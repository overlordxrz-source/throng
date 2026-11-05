"""Learner testbed (Cam's Step 2, 2026-11-04): can the production learner learn one-body navigation at all?

Real network (AgentNetworkJax, fresh init, production dims), real ppo_update/_minibatch_step/ppo_loss,
production action sampling (masked logits + alarm head, joint log-prob). The world is minimal: a 15x15
torus, one hearth (all four hearth_rel slots point at it), agents always hold wood and are informed,
deposit = Craft within hearth_radius of the hearth, then respawn at a random cell (still holding, no
done -- agents continue, as in production). No reds, comms, neighbours, food or ecology.
E copies of the single-agent world run in parallel (independent, no interaction) to fill a batch.

Arms (Cam):
  A  production MC (rl_jax.compute_gae as is), deposit only
  B  production MC + potential shaping (beta 24, gamma .999)
  C  GAE lambda .95 + bootstrap gamma*V(s_T) at the rollout boundary, deposit only
  D  C + shaping
  E  frozen-policy control (lr=0): A's reward and estimator, no parameter update -- the noise floor for
     distance change from deposit/respawn dynamics and starting-weight biases (Cam, 2026-11-04). Adam with
     lr=0 applies an exactly-zero update, so E skips the PPO step instead of running it (same parameters).
Arms C/D monkeypatch rl_jax.compute_gae for the duration of ppo_update; nothing in jax_sim/ changes.

Usage: python scripts/learner_testbed.py --arm A --seed 0 --updates 20 --out runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402

from jax_sim import rl_jax  # noqa: E402
from jax_sim.action_space import mask_disabled_actions  # noqa: E402
from jax_sim.network_jax import AgentNetworkJax, init_agent_params, make_model_apply  # noqa: E402
from jax_sim.obs_layout import make_obs_layout  # noqa: E402
from jax_sim.main_jax import _normalize_config  # noqa: E402
from jax_sim.rl_jax import PopArt, RunningReturnStd, create_optimizer, ppo_update  # noqa: E402

ACTION_CRAFT = 10
DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]] + [[0, 0]] * 7)  # apply_moves order

_production_compute_gae = rl_jax.compute_gae


def gae_bootstrap(rewards, values, dones, last_values, gamma, lam):
    """Standard GAE(lambda) with gamma*V(s_T) bootstrap at the rollout boundary; same normalisation and
    +/-10 clip as the production function so only the estimator differs."""
    rewards = np.asarray(rewards, np.float32)
    values = np.asarray(values, np.float32)
    dones = np.asarray(dones, np.float32)
    T, N = rewards.shape
    adv = np.zeros((T, N), np.float32)
    nxt_v = np.asarray(last_values, np.float32)
    carry = np.zeros(N, np.float32)
    for t in range(T - 1, -1, -1):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt_v * nonterm - values[t]
        carry = delta + gamma * lam * nonterm * carry
        adv[t] = carry
        nxt_v = values[t]
    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return np.clip(adv, -10.0, 10.0), returns


def testbed_hearth_rel(pos, hearth, G):
    """The 8-dim hearth_rel block, built as observations_jax builds it (toroidal signed diff / (G/2), 4 hearths x
    (axis0, axis1)); the testbed's single hearth fills all four slots. Checked against build_observations_jax in
    tests/test_testbed_obs_matches_production.py."""
    diff = (np.asarray(hearth)[None, :] - pos + G / 2) % G - G / 2
    rel = (diff / (G / 2.0)).astype(np.float32)
    return np.tile(rel, (1, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list("ABCDE"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--hearth-radius", type=int, default=1)
    ap.add_argument("--critic-fix", action="store_true", help="value_clip=False + running return std (critic-fix branch)")
    ap.add_argument("--vf-coef", type=float, default=None, help="override ppo_value_coef (default: config, 0.1)")
    ap.add_argument("--popart", action="store_true", help="PopArt value normalisation (critic-fix branch); implies no clip")
    ap.add_argument("--save-params", default=None, help="write the final params (flax msgpack) here")
    ap.add_argument("--eval-grids", default="", help="comma list: evaluate the frozen final policy at these sizes")
    ap.add_argument("--eval-updates", type=int, default=10, help="rollouts per eval size")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timing-only", action="store_true", help="print timing, suppress learning metrics")
    a = ap.parse_args()

    cfg = _normalize_config(yaml.safe_load((ROOT / "config.yaml").read_text()))  # as production
    T = int(cfg["ppo_rollout_steps"])            # 512
    gamma = float(cfg["ppo_gamma"])              # 0.999
    lam = float(cfg["ppo_gae_lam"])              # 0.95
    beta = float(cfg["ctd_competence_ramp"]["hearth_shaping_beta"]) if a.arm in "BD" else 0.0
    R_dep = float(cfg["reward_hearth_deposit"]) + float(cfg["reward_hearth_completion"])  # solo completion
    n_layers = int(cfg["n_layers"])
    hidden = int(cfg.get("hidden_dim", cfg.get("agent_hidden_dim", 256)))
    G, E, hr = a.grid, a.envs, a.hearth_radius

    layout = make_obs_layout(
        signal_dim=cfg["signal_dim"], symbol_dim=cfg.get("symbol_dim", 8), memory_slots=cfg.get("memory_slots", 0),
        neighbor_k=cfg["neighbor_k"], local_cells=(2 * cfg["local_obs_radius"] + 1) ** 2,
        env_channels=int(cfg.get("env_channels", 15)), own_state_dim=int(cfg.get("own_state_dim", 22)),
    )
    assert layout.own_state_dim == 29, "testbed fills the 29-dim Phase 19 own_state layout"
    model = AgentNetworkJax(
        hidden_dim=hidden, n_heads=cfg["n_heads"], n_layers=n_layers, obs_dim=layout.total_dim,
        signal_dim=cfg["signal_dim"], symbol_dim=cfg["symbol_dim"], vocab_size=cfg["vocab_size"],
        vq_beta=float(cfg.get("vq_beta", 0.25)), vq_dead_code_reset=bool(cfg.get("vq_dead_code_reset", True)),
        memory_slots=cfg.get("memory_slots", 0), fwd_env_dim=layout.loc_env_end - layout.loc_env_start,
        cross_attn_enabled=bool((cfg.get("phase9_canvas") or {}).get("cross_attn_enabled", False)),
        cross_attn_num_heads=int((cfg.get("phase9_canvas") or {}).get("cross_attn_num_heads", cfg["n_heads"])),
        env_channels=int(cfg.get("env_channels", 15)), own_state_dim=layout.own_state_dim,
        n_actions=int(cfg["n_actions"]), local_cells=layout.local_cells, neighbor_k=cfg["neighbor_k"],
    )
    apply_fn = make_model_apply(model)
    key = jax.random.PRNGKey(a.seed)
    key, k_init = jax.random.split(key)
    params = init_agent_params(model, k_init, jnp.zeros((1, hidden)), jnp.zeros((1, layout.total_dim)), n_layers)
    opt = create_optimizer(float(cfg["ppo_lr"]), float(cfg["ppo_max_grad_norm"]))
    opt_state = opt.init(params)
    return_rms = RunningReturnStd() if a.critic_fix else None
    vf_coef = a.vf_coef if a.vf_coef is not None else float(cfg.get("ppo_value_coef", 0.25))
    gpu = jax.devices()[0].device_kind

    @jax.jit
    def policy_step(params, carries, obs, k):
        new_c, outs = apply_fn(params, carries, obs, n_layers)
        logits = mask_disabled_actions(outs.action_logits, axis=-1)
        k1, k2 = jax.random.split(k)
        act = jax.random.categorical(k1, logits)
        alarm = jax.random.categorical(k2, outs.alarm_out)
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), act[:, None], -1)[:, 0]
        lp = lp + jnp.take_along_axis(jax.nn.log_softmax(outs.alarm_out, -1), alarm[:, None], -1)[:, 0]
        lvq = outs.loss_vq.sum(-1) if outs.loss_vq.ndim > 1 else outs.loss_vq
        return new_c, act, alarm, lp, outs.values, lvq

    rng = np.random.default_rng(a.seed)
    popart = PopArt(beta=0.1) if a.popart else None

    def to_value(v):  # PopArt: the head's output is the normalised value
        v = np.asarray(v)
        return popart.denormalise(v) if popart is not None else v

    class World:
        """E independent copies of the one-agent, one-hearth G x G torus."""

        def __init__(self, G):
            self.G, self.H, self.d_max = G, np.array([G // 2, G // 2]), G // 2
            self.pos = self.respawn(E)

        def cheb(self, p):
            d = np.abs(p - self.H)
            d = np.minimum(d, self.G - d)
            return d.max(-1)

        def respawn(self, n):
            p = rng.integers(0, self.G, (n, 2))
            bad = self.cheb(p) <= hr
            if bad.any():
                p[bad] = self.respawn(int(bad.sum()))
            return p

        def build_obs(self, k):
            pos, G = self.pos, self.G
            obs = np.zeros((E, layout.total_dim), np.float32)
            s = layout.own_state_start
            obs[:, s + 2] = 1.0                                  # energy (keeps the feral alarm mask off)
            obs[:, s + 3] = n_layers / 6.0                       # nl_norm
            obs[:, s + 4] = pos[:, 0] / G
            obs[:, s + 5] = pos[:, 1] / G
            obs[:, s + 6] = 1.0                                  # inv_wood: always holding
            obs[:, s + 12] = 1.0                                 # can_see_recipe
            obs[:, s + 13: s + 21] = testbed_hearth_rel(pos, self.H, G)  # same block as production (tested)
            obs[:, s + 21: s + 25] = 0.0                          # hearth_need(wood=0)/4 * informed = 0
            obs[:, s + 25: s + 29] = np.asarray(jax.random.normal(k, (E, 4)))  # intrinsic entropy, as production
            return obs

    def rollout(world, params, carries, key, shaping_beta):
        B = {k: [] for k in ("obs", "actions", "alarm_actions", "log_probs", "rewards", "values", "carries", "loss_vq")}
        st = dict(dist=0.0, deps=0, toward=0, moves=0)
        for t in range(T):
            key, k_obs, k_act = jax.random.split(key, 3)
            obs = world.build_obs(k_obs)
            B["carries"].append(np.asarray(carries))
            carries, act, alarm, lp, val, lvq = policy_step(params, carries, jnp.asarray(obs), k_act)
            act = np.asarray(act)
            d0 = world.cheb(world.pos)
            new = (world.pos + DELTAS[act]) % world.G
            d1 = world.cheb(new)
            dep = (act == ACTION_CRAFT) & (d0 <= hr)
            r = np.where(dep, R_dep, shaping_beta * (gamma * (-d1 / world.d_max) + d0 / world.d_max)).astype(np.float32)
            st["moves"] += int((d1 != d0).sum())
            st["toward"] += int((d1 < d0).sum())
            st["dist"] += float(d0.sum())
            st["deps"] += int(dep.sum())
            world.pos = np.where(dep[:, None], world.respawn(E), new)
            for k_, v_ in (("obs", obs), ("actions", act), ("alarm_actions", np.asarray(alarm)),
                           ("log_probs", np.asarray(lp)), ("rewards", r), ("values", to_value(val)),
                           ("loss_vq", np.asarray(lvq))):
                B[k_].append(v_)
        return B, st, carries, key

    def emit(rec):
        if a.timing_only:
            print(json.dumps({k: rec[k] for k in ("arm", "seed", "grid", "gpu", "update", "t_rollout", "t_update")}), flush=True)
            return
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    base = dict(arm=a.arm, seed=a.seed, gpu=gpu, critic_fix=a.critic_fix, popart=a.popart, vf_coef=vf_coef)
    world = World(G)
    carries = jnp.zeros((E, hidden))
    for u in range(1, a.updates + 1):
        t0 = time.time()
        B, st, carries, key = rollout(world, params, carries, key, beta)
        t_roll = time.time() - t0
        batch = {k: jnp.asarray(np.stack(v)) for k, v in B.items()}
        batch["dones"] = jnp.zeros((T, E), jnp.float32)
        batch["alive"] = jnp.ones((T, E), bool)
        values_np = np.stack(B["values"])
        rewards_np = np.stack(B["rewards"])

        # value of s_T, for the bootstrap arms and for logging
        key, k_obs, k_act = jax.random.split(key, 3)
        _, _, _, _, v_last, _ = policy_step(params, carries, jnp.asarray(world.build_obs(k_obs)), k_act)
        v_last = to_value(v_last)
        if a.arm in "CD":
            rl_jax.compute_gae = lambda r_, v_, d_, gamma, lam, **kw: gae_bootstrap(r_, v_, d_, v_last, gamma, lam)
            _, targets = gae_bootstrap(rewards_np, values_np, np.zeros_like(rewards_np), v_last, gamma, lam)
        else:
            rl_jax.compute_gae = _production_compute_gae
            _, targets = _production_compute_gae(rewards_np, values_np, np.zeros_like(rewards_np), gamma=gamma, lam=lam)
        var_target = float(np.var(targets))
        var_resid = float(np.var(targets - values_np))
        ev = 1.0 - var_resid / (var_target + 1e-8)

        key, k_up = jax.random.split(key)
        if a.arm == "E":
            mets = {}
        else:
            params, opt_state, mets = ppo_update(
                params, opt_state, opt, apply_fn, batch, n_layers, k_up,
                clip_eps=float(cfg.get("ppo_clip_eps", 0.2)), vf_coef=vf_coef,
                ent_coef=float(cfg.get("ppo_entropy_coef", 0.02)), vq_coef=float(cfg.get("vq_loss_coef", 0.1)),
                minibatch_size=int(cfg["ppo_minibatch_size"]), gamma=gamma, lam=lam, team="blue",
                alarm_ent_coef=float(cfg.get("alarm_ent_coef", 0.0)), freeze_comms=True,
                return_rms=return_rms, value_clip=not (a.critic_fix or a.popart), popart=popart,
            )
        rl_jax.compute_gae = _production_compute_gae
        emit(dict(base, phase="train", grid=G, update=u, mean_dist=st["dist"] / (T * E), deposits_per_env=st["deps"] / E,
                  frac_moves_toward=st["toward"] / max(st["moves"], 1), ev=float(ev), var_target=var_target,
                  var_resid=var_resid, v_mean=float(values_np.mean()), target_mean=float(np.mean(targets)),
                  entropy=float(mets.get("ppo_entropy", np.nan)), clip=float(mets.get("ppo_clip_frac", np.nan)),
                  t_rollout=round(t_roll, 1), t_update=round(time.time() - t0, 1)))

    if a.save_params:
        import flax.serialization
        Path(a.save_params).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save_params).write_bytes(flax.serialization.to_bytes(params))
        print(f"[testbed] saved final params -> {a.save_params}", flush=True)

    # Transfer evaluation: the FROZEN final policy at other map sizes (no update, fresh world and carries).
    for G_eval in [int(x) for x in a.eval_grids.split(",") if x]:
        ew, ec = World(G_eval), jnp.zeros((E, hidden))
        for u in range(1, a.eval_updates + 1):
            t0 = time.time()
            _, st, ec, key = rollout(ew, params, ec, key, 0.0)
            emit(dict(base, phase="eval", trained_grid=G, grid=G_eval, update=u, mean_dist=st["dist"] / (T * E),
                      deposits_per_env=st["deps"] / E, frac_moves_toward=st["toward"] / max(st["moves"], 1),
                      t_rollout=round(time.time() - t0, 1), t_update=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
