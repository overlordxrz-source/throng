"""Communication experiment in the learner testbed (Cam, 2026-11-05; draft docs/drafts/2026-11-05_testbed_comm_
experiment_DRAFT.md, registration in docs/RESEARCH_PROTOCOL.md).

World: W independent G x G tori, each with the four production hearths (generate_hearth_positions) and TWO agents
sharing one network -- a SENDER (informed: its 4 hearth_need dims carry one-hot of the active hearth,
can_see_recipe=1) and a LISTENER (those dims 0). Exactly one hearth is active; Craft within radius 1 of it is a
deposit (2.7 to the depositor), the depositor respawns, and the active hearth re-rolls uniformly among the other
three (also every 256 steps without a deposit). A listener deposit also pays the sender 2.7 (shared reward).
Everyone sees all four hearth_rel vectors (production formula). The testbed has no local patch, so agents cannot
see each other: spatial following is impossible by construction; the only route for information is the wire.

Channel fix: the token is a policy action (rl_jax.token_logits, tau), sampled each step, broadcast as the codebook
vectors, its log-prob in the PPO ratio; the comms subtree trains (freeze_comms=False). Estimator C' (GAE 0.95 +
bootstrap), critic PopArt, vf 0.5.

Arms:  COMM   wire live (each agent's broadcast fills the other's first neighbour-signal slot, one step later)
       NOSIG  wire zeroed both ways (what a listener does with no channel)
       ORACLE wire zeroed; the listener's 4 need dims carry the one-hot too (apparatus check)
After training (COMM): MI(token; active) over the last 10 updates' sender steps, and the paired token-swap eval.
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
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402

from jax_sim import rl_jax  # noqa: E402
from jax_sim.action_space import mask_disabled_actions  # noqa: E402
from jax_sim.grid_jax import generate_hearth_positions  # noqa: E402
from jax_sim.main_jax import _normalize_config  # noqa: E402
from jax_sim.network_jax import AgentNetworkJax, init_agent_params, make_model_apply  # noqa: E402
from jax_sim.obs_layout import make_obs_layout  # noqa: E402
from jax_sim.rl_jax import PopArt, create_optimizer, ppo_update, token_logits  # noqa: E402
from learner_testbed import ACTION_CRAFT, DELTAS, gae_bootstrap  # noqa: E402

_production_compute_gae = rl_jax.compute_gae
REROLL_STEPS = 256
R_DEP = 2.7


def hearth_rel_multi(pos, hearths, G):
    """(N, 8) hearth_rel for 4 distinct hearths, exactly observations_jax's formula and slot order."""
    diff = (hearths[None, :, :] - pos[:, None, :] + G / 2) % G - G / 2   # (N, 4, 2)
    return (diff / (G / 2.0)).reshape(pos.shape[0], 8).astype(np.float32)


def mi_miller_madow(x, y, kx, ky):
    joint = np.zeros((kx, ky))
    np.add.at(joint, (x, y), 1)
    n = joint.sum()
    p = joint / n
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())
    # Miller-Madow: subtract (cells_xy - cells_x - cells_y + 1) / (2 n ln 2)
    corr = ((joint > 0).sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2 * n * np.log(2))
    return mi - corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["COMM", "NOSIG", "ORACLE"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--hearth-radius", type=int, default=1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--token-ent-coef", type=float, default=0.01)
    ap.add_argument("--warm-start", default=None, help="flax msgpack params from a 32x32 PopArt testbed run")
    ap.add_argument("--swap-pairs", type=int, default=10)
    ap.add_argument("--craft-cost", type=float, default=0.0,
                    help="cost of a Craft that is not a deposit (closes the camping loophole; final run)")
    ap.add_argument("--halt-update", type=int, default=0, help="auto-halt check after this update (0 = off)")
    ap.add_argument("--halt-sender-dep", type=float, default=0.0,
                    help="halt unless mean sender deposits/world over updates 1..halt-update >= this")
    ap.add_argument("--halt-token-entropy", type=float, default=0.0,
                    help="halt unless token entropy (3 slots summed) at halt-update >= this")
    ap.add_argument("--save-params", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timing-only", action="store_true")
    a = ap.parse_args()

    cfg = _normalize_config(yaml.safe_load((ROOT / "config.yaml").read_text()))
    T = int(cfg["ppo_rollout_steps"])
    gamma, lam = float(cfg["ppo_gamma"]), float(cfg["ppo_gae_lam"])
    nl = int(cfg["n_layers"])
    hidden = int(cfg["hidden_dim"])
    G, W, hr = a.grid, a.worlds, a.hearth_radius
    N = 2 * W                                   # agent 2w = sender of world w, 2w+1 = listener
    HEARTHS = np.asarray(generate_hearth_positions(G))
    L = make_obs_layout(signal_dim=cfg["signal_dim"], symbol_dim=cfg["symbol_dim"],
                        memory_slots=cfg.get("memory_slots", 0), neighbor_k=cfg["neighbor_k"],
                        local_cells=(2 * cfg["local_obs_radius"] + 1) ** 2,
                        env_channels=int(cfg.get("env_channels", 15)), own_state_dim=int(cfg.get("own_state_dim", 29)))
    assert L.own_state_dim == 29
    model = AgentNetworkJax(
        hidden_dim=hidden, n_heads=cfg["n_heads"], n_layers=nl, obs_dim=L.total_dim, signal_dim=cfg["signal_dim"],
        symbol_dim=cfg["symbol_dim"], vocab_size=cfg["vocab_size"], vq_beta=float(cfg.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(cfg.get("vq_dead_code_reset", True)), memory_slots=cfg.get("memory_slots", 0),
        fwd_env_dim=L.loc_env_end - L.loc_env_start,
        cross_attn_enabled=bool((cfg.get("phase9_canvas") or {}).get("cross_attn_enabled", False)),
        cross_attn_num_heads=int((cfg.get("phase9_canvas") or {}).get("cross_attn_num_heads", cfg["n_heads"])),
        env_channels=int(cfg.get("env_channels", 15)), own_state_dim=L.own_state_dim, n_actions=int(cfg["n_actions"]),
        local_cells=L.local_cells, neighbor_k=cfg["neighbor_k"])
    apply_fn = make_model_apply(model)
    key = jax.random.PRNGKey(a.seed)
    key, k_init = jax.random.split(key)
    params = init_agent_params(model, k_init, jnp.zeros((1, hidden)), jnp.zeros((1, L.total_dim)), nl)
    if a.warm_start:
        import flax.serialization
        params = flax.serialization.from_bytes(params, Path(a.warm_start).read_bytes())
        print(f"[comm] warm start <- {a.warm_start}", flush=True)
    opt = create_optimizer(float(cfg["ppo_lr"]), float(cfg["ppo_max_grad_norm"]))
    opt_state = opt.init(params)
    popart = PopArt(beta=0.1)
    gpu = jax.devices()[0].device_kind
    rng = np.random.default_rng(a.seed)
    is_listener = np.tile([False, True], W)
    ns, ne = L.nb_sigs_start, L.nb_sigs_start + int(cfg["signal_dim"])
    s0 = L.own_state_start

    @jax.jit
    def step_fn(params, carries, obs, k):
        new_c, o = apply_fn(params, carries, obs, nl)
        logits = mask_disabled_actions(o.action_logits, axis=-1)
        k1, k2, k3 = jax.random.split(k, 3)
        act = jax.random.categorical(k1, logits)
        alarm = jax.random.categorical(k2, o.alarm_out)
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), act[:, None], -1)[:, 0]
        lp = lp + jnp.take_along_axis(jax.nn.log_softmax(o.alarm_out, -1), alarm[:, None], -1)[:, 0]
        toks, tent, bc = [], 0.0, [jnp.zeros((obs.shape[0], 8))]
        p = params["params"] if "params" in params else params
        for i, lg in enumerate(token_logits(params, o.z_e, a.tau)):
            kk = jax.random.fold_in(k3, i)
            t = jax.random.categorical(kk, lg)
            lsm = jax.nn.log_softmax(lg, -1)
            lp = lp + jnp.take_along_axis(lsm, t[:, None], -1)[:, 0]
            tent = tent - jnp.sum(jnp.exp(lsm) * lsm, -1)
            toks.append(t)
            bc.append(p[rl_jax.TOKEN_SLOTS[i][2]]["embedding"][t])
        lvq = o.loss_vq.sum(-1) if o.loss_vq.ndim > 1 else o.loss_vq
        return new_c, act, alarm, lp, o.values, lvq, jnp.stack(toks, -1), jnp.concatenate(bc, -1), tent

    @jax.jit
    def swap_tokens(params, carries, obs, k):
        """Tokens a sender WOULD emit for obs (used with T' injected); same key -> paired sampling."""
        _, o = apply_fn(params, carries, obs, nl)
        p = params["params"] if "params" in params else params
        k3 = jax.random.split(k, 3)[2]
        bc = [jnp.zeros((obs.shape[0], 8))]
        for i, lg in enumerate(token_logits(params, o.z_e, a.tau)):
            t = jax.random.categorical(jax.random.fold_in(k3, i), lg)
            bc.append(p[rl_jax.TOKEN_SLOTS[i][2]]["embedding"][t])
        return jnp.concatenate(bc, -1)

    class World:
        def __init__(self, r):
            self.r = r
            self.pos = np.zeros((N, 2), int)
            self.pos[:] = self.respawn(N)
            self.active = self.r.integers(0, 4, W)
            self.since = np.zeros(W, int)
            self.wire = np.zeros((N, int(cfg["signal_dim"])), np.float32)   # last step's broadcast per agent

        def dist_to(self, pos, h):
            d = np.abs(pos - h)
            d = np.minimum(d, G - d)
            return d.max(-1)

        def near_any(self, pos):
            return np.stack([self.dist_to(pos, h) <= hr for h in HEARTHS], 1)      # (n, 4)

        def respawn(self, n):
            p = self.r.integers(0, G, (n, 2))
            bad = self.near_any(p).any(1)
            if bad.any():
                p[bad] = self.respawn(int(bad.sum()))
            return p

        def onehot(self, idx):
            return np.eye(4, dtype=np.float32)[idx]

        def build_obs(self, k, active_override=None):
            act = self.active if active_override is None else active_override
            obs = np.zeros((N, L.total_dim), np.float32)
            obs[:, s0 + 2] = 1.0
            obs[:, s0 + 3] = nl / 6.0
            obs[:, s0 + 4] = self.pos[:, 0] / G
            obs[:, s0 + 5] = self.pos[:, 1] / G
            obs[:, s0 + 6] = 1.0
            obs[:, s0 + 13: s0 + 21] = hearth_rel_multi(self.pos, HEARTHS, G)
            oh = self.onehot(np.repeat(act, 2))                                     # (N, 4)
            informed = ~is_listener if a.arm != "ORACLE" else np.ones(N, bool)
            obs[:, s0 + 12] = informed
            obs[:, s0 + 21: s0 + 25] = oh * informed[:, None]
            obs[:, s0 + 25: s0 + 29] = np.asarray(jax.random.normal(k, (N, 4)))
            if a.arm == "COMM":                                                     # partner's last broadcast
                obs[:, ns:ne] = self.wire.reshape(W, 2, -1)[:, ::-1].reshape(N, -1)
            return obs

    def apparatus(obs, world, bc, toks, params):
        lst = obs[is_listener, s0 + 21: s0 + 25]
        snd = obs[~is_listener, s0 + 21: s0 + 25]
        ok = np.array_equal(snd.argmax(1), world.active) and np.allclose(snd.sum(1), 1)
        ok &= np.allclose(lst.sum(1), 1) if a.arm == "ORACLE" else bool((lst == 0).all())
        p = params["params"] if "params" in params else params
        for i, (lo, hi, cbk) in enumerate(rl_jax.TOKEN_SLOTS):
            ok &= np.allclose(bc[:, lo:hi], np.asarray(p[cbk]["embedding"])[toks[:, i]])
        if a.arm != "COMM":
            ok &= bool((obs[:, ns:ne] == 0).all())
        if not ok:
            print("[APPARATUS] FATAL: target dims / one-hot / token-broadcast / wire check failed", flush=True)
            raise SystemExit(1)

    def run_rollout(world, params, carries, key, collect, swap=False):
        """One T-step rollout. collect=True returns the PPO batch. swap=True: the listener receives the token its
        sender would emit with active -> (active + 1) % 4 (eval only)."""
        B = {k: [] for k in ("obs", "actions", "alarm_actions", "log_probs", "rewards", "values", "carries",
                             "loss_vq", "token_actions")} if collect else None
        st = dict(l_dep=0, s_dep=0, l_att=0, l_att_ok=0, s_att=0, tent=0.0, mu_swap=[], mu_true=[], snd_tok=[],
                  snd_T=[], epoch=[], swap_diff=0, swap_n=0)
        epoch_id = np.arange(W)
        for t in range(T):
            key, k_obs, k_act = jax.random.split(key, 3)
            obs = world.build_obs(k_obs)
            if collect:
                B["carries"].append(np.asarray(carries))
            c_prev = carries
            carries, act, alarm, lp, val, lvq, toks, bc, tent = step_fn(params, carries, jnp.asarray(obs), k_act)
            act, toks, bc = np.asarray(act), np.asarray(toks), np.asarray(bc)
            if t == 0 or t % 64 == 0:
                apparatus(obs, world, bc, toks, params)
            st["tent"] += float(np.asarray(tent).mean())
            st["snd_tok"].append(toks[~is_listener]); st["snd_T"].append(world.active.copy())
            st["epoch"].append(epoch_id.copy())
            if swap:   # listener hears what its sender would say about T' = (T + 1) % 4
                # Each agent reads its PARTNER's wire entry, so the swap replaces the SENDER's broadcast; the
                # sender's own action, carry and reward still come from the true T (sampled with the same key).
                sw_obs = world.build_obs(k_obs, active_override=(world.active + 1) % 4)
                bc_sw = np.asarray(swap_tokens(params, c_prev, jnp.asarray(sw_obs), k_act))
                st["swap_diff"] += int((~np.isclose(bc_sw[~is_listener], bc[~is_listener]).all(1)).sum())
                st["swap_n"] += W
                bc = bc.copy()
                bc[~is_listener] = bc_sw[~is_listener]
                heard = bc.reshape(W, 2, -1)[:, ::-1].reshape(N, -1)[is_listener]   # what build_obs will hand over
                if not np.allclose(heard, bc_sw[~is_listener]):
                    print("[APPARATUS] FATAL: swap wiring -- listener does not receive the T' broadcast", flush=True)
                    raise SystemExit(1)
            # movement metrics for listeners (m.u toward T' and toward T)
            new = (world.pos + DELTAS[act]) % G
            for tgt_name, tgt_idx in (("mu_swap", (world.active + 1) % 4), ("mu_true", world.active)):
                h = HEARTHS[np.repeat(tgt_idx, 2)]
                dvec = (h - world.pos + G / 2) % G - G / 2
                nrm = np.linalg.norm(dvec, axis=1)
                mv = DELTAS[act].astype(float)
                m = is_listener & (act >= 1) & (act <= 4) & (nrm > 0)
                st[tgt_name].append(float(np.sum((mv[m] * dvec[m]).sum(1) / nrm[m])) if m.any() else 0.0)
                st[tgt_name + "_n"] = st.get(tgt_name + "_n", 0) + int(m.sum())
            near = world.near_any(world.pos)                                        # pre-move, as resolve_*
            craft = act == ACTION_CRAFT
            at_active = near[np.arange(N), np.repeat(world.active, 2)]
            dep = craft & at_active
            att = craft & near.any(1)
            st["l_att"] += int((att & is_listener).sum()); st["l_att_ok"] += int((dep & is_listener).sum())
            st["s_att"] += int((att & ~is_listener).sum())
            st["l_dep"] += int((dep & is_listener).sum()); st["s_dep"] += int((dep & ~is_listener).sum())
            r = np.where(dep, R_DEP, 0.0) - a.craft_cost * (craft & ~dep)
            ldep_w = dep[is_listener]                                                # shared reward to sender
            r[~is_listener] += R_DEP * ldep_w
            world.pos = np.where(dep[:, None], world.respawn(N), new)
            dep_w = dep.reshape(W, 2).any(1)
            world.since += 1
            reroll = dep_w | (world.since >= REROLL_STEPS)
            if reroll.any():
                world.active[reroll] = (world.active[reroll] + world.r.integers(1, 4, int(reroll.sum()))) % 4
                world.since[reroll] = 0
                epoch_id = np.where(reroll, epoch_id + W * (t + 1), epoch_id)
            world.wire = bc if a.arm == "COMM" else np.zeros_like(bc)
            if collect:
                for k_, v_ in (("obs", obs), ("actions", act), ("alarm_actions", np.asarray(alarm)),
                               ("log_probs", np.asarray(lp)), ("rewards", r.astype(np.float32)),
                               ("values", popart.denormalise(np.asarray(val))), ("loss_vq", np.asarray(lvq)),
                               ("token_actions", toks)):
                    B[k_].append(v_)
        return B, st, carries, key

    def emit(rec):
        if a.timing_only:
            print(json.dumps({k: rec.get(k) for k in ("arm", "seed", "update", "t_update")}), flush=True)
            return
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    base = dict(arm=a.arm, seed=a.seed, grid=G, gpu=gpu, tau=a.tau, warm_start=bool(a.warm_start),
                craft_cost=a.craft_cost)
    halt_sdep = []
    world, carries = World(rng), jnp.zeros((N, hidden))
    mi_tok, mi_T, mi_ep = [], [], []
    for u in range(1, a.updates + 1):
        t0 = time.time()
        B, st, carries, key = run_rollout(world, params, carries, key, collect=True)
        batch = {k: jnp.asarray(np.stack(v)) for k, v in B.items()}
        batch["dones"] = jnp.zeros((T, N), jnp.float32)
        batch["alive"] = jnp.ones((T, N), bool)
        values_np, rewards_np = np.stack(B["values"]), np.stack(B["rewards"])
        key, k_obs, k_act = jax.random.split(key, 3)
        v_last = popart.denormalise(np.asarray(step_fn(params, carries, jnp.asarray(world.build_obs(k_obs)), k_act)[4]))
        rl_jax.compute_gae = lambda r_, v_, d_, gamma, lam, **kw: gae_bootstrap(r_, v_, d_, v_last, gamma, lam)
        _, targets = gae_bootstrap(rewards_np, values_np, np.zeros_like(rewards_np), v_last, gamma, lam)
        key, k_up = jax.random.split(key)
        params, opt_state, mets = ppo_update(
            params, opt_state, opt, apply_fn, batch, nl, k_up, clip_eps=float(cfg.get("ppo_clip_eps", 0.2)),
            vf_coef=a.vf_coef, ent_coef=float(cfg.get("ppo_entropy_coef", 0.02)),
            vq_coef=float(cfg.get("vq_loss_coef", 0.1)), minibatch_size=int(cfg["ppo_minibatch_size"]), gamma=gamma,
            lam=lam, team="blue", alarm_ent_coef=float(cfg.get("alarm_ent_coef", 0.0)), freeze_comms=False,
            value_clip=False, popart=popart, token_tau=a.tau, token_ent_coef=a.token_ent_coef)
        rl_jax.compute_gae = _production_compute_gae
        if u > a.updates - 10:
            mi_tok.append(np.concatenate(st["snd_tok"])); mi_T.append(np.concatenate(st["snd_T"]))
            mi_ep.append(np.concatenate(st["epoch"]) + u * 10 ** 7)
        ev = 1 - np.var(targets - values_np) / (np.var(targets) + 1e-8)
        halt_sdep.append(st["s_dep"] / W)
        if a.halt_update and u == a.halt_update:
            sdep, tent = float(np.mean(halt_sdep)), st["tent"] / T
            ok = sdep >= a.halt_sender_dep and tent >= a.halt_token_entropy
            print(f"[AUTO-HALT CHECK] update {u}: sender dep/world {sdep:.2f} (need >= {a.halt_sender_dep}) | token "
                  f"entropy {tent:.2f} (need >= {a.halt_token_entropy}) -> {'continue' if ok else 'HALT'}", flush=True)
            if not ok:
                emit(dict(base, phase="halt", update=u, sender_dep=sdep, token_entropy=tent))
                raise SystemExit(3)
        emit(dict(base, phase="train", update=u, listener_dep_per_world=st["l_dep"] / W,
                  sender_dep_per_world=st["s_dep"] / W, listener_attempts=st["l_att"], sender_attempts=st["s_att"],
                  listener_correct_frac=st["l_att_ok"] / max(st["l_att"], 1),
                  token_entropy=st["tent"] / T, ev=float(ev), entropy=float(mets.get("ppo_entropy", np.nan)),
                  ppo_token_entropy=float(mets.get("ppo_token_entropy", np.nan)), t_update=round(time.time() - t0, 1)))

    if a.save_params:
        import flax.serialization
        Path(a.save_params).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save_params).write_bytes(flax.serialization.to_bytes(params))

    if a.timing_only:
        return
    # MI(token; active) per slot over the last 10 updates' sender steps, block-permutation null over epochs
    tok, Tt, ep = np.concatenate(mi_tok), np.concatenate(mi_T), np.concatenate(mi_ep)
    uniq, inv = np.unique(ep, return_inverse=True)
    ep_T = np.zeros(len(uniq), int); ep_T[inv] = Tt
    prng = np.random.default_rng(1000 + a.seed)
    for i in range(3):
        mi = mi_miller_madow(tok[:, i], Tt, int(cfg["vocab_size"]), 4)
        null = np.array([mi_miller_madow(tok[:, i], prng.permutation(ep_T)[inv], int(cfg["vocab_size"]), 4)
                         for _ in range(1000)])
        emit(dict(base, phase="mi", slot=i, mi_bits=mi, null_p99=float(np.percentile(null, 99)),
                  null_mean=float(null.mean()), n=int(len(Tt)), epochs=int(len(uniq))))

    # Paired token-swap evaluation (COMM only): frozen params, same initial world + keys per pair
    if a.arm == "COMM":
        for k in range(a.swap_pairs):
            res = {}
            for arm_name, swap in (("control", False), ("swap", True)):
                w = World(np.random.default_rng(10_000 + k))
                _, st, _, _ = run_rollout(w, params, jnp.zeros((N, hidden)), jax.random.PRNGKey(10_000 + k),
                                          collect=False, swap=swap)
                res[arm_name] = (sum(st["mu_swap"]) / max(st["mu_swap_n"], 1),
                                 sum(st["mu_true"]) / max(st["mu_true_n"], 1), st["l_dep"] / W,
                                 st["swap_diff"] / max(st["swap_n"], 1))
            emit(dict(base, phase="swap", pair=k, mu_Tprime_control=res["control"][0], mu_Tprime_swap=res["swap"][0],
                      mu_T_control=res["control"][1], mu_T_swap=res["swap"][1],
                      listener_dep_control=res["control"][2], listener_dep_swap=res["swap"][2],
                      swapped_token_differs_frac=res["swap"][3]))


if __name__ == "__main__":
    main()
