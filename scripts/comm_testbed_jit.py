"""Step 0 (Cam, 2026-11-08): the comm testbed (scripts/comm_testbed.py, ceadaf8) with its world stepped inside a
jitted lax.scan instead of a Python loop. Same network, same ppo_update, same estimator (C'), same PopArt, same
obs layout and formulas, same JAX key schedule (key, k_obs, k_act = split(key, 3) per step). The only intended
difference: the world's random draws (respawn cells, re-roll increments, initial state) come from a JAX env key
instead of numpy, with the same distributions (uniform over cells not within radius 1 of a hearth; uniform on
{1, 2, 3}). Equivalence against the Python path: scripts/equiv_comm_jit.py (registered bar, docs/RESEARCH_PROTOCOL).

--equiv PATH: run one rollout with the draws injected from PATH.draws.npz and save PATH.jit.npz (test only).
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
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--craft-cost", type=float, default=0.0)
    ap.add_argument("--halt-update", type=int, default=0)
    ap.add_argument("--halt-sender-dep", type=float, default=0.0)
    ap.add_argument("--save-params", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--equiv", default=None, help="TEST ONLY: one rollout with injected draws -> PATH.jit.npz")
    ap.add_argument("--ps", type=float, default=0.0, help="positive signalling coefficient (0 = off)")
    ap.add_argument("--apparatus-update", type=int, default=2,
                    help="Cam 2026-11-08: halt after this update if the PS gradient misses the message heads, messages are constant, or the listener wire input is zero (0 = off)")
    ap.add_argument("--mi-null-perms", type=int, default=100, help="block permutations for the per-update MI null")
    ap.add_argument("--swap-pairs", type=int, default=10)
    ap.add_argument("--eval-at", type=int, nargs="*", default=[])
    ap.add_argument("--no-eval", action="store_true", help="skip the end-of-run MI/swap eval (timing)")
    a = ap.parse_args()

    cfg = _normalize_config(yaml.safe_load((ROOT / "config.yaml").read_text()))
    T = int(cfg["ppo_rollout_steps"])
    gamma, lam = float(cfg["ppo_gamma"]), float(cfg["ppo_gae_lam"])
    nl = int(cfg["n_layers"])
    hidden = int(cfg["hidden_dim"])
    G, W, hr = a.grid, a.worlds, a.hearth_radius
    N = 2 * W
    HEARTHS_NP = np.asarray(generate_hearth_positions(G))
    HEARTHS = jnp.asarray(HEARTHS_NP)
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
        print(f"[comm-jit] warm start <- {a.warm_start}", flush=True)
    opt = create_optimizer(float(cfg["ppo_lr"]), float(cfg["ppo_max_grad_norm"]))
    opt_state = opt.init(params)
    popart = PopArt(beta=0.1)
    gpu = jax.devices()[0].device_kind
    is_listener_np = np.tile([False, True], W)
    is_listener = jnp.asarray(is_listener_np)
    ns, ne = L.nb_sigs_start, L.nb_sigs_start + int(cfg["signal_dim"])
    s0 = L.own_state_start
    informed = jnp.ones(N, bool) if a.arm == "ORACLE" else ~is_listener
    DEL = jnp.asarray(DELTAS)
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    dd = np.abs(cells[:, None, :] - HEARTHS_NP[None]); dd = np.minimum(dd, G - dd).max(-1)
    VALID = jnp.asarray(cells[(dd > hr).all(1)])

    # ---- step body: identical to comm_testbed.step_fn
    def step_core(params, carries, obs, k):
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
        return new_c, act, alarm, lp, o.values, lvq, jnp.stack(toks, -1), jnp.concatenate(bc, -1), tent, logits

    # ---- obs builder: identical formulas to comm_testbed.World.build_obs
    def build_obs(pos, active, wire, k):
        obs = jnp.zeros((N, L.total_dim), jnp.float32)
        obs = obs.at[:, s0 + 2].set(1.0).at[:, s0 + 3].set(nl / 6.0)
        obs = obs.at[:, s0 + 4].set(pos[:, 0] / G).at[:, s0 + 5].set(pos[:, 1] / G).at[:, s0 + 6].set(1.0)
        diff = (HEARTHS[None, :, :] - pos[:, None, :] + G / 2) % G - G / 2
        obs = obs.at[:, s0 + 13: s0 + 21].set((diff / (G / 2.0)).reshape(N, 8).astype(jnp.float32))
        oh = jax.nn.one_hot(jnp.repeat(active, 2), 4, dtype=jnp.float32)
        obs = obs.at[:, s0 + 12].set(informed.astype(jnp.float32))
        obs = obs.at[:, s0 + 21: s0 + 25].set(oh * informed[:, None])
        obs = obs.at[:, s0 + 25: s0 + 29].set(jax.random.normal(k, (N, 4)))
        if a.arm == "COMM":
            obs = obs.at[:, ns:ne].set(wire.reshape(W, 2, -1)[:, ::-1].reshape(N, -1))
        return obs

    _t = np.zeros((2, 2, 2), np.float64)
    for d_ in (0, 1):
        for c_ in (0, 1):
            for s_ in (0, 1):
                v = (R_DEP if d_ else 0.0) - a.craft_cost * float(c_)   # np.where(dep, R_DEP, 0.0) - cost * (craft & ~dep)
                v = v + R_DEP * float(s_)                              # r[~is_listener] += R_DEP * ldep_w
                _t[d_, c_, s_] = v
    R_TABLE = jnp.asarray(_t.astype(np.float32))

    def near_any(pos):
        d = jnp.abs(pos[:, None, :] - HEARTHS[None]); d = jnp.minimum(d, G - d).max(-1)
        return d <= hr

    @jax.jit
    def rollout(params, state, key, resp, inc):
        """T steps. resp: (T, N, 2) respawn cells; inc: (T, W) re-roll increments, consumed in world order."""
        def step(carry, x):
            pos, active, since, wire, carries, key = carry
            resp_t, inc_t = x
            key, k_obs, k_act = jax.random.split(key, 3)
            obs = build_obs(pos, active, wire, k_obs)
            new_c, act, alarm, lp, val, lvq, toks, bc, tent, logits = step_core(params, carries, obs, k_act)
            new = (pos + DEL[act]) % G
            near = near_any(pos)
            craft = act == ACTION_CRAFT
            at_active = near[jnp.arange(N), jnp.repeat(active, 2)]
            dep = craft & at_active
            att = craft & near.any(1)
            # Reward via a table built in float64 in the reference's order and then cast to float32, so the stored
            # float32 reward is bitwise the reference's (it sums in float64; summing in float32 differed by 1 ulp)
            ldep_w = dep[is_listener]
            shared = (~is_listener) & jnp.repeat(ldep_w, 2)
            r = R_TABLE[dep.astype(jnp.int32), (craft & ~dep).astype(jnp.int32), shared.astype(jnp.int32)]
            pos2 = jnp.where(dep[:, None], resp_t, new)
            dep_w = dep.reshape(W, 2).any(1)
            since2 = since + 1
            reroll = dep_w | (since2 >= REROLL_STEPS)
            rank = jnp.cumsum(reroll) - 1                                  # k-th re-rolling world takes inc_t[k]
            active2 = jnp.where(reroll, (active + inc_t[jnp.clip(rank, 0, W - 1)]) % 4, active)
            since2 = jnp.where(reroll, 0, since2)
            wire2 = bc if a.arm == "COMM" else jnp.zeros_like(bc)
            rec = dict(obs=obs, actions=act, alarm_actions=alarm, log_probs=lp, rewards=r.astype(jnp.float32),
                       values=val, carries=carries, loss_vq=lvq, token_actions=toks, tent=tent, dep=dep, att=att,
                       active=active, reroll=reroll, bc=bc, logits=logits)
            return (pos2, active2, since2, wire2, new_c, key), rec
        carry, recs = jax.lax.scan(step, state, (resp, inc))
        return carry, recs

    def env_draws(k):
        k1, k2 = jax.random.split(k)
        return (VALID[jax.random.randint(k1, (T, N), 0, VALID.shape[0])],
                jax.random.randint(k2, (T, W), 1, 4))

    if a.equiv:
        D = np.load(a.equiv + ".draws.npz")
        st0 = (jnp.asarray(D["init_pos"]), jnp.asarray(D["init_active"]), jnp.zeros(W, jnp.int32),
               jnp.zeros((N, int(cfg["signal_dim"])), jnp.float32), jnp.zeros((N, hidden)),
               jax.random.PRNGKey(int(D["key_seed"])))
        _, R = rollout(params, st0, None, jnp.asarray(D["resp"]), jnp.asarray(D["inc"]))
        np.savez(a.equiv + ".jit.npz", **{k: np.asarray(v) for k, v in R.items()})
        print(f"[equiv] jit rollout saved: {R['actions'].shape[0]} steps", flush=True)
        return

    # ---- eval rollout (MI / swaps); the TRAINING rollout above is untouched
    @jax.jit
    def rollout_eval(params, state, resp, inc, table, mode):
        """mode 0 = control, 1 = constructed swap (sender's own policy sample for T' = (T + 1) % 4, same pre-step
        carry and key, as comm_testbed's swap_tokens), 2 = empirical swap (a full token triple drawn from the
        triples this sender emitted for T' in the paired control rollout)."""
        cbs = [(params["params"] if "params" in params else params)[k]["embedding"] for _, _, k in rl_jax.TOKEN_SLOTS]

        def step(carry, x):
            pos, active, since, wire, carries, key = carry
            resp_t, inc_t = x
            key, k_obs, k_act = jax.random.split(key, 3)
            obs = build_obs(pos, active, wire, k_obs)
            new_c, act, alarm, lp, val, lvq, toks, bc, tent, logits = step_core(params, carries, obs, k_act)
            tp = (active + 1) % 4
            sw_obs = build_obs(pos, tp, wire, k_obs)
            _, _, _, _, _, _, toks_sw, bc_sw, _, _ = step_core(params, carries, sw_obs, k_act)
            idx = jax.random.randint(jax.random.fold_in(k_act, 77), (N,), 0, table.shape[1])
            trip = table[jnp.repeat(tp, 2), idx]
            bc_emp = jnp.concatenate([jnp.zeros((N, 8))] + [cbs[i][trip[:, i]] for i in range(3)], -1)
            used_bc = jnp.where(mode == 1, bc_sw, bc_emp)
            used_tok = jnp.where(mode == 1, toks_sw, trip)
            bc_out = jnp.where((mode > 0) & ~is_listener[:, None], used_bc, bc)
            new = (pos + DEL[act]) % G
            near = near_any(pos)
            craft = act == ACTION_CRAFT
            dep = craft & near[jnp.arange(N), jnp.repeat(active, 2)]
            att_tp = craft & near[jnp.arange(N), jnp.repeat(tp, 2)]
            mv = DEL[act].astype(jnp.float32)

            def mu(tgt):
                h = HEARTHS[jnp.repeat(tgt, 2)]
                dv = (h - pos + G / 2) % G - G / 2
                nr = jnp.linalg.norm(dv.astype(jnp.float32), axis=1)
                m = is_listener & (act >= 1) & (act <= 4) & (nr > 0)
                return jnp.where(m, (mv * dv).sum(1) / jnp.maximum(nr, 1e-6), 0.0), m
            mu_sw, m_sw = mu(tp)
            mu_tr, m_tr = mu(active)
            pos2 = jnp.where(dep[:, None], resp_t, new)
            dep_w = dep.reshape(W, 2).any(1)
            since2 = since + 1
            reroll = dep_w | (since2 >= REROLL_STEPS)
            rank = jnp.cumsum(reroll) - 1
            active2 = jnp.where(reroll, (active + inc_t[jnp.clip(rank, 0, W - 1)]) % 4, active)
            since2 = jnp.where(reroll, 0, since2)
            wire2 = bc_out if a.arm == "COMM" else jnp.zeros_like(bc_out)
            heard_next = wire2.reshape(W, 2, -1)[:, ::-1].reshape(N, -1)
            rec = dict(token_actions=toks, used_tok=used_tok, used_bc=used_bc, heard_next=heard_next, active=active,
                       dep=dep, att_tp=att_tp, mu_sw=mu_sw, m_sw=m_sw, mu_tr=mu_tr, m_tr=m_tr, reroll=reroll)
            return (pos2, active2, since2, wire2, new_c, key), rec
        _, recs = jax.lax.scan(step, state, (resp, inc))
        return recs

    from comm_testbed import mi_miller_madow
    V_ = int(cfg["vocab_size"])

    def mi_with_null(tok, lab, epoch, rng, perms):
        """Miller-Madow MI per slot (bits) of sender tokens vs the active hearth, mean over slots, and the p95 of a
        block-permutation null (labels permuted across re-roll epochs, as in comm_testbed)."""
        uniq, inv = np.unique(epoch, return_inverse=True)
        ep_lab = np.zeros(len(uniq), int); ep_lab[inv] = lab
        mi = float(np.mean([mi_miller_madow(tok[:, i], lab, V_, 4) for i in range(3)]))
        null = [float(np.mean([mi_miller_madow(tok[:, i], rng.permutation(ep_lab)[inv], V_, 4) for i in range(3)]))
                for _ in range(perms)]
        return mi, float(np.percentile(null, 95)) if perms else float("nan")

    def epochs_of(reroll):     # (T, W) bool -> epoch id per (t, w), new epoch after each re-roll
        c = np.cumsum(np.vstack([np.zeros((1, W), int), reroll[:-1].astype(int)]), 0)
        return (c * W + np.arange(W)[None]).ravel()

    def evaluate(params, u, buf):
        rng_e = np.random.default_rng(1000 + 7919 * a.seed + u)
        if buf:
            tok = np.concatenate([b[0] for b in buf]); lab = np.concatenate([b[1] for b in buf])
            ep = np.concatenate([b[2] + 10 ** 7 * j for j, b in enumerate(buf)])
            uniq, inv = np.unique(ep, return_inverse=True)
            ep_lab = np.zeros(len(uniq), int); ep_lab[inv] = lab
            for i in range(3):
                mi = mi_miller_madow(tok[:, i], lab, V_, 4)
                null = np.array([mi_miller_madow(tok[:, i], rng_e.permutation(ep_lab)[inv], V_, 4) for _ in range(1000)])
                emit(dict(base, phase="mi", update=u, slot=i, mi_bits=mi, null_p99=float(np.percentile(null, 99)),
                          null_mean=float(null.mean()), n=int(len(lab)), epochs=int(len(uniq))))
        if a.arm != "COMM":
            return
        enc = lambda x: (x[..., 0] * V_ + x[..., 1]) * V_ + x[..., 2]  # noqa: E731
        for k in range(a.swap_pairs):
            ek = jax.random.PRNGKey(10_000 + k)
            k0, k1, kd = jax.random.split(ek, 3)
            st0 = (VALID[jax.random.randint(k0, (N,), 0, VALID.shape[0])], jax.random.randint(k1, (W,), 0, 4),
                   jnp.zeros(W, jnp.int32), jnp.zeros((N, int(cfg["signal_dim"])), jnp.float32), jnp.zeros((N, hidden)),
                   jax.random.PRNGKey(20_000 + k))
            resp, inc = env_draws(kd)
            dummy = jnp.zeros((4, 512, 3), jnp.int32)
            Rc = jax.tree_util.tree_map(np.asarray, rollout_eval(params, st0, resp, inc, dummy, 0))
            trip_c, lab = Rc["token_actions"][:, ~is_listener_np], Rc["active"]
            sets = [set(enc(trip_c[lab == t]).tolist()) for t in range(4)]
            table = np.zeros((4, 512, 3), np.int32)
            for t in range(4):
                tt = trip_c[lab == t]
                if len(tt) == 0:
                    print(f"[APPARATUS] FATAL: no control tokens for hearth {t}", flush=True); raise SystemExit(1)
                table[t] = tt[rng_e.integers(0, len(tt), 512)]
            c_mu_sw = Rc["mu_sw"].sum() / max(Rc["m_sw"].sum(), 1); c_mu_tr = Rc["mu_tr"].sum() / max(Rc["m_tr"].sum(), 1)
            c_dep = float(Rc["dep"][:, is_listener_np].sum() / W); c_att = float(Rc["att_tp"][:, is_listener_np].sum() / W)
            for mode, name in ((1, "constructed"), (2, "empirical")):
                Rk = jax.tree_util.tree_map(np.asarray, rollout_eval(params, st0, resp, inc, jnp.asarray(table), mode))
                if not np.allclose(Rk["heard_next"][:, is_listener_np], Rk["used_bc"][:, ~is_listener_np]):
                    print("[APPARATUS] FATAL: swap wiring", flush=True); raise SystemExit(1)
                used = Rk["used_tok"][:, ~is_listener_np]; lab_sw = (Rk["active"] + 1) % 4
                seen = float(np.mean([e in sets[t] for e, t in zip(enc(used).ravel().tolist(), lab_sw.ravel().tolist())]))
                emit(dict(base, phase="swap", update=u, mode=name, pair=k, mu_Tprime_control=float(c_mu_sw),
                          mu_Tprime_swap=float(Rk["mu_sw"].sum() / max(Rk["m_sw"].sum(), 1)), mu_T_control=float(c_mu_tr),
                          mu_T_swap=float(Rk["mu_tr"].sum() / max(Rk["m_tr"].sum(), 1)), listener_dep_control=c_dep,
                          listener_dep_swap=float(Rk["dep"][:, is_listener_np].sum() / W),
                          listener_crafts_at_Tprime_control=c_att,
                          listener_crafts_at_Tprime_swap=float(Rk["att_tp"][:, is_listener_np].sum() / W),
                          swapped_token_differs_frac=float(np.any(used != Rk["token_actions"][:, ~is_listener_np], -1).mean()),
                          swap_token_seen_for_Tprime=seen))

    def emit(rec):
        if a.timing_only:
            print(json.dumps({k: rec.get(k) for k in ("arm", "seed", "update", "t_update", "t_rollout", "t_ppo")}),
                  flush=True)
            return
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    base = dict(arm=a.arm, seed=a.seed, grid=G, gpu=gpu, tau=a.tau, warm_start=bool(a.warm_start),
                craft_cost=a.craft_cost, impl="jit")
    env_key = jax.random.PRNGKey(10_007 + a.seed)
    env_key, k0, k1 = jax.random.split(env_key, 3)
    state = (VALID[jax.random.randint(k0, (N,), 0, VALID.shape[0])], jax.random.randint(k1, (W,), 0, 4),
             jnp.zeros(W, jnp.int32), jnp.zeros((N, int(cfg["signal_dim"])), jnp.float32), jnp.zeros((N, hidden)),
             None)
    halt_sdep = []
    from collections import deque
    mi_buf = deque(maxlen=10)
    for u in range(1, a.updates + 1):
        t0 = time.time()
        env_key, k_env = jax.random.split(env_key)
        resp, inc = env_draws(k_env)
        state = state[:5] + (key,)
        state, R = rollout(params, state, None, resp, inc)
        key = state[5]
        R = jax.tree_util.tree_map(np.asarray, R)
        t_roll = time.time() - t0
        # apparatus, as comm_testbed.apparatus at t == 0 and every 64 steps
        p_ = params["params"] if "params" in params else params
        for t in range(0, T, 64):
            obs_t = R["obs"][t]
            snd, lst = obs_t[~is_listener_np, s0 + 21: s0 + 25], obs_t[is_listener_np, s0 + 21: s0 + 25]
            ok = np.array_equal(snd.argmax(1), R["active"][t]) and np.allclose(snd.sum(1), 1)
            ok &= np.allclose(lst.sum(1), 1) if a.arm == "ORACLE" else bool((lst == 0).all())
            for i, (lo, hi, cbk) in enumerate(rl_jax.TOKEN_SLOTS):
                ok &= np.allclose(R["bc"][t][:, lo:hi], np.asarray(p_[cbk]["embedding"])[R["token_actions"][t][:, i]])
            if a.arm != "COMM":
                ok &= bool((obs_t[:, ns:ne] == 0).all())
            if not ok:
                print("[APPARATUS] FATAL: target dims / one-hot / token-broadcast / wire check failed", flush=True)
                raise SystemExit(1)
        batch = {k: jnp.asarray(R[k]) for k in ("obs", "actions", "alarm_actions", "log_probs", "rewards", "carries",
                                                 "loss_vq", "token_actions")}
        values_np = popart.denormalise(R["values"])
        batch["values"] = jnp.asarray(values_np)
        batch["dones"] = jnp.zeros((T, N), jnp.float32)
        batch["alive"] = jnp.ones((T, N), bool)
        rewards_np = R["rewards"]
        key, k_obs, k_act = jax.random.split(key, 3)
        o_last = build_obs(state[0], state[1], state[3], k_obs)
        v_last = popart.denormalise(np.asarray(step_core(params, state[4], o_last, k_act)[4]))
        rl_jax.compute_gae = lambda r_, v_, d_, gamma, lam, **kw: gae_bootstrap(r_, v_, d_, v_last, gamma, lam)
        _, targets = gae_bootstrap(rewards_np, values_np, np.zeros_like(rewards_np), v_last, gamma, lam)
        key, k_up = jax.random.split(key)
        t1 = time.time()
        params, opt_state, mets = ppo_update(
            params, opt_state, opt, apply_fn, batch, nl, k_up, clip_eps=float(cfg.get("ppo_clip_eps", 0.2)),
            vf_coef=a.vf_coef, ent_coef=float(cfg.get("ppo_entropy_coef", 0.02)),
            vq_coef=float(cfg.get("vq_loss_coef", 0.1)), minibatch_size=int(cfg["ppo_minibatch_size"]), gamma=gamma,
            lam=lam, team="blue", alarm_ent_coef=float(cfg.get("alarm_ent_coef", 0.0)), freeze_comms=False,
            value_clip=False, popart=popart, token_tau=a.tau, token_ent_coef=a.token_ent_coef,
            ps_coef=a.ps, ps_cols=((s0 + 12, s0 + 21) if a.ps > 0 else None))
        rl_jax.compute_gae = _production_compute_gae
        t_ppo = time.time() - t1
        ev = 1 - np.var(targets - values_np) / (np.var(targets) + 1e-8)
        dep, att = R["dep"], R["att"]
        s_dep = int(dep[:, ~is_listener_np].sum())
        halt_sdep.append(s_dep / W)
        if a.halt_update and u == a.halt_update:
            sdep = float(np.mean(halt_sdep))
            ok = sdep >= a.halt_sender_dep
            print(f"[AUTO-HALT CHECK] update {u}: sender dep/world {sdep:.2f} (need >= {a.halt_sender_dep}) -> "
                  f"{'continue' if ok else 'HALT'}", flush=True)
            if not ok:
                emit(dict(base, phase="halt", update=u, sender_dep=sdep)); raise SystemExit(3)
        snd_tok = R["token_actions"][:, ~is_listener_np].reshape(-1, 3)
        snd_lab = R["active"].reshape(-1); snd_ep = epochs_of(R["reroll"])
        mi_buf.append((snd_tok, snd_lab, snd_ep))
        mi_u, mi_null95 = mi_with_null(snd_tok, snd_lab, snd_ep, np.random.default_rng(u), a.mi_null_perms)
        if a.apparatus_update and u == a.apparatus_update:
            chk = {}
            if a.ps > 0:
                mbo = jnp.asarray(R["obs"][:, :].reshape(-1, L.total_dim)[:512])
                mbc = jnp.asarray(R["carries"].reshape(-1, hidden)[:512])

                def _psf(p):
                    _, o_ = apply_fn(p, mbc, mbo, nl)
                    return rl_jax.positive_signalling_mi(p, o_.z_e, mbo, a.tau, (s0 + 12, s0 + 21))
                gps = jax.grad(_psf)(params); gpp = gps["params"] if "params" in gps else gps
                chk["ps_grad_message_heads"] = float(np.sqrt(sum(float(jnp.sum(x ** 2)) for h in
                                                     ("head_signal_slot0", "head_signal_slot1", "head_signal_slot2")
                                                     for x in jax.tree_util.tree_leaves(gpp[h]))))
            chk["distinct_sender_messages"] = int(len(np.unique(snd_tok, axis=0)))
            if a.arm == "COMM":
                lw = R["obs"][1:, is_listener_np, ns:ne]
                chk["listener_wire_nonzero_frac"] = float((np.abs(lw).sum(-1) > 0).mean())
            ok = chk.get("ps_grad_message_heads", 1.0) > 0 and chk["distinct_sender_messages"] > 1 and \
                chk.get("listener_wire_nonzero_frac", 1.0) > 0
            print(f"[APPARATUS CHECK] update {u}: {chk} -> {'continue' if ok else 'HALT'}", flush=True)
            emit(dict(base, phase="apparatus", update=u, ok=bool(ok), **chk))
            if not ok:
                raise SystemExit(3)
        l_att = int(att[:, is_listener_np].sum()); l_dep = int(dep[:, is_listener_np].sum())
        emit(dict(base, phase="train", update=u, listener_dep_per_world=l_dep / W, sender_dep_per_world=s_dep / W,
                  listener_attempts=l_att, sender_attempts=int(att[:, ~is_listener_np].sum()),
                  listener_correct_frac=l_dep / max(l_att, 1), token_entropy=float(R["tent"].mean()), ev=float(ev),
                  entropy=float(mets.get("ppo_entropy", np.nan)),
                  ppo_token_entropy=float(mets.get("ppo_token_entropy", np.nan)),
                  mi_bits=mi_u, mi_null_p95=mi_null95, ppo_ps_mi=float(mets.get('ppo_ps_mi', 0.0)),
                  t_update=round(time.time() - t0, 2), t_rollout=round(t_roll, 2), t_ppo=round(t_ppo, 2)))
        if u in a.eval_at and u != a.updates:
            evaluate(params, u, list(mi_buf)[-10:])

    if not a.no_eval:
        evaluate(params, a.updates, list(mi_buf))
    if a.save_params:
        import flax.serialization
        Path(a.save_params).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save_params).write_bytes(flax.serialization.to_bytes(params))


if __name__ == "__main__":
    main()
