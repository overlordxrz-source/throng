"""CPU toy of the comm testbed (Cam, 2026-11-07): reproduce the live-wire token collapse cheaply, then test
positive signalling / positive listening (Eccles et al. 2019) before any GPU money.

KEPT IDENTICAL to scripts/comm_testbed.py (ceadaf8): the world rules (four production hearths via
generate_hearth_positions, sender = informed / listener = uninformed, one active hearth, Craft within radius 1 of it
deposits 2.7, depositor respawns away from hearths, active re-rolls on ANY deposit in the world or after REROLL
steps, listener deposit pays the sender 2.7, cost per non-deposit Craft), the symmetric one-step-delayed wire
(each agent's broadcast = its sampled tokens' codebook vectors, heard by its partner), the token policy
(rl_jax.token_logits, tau 0.08, log-prob in the PPO ratio, entropy coef 0.01), the VQ loss
(network_jax.vector_quantize_signals, beta 0.25, vq_coef 0.1), rl_jax.ppo_loss itself, C' (learner_testbed.
gae_bootstrap: GAE 0.95 + bootstrap, gamma 0.999), PopArt (rl_jax.PopArt, beta 0.1), vf 0.5, ent 0.02, Adam 3e-4,
the comms pathway (separate feed-forward branch from the obs, as network_jax), the /2 action-logit temperature,
the per-group (head_action / head_value / trunk) gradient clip at 2.0, minibatch 512, one epoch, the paired
token-swap eval and its apparatus check, the hearth_rel formula, the 4 noise dims.

CHANGED (the toy): G = 12 (hearths at 3/9), REROLL = 96 (256 x 12/32, rounded), T = 128 rollout steps, a small
MLP + GRU (64) instead of AgentNetworkJax, a 59-dim obs (pos, hearth_rel, informed flag, need one-hot, 4 noise
dims, 40-dim wire) instead of the production layout, no alarm head, the whole rollout jitted with lax.scan.

Arms: ORACLE (wire zeroed, both agents informed) | NOSIG (wire zeroed) | COMM (+ --ps / --pl aux losses).
  --ps C   positive signalling: + C * sum_slots MI(token; active hearth) on SENDER rows (batch estimate,
           H(mean p) - sum_T w_T H(mean p | T); maximised).
  --pl C   positive listening: + C * mean L1(pi(a | wire), stopgrad pi(a | wire = 0)) on LISTENER rows (maximised).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import flax.linen as nn  # noqa: E402
import flax.serialization  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from jax_sim import rl_jax  # noqa: E402
from jax_sim.action_space import mask_disabled_actions  # noqa: E402
from jax_sim.grid_jax import generate_hearth_positions  # noqa: E402
from jax_sim.network_jax import vector_quantize_signals  # noqa: E402
from jax_sim.rl_jax import PopArt, ppo_loss, token_logits  # noqa: E402
from learner_testbed import ACTION_CRAFT, DELTAS, gae_bootstrap  # noqa: E402

R_DEP = 2.7
N_ACT, V, SIG = 12, 64, 40
GAMMA, LAM, LR, CLIP, ENT, VQC, VF, MB = 0.999, 0.95, 3e-4, 0.2, 0.02, 0.1, 0.5, 512
TAU, TOK_ENT = 0.08, 0.01
# obs layout (toy)
O_POS, O_HR, O_INF, O_NEED, O_NOISE, O_WIRE, D = 0, 2, 10, 11, 15, 19, 59


class Outs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ToyNet(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, carries, obs):
        x = nn.relu(nn.Dense(self.hidden, name="enc_0")(obs))
        x = nn.relu(nn.Dense(self.hidden, name="enc_1")(x))
        new_c, h = nn.GRUCell(self.hidden, name="gru")(carries, x)
        logits = nn.Dense(N_ACT, name="head_action")(h) / 2.0       # production temperature 2.0
        v = nn.Dense(1, name="head_value")(h)[..., 0]
        # Comms pathway as in production (network_jax: gwt_comms_1 -> head_signal_slot{0,1,2}): a separate
        # feed-forward branch from the CURRENT obs, independent of the recurrent trunk; z_e[:, :8] = 0.
        h_c = nn.relu(nn.Dense(self.hidden, name="gwt_comms_1")(obs))
        zs, lvq = [jnp.zeros((obs.shape[0], 8))], []
        for i, (lo, hi, k) in enumerate(rl_jax.TOKEN_SLOTS):
            z = nn.Dense(hi - lo, name=f"head_signal_slot{i}")(h_c)
            cb = nn.Embed(V, hi - lo, name=k).embedding
            _, _, l = vector_quantize_signals(z, cb, beta=0.25)
            zs.append(z); lvq.append(l)
        z_e = jnp.concatenate(zs, -1)
        return new_c, Outs(action_logits=logits, values=v, z_e=z_e, loss_vq=jnp.stack(lvq, -1), alarm_out=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["COMM", "NOSIG", "ORACLE"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--reroll", type=int, default=96)
    ap.add_argument("--craft-cost", type=float, default=0.45)
    ap.add_argument("--ps", type=float, default=0.0)
    ap.add_argument("--pl", type=float, default=0.0)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--save-params", default=None)
    ap.add_argument("--swap-pairs", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--wire-init-scale", type=float, default=1.0,
                    help="after warm start, scale the (never-trained) wire-input rows of enc_0 and gwt_comms_1 (fidelity calibration)")
    ap.add_argument("--measure-sens", action="store_true", help="print listener sensitivity at the loaded params and exit")
    a = ap.parse_args()

    G, W, T, H = a.grid, a.worlds, a.steps, 64
    N = 2 * W
    HEARTHS = jnp.asarray(np.asarray(generate_hearth_positions(G)))
    is_listener = jnp.tile(jnp.array([False, True]), W)
    live_wire = a.arm == "COMM"
    oracle = a.arm == "ORACLE"
    informed = jnp.ones(N, bool) if oracle else ~is_listener
    DEL = jnp.asarray(DELTAS)
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    hn = np.asarray(HEARTHS)
    dd = np.abs(cells[:, None, :] - hn[None]); dd = np.minimum(dd, G - dd).max(-1)
    VALID = jnp.asarray(cells[(dd > 1).all(1)])           # respawn cells: not within radius 1 of any hearth

    model = ToyNet(H)

    def apply_fn(params, carries, obs, n_layers=1, detach_value=False, **kw):
        return model.apply(params, carries, obs)

    key = jax.random.PRNGKey(a.seed)
    key, k_init = jax.random.split(key)
    params = model.init(k_init, jnp.zeros((1, H)), jnp.zeros((1, D)))
    if a.warm_start:
        params = flax.serialization.from_bytes(params, Path(a.warm_start).read_bytes())
        if a.wire_init_scale != 1.0:
            # The wire was zero throughout ORACLE training, so these rows are still at init (verified bit-identical).
            # Rescaling them sets the listener's start sensitivity to the real network's without scaling the input
            # (an input gain would also scale the learning signal under Adam).
            pp = {k: dict(v) for k, v in params["params"].items()}
            for n in ("enc_0", "gwt_comms_1"):
                pp[n]["kernel"] = jnp.asarray(pp[n]["kernel"]).at[O_WIRE:].multiply(a.wire_init_scale)
            params = {"params": pp}
    opt = optax.adam(LR)
    opt_state = opt.init(params)
    popart = PopArt(beta=0.1)

    # ---------------------------------------------------------------- env (jnp)
    def near(pos):                                                     # (n, 4) Chebyshev radius 1, torus
        d = jnp.abs(pos[:, None, :] - HEARTHS[None]); d = jnp.minimum(d, G - d).max(-1)
        return d <= 1

    def build_obs(pos, active, wire, k):
        diff = (HEARTHS[None] - pos[:, None, :] + G / 2) % G - G / 2
        hr = (diff / (G / 2.0)).reshape(N, 8)
        need = jax.nn.one_hot(jnp.repeat(active, 2), 4) * informed[:, None]
        heard = wire.reshape(W, 2, -1)[:, ::-1].reshape(N, -1) if live_wire else jnp.zeros((N, SIG))
        return jnp.concatenate([pos / G, hr, informed[:, None].astype(jnp.float32), need,
                                jax.random.normal(k, (N, 4)), heard], -1)

    def policy(params, carries, obs, k):
        new_c, o = apply_fn(params, carries, obs)
        logits = mask_disabled_actions(o.action_logits, axis=-1)
        k1, k3 = jax.random.split(k)
        act = jax.random.categorical(k1, logits)
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), act[:, None], -1)[:, 0]
        p = params["params"]
        toks, bc, ent = [], [jnp.zeros((N, 8))], []
        for i, lg in enumerate(token_logits(params, o.z_e, TAU)):
            t = jax.random.categorical(jax.random.fold_in(k3, i), lg)
            lsm = jax.nn.log_softmax(lg, -1)
            lp = lp + jnp.take_along_axis(lsm, t[:, None], -1)[:, 0]
            ent.append(-jnp.sum(jnp.exp(lsm) * lsm, -1))
            toks.append(t); bc.append(p[rl_jax.TOKEN_SLOTS[i][2]]["embedding"][t])
        return (new_c, logits, act, lp, o.values, o.loss_vq.sum(-1), jnp.stack(toks, -1),
                jnp.concatenate(bc, -1), jnp.stack(ent, -1).sum(-1))

    def tv(l1, l2):
        p1, p2 = jax.nn.softmax(l1, -1), jax.nn.softmax(l2, -1)
        return 0.5 * jnp.abs(p1 - p2).sum(-1)

    @functools.partial(jax.jit, static_argnames=("swap",))
    def rollout(params, state, key, swap=False):
        """lax.scan over T steps. swap=True (eval): the listener hears what its sender WOULD broadcast for
        T' = (T + 1) % 4 (same pre-step carry, same key); the sender's own action/reward come from the true T."""
        cb = [params["params"][k]["embedding"] for _, _, k in rl_jax.TOKEN_SLOTS]

        def step(st, k):
            pos, active, since, wire, carries = st
            k_obs, k_act, k_rs, k_rr, k_rand = jax.random.split(k, 5)
            obs = build_obs(pos, active, wire, k_obs)
            new_c, logits, act, lp, val, lvq, toks, bc, ent = policy(params, carries, obs, k_act)
            # sender's counterfactual broadcast for T' (paired key) -- used by the swap eval and the swap-diff metric
            obs_sw = build_obs(pos, (active + 1) % 4, wire, k_obs)
            _, _, _, _, _, _, toks_sw, bc_sw, _ = policy(params, carries, obs_sw, k_act)
            swap_diff = jnp.any(toks_sw != toks, -1) & ~is_listener
            # listener sensitivity: received wire replaced by a uniform random token triple (diagnostic only)
            rt = jax.random.randint(k_rand, (N, 3), 0, V)
            rbc = jnp.concatenate([jnp.zeros((N, 8))] + [cb[i][rt[:, i]] for i in range(3)], -1)
            w_r = jnp.where(is_listener[:, None], wire, rbc)          # replace the SENDER's entry (partner of listener)
            _, o_r = apply_fn(params, carries, build_obs(pos, active, w_r, k_obs))
            sens = tv(logits, mask_disabled_actions(o_r.action_logits, axis=-1))
            bc_out = jnp.where(is_listener[:, None], bc, bc_sw) if swap else bc
            # movement toward T' and T (listeners)
            mv = DEL[act].astype(jnp.float32)
            def mu(tgt):
                h = HEARTHS[jnp.repeat(tgt, 2)]
                dv = (h - pos + G / 2) % G - G / 2
                nr = jnp.linalg.norm(dv, axis=1)
                m = is_listener & (act >= 1) & (act <= 4) & (nr > 0)
                return jnp.where(m, (mv * dv).sum(1) / jnp.maximum(nr, 1e-6), 0.0), m
            mu_sw, m_sw = mu((active + 1) % 4)
            mu_tr, m_tr = mu(active)
            nr_ = near(pos)
            craft = act == ACTION_CRAFT
            at_active = nr_[jnp.arange(N), jnp.repeat(active, 2)]
            dep = craft & at_active
            att = craft & nr_.any(1)
            r = jnp.where(dep, R_DEP, 0.0) - a.craft_cost * (craft & ~dep)
            ldep_w = dep[is_listener]
            r = r + jnp.where(~is_listener, R_DEP * jnp.repeat(ldep_w, 2), 0.0)   # shared reward to the sender
            newpos = (pos + DEL[act]) % G
            rs = VALID[jax.random.randint(k_rs, (N,), 0, VALID.shape[0])]
            pos2 = jnp.where(dep[:, None], rs, newpos)
            dep_w = dep.reshape(W, 2).any(1)
            since2 = since + 1
            reroll = dep_w | (since2 >= a.reroll)
            active2 = jnp.where(reroll, (active + jax.random.randint(k_rr, (W,), 1, 4)) % 4, active)
            since2 = jnp.where(reroll, 0, since2)
            wire2 = bc_out if live_wire else jnp.zeros_like(bc_out)
            heard_next = wire2.reshape(W, 2, -1)[:, ::-1].reshape(N, -1)
            rec = dict(obs=obs, actions=act, log_probs=lp, rewards=r.astype(jnp.float32), values=val,
                       carries=carries, loss_vq=lvq, token_actions=toks, ent=ent, dep=dep, att=att,
                       swap_diff=swap_diff, sens=sens, mu_sw=mu_sw, m_sw=m_sw, mu_tr=mu_tr, m_tr=m_tr,
                       active=active, heard_next=heard_next, bc_sw=bc_sw)
            return (pos2, active2, since2, wire2, new_c), rec

        ks = jax.random.split(key, T)
        st, recs = jax.lax.scan(step, state, ks)
        return st, recs

    def init_state(r_key):
        k1, k2 = jax.random.split(r_key)
        pos = VALID[jax.random.randint(k1, (N,), 0, VALID.shape[0])]
        return (pos, jax.random.randint(k2, (W,), 0, 4), jnp.zeros(W, jnp.int32), jnp.zeros((N, SIG)),
                jnp.zeros((N, H)))

    # ---------------------------------------------------------------- aux losses
    def ps_term(params, obs, carries, role_mask=True):
        """MI(token; active hearth) on sender rows, summed over the 3 slots (nats); batch estimate."""
        _, o = apply_fn(params, carries, obs)
        snd = obs[:, O_INF] > 0.5
        if oracle:
            snd = snd & False
        if not role_mask:          # mutation-test only
            snd = jnp.ones_like(snd)
        Tl = jnp.argmax(obs[:, O_NEED:O_NEED + 4], -1)
        w_s = snd.astype(jnp.float32)
        mi = 0.0
        for lg in token_logits(params, o.z_e, TAU):
            p = jax.nn.softmax(lg, -1)
            pbar = (p * w_s[:, None]).sum(0) / (w_s.sum() + 1e-8)
            h_all = -(pbar * jnp.log(pbar + 1e-10)).sum()
            h_cond = 0.0
            for tt in range(4):
                wt = w_s * (Tl == tt)
                pt = (p * wt[:, None]).sum(0) / (wt.sum() + 1e-8)
                h_cond = h_cond + (wt.sum() / (w_s.sum() + 1e-8)) * -(pt * jnp.log(pt + 1e-10)).sum()
            mi = mi + h_all - h_cond
        return mi

    def pl_term(params, obs, carries, role_mask=True):
        """mean L1 between the listener's action distribution with the wire and (stop-grad) with the wire zeroed."""
        _, o = apply_fn(params, carries, obs)
        obs0 = obs.at[:, O_WIRE:].set(0.0)
        _, o0 = apply_fn(params, carries, obs0)
        p = jax.nn.softmax(mask_disabled_actions(o.action_logits, axis=-1), -1)
        p0 = jax.lax.stop_gradient(jax.nn.softmax(mask_disabled_actions(o0.action_logits, axis=-1), -1))
        lst = (obs[:, O_INF] < 0.5).astype(jnp.float32)
        if not role_mask:          # mutation-test only
            lst = jnp.ones_like(lst)
        return (jnp.abs(p - p0).sum(-1) * lst).sum() / (lst.sum() + 1e-8)

    def total_loss(params, mb, vs, vsh):
        loss, mets = ppo_loss(params, apply_fn, mb["obs"], mb["actions"], None, mb["log_probs"], mb["adv"],
                              mb["ret"], mb["carries"], 1, mb["values"], clip_eps=CLIP, vf_coef=VF, ent_coef=ENT,
                              vq_coef=VQC, value_scale=vs, value_clip=False, value_shift=vsh, popart_mode=True,
                              token_actions=mb["token_actions"], token_tau=TAU, token_ent_coef=TOK_ENT)
        ps = ps_term(params, mb["obs"], mb["carries"]) if a.ps else jnp.array(0.0)
        pl = pl_term(params, mb["obs"], mb["carries"]) if a.pl else jnp.array(0.0)
        mets = dict(mets, aux_ps=ps, aux_pl=pl)
        return loss - a.ps * ps - a.pl * pl, mets

    def _clip_group(g, max_norm=2.0):
        norm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
        s = jnp.minimum(1.0, max_norm / (norm + 1e-6))
        return jax.tree_util.tree_map(lambda x: x * s, g)

    @jax.jit
    def mb_step(params, opt_state, mb, vs, vsh):
        (loss, mets), grads = jax.value_and_grad(total_loss, has_aux=True)(params, mb, vs, vsh)
        g = dict(grads["params"])
        groups = [["head_action"], ["head_value"], [k for k in g if k not in ("head_action", "head_value")]]
        for grp in groups:
            c = _clip_group({k: g[k] for k in grp})
            g.update(c)
        grads = {"params": g}
        upd, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, mets

    # ---------------------------------------------------------------- self-test of the aux losses
    if a.selftest:
        st_, R0 = rollout(params, init_state(jax.random.PRNGKey(7)), jax.random.PRNGKey(8))
        obs = jnp.asarray(R0["obs"]).reshape(-1, D)[:2048]
        car = jnp.asarray(R0["carries"]).reshape(-1, H)[:2048]
        lst_rows = np.asarray(obs[:, O_INF] < 0.5)
        ok = True
        for name, fn, own_rows in (("PS", ps_term, ~lst_rows), ("PL", pl_term, lst_rows)):
            g_obs = np.asarray(jax.grad(lambda o: fn(params, o, car))(obs))
            other = np.abs(g_obs[~own_rows]).max(); own = np.abs(g_obs[own_rows]).max()
            g_p = jax.grad(lambda p: fn(p, obs, car))(params)
            gn = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g_p))))
            p2 = jax.tree_util.tree_map(lambda x, g: x + 1e-2 * g / (gn + 1e-12), params, g_p)
            before, after = float(fn(params, obs, car)), float(fn(p2, obs, car))
            res = dict(term=name, grad_obs_own_rows=float(own), grad_obs_other_rows=float(other), grad_norm=gn,
                       value_before=before, value_after_ascent=after)
            print(json.dumps(res), flush=True)
            ok &= other == 0.0 and own > 0 and gn > 0 and after > before
        # mutation: PS computed on LISTENER rows instead must be caught by the role check
        def ps_mut(p, o, c):
            return ps_term(p, o, c, role_mask=False)
        g_m = np.asarray(jax.grad(lambda o: ps_mut(params, o, car))(obs))
        caught = np.abs(g_m[lst_rows]).max() > 0
        print(json.dumps(dict(mutant="PS without the sender-row mask", caught=bool(caught))), flush=True)
        ok &= bool(caught)
        g_m2 = np.asarray(jax.grad(lambda o: pl_term(params, o, car, role_mask=False))(obs))
        caught2 = np.abs(g_m2[~lst_rows]).max() > 0
        print(json.dumps(dict(mutant="PL without the listener-row mask", caught=bool(caught2))), flush=True)
        ok &= bool(caught2)
        print("SELFTEST", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)

    if a.measure_sens:
        vals = []
        for k in range(4):
            _, Rm = rollout(params, init_state(jax.random.PRNGKey(50 + k)), jax.random.PRNGKey(60 + k))
            vals.append(float(np.asarray(Rm["sens"])[:, np.asarray(is_listener)].mean()))
        print(json.dumps(dict(wire_init_scale=a.wire_init_scale, listener_sens_tv=float(np.mean(vals)))), flush=True)
        raise SystemExit(0)

    # ---------------------------------------------------------------- coefficient calibration (registered rule)
    if a.calibrate:
        # c_aux = 0.25 * ||grad PPO|| / ||grad aux||, full-parameter L2 norms (pre-clip), mean over 4 minibatches
        # of 512 from one COMM rollout at the warm-start params.
        _, Rc = rollout(params, init_state(jax.random.PRNGKey(3)), jax.random.PRNGKey(4))
        Rc = jax.tree_util.tree_map(np.asarray, Rc)
        vals = popart.denormalise(Rc["values"])
        adv, ret = gae_bootstrap(Rc["rewards"], vals, np.zeros_like(Rc["rewards"]), vals[-1], GAMMA, LAM)
        params_c = popart.update_and_preserve(ret, params)
        M = T * N
        flat = {k: Rc[k].reshape((M,) + Rc[k].shape[2:]) for k in ("obs", "actions", "log_probs", "carries",
                                                                    "token_actions")}
        flat["values"], flat["adv"], flat["ret"] = vals.reshape(M), adv.reshape(M), ret.reshape(M).astype(np.float32)
        perm = np.random.default_rng(0).permutation(M)
        gn = lambda g: float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g))))  # noqa: E731
        r_ps, r_pl = [], []
        for i in range(4):
            mb = {k: jnp.asarray(v[perm[i * MB:(i + 1) * MB]]) for k, v in flat.items()}
            g_ppo = jax.grad(lambda p: ppo_loss(
                p, apply_fn, mb["obs"], mb["actions"], None, mb["log_probs"], mb["adv"], mb["ret"], mb["carries"], 1,
                mb["values"], clip_eps=CLIP, vf_coef=VF, ent_coef=ENT, vq_coef=VQC, value_scale=popart.sigma,
                value_clip=False, value_shift=popart.mu, popart_mode=True, token_actions=mb["token_actions"],
                token_tau=TAU, token_ent_coef=TOK_ENT)[0])(params_c)
            g_ps = jax.grad(lambda p: ps_term(p, mb["obs"], mb["carries"]))(params_c)
            g_pl = jax.grad(lambda p: pl_term(p, mb["obs"], mb["carries"]))(params_c)
            r_ps.append(gn(g_ppo) / gn(g_ps)); r_pl.append(gn(g_ppo) / gn(g_pl))
        print(json.dumps(dict(calibration="0.25 * |grad ppo| / |grad aux|", c_ps=0.25 * float(np.mean(r_ps)),
                              c_pl=0.25 * float(np.mean(r_pl)), ratios_ps=r_ps, ratios_pl=r_pl)), flush=True)
        raise SystemExit(0)

    # ---------------------------------------------------------------- loop
    def emit(rec):
        rec = dict(arm=a.arm, seed=a.seed, ps=a.ps, pl=a.pl, tag=a.tag, craft_cost=a.craft_cost,
                   warm_start=bool(a.warm_start), **rec)
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    np_rng = np.random.default_rng(a.seed)
    key, k_s = jax.random.split(key)
    state = init_state(k_s)
    lis = np.asarray(is_listener)
    mi_tok, mi_T = [], []
    for u in range(1, a.updates + 1):
        t0 = time.time()
        key, k_r, k_l = jax.random.split(key, 3)
        state, R = rollout(params, state, k_r)
        R = jax.tree_util.tree_map(np.asarray, R)
        values = popart.denormalise(R["values"])
        obs_last = build_obs(state[0], state[1], state[3], k_l)
        _, o_last = apply_fn(params, state[4], obs_last)
        v_last = popart.denormalise(np.asarray(o_last.values))
        adv, ret = gae_bootstrap(R["rewards"], values, np.zeros_like(R["rewards"]), v_last, GAMMA, LAM)
        params = popart.update_and_preserve(ret, params)
        M = T * N
        flat = {k: R[k].reshape((M,) + R[k].shape[2:]) for k in ("obs", "actions", "log_probs", "carries",
                                                                  "token_actions")}
        flat["values"] = values.reshape(M).astype(np.float32)
        flat["adv"], flat["ret"] = adv.reshape(M), ret.reshape(M).astype(np.float32)
        perm = np_rng.permutation(M)
        msum = {}
        for i in range(M // MB):
            idx = perm[i * MB:(i + 1) * MB]
            mb = {k: jnp.asarray(v[idx]) for k, v in flat.items()}
            params, opt_state, mets = mb_step(params, opt_state, mb, popart.sigma, popart.mu)
            for k, v in mets.items():
                msum[k] = msum.get(k, 0.0) + float(v)
        mets = {k: v / (M // MB) for k, v in msum.items()}
        dep, att, ent = R["dep"], R["att"], R["ent"]
        ev = 1 - np.var(ret - values) / (np.var(ret) + 1e-8)
        rec = dict(phase="train", update=u,
                   listener_dep_per_world=float(dep[:, lis].sum() / W), sender_dep_per_world=float(dep[:, ~lis].sum() / W),
                   listener_attempts=int(att[:, lis].sum()), sender_attempts=int(att[:, ~lis].sum()),
                   listener_crafts=int((R["actions"][:, lis] == ACTION_CRAFT).sum()),
                   listener_correct_frac=float(dep[:, lis].sum() / max(att[:, lis].sum(), 1)),
                   token_entropy_sender=float(ent[:, ~lis].mean()), token_entropy_listener=float(ent[:, lis].mean()),
                   swap_diff=float(R["swap_diff"][:, ~lis].mean()), listener_sens_tv=float(R["sens"][:, lis].mean()),
                   ev=float(ev), aux_ps=mets.get("aux_ps", 0.0), aux_pl=mets.get("aux_pl", 0.0),
                   ppo_entropy=mets.get("ppo_entropy"), t_update=round(time.time() - t0, 2))
        emit(rec)
        if u > a.updates - 10:
            mi_tok.append(R["token_actions"][:, ~lis].reshape(-1, 3)); mi_T.append(R["active"].reshape(-1))

    if a.save_params:
        Path(a.save_params).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save_params).write_bytes(flax.serialization.to_bytes(params))

    from comm_testbed import mi_miller_madow
    tok, Tt = np.concatenate(mi_tok), np.concatenate(mi_T)
    for i in range(3):
        mi = mi_miller_madow(tok[:, i], Tt, V, 4)
        null = np.array([mi_miller_madow(tok[:, i], np_rng.permutation(Tt), V, 4) for _ in range(200)])
        emit(dict(phase="mi", slot=i, mi_bits=mi, null_p99=float(np.percentile(null, 99)), n=int(len(Tt))))

    if a.arm == "COMM":
        for k in range(a.swap_pairs):
            res = {}
            for name, sw in (("control", False), ("swap", True)):
                st0 = init_state(jax.random.PRNGKey(10_000 + k))
                _, Rk = rollout(params, st0, jax.random.PRNGKey(20_000 + k), swap=sw)
                Rk = jax.tree_util.tree_map(np.asarray, Rk)
                if sw:   # apparatus: what each listener hears next step == its sender's T' broadcast
                    heard = Rk["heard_next"][:, lis]
                    if not np.allclose(heard, Rk["bc_sw"][:, ~lis]):
                        print("[APPARATUS] FATAL: swap wiring", flush=True); raise SystemExit(1)
                res[name] = (Rk["mu_sw"].sum() / max(Rk["m_sw"].sum(), 1), Rk["mu_tr"].sum() / max(Rk["m_tr"].sum(), 1),
                             float(Rk["dep"][:, lis].sum() / W), float(Rk["swap_diff"][:, ~lis].mean()))
            emit(dict(phase="swap", pair=k, mu_Tprime_control=float(res["control"][0]),
                      mu_Tprime_swap=float(res["swap"][0]), mu_T_control=float(res["control"][1]),
                      mu_T_swap=float(res["swap"][1]), listener_dep_control=res["control"][2],
                      listener_dep_swap=res["swap"][2], swapped_token_differs_frac=res["swap"][3]))


if __name__ == "__main__":
    main()
