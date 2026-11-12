"""Separate-brains population toy (Cam Nov 9, item (e); design in docs/RESEARCH_PROTOCOL.md "separate-brains toy").

The comm toy's world (G 12, 4 fires, sender sees the lit fire, listener hears the sender's 3-token broadcast one step
later, wire-blind message head), but B brains with their own weights / Adam / PopArt instead of one shared brain:
  * every world pairs an ordered (sender brain, listener brain); pairs reshuffled each update (brains.pair_schedule);
  * founders start from navigators 0..B-1 of the pool; PS (founders' formation only) annealed to 0 over 1,000-1,500;
  * message-head weight decay (the calibrated erosion strength) from --erode-from;
  * staggered generations: slot b is first replaced at --first-repl + b * --stagger, then every --life updates; a
    newborn takes a navigator start no living brain started from, a fresh Adam / PopArt, age 0;
  * newborns copy: during the sensitive period, an imitation loss (cross-entropy of the newborn's token policy on elder
    senders' tokens for the elders' observations; message head only, wire-blind pathway) with coefficient --imit;
  * speech that settles: message-head updates x plasticity(age); varied learning speeds: per-lineage --lr-mults;
  * --msg-grad sender: the message head trains on sender rows only (the rung-3 rerun's stop-grad rule);
  * --arm NOSIG = wire off (the control population: copying ON, no signal).
Evals at --eval-at: cross-spot and same-spot empirical swaps over all brains (10 pairs), joint MI, per-brain code
drift. Emits JSONL like the toy. Laptop smoke first; Modal via scripts/modal_comm_toy.py plan "pop".
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flax.serialization  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import brains  # noqa: E402
from comm_toy import (CLIP, ENT, GAMMA, LAM, LR, MB, N_ACT, O_INF, O_NEED, O_WIRE, R_DEP, SIG, TOK_ENT, V, VF,  # noqa: E402
                      VQC, D, ToyNet)
from jax_sim import rl_jax  # noqa: E402
from jax_sim.action_space import mask_disabled_actions  # noqa: E402
from jax_sim.grid_jax import generate_hearth_positions  # noqa: E402
from jax_sim.rl_jax import PopArt  # noqa: E402
from learner_testbed import ACTION_CRAFT, DELTAS, gae_bootstrap  # noqa: E402

TAU = 0.08
MSGK = brains.MSG_KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="COMM", choices=["COMM", "NOSIG"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=9000)
    ap.add_argument("--brains", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--reroll", type=int, default=96)
    ap.add_argument("--craft-cost", type=float, default=0.45)
    ap.add_argument("--navs", nargs="+", required=True, help="navigator params, founders first (>= brains + newborn pool)")
    ap.add_argument("--wire-init-scale", type=float, default=0.06)
    ap.add_argument("--ps", type=float, default=1.99)
    ap.add_argument("--ps-anneal", type=int, nargs=2, default=[1000, 1500])
    ap.add_argument("--erode-from", type=int, default=1500)
    ap.add_argument("--erode-wd", type=float, default=None)
    ap.add_argument("--msg-grad", default="sender", choices=["all", "sender"])
    ap.add_argument("--first-repl", type=int, default=1500)
    ap.add_argument("--stagger", type=int, default=500)
    ap.add_argument("--life", type=int, default=2000)
    ap.add_argument("--sensitive", type=int, default=1000)
    ap.add_argument("--settle", type=int, default=500)
    ap.add_argument("--plast-floor", type=float, default=0.1)
    ap.add_argument("--lr-mults", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--imit", type=float, default=1.0)
    ap.add_argument("--imit-batch", type=int, default=512)
    ap.add_argument("--eval-at", type=int, nargs="*", default=[])
    ap.add_argument("--swap-pairs", type=int, default=10)
    ap.add_argument("--save-at", type=int, nargs="*", default=[])
    ap.add_argument("--save-prefix", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fifth-fire", action="store_true", help="a 5th fire (centre), in the obs from update 0, lit only from --lit-from")
    ap.add_argument("--lit-from", type=int, default=7500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-imit", action="store_true", help="diagnostic (Cam Nov 10, 4b): log each newborn's copy loss and token agreement with the copied elders, before each copy step")
    ap.add_argument("--copy-unit", default="", help="Nov 12 unit test (Cam work order ii): a newborn copies a FROZEN elder for --copy-steps; modes ce | soft | embed (comma list); exit")
    ap.add_argument("--copy-steps", type=int, default=500); ap.add_argument("--copy-tau", type=float, default=1.0)
    ap.add_argument("--copy-elder", default="fire", choices=["fire", "navigator"], help="fire: a frozen elder whose top word is a fixed word per lit fire (a working founder); navigator: an untrained navigator's own tokens")
    ap.add_argument("--copy-diag", default=None, help="diagnostic: load these brains (msgpack), roll out once, and compare each brain's argmax tokens with every other brain's on the same sender inputs; exit")
    ap.add_argument("--copy-diag-ages", type=int, nargs="*", default=[], help="the brains' ages at the saved update (to say who copied whom)")
    ap.add_argument("--ckpt", default=None, help="checkpoint path: resume from it if present (Cam Nov 10)")
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--stop-after", type=int, default=0, help="test only: exit abruptly after this update (simulated preemption)")
    a = ap.parse_args()

    G, W, T, B, H = a.grid, a.worlds, a.steps, a.brains, 64
    NF = 5 if a.fifth_fire else 4
    if a.fifth_fire:   # obs: pos 2 | fire offsets 10 | informed 1 | need 5 | noise 4 | wire 40 -> D 62
        import comm_toy
        O_INF_, O_NEED_, O_WIRE_, D_ = 12, 13, 22, 62
        comm_toy.O_WIRE = O_WIRE_          # ToyNet's wire-blind pathway reads this module global at trace time
    else:
        O_INF_, O_NEED_, O_WIRE_, D_ = O_INF, O_NEED, O_WIRE, D
    N = 2 * W
    assert len(a.lr_mults) == B and len(a.navs) >= 2 * B
    HEARTHS = np.asarray(generate_hearth_positions(G))
    if a.fifth_fire:
        HEARTHS = np.concatenate([HEARTHS, [[G // 2, G // 2]]], 0)
    HEARTHS = jnp.asarray(HEARTHS)
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    dd = np.abs(cells[:, None, :] - np.asarray(HEARTHS)[None]); dd = np.minimum(dd, G - dd).max(-1)
    VALID = jnp.asarray(cells[(dd > 1).all(1)])
    is_lis = np.tile([False, True], W); lis = jnp.asarray(is_lis)
    live_wire = a.arm == "COMM"
    DEL = jnp.asarray(DELTAS)
    out_f = open(a.out, "a")

    def emit(rec):
        out_f.write(json.dumps(dict(arm=a.arm, seed=a.seed, tag=a.tag, **rec)) + "\n"); out_f.flush()

    model = ToyNet(H, wire_blind=1)
    templ0 = model.init(jax.random.PRNGKey(0), jnp.zeros((1, H)), jnp.zeros((1, D)))     # the navigators' 4-fire shape
    templ = model.init(jax.random.PRNGKey(0), jnp.zeros((1, H)), jnp.zeros((1, D_)))

    def expand(p):
        """4-fire input rows -> 5-fire layout; the new rows (fire-5 offset, fire-5 need) are ZERO, so the expanded
        navigator computes exactly what the original did on the shared dims"""
        if not a.fifth_fire:
            return p
        pp = {k: dict(v) for k, v in p["params"].items()}
        for n in ("enc_0", "gwt_comms_1"):
            k0 = np.asarray(pp[n]["kernel"]); z = np.zeros((1, k0.shape[1]), k0.dtype)
            k1 = np.concatenate([k0[0:10], z, z, k0[10:11], k0[11:15], z, k0[15:19], k0[19:59]], 0)
            assert k1.shape[0] == D_
            pp[n]["kernel"] = jnp.asarray(k1)
        return {"params": pp}

    def nav_params(i):
        p = expand(flax.serialization.from_bytes(templ0, Path(a.navs[i]).read_bytes()))
        pp = {k: dict(v) for k, v in p["params"].items()}
        for n in ("enc_0", "gwt_comms_1"):     # the wire rows were never trained (navigators ran wire-off)
            pp[n]["kernel"] = jnp.asarray(pp[n]["kernel"]).at[O_WIRE_:].multiply(a.wire_init_scale)
        return {"params": pp}

    S = brains.stack([nav_params(b) for b in range(B)])
    opt = optax.adam(LR)
    OS = brains.opt_init_per_brain(opt, S)
    pop = [PopArt(beta=0.1) for _ in range(B)]
    nav_of = list(range(B)); next_nav = B                      # which navigator each slot started from
    born = np.zeros(B, np.int64); gen = np.ones(B, np.int64)
    repl = {a.first_repl + b * a.stagger: b for b in range(B)}
    rng = np.random.default_rng(a.seed)

    def fwd(S_, ids, carries, obs):
        """per-row brain forward + per-row token logits (each row with its own brain's codebook)"""
        ids = jnp.asarray(ids); rows = jnp.arange(ids.shape[0])
        outs, tls = [], []
        for b in range(B):
            p = brains.take(S_, b)
            c, o = model.apply(p, carries, obs)
            outs.append((c, o)); tls.append(rl_jax.token_logits(p, o.z_e, TAU))
        sel = lambda xs: jnp.stack(xs)[ids, rows]  # noqa: E731
        new_c = sel([c for c, _ in outs])
        logits = sel([o.action_logits for _, o in outs]); vals = sel([o.values for _, o in outs])
        lvq = sel([o.loss_vq.sum(-1) for _, o in outs])
        tl = [sel([t[i] for t in tls]) for i in range(3)]
        cbs = [jnp.stack([brains.take(S_, b)["params"][k]["embedding"] for b in range(B)])[ids] for _, _, k in rl_jax.TOKEN_SLOTS]
        return new_c, logits, vals, lvq, tl, cbs

    def build_obs(pos, active, wire, k):
        diff = (HEARTHS[None] - pos[:, None, :] + G / 2) % G - G / 2
        hr = (diff / (G / 2.0)).reshape(N, 2 * NF)
        inf_ = ~lis
        need = jax.nn.one_hot(jnp.repeat(active, 2), NF) * inf_[:, None]
        heard = wire.reshape(W, 2, -1)[:, ::-1].reshape(N, -1) if live_wire else jnp.zeros((N, SIG))
        noise = jax.random.normal(k, (N, 4))
        return jnp.concatenate([pos / G, hr, inf_[:, None].astype(jnp.float32), need, noise, heard], -1)

    def init_state(k):
        k1, k2 = jax.random.split(k)
        return (VALID[jax.random.randint(k1, (N,), 0, VALID.shape[0])], jax.random.randint(k2, (W,), 0, 4),
                jnp.zeros(W, jnp.int32), jnp.zeros((N, SIG)), jnp.zeros((N, H)))

    @functools.partial(jax.jit, static_argnames=("nlit", "fixed"))
    def rollout(S_, ids, state, key, swap_table=None, shift=1, nlit=4, fixed=-1):
        def step(st, k):
            pos, active, since, wire, carries = st
            k_obs, k_act, k_rs, k_rr, k_tab = jax.random.split(k, 5)
            obs = build_obs(pos, active, wire, k_obs)
            new_c, logits, vals, lvq, tl, cbs = fwd(S_, ids, carries, obs)
            logits = mask_disabled_actions(logits, axis=-1)
            ka, kt = jax.random.split(k_act)
            act = jax.random.categorical(ka, logits)
            lp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), act[:, None], -1)[:, 0]
            toks, bc = [], [jnp.zeros((N, 8))]
            for i in range(3):
                t = jax.random.categorical(jax.random.fold_in(kt, i), tl[i])
                lp = lp + jnp.take_along_axis(jax.nn.log_softmax(tl[i], -1), t[:, None], -1)[:, 0]
                toks.append(t); bc.append(jnp.take_along_axis(cbs[i], t[:, None, None], 1)[:, 0])
            toks = jnp.stack(toks, -1); bc = jnp.concatenate(bc, -1)
            if swap_table is not None:    # eval: each sender's broadcast replaced by one of ITS OWN triples for (T+shift)%4
                sb = jnp.asarray(ids)[0::2]                                            # sender brain per world
                idx = jax.random.randint(k_tab, (W,), 0, swap_table.shape[2])
                tgt = jnp.full_like(active, fixed) if fixed >= 0 else (active + shift) % nlit
                trip = swap_table[sb, tgt, idx]                                        # (W, 3)
                cbs_s = [c[0::2] for c in cbs]
                bsw = jnp.concatenate([jnp.zeros((W, 8))] + [jnp.take_along_axis(cbs_s[i], trip[:, i][:, None, None], 1)[:, 0]
                                                              for i in range(3)], -1)
                bc = bc.at[0::2].set(bsw)
            near = jnp.abs(pos[:, None, :] - HEARTHS[None]); near = jnp.minimum(near, G - near).max(-1) <= 1
            craft = act == ACTION_CRAFT
            dep = craft & near[jnp.arange(N), jnp.repeat(active, 2)]
            att_Tp = craft & near[jnp.arange(N), jnp.repeat((active + shift) % nlit, 2)]
            att_last = craft & near[:, NF - 1]                                         # Crafts at the last fire (fire 5)
            r = jnp.where(dep, R_DEP, 0.0) - a.craft_cost * (craft & ~dep)
            ldep = dep[lis]
            r = r + jnp.where(~lis, R_DEP * jnp.repeat(ldep, 2), 0.0)
            pos2 = jnp.where(dep[:, None], VALID[jax.random.randint(k_rs, (N,), 0, VALID.shape[0])], (pos + DEL[act]) % G)
            dep_w = dep.reshape(W, 2).any(1); since2 = since + 1
            reroll = dep_w | (since2 >= a.reroll)
            active2 = jnp.where(reroll, (active + jax.random.randint(k_rr, (W,), 1, nlit)) % nlit, active)
            since2 = jnp.where(reroll, 0, since2)
            wire2 = bc if live_wire else jnp.zeros_like(bc)
            rec = dict(obs=obs, actions=act, log_probs=lp, rewards=r.astype(jnp.float32), values=vals, carries=carries,
                       token_actions=toks, dep=dep, att_Tp=att_Tp, att_last=att_last, active=active)
            return (pos2, active2, since2, wire2, new_c), rec
        st, R = jax.lax.scan(step, state, jax.random.split(key, T))
        return st, R

    def ps_term(S_, ids, obs, carries, w_rows):
        """MI(token; active fire) on sender rows of the brains with w_rows > 0 (founders in formation), per brain, summed"""
        _, _, _, _, tl, _ = fwd(S_, ids, carries, obs)
        snd = (obs[:, O_INF_] > 0.5).astype(jnp.float32) * w_rows
        Tl = jnp.argmax(obs[:, O_NEED_:O_NEED_ + NF], -1)
        tot = 0.0
        for b in range(B):
            wb = snd * (jnp.asarray(ids) == b)
            for lg in tl:
                p = jax.nn.softmax(lg, -1)
                pbar = (p * wb[:, None]).sum(0) / (wb.sum() + 1e-8)
                h_all = -(pbar * jnp.log(pbar + 1e-10)).sum()
                h_c = 0.0
                for t in range(4):
                    wt = wb * (Tl == t)
                    pt = (p * wt[:, None]).sum(0) / (wt.sum() + 1e-8)
                    h_c = h_c + (wt.sum() / (wb.sum() + 1e-8)) * -(pt * jnp.log(pt + 1e-10)).sum()
                tot = tot + jnp.where(wb.sum() > 0, h_all - h_c, 0.0)
        return tot

    def loss_fn(S_, mb, sig, mu, ps_c, mask, ps_w):
        """PPO (ratio = exp(lp - old), clip), PopArt value Huber, entropies, VQ, logit penalty; mean over `mask` rows;
        sig / mu = per-row PopArt stats of the row's brain"""
        _, logits, vals, lvq, tl, _ = fwd(S_, mb["ids"], mb["carries"], mb["obs"])
        logits = mask_disabled_actions(logits, axis=-1)
        lsm = jax.nn.log_softmax(logits, -1)
        lp = jnp.take_along_axis(lsm, mb["actions"][:, None], -1)[:, 0]
        tent = 0.0
        for i in range(3):
            l_ = jax.nn.log_softmax(tl[i], -1)
            lp = lp + jnp.take_along_axis(l_, mb["token_actions"][:, i:i + 1], -1)[:, 0]
            tent = tent - jnp.sum(jnp.exp(l_) * l_, -1)
        ratio = jnp.clip(jnp.exp(lp - mb["log_probs"]), 0.0, 10.0)
        adv = mb["adv"]
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv)
        err = jnp.abs(vals - (mb["ret"] - mu) / sig)
        vf = jnp.where(err < 0.5, 0.5 * err ** 2, 0.5 * (err - 0.25))
        ent = -jnp.sum(jnp.exp(lsm) * lsm, -1)
        pen = 0.01 * jnp.mean(logits ** 2, -1)
        den = mask.sum() + 1e-8
        L = ((pg + VF * vf - ENT * ent - TOK_ENT * tent + pen + VQC * lvq) * mask).sum() / den
        ps = ps_term(S_, mb["ids"], mb["obs"], mb["carries"], ps_w) if a.ps else 0.0
        return L - ps_c * ps

    GROUPS = [["head_action"], ["head_value"], [k for k in templ["params"] if k not in ("head_action", "head_value")]]

    @jax.jit
    def grads(S_, mb, sig, mu, ps_c, ps_w):
        ones = jnp.ones(mb["obs"].shape[0])
        g = jax.grad(loss_fn)(S_, mb, sig, mu, ps_c, ones, ps_w)
        if a.msg_grad == "sender":
            snd = (mb["obs"][:, O_INF_] > 0.5).astype(jnp.float32)
            g0 = jax.grad(loss_fn)(S_, mb, sig, mu, 0.0, snd, ps_w)                       # sender rows, no PS
            gps = jax.grad(lambda s_: -ps_c * ps_term(s_, mb["ids"], mb["obs"], mb["carries"], ps_w))(S_) if a.ps else None
            sc = snd.sum() / snd.shape[0]
            gg = dict(g["params"])
            for k in MSGK:
                gg[k] = jax.tree_util.tree_map(lambda x, y: x * sc + y, g0["params"][k], gps["params"][k]) if gps is not None \
                    else jax.tree_util.tree_map(lambda x: x * sc, g0["params"][k])
            g = {"params": gg}
        return g

    @jax.jit
    def imit_grads(S_, ob, toks, idsn, w):
        """cross-entropy of each newborn's token policy on elder senders' tokens for the elders' observations;
        w = per-row weight (1 for rows assigned to a newborn in its sensitive period)"""
        def L(s_):
            _, _, _, _, tl, _ = fwd(s_, idsn, jnp.zeros((ob.shape[0], H)), ob)
            ce = sum(-jnp.take_along_axis(jax.nn.log_softmax(tl[i], -1), toks[:, i:i + 1], -1)[:, 0] for i in range(3))
            return (ce * w).sum() / (w.sum() + 1e-8)
        g = jax.grad(L)(S_)
        return {"params": {k: (g["params"][k] if k in MSGK else jax.tree_util.tree_map(jnp.zeros_like, g["params"][k]))
                           for k in g["params"]}}

    def apply(S_, OS_, g, u):
        g, _ = brains.clip_per_brain(g, GROUPS)
        upd, OS_ = brains.opt_update_per_brain(opt, g, OS_, S_)
        age = u - born
        upd = brains.scale_updates(upd, np.asarray(a.lr_mults), brains.plasticity(age, a.sensitive, a.settle, a.plast_floor))
        S_ = optax.apply_updates(S_, upd)
        return S_, OS_

    def head_vec(S_, b):
        return np.concatenate([np.asarray(x).ravel() for k in MSGK for x in jax.tree_util.tree_leaves(brains.take(S_, b)["params"][k])])

    def evaluate(S_, u):
        ev_rng = np.random.default_rng(1000 + 7919 * a.seed + u)
        nlit = NF if (a.fifth_fire and u >= a.lit_from) else 4
        res = []
        for k in range(a.swap_pairs):
            sch = brains.pair_schedule(W, B, np.random.default_rng(30_000 + k))
            ids = brains.agent_brain_map(*sch)
            st0 = init_state(jax.random.PRNGKey(10_000 + k))
            _, Rc = rollout(S_, ids, st0, jax.random.PRNGKey(20_000 + k), nlit=nlit)
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            trip = Rc["token_actions"][:, 0::2]; lab = Rc["active"]; sb = np.asarray(ids)[0::2]
            table = np.zeros((B, NF, 256, 3), np.int32)
            for b in range(B):
                for t in range(nlit):
                    tt = trip[(lab == t) & (sb[None, :] == b)]
                    table[b, t] = tt[ev_rng.integers(0, len(tt), 256)] if len(tt) else 0
            row = dict(phase="pop_swap", update=u, pair=k, nlit=nlit, dep_control=float(Rc["dep"][:, is_lis].sum() / W),
                       crafts_Tp_control=float(Rc["att_Tp"][:, is_lis].sum() / W),
                       crafts_last_control=float(Rc["att_last"][:, is_lis].sum() / W))
            if live_wire:
                if nlit == NF and a.fifth_fire:   # redirection to fire 5: every sender's message replaced by its own fire-5 word
                    _, Rk = rollout(S_, ids, st0, jax.random.PRNGKey(20_000 + k), jnp.asarray(table), 1, nlit=nlit, fixed=NF - 1)
                    Rk = jax.tree_util.tree_map(np.asarray, Rk)
                    row["crafts_last_to5"] = float(Rk["att_last"][:, is_lis].sum() / W)
                for nm, sh in (("cross", 1), ("same", 0)):
                    _, Rk = rollout(S_, ids, st0, jax.random.PRNGKey(20_000 + k), jnp.asarray(table), sh, nlit=nlit)
                    Rk = jax.tree_util.tree_map(np.asarray, Rk)
                    row[f"dep_{nm}"] = float(Rk["dep"][:, is_lis].sum() / W)
                    if nm == "cross":
                        row["crafts_Tp_cross"] = float(Rk["att_Tp"][:, is_lis].sum() / W)
            emit(row); res.append((trip.reshape(-1, 3), lab.reshape(-1), np.repeat(sb[None], T, 0).reshape(-1)))
        tok = np.concatenate([r[0] for r in res]); lab = np.concatenate([r[1] for r in res]); bb = np.concatenate([r[2] for r in res])
        words = [[[int(np.bincount(tok[(lab == t) & (bb == b), i], minlength=V).argmax()) if ((lab == t) & (bb == b)).any() else -1
                   for i in range(3)] for t in range(NF)] for b in range(B)]
        emit(dict(phase="pop_words", update=u, words=words, gen=gen.tolist(), born=born.tolist(), nav_of=nav_of))
        np.savez_compressed(f"{a.out}.u{u}.tok.npz", tok=tok, lab=lab, brain=bb)

    if a.selftest:
        """apparatus (before any paid run), on one real rollout + minibatch of the population:
        (1) brain isolation in the actual loss: perturbing every row of brain 0 leaves brains 1..B-1's gradients
            bit-identical (and changes brain 0's);
        (2) the sender rule: perturbing every LISTENER row leaves every brain's message-head gradient bit-identical
            (only when --msg-grad sender; with 'all' it must change -- the mutant);
        (3) imitation gradients touch the message head only;
        (4) PS weight 0 on non-founder rows: the PS term is exactly 0 when no row is a founder's;
        (5) wire off (NOSIG): every heard wire entry in the obs is exactly 0."""
        vec = lambda g, b, keys=None: np.concatenate([np.asarray(x).ravel() for k in (keys or g["params"])  # noqa: E731
                                                      for x in jax.tree_util.tree_leaves(brains.take(g, b)["params"][k])])
        sch = brains.pair_schedule(W, B, np.random.default_rng(0)); ids = brains.agent_brain_map(*sch)
        _, R = rollout(S, ids, init_state(jax.random.PRNGKey(5)), jax.random.PRNGKey(6))
        R = jax.tree_util.tree_map(np.asarray, R)
        M = T * N; idsT = np.broadcast_to(ids, (T, N)).reshape(M)
        flat = dict(obs=R["obs"].reshape(M, -1), actions=R["actions"].reshape(M), log_probs=R["log_probs"].reshape(M),
                    carries=R["carries"].reshape(M, H), token_actions=R["token_actions"].reshape(M, 3),
                    adv=np.random.default_rng(1).normal(size=M).astype(np.float32),
                    ret=np.random.default_rng(2).normal(size=M).astype(np.float32), ids=idsT)
        ix = np.random.default_rng(3).permutation(M)[:MB]
        mb = {k: jnp.asarray(v[ix]) for k, v in flat.items()}
        one = jnp.ones(MB); fw = jnp.ones(MB)
        g = grads(S, mb, one, 0 * one, 0.5, fw)
        def pert(mask):
            m2 = dict(mb); r_ = np.random.default_rng(9)
            for k in ("adv", "ret", "log_probs"):
                v = np.array(mb[k]); v[mask] = r_.normal(size=mask.sum()).astype(v.dtype) * 5; m2[k] = jnp.asarray(v)
            v = np.array(mb["actions"]); v[mask] = r_.integers(0, 5, mask.sum()); m2["actions"] = jnp.asarray(v)
            v = np.array(mb["token_actions"]); v[mask] = r_.integers(0, V, (mask.sum(), 3)); m2["token_actions"] = jnp.asarray(v)
            return m2
        g_b0 = grads(S, pert(np.asarray(mb["ids"]) == 0), one, 0 * one, 0.5, fw)
        res = dict(isolation_other_brains_identical=all(np.array_equal(vec(g, b), vec(g_b0, b)) for b in range(1, B)),
                   isolation_brain0_changes=not np.array_equal(vec(g, 0), vec(g_b0, 0)))
        lis_rows = np.asarray(mb["obs"][:, O_INF_] < 0.5)
        g_l = grads(S, pert(lis_rows), one, 0 * one, 0.5, fw)
        same = all(np.array_equal(vec(g, b, MSGK), vec(g_l, b, MSGK)) for b in range(B))
        res["sender_rule_listener_rows_no_msg_grad"] = bool(same) if a.msg_grad == "sender" else (not same)
        gi = imit_grads(S, mb["obs"][:64], mb["token_actions"][:64], mb["ids"][:64], jnp.ones(64))
        other = [k for k in gi["params"] if k not in MSGK]
        res["imitation_msg_head_only"] = all(float(jnp.abs(x).sum()) == 0.0 for k in other
                                             for x in jax.tree_util.tree_leaves(gi["params"][k]))
        res["imitation_moves_msg_head"] = any(float(jnp.abs(x).sum()) > 0 for k in MSGK for x in jax.tree_util.tree_leaves(gi["params"][k]))
        res["ps_zero_without_founders"] = float(ps_term(S, mb["ids"], mb["obs"], mb["carries"], 0 * fw)) == 0.0
        res["ps_nonzero_with_founders"] = float(ps_term(S, mb["ids"], mb["obs"], mb["carries"], fw)) != 0.0
        if a.fifth_fire:   # the expanded navigator computes what the original did, whatever the new fire-5 dims hold
            p0 = flax.serialization.from_bytes(templ0, Path(a.navs[0]).read_bytes()); p1 = expand(p0)
            r_ = np.random.default_rng(4); o59 = r_.normal(size=(64, 59)).astype(np.float32); o59[:, 19:] = 0.0
            o62 = np.concatenate([o59[:, :10], r_.normal(size=(64, 2)), o59[:, 10:15], r_.normal(size=(64, 1)),
                                  o59[:, 15:19], o59[:, 19:]], 1).astype(np.float32)
            c = jnp.zeros((64, H))
            _, q0 = model.apply(p0, c, jnp.asarray(o59)); _, q1 = model.apply(p1, c, jnp.asarray(o62))
            res["expand_preserves_navigator"] = bool(np.allclose(q0.action_logits, q1.action_logits, atol=1e-5)
                                                     and np.allclose(q0.values, q1.values, atol=1e-5)
                                                     and np.allclose(q0.z_e, q1.z_e, atol=1e-5))
            res["fifth_fire_need_reaches_msg_path"] = bool(O_WIRE_ == 22 and O_NEED_ + NF <= O_WIRE_)
        if not live_wire:
            res["wire_off_heard_zero"] = bool(np.all(R["obs"][:, :, O_WIRE_:] == 0.0))
        res["PASS"] = bool(all(res.values()))
        print(json.dumps(res, indent=1), flush=True); emit(dict(phase="selftest", **res))
        raise SystemExit(0 if res["PASS"] else 1)

    if a.copy_unit:                                   # Nov 12: copy-pathway unit test (newborn b=1 copies frozen elder e=0)
        TS = rl_jax.TOKEN_SLOTS

        def zfwd(S_, b, ob):
            p = brains.take(S_, b); _, o = model.apply(p, jnp.zeros((ob.shape[0], H)), ob)
            return p, o.z_e

        def argtok1(S_, b, ob):
            p, z = zfwd(S_, b, ob)
            return jnp.stack([jnp.argmax(t, -1) for t in rl_jax.token_logits(p, z, TAU)], 1)

        def copy_loss(S_, ob, tk, mode):
            p, z = zfwd(S_, 1, ob)
            if mode == "embed":                       # pull z_e toward the newborn's OWN codebook entry at the elder's index
                L = 0.0
                for i, (lo, hi, ck) in enumerate(TS):
                    tgt = jax.lax.stop_gradient(p["params"][ck]["embedding"])[tk[:, i]]
                    L = L + jnp.sum((z[:, lo:hi] - tgt) ** 2, -1).mean()
                return L
            tau = TAU if mode == "ce" else a.copy_tau
            tl = rl_jax.token_logits(p, z, tau)
            return sum(-jnp.take_along_axis(jax.nn.log_softmax(tl[i], -1), tk[:, i:i + 1], -1)[:, 0].mean() for i in range(3))

        def cgrads(S_, ob, tk, mode):
            g = jax.grad(copy_loss)(S_, ob, tk, mode)
            return {"params": {k: (g["params"][k] if k in MSGK else jax.tree_util.tree_map(jnp.zeros_like, g["params"][k])) for k in g["params"]}}
        cg = jax.jit(cgrads, static_argnums=(3,))
        sch = brains.pair_schedule(W, B, np.random.default_rng(0)); ids = brains.agent_brain_map(*sch)
        pools, labs = [], []
        for k_ in range(4):                           # situations: elder sender inputs from 4 rollouts (3 train, 1 held out)
            _, R = rollout(S, ids, init_state(jax.random.PRNGKey(50 + k_)), jax.random.PRNGKey(60 + k_))
            R = jax.tree_util.tree_map(np.asarray, R); T_, N_ = R["obs"].shape[:2]
            rows = (~np.broadcast_to(is_lis, (T_, N_))).reshape(-1) & (np.broadcast_to(ids, (T_, N_)).reshape(-1) == 0)
            pools.append(jnp.asarray(R["obs"].reshape(T_ * N_, -1)[rows]))
            labs.append(np.repeat(R["active"], 2, axis=1).reshape(-1)[rows])      # the lit fire each sender row sees
        tr = jnp.concatenate(pools[:3]); ho = pools[3]; lab_tr = np.concatenate(labs[:3]); lab_ho = labs[3]
        S0 = S
        if a.copy_elder == "fire":                    # the elder's top word per situation: a fixed word per lit fire
            WORD = np.array([[(7 * f + 3) % V, (13 * f + 5) % V, (29 * f + 11) % V] for f in range(8)])
            el_tr = jnp.asarray(WORD[lab_tr]); el_ho = WORD[lab_ho]
        else:
            el_tr = argtok1(S0, 0, tr); el_ho = np.asarray(argtok1(S0, 0, ho))
        words = len({tuple(x) for x in el_ho.tolist()})
        out = dict(elder=a.copy_elder, situations_heldout=int(len(set(lab_ho.tolist()))), steps=a.copy_steps, copy_tau=a.copy_tau, train_rows=int(tr.shape[0]), heldout_rows=int(ho.shape[0]),
                   elder_distinct_words_heldout=words, modes={})
        fresh = brains.stack([nav_params(j % len(a.navs)) for j in range(B, 2 * B)])
        chance_slot = float(np.mean([(np.asarray(argtok1(fresh, j, ho)) == el_ho).mean() for j in range(B)]))
        chance_word = float(np.mean([(np.asarray(argtok1(fresh, j, ho)) == el_ho).all(1).mean() for j in range(B)]))
        rr = np.random.default_rng(7)
        for mode in a.copy_unit.split(","):
            Sx, OSx = S0, brains.opt_init_per_brain(opt, S0); curve = []
            for st in range(1, a.copy_steps + 1):
                bi = rr.integers(0, tr.shape[0], 512)
                g = cg(Sx, tr[bi], el_tr[bi], mode)
                Sx, OSx = apply(Sx, OSx, g, st)       # born = 0, so age = st: inside the sensitive period
                if st in (1, 50, 100, 200, 300, 500, 750, 1000, 1500, a.copy_steps):
                    nb = np.asarray(argtok1(Sx, 1, ho))
                    curve.append(dict(step=st, slot_agree=float((nb == el_ho).mean()), word_agree=float((nb == el_ho).all(1).mean())))
            # the elder must stay frozen: brain 0 untouched
            frozen = all(bool(jnp.array_equal(x[0], y[0])) for x, y in zip(jax.tree_util.tree_leaves(Sx), jax.tree_util.tree_leaves(S0)))
            out["modes"][mode] = dict(curve=curve, elder_frozen=frozen, final_slot_agree=curve[-1]["slot_agree"],
                                      final_word_agree=curve[-1]["word_agree"], passes_50pct_slot=curve[-1]["slot_agree"] >= 0.5)
        out["chance_slot_fresh_navigators"] = chance_slot; out["chance_word_fresh_navigators"] = chance_word
        print("COPYUNIT", json.dumps(out), flush=True); raise SystemExit(0)

    if a.copy_diag:                                   # 4b: did newborns come to emit their elders' tokens on the same inputs?
        S = jax.tree_util.tree_map(jnp.asarray, flax.serialization.from_bytes(S, open(a.copy_diag, "rb").read()))
        sch = brains.pair_schedule(W, B, np.random.default_rng(0)); ids = brains.agent_brain_map(*sch)
        _, R = rollout(S, ids, init_state(jax.random.PRNGKey(5)), jax.random.PRNGKey(6))
        R = jax.tree_util.tree_map(np.asarray, R); T_, N_ = R["obs"].shape[:2]
        idsT = np.broadcast_to(ids, (T_, N_)).reshape(-1); snd = ~np.broadcast_to(is_lis, (T_, N_)).reshape(-1)
        obs = jnp.asarray(R["obs"].reshape(T_ * N_, -1)); car = jnp.zeros((obs.shape[0], H))
        def argtok(Sx, b):
            _, _, _, _, tl, _ = fwd(Sx, jnp.full(obs.shape[0], b), car, obs)
            return np.stack([np.asarray(t.argmax(-1)) for t in tl], 1)
        tok = {b: argtok(S, b) for b in range(B)}
        fresh = brains.stack([nav_params((B + j) % len(a.navs)) for j in range(B)])     # untrained navigators: chance
        tokf = {b: argtok(fresh, b) for b in range(B)}
        out = dict(ages=a.copy_diag_ages, pairs=[])
        for e in range(B):                            # on brain e's own sender inputs
            rows = snd & (idsT == e)
            for b in range(B):
                if b == e:
                    continue
                ag = float((tok[b][rows] == tok[e][rows]).mean()); ch = float(np.mean([(tokf[j][rows] == tok[e][rows]).mean() for j in range(B)]))
                out["pairs"].append(dict(copier=b, elder=e, agreement=ag, chance_fresh_navigators=ch, rows=int(rows.sum())))
        print("COPYDIAG", json.dumps(out), flush=True); raise SystemExit(0)

    key = jax.random.PRNGKey(a.seed)
    state = init_state(jax.random.fold_in(key, 1))
    elder_buf = None
    prev_head = [head_vec(S, b) for b in range(B)]
    u0 = 1
    if a.ckpt:
        import ckpt as _ck
        C = _ck.load(a.ckpt)
        if C is not None:                                      # resume: restore the loop state, drop rows after it
            out_f.flush(); _ck.truncate(a.out, C["out_offset"]); out_f.seek(0, 2)
            S, OS, pop, born, gen, repl = C["S"], C["OS"], C["pop"], C["born"], C["gen"], C["repl"]
            nav_of, next_nav, rng, key, state, elder_buf, prev_head = (C["nav_of"], C["next_nav"], C["rng"], C["key"],
                                                                       C["state"], C["elder_buf"], C["prev_head"])
            S = jax.tree_util.tree_map(jnp.asarray, S); OS = jax.tree_util.tree_map(jnp.asarray, OS)
            key = jnp.asarray(key); state = jax.tree_util.tree_map(jnp.asarray, state)
            u0 = C["u"] + 1
            emit(dict(phase="resume", update=C["u"]))
    for u in range(u0, a.updates + 1):
        t0 = time.time()
        if u in repl:                                         # a newborn replaces slot b
            b = repl.pop(u)
            in_use = set(nav_of)
            while next_nav % len(a.navs) in in_use:
                next_nav += 1
            nb = nav_params(next_nav % len(a.navs)); nav_of[b] = next_nav % len(a.navs); next_nav += 1
            S, OS = brains.newborn_reset(S, OS, b, nb, opt.init(nb))
            pop[b] = PopArt(beta=0.1); born[b] = u; gen[b] += 1
            repl[u + a.life] = b
            emit(dict(phase="birth", update=u, slot=b, gen=int(gen[b]), nav=nav_of[b]))
        sch = brains.pair_schedule(W, B, rng)
        ids = brains.agent_brain_map(*sch)
        key, kr = jax.random.split(key)
        state, R = rollout(S, ids, state, kr, nlit=NF if (a.fifth_fire and u >= a.lit_from) else 4)
        R = jax.tree_util.tree_map(np.asarray, R)
        idsT = np.broadcast_to(ids, (T, N))
        # per-brain PopArt: denormalise each row with its brain's stats, update each brain's stats on its own returns
        sig_b = np.array([p.sigma for p in pop]); mu_b = np.array([p.mu for p in pop])
        values = R["values"] * sig_b[idsT] + mu_b[idsT]
        adv, ret = gae_bootstrap(R["rewards"], values, np.zeros_like(R["rewards"]), values[-1], GAMMA, LAM)
        for b in range(B):
            m = idsT == b
            Pb = brains.take(S, b)
            Pb = pop[b].update_and_preserve(ret[m], Pb)
            S = jax.tree_util.tree_map(lambda X, x: X.at[b].set(x), S, Pb)
        sig_b = np.array([p.sigma for p in pop]); mu_b = np.array([p.mu for p in pop])
        M = T * N
        flat = dict(obs=R["obs"].reshape(M, -1), actions=R["actions"].reshape(M), log_probs=R["log_probs"].reshape(M),
                    carries=R["carries"].reshape(M, H), token_actions=R["token_actions"].reshape(M, 3),
                    adv=adv.reshape(M).astype(np.float32), ret=ret.reshape(M).astype(np.float32), ids=idsT.reshape(M))
        s0, e0 = a.ps_anneal
        ps_c = a.ps * float(np.clip((e0 - u) / (e0 - s0), 0.0, 1.0)) if u > s0 else a.ps
        founder = (gen == 1).astype(np.float32)                 # PS only for founders' rows
        perm = rng.permutation(M)
        for i in range(M // MB):
            ix = perm[i * MB:(i + 1) * MB]
            mb = {k: jnp.asarray(v[ix]) for k, v in flat.items()}
            sig = jnp.asarray(sig_b[flat["ids"][ix]], jnp.float32); mu = jnp.asarray(mu_b[flat["ids"][ix]], jnp.float32)
            g = grads(S, mb, sig, mu, ps_c, jnp.asarray(founder[flat["ids"][ix]]))
            S, OS = apply(S, OS, g, u)
            if a.erode_wd is not None and u >= a.erode_from:
                pp = dict(S["params"])
                for hk in MSGK:
                    pp[hk] = jax.tree_util.tree_map(lambda x: x * (1.0 - a.erode_wd), pp[hk])
                S = {"params": pp}
        # copying: newborns in their sensitive period imitate elder senders from this rollout
        ages = u - born
        newb = [b for b in range(B) if gen[b] > 1 and ages[b] < a.sensitive]
        if newb and a.imit > 0:
            idsf = flat["ids"]; snd = ~np.broadcast_to(is_lis, (T, N)).reshape(M)
            rows_all, own = [], []
            for b in newb:
                el = brains.elder_rows(idsf, snd, ages, b)
                if el.any():
                    pick = rng.choice(np.where(el)[0], a.imit_batch)
                    rows_all.append(pick); own.append(np.full(a.imit_batch, b))
            if rows_all:
                rr = np.concatenate(rows_all); ob = jnp.asarray(flat["obs"][rr]); tk = jnp.asarray(flat["token_actions"][rr])
                if a.log_imit:                        # before the copy step: CE and argmax agreement on the copied inputs
                    ownv = np.concatenate(own)
                    _, _, _, _, tl_, _ = fwd(S, jnp.asarray(ownv), jnp.zeros((len(rr), H)), ob)
                    ce_ = sum(-np.take_along_axis(np.asarray(jax.nn.log_softmax(tl_[i], -1)), np.asarray(tk)[:, i:i + 1], -1)[:, 0] for i in range(3))
                    am = np.stack([np.asarray(tl_[i].argmax(-1)) for i in range(3)], 1); tkn = np.asarray(tk)
                    shuf = tkn[np.random.default_rng(u).permutation(len(tkn))]
                    for b in newb:
                        m_ = ownv == b
                        if m_.any():
                            emit(dict(phase="imit", update=u, newborn=int(b), age=int(ages[b]), ce=float(ce_[m_].mean()),
                                      agree=float((am[m_] == tkn[m_]).mean()), agree_shuffled=float((am[m_] == shuf[m_]).mean())))
                gi = imit_grads(S, ob, tk, jnp.asarray(np.concatenate(own)), jnp.ones(len(rr)))
                gi = jax.tree_util.tree_map(lambda x: x * a.imit, gi)
                S, OS = apply(S, OS, gi, u)
        dep = R["dep"]; heads = [head_vec(S, b) for b in range(B)]
        drift = [float(np.linalg.norm(h - p)) for h, p in zip(heads, prev_head)]; prev_head = heads
        emit(dict(phase="train", update=u, listener_dep_per_world=float(dep[:, is_lis].sum() / W),
                  sender_dep_per_world=float(dep[:, ~is_lis].sum() / W),
                  dep_by_brain_as_listener=[float(dep[:, is_lis][:, sch[1] == b].sum() / max((sch[1] == b).sum(), 1)) for b in range(B)],
                  gen=gen.tolist(), age=(u - born).tolist(), ps_coef=ps_c, head_step=drift, t_update=round(time.time() - t0, 2)))
        if u in a.eval_at or u == a.updates:
            evaluate(S, u)
        if u in a.save_at and a.save_prefix:
            Path(a.save_prefix).parent.mkdir(parents=True, exist_ok=True)
            Path(f"{a.save_prefix}_u{u}_brains.msgpack").write_bytes(flax.serialization.to_bytes(S))
        if a.ckpt and (u % a.ckpt_every == 0) and u < a.updates:
            import ckpt as _ck
            out_f.flush()
            _ck.save(a.ckpt, dict(u=u, out_offset=os.path.getsize(a.out), S=S, OS=OS, pop=pop, born=born, gen=gen, repl=repl,
                                  nav_of=nav_of, next_nav=next_nav, rng=rng, key=key, state=state, elder_buf=elder_buf,
                                  prev_head=prev_head))
        if a.stop_after and u == a.stop_after:
            out_f.flush(); os._exit(3)


if __name__ == "__main__":
    main()
