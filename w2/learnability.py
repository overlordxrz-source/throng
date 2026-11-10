"""W2 learnability gate (Cam Nov 9 ~20:15; levers Nov 10 ~00:35): can a newborn learn to LISTEN within one lifetime?

Per world: 4 frozen speakers ("beacons": stationary, immortal, at the founders' cluster) broadcasting a fixed scripted
code (the 8-sector direction from the beacon to its nearest food patch), NL learners (fresh W2 navigators; new input
rows / heads at zero or fresh init; own per-agent learning; natural lives), and optionally NE elders (--copy):
immortal scripted listeners that forage by the heard code, the demonstrators the young copy.
A learner hears the nearest beacon within Chebyshev radius 9: one-hot(sector) + flag + offset / 9.
TEST: at death the learner's end-of-life weights are frozen. Food eaten over 300 steps from a start near the beacons
(learner alone, no elders), true code vs a cross-direction swap (every heard sector shifted by a random 1..7),
16 paired episodes. A life PASSES iff drop >= 30% AND paired drop > 2 SE. GATE PASS iff >= 50% of lives pass.
LEVERS:
  --juvenile J    provisioning: energy kept >= E0 for the first J steps of life; the young stays within hearing range
                  (default 500: the W2 default from Nov 10; the beacon world has no parents, so nobody pays)
  --pred C        predictive learning: a self-supervised head on the shared trunk (heard messages are inputs)
                  predicting the next view window (98), own energy change (1), and the max food in view 4 steps ahead
                  (1); MSE, weight C (0 = off)
  --replay K      sleep replay: after each rollout, K extra passes over the agent's own memory since birth (the last
                  --mem rollouts), predictive loss only
  --copy C        copying elders' actions (M approved, Nov 10): behaviour cloning on the (obs, action) of elders within
                  hearing range, young only (age < --young), weight C
  --sanity        positive control: the learners themselves are scripted listeners
usage: python w2/learnability.py --navs "w2_out/nav/*.msgpack" [--updates 150] [--pred 1.0] [--replay 8] [--copy 1.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402
from w2.worlds import world_kw  # noqa: E402

NB, R_HEAR, HD = 4, 9, 11


class NetP(nn.Module):
    """the navigator MLP (same layer names, so navigator weights load) + a predictive head on the shared trunk"""
    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(128, name="Dense_0")(x)); h = nn.tanh(nn.Dense(128, name="Dense_1")(h))
        return nn.Dense(6, name="Dense_2")(h), nn.Dense(1, name="Dense_3")(h)[..., 0], nn.Dense(100, name="pred")(h)


def sector(off):
    ang = jnp.arctan2(off[..., 1].astype(jnp.float32), off[..., 0].astype(jnp.float32))
    return jnp.mod(jnp.round(ang / (2 * jnp.pi / 8)).astype(jnp.int32), 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", required=True); ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--worlds", type=int, default=16); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--juvenile", type=int, default=500)
    ap.add_argument("--long-life", action="store_true"); ap.add_argument("--random-code", action="store_true")
    ap.add_argument("--pred", type=float, default=0.0); ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--mem", type=int, default=4); ap.add_argument("--copy", type=float, default=0.0)
    ap.add_argument("--young", type=int, default=1000); ap.add_argument("--n-elders", type=int, default=2)
    ap.add_argument("--n-learners", type=int, default=4)
    ap.add_argument("--eval-eps", type=int, default=16); ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--copy-replay", action="store_true", help="Cam Nov 10: the K replay passes also cover the copy loss (supervised)")
    ap.add_argument("--young-lr", type=float, default=1.0, help="young plasticity: update x this at birth, linear to x1 by the end of --juvenile")
    ap.add_argument("--diag", type=int, default=0, help="free diagnostic: on this many lives, does the predictive head use the message?")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="w2_out/learnability.json")
    ap.add_argument("--world", default="proto", help='"proto" or "d0,P,relocate" (Cam Nov 10 4b)')
    ap.add_argument("--beacon-spread", type=int, default=2, help="beacons at (+-k, +-k) around the cluster centre (Cam Nov 10 coverage: spread)")
    ap.add_argument("--beacon-walk", action="store_true", help="beacons random-walk at agent speed (one random move per step)")
    ap.add_argument("--seek-age", type=int, default=0, help="young-seek applies while age < this (0: during the juvenile period, as before); the same scaffolding in every arm, independent of the juvenile lever")
    ap.add_argument("--young-seek", action="store_true", help="a young that hears nothing steps toward the nearest beacon (was: stays)")
    ap.add_argument("--no-train", action="store_true", help="apparatus gate: no learning; per-life coverage + life test (+ blind on the same lives)")
    ap.add_argument("--test", default="shift", choices=["shift", "heard25"], help="heard25: swap = a derangement among the sectors that life heard >= --min-heard times")
    ap.add_argument("--min-heard", type=int, default=25); ap.add_argument("--test-window", type=int, default=300)
    ap.add_argument("--min-sectors", type=int, default=2, help="Cam Nov 10: a life is scorable only if it heard >= this many sectors >= --min-heard times (ruled: 5); pass rates over eligible lives")
    ap.add_argument("--score", default="window", choices=["window", "last-heard"], help="last-heard: in the heard25 test, only food eaten while the last heard TRUE sector is one the life heard >= --min-heard times counts")
    ap.add_argument("--test-start", default="fresh", choices=["fresh", "life"], help="life: each probe episode starts in the life's own world at its end (patch centres + beacon positions), the learner beside a random beacon")
    ap.add_argument("--snap-ages", default="", help="comma-separated ages: snapshot each life's weights there; the heard25 test also runs on them (steps to first reliance)")
    ap.add_argument("--ckpt", default=None, help="checkpoint path; resume from it if present (Cam Nov 10: --ckpt on for Modal runs)")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--stop-after", type=int, default=0, help="test only: exit abruptly after this many updates (simulated preemption)")
    ap.add_argument("--hearing-probe", action="store_true", help="Cam Nov 10 ~05:15: supervised probe of the hearing pathway")
    ap.add_argument("--probe-steps", type=int, default=300); ap.add_argument("--probe-lrs", default="1,4,16")
    ap.add_argument("--probe-collect", type=int, default=32, help="teacher rollouts (x T steps) per data set")
    ap.add_argument("--probe-pool", action="store_true", help="diagnostic: every learner trains on teacher data pooled over all training worlds (full sector coverage)")
    ap.add_argument("--probe-train", default="hear", choices=["hear", "all"], help="hear: heard rows + policy head only; all: every layer (diagnostic)")
    a = ap.parse_args()
    NL = a.n_learners; NE = a.n_elders if a.copy > 0 else 0; M = NL + NE
    if a.world == "proto":
        cfg = env.Cfg(L=96, N=NB + M, P=6, patch_sigma=1.0, d0=0.004 if a.long_life else 0.008, reproduce=False, A_max=3000)
    else:
        cfg = env.Cfg(L=96, N=NB + M, patch_sigma=1.0, reproduce=False, A_max=3000, **world_kw(a.world))
    W, T = a.worlds, a.T; net = NetP()
    navs = sorted(glob.glob(a.navs)); K = len(navs)
    D0 = int(env.observe(cfg, env.reset(cfg, jax.random.PRNGKey(0))).shape[1])          # 100
    templ0 = Net().init(jax.random.PRNGKey(0), jnp.zeros((1, D0)))
    templ = net.init(jax.random.PRNGKey(1), jnp.zeros((1, D0 + HD)))

    def expand(p):          # hearing rows appended as zeros (the newborn starts exactly as its navigator); pred head fresh
        q = {k: dict(v) for k, v in templ["params"].items()}
        for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
            q[k] = dict(p["params"][k])
        q["Dense_0"]["kernel"] = jnp.concatenate([p["params"]["Dense_0"]["kernel"], jnp.zeros((HD, 128))], 0)
        return {"params": q}
    NAV = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[expand(flax.serialization.from_bytes(templ0, open(f, "rb").read())) for f in navs])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))

    def beacons_reset(s, key):
        c = jax.random.randint(key, (2,), 0, cfg.L)
        k_ = a.beacon_spread
        bpos = (c[None] + jnp.array([[-k_, -k_], [-k_, k_], [k_, -k_], [k_, k_]])) % cfg.L
        mpos = (c[None] + jax.random.randint(jax.random.fold_in(key, 1), (M, 2), -3, 4)) % cfg.L
        return s._replace(pos=jnp.concatenate([bpos, mpos], 0)), c

    def hear(s, shift, perm=None):
        """(M, HD) for the non-beacon agents: nearest beacon within radius; its sector (+ shift, or mapped through perm)
        one-hot, flag, offset / R"""
        sec = jnp.mod(sector(env.oracle_offset(cfg, s))[:NB] + shift, 8)
        if perm is not None:
            sec = perm[sec]
        lp, bp = s.pos[NB:], s.pos[:NB]
        d = (bp[None] - lp[:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
        cheb = jnp.abs(d).max(-1); j = jnp.argmin(cheb, 1); dj = d[jnp.arange(M), j]
        ok = cheb[jnp.arange(M), j] <= R_HEAR
        return jnp.concatenate([jax.nn.one_hot(sec[j], 8), jnp.ones((M, 1)), dj / R_HEAR], 1) * ok[:, None]

    def scripted(s, h, fallback):
        """a listener that eats on food, heads for visible food, else walks the heard sector"""
        pos = s.pos[NB:]; here = s.food[pos[:, 0], pos[:, 1]]
        wv = env.window(cfg, s.food, pos, cfg.r).reshape(M, -1); best = jnp.argmax(wv, 1); seen = wv.max(1) > 0.2
        o_ = jnp.arange(-cfg.r, cfg.r + 1)
        tw = lambda dx, dy: jnp.where(jnp.abs(dx) >= jnp.abs(dy), jnp.where(dx < 0, 1, 2), jnp.where(dy > 0, 3, 4))  # noqa: E731
        vis = tw(o_[best // (2 * cfg.r + 1)], o_[best % (2 * cfg.r + 1)])
        ang = jnp.argmax(h[:, :8], 1) * (2 * jnp.pi / 8)
        hd = tw(jnp.round(jnp.cos(ang) * 3), jnp.round(jnp.sin(ang) * 3))
        return jnp.where(here >= 0.2, env.EAT, jnp.where(seen, vis, jnp.where(h[:, 8] > 0.5, hd, fallback)))

    def step_all(P, s, key, shift, young, perm=None, mute=False, seek_ok=True, rand_fb=False):
        k1, k2 = jax.random.split(key)
        h = hear(s, shift, perm)
        if mute:                                  # silence: nobody hears anything (VoI baseline)
            h = jnp.zeros_like(h)
        o = jnp.concatenate([env.observe(cfg, s)[NB:], h], 1)                              # (M, D)
        lg, v, _ = jax.vmap(lambda p, x: net.apply(p, x[None]))(P, o[:NL]); lg, v = lg[:, 0], v[:, 0]
        act_l = jax.random.categorical(k1, lg)
        fb_l = jax.random.randint(jax.random.fold_in(key, 79), (NL,), 1, 5) if rand_fb else act_l
        sc = scripted(s, h, jnp.concatenate([fb_l, jax.random.randint(k2, (NE,), 1, 5)]))
        if a.sanity:
            act_l = sc[:NL]
        if a.young_seek and a.seek_age and seek_ok:       # scaffolding independent of the juvenile lever (Cam Nov 10)
            db = (s.pos[:NB][None] - s.pos[NB:NB + NL][:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
            dn_ = db[jnp.arange(NL), jnp.argmin(jnp.abs(db).max(-1), 1)]
            seek = jnp.where(jnp.abs(dn_[:, 0]) >= jnp.abs(dn_[:, 1]), jnp.where(dn_[:, 0] < 0, 1, 2), jnp.where(dn_[:, 1] > 0, 3, 4))
            far = jnp.abs(db).max(-1).min(1) > R_HEAR
            act_l = jnp.where((s.age[NB:NB + NL] < a.seek_age) & far, seek, act_l)
        elif a.juvenile:                                  # the young stay within hearing range of the beacons
            if a.young_seek:                              # ... by stepping toward the nearest beacon (beacons may wander)
                db = (s.pos[:NB][None] - s.pos[NB:NB + NL][:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
                dn_ = db[jnp.arange(NL), jnp.argmin(jnp.abs(db).max(-1), 1)]
                seek = jnp.where(jnp.abs(dn_[:, 0]) >= jnp.abs(dn_[:, 1]), jnp.where(dn_[:, 0] < 0, 1, 2), jnp.where(dn_[:, 1] > 0, 3, 4))
                far = jnp.abs(db).max(-1).min(1) > R_HEAR                  # geometry, never the message channel (Cam)
                act_l = jnp.where(young & far, seek, act_l)
            else:
                act_l = jnp.where(young & (h[:NL, 8] < 0.5), 0, act_l)
        lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act_l[:, None], 1)[:, 0]
        act_m = jnp.concatenate([act_l, sc[NL:]])
        act_b = jax.random.randint(jax.random.fold_in(key, 77), (NB,), 1, 5) if a.beacon_walk else jnp.zeros(NB, jnp.int32)
        s2, r, info = env.step(cfg, s, jnp.concatenate([act_b, act_m]), k2)
        E = s2.energy.at[:NB].set(1.0).at[NB + NL:].set(1.0)                              # beacons + elders immortal
        if a.juvenile:
            E = E.at[NB:NB + NL].set(jnp.where(young & s2.alive[NB:NB + NL], jnp.maximum(E[NB:NB + NL], cfg.E0), E[NB:NB + NL]))
        al = s2.alive.at[:NB].set(True).at[NB + NL:].set(True)
        s2 = s2._replace(energy=E, alive=al)
        dd = (s.pos[NB + NL:][None] - s.pos[NB:NB + NL][:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
        near_e = jnp.abs(dd).max(-1) <= R_HEAR                                             # (NL, NE)
        return s2, (o[:NL], act_l, lp, v, r[NB:NB + NL] * 20.0, info["dead"][NB:NB + NL], info["take"][NB:NB + NL],
                    o[NL:], act_m[NL:], near_e)

    def roll(P, s, key):
        def body(s, k):
            k1, k2 = jax.random.split(k)
            shift = jax.random.randint(k2, (), 0, 8) if a.random_code else 0
            young = s.age[NB:NB + NL] < a.juvenile
            yc = s.age[NB:NB + NL] < a.young
            s2, rec = step_all(P, s, k1, shift, young)
            return s2, rec + (s.alive[NB:NB + NL], yc)
        return jax.lax.scan(body, s, jax.random.split(key, T))
    roll_w = jax.jit(jax.vmap(roll))

    def pred_loss(p, o, alive):
        """targets from the agent's own consecutive steps: next view (98), energy change (1), max food in view 4 ahead (1)"""
        _, _, y = net.apply(p, o)
        nxt = o[1:, :98]; de = o[1:, 98] - o[:-1, 98]
        f4 = jnp.concatenate([o[4:, :49].max(1), jnp.zeros(4)])
        m4 = jnp.concatenate([alive[:-4] & alive[4:], jnp.zeros(4, bool)]).astype(jnp.float32)
        m1 = (alive[:-1] & alive[1:]).astype(jnp.float32)
        l1 = (((y[:-1, :98] - nxt) ** 2).mean(1) * m1).sum() / (m1.sum() + 1e-8)
        l2 = (((y[:-1, 98] - de * 20.0) ** 2) * m1).sum() / (m1.sum() + 1e-8)
        l3 = (((y[:, 99] - f4) ** 2) * m4).sum() / (m4.sum() + 1e-8)
        return l1 + l2 + l3

    def loss(p, o, a_, lp0, adv, ret, m, eo, ea, em):
        lg, v, _ = net.apply(p, o); ls = jax.nn.log_softmax(lg)
        lp = jnp.take_along_axis(ls, a_[:, None], 1)[:, 0]; ratio = jnp.exp(lp - lp0)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        L = ((pg + 0.5 * (v - ret) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)) * m).sum() / (m.sum() + 1e-8)
        if a.pred:
            L = L + a.pred * pred_loss(p, o, m > 0)
        if a.copy and NE:
            lge, _, _ = net.apply(p, eo)
            ce = -jnp.take_along_axis(jax.nn.log_softmax(lge), ea[:, None], 1)[:, 0]
            L = L + a.copy * (ce * em).sum() / (em.sum() + 1e-8)
        return L

    @jax.jit
    def upd(P, OS, f, *b):
        def one(p, os_, f_, *bb):
            g = jax.grad(loss)(p, *bb); u, os_ = opt.update(g, os_, p)
            return optax.apply_updates(p, jax.tree_util.tree_map(lambda x: x * f_, u)), os_
        return jax.vmap(jax.vmap(one))(P, OS, f, *b)

    def replay_loss(q, o, m, eo, ea, em):
        L = a.pred * pred_loss(q, o, m > 0) if a.pred else 0.0
        if a.copy_replay and a.copy and NE:
            lge, _, _ = net.apply(q, eo)
            ce = -jnp.take_along_axis(jax.nn.log_softmax(lge), ea[:, None], 1)[:, 0]
            L = L + a.copy * (ce * em).sum() / (em.sum() + 1e-8)
        return L

    @jax.jit
    def upd_replay(P, OS, f, o, m, eo, ea, em):
        def one(p, os_, f_, o, m, eo, ea, em):
            g = jax.grad(replay_loss)(p, o, m, eo, ea, em); u, os_ = opt.update(g, os_, p)
            return optax.apply_updates(p, jax.tree_util.tree_map(lambda x: x * f_, u)), os_
        return jax.vmap(jax.vmap(one))(P, OS, f, o, m, eo, ea, em)

    def eval_life(p, centres, key, shift_mode, perm=None, bpos=None, smask=None, mute=False, rand_fb=False):
        s = env.reset(cfg, key)
        s = s._replace(centres=centres, cap=env.capacity(cfg, centres), food=env.capacity(cfg, centres))
        s, _ = beacons_reset(s, jax.random.fold_in(key, 3))
        if bpos is not None:                      # --test-start life: the life's beacons; the learner beside one of them
            j = jax.random.randint(jax.random.fold_in(key, 11), (), 0, NB)
            lp0 = (bpos[j] + jax.random.randint(jax.random.fold_in(key, 12), (2,), -3, 4)) % cfg.L
            s = s._replace(pos=s.pos.at[:NB].set(bpos).at[NB].set(lp0))
        s = s._replace(alive=jnp.array([True] * NB + [True] + [False] * (M - 1)))
        sh = jax.random.randint(jax.random.fold_in(key, 9), (), 1, 8) * shift_mode
        P1 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NL,) + x.shape), p)
        def body(c, k):
            s, eaten, last = c
            if smask is not None:                     # the TRUE sector heard now (before any swap), for attribution
                ht = hear(s, 0)[0]; last = jnp.where(ht[8] > 0.5, jnp.argmax(ht[:8]), last)
            s2, rec = step_all(P1, s, k, sh, jnp.zeros(NL, bool), perm, mute, seek_ok=False, rand_fb=rand_fb)
            s2 = s2._replace(alive=s2.alive.at[NB + 1:].set(False))                      # only this learner (no elders)
            credit = 1.0 if smask is None else jnp.where(last >= 0, smask[jnp.maximum(last, 0)], 0.0)
            return (s2, eaten + rec[6][0] * credit, last), None
        (s, eaten, _), _ = jax.lax.scan(body, (s, 0.0, jnp.int32(-1)), jax.random.split(jax.random.fold_in(key, 5), a.test_window))
        return eaten
    ev = jax.jit(jax.vmap(jax.vmap(eval_life, in_axes=(None, None, 0, None)), in_axes=(0, 0, 0, None)))
    def make_evp(mute=False, rand_fb=False):   # the heard25 test: per-life, per-episode sector permutation (identity = true message); traced on first call
        life_start = a.test_start == "life"; lh = a.score == "last-heard"
        def one(p, c, k, q, b, m):
            return eval_life(p, c, k, 0, q, b if life_start else None, m if lh else None, mute, rand_fb)
        f = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, None, 0, 0, None, None)), in_axes=(0, 0, 0, 0, 0, 0)))
        return lambda p, c, k, q, b, m: f(p, c, k, q, b, m)

    rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed)
    states = []
    for w in range(W):
        s0 = env.reset(cfg, jax.random.PRNGKey(1000 + w)); s0, _ = beacons_reset(s0, jax.random.PRNGKey(2000 + w)); states.append(s0)
    S = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *states)
    nav_of = rng.integers(0, K, (W, NL))
    P = jax.tree_util.tree_map(lambda x: x[nav_of], NAV); OS = jax.vmap(jax.vmap(opt.init))(P)
    if a.hearing_probe:
        return hearing_probe(a, cfg, net, opt, NAV, nav_of, S, roll_w, ev, hear, W, NL, T, M)
    mem_o, mem_m, mem_e = [], [], []            # replay memory: the last a.mem rollouts, (W, NL, T, D) each (+ elder data)
    since_birth = np.zeros((W, NL), int)        # rollouts since birth (replay uses own memory only)
    gsteps = np.zeros((W, NL), int)
    cnt = np.zeros((W, NL, 8), np.int64)       # per-life heard-sector counts (coverage)
    SNAP = [int(x) for x in a.snap_ages.split(",")] if a.snap_ages else []
    snaps = {}; prev_age = np.zeros((W, NL), np.int64)
    lives = []
    it0 = 0
    if a.ckpt:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        import ckpt as _ck
        Ck = _ck.load(a.ckpt)
        if Ck is not None:
            tj = lambda t: jax.tree_util.tree_map(jnp.asarray, t)  # noqa: E731
            P, OS, S = tj(Ck["P"]), tj(Ck["OS"]), tj(Ck["S"]); key = jnp.asarray(Ck["key"]); rng = Ck["rng"]
            mem_o, mem_m = Ck["mem_o"], Ck["mem_m"]; mem_e = [(jnp.asarray(e0), jnp.asarray(e1), e2) for e0, e1, e2 in Ck["mem_e"]]
            since_birth, gsteps, cnt, lives, nav_of = Ck["since_birth"], Ck["gsteps"], Ck["cnt"], Ck["lives"], Ck["nav_of"]
            snaps, prev_age = Ck.get("snaps", {}), Ck.get("prev_age", prev_age)
            it0 = Ck["it"] + 1; print(f"resumed from update {Ck['it'] + 1}", flush=True)
    for it in range(it0, a.updates):
        key, k = jax.random.split(key)
        S, (o, act, lp, v, r, dn, take, eo, ea, ne, alive, yc) = roll_w(P, S, jax.random.split(k, W))
        o, act, lp, v, r, dn, alive = (np.asarray(x) for x in (o, act, lp, v, r, dn, alive))   # (W, T, NL, ...)
        hm = (o[..., 108] > 0.5) & alive
        cnt += np.einsum("wtl,wtlk->wlk", hm.astype(np.int64), (o[..., 100:108] > 0.5).astype(np.int64))
        adv = np.zeros_like(r); last = 0.0
        for t in reversed(range(T)):
            nv = v[:, t + 1] if t + 1 < T else v[:, t]; nt = 1.0 - dn[:, t]
            last = r[:, t] + 0.99 * nv * nt - v[:, t] + 0.99 * 0.95 * nt * last; adv[:, t] = last
        ret = adv + v; m = alive.astype(np.float32)
        A = (adv - (adv * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)) / (adv.std() + 1e-8)
        tr = lambda x: jnp.asarray(np.swapaxes(np.asarray(x), 1, 2))  # noqa: E731
        Dd = o.shape[-1]
        if NE:
            eo_, ea_ = np.asarray(eo), np.asarray(ea)                                       # (W, T, NE, D), (W, T, NE)
            em = (np.asarray(ne) & np.asarray(yc)[..., None] & alive[..., None]).astype(np.float32)   # (W, T, NL, NE)
            EO = jnp.asarray(np.broadcast_to(eo_[:, None], (W, NL) + eo_.shape[1:]).reshape(W, NL, T * NE, Dd))
            EA = jnp.asarray(np.broadcast_to(ea_[:, None], (W, NL) + ea_.shape[1:]).reshape(W, NL, T * NE))
            EM = jnp.asarray(np.swapaxes(em, 1, 2).reshape(W, NL, T * NE))
        else:
            EO = jnp.zeros((W, NL, 1, Dd)); EA = jnp.zeros((W, NL, 1), jnp.int32); EM = jnp.zeros((W, NL, 1))
        agel = np.asarray(S.age)[:, NB:NB + NL]
        fy = jnp.asarray(1.0 + (a.young_lr - 1.0) * np.clip(1.0 - agel / max(a.juvenile, 1), 0.0, 1.0), jnp.float32)
        for _ in range(0 if a.no_train else a.epochs):
            P, OS = upd(P, OS, fy, tr(o), tr(act), tr(lp), tr(A.astype(np.float32)), tr(ret.astype(np.float32)), tr(m), EO, EA, EM)
        gsteps += a.epochs
        since_birth += 1
        if a.replay and (a.pred or a.copy_replay) and not a.no_train:
            mem_o.append(np.swapaxes(o, 1, 2)); mem_m.append(np.swapaxes(m, 1, 2)); mem_e.append((EO, EA, np.asarray(EM)))
            mem_o, mem_m, mem_e = mem_o[-a.mem:], mem_m[-a.mem:], mem_e[-a.mem:]
            for _ in range(a.replay):
                for j in range(len(mem_o)):          # own memory since birth only: older rollouts masked per learner
                    age_ok = (len(mem_o) - j) <= since_birth
                    eoj, eaj, emj = mem_e[j]
                    P, OS = upd_replay(P, OS, fy, jnp.asarray(mem_o[j]), jnp.asarray(mem_m[j] * age_ok[..., None]),
                                       eoj, eaj, jnp.asarray(emj * age_ok[..., None]))
            gsteps += a.replay * len(mem_o)
        dead = ~np.asarray(S.alive)[:, NB:NB + NL]
        ages = np.asarray(S.age)[:, NB:NB + NL]
        for k_ in SNAP:                             # weights snapshot when a life first reaches age k_
            for w, l in zip(*np.where((prev_age < k_) & (ages >= k_) & ~dead)):
                snaps.setdefault((int(w), int(l)), []).append((k_, jax.tree_util.tree_map(lambda x: np.asarray(x[w, l]), P)))
        prev_age = np.where(dead, 0, ages)
        for w, l in zip(*np.where(dead)):
            lives.append(dict(lifespan=int(ages[w, l]), grad_steps=int(gsteps[w, l]), heard=cnt[w, l].copy(), bpos=np.asarray(S.pos)[w, :NB].copy(),
                              params=jax.tree_util.tree_map(lambda x: x[w, l], P), centres=np.asarray(S.centres[w]),
                              snaps=snaps.pop((int(w), int(l)), [])))
            nav_of[w, l] = rng.integers(0, K)
        if dead.any():
            fresh = jax.tree_util.tree_map(lambda x: x[nav_of], NAV); dm = jnp.asarray(dead)
            P = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), P, fresh)
            OS = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), OS,
                                        jax.vmap(jax.vmap(opt.init))(fresh))
            since_birth[dead] = 0; gsteps[dead] = 0; cnt[dead] = 0
            npos = (np.asarray(S.pos)[:, :NB].mean(1, keepdims=True).astype(int) + rng.integers(-3, 4, (W, NL, 2))) % cfg.L
            pos = np.asarray(S.pos).copy(); E = np.asarray(S.energy).copy(); age = np.asarray(S.age).copy(); al = np.asarray(S.alive).copy()
            sl = slice(NB, NB + NL)
            pos[:, sl][dead] = npos[dead]; E[:, sl][dead] = cfg.E0; age[:, sl][dead] = 0; al[:, sl][dead] = True
            S = S._replace(pos=jnp.asarray(pos), energy=jnp.asarray(E), age=jnp.asarray(age), alive=jnp.asarray(al))
        if a.ckpt and (it + 1) % a.ckpt_every == 0 and it + 1 < a.updates:
            _ck.save(a.ckpt, dict(it=it, P=P, OS=OS, S=S, key=key, rng=rng, mem_o=mem_o, mem_m=mem_m, mem_e=mem_e,
                                  since_birth=since_birth, gsteps=gsteps, cnt=cnt, lives=lives, nav_of=nav_of, snaps=snaps, prev_age=prev_age))
        if a.stop_after and it + 1 == a.stop_after:
            os._exit(3)
    if not lives:
        print("no completed lives"); return
    diag = None
    if a.diag and a.pred:
        def pred_trace(p, centres, key):
            """a true-message test episode; at each step the head's food-in-view-4-ahead prediction with the heard
            sector as heard and with it shifted by +4 (on the SAME observation), plus the realised target"""
            s = env.reset(cfg, key)
            s = s._replace(centres=centres, cap=env.capacity(cfg, centres), food=env.capacity(cfg, centres))
            s, _ = beacons_reset(s, jax.random.fold_in(key, 3))
            s = s._replace(alive=jnp.array([True] * NB + [True] + [False] * (M - 1)))
            P1 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NL,) + x.shape), p)
            def body(s, k):
                h = hear(s, 0); o = jnp.concatenate([env.observe(cfg, s)[NB:], h], 1)[0]
                osw = o.at[100:108].set(jnp.roll(o[100:108], 4))
                _, _, y = net.apply(p, o[None]); _, _, ys = net.apply(p, osw[None])
                s2, _ = step_all(P1, s, k, 0, jnp.zeros(NL, bool))
                s2 = s2._replace(alive=s2.alive.at[NB + 1:].set(False))
                return s2, (y[0, 99], ys[0, 99], o[:49].max(), o[108] > 0.5)
            _, (yt, ys, fv, heard) = jax.lax.scan(body, s, jax.random.split(jax.random.fold_in(key, 5), 300))
            return yt, ys, fv, heard
        idx = np.linspace(0, len(lives) - 1, min(a.diag, len(lives))).astype(int)
        errs_t, errs_s = [], []
        for i in idx:
            for e in range(4):
                yt, ys, fv, hd = (np.asarray(x) for x in jax.jit(pred_trace)(lives[i]["params"], jnp.asarray(lives[i]["centres"]),
                                                                              jax.random.PRNGKey(9000 + 10 * int(i) + e)))
                tgt = np.concatenate([fv[4:], np.full(4, np.nan)]); mk = hd & ~np.isnan(tgt)
                if mk.sum() > 5:
                    errs_t.append(np.mean((yt[mk] - tgt[mk]) ** 2)); errs_s.append(np.mean((ys[mk] - tgt[mk]) ** 2))
        et, es_ = np.array(errs_t), np.array(errs_s); dd = es_ - et
        diag = dict(n=len(et), mse_true=float(et.mean()), mse_swapped=float(es_.mean()),
                    swap_increase=float(dd.mean() / max(et.mean(), 1e-9)), paired_z=float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)))
        print("DIAG predictive head uses the message?", json.dumps(diag), flush=True)
    Pz = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[x["params"] for x in lives])
    C = jnp.asarray(np.stack([x["centres"] for x in lives]))
    keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(7000 + i), a.eval_eps))(jnp.arange(len(lives)))
    scorable = np.ones(len(lives), bool)
    if a.test == "heard25":
        rq = np.random.default_rng(4242); E_ = a.eval_eps
        ident = np.tile(np.arange(8), (len(lives), E_, 1)); der = ident.copy()
        for i, x in enumerate(lives):
            Sset = np.where(x["heard"] >= a.min_heard)[0]
            if len(Sset) < max(2, a.min_sectors):
                scorable[i] = False
            if len(Sset) < 2:
                continue
            for e in range(E_):
                while True:
                    q = rq.permutation(Sset)
                    if np.all(q != Sset):
                        break
                der[i, e, Sset] = q
        evp = make_evp(); BP = jnp.asarray(np.stack([x["bpos"] for x in lives]))
        SM = jnp.asarray(np.stack([(x["heard"] >= a.min_heard).astype(np.float32) for x in lives]))
        true = np.asarray(evp(Pz, C, keys, jnp.asarray(ident), BP, SM)); swap = np.asarray(evp(Pz, C, keys, jnp.asarray(der), BP, SM))
    else:
        true = np.asarray(ev(Pz, C, keys, 0)); swap = np.asarray(ev(Pz, C, keys, 1))
    blind = silent = blind_swap = None; evaluate_voi_r = None
    if a.no_train and a.sanity and a.test == "heard25":   # the blind comparison on the SAME lives / keys: the navigator policy
        a.sanity = False
        evn = make_evp()
        blind = np.asarray(evn(Pz, C, keys, jnp.asarray(np.tile(np.arange(8), (len(lives), a.eval_eps, 1))), BP, SM))
        blind_swap = np.asarray(evn(Pz, C, keys, jnp.asarray(der), BP, SM))        # negative control: the non-listener's life test
        a.sanity = True
        I8 = jnp.asarray(np.tile(np.arange(8), (len(lives), a.eval_eps, 1)))
        silent = np.asarray(make_evp(mute=True)(Pz, C, keys, I8, BP, SM))
        # appetite-free VoI (Nov 10): the scripted listener's own uninformed move (random) when it hears nothing, both arms
        true_r = np.asarray(make_evp(rand_fb=True)(Pz, C, keys, I8, BP, SM)); silent_r = np.asarray(make_evp(mute=True, rand_fb=True)(Pz, C, keys, I8, BP, SM))
        evaluate_voi_r = (true_r, silent_r)
    res = []
    for i, x in enumerate(lives):
        t, s_ = true[i], swap[i]; d = t - s_; se = d.std(ddof=1) / np.sqrt(len(d))
        drop = 1 - s_.sum() / max(t.sum(), 1e-9)
        res.append(dict(lifespan=x["lifespan"], grad_steps=x["grad_steps"], eaten_true=float(t.mean()), eaten_swap=float(s_.mean()),
                        drop=float(drop), passed=bool(scorable[i] and drop >= 0.30 and d.mean() > 2 * se), scorable=bool(scorable[i]),
                        distinct_heard=int((x["heard"] > 0).sum()), sectors_ge_min=int((x["heard"] >= a.min_heard).sum()),
                        eaten_blind=float(blind[i].mean()) if blind is not None else None))
    frac = np.mean([r_["passed"] for r_ in res])
    summ = dict(variant=dict(juvenile=a.juvenile, long_life=a.long_life, pred=a.pred, replay=a.replay, copy=a.copy,
                             copy_replay=a.copy_replay, young_lr=a.young_lr,
                             n_elders=NE, sanity=a.sanity, random_code=a.random_code, updates=a.updates, T=T, epochs=a.epochs),
                lives=len(res), pass_frac=float(frac), median_drop=float(np.median([r_["drop"] for r_ in res])),
                median_lifespan=float(np.median([r_["lifespan"] for r_ in res])),
                median_grad_steps_per_life=float(np.median([r_["grad_steps"] for r_ in res])),
                eaten_true_mean=float(np.mean([r_["eaten_true"] for r_ in res])),
                eaten_swap_mean=float(np.mean([r_["eaten_swap"] for r_ in res])), gate="PASS" if frac >= 0.5 else "FAIL",
                diag=diag)
    if a.no_train:
        dh = np.array([r_["distinct_heard"] for r_ in res]); gm = np.array([r_["sectors_ge_min"] for r_ in res])
        summ["apparatus"] = dict(
            test_start=a.test_start, score=a.score, beacon_spread=a.beacon_spread, beacon_walk=a.beacon_walk, young_seek=a.young_seek, world=a.world, test=a.test,
            test_window=a.test_window, eval_eps=a.eval_eps, policy="scripted listener" if a.sanity else "navigator (no training)",
            coverage_median_distinct=float(np.median(dh)), coverage_median_sectors_ge_min=float(np.median(gm)),
            coverage_hist_distinct=np.bincount(dh, minlength=9).tolist(),
            gate_i=bool(np.median(dh) >= 4 and np.median(gm) >= 3), scorable_frac=float(scorable.mean()))
        if blind is not None:
            eb = float(np.mean([r_["eaten_blind"] for r_ in res])); summ["apparatus"]["eaten_blind_mean"] = eb
            summ["apparatus"]["scripted_over_blind"] = float(summ["eaten_true_mean"] / max(eb, 1e-9))
            summ["apparatus"]["gate_ii"] = bool(frac >= 0.9 and summ["eaten_true_mean"] >= 2 * eb)
            es = float(silent.mean()); summ["apparatus"]["eaten_scripted_silence_mean"] = es
            summ["apparatus"]["scripted_true_over_silence"] = float(summ["eaten_true_mean"] / max(es, 1e-9))
    if a.test == "heard25":
        summ["heard25"] = heard25_report(a, lives, res, scorable, true, swap, der, blind, blind_swap, silent)
        if evaluate_voi_r is not None:
            el = np.asarray(scorable); tm, sm = evaluate_voi_r[0][el].mean(1), evaluate_voi_r[1][el].mean(1); d = tm - sm
            summ["heard25"]["voi_random_fallback"] = dict(eaten_true=float(tm.mean()), eaten_silence=float(sm.mean()),
                                                          ratio=float(tm.mean() / max(sm.mean(), 1e-9)),
                                                          paired_z=float(d.mean() / max(d.std(ddof=1) / np.sqrt(len(d)), 1e-12)))
        if SNAP:                                     # the same test on the age snapshots: steps to first reliance
            byage = {}; first = np.full(len(lives), np.inf)
            for k_ in SNAP:
                idx = [i for i, x in enumerate(lives) if scorable[i] and any(kk == k_ for kk, _ in x["snaps"])]
                if not idx:
                    byage[str(k_)] = dict(n=0); continue
                Pk = jax.tree_util.tree_map(lambda *z: jnp.stack([jnp.asarray(q) for q in z]), *[dict(x["snaps"])[k_] for x in (lives[i] for i in idx)])
                ii = np.array(idx); tk = np.asarray(evp(Pk, C[ii], keys[ii], jnp.asarray(ident[ii]), BP[ii], SM[ii]))
                sk = np.asarray(evp(Pk, C[ii], keys[ii], jnp.asarray(der[ii]), BP[ii], SM[ii]))
                pk = np.array([_life_pass(tk[j], sk[j]) for j in range(len(idx))])
                for j, i in enumerate(idx):
                    res[i].setdefault("snap_pass", {})[str(k_)] = bool(pk[j])     # per-life, for paired age trends
                    if pk[j]:
                        first[i] = min(first[i], k_)
                byage[str(k_)] = dict(n=len(idx), pass_frac=float(pk.mean()), ci95=_wilson(int(pk.sum()), len(idx)))
            endp = np.array([r_["passed"] for r_ in res])
            for i in range(len(lives)):
                if scorable[i] and endp[i]:
                    first[i] = min(first[i], lives[i]["lifespan"])
            fe = first[np.asarray(scorable)]
            summ["heard25"]["by_age"] = byage
            summ["heard25"]["first_reliance"] = dict(ever_frac=float(np.isfinite(fe).mean()) if len(fe) else None,
                                                     median_age_among_ever=float(np.median(fe[np.isfinite(fe)])) if np.isfinite(fe).any() else None,
                                                     median_lifespan=float(np.median([x["lifespan"] for x in lives])))
        summ["gate"] = "PASS" if summ["heard25"]["learner_pass_frac_eligible"] >= 0.5 else "FAIL"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summ, lives=res), open(a.out, "w"), indent=1, default=lambda z: z.tolist() if hasattr(z, "tolist") else str(z))
    print(json.dumps(summ))


def _life_pass(t, s_):
    d = t - s_; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
    return bool((1 - s_.sum() / max(t.sum(), 1e-9)) >= 0.30 and d.mean() > 2 * se)


def _wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def heard25_report(a, lives, res, scorable, true, swap, der, blind, blind_swap, silent):
    """Cam Nov 10 rulings: last-heard metric; a life is eligible iff it heard >= min-sectors sectors >= min-heard x; pass
    rates over ELIGIBLE lives (>= 50 needed), eligibility rate, stratified 5 / 6 / 7-8; secondary (no bar): probes whose
    swap moved the well-heard sectors >= 90 degrees on average; apparatus: negative control (non-listener) and VoI (true vs silence)."""
    el = np.asarray(scorable); n_el = int(el.sum()); gm = np.array([r_["sectors_ge_min"] for r_ in res])
    passed = np.array([r_["passed"] for r_ in res])
    out = dict(min_sectors=a.min_sectors, min_heard=a.min_heard, score=a.score, lives=len(res), eligible=n_el,
               eligibility_rate=float(el.mean()), enough_eligible=n_el >= 50,
               learner_pass_frac_eligible=float(passed[el].mean()) if n_el else float("nan"),
               learner_pass_ci95=_wilson(int(passed[el].sum()), n_el))
    out["stratified"] = {k: dict(n=int(m.sum()), pass_frac=float(passed[m].mean()) if m.any() else None)
                         for k, m in (("5", el & (gm == 5)), ("6", el & (gm == 6)), ("7-8", el & (gm >= 7)))}
    # secondary: per life, only the probes whose derangement moved every well-heard sector >= 90 degrees (>= 2 sectors)
    sec = []
    for i, x in enumerate(lives):
        if not el[i]:
            continue
        S_ = np.where(x["heard"] >= a.min_heard)[0]
        mv = np.abs(der[i][:, S_] - S_[None]); mv = np.minimum(mv, 8 - mv)
        ok = mv.mean(1) >= 2                        # mean move over the well-heard sectors >= 90 degrees (2 sector steps)
        if ok.sum() >= 4:
            sec.append(_life_pass(true[i][ok], swap[i][ok]))
    out["secondary_swap_ge90deg"] = dict(lives_with_ge4_such_probes=len(sec), pass_frac=float(np.mean(sec)) if sec else None)
    if blind_swap is not None:
        neg = np.array([_life_pass(blind[i], blind_swap[i]) for i in range(len(lives))])
        out["negative_control_nonlistener"] = dict(pass_frac=float(neg[el].mean()) if n_el else None,
                                                   ci95=_wilson(int(neg[el].sum()), n_el), ok=bool(neg[el].mean() <= 0.10) if n_el else False)
        out["scripted_control"] = dict(pass_frac=out["learner_pass_frac_eligible"], ci95=out["learner_pass_ci95"],
                                       ok=bool(out["learner_pass_frac_eligible"] >= 0.90))
    if silent is not None:
        tm, sm = true[el].mean(1), silent[el].mean(1); d = tm - sm; se = d.std(ddof=1) / np.sqrt(len(d))
        ratio = float(tm.mean() / max(sm.mean(), 1e-9))
        out["voi_true_vs_silence"] = dict(eaten_true=float(tm.mean()), eaten_silence=float(sm.mean()), ratio=ratio,
                                          paired_z=float(d.mean() / max(se, 1e-12)), ok=bool(ratio >= 1.5 and d.mean() > 2 * se))
    if blind_swap is not None and silent is not None:
        out["gate_ii"] = bool(out["scripted_control"]["ok"] and out["negative_control_nonlistener"]["ok"]
                              and out["voi_true_vs_silence"]["ok"] and out["enough_eligible"])
    return out


def hearing_probe(a, cfg, net, opt, NAV, nav_of, S, roll_w, ev, hear, W, NL, T, M):
    """Cam Nov 10 ~05:15 (4a). Train ONLY the hearing rows (Dense_0 kernel rows 100:111) + the policy head (Dense_2)
    of each learner, starting from its navigator, by cross-entropy on the scripted listener's actions given
    (obs, heard), Adam at the arm's rate (3e-4) x a multiplier, --probe-steps steps of batch 128 from the learner's own
    world's teacher data. Report held-out action accuracy (other worlds) and the harness's unchanged life-test gate.
    Pipeline checks: the input layout, the heard sector vs an independent beacon -> nearest-patch sector, the
    gradient reaching the heard rows, the heard input changing the logits after training."""
    P0 = jax.tree_util.tree_map(lambda x: x[nav_of], NAV)
    a.sanity = True                                           # the scripted listener drives the collection rollouts

    def collect(S, seed):
        key = jax.random.PRNGKey(seed); Os, As, Ms = [], [], []
        for _ in range(a.probe_collect):
            key, k = jax.random.split(key)
            S, rec = roll_w(P0, S, jax.random.split(k, W))
            Os.append(np.asarray(rec[0])); As.append(np.asarray(rec[1])); Ms.append(np.asarray(rec[10]))
        return np.concatenate(Os, 1), np.concatenate(As, 1), np.concatenate(Ms, 1)       # (W, Tn, NL, ...)

    def fresh_worlds(base):
        st = []
        for w in range(W):
            s0 = env.reset(cfg, jax.random.PRNGKey(base + w))
            c = jax.random.randint(jax.random.PRNGKey(base + 500 + w), (2,), 0, cfg.L)
            bpos = (c[None] + jnp.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])) % cfg.L
            mpos = (c[None] + jax.random.randint(jax.random.PRNGKey(base + 900 + w), (M, 2), -3, 4)) % cfg.L
            st.append(s0._replace(pos=jnp.concatenate([bpos, mpos], 0)))
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *st)

    O, A, Mk = collect(S, 11)
    Oh, Ah, Mh = collect(fresh_worlds(30_000), 12)                                     # held-out worlds
    a.sanity = False                                          # the life test (traced below) uses the learners' own policy
    D = O.shape[-1]
    heard = (O[..., 108] > 0.5) & Mk; heard_h = (Oh[..., 108] > 0.5) & Mh
    # code-driven states: nothing to eat underfoot (centre cell < 0.2) and no food seen (> 0.2) in view
    def code_driven(o):
        r = cfg.r; wv = o[..., :(2 * r + 1) ** 2]; centre = wv[..., ((2 * r + 1) ** 2) // 2]
        return (centre < 0.2) & (wv.max(-1) <= 0.2)
    cd_h = heard_h & code_driven(Oh)

    # ---- pipeline checks
    chk = dict(input_dim=int(D), heard_dims="100:108 one-hot sector, 108 flag, 109:111 offset / R_HEAR",
               kernel_rows=int(P0["params"]["Dense_0"]["kernel"].shape[-2]))
    oh = O[heard]; chk["onehot_sums_to_1"] = bool(np.allclose(oh[:, 100:108].sum(1), 1.0))
    chk["offset_within_unit"] = bool(np.abs(oh[:, 109:111]).max() <= 1.0 + 1e-6)
    chk["unheard_all_zero"] = bool(np.abs(O[(O[..., 108] < 0.5) & Mk][:, 100:111]).max() == 0.0) if ((O[..., 108] < 0.5) & Mk).any() else True
    chk["heard_frac_alive_states"] = float(heard.sum() / max(Mk.sum(), 1))
    chk["code_driven_frac_heard_heldout"] = float(cd_h.sum() / max(heard_h.sum(), 1))
    # independent sector: each world's beacons -> nearest patch, unclipped torus offset, vs the beacon's broadcast
    s_ = jax.tree_util.tree_map(lambda x: x[0], S)
    bp = np.asarray(s_.pos[:NB]); cen = np.asarray(s_.centres); L = cfg.L
    d = (cen[None] - bp[:, None] + L // 2) % L - L // 2; i = np.abs(d).sum(-1).argmin(1); off = d[np.arange(NB), i]
    true_sec = np.mod(np.round(np.arctan2(off[:, 1], off[:, 0]) / (2 * np.pi / 8)).astype(int), 8)
    bc_sec = np.asarray(sector(env.oracle_offset(cfg, s_))[:NB])
    chk["beacon_offsets_unclipped"] = off.tolist(); chk["r_oracle"] = int(cfg.r_oracle)
    chk["broadcast_sector"] = bc_sec.tolist(); chk["true_direction_sector"] = true_sec.tolist()
    chk["broadcast_eq_true_direction"] = float((bc_sec == true_sec).mean())
    print("PIPELINE", json.dumps(chk), flush=True)

    tm = {"params": {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} for k, v in P0["params"].items()}}
    for k in tm["params"]:
        for kk in tm["params"][k]:
            tm["params"][k][kk] = jnp.ones_like(tm["params"][k][kk]) if k == "Dense_2" else tm["params"][k][kk]
    tm["params"]["Dense_0"]["kernel"] = jnp.zeros_like(tm["params"]["Dense_0"]["kernel"]).at[..., 100:, :].set(1.0)
    if a.probe_train == "all":
        tm = jax.tree_util.tree_map(jnp.ones_like, tm)
    tm1 = jax.tree_util.tree_map(lambda x: x[0, 0], tm)
    # target consistency: on heard code-driven states the scripted action must be the heading of the heard sector
    def heading(sec):
        ang = sec * (2 * np.pi / 8); dx = np.round(np.cos(ang) * 3); dy = np.round(np.sin(ang) * 3)
        return np.where(np.abs(dx) >= np.abs(dy), np.where(dx < 0, 1, 2), np.where(dy > 0, 3, 4))
    Ocd, Acd = Oh[cd_h], Ah[cd_h]
    chk["target_eq_heading_of_heard_sector_codedriven"] = float((heading(Ocd[:, 100:108].argmax(1)) == Acd).mean()) if len(Acd) else None
    chk["codedriven_action_hist"] = np.bincount(Acd, minlength=6).tolist() if len(Acd) else None
    # first-layer saturation of the navigator trunk on heard states
    p00 = jax.tree_util.tree_map(lambda x: x[0, 0], P0)
    pre = np.asarray(jnp.asarray(oh[:2000]) @ p00["params"]["Dense_0"]["kernel"] + p00["params"]["Dense_0"]["bias"])
    chk["dense0_tanh_saturated_frac(|tanh|>0.99)"] = float((np.abs(np.tanh(pre)) > 0.99).mean())
    chk["dense0_preact_abs_median"] = float(np.median(np.abs(pre)))
    print("PIPELINE2", json.dumps(chk), flush=True)

    def ce(p, o, y):
        lg, _, _ = net.apply(p, o)
        return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], 1)[:, 0].mean()

    def one(p, os_, o, y, f):
        g = jax.tree_util.tree_map(lambda x, m: x * m, jax.grad(ce)(p, o, y), tm1)
        u, os_ = opt.update(g, os_, p)
        return optax.apply_updates(p, jax.tree_util.tree_map(lambda x: x * f, u)), os_
    upd = jax.jit(jax.vmap(jax.vmap(one, in_axes=(0, 0, 0, 0, None)), in_axes=(0, 0, 0, 0, None)))
    g0 = jax.grad(ce)(jax.tree_util.tree_map(lambda x: x[0, 0], P0), jnp.asarray(oh[:128]), jnp.asarray(A[heard][:128]))
    chk_g = float(jnp.abs(g0["params"]["Dense_0"]["kernel"][100:]).max())
    print("PIPELINE grad into heard rows at step 0 (max |g|):", chk_g, flush=True)

    rng = np.random.default_rng(a.seed + 77)
    Opool, Apool = O[heard], A[heard]
    idx = [[np.where(heard[w, :, l])[0] for l in range(NL)] for w in range(W)]
    Ohh, Ahh = Oh[heard_h], Ah[heard_h]; CDm = code_driven(Ohh)
    sub = rng.choice(len(Ohh), min(4000, len(Ohh)), replace=False); Ohs, Ahs, CDs = jnp.asarray(Ohh[sub]), Ahh[sub], CDm[sub]
    acc_f = jax.jit(jax.vmap(jax.vmap(lambda p: net.apply(p, Ohs)[0].argmax(-1))))

    def accuracy(P):
        pr = np.asarray(acc_f(P)); hit = pr == Ahs[None, None]
        return float(hit.mean()), float(hit[..., CDs].mean()) if CDs.any() else float("nan")

    out = dict(checks=chk, grad_heard_rows_step0=chk_g, n_train_heard=int(heard.sum()), n_heldout_heard=int(heard_h.sum()),
               n_heldout_code_driven=int(cd_h.sum()))
    a0 = accuracy(P0); out["navigator_accuracy"] = dict(all=a0[0], code_driven=a0[1])
    print("navigator (untrained) held-out accuracy all / code-driven:", a0, flush=True)
    C = jnp.asarray(np.repeat(np.asarray(S.centres), NL, 0))
    keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(7000 + i), a.eval_eps))(jnp.arange(W * NL))

    def gate(P):
        Pz = jax.tree_util.tree_map(lambda x: x.reshape((W * NL,) + x.shape[2:]), P)
        t = np.asarray(ev(Pz, C, keys, 0)); s2 = np.asarray(ev(Pz, C, keys, 1)); dd = t - s2
        se = dd.std(1, ddof=1) / np.sqrt(dd.shape[1]); drop = 1 - s2.sum(1) / np.maximum(t.sum(1), 1e-9)
        ok = (drop >= 0.30) & (dd.mean(1) > 2 * se)
        return dict(pass_frac=float(ok.mean()), median_drop=float(np.median(drop)), eaten_true=float(t.mean()), eaten_swap=float(s2.mean()))
    out["navigator_gate"] = gate(P0); print("navigator gate", json.dumps(out["navigator_gate"]), flush=True)
    out["runs"] = {}
    for fm in [float(x) for x in a.probe_lrs.split(",")]:
        P = P0; OS = jax.vmap(jax.vmap(opt.init))(P0); curve = []
        for st in range(a.probe_steps):
            if a.probe_pool:
                j = rng.choice(len(Opool), (W, NL, 128)); ob = jnp.asarray(Opool[j]); yb = jnp.asarray(Apool[j])
            else:
                b = np.stack([np.stack([rng.choice(idx[w][l], 128) if len(idx[w][l]) else np.zeros(128, int) for l in range(NL)]) for w in range(W)])
                ob = jnp.asarray(O[np.arange(W)[:, None, None], b, np.arange(NL)[None, :, None]])
                yb = jnp.asarray(A[np.arange(W)[:, None, None], b, np.arange(NL)[None, :, None]])
            P, OS = upd(P, OS, ob, yb, fm)
            if (st + 1) in (50, 100, 200, 300, 1000, 2000, a.probe_steps):
                curve.append((st + 1,) + accuracy(P))
        acc = accuracy(P); g = gate(P)
        # the heard input reaches the policy: flip the heard sector by 4 on held-out code-driven states
        o_sw = Ohs.at[:, 100:108].set(jnp.roll(Ohs[:, 100:108], 4, axis=1))
        l1 = np.asarray(jax.vmap(jax.vmap(lambda p: net.apply(p, Ohs)[0]))(P)); l2 = np.asarray(jax.vmap(jax.vmap(lambda p: net.apply(p, o_sw)[0]))(P))
        # coverage: which heard sectors did each learner see in training? accuracy on held-out code-driven states split
        # by whether the state's sector was among them
        seen = np.zeros((W, NL, 8), bool)
        for w in range(W):
            for l in range(NL):
                if a.probe_pool:
                    seen[w, l, np.unique(Opool[:, 100:108].argmax(1))] = True
                elif len(idx[w][l]):
                    seen[w, l, np.unique(O[w, idx[w][l], l, 100:108].argmax(1))] = True
        secs = np.asarray(Ohs[:, 100:108]).argmax(1)
        prd = np.asarray(acc_f(P)); hit = prd == Ahs[None, None]
        sm = seen[:, :, secs] & CDs[None, None]; um = (~seen[:, :, secs]) & CDs[None, None]
        cov = dict(distinct_sectors_heard_per_learner_median=float(np.median(seen.sum(-1))),
                   distinct_sectors_hist=np.bincount(seen.sum(-1).ravel(), minlength=9).tolist(),
                   acc_codedriven_seen_sector=float(hit[sm].mean()) if sm.any() else None,
                   acc_codedriven_unseen_sector=float(hit[um].mean()) if um.any() else None,
                   frac_codedriven_unseen=float(um.sum() / max(sm.sum() + um.sum(), 1)))
        # training-set accuracy (pooled mode: on a sample of the pooled training states; code-driven split)
        if a.probe_pool:
            jt = rng.choice(len(Opool), min(4000, len(Opool)), replace=False)
            Ot, At = jnp.asarray(Opool[jt]), Apool[jt]; CDt = code_driven(Opool[jt])
            pt = np.asarray(jax.vmap(jax.vmap(lambda p: net.apply(p, Ot)[0].argmax(-1)))(P)); ht = pt == At[None, None]
            cov["train_acc_all"] = float(ht.mean()); cov["train_acc_code_driven"] = float(ht[..., CDt].mean())
        r = dict(coverage=cov, lr=3e-4 * fm, steps=a.probe_steps, heldout_accuracy_all=acc[0], heldout_accuracy_code_driven=acc[1],
                 curve=curve, gate=g, gate_pass=g["pass_frac"] >= 0.5, accuracy_pass=acc[0] >= 0.9,
                 logit_change_on_sector_flip=float(np.abs(l1 - l2).mean()))
        out["runs"][f"x{fm:g}"] = r; print("PROBE", json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
