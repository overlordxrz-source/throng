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
    a = ap.parse_args()
    NL = a.n_learners; NE = a.n_elders if a.copy > 0 else 0; M = NL + NE
    cfg = env.Cfg(L=96, N=NB + M, P=6, patch_sigma=1.0, d0=0.004 if a.long_life else 0.008, reproduce=False, A_max=3000)
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
        bpos = (c[None] + jnp.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])) % cfg.L
        mpos = (c[None] + jax.random.randint(jax.random.fold_in(key, 1), (M, 2), -3, 4)) % cfg.L
        return s._replace(pos=jnp.concatenate([bpos, mpos], 0)), c

    def hear(s, shift):
        """(M, HD) for the non-beacon agents: nearest beacon within radius; its sector (+ shift) one-hot, flag, offset / R"""
        sec = jnp.mod(sector(env.oracle_offset(cfg, s))[:NB] + shift, 8)
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

    def step_all(P, s, key, shift, young):
        k1, k2 = jax.random.split(key)
        h = hear(s, shift)
        o = jnp.concatenate([env.observe(cfg, s)[NB:], h], 1)                              # (M, D)
        lg, v, _ = jax.vmap(lambda p, x: net.apply(p, x[None]))(P, o[:NL]); lg, v = lg[:, 0], v[:, 0]
        act_l = jax.random.categorical(k1, lg)
        sc = scripted(s, h, jnp.concatenate([act_l, jax.random.randint(k2, (NE,), 1, 5)]))
        if a.sanity:
            act_l = sc[:NL]
        if a.juvenile:                                    # the young stay within hearing range of the beacons
            act_l = jnp.where(young & (h[:NL, 8] < 0.5), 0, act_l)
        lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act_l[:, None], 1)[:, 0]
        act_m = jnp.concatenate([act_l, sc[NL:]])
        s2, r, info = env.step(cfg, s, jnp.concatenate([jnp.zeros(NB, jnp.int32), act_m]), k2)
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

    def eval_life(p, centres, key, shift_mode):
        s = env.reset(cfg, key)
        s = s._replace(centres=centres, cap=env.capacity(cfg, centres), food=env.capacity(cfg, centres))
        s, _ = beacons_reset(s, jax.random.fold_in(key, 3))
        s = s._replace(alive=jnp.array([True] * NB + [True] + [False] * (M - 1)))
        sh = jax.random.randint(jax.random.fold_in(key, 9), (), 1, 8) * shift_mode
        P1 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NL,) + x.shape), p)
        def body(c, k):
            s, eaten = c
            s2, rec = step_all(P1, s, k, sh, jnp.zeros(NL, bool))
            s2 = s2._replace(alive=s2.alive.at[NB + 1:].set(False))                      # only this learner (no elders)
            return (s2, eaten + rec[6][0]), None
        (s, eaten), _ = jax.lax.scan(body, (s, 0.0), jax.random.split(jax.random.fold_in(key, 5), 300))
        return eaten
    ev = jax.jit(jax.vmap(jax.vmap(eval_life, in_axes=(None, None, 0, None)), in_axes=(0, 0, 0, None)))

    rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed)
    states = []
    for w in range(W):
        s0 = env.reset(cfg, jax.random.PRNGKey(1000 + w)); s0, _ = beacons_reset(s0, jax.random.PRNGKey(2000 + w)); states.append(s0)
    S = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *states)
    nav_of = rng.integers(0, K, (W, NL))
    P = jax.tree_util.tree_map(lambda x: x[nav_of], NAV); OS = jax.vmap(jax.vmap(opt.init))(P)
    mem_o, mem_m, mem_e = [], [], []            # replay memory: the last a.mem rollouts, (W, NL, T, D) each (+ elder data)
    since_birth = np.zeros((W, NL), int)        # rollouts since birth (replay uses own memory only)
    gsteps = np.zeros((W, NL), int)
    lives = []
    for it in range(a.updates):
        key, k = jax.random.split(key)
        S, (o, act, lp, v, r, dn, take, eo, ea, ne, alive, yc) = roll_w(P, S, jax.random.split(k, W))
        o, act, lp, v, r, dn, alive = (np.asarray(x) for x in (o, act, lp, v, r, dn, alive))   # (W, T, NL, ...)
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
        for _ in range(a.epochs):
            P, OS = upd(P, OS, fy, tr(o), tr(act), tr(lp), tr(A.astype(np.float32)), tr(ret.astype(np.float32)), tr(m), EO, EA, EM)
        gsteps += a.epochs
        since_birth += 1
        if a.replay and (a.pred or a.copy_replay):
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
        for w, l in zip(*np.where(dead)):
            lives.append(dict(lifespan=int(ages[w, l]), grad_steps=int(gsteps[w, l]),
                              params=jax.tree_util.tree_map(lambda x: x[w, l], P), centres=np.asarray(S.centres[w])))
            nav_of[w, l] = rng.integers(0, K)
        if dead.any():
            fresh = jax.tree_util.tree_map(lambda x: x[nav_of], NAV); dm = jnp.asarray(dead)
            P = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), P, fresh)
            OS = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), OS,
                                        jax.vmap(jax.vmap(opt.init))(fresh))
            since_birth[dead] = 0; gsteps[dead] = 0
            npos = (np.asarray(S.pos)[:, :NB].mean(1, keepdims=True).astype(int) + rng.integers(-3, 4, (W, NL, 2))) % cfg.L
            pos = np.asarray(S.pos).copy(); E = np.asarray(S.energy).copy(); age = np.asarray(S.age).copy(); al = np.asarray(S.alive).copy()
            sl = slice(NB, NB + NL)
            pos[:, sl][dead] = npos[dead]; E[:, sl][dead] = cfg.E0; age[:, sl][dead] = 0; al[:, sl][dead] = True
            S = S._replace(pos=jnp.asarray(pos), energy=jnp.asarray(E), age=jnp.asarray(age), alive=jnp.asarray(al))
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
    true = np.asarray(ev(Pz, C, keys, 0)); swap = np.asarray(ev(Pz, C, keys, 1))
    res = []
    for i, x in enumerate(lives):
        t, s_ = true[i], swap[i]; d = t - s_; se = d.std(ddof=1) / np.sqrt(len(d))
        drop = 1 - s_.sum() / max(t.sum(), 1e-9)
        res.append(dict(lifespan=x["lifespan"], grad_steps=x["grad_steps"], eaten_true=float(t.mean()), eaten_swap=float(s_.mean()),
                        drop=float(drop), passed=bool(drop >= 0.30 and d.mean() > 2 * se)))
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
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summ, lives=res), open(a.out, "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
