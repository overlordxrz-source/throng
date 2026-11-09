"""W2 learnability gate (Cam Nov 9 ~20:15): can a newborn learn to LISTEN within one lifetime?

Per world: 4 frozen speakers ("beacons": stationary, immortal, at the founders' cluster) broadcasting a fixed scripted
code (the 8-sector direction from the beacon to its nearest food patch), and 4 learners (fresh W2 navigators, hearing
input rows initialised at zero, own per-agent PPO, natural lives). A learner hears the nearest beacon within Chebyshev
radius 9: one-hot(sector) + a heard flag + the beacon's offset / 9. When a learner dies, its end-of-life weights are
frozen and tested: food eaten over 300 steps from a start near the beacons, true code vs a cross-direction swap (every
heard sector shifted by a random 1..7), 4 paired episodes. A life PASSES iff drop >= 30% and paired drop > 2 SE.
GATE PASS iff >= 50% of lives pass.
Variants (Cam's fallbacks, re-gate after each): --juvenile J (energy provisioned to >= E0 for the first J steps of a
life; the young stays within hearing range: J steps of no movement beyond radius 9 from the beacons) and --long-life
(drain halved; P2 must still hold). --random-code = apparatus control (uninformative messages: reliance must be ~0).
usage: python w2/learnability.py --navs "w2_out/nav/*.msgpack" [--updates 150] [--juvenile 0] [--long-life] [--random-code]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402

NB, NL, R_HEAR, HD = 4, 4, 9, 11


def sector(off):
    ang = jnp.arctan2(off[..., 1].astype(jnp.float32), off[..., 0].astype(jnp.float32))
    return jnp.mod(jnp.round(ang / (2 * jnp.pi / 8)).astype(jnp.int32), 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", required=True); ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--worlds", type=int, default=16); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--juvenile", type=int, default=0)
    ap.add_argument("--long-life", action="store_true"); ap.add_argument("--random-code", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="w2_out/learnability.json")
    a = ap.parse_args()
    cfg = env.Cfg(L=96, N=NB + NL, P=6, patch_sigma=1.0, d0=0.004 if a.long_life else 0.008, reproduce=False,
                  A_max=3000)
    W, T = a.worlds, a.T; net = Net()
    navs = sorted(glob.glob(a.navs)); K = len(navs)
    D0 = int(env.observe(cfg, env.reset(cfg, jax.random.PRNGKey(0))).shape[1])
    templ0 = net.init(jax.random.PRNGKey(0), jnp.zeros((1, D0)))

    def expand(p):          # hearing rows appended as zeros: the newborn starts exactly as its navigator
        d = jax.tree_util.tree_map(lambda x: x, p)
        k = d["params"]["Dense_0"]["kernel"]
        d["params"]["Dense_0"]["kernel"] = jnp.concatenate([k, jnp.zeros((HD, k.shape[1]))], 0)
        return d
    NAV = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[expand(flax.serialization.from_bytes(templ0, open(f, "rb").read())) for f in navs])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))

    def beacons_reset(s, key):
        c = jax.random.randint(key, (2,), 0, cfg.L)
        bpos = (c[None] + jnp.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])) % cfg.L
        lpos = (c[None] + jax.random.randint(jax.random.fold_in(key, 1), (NL, 2), -3, 4)) % cfg.L
        return s._replace(pos=jnp.concatenate([bpos, lpos], 0)), c

    def hear(s, shift):
        """(NL, HD): nearest beacon within radius; its sector (+ shift, mod 8) one-hot, flag, offset / R"""
        sec = sector(env.oracle_offset(cfg, s))[:NB]
        sec = jnp.mod(sec + shift, 8)
        lp, bp = s.pos[NB:], s.pos[:NB]
        d = (bp[None] - lp[:, None] + cfg.L // 2) % cfg.L - cfg.L // 2                   # (NL, NB, 2)
        cheb = jnp.abs(d).max(-1); j = jnp.argmin(cheb, 1); dj = d[jnp.arange(NL), j]
        ok = cheb[jnp.arange(NL), j] <= R_HEAR
        f = jnp.concatenate([jax.nn.one_hot(sec[j], 8), jnp.ones((NL, 1)), dj / R_HEAR], 1)
        return f * ok[:, None]

    def obs_l(s, shift):
        return jnp.concatenate([env.observe(cfg, s)[NB:], hear(s, shift)], 1)

    def step_all(P, s, key, shift, young):
        k1, k2 = jax.random.split(key)
        o = obs_l(s, shift)
        lg, v = jax.vmap(lambda p, x: net.apply(p, x[None]))(P, o); lg, v = lg[:, 0], v[:, 0]
        act_l = jax.random.categorical(k1, lg)
        if a.juvenile:                                    # the young stays within hearing range of the beacons
            far = hear(s, 0)[:, 8] < 0.5
            act_l = jnp.where(young & far, 0, act_l)
        lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act_l[:, None], 1)[:, 0]
        act = jnp.concatenate([jnp.zeros(NB, jnp.int32), act_l])
        s2, r, info = env.step(cfg, s, act, k2)
        E = s2.energy.at[:NB].set(1.0)                    # beacons are immortal and stationary
        if a.juvenile:                                    # parental provisioning (the parent's cost is not modelled here)
            E = E.at[NB:].set(jnp.where(young & s2.alive[NB:], jnp.maximum(E[NB:], cfg.E0), E[NB:]))
        s2 = s2._replace(energy=E, alive=s2.alive.at[:NB].set(True))
        return s2, (o, act_l, lp, v, r[NB:] * 20.0, info["dead"][NB:], info["take"][NB:])

    def roll(P, s, key, codekey):
        def body(s, k):
            k1, k2 = jax.random.split(k)
            shift = jax.random.randint(k2, (), 0, 8) if a.random_code else 0
            young = s.age[NB:] < a.juvenile
            s2, rec = step_all(P, s, k1, shift, young)
            return s2, rec + (s.alive[NB:],)
        return jax.lax.scan(body, s, jax.random.split(key, T))
    roll_w = jax.jit(jax.vmap(roll, in_axes=(0, 0, 0, 0)))

    def loss(p, o, a_, lp0, adv, ret, m):
        lg, v = net.apply(p, o); ls = jax.nn.log_softmax(lg)
        lp = jnp.take_along_axis(ls, a_[:, None], 1)[:, 0]; ratio = jnp.exp(lp - lp0)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        return ((pg + 0.5 * (v - ret) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)) * m).sum() / (m.sum() + 1e-8)

    @jax.jit
    def upd(P, OS, o, a_, lp, adv, ret, m):
        def one(p, os_, *b):
            g = jax.grad(loss)(p, *b); u, os_ = opt.update(g, os_, p); return optax.apply_updates(p, u), os_
        return jax.vmap(jax.vmap(one))(P, OS, o, a_, lp, adv, ret, m)

    def eval_life(p, centres, key, shift_mode):
        """food eaten in 300 steps from a start near the beacons, this learner alone (other learner slots dead)"""
        s = env.reset(cfg, key)
        s = s._replace(centres=centres, cap=env.capacity(cfg, centres), food=env.capacity(cfg, centres))
        s, _ = beacons_reset(s, jax.random.fold_in(key, 3))
        s = s._replace(alive=jnp.array([True] * NB + [True] + [False] * (NL - 1)))
        sh = jax.random.randint(jax.random.fold_in(key, 9), (), 1, 8) * shift_mode
        P1 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NL,) + x.shape), p)
        def body(c, k):
            s, eaten = c
            s2, rec = step_all(P1, s, k, sh, jnp.zeros(NL, bool))
            return (s2, eaten + rec[6][0]), None
        (s, eaten), _ = jax.lax.scan(body, (s, 0.0), jax.random.split(jax.random.fold_in(key, 5), 300))
        return eaten
    ev = jax.jit(jax.vmap(jax.vmap(eval_life, in_axes=(None, None, 0, None)), in_axes=(0, 0, 0, None)))

    rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed)
    states = []; nav_of = np.zeros((W, NL), int); P_list = []
    for w in range(W):
        s0 = env.reset(cfg, jax.random.PRNGKey(1000 + w)); s0, _ = beacons_reset(s0, jax.random.PRNGKey(2000 + w)); states.append(s0)
    S = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *states)
    nav_of[:] = rng.integers(0, K, (W, NL))
    P = jax.tree_util.tree_map(lambda x: x[nav_of], NAV); OS = jax.vmap(jax.vmap(opt.init))(P)
    lives = []
    for it in range(a.updates):
        key, k = jax.random.split(key)
        S, (o, act, lp, v, r, dn, take, alive) = roll_w(P, S, jax.random.split(k, W), jax.random.split(k, W))
        o, act, lp, v, r, dn, alive = (np.asarray(x) for x in (o, act, lp, v, r, dn, alive))   # (W, T, NL, ...)
        adv = np.zeros_like(r); last = 0.0
        for t in reversed(range(T)):
            nv = v[:, t + 1] if t + 1 < T else v[:, t]; nt = 1.0 - dn[:, t]
            last = r[:, t] + 0.99 * nv * nt - v[:, t] + 0.99 * 0.95 * nt * last; adv[:, t] = last
        ret = adv + v; m = alive.astype(np.float32)
        A = (adv - (adv * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)) / (adv.std() + 1e-8)
        tr = lambda x: jnp.asarray(np.swapaxes(x, 1, 2))  # noqa: E731
        for _ in range(a.epochs):
            P, OS = upd(P, OS, tr(o), tr(act), tr(lp), tr(A.astype(np.float32)), tr(ret.astype(np.float32)), tr(m))
        # deaths: freeze the end-of-life weights for the test, then a fresh newborn takes the slot
        dead = ~np.asarray(S.alive)[:, NB:]
        ages = np.asarray(S.age)[:, NB:]
        for w, l in zip(*np.where(dead)):
            lives.append(dict(w=int(w), l=int(l), lifespan=int(ages[w, l]), nav=int(nav_of[w, l]),
                              params=jax.tree_util.tree_map(lambda x: x[w, l], P), centres=np.asarray(S.centres[w])))
            nav_of[w, l] = rng.integers(0, K)
        if dead.any():
            fresh = jax.tree_util.tree_map(lambda x: x[nav_of], NAV)
            dm = jnp.asarray(dead)
            P = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), P, fresh)
            OS = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), OS,
                                        jax.vmap(jax.vmap(opt.init))(fresh))
            key, kk = jax.random.split(key)
            npos = (np.asarray(S.pos)[:, :NB].mean(1, keepdims=True).astype(int) + rng.integers(-3, 4, (W, NL, 2))) % cfg.L
            pos = np.asarray(S.pos).copy(); E = np.asarray(S.energy).copy(); age = np.asarray(S.age).copy(); al = np.asarray(S.alive).copy()
            pos[:, NB:][dead] = npos[dead]; E[:, NB:][dead] = cfg.E0; age[:, NB:][dead] = 0; al[:, NB:][dead] = True
            S = S._replace(pos=jnp.asarray(pos), energy=jnp.asarray(E), age=jnp.asarray(age), alive=jnp.asarray(al))
    # the test
    if not lives:
        print("no completed lives"); return
    Pz = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[x["params"] for x in lives])
    C = jnp.asarray(np.stack([x["centres"] for x in lives]))
    keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(7000 + i), 4))(jnp.arange(len(lives)))
    true = np.asarray(ev(Pz, C, keys, 0)); swap = np.asarray(ev(Pz, C, keys, 1))
    res = []
    for i, x in enumerate(lives):
        t, s_ = true[i], swap[i]; d = t - s_; se = d.std(ddof=1) / np.sqrt(len(d))
        drop = 1 - s_.sum() / max(t.sum(), 1e-9); ok = bool(drop >= 0.30 and d.mean() > 2 * se)
        res.append(dict(lifespan=x["lifespan"], eaten_true=float(t.mean()), eaten_swap=float(s_.mean()), drop=float(drop), passed=ok))
    frac = np.mean([r_["passed"] for r_ in res])
    summ = dict(variant=dict(juvenile=a.juvenile, long_life=a.long_life, random_code=a.random_code, updates=a.updates, T=T,
                             epochs=a.epochs), lives=len(res), pass_frac=float(frac),
                median_drop=float(np.median([r_["drop"] for r_ in res])), median_lifespan=float(np.median([r_["lifespan"] for r_ in res])),
                gate="PASS" if frac >= 0.5 else "FAIL")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summ, lives=res), open(a.out, "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
