"""W2 fast-learner prior (Cam Nov 10 ~02:25: APPROVED to build; docs/W2_FAST_LEARNER_DESIGN.md). Laptop first, cap $3.

RL^2-style: a GRU agent over [obs (100), heard (11: one-hot token + flag + speaker offset), previous action one-hot (6),
previous reward]. Its WEIGHTS are meta-trained across many lives; within a life only its hidden state adapts.
World = the learnability gate's beacon world (4 stationary immortal speakers per world, 4 learners). Each life the
speakers use a FRESH RANDOM CODE: token = sigma(sector), sigma a random permutation of the 8 food sectors, so no
particular mapping can be learned across lives -- only "pick up whatever code is in use".
Leak test: codes are split into two families by sigma(0)'s parity; meta-training uses family A only; the gate is run on
family B (held out). Gates (design doc):
  G1  within-life reliance: a life adapts for --adapt steps, then the state is branched into a true-code continuation
      and a cross-direction-swap continuation (300 steps each, same keys); food eaten drop >= 30% (paired > 2 SE over
      16 branches) in >= 50% of lives, held-out code family B
  G2  no-code control: a meta-learner trained with uninformative tokens shows reliance ~0
  G3  leak: reliance on family B ~= family A (within SE)
  G4  foraging kept: median life without any messages >= 0.9x the W2 navigators'
Lives here have a fixed length (--life steps, juvenile provisioning throughout) so that every life is a full
adaptation episode; the hidden state resets at each life start.
usage: python w2/fastlearner.py --updates 300 [--null-code] [--eval-only CKPT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402

NB, NL, R_HEAR, H = 4, 4, 9, 128
PERMS_A = PERMS_B = None


class Agent(nn.Module):
    @nn.compact
    def __call__(self, h, x):
        z = nn.tanh(nn.Dense(H, name="enc")(x))
        h, y = nn.GRUCell(H, name="gru")(h, z)
        return h, nn.Dense(6, name="pi")(y), nn.Dense(1, name="v")(y)[..., 0]


def sector(off):
    ang = jnp.arctan2(off[..., 1].astype(jnp.float32), off[..., 0].astype(jnp.float32))
    return jnp.mod(jnp.round(ang / (2 * jnp.pi / 8)).astype(jnp.int32), 8)


def code_families(seed=0):
    """all 8! permutations is large; sample 4,000 random ones and split by parity of sigma(0)"""
    r = np.random.default_rng(seed)
    P = np.stack([r.permutation(8) for _ in range(4000)])
    return P[P[:, 0] % 2 == 0], P[P[:, 0] % 2 == 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300); ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--T", type=int, default=64); ap.add_argument("--life", type=int, default=640)
    ap.add_argument("--adapt", type=int, default=300); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--null-code", action="store_true", help="G2 control: tokens carry no information")
    ap.add_argument("--eval-lives", type=int, default=64); ap.add_argument("--branches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="w2_out/fastlearner")
    ap.add_argument("--eval-only", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = env.Cfg(L=96, N=NB + NL, P=6, patch_sigma=1.0, d0=0.008, reproduce=False, A_max=100_000)
    W, T = a.worlds, a.T; net = Agent()
    D = 100 + 11 + 6 + 1
    params = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, H)), jnp.zeros((1, D)))
    FA, FB = code_families()
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)); ost = opt.init(params)

    def new_world(key, perm):
        s = env.reset(cfg, key)
        c = jax.random.randint(jax.random.fold_in(key, 1), (2,), 0, cfg.L)
        bpos = (c[None] + jnp.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])) % cfg.L
        lpos = (c[None] + jax.random.randint(jax.random.fold_in(key, 2), (NL, 2), -3, 4)) % cfg.L
        return s._replace(pos=jnp.concatenate([bpos, lpos], 0)), perm

    def hear(s, perm, shift, key, null):
        sec = sector(env.oracle_offset(cfg, s))[:NB]
        tok = perm[jnp.mod(sec + shift, 8)]
        if null:
            tok = jax.random.randint(key, (NB,), 0, 8)
        lp, bp = s.pos[NB:], s.pos[:NB]
        d = (bp[None] - lp[:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
        cheb = jnp.abs(d).max(-1); j = jnp.argmin(cheb, 1); dj = d[jnp.arange(NL), j]
        ok = cheb[jnp.arange(NL), j] <= R_HEAR
        return jnp.concatenate([jax.nn.one_hot(tok[j], 8), jnp.ones((NL, 1)), dj / R_HEAR], 1) * ok[:, None]

    def step(p, s, perm, h, pa, pr, key, shift, null, no_msg=False):
        k1, k2, k3 = jax.random.split(key, 3)
        hd = hear(s, perm, shift, k3, null)
        if no_msg:
            hd = jnp.zeros_like(hd)
        x = jnp.concatenate([env.observe(cfg, s)[NB:], hd, jax.nn.one_hot(pa, 6), pr[:, None]], 1)
        h2, lg, v = net.apply(p, h, x)
        act = jax.random.categorical(k1, lg)
        lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act[:, None], 1)[:, 0]
        s2, r, info = env.step(cfg, s, jnp.concatenate([jnp.zeros(NB, jnp.int32), act]), k2)
        E = s2.energy.at[:NB].set(1.0)
        E = E.at[NB:].set(jnp.maximum(E[NB:], 0.02))                                   # lives have a fixed length (no death)
        s2 = s2._replace(energy=E, alive=s2.alive.at[:].set(True))
        rr = r[NB:] * 20.0
        return s2, h2, act, rr, (x, act, lp, v, rr, info["take"][NB:])

    def roll(p, s, perm, h, pa, pr, key):
        def body(c, k):
            s, h, pa, pr = c
            s2, h2, act, rr, rec = step(p, s, perm, h, pa, pr, k, 0, a.null_code)
            return (s2, h2, act, rr), rec + (h,)
        (s, h, pa, pr), R = jax.lax.scan(body, (s, h, pa, pr), jax.random.split(key, T))
        return s, h, pa, pr, R
    roll_w = jax.jit(jax.vmap(roll, in_axes=(None, 0, 0, 0, 0, 0, 0)))

    def loss(p, x, h0, act, lp0, adv, ret):
        """recompute the GRU over the chunk from its stored start state (truncated BPTT over T)"""
        def body(h, xt):
            h2, lg, v = net.apply(p, h, xt)
            return h2, (lg, v)
        _, (lg, v) = jax.lax.scan(body, h0, x)                                           # x (T, NL, D)
        ls = jax.nn.log_softmax(lg); lp = jnp.take_along_axis(ls, act[..., None], -1)[..., 0]
        ratio = jnp.exp(lp - lp0)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        return (pg + 0.5 * (v - ret) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)).mean()

    @jax.jit
    def upd(p, ost, x, h0, act, lp0, adv, ret):
        g = jax.grad(lambda q: jax.vmap(loss, in_axes=(None, 0, 0, 0, 0, 0, 0))(q, x, h0, act, lp0, adv, ret).mean())(p)
        u, ost = opt.update(g, ost, p); return optax.apply_updates(p, u), ost

    rng = np.random.default_rng(a.seed)

    def fresh(W_, fam):
        perms = jnp.asarray(fam[rng.integers(0, len(fam), W_)])
        ks = jax.random.split(jax.random.PRNGKey(int(rng.integers(1 << 30))), W_)
        S, Pm = jax.vmap(new_world)(ks, perms)
        return S, Pm, jnp.zeros((W_, NL, H)), jnp.zeros((W_, NL), jnp.int32), jnp.zeros((W_, NL))

    if a.eval_only:
        params = flax.serialization.from_bytes(params, open(a.eval_only, "rb").read())
    else:
        S, Pm, Hh, PA, PR = fresh(W, FA); t_life = 0; key = jax.random.PRNGKey(a.seed + 1); log = open(f"{a.out}/train.jsonl", "w")
        for it in range(a.updates):
            t0 = time.time()
            key, k = jax.random.split(key)
            S, Hh, PA, PR, (x, act, lp, v, r, take, h0s) = roll_w(params, S, Pm, Hh, PA, PR, jax.random.split(k, W))
            v_, r_ = np.asarray(v), np.asarray(r)                                      # (W, T, NL)
            adv = np.zeros_like(r_); last = 0.0
            for t in reversed(range(T)):
                nv = v_[:, t + 1] if t + 1 < T else v_[:, t]
                last = r_[:, t] + 0.99 * nv - v_[:, t] + 0.99 * 0.95 * last; adv[:, t] = last
            ret = adv + v_; A = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(2):
                params, ost = upd(params, ost, x, h0s[:, 0], act, lp, jnp.asarray(A, jnp.float32), jnp.asarray(ret, jnp.float32))
            t_life += T
            if t_life >= a.life:                                                        # every life ends: new code, new world
                S, Pm, Hh, PA, PR = fresh(W, FA); t_life = 0
            log.write(json.dumps(dict(update=it + 1, eaten_per_step=float(np.asarray(take).mean()), t=round(time.time() - t0, 2))) + "\n"); log.flush()
        open(f"{a.out}/params.msgpack", "wb").write(flax.serialization.to_bytes(params))

    # ---- gates
    def life_test(p, perm, key, shift_mode, adapt, no_msg=False):
        s, _ = new_world(key, perm)
        h = jnp.zeros((NL, H)); pa = jnp.zeros(NL, jnp.int32); pr = jnp.zeros(NL)
        def body(c, k):
            s, h, pa, pr = c
            s2, h2, act, rr, _ = step(p, s, perm, h, pa, pr, k, 0, a.null_code, no_msg)
            return (s2, h2, act, rr), None
        (s, h, pa, pr), _ = jax.lax.scan(body, (s, h, pa, pr), jax.random.split(jax.random.fold_in(key, 1), adapt))
        sh = jax.random.randint(jax.random.fold_in(key, 9), (), 1, 8) * shift_mode
        def body2(c, k):
            s, h, pa, pr, eaten = c
            s2, h2, act, rr, rec = step(p, s, perm, h, pa, pr, k, sh, a.null_code, no_msg)
            return (s2, h2, act, rr, eaten + rec[5].sum()), None
        (_, _, _, _, eaten), _ = jax.lax.scan(body2, (s, h, pa, pr, 0.0), jax.random.split(jax.random.fold_in(key, 2), 300))
        return eaten
    lt = jax.jit(jax.vmap(jax.vmap(life_test, in_axes=(None, None, 0, None, None)), in_axes=(None, 0, 0, None, None)), static_argnums=(3, 4))

    def gate(fam, n, tag):
        perms = jnp.asarray(fam[np.random.default_rng(99).integers(0, len(fam), n)])
        keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(5000 + i), a.branches))(jnp.arange(n))
        tru = np.asarray(lt(params, perms, keys, 0, a.adapt)); swp = np.asarray(lt(params, perms, keys, 1, a.adapt))
        d = tru - swp; drop = 1 - swp.sum(1) / np.maximum(tru.sum(1), 1e-9)
        se = d.std(1, ddof=1) / np.sqrt(d.shape[1]); ok = (drop >= 0.30) & (d.mean(1) > 2 * se)
        return dict(family=tag, lives=n, pass_frac=float(ok.mean()), median_drop=float(np.median(drop)),
                    eaten_true=float(tru.mean()), eaten_swap=float(swp.mean()))
    gB = gate(FB, a.eval_lives, "B (held out)"); gA = gate(FA, a.eval_lives, "A (training)")
    res = dict(null_code=a.null_code, G1=gB, leak_check_A=gA, G1_pass=gB["pass_frac"] >= 0.5,
               G3_leak_ok=abs(gA["median_drop"] - gB["median_drop"]) < 0.1)
    print(json.dumps(res)); json.dump(res, open(f"{a.out}/gates{'_null' if a.null_code else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
