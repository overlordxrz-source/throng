"""W2 prototype gates P1 / P2 with learning (docs/OWN_WORLD_SPEC.md section 9). Laptop, $0.

A small shared-brain PPO (MLP 2 x 128, actions stay / 4 moves / eat) on the tuned world, homeostatic reward only.
Training: dead agents respawn (random cell, E0) so the batch stays full. Evaluation: the frozen policy, no respawn,
the same lifespan test as w2/scripted.py (16 worlds x 64 agents x 3,000 steps).
  P1: learned blind agent's lifespan > random's (by > 2 SE).
  P2: learned ORACLE agent (sees the clipped offset to the nearest patch) lives >= 2x a learned blind one.
usage: python w2/learn.py [--updates 300]
"""
from __future__ import annotations

import argparse
import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env, scripted  # noqa: E402

CFG = env.Cfg(L=96, N=64, P=6, patch_sigma=1.0, d0=0.008, reproduce=False, A_max=100_000)


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(128)(x)); h = nn.tanh(nn.Dense(128)(h))
        return nn.Dense(6)(h), nn.Dense(1)(h)[..., 0]


def train(oracle: bool, updates: int, seed: int, W=8, T=128, gamma=0.99, lam=0.95, lr=3e-4, juvenile=0):
    cfg = CFG; net = Net()
    D = int(env.observe(cfg, env.reset(cfg, jax.random.PRNGKey(0)), oracle).shape[1])
    params = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, D)))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)); ost = opt.init(params)

    def respawn(s, k):
        dead = ~s.alive
        pos = jnp.where(dead[:, None], jax.random.randint(k, (cfg.N, 2), 0, cfg.L), s.pos)
        return s._replace(pos=pos, energy=jnp.where(dead, cfg.E0, s.energy), age=jnp.where(dead, 0, s.age),
                          alive=jnp.ones(cfg.N, bool))

    def roll(params, s, k):
        def body(s, kk):
            k1, k2, k3 = jax.random.split(kk, 3)
            s = respawn(s, k3)
            o = env.observe(cfg, s, oracle)
            lg, v = net.apply(params, o)
            a = jax.random.categorical(k1, lg)
            lp = jnp.take_along_axis(jax.nn.log_softmax(lg), a[:, None], 1)[:, 0]
            s2, r, info = env.step(cfg, s, a, k2)
            if juvenile:   # provisioning: energy >= E0 for the first `juvenile` steps of life (nobody pays here)
                s2 = s2._replace(energy=jnp.where((s2.age < juvenile) & s2.alive, jnp.maximum(s2.energy, cfg.E0), s2.energy))
            return s2, (o, a, lp, v, r, info["dead"].astype(jnp.float32))
        return jax.lax.scan(body, s, jax.random.split(k, T))

    roll_w = jax.jit(jax.vmap(roll, in_axes=(None, 0, 0)))

    def loss(p, o, a, lp0, adv, ret):
        lg, v = net.apply(p, o)
        ls = jax.nn.log_softmax(lg)
        lp = jnp.take_along_axis(ls, a[:, None], 1)[:, 0]
        ratio = jnp.exp(lp - lp0)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv).mean()
        return pg + 0.5 * ((v - ret) ** 2).mean() - 0.01 * (-(jnp.exp(ls) * ls).sum(-1)).mean()

    @jax.jit
    def upd(p, ost, o, a, lp0, adv, ret):
        g = jax.grad(loss)(p, o, a, lp0, adv, ret)
        u, ost = opt.update(g, ost, p)
        return optax.apply_updates(p, u), ost

    states = jax.vmap(lambda k: env.reset(cfg, k))(jax.random.split(jax.random.PRNGKey(seed + 100), W))
    key = jax.random.PRNGKey(seed + 200)
    for it in range(updates):
        key, k = jax.random.split(key)
        states, (o, a, lp, v, r, dn) = roll_w(params, states, jax.random.split(k, W))
        o, a, lp, v, r, dn = (np.asarray(x) for x in (o, a, lp, v, r, dn))      # (W, T, N, ...)
        r = r * 20.0                                                               # drive deltas are tiny
        adv = np.zeros_like(r); last = 0.0
        for t in reversed(range(T)):
            nv = v[:, t + 1] if t + 1 < T else v[:, t]
            nonterm = 1.0 - dn[:, t]
            delta = r[:, t] + gamma * nv * nonterm - v[:, t]
            last = delta + gamma * lam * nonterm * last
            adv[:, t] = last
        ret = adv + v
        f = lambda x: jnp.asarray(x.reshape((-1,) + x.shape[3:]))  # noqa: E731
        A = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.random.default_rng(it).permutation(W * T * cfg.N)
        O, Aa, LP, AD, RT = f(o), f(a), f(lp), f(A), f(ret)
        for j in range(0, len(idx), 4096):
            b = idx[j:j + 4096]
            params, ost = upd(params, ost, O[b], Aa[b], LP[b], AD[b], RT[b])
    return params, net


def eval_lifespan(params, net, oracle, worlds=16, T=3000, seed=3, juvenile=0, random_policy=False, stat="mean"):
    cfg = CFG

    def one(k):
        s = env.reset(cfg, k)
        def body(c, kk):
            s, life = c
            lg, _ = net.apply(params, env.observe(cfg, s, oracle))
            a = jax.random.randint(kk, (cfg.N,), 0, 6) if random_policy else jax.random.categorical(kk, lg)
            s2, _, _ = env.step(cfg, s, a, kk)
            if juvenile:
                s2 = s2._replace(energy=jnp.where((s2.age < juvenile) & s2.alive, jnp.maximum(s2.energy, cfg.E0), s2.energy))
            return (s2, life + s2.alive), None
        (s, life), _ = jax.lax.scan(body, (s, jnp.zeros(cfg.N, jnp.int32)), jax.random.split(jax.random.fold_in(k, 7), T))
        return life
    lives = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(seed), worlds))
    if stat == "median":
        L = np.asarray(lives).ravel(); bs = np.random.default_rng(0)
        meds = [np.median(bs.choice(L, L.size)) for _ in range(200)]
        return float(np.median(L)), float(np.std(meds))           # median and its bootstrap SE
    return float(lives.mean()), float(lives.std() / np.sqrt(lives.size))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--updates", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = scripted.lifespan(CFG, "random")
    pb, nb = train(False, a.updates, a.seed); blind = eval_lifespan(pb, nb, False)
    po, no = train(True, a.updates, a.seed); orac = eval_lifespan(po, no, True)
    print(f"random  : {rnd[0]:.1f} (SE {rnd[1]:.1f})")
    print(f"learned blind : {blind[0]:.1f} (SE {blind[1]:.1f})  -> P1 {'PASS' if blind[0] - rnd[0] > 2 * np.hypot(blind[1], rnd[1]) else 'FAIL'}")
    print(f"learned oracle: {orac[0]:.1f} (SE {orac[1]:.1f})  -> P2 oracle / blind {orac[0] / blind[0]:.2f}x "
          f"{'PASS' if orac[0] >= 2 * blind[0] else 'FAIL'}")
