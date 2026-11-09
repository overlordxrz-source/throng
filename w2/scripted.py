"""W2 prototype gates before learning (docs/OWN_WORLD_SPEC.md section 9): scripted P2 premise + P3 throughput.

Policies (jitted, per agent):
  random       uniform over stay / 4 moves / eat
  blind        eat if standing on food >= 0.2; else step toward the richest visible cell in the 7 x 7 window; else
               keep a persistent random heading (a blind forager)
  oracle       as blind, but with nothing visible it steps toward the ORACLE offset (nearest patch centre, clipped)
Lifespan: agents start at E0 with no reproduction; mean steps alive over T steps (capped at T), 8 worlds x 64 agents.
usage: python w2/scripted.py
"""
from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402


def toward(off):
    """an action reducing the larger axis of an offset (dx, dy); 0 if already there"""
    dx, dy = off[..., 0], off[..., 1]
    return jnp.where((dx == 0) & (dy == 0), 0,
                     jnp.where(jnp.abs(dx) >= jnp.abs(dy), jnp.where(dx < 0, 1, 2), jnp.where(dy > 0, 3, 4)))


def policy(cfg, s, key, kind, heading):
    r = cfg.r
    if kind == "random":
        return jax.random.randint(key, (cfg.N,), 0, 6), heading
    here = s.food[s.pos[:, 0], s.pos[:, 1]]
    w = env.window(cfg, s.food, s.pos, r).reshape(cfg.N, -1)
    best = jnp.argmax(w, 1); seen = w.max(1) > 0.2
    o = jnp.arange(-r, r + 1)
    off_seen = jnp.stack([o[best // (2 * r + 1)], o[best % (2 * r + 1)]], -1)
    if kind == "oracle":
        fallback = toward(env.oracle_offset(cfg, s))
    else:
        turn = jax.random.uniform(key, (cfg.N,)) < 0.05
        heading = jnp.where(turn, jax.random.randint(jax.random.fold_in(key, 1), (cfg.N,), 1, 5), heading)
        fallback = heading
    act = jnp.where(here >= 0.2, env.EAT, jnp.where(seen, toward(off_seen), fallback))
    return act, heading


def lifespan(cfg, kind, worlds=16, T=3000, seed=3):
    def one(k):
        s = env.reset(cfg, k)
        def body(c, kk):
            s, h, life = c
            act, h = policy(cfg, s, kk, kind, h)
            s2, _, _ = env.step(cfg, s, act, kk)
            return (s2, h, life + s2.alive), None
        (s, _, life), _ = jax.lax.scan(body, (s, jnp.ones(cfg.N, jnp.int32), jnp.zeros(cfg.N, jnp.int32)),
                                       jax.random.split(jax.random.fold_in(k, 7), T))
        return life
    lives = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(seed), worlds))
    return float(lives.mean()), float(lives.std() / np.sqrt(lives.size))


def throughput(cfg, worlds=64, T=200):
    def one(k):
        s = env.reset(cfg, k)
        def body(s, kk):
            act = jax.random.randint(kk, (cfg.N,), 0, 6)
            s2, r, _ = env.step(cfg, s, act, kk)
            return s2, env.observe(cfg, s2, oracle=True).sum()
        return jax.lax.scan(body, s, jax.random.split(k, T))[1]
    f = jax.jit(jax.vmap(one))
    ks = jax.random.split(jax.random.PRNGKey(1), worlds)
    jax.block_until_ready(f(ks)); t = time.time(); jax.block_until_ready(f(ks)); dt = time.time() - t
    return worlds * T * cfg.N / dt


if __name__ == "__main__":
    cfg = env.Cfg(L=96, N=64, P=6, patch_sigma=1.0, d0=0.008, reproduce=False, A_max=100_000)   # tuned Nov 9 (P2 premise)
    res = {k: lifespan(cfg, k) for k in ("random", "blind", "oracle")}
    for k, (m, se) in res.items():
        print(f"{k:7s}: mean lifespan {m:7.1f} steps (SE {se:.1f}) of 3,000")
    print(f"P2 premise (oracle / blind >= 2): {res['oracle'][0] / res['blind'][0]:.2f}x -> "
          f"{'OK' if res['oracle'][0] >= 2 * res['blind'][0] else 'RETUNE THE WORLD'}")
    print(f"blind / random: {res['blind'][0] / res['random'][0]:.2f}x")
    print(f"P3 throughput (env + obs, laptop CPU, 64 worlds x 64 agents): {throughput(cfg):,.0f} agent-steps/s")
