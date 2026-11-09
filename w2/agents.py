"""W2 per-agent brains + PPO (Cam Nov 9 ~17:50: one brain per agent; W2 v1 = 8 agents, B = 8). Laptop first.

Each agent slot has its own MLP (the navigator architecture, w2/learn.Net) and its own Adam. W independent replicate
worlds run in parallel (vmapped); brains never cross worlds. Inside the rollout scan, a newborn's slot gets fresh weights
from a W2 navigator that no living agent in its world started from, and its optimizer state is reset after the rollout.
Each brain's PPO uses only its own rows from its current individual (rows of a slot's previous occupant are masked).
Reward: the env's (homeostatic; inclusive fitness when cfg.inclusive). Founders: 8 agents spawned clustered. If a world
goes extinct it is re-seeded with founders (recorded). No communication yet (the comm run adds the wire).
usage: python w2/agents.py --navs w2_out/nav/*.msgpack [--updates 100] [--inclusive]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402


def make_cfg(inclusive: bool, N=8):
    return env.Cfg(L=96, N=N, P=6, patch_sigma=1.0, d0=0.008, reproduce=True, rep_mode="setpoint", A_max=3000,
                   A_min=100, inclusive=inclusive)


def clustered_reset(cfg, key, radius=4):
    s = env.reset(cfg, key)
    k1, k2 = jax.random.split(jax.random.fold_in(key, 1))
    c = jax.random.randint(k1, (2,), 0, cfg.L)
    pos = (c[None] + jax.random.randint(k2, (cfg.N, 2), -radius, radius + 1)) % cfg.L
    return s._replace(pos=pos)


def run(navs, updates, inclusive, W=8, T=256, seed=0, gamma=0.99, lam=0.95, lr=3e-4, log=None):
    cfg = make_cfg(inclusive); N = cfg.N; net = Net()
    D = int(env.observe(cfg, env.reset(cfg, jax.random.PRNGKey(0))).shape[1])
    templ = net.init(jax.random.PRNGKey(0), jnp.zeros((1, D)))
    NAV = [flax.serialization.from_bytes(templ, open(p, "rb").read()) for p in navs]
    NAVS = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *NAV)                     # (K, ...)
    K = len(navs)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    # founders: world w's slot n starts from navigator n % K
    nav_of = jnp.tile(jnp.arange(N) % K, (W, 1))
    P = jax.tree_util.tree_map(lambda x: x[nav_of], NAVS)                           # (W, N, ...)
    OS = jax.vmap(jax.vmap(opt.init))(P)

    def pick_navs(nav_of, alive, births):
        """each newborn gets a navigator no living agent of its world started from (k-th newborn -> k-th free)"""
        in_use = jnp.zeros(K, bool).at[nav_of].max(alive & ~births)
        free_rank = jnp.cumsum(~in_use) - 1
        b_rank = jnp.cumsum(births) - 1
        choice = jnp.argmax((free_rank[None, :] == b_rank[:, None]) & ~in_use[None, :], 1)
        return jnp.where(births, choice, nav_of)

    def world_roll(Pw, s, nav_w, key):
        def body(c, k):
            Pw, s, nav_w, born_t, t = c
            k1, k2 = jax.random.split(k)
            o = env.observe(cfg, s)
            lg, v = jax.vmap(lambda p, x: net.apply(p, x[None]))(Pw, o)
            lg, v = lg[:, 0], v[:, 0]
            a = jax.random.categorical(k1, lg)
            lp = jnp.take_along_axis(jax.nn.log_softmax(lg), a[:, None], 1)[:, 0]
            alive0 = s.alive
            s2, r, info = env.step(cfg, s, a, k2)
            births = info["births"]
            nav2 = pick_navs(nav_w, s2.alive, births)
            Pw = jax.tree_util.tree_map(lambda P_, Nv: jnp.where(births.reshape((-1,) + (1,) * (P_.ndim - 1)), Nv[nav2], P_), Pw, NAVS)
            born_t = jnp.where(births, t + 1, born_t)
            # extinction: re-seed founders (recorded)
            ext = ~s2.alive.any()
            s2 = jax.tree_util.tree_map(lambda x, y: jnp.where(ext, y, x), s2, clustered_reset(cfg, k2))
            rec = dict(o=o, a=a, lp=lp, v=v, r=r * 20.0, alive=alive0, dead=info["dead"], births=births, ext=ext)
            return (Pw, s2, nav2, born_t, t + 1), rec
        (Pw, s, nav_w, born_t, _), R = jax.lax.scan(body, (Pw, s, nav_w, jnp.full(N, -1), 0), jax.random.split(key, T))
        return Pw, s, nav_w, born_t, R

    roll = jax.jit(jax.vmap(world_roll))

    def brain_loss(p, o, a, lp0, adv, ret, m):
        lg, v = net.apply(p, o)
        ls = jax.nn.log_softmax(lg); lp = jnp.take_along_axis(ls, a[:, None], 1)[:, 0]
        ratio = jnp.exp(lp - lp0)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        ent = -(jnp.exp(ls) * ls).sum(-1)
        den = m.sum() + 1e-8
        return ((pg + 0.5 * (v - ret) ** 2 - 0.01 * ent) * m).sum() / den

    @jax.jit
    def update(P, OS, o, a, lp, adv, ret, m):
        def one(p, os_, o, a, lp, adv, ret, m):
            g = jax.grad(brain_loss)(p, o, a, lp, adv, ret, m)
            u, os_ = opt.update(g, os_, p)
            return optax.apply_updates(p, u), os_
        return jax.vmap(jax.vmap(one))(P, OS, o, a, lp, adv, ret, m)

    states = jax.vmap(lambda k: clustered_reset(cfg, k))(jax.random.split(jax.random.PRNGKey(seed + 100), W))
    key = jax.random.PRNGKey(seed + 200)
    tot_births = 0; steps = 0; hist = []
    for it in range(updates):
        t0 = time.time()
        key, k = jax.random.split(key)
        P, states, nav_of, born_t, R = roll(P, states, nav_of, jax.random.split(k, W))
        # reset Adam for slots that were (re)born this rollout
        reborn = np.asarray(born_t) >= 0
        fresh = jax.vmap(jax.vmap(opt.init))(P)
        OS = jax.tree_util.tree_map(lambda x, y: jnp.where(jnp.asarray(reborn).reshape((W, N) + (1,) * (x.ndim - 2)), y, x), OS, fresh)
        R = jax.tree_util.tree_map(np.asarray, R)                                    # (W, T, N, ...)
        tvec = np.arange(T)[None, :, None]
        valid = R["alive"] & (tvec >= np.maximum(np.asarray(born_t), 0)[:, None, :])   # current individual's rows only
        v, r, dn = R["v"], R["r"], R["dead"].astype(np.float32)
        adv = np.zeros_like(r); last = 0.0
        for t in reversed(range(T)):
            nv = v[:, t + 1] if t + 1 < T else v[:, t]
            nonterm = 1.0 - dn[:, t]
            delta = r[:, t] + gamma * nv * nonterm - v[:, t]
            last = delta + gamma * lam * nonterm * last
            adv[:, t] = last
        ret = adv + v
        m = valid.astype(np.float32)
        mu = (adv * m).sum((1,), keepdims=True) / np.maximum(m.sum((1,), keepdims=True), 1)
        sd = np.sqrt(((adv - mu) ** 2 * m).sum((1,), keepdims=True) / np.maximum(m.sum((1,), keepdims=True), 1)) + 1e-8
        A = (adv - mu) / sd
        tr = lambda x: jnp.asarray(np.swapaxes(x, 1, 2))  # noqa: E731  (W, T, N, ...) -> (W, N, T, ...)
        P, OS = update(P, OS, tr(R["o"]), tr(R["a"]), tr(R["lp"]), tr(A.astype(np.float32)), tr(ret.astype(np.float32)), tr(m))
        nb = int(R["births"].sum()); tot_births += nb; steps += W * T
        rec = dict(update=it + 1, births_per_1000_steps_per_world=1000 * nb / (W * T), alive_mean=float(R["alive"].mean() * N),
                   deaths=int(R["dead"].sum()), extinctions=int(R["ext"].sum()), t_update=round(time.time() - t0, 2))
        hist.append(rec)
        if log:
            log.write(json.dumps(rec) + "\n"); log.flush()
    return hist, tot_births, steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", nargs="+", required=True); ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--inclusive", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=8); ap.add_argument("--out", default="w2_out/agents.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    h, nb, st = run(sorted(sum((glob.glob(x) for x in a.navs), [])), a.updates, a.inclusive, W=a.worlds, seed=a.seed,
                    log=open(a.out, "w"))
    last = h[-max(1, len(h) // 5):]
    print(f"births per 1,000 steps per world: overall {1000 * nb / st:.2f}, last 20% {np.mean([x['births_per_1000_steps_per_world'] for x in last]):.2f}"
          f" | alive mean {np.mean([x['alive_mean'] for x in last]):.2f} / 8 | extinctions {sum(x['extinctions'] for x in h)}"
          f" | t_update {np.mean([x['t_update'] for x in h[2:]]):.2f} s")
