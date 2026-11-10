"""W2 prototype world core (docs/OWN_WORLD_SPEC.md, Cam Nov 9 ~07:50). Pure JAX, fixed shapes, jit / vmap / scan.

State (one world): food (L, L); pos (N, 2) int; energy, age (N,); alive (N,) bool. Births and deaths are mask flips.
Senses are absolute, local and limited range: a (2r+1)^2 egocentric window (food, agents), own energy and age, and
(ORACLE only) the offset to the nearest patch centre, clipped, in cells. Nothing is divided by L.
Actions: 0 stay, 1..4 move (-x, +x, +y, -y), 5 eat. Reward: homeostatic drive reduction (Keramati & Gutkin 2014).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], jnp.int32)
EAT = 5


class Cfg(NamedTuple):
    L: int = 64
    N: int = 64
    P: int = 6              # food patches
    patch_sigma: float = 2.0
    regrow: float = 0.02    # food += regrow * (cap - food) per step
    r: int = 3              # view radius
    r_oracle: int = 16      # ORACLE offset clip (cells)
    d0: float = 0.005       # basal drain
    d_move: float = 0.003
    bite: float = 0.2
    eff: float = 0.5        # energy per unit food eaten
    E0: float = 0.6
    E_rep: float = 0.95
    A_min: int = 100
    A_max: int = 5000
    c_rep: float = 0.3          # the parent pays this (Cam Nov 9 ~17:50: 0.3)
    reproduce: bool = False
    rep_mode: str = "setpoint"   # Cam Nov 9 ~17:50: birth after E >= E_set held rep_hold steps (else: E >= E_rep)
    rep_hold: int = 50
    E_child: float = 0.5
    E_set: float = 0.8
    drive_n: float = 2.0
    inclusive: bool = False      # inclusive-fitness reward: + mean over living kin of relatedness x their drive reduction
    juvenile: int = 0            # Cam Nov 10: for the first J steps of life a newborn is topped up to E0 by its living parent
    relocate: float = 0.0        # per-step probability that each food patch jumps to a random location


class State(NamedTuple):
    food: jnp.ndarray
    cap: jnp.ndarray
    centres: jnp.ndarray
    pos: jnp.ndarray
    energy: jnp.ndarray
    age: jnp.ndarray
    alive: jnp.ndarray
    fed: jnp.ndarray = None       # consecutive steps at E >= E_set
    rel: jnp.ndarray = None       # (N, N) relatedness, 0 on the diagonal and for dead slots
    parent: jnp.ndarray = None    # parent slot (-1 for founders)


def capacity(cfg: Cfg, centres):
    g = jnp.arange(cfg.L)
    d = jnp.abs(g[None, :, None] - centres[:, 0][:, None, None]); dx = jnp.minimum(d, cfg.L - d)
    d = jnp.abs(g[None, None, :] - centres[:, 1][:, None, None]); dy = jnp.minimum(d, cfg.L - d)
    return jnp.clip(jnp.exp(-(dx ** 2 + dy ** 2) / (2 * cfg.patch_sigma ** 2)).sum(0), 0.0, 1.0)


def reset(cfg: Cfg, key) -> State:
    k1, k2 = jax.random.split(key)
    centres = jax.random.randint(k1, (cfg.P, 2), 0, cfg.L)
    cap = capacity(cfg, centres)
    pos = jax.random.randint(k2, (cfg.N, 2), 0, cfg.L)
    rel = 0.5 * (1.0 - jnp.eye(cfg.N))                                    # founders' cluster: 0.5 to each other
    return State(cap, cap, centres, pos, jnp.full(cfg.N, cfg.E0), jnp.zeros(cfg.N, jnp.int32), jnp.ones(cfg.N, bool),
                 jnp.zeros(cfg.N, jnp.int32), rel, jnp.full(cfg.N, -1, jnp.int32))


def window(cfg: Cfg, grid, pos, r):
    o = jnp.arange(-r, r + 1)
    ix = (pos[:, 0][:, None, None] + o[None, :, None]) % cfg.L
    iy = (pos[:, 1][:, None, None] + o[None, None, :]) % cfg.L
    return grid[ix, iy]                                                     # (N, 2r+1, 2r+1)


def occupancy(cfg: Cfg, pos, alive):
    return jnp.zeros((cfg.L, cfg.L)).at[pos[:, 0], pos[:, 1]].add(alive.astype(jnp.float32))


def oracle_offset(cfg: Cfg, s: State):
    d = (s.centres[None, :, :] - s.pos[:, None, :] + cfg.L // 2) % cfg.L - cfg.L // 2   # (N, P, 2) torus offsets
    i = jnp.argmin(jnp.abs(d).sum(-1), 1)
    off = d[jnp.arange(cfg.N), i]
    return jnp.clip(off, -cfg.r_oracle, cfg.r_oracle)


def observe(cfg: Cfg, s: State, oracle: bool = False):
    f = window(cfg, s.food, s.pos, cfg.r).reshape(cfg.N, -1)
    a = window(cfg, occupancy(cfg, s.pos, s.alive), s.pos, cfg.r).reshape(cfg.N, -1)
    intero = jnp.stack([s.energy, s.age / cfg.A_max], 1)
    parts = [f, a, intero]
    if oracle:
        parts.append(oracle_offset(cfg, s).astype(jnp.float32) / cfg.r_oracle)   # clipped offset, a fixed unit
    return jnp.concatenate(parts, 1)


def drive(cfg: Cfg, E):
    return jnp.abs(cfg.E_set - E) ** cfg.drive_n


def step(cfg: Cfg, s: State, act, key):
    """returns (state', reward (N,), info)"""
    alive = s.alive
    mv = MOVES[act] * alive[:, None]
    pos = (s.pos + mv) % cfg.L
    moved = jnp.any(mv != 0, 1)
    eat = (act == EAT) & alive
    here = s.food[pos[:, 0], pos[:, 1]]
    # several eaters on one cell share it: each takes bite, capped by the cell's food split evenly
    n_here = jnp.zeros((cfg.L, cfg.L)).at[pos[:, 0], pos[:, 1]].add(eat.astype(jnp.float32))[pos[:, 0], pos[:, 1]]
    take = jnp.where(eat, jnp.minimum(cfg.bite, here / jnp.maximum(n_here, 1.0)), 0.0)
    food = s.food.at[pos[:, 0], pos[:, 1]].add(-take)
    cap, centres = s.cap, s.centres
    if cfg.relocate > 0:           # food that relocates: fresh information keeps its value
        kr1, kr2 = jax.random.split(jax.random.fold_in(key, 77))
        jump = jax.random.uniform(kr1, (cfg.P,)) < cfg.relocate
        centres = jnp.where(jump[:, None], jax.random.randint(kr2, (cfg.P, 2), 0, cfg.L), centres)
        cap = capacity(cfg, centres)
        food = jnp.minimum(food, cap)
    food = food + cfg.regrow * (cap - food)
    E_prev = s.energy
    E = jnp.clip(E_prev + cfg.eff * take - (cfg.d0 + cfg.d_move * moved) * alive, 0.0, 1.0)
    age = s.age + alive.astype(jnp.int32)
    dead = alive & ((E <= 0.0) | (age >= cfg.A_max))
    alive2 = alive & ~dead
    r = jnp.where(alive, drive(cfg, E_prev) - drive(cfg, E), 0.0)
    rel = s.rel if s.rel is not None else jnp.zeros((cfg.N, cfg.N))
    if cfg.inclusive:                     # own drive reduction + mean over living kin of relatedness x theirs
        kin = (rel > 0) & alive[None, :]
        r = r + jnp.where(alive, (rel * r[None, :] * kin).sum(1) / jnp.maximum(kin.sum(1), 1), 0.0)
    fed = jnp.where(alive2 & (E >= cfg.E_set), (s.fed if s.fed is not None else 0) + 1, 0)
    par = s.parent if s.parent is not None else jnp.full(cfg.N, -1, jnp.int32)
    rel = jnp.where(alive2[:, None] & alive2[None, :], rel, 0.0)            # the dead leave the kin table
    births = jnp.zeros(cfg.N, bool)
    if cfg.reproduce:
        if cfg.rep_mode == "setpoint":
            parent = alive2 & (fed >= cfg.rep_hold) & (age >= cfg.A_min)
        else:
            parent = alive2 & (E >= cfg.E_rep) & (age >= cfg.A_min)
        free = ~alive2
        pr = jnp.cumsum(parent) * parent                                     # k-th parent -> k-th free slot
        fr = jnp.cumsum(free) * free
        nfree = free.sum()
        ok_parent = parent & (pr <= nfree)
        # child slot index for each parent rank k: position of the k-th free slot
        slot_of_rank = jnp.zeros(cfg.N + 1, jnp.int32).at[fr].set(jnp.arange(cfg.N))
        child = jnp.where(ok_parent, slot_of_rank[pr], cfg.N)                # N = no child (dropped by mode='drop')
        off = MOVES[1 + jax.random.randint(key, (cfg.N,), 0, 4)]
        cpos = (pos + off) % cfg.L
        E = jnp.where(ok_parent, E - cfg.c_rep, E)
        fed = jnp.where(ok_parent, 0, fed)
        pos = pos.at[child].set(cpos, mode="drop")
        e_child = cfg.E_child if cfg.rep_mode == "setpoint" else cfg.c_rep * 0.9
        E = E.at[child].set(e_child, mode="drop")
        age = age.at[child].set(0, mode="drop")
        births = jnp.zeros(cfg.N, bool).at[child].set(True, mode="drop")
        cp = jnp.full(cfg.N, -1, jnp.int32).at[child].set(jnp.arange(cfg.N), mode="drop")   # newborn slot -> its parent
        par = jnp.where(births, cp, par)
        fed = jnp.where(births, 0, fed)
        # relatedness: newborn row = 0.5 x its parent's row; parent and siblings (same parent) 0.5; then symmetric
        pc = jnp.clip(cp, 0, cfg.N - 1)
        row = 0.5 * rel[pc]
        row = jnp.where(jnp.arange(cfg.N)[None, :] == pc[:, None], 0.5, row)
        row = jnp.where((par[None, :] == cp[:, None]) & (cp[:, None] >= 0), 0.5, row)
        rel = jnp.where(births[:, None], row, rel)
        rel = jnp.where(births[None, :], rel.T, rel)
        rel = rel * (1.0 - jnp.eye(cfg.N))
        alive2 = alive2 | births
    if cfg.juvenile and cfg.reproduce:   # parent-paid provisioning
        young = alive2 & (age < cfg.juvenile) & (par >= 0)
        pidx = jnp.clip(par, 0, cfg.N - 1)
        need = jnp.where(young & alive2[pidx], jnp.maximum(cfg.E0 - E, 0.0), 0.0)
        paid = jnp.minimum(need, jnp.zeros(cfg.N).at[pidx].add(0.0) + E[pidx])        # can't pay more than it has
        E = E + paid
        E = E - jnp.zeros(cfg.N).at[pidx].add(paid)
        E = jnp.clip(E, 0.0, 1.0)
    s2 = State(food, cap, centres, pos, E, age, alive2, fed, rel, par)
    return s2, r, dict(dead=dead, births=births, take=take, moved=moved)
