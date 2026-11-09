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
    c_rep: float = 0.4
    reproduce: bool = False
    E_set: float = 0.8
    drive_n: float = 2.0


class State(NamedTuple):
    food: jnp.ndarray
    cap: jnp.ndarray
    centres: jnp.ndarray
    pos: jnp.ndarray
    energy: jnp.ndarray
    age: jnp.ndarray
    alive: jnp.ndarray


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
    return State(cap, cap, centres, pos, jnp.full(cfg.N, cfg.E0), jnp.zeros(cfg.N, jnp.int32), jnp.ones(cfg.N, bool))


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
    food = food + cfg.regrow * (s.cap - food)
    E_prev = s.energy
    E = jnp.clip(E_prev + cfg.eff * take - (cfg.d0 + cfg.d_move * moved) * alive, 0.0, 1.0)
    age = s.age + alive.astype(jnp.int32)
    dead = alive & ((E <= 0.0) | (age >= cfg.A_max))
    alive2 = alive & ~dead
    r = jnp.where(alive, drive(cfg, E_prev) - drive(cfg, E), 0.0)
    births = jnp.zeros(cfg.N, bool)
    if cfg.reproduce:
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
        pos = pos.at[child].set(cpos, mode="drop")
        E = E.at[child].set(cfg.c_rep * 0.9, mode="drop")
        age = age.at[child].set(0, mode="drop")
        births = jnp.zeros(cfg.N, bool).at[child].set(True, mode="drop")
        alive2 = alive2 | births
    s2 = State(food, s.cap, s.centres, pos, E, age, alive2)
    return s2, r, dict(dead=dead, births=births, take=take, moved=moved)
