"""
jax_sim/grid_jax.py — Environment state + ops (pure JAX, @jit-friendly)

All functions accept/return JAX arrays.  No in-place mutation.
Grid layers are immutable — every "write" returns a new array.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict, Any

# ── Grid layers (immutable) ──────────────────────────────────────────────────

class GridState:
    """All environment layers packed into a pytree-friendly dict."""
    def __init__(self, size: int, symbol_dim: int = 16):
        self.size = size
        self.symbol_dim = symbol_dim
        self.symbols       = jnp.zeros((size, size, symbol_dim), dtype=jnp.float32)
        self.walls         = jnp.zeros((size, size), dtype=jnp.bool_)
        self.resources     = jnp.zeros((size, size), dtype=jnp.float32)
        self.shelter_spots = jnp.zeros((size, size), dtype=jnp.bool_)
        self.contested_res = jnp.zeros((size, size), dtype=jnp.float32)
        self.scent_trails  = jnp.zeros((size, size), dtype=jnp.float32)
        self.cultural_fast = jnp.zeros((size, size, symbol_dim), dtype=jnp.float32)
        self.cultural_slow = jnp.zeros((size, size, symbol_dim), dtype=jnp.float32)
        # Puzzle
        self.puzzle_grid   = jnp.zeros((size, size), dtype=jnp.float32)

    def tree_flatten(self):
        children = (
            self.symbols, self.walls, self.resources,
            self.shelter_spots, self.contested_res, self.scent_trails,
            self.cultural_fast, self.cultural_slow, self.puzzle_grid,
        )
        aux = (self.size, self.symbol_dim)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        size, symbol_dim = aux
        gs = cls(size, symbol_dim)
        (gs.symbols, gs.walls, gs.resources,
         gs.shelter_spots, gs.contested_res, gs.scent_trails,
         gs.cultural_fast, gs.cultural_slow, gs.puzzle_grid) = children
        return gs

    def replace(self, **kwargs):
        """Immutable update: return new GridState with replaced fields."""
        gs = GridState(self.size, self.symbol_dim)
        gs.symbols = kwargs.get("symbols", self.symbols)
        gs.walls = kwargs.get("walls", self.walls)
        gs.resources = kwargs.get("resources", self.resources)
        gs.shelter_spots = kwargs.get("shelter_spots", self.shelter_spots)
        gs.contested_res = kwargs.get("contested_res", self.contested_res)
        gs.scent_trails = kwargs.get("scent_trails", self.scent_trails)
        gs.cultural_fast = kwargs.get("cultural_fast", self.cultural_fast)
        gs.cultural_slow = kwargs.get("cultural_slow", self.cultural_slow)
        gs.puzzle_grid = kwargs.get("puzzle_grid", self.puzzle_grid)
        return gs

jax.tree_util.register_pytree_node(
    GridState,
    lambda gs: gs.tree_flatten(),
    lambda aux, children: GridState.tree_unflatten(aux, children)
)


# ── Toroidal helpers ─────────────────────────────────────────────────────────

def wrap(pos: jnp.ndarray, size: int) -> jnp.ndarray:
    """pos: (..., 2)  ->  wrapped positions"""
    return jnp.mod(pos, size)


def chebyshev_dist(a: jnp.ndarray, b: jnp.ndarray, size: int) -> jnp.ndarray:
    """a,b: (2,) or (N,2); returns scalar or (N,)"""
    diff = jnp.abs(a - b)
    diff = jnp.minimum(diff, size - diff)
    return jnp.max(diff, axis=-1)


# ── Movement ─────────────────────────────────────────────────────────────────

def apply_moves(
    positions: jnp.ndarray,  # (max_pop, 2) int32
    actions:   jnp.ndarray,  # (max_pop,) int32  0=stay 1=N 2=S 3=E 4=W
    alive:     jnp.ndarray,  # (max_pop,) bool
    grid_size: int,
    walls:     jnp.ndarray,  # (size, size) bool
) -> jnp.ndarray:
    """Return new positions after movement (collision with walls = stay)."""
    # Action → delta
    deltas = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    new_pos = positions + deltas[actions]
    new_pos = wrap(new_pos, grid_size)
    # Wall collision: if target cell is wall, stay
    wall_hit = walls[new_pos[:, 0], new_pos[:, 1]]
    new_pos = jnp.where(wall_hit[:, None] | ~alive[:, None], positions, new_pos)
    return new_pos


# ── Resource consumption ───────────────────────────────────────────────────

def consume_resources(
    positions: jnp.ndarray,  # (max_pop, 2) int32
    alive:     jnp.ndarray,  # (max_pop,) bool
    resources: jnp.ndarray,  # (size, size) float32
    decay:     float = 0.05,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (energy_gained, new_resources)."""
    max_pop = positions.shape[0]
    # Gather resource at each position
    gathered = resources[positions[:, 0], positions[:, 1]] * alive.astype(jnp.float32)
    energy = jnp.clip(gathered, 0.0, 1.0)
    # Subtract from grid
    mask = jnp.zeros_like(resources).at[positions[:, 0], positions[:, 1]].add(
        alive.astype(jnp.float32)
    )
    new_res = resources - decay * mask
    new_res = jnp.clip(new_res, 0.0, 1.0)
    return energy, new_res


# ── Catch detection ──────────────────────────────────────────────────────────

def apply_catches(
    b_pos: jnp.ndarray,  # (max_pop_b, 2)
    b_alive: jnp.ndarray,
    r_pos: jnp.ndarray,  # (max_pop_r, 2)
    r_alive: jnp.ndarray,
    grid_size: int,
    catch_radius: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return (b_new_alive, r_catch_reward, b_catch_penalty, caught_idx_mask)
    Red catches blue if within catch_radius Chebyshev distance.
    """
    max_b = b_pos.shape[0]
    max_r = r_pos.shape[0]

    # Compute pairwise Chebyshev distances (toroidal)
    diff = jnp.abs(b_pos[:, None, :] - r_pos[None, :, :])  # (B, R, 2)
    diff = jnp.minimum(diff, grid_size - diff)
    dist = jnp.max(diff, axis=-1)  # (B, R)

    caught = (dist <= catch_radius) & b_alive[:, None] & r_alive[None, :]
    caught_b = jnp.any(caught, axis=1)  # (B,)

    b_new_alive = b_alive & ~caught_b

    # Red reward per catch (one per blue caught)
    r_catch_rew = jnp.sum(caught, axis=0).astype(jnp.float32)  # (R,)
    b_penalty = -1.0 * caught_b.astype(jnp.float32)

    return b_new_alive, r_catch_rew, b_penalty, caught_b


# ── Symbol / culture writes ─────────────────────────────────────────────────

def write_to_grid(
    grid: jnp.ndarray,       # (size, size, D) or (size, size)
    positions: jnp.ndarray,  # (max_pop, 2) int32
    values:    jnp.ndarray,  # (max_pop, D) or (max_pop,) float32
    alive:     jnp.ndarray,  # (max_pop,) bool
    intensity: float = 1.0,
) -> jnp.ndarray:
    """Scatter-add values into grid at positions. Returns new grid."""
    if grid.ndim == 2:
        # Scalar grid
        alive_f = alive.astype(jnp.float32)
        updates = values * alive_f * intensity
        new_grid = grid.at[positions[:, 0], positions[:, 1]].add(updates)
        return jnp.clip(new_grid, 0.0, 1.0)
    else:
        # Multi-channel grid
        alive_f = alive.astype(jnp.float32)[:, None]
        updates = values * alive_f * intensity
        new_grid = grid.at[positions[:, 0], positions[:, 1]].add(updates)
        return jnp.clip(new_grid, -1.0, 1.0)


def decay_grid(grid: jnp.ndarray, decay: float) -> jnp.ndarray:
    return grid * decay


# ── Local patch extraction ───────────────────────────────────────────────────

def get_local_patches(
    grid_layer: jnp.ndarray,  # (size, size) or (size, size, D)
    positions: jnp.ndarray,   # (max_pop, 2) int32
    radius: int,
    grid_size: int,
) -> jnp.ndarray:
    """
    Extract (max_pop, W, [D]) local patches via toroidal indexing.
    W = (2*radius+1)**2
    """
    max_pop = positions.shape[0]
    W = (2 * radius + 1) ** 2
    # Relative offsets
    dy = jnp.arange(-radius, radius + 1)
    dx = jnp.arange(-radius, radius + 1)
    dyg, dxg = jnp.meshgrid(dy, dx, indexing='ij')
    offsets = jnp.stack([dyg.ravel(), dxg.ravel()], axis=1)  # (W, 2)

    # Absolute cell coords (toroidal)
    cells = positions[:, None, :] + offsets[None, :, :]  # (max_pop, W, 2)
    cells = wrap(cells, grid_size)

    if grid_layer.ndim == 2:
        patches = grid_layer[cells[:, :, 0], cells[:, :, 1]]  # (max_pop, W)
    else:
        D = grid_layer.shape[2]
        patches = grid_layer[cells[:, :, 0], cells[:, :, 1]]  # (max_pop, W, D)
    return patches


# ── Neighbour signal aggregation ───────────────────────────────────────────

def get_neighbour_signals(
    positions: jnp.ndarray,   # (max_pop, 2)
    signals:   jnp.ndarray,   # (max_pop, sig_dim)
    alive:     jnp.ndarray,   # (max_pop,) bool
    k: int,
    grid_size: int,
) -> jnp.ndarray:
    """
    For each agent, find k nearest alive neighbours and return their signals.
    Returns (max_pop, k, sig_dim) with zeros for missing neighbours.
    """
    max_pop = positions.shape[0]
    sig_dim = signals.shape[1]

    # Pairwise distances (Chebyshev on torus)
    diff = jnp.abs(positions[:, None, :] - positions[None, :, :])
    diff = jnp.minimum(diff, grid_size - diff)
    dists = jnp.max(diff, axis=-1)  # (max_pop, max_pop)

    # Self-distance = large so we don't pick self
    dists = dists + jnp.eye(max_pop) * (grid_size * 10)

    # Mask dead agents
    dists = jnp.where(alive[None, :], dists, grid_size * 100)

    # Get k nearest indices
    _, neighbour_idx = jax.lax.top_k(-dists, k)  # (max_pop, k)

    # Gather signals
    nb_sigs = signals[neighbour_idx]  # (max_pop, k, sig_dim)
    valid = alive[neighbour_idx][:, :, None]  # (max_pop, k, 1)
    nb_sigs = jnp.where(valid, nb_sigs, 0.0)
    return nb_sigs


# ── Puzzle check ───────────────────────────────────────────────────────────

def check_puzzle_solved(
    positions: jnp.ndarray,   # (max_pop, 2)
    alive:     jnp.ndarray,
    puzzle_nodes: jnp.ndarray,  # (n_nodes, 6) — [ay, ax, by, bx, ry, rx]
    active:    jnp.ndarray,     # (n_nodes,) bool
    grid_size: int,
    switch_dist: int = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (rewards_per_agent, solved_mask).
    puzzle_nodes: (n_nodes, 6) with [switch_a_y, switch_a_x, switch_b_y, switch_b_x, reward_y, reward_x]
    """
    max_pop = positions.shape[0]
    n_nodes = puzzle_nodes.shape[0]
    rewards = jnp.zeros(max_pop, dtype=jnp.float32)
    solved_any = jnp.zeros(max_pop, dtype=jnp.bool_)

    alive_pos = positions  # all positions, we'll mask by alive later

    def _check_one_node(carry, node_info):
        rew, solved, active_flag = carry
        ay, ax, by, bx, ry, rx = node_info[:6]
        is_active = node_info[6] > 0.5  # active flag

        # Distances from all agents to both switches
        dist_a = chebyshev_dist(alive_pos, jnp.array([ay, ax]), grid_size)
        dist_b = chebyshev_dist(alive_pos, jnp.array([by, bx]), grid_size)

        on_a = (dist_a <= switch_dist) & alive
        on_b = (dist_b <= switch_dist) & alive

        solved_this = is_active & jnp.any(on_a) & jnp.any(on_b)

        # Reward agents on either switch
        on_either = on_a | on_b
        new_rew = jnp.where(solved_this, rew + on_either.astype(jnp.float32), rew)
        new_solved = solved | (solved_this & on_either)

        return (new_rew, new_solved, active_flag), None

    # Stack active flags into node_info
    node_data = jnp.concatenate([puzzle_nodes, active[:, None].astype(jnp.float32)], axis=1)

    (rewards, solved_any, _), _ = lax.scan(
        _check_one_node,
        (rewards, solved_any, jnp.zeros((), dtype=jnp.bool_)),
        node_data,
    )

    return rewards, solved_any
