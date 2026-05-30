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
        self.puzzle_nodes  = jnp.zeros((3, 6), dtype=jnp.int32) # [ay, ax, by, bx, ry, rx]
        self.puzzle_active = jnp.zeros((3,), dtype=jnp.bool_)
        self.puzzle_cooldown = jnp.zeros((3,), dtype=jnp.int32)

    def tree_flatten(self):
        children = (
            self.symbols, self.walls, self.resources,
            self.shelter_spots, self.contested_res, self.scent_trails,
            self.cultural_fast, self.cultural_slow, self.puzzle_grid,
            self.puzzle_nodes, self.puzzle_active, self.puzzle_cooldown,
        )
        aux = (self.size, self.symbol_dim)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        size, symbol_dim = aux
        gs = cls(size, symbol_dim)
        (gs.symbols, gs.walls, gs.resources,
         gs.shelter_spots, gs.contested_res, gs.scent_trails,
         gs.cultural_fast, gs.cultural_slow, gs.puzzle_grid,
         gs.puzzle_nodes, gs.puzzle_active, gs.puzzle_cooldown) = children
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
        gs.puzzle_nodes = kwargs.get("puzzle_nodes", self.puzzle_nodes)
        gs.puzzle_active = kwargs.get("puzzle_active", self.puzzle_active)
        gs.puzzle_cooldown = kwargs.get("puzzle_cooldown", self.puzzle_cooldown)
        return gs

jax.tree_util.register_pytree_node(
    GridState,
    lambda gs: gs.tree_flatten(),
    lambda aux, children: GridState.tree_unflatten(aux, children)
)


# ── World generation (called once at init, not JIT) ──────────────────────

def generate_resource_patches(
    key: jnp.ndarray,
    grid_size: int,
    n_patches: int = 20,
    patch_radius: float = 5.0,
) -> jnp.ndarray:
    """Generate Gaussian resource patches on the toroidal grid."""
    resources = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)
    keys = jax.random.split(key, n_patches)
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(grid_size), jnp.arange(grid_size), indexing='ij'
    )
    for i in range(n_patches):
        cy, cx = jax.random.randint(keys[i], (2,), 0, grid_size)
        dy = jnp.minimum(jnp.abs(y_coords - cy), grid_size - jnp.abs(y_coords - cy))
        dx = jnp.minimum(jnp.abs(x_coords - cx), grid_size - jnp.abs(x_coords - cx))
        patch = jnp.exp(-(dy**2 + dx**2) / (2.0 * patch_radius**2))
        resources = resources + patch
    return jnp.clip(resources, 0.0, 1.0)


def generate_shelter_spots(
    key: jnp.ndarray,
    grid_size: int,
    n_spots: int = 5,
    radius: int = 2,
) -> jnp.ndarray:
    """Place shelter zones where blues are protected from red catches."""
    shelter = jnp.zeros((grid_size, grid_size), dtype=jnp.bool_)
    keys = jax.random.split(key, n_spots)
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(grid_size), jnp.arange(grid_size), indexing='ij'
    )
    for i in range(n_spots):
        cy, cx = jax.random.randint(keys[i], (2,), 0, grid_size)
        dy = jnp.minimum(jnp.abs(y_coords - cy), grid_size - jnp.abs(y_coords - cy))
        dx = jnp.minimum(jnp.abs(x_coords - cx), grid_size - jnp.abs(x_coords - cx))
        dist = jnp.maximum(dy, dx)
        shelter = shelter | (dist <= radius)
    return shelter


def generate_contested_nodes(
    key: jnp.ndarray,
    grid_size: int,
    n_nodes: int = 3,
    yield_mult: float = 3.0,
    radius: int = 2,
) -> jnp.ndarray:
    """Place high-yield contested resource nodes (require 2+ agents to harvest)."""
    contested = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)
    keys = jax.random.split(key, n_nodes)
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(grid_size), jnp.arange(grid_size), indexing='ij'
    )
    for i in range(n_nodes):
        cy, cx = jax.random.randint(keys[i], (2,), 0, grid_size)
        dy = jnp.minimum(jnp.abs(y_coords - cy), grid_size - jnp.abs(y_coords - cy))
        dx = jnp.minimum(jnp.abs(x_coords - cx), grid_size - jnp.abs(x_coords - cx))
        dist = jnp.maximum(dy, dx)
        contested = jnp.where(dist <= radius, yield_mult, contested)
    return contested


def update_scent_trails(
    scent_trails: jnp.ndarray,
    r_positions: jnp.ndarray,
    r_alive: jnp.ndarray,
    intensity: float = 0.8,
    decay_steps: int = 20,
) -> jnp.ndarray:
    """Reds deposit scent; trails decay over time."""
    decay_rate = 1.0 / max(decay_steps, 1)
    new_scent = scent_trails * (1.0 - decay_rate)
    alive_f = r_alive.astype(jnp.float32)
    new_scent = new_scent.at[r_positions[:, 0], r_positions[:, 1]].add(
        intensity * alive_f
    )
    return jnp.clip(new_scent, 0.0, 1.0)


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
    catch_prob: float = 1.0,
    rng: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return (b_new_alive, r_catch_reward, b_catch_penalty, caught_idx_mask)
    Red catches blue if within catch_radius Chebyshev distance.
    If catch_prob < 1.0, each in-range blue survives with probability (1 - catch_prob)
    (predator jitter / safety bubble).
    """
    max_b = b_pos.shape[0]

    # Compute pairwise Chebyshev distances (toroidal)
    diff = jnp.abs(b_pos[:, None, :] - r_pos[None, :, :])  # (B, R, 2)
    diff = jnp.minimum(diff, grid_size - diff)
    dist = jnp.max(diff, axis=-1)  # (B, R)

    in_range = (dist <= catch_radius) & b_alive[:, None] & r_alive[None, :]
    in_range_b = jnp.any(in_range, axis=1)  # (B,)

    if catch_prob >= 1.0:
        caught_b = in_range_b
        caught = in_range
    else:
        roll = jax.random.uniform(rng, (max_b,))
        caught_b = in_range_b & (roll < catch_prob)
        caught = in_range & caught_b[:, None]

    b_new_alive = b_alive & ~caught_b

    # Red reward per successful catch (one per blue actually caught)
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


# ── Puzzle logic ───────────────────────────────────────────────────────────

def generate_puzzle_nodes(
    key: jnp.ndarray,
    n_nodes: int,
    grid_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (puzzle_nodes, active, cooldown).
    puzzle_nodes: (n_nodes, 6) -> [ay, ax, by, bx, ry, rx]
    """
    keys = jax.random.split(key, n_nodes)
    def _gen_one(k):
        k1, k2 = jax.random.split(k)
        ay, ax = jax.random.randint(k1, (2,), 0, grid_size)
        by = (ay + grid_size // 2) % grid_size
        bx = (ax + grid_size // 2) % grid_size
        ry = (ay + by) // 2
        rx = (ax + bx) // 2
        return jnp.array([ay, ax, by, bx, ry, rx], dtype=jnp.int32)
    
    nodes = jax.vmap(_gen_one)(keys)
    active = jnp.ones((n_nodes,), dtype=jnp.bool_)
    cooldown = jnp.zeros((n_nodes,), dtype=jnp.int32)
    return nodes, active, cooldown


def update_puzzle_grid(
    grid_size: int,
    puzzle_nodes: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Renders the puzzle nodes onto a 2D observation layer."""
    pg = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)
    def _add_node(carry, data):
        g = carry
        ay, ax, by, bx, ry, rx, is_act = data
        g = jnp.where(is_act > 0, g.at[ay, ax].set(0.5), g)
        g = jnp.where(is_act > 0, g.at[by, bx].set(0.6), g)
        g = jnp.where(is_act > 0, g.at[ry, rx].set(0.9), g)
        return g, None

    data = jnp.concatenate([puzzle_nodes, active[:, None].astype(jnp.int32)], axis=1)
    pg, _ = lax.scan(_add_node, pg, data)
    return pg


def decay_puzzle_timeout(
    active: jnp.ndarray,
    cooldown: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ticks down cooldowns and reactivates nodes."""
    cooldown = jnp.maximum(0, cooldown - 1)
    reactivate = (cooldown == 0) & (~active)
    new_active = active | reactivate
    return new_active, cooldown


def check_puzzle_solved(
    positions: jnp.ndarray,   # (max_pop, 2)
    alive:     jnp.ndarray,
    puzzle_nodes: jnp.ndarray,  # (n_nodes, 6)
    active:    jnp.ndarray,
    cooldown:  jnp.ndarray,
    grid_size: int,
    switch_dist: int = 2,
    cooldown_time: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (rewards_per_agent, solved_mask, new_active, new_cooldown).
    """
    max_pop = positions.shape[0]
    n_nodes = puzzle_nodes.shape[0]
    rewards = jnp.zeros(max_pop, dtype=jnp.float32)
    solved_any = jnp.zeros(max_pop, dtype=jnp.bool_)

    def _check_one_node(carry, node_data):
        rew, solved, n_act, n_cool = carry
        ay, ax, by, bx, ry, rx, is_active, curr_cool = node_data
        is_active = is_active > 0

        dist_a = chebyshev_dist(positions, jnp.array([ay, ax]), grid_size)
        dist_b = chebyshev_dist(positions, jnp.array([by, bx]), grid_size)

        on_a = (dist_a <= switch_dist) & alive
        on_b = (dist_b <= switch_dist) & alive

        solved_this = is_active & jnp.any(on_a) & jnp.any(on_b)

        on_either = on_a | on_b
        new_rew = jnp.where(solved_this, rew + on_either.astype(jnp.float32), rew)
        new_solved = solved | (solved_this & on_either)
        
        # update state
        out_act = jnp.where(solved_this, jnp.zeros_like(is_active, dtype=jnp.bool_), is_active)
        out_cool = jnp.where(solved_this, jnp.array(cooldown_time, dtype=jnp.int32), curr_cool)

        return (new_rew, new_solved, out_act, out_cool), None

    node_data = jnp.concatenate([
        puzzle_nodes, 
        active[:, None].astype(jnp.int32),
        cooldown[:, None]
    ], axis=1)

    init_state = (rewards, solved_any, jnp.zeros((), dtype=jnp.bool_), jnp.zeros((), dtype=jnp.int32))
    # scan over nodes is hard to unpack into arrays directly, let's vmap it over nodes instead.
    # Actually lax.scan can collect the state arrays by doing:
    def _scan_fn(carry, nd):
        rew, solved = carry
        ay, ax, by, bx, ry, rx, is_active, curr_cool = nd
        is_active = is_active > 0
        dist_a = chebyshev_dist(positions, jnp.array([ay, ax]), grid_size)
        dist_b = chebyshev_dist(positions, jnp.array([by, bx]), grid_size)
        on_a = (dist_a <= switch_dist) & alive
        on_b = (dist_b <= switch_dist) & alive
        solved_this = is_active & jnp.any(on_a) & jnp.any(on_b)
        on_either = on_a | on_b
        new_rew = jnp.where(solved_this, rew + on_either.astype(jnp.float32), rew)
        new_solved = solved | (solved_this & on_either)
        out_act = jnp.where(solved_this, False, is_active)
        out_cool = jnp.where(solved_this, cooldown_time, curr_cool)
        return (new_rew, new_solved), jnp.stack([out_act, out_cool])

    (rewards, solved_any), out_states = lax.scan(_scan_fn, (rewards, solved_any), node_data)
    new_active = out_states[:, 0].astype(jnp.bool_)
    new_cooldown = out_states[:, 1].astype(jnp.int32)

    return rewards, solved_any, new_active, new_cooldown
