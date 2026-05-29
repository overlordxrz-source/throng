"""JAX observation builder (separate module so git pull + train_entry evicts stale code)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_sim.grid_jax import GridState, get_local_patches, get_neighbour_signals
from jax_sim.population_jax import PopState

RED_SENSE_API_VERSION = 2


def _mask_loc_env_red_channel(
    loc_env: jnp.ndarray,
    pop: PopState,
    red_map: jnp.ndarray,
    shelter_spots: jnp.ndarray,
    det_r: int,
    gs: int,
) -> jnp.ndarray:
    """Zero loc_env red-presence channel when no live red within Chebyshev det_r."""
    coords_y, coords_x = jnp.meshgrid(
        jnp.arange(gs), jnp.arange(gs), indexing="ij"
    )
    grid_coords = jnp.stack([coords_y, coords_x], axis=-1).astype(jnp.float32)
    b_pos = pop.positions.astype(jnp.float32)
    diff = jnp.abs(b_pos[:, None, None, :] - grid_coords[None, :, :, :])
    diff = jnp.minimum(diff, gs - diff)
    cheb = jnp.max(diff, axis=-1)
    cheb_at_reds = jnp.where(red_map[None, :, :], cheb, jnp.inf)
    min_dist = cheb_at_reds.min(axis=(1, 2))
    in_shelter = shelter_spots[pop.positions[:, 0], pop.positions[:, 1]]
    effective_det_r = jnp.where(in_shelter, float(det_r) * 2.0, float(det_r))
    blind = jnp.isfinite(min_dist) & (min_dist > effective_det_r) & pop.alive
    return loc_env.at[:, :, 1].set(
        jnp.where(blind[:, None], 0.0, loc_env[:, :, 1])
    )


def build_observations_jax(
    pop: PopState,
    grid: GridState,
    blue_map: jnp.ndarray,
    red_map: jnp.ndarray,
    config: dict,
    step: int,
    key: jnp.ndarray = None,
    limit_red_sensing: bool = False,
) -> jnp.ndarray:
    """Build flat observation vector for all agents."""
    gs = config["grid_size"]
    K = config["neighbor_k"]
    r = config["local_obs_radius"]
    W = (2 * r + 1) ** 2
    N = pop.max_pop

    norm_age = pop.ages.astype(jnp.float32) / float(config["max_age"])
    norm_x = pop.positions[:, 0].astype(jnp.float32) / gs
    norm_y = pop.positions[:, 1].astype(jnp.float32) / gs
    energy = pop.energy
    nl_norm = pop.n_layers.astype(jnp.float32) / 6.0
    mat_frac = jnp.zeros(N, dtype=jnp.float32)

    own_state = jnp.stack([norm_age, mat_frac, energy, nl_norm, norm_x, norm_y], axis=1)

    nb_sigs = get_neighbour_signals(
        pop.positions, pop.signals, pop.alive, K, gs,
    )

    loc_sym = get_local_patches(grid.symbols, pop.positions, r, gs)
    loc_pres = jnp.concatenate([
        get_local_patches(blue_map.astype(jnp.float32), pop.positions, r, gs)[..., None],
        get_local_patches(red_map.astype(jnp.float32), pop.positions, r, gs)[..., None],
    ], axis=-1)
    loc_wall = get_local_patches(grid.walls.astype(jnp.float32), pop.positions, r, gs)[..., None]
    loc_res = get_local_patches(grid.resources, pop.positions, r, gs)[..., None]
    loc_shelter = get_local_patches(grid.shelter_spots.astype(jnp.float32), pop.positions, r, gs)[..., None]
    loc_contested = get_local_patches(grid.contested_res, pop.positions, r, gs)[..., None]
    loc_scent = get_local_patches(grid.scent_trails, pop.positions, r, gs)[..., None]
    loc_puzzle = get_local_patches(grid.puzzle_grid, pop.positions, r, gs)[..., None]

    loc_env = jnp.concatenate([
        loc_pres, loc_wall, loc_res, loc_shelter, loc_contested, loc_scent, loc_puzzle
    ], axis=-1)

    if key is not None:
        k_noise, k_gate = jax.random.split(key)
        noise_std = float(config.get("resource_obs_noise", 0.0))
        if noise_std > 0:
            res_noise = jax.random.normal(k_noise, loc_res.shape) * noise_std
            loc_env = loc_env.at[:, :, 3].add(res_noise.squeeze(-1))

        gate_frac = float(config.get("signal_gate_mask_frac", 0.0))
        if config.get("signal_gate_enabled", False) and gate_frac > 0:
            gate_mask = jax.random.bernoulli(k_gate, 1.0 - gate_frac, loc_env.shape)
            loc_env = loc_env * gate_mask

    loc_cult_fast = get_local_patches(grid.cultural_fast, pop.positions, r, gs)
    loc_cult_slow = get_local_patches(grid.cultural_slow, pop.positions, r, gs)

    det_r = int(config.get("red_detection_radius", 0))
    if det_r > 0 and limit_red_sensing:
        loc_env = _mask_loc_env_red_channel(
            loc_env, pop, red_map, grid.shelter_spots, det_r, gs
        )

    parts = [
        own_state,
        nb_sigs.reshape(N, -1),
        loc_sym.reshape(N, -1),
        loc_env.reshape(N, -1),
        pop.signals,
    ]

    if pop.memory_buffer is not None:
        parts.append(pop.memory_buffer.reshape(N, -1))

    parts.append(loc_cult_fast.reshape(N, -1))
    parts.append(loc_cult_slow.reshape(N, -1))

    obs = jnp.concatenate(parts, axis=1)
    return jnp.where(pop.alive[:, None], obs, 0.0)
