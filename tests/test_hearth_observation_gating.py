"""Cam's hearth spec (2026-09-21): "Hearth positions are visible to every
agent. Hearth needs are gated by can_see_recipe." That's the whole design in
one line -- this test pins it against the real observation builder, not a
reimplementation: own_state's hearth_rel block (8 dims: 4 hearths x
toroidal-wrapped dx,dy) must be identical for informed and uninformed
agents at the same position, while the hearth_need block (4 dims, one per
hearth) must be exactly zero for an uninformed agent and equal to
hearth_need/4.0 for an informed one.
"""

import yaml
import jax.numpy as jnp

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.obs_layout import make_obs_layout
from jax_sim.observations_jax import build_observations_jax
from jax_sim.grid_jax import GridState
from jax_sim.population_jax import PopState

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})

GRID_SIZE = int(_cfg["grid_size"])
_layout = make_obs_layout(
    signal_dim=int(_cfg["signal_dim"]), symbol_dim=int(_cfg["symbol_dim"]),
    memory_slots=int(_cfg.get("memory_slots", 0)), neighbor_k=int(_cfg["neighbor_k"]),
    local_cells=(2 * int(_cfg["local_obs_radius"]) + 1) ** 2,
    env_channels=int(_cfg.get("env_channels", 15)), own_state_dim=int(_cfg.get("own_state_dim", 29)),
)
assert _layout.own_state_dim == 29, "test assumes the live 29-dim own_state layout"

# own_state column layout (see jax_sim/observations_jax.py's build_observations_jax):
# [0:13) base fields (incl. can_see_recipe at 12), [13:21) hearth_rel, [21:25) hearth_need,
# [25:29) intrinsic_entropy.
HEARTH_REL_SLICE = slice(13, 21)
HEARTH_NEED_SLICE = slice(21, 25)


def _make_pop_and_grid(n=4, can_see_recipe=None, positions=None, memory_slots=0):
    pop = PopState(n, hidden_dim=int(_cfg["hidden_dim"]), signal_dim=int(_cfg["signal_dim"]),
                   memory_slots=memory_slots)
    if positions is None:
        positions = jnp.zeros((n, 2), dtype=jnp.int32)
    if can_see_recipe is None:
        can_see_recipe = jnp.zeros((n,), dtype=bool)
    pop = pop.replace(
        alive=jnp.ones((n,), dtype=bool),
        positions=jnp.array(positions, dtype=jnp.int32),
        can_see_recipe=jnp.array(can_see_recipe, dtype=bool),
        energy=jnp.full((n,), 0.5),
    )
    grid = GridState(GRID_SIZE, symbol_dim=int(_cfg["symbol_dim"]), max_pop=n)
    grid = grid.replace(hearth_need=jnp.array([1, 2, 3, 4], dtype=jnp.int32))
    return pop, grid


def _build_obs(pop, grid):
    blue_map = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    blue_map = blue_map.at[pop.positions[:, 0], pop.positions[:, 1]].set(True)
    red_map = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    return build_observations_jax(pop, grid, blue_map, red_map, _cfg, step=0)


def test_hearth_positions_visible_regardless_of_can_see_recipe():
    # neighbor_k (6) requires at least 7 agents for get_neighbour_signals'
    # top_k; pad with agents elsewhere on the map, unrelated to the check.
    n = 8
    positions = jnp.zeros((n, 2), dtype=jnp.int32).at[0].set(jnp.array([10, 10])).at[1].set(jnp.array([10, 10]))
    can_see_recipe = jnp.zeros((n,), dtype=bool).at[1].set(True)
    pop, grid = _make_pop_and_grid(n=n, can_see_recipe=can_see_recipe, positions=positions)
    obs = _build_obs(pop, grid)
    rel_uninformed = obs[0, HEARTH_REL_SLICE]
    rel_informed = obs[1, HEARTH_REL_SLICE]
    assert jnp.allclose(rel_uninformed, rel_informed), (
        "hearth-relative positions must be identical for informed and uninformed agents "
        "at the same position -- position visibility does not depend on can_see_recipe"
    )
    assert not bool(jnp.allclose(rel_uninformed, 0.0)), (
        "sanity: an agent not standing on any hearth should see nonzero relative offsets"
    )


def test_hearth_need_is_zero_for_uninformed_and_present_for_informed():
    n = 8
    can_see_recipe = jnp.zeros((n,), dtype=bool).at[1].set(True)
    pop, grid = _make_pop_and_grid(n=n, can_see_recipe=can_see_recipe)
    obs = _build_obs(pop, grid)
    need_uninformed = obs[0, HEARTH_NEED_SLICE]
    need_informed = obs[1, HEARTH_NEED_SLICE]
    assert bool(jnp.allclose(need_uninformed, 0.0)), (
        "uninformed agent must see all-zero hearth needs -- this IS the gate"
    )
    expected = jnp.array([1, 2, 3, 4], dtype=jnp.float32) / 4.0
    assert jnp.allclose(need_informed, expected), (
        f"informed agent should see hearth_need/4.0 = {expected}, got {need_informed}"
    )


def test_hearth_rel_is_zero_when_standing_on_that_hearth():
    from jax_sim.grid_jax import generate_hearth_positions
    hearth_positions = generate_hearth_positions(GRID_SIZE)
    n = 8
    positions = jnp.zeros((n, 2), dtype=jnp.int32).at[0].set(hearth_positions[0])
    can_see_recipe = jnp.zeros((n,), dtype=bool).at[0].set(True)
    pop, grid = _make_pop_and_grid(n=n, can_see_recipe=can_see_recipe, positions=positions)
    obs = _build_obs(pop, grid)
    rel = obs[0, HEARTH_REL_SLICE].reshape(4, 2)
    assert jnp.allclose(rel[0], 0.0), "standing exactly on hearth 0 should give it zero relative offset"
    assert not bool(jnp.allclose(rel[1], 0.0)), "the other hearths should not also read zero"


if __name__ == "__main__":
    test_hearth_positions_visible_regardless_of_can_see_recipe()
    test_hearth_need_is_zero_for_uninformed_and_present_for_informed()
    test_hearth_rel_is_zero_when_standing_on_that_hearth()
    print(
        "OK: hearth positions are visible in own_state regardless of can_see_recipe, and "
        "hearth needs are exactly zero for uninformed agents and correctly gated-through for "
        "informed ones -- verified against the real build_observations_jax, at the live "
        "29-dim own_state layout."
    )
