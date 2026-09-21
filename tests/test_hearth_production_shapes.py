"""Cam's hearth spec (2026-09-21): prove hearth deposits and completions fire
at production shapes -- grid_size=128, max_pop=200 (config.yaml) -- embedded
in a realistic scattered population, at both N=1 (stage 0, solo-satisfiable)
and N=2 (stage 1, the first real coordination stage), with reward payment
computed the same one-line way jax_sim/main_jax.py does it (not a
reimplementation of grid logic). Mirrors the discipline
test_pair_craft_success_production_shapes.py established for the mechanism
it replaces.
"""

import jax
import jax.numpy as jnp

from jax_sim.grid_jax import resolve_hearth_deposits, generate_hearth_positions

GRID_SIZE = 128
MAX_POP = 200
REWARD_HEARTH_DEPOSIT = 0.3    # config.yaml default
REWARD_HEARTH_COMPLETION = 2.4  # config.yaml default (8x deposit)
DECAY_FACTOR = 0.5 ** (1.0 / 200.0)  # config.yaml hearth_decay_half_life_steps default


def _scattered_population(key, n=MAX_POP, exclude_positions=()):
    positions = jax.random.randint(key, (n, 2), 0, GRID_SIZE)
    for i, p in enumerate(exclude_positions):
        positions = positions.at[i].set(jnp.array(p))
    return positions


def test_solo_bootstrap_fires_at_production_shapes():
    """N=1: a single ideal agent, embedded in 199 scattered mostly-idle
    others (some of whom crafting elsewhere, uselessly), deposits and
    completes alone."""
    key = jax.random.PRNGKey(0)
    hearth_positions = generate_hearth_positions(GRID_SIZE)
    hearth_need = jnp.zeros((4,), dtype=jnp.int32)  # all want wood
    hearth_fill = jnp.zeros((4,), dtype=jnp.float32)
    hearth_credit = jnp.zeros((4, MAX_POP), dtype=jnp.float32)

    positions = _scattered_population(key, exclude_positions=[tuple(int(v) for v in hearth_positions[0])])
    is_craft = jnp.zeros((MAX_POP,), dtype=bool).at[0].set(True)
    # A third, distant agent also crafts uselessly (noise, no partner, empty-handed).
    is_craft = is_craft.at[50].set(True)
    inv_wood = jnp.zeros((MAX_POP,), dtype=jnp.int32).at[0].set(1)
    inv_stone, inv_flint, inv_clay, inv_vine = (jnp.zeros((MAX_POP,), dtype=jnp.int32) for _ in range(4))

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(DECAY_FACTOR),
        reroll_key=jax.random.PRNGKey(99),
    )
    assert bool(out["deposited"][0])
    assert bool(out["completed"][0])
    assert not bool(out["attempted"][50]), "distant empty-handed crafter shouldn't even count as an attempt"
    assert int(out["deposited"].astype(jnp.int32).sum()) == 1, "no one else should have deposited"

    b_rew = REWARD_HEARTH_DEPOSIT * out["deposited"].astype(jnp.float32)
    b_rew = b_rew + REWARD_HEARTH_COMPLETION * jnp.sum(out["completion_share"], axis=0)
    assert abs(float(b_rew[0]) - (REWARD_HEARTH_DEPOSIT + REWARD_HEARTH_COMPLETION)) < 1e-6
    assert float(jnp.sum(b_rew)) == float(b_rew[0]), "reward should be entirely localized to the one depositor"


def test_pair_coordination_fires_at_production_shapes():
    """N=2: two ideal agents at the same hearth, same step, both holding the
    needed material, complete and split the completion reward -- embedded
    in the same 200-agent scattered population with unrelated noise."""
    key = jax.random.PRNGKey(1)
    hearth_positions = generate_hearth_positions(GRID_SIZE)
    hearth_need = jnp.zeros((4,), dtype=jnp.int32)
    hearth_fill = jnp.zeros((4,), dtype=jnp.float32)
    hearth_credit = jnp.zeros((4, MAX_POP), dtype=jnp.float32)

    positions = _scattered_population(
        key, exclude_positions=[
            tuple(int(v) for v in hearth_positions[0]),
            tuple(int(v) for v in hearth_positions[0]),
        ],
    )
    is_craft = jnp.zeros((MAX_POP,), dtype=bool).at[0].set(True).at[1].set(True)
    is_craft = is_craft.at[75].set(True)  # distant noise crafter
    inv_wood = jnp.zeros((MAX_POP,), dtype=jnp.int32).at[0].set(1).at[1].set(1)
    inv_stone = jnp.zeros((MAX_POP,), dtype=jnp.int32).at[75].set(1)  # noise: wrong hearth's material anyway
    inv_flint, inv_clay, inv_vine = (jnp.zeros((MAX_POP,), dtype=jnp.int32) for _ in range(3))

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(2), decay_factor=jnp.array(DECAY_FACTOR),
        reroll_key=jax.random.PRNGKey(98),
    )
    assert bool(out["deposited"][0]) and bool(out["deposited"][1])
    assert bool(out["completed"][0])
    assert int(out["deposited"].astype(jnp.int32).sum()) == 2

    b_rew = REWARD_HEARTH_DEPOSIT * out["deposited"].astype(jnp.float32)
    b_rew = b_rew + REWARD_HEARTH_COMPLETION * jnp.sum(out["completion_share"], axis=0)
    expected_each = REWARD_HEARTH_DEPOSIT + REWARD_HEARTH_COMPLETION / 2.0
    assert abs(float(b_rew[0]) - expected_each) < 1e-6
    assert abs(float(b_rew[1]) - expected_each) < 1e-6
    assert float(b_rew[75]) == 0.0, "noise crafter holding the wrong material earns nothing"


if __name__ == "__main__":
    test_solo_bootstrap_fires_at_production_shapes()
    test_pair_coordination_fires_at_production_shapes()
    print(
        "OK: hearth deposits and completions fire correctly at production shapes "
        "(grid_size=128, max_pop=200) for both N=1 (solo bootstrap) and N=2 (pair "
        "coordination), with reward payment matching the exact config-scaled "
        "computation jax_sim/main_jax.py performs, embedded in a realistic scattered "
        "population including unrelated noise crafters."
    )
