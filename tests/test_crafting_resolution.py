"""Pins jax_sim.grid_jax.resolve_crafting() -- the crafting resolution + failure-
cause breakdown Cam required (2026-09-13) before landing the flint/clay/vine spawn
fix: "a persistent 0% rate after the fix is uninterpretable... only 'wrong
materials' is evidence about the communication channel; 'too few crafters' and
'nobody carrying anything' are logistics."

Tests against the real function jax_sim/main_jax.py calls, not a reimplementation.
Covers all four outcomes with small, hand-constructed populations: success,
futile_uncoordinated (not enough co-located crafters), futile_empty (co-located
crafters carrying nothing), futile_wrong_mats (enough crafters, carrying
something, but not what the recipe needs) -- plus the exhaustive/mutually-
exclusive partition property itself.
"""

import jax.numpy as jnp

from jax_sim.grid_jax import resolve_crafting

GRID_SIZE = 128


def _inv(n, wood=None, stone=None, flint=None, clay=None, vine=None):
    def _col(x):
        return jnp.zeros((n,), dtype=jnp.int32) if x is None else jnp.array(x, dtype=jnp.int32)
    return _col(wood), _col(stone), _col(flint), _col(clay), _col(vine)


def test_success_two_agents_co_located_with_complementary_materials():
    # Recipe needs 1 wood + 1 stone. Two adjacent agents, one holding each.
    positions = jnp.array([[10, 10], [10, 11]])
    is_craft = jnp.array([True, True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(
        2,
        wood=[1, 0], stone=[0, 1],
    )
    recipe = jnp.array([1, 1, 0, 0, 0], dtype=jnp.int32)
    out = resolve_crafting(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        recipe, GRID_SIZE,
    )
    assert bool(jnp.all(out["craft_success"])), "co-located complementary pair should succeed"
    assert not bool(jnp.any(out["futile_uncoordinated"]))
    assert not bool(jnp.any(out["futile_wrong_mats"]))
    assert not bool(jnp.any(out["futile_empty"]))


def test_futile_uncoordinated_too_few_crafters():
    # Recipe needs 2 wood + 1 stone (total 3 units) but only 2 agents are crafting.
    positions = jnp.array([[10, 10], [10, 11]])
    is_craft = jnp.array([True, True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(
        2,
        wood=[1, 1], stone=[0, 0],
    )
    recipe = jnp.array([2, 1, 0, 0, 0], dtype=jnp.int32)  # total_req = 3 > group_size = 2
    out = resolve_crafting(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        recipe, GRID_SIZE,
    )
    assert bool(jnp.all(out["futile_uncoordinated"])), (
        "group smaller than the recipe's total unit count must be futile_uncoordinated"
    )
    assert not bool(jnp.any(out["craft_success"]))
    assert not bool(jnp.any(out["futile_wrong_mats"]))
    assert not bool(jnp.any(out["futile_empty"]))


def test_futile_empty_enough_crafters_none_carrying():
    # Recipe needs 1 wood. Two adjacent crafters, both empty-handed.
    positions = jnp.array([[10, 10], [10, 11]])
    is_craft = jnp.array([True, True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(2, wood=[0, 0])
    recipe = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    out = resolve_crafting(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        recipe, GRID_SIZE,
    )
    assert bool(jnp.all(out["futile_empty"])), (
        "enough crafters but none carrying anything must be futile_empty"
    )
    assert not bool(jnp.any(out["craft_success"]))
    assert not bool(jnp.any(out["futile_uncoordinated"]))
    assert not bool(jnp.any(out["futile_wrong_mats"]))


def test_futile_wrong_mats_enough_crafters_carrying_but_mismatched():
    # Recipe needs 1 flint. Two adjacent crafters, both carrying wood (not flint).
    positions = jnp.array([[10, 10], [10, 11]])
    is_craft = jnp.array([True, True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(2, wood=[1, 1])
    recipe = jnp.array([0, 0, 1, 0, 0], dtype=jnp.int32)  # needs flint, they have wood
    out = resolve_crafting(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        recipe, GRID_SIZE,
    )
    assert bool(jnp.all(out["futile_wrong_mats"])), (
        "enough carrying crafters but the wrong materials must be futile_wrong_mats -- "
        "this is the only category that's evidence about the communication channel"
    )
    assert not bool(jnp.any(out["craft_success"]))
    assert not bool(jnp.any(out["futile_uncoordinated"]))
    assert not bool(jnp.any(out["futile_empty"]))


def test_non_adjacent_craft_attempts_are_uncoordinated_not_wrong_mats():
    # Two agents far apart, each crafting alone, recipe needs 2 units -- each
    # agent's own craft_group is just itself (size 1 < total_req 2).
    positions = jnp.array([[10, 10], [100, 100]])
    is_craft = jnp.array([True, True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(2, wood=[1, 1])
    recipe = jnp.array([2, 0, 0, 0, 0], dtype=jnp.int32)
    out = resolve_crafting(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        recipe, GRID_SIZE,
    )
    assert bool(jnp.all(out["futile_uncoordinated"]))


def test_categories_are_mutually_exclusive_and_exhaustive_over_random_population():
    """Structural property: for every crafting agent, exactly one of
    craft_success / futile_uncoordinated / futile_wrong_mats / futile_empty holds,
    and non-crafting agents get none of the four."""
    import jax
    rng = jax.random.PRNGKey(0)
    N = 40
    k1, k2, k3 = jax.random.split(rng, 3)
    positions = jax.random.randint(k1, (N, 2), 0, GRID_SIZE)
    is_craft = jax.random.bernoulli(k2, 0.5, (N,))
    inv_keys = jax.random.split(k3, 5)
    # Random single-material inventories (respecting capacity=1) plus some empty.
    mats = jax.random.randint(inv_keys[0], (N,), 0, 6)  # 0-4 = material, 5 = empty
    inv_wood = (mats == 0).astype(jnp.int32)
    inv_stone = (mats == 1).astype(jnp.int32)
    inv_flint = (mats == 2).astype(jnp.int32)
    inv_clay = (mats == 3).astype(jnp.int32)
    inv_vine = (mats == 4).astype(jnp.int32)
    recipe = jnp.array([1, 1, 1, 0, 0], dtype=jnp.int32)

    out = resolve_crafting(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        recipe, GRID_SIZE,
    )
    total = (
        out["craft_success"].astype(jnp.int32)
        + out["futile_uncoordinated"].astype(jnp.int32)
        + out["futile_wrong_mats"].astype(jnp.int32)
        + out["futile_empty"].astype(jnp.int32)
    )
    assert bool(jnp.all(total[is_craft] == 1)), "every crafting agent must land in exactly one category"
    assert bool(jnp.all(total[~is_craft] == 0)), "non-crafting agents must land in none"


if __name__ == "__main__":
    test_success_two_agents_co_located_with_complementary_materials()
    test_futile_uncoordinated_too_few_crafters()
    test_futile_empty_enough_crafters_none_carrying()
    test_futile_wrong_mats_enough_crafters_carrying_but_mismatched()
    test_non_adjacent_craft_attempts_are_uncoordinated_not_wrong_mats()
    test_categories_are_mutually_exclusive_and_exhaustive_over_random_population()
    print("OK: resolve_crafting() classifies success and all three failure causes "
          "correctly, mutually exclusively, and exhaustively.")
