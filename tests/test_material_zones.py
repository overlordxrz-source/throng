"""Pins the crafting spawn-zone invariant Cam approved (2026-09-13): wood/stone
split west/east, flint/clay/vine split north/mid/south, orthogonal axes so every
cell holds at most 2 of the 5 materials -- the property that makes co-location a
coordination problem rather than a solo pickup, per the explicit design argument
(if every material were everywhere, a blind agent picks up whatever it's standing
on and co-located groups assemble correct combinations by chance).

Tests against jax_sim.grid_jax.material_zone_masks() itself -- the function
jax_sim/main_jax.py's spawn code actually calls -- not a reimplementation.
"""

import jax.numpy as jnp

from jax_sim.grid_jax import material_zone_masks

GRID_SIZE = 128


def test_each_axis_is_a_proper_partition():
    zones = material_zone_masks(GRID_SIZE)
    # west/east: every cell in exactly one, never both, never neither.
    xor_wood_stone = zones["wood"] ^ zones["stone"]
    assert bool(jnp.all(xor_wood_stone)), "wood/stone zones overlap or leave a gap"

    # north/mid/south: every cell in exactly one of the three.
    counts = (
        zones["flint"].astype(jnp.int32)
        + zones["clay"].astype(jnp.int32)
        + zones["vine"].astype(jnp.int32)
    )
    assert bool(jnp.all(counts == 1)), "flint/clay/vine zones overlap or leave a gap"


def test_at_most_two_materials_per_cell():
    """The property Cam named as the one doing the work: no cell is eligible for
    more than 2 of the 5 materials, so no single region can supply a recipe that
    needs 3+ distinct materials -- solo gathering from one spot is structurally
    insufficient."""
    zones = material_zone_masks(GRID_SIZE)
    total = sum(z.astype(jnp.int32) for z in zones.values())
    assert int(jnp.max(total)) <= 2, (
        f"some cell is eligible for {int(jnp.max(total))} materials, expected <=2"
    )
    assert int(jnp.min(total)) == 2, (
        "some cell is eligible for fewer than 2 materials -- every cell should get "
        "exactly one from the west/east split and one from the north/mid/south split"
    )


def test_all_five_materials_actually_occupy_grid_area():
    """Regression pin for Rule 13 #10: flint/clay/vine zones must be non-empty,
    or the fix that spawns them there never actually places anything."""
    zones = material_zone_masks(GRID_SIZE)
    for name, mask in zones.items():
        n_cells = int(jnp.sum(mask.astype(jnp.int32)))
        assert n_cells > 0, f"{name}'s zone is empty -- it can never spawn"


if __name__ == "__main__":
    test_each_axis_is_a_proper_partition()
    test_at_most_two_materials_per_cell()
    test_all_five_materials_actually_occupy_grid_area()
    print("OK: material zones partition the grid on two orthogonal axes; every "
          "cell is eligible for exactly 2 of the 5 crafting materials.")
