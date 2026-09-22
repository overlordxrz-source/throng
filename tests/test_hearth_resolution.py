"""Pins jax_sim.grid_jax.resolve_hearth_deposits() -- Phase 19 hearths
(Cam's spec, 2026-09-21), which replaces Phase 18.7's pair-adjacency
resolve_crafting() entirely: a two-body same-instant simultaneity problem
becomes a one-body navigation-plus-information problem. CRAFT (action 10,
same id) on a hearth tile holding that hearth's needed material deposits;
deposits decay; reaching the requirement N completes the hearth and splits
a larger reward among whoever's (decayed) deposits are still represented in
the fill at that instant.

Tests against the real function jax_sim/main_jax.py calls, not a
reimplementation.
"""

import jax
import jax.numpy as jnp

from jax_sim.grid_jax import resolve_hearth_deposits, generate_hearth_positions

GRID_SIZE = 128
MAX_POP = 200


def _inv(n, wood=None, stone=None, flint=None, clay=None, vine=None):
    def _col(x):
        return jnp.zeros((n,), dtype=jnp.int32) if x is None else jnp.array(x, dtype=jnp.int32)
    return _col(wood), _col(stone), _col(flint), _col(clay), _col(vine)


def _fresh_hearth_state(max_pop=MAX_POP):
    hearth_positions = generate_hearth_positions(GRID_SIZE)
    hearth_need = jnp.zeros((4,), dtype=jnp.int32)  # all want "wood" (material 0)
    hearth_fill = jnp.zeros((4,), dtype=jnp.float32)
    hearth_credit = jnp.zeros((4, max_pop), dtype=jnp.float32)
    return hearth_positions, hearth_need, hearth_fill, hearth_credit


def test_hearth_positions_are_one_per_quadrant_and_deterministic():
    pos = generate_hearth_positions(GRID_SIZE)
    assert pos.shape == (4, 2)
    q = GRID_SIZE // 4
    expected = jnp.array([[q, q], [q, 3 * q], [3 * q, q], [3 * q, 3 * q]])
    assert bool(jnp.all(pos == expected))
    # deterministic: calling again gives the exact same positions (no RNG)
    assert bool(jnp.all(generate_hearth_positions(GRID_SIZE) == pos))


def test_solo_deposit_completes_a_stage0_n1_hearth():
    """N=1 must be solo-satisfiable -- the whole point of the bootstrap."""
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 4)
    positions = jnp.stack([hearth_positions[0], jnp.array([0, 0]), jnp.array([1, 1]), jnp.array([2, 2])])
    is_craft = jnp.array([True, False, False, False])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1, 0, 0, 0])

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit[:, :n],
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0),
    )
    assert bool(out["deposited"][0]), "agent on the hearth holding the needed material should deposit"
    assert bool(out["completed"][0]), "N=1 must complete from a single solo deposit"
    assert float(out["completion_share"][0, 0]) == 1.0, "sole depositor gets the entire completion share"
    assert float(out["new_hearth_fill"][0]) == 0.0, "fill resets to 0 immediately on completion"


def test_n2_requires_two_deposits_not_satisfiable_by_one():
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 4)
    positions = jnp.stack([hearth_positions[0], jnp.array([0, 0]), jnp.array([1, 1]), jnp.array([2, 2])])
    is_craft = jnp.array([True, False, False, False])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1, 0, 0, 0])

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit[:, :n],
        hearth_n_required=jnp.array(2), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0),
    )
    assert bool(out["deposited"][0]), "the deposit itself should still register"
    assert not bool(out["completed"][0]), "one deposit must not complete an N=2 hearth"
    assert float(out["new_hearth_fill"][0]) == 1.0, "fill should carry the single deposit forward"


def test_n2_completes_with_two_agents_same_step_and_splits_evenly():
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 4)
    positions = jnp.stack([hearth_positions[0], hearth_positions[0], jnp.array([1, 1]), jnp.array([2, 2])])
    is_craft = jnp.array([True, True, False, False])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1, 1, 0, 0])

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit[:, :n],
        hearth_n_required=jnp.array(2), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0),
    )
    assert bool(out["deposited"][0]) and bool(out["deposited"][1])
    assert bool(out["completed"][0])
    assert abs(float(out["completion_share"][0, 0]) - 0.5) < 1e-6
    assert abs(float(out["completion_share"][0, 1]) - 0.5) < 1e-6
    assert float(out["completion_share"][0, 2:].sum()) == 0.0, "non-contributors get no share"


def test_two_single_unit_deposits_one_step_apart_fall_just_short():
    """With any decay strictly below 1.0, two single-unit deposits spaced a
    step apart can never sum to exactly N=2 -- decay always costs a
    fractional shortfall. This is decay doing its job (Cam: "a half-filled
    hearth doesn't sit indefinitely"), not a bug: it means N>=2 effectively
    rewards prompt, near-simultaneous contribution over a slow relay, even
    though nothing enforces literal same-step adjacency the way the old
    pair-crafting mechanism did."""
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 2)
    positions = jnp.stack([hearth_positions[0], jnp.array([5, 5])])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1, 0])
    decay = jnp.array(0.99654)  # production half-life-200 decay factor

    out1 = resolve_hearth_deposits(
        positions, jnp.array([True, False]),
        inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(2), decay_factor=decay,
        reroll_key=jax.random.PRNGKey(1),
    )
    assert not bool(out1["completed"][0])
    assert float(out1["new_hearth_fill"][0]) == 1.0

    positions2 = jnp.stack([hearth_positions[0], hearth_positions[0]])
    inv_wood2, inv_stone2, inv_flint2, inv_clay2, inv_vine2 = _inv(n, wood=[0, 1])
    out2 = resolve_hearth_deposits(
        positions2, jnp.array([False, True]),
        inv_wood2, inv_stone2, inv_flint2, inv_clay2, inv_vine2,
        hearth_positions, hearth_need, out1["new_hearth_fill"], out1["new_hearth_credit"],
        hearth_n_required=jnp.array(2), decay_factor=decay,
        reroll_key=jax.random.PRNGKey(2),
    )
    assert not bool(out2["completed"][0]), (
        "one step of decay on the first deposit should leave the pair just under N=2"
    )
    assert 1.99 < float(out2["new_hearth_fill"][0]) < 2.0

    # A third, same-step deposit finishes it -- and only the two most recent
    # contributors (agent 1 from the prior step, agent 2 arriving now) still
    # have any undecayed credit worth splitting.
    positions3 = jnp.stack([hearth_positions[0], hearth_positions[0], hearth_positions[0]])
    hearth_credit3 = jnp.concatenate(
        [out2["new_hearth_credit"], jnp.zeros((4, 1), dtype=jnp.float32)], axis=1
    )
    inv_wood3, inv_stone3, inv_flint3, inv_clay3, inv_vine3 = _inv(3, wood=[0, 0, 1])
    out3 = resolve_hearth_deposits(
        positions3, jnp.array([False, False, True]),
        inv_wood3, inv_stone3, inv_flint3, inv_clay3, inv_vine3,
        hearth_positions, hearth_need, out2["new_hearth_fill"], hearth_credit3,
        hearth_n_required=jnp.array(2), decay_factor=decay,
        reroll_key=jax.random.PRNGKey(3),
    )
    assert bool(out3["completed"][0]), "a third deposit should finally cross N=2"


def test_decay_alone_can_prevent_completion():
    """A lone deposit that sits too long should rot away before a second
    deposit arrives, if enough steps of decay separate them."""
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 2)
    positions = jnp.stack([hearth_positions[0], jnp.array([5, 5])])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1, 0])
    decay = jnp.array(0.5)  # aggressive decay for a fast, legible test

    fill, credit = hearth_fill, hearth_credit
    out = None
    for step in range(20):
        is_craft = jnp.array([step == 0, step == 19])
        pos = positions if step == 0 else jnp.stack([hearth_positions[0], hearth_positions[0]])
        wood = [1, 0] if step == 0 else [0, 1]
        iw, ist, ifl, icl, ivn = _inv(n, wood=wood)
        out = resolve_hearth_deposits(
            pos, is_craft, iw, ist, ifl, icl, ivn,
            hearth_positions, hearth_need, fill, credit,
            hearth_n_required=jnp.array(2), decay_factor=decay,
            reroll_key=jax.random.PRNGKey(step),
        )
        fill, credit = out["new_hearth_fill"], out["new_hearth_credit"]
    assert not bool(out["completed"][0]), (
        "agent 0's deposit should have decayed to near-nothing after 19 steps at decay=0.5, "
        "leaving the second deposit short of N=2"
    )


def test_wrong_material_does_not_deposit():
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 1)
    positions = hearth_positions[0:1]
    is_craft = jnp.array([True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, stone=[1])  # hearth wants wood

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0),
    )
    assert not bool(out["deposited"][0])
    assert bool(out["attempted"][0]), "still counts as an attempt (a real material guess) for Gate 2"
    assert not bool(out["completed"][0])


def test_empty_handed_does_not_count_as_an_attempt():
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 1)
    positions = hearth_positions[0:1]
    is_craft = jnp.array([True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n)  # empty-handed

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0),
    )
    assert not bool(out["deposited"][0])
    assert not bool(out["attempted"][0]), "empty-handed is logistics, not a material guess"


def test_off_hearth_craft_is_neither_attempt_nor_deposit():
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 1)
    positions = jnp.array([[0, 0]])  # nowhere near a hearth
    is_craft = jnp.array([True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1])

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0),
    )
    assert not bool(out["deposited"][0])
    assert not bool(out["attempted"][0])


def test_need_rerolls_only_the_completed_hearth():
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 1)
    hearth_need = jnp.array([0, 2, 3, 4])  # distinct needs per hearth
    positions = hearth_positions[0:1]
    is_craft = jnp.array([True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1])

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(3),
    )
    assert bool(out["completed"][0])
    assert bool(jnp.all(out["new_hearth_need"][1:] == hearth_need[1:])), (
        "hearths that didn't complete this step must keep their existing need"
    )


def test_max_pop_population_shape_runs_clean():
    """Sanity check at a realistic max_pop (200) with a scattered, mostly-idle
    population -- catches shape/broadcast bugs a 1-4 agent unit test can miss."""
    n = MAX_POP
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n)
    key = jax.random.PRNGKey(42)
    k_pos, k_inv = jax.random.split(key)
    positions = jax.random.randint(k_pos, (n, 2), 0, GRID_SIZE)
    positions = positions.at[0].set(hearth_positions[0])
    is_craft = jnp.zeros((n,), dtype=bool).at[0].set(True)
    inv_wood = jnp.zeros((n,), dtype=jnp.int32).at[0].set(1)
    inv_stone, inv_flint, inv_clay, inv_vine = (jnp.zeros((n,), dtype=jnp.int32) for _ in range(4))

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99654),
        reroll_key=jax.random.PRNGKey(7),
    )
    assert out["deposited"].shape == (n,)
    assert out["completion_share"].shape == (4, n)
    assert bool(out["deposited"][0])
    assert bool(out["completed"][0])


def test_hearth_radius_widens_the_footprint_without_changing_reward_math():
    """Cam, 2026-09-22: radius=0 (default) is the single-tile hearth; a
    wider radius must accept a deposit from anywhere within Chebyshev
    distance, deposit/completion math otherwise unchanged."""
    hearth_positions, hearth_need, hearth_fill, hearth_credit = _fresh_hearth_state(n := 1)
    # 2 tiles away from hearth 0 (32,32) -- inside radius=3, outside radius=0/1.
    positions = jnp.array([[34, 34]])
    is_craft = jnp.array([True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(n, wood=[1])

    out_r0 = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0), grid_size=GRID_SIZE, hearth_radius=0,
    )
    assert not bool(out_r0["attempted"][0]), "2 tiles away must miss at radius=0"

    out_r3 = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0), grid_size=GRID_SIZE, hearth_radius=3,
    )
    assert bool(out_r3["deposited"][0]), "2 tiles away must hit at radius=3 (7x7 footprint)"
    assert bool(out_r3["completed"][0])
    assert float(out_r3["completion_share"][0, 0]) == 1.0

    out_r3_far = resolve_hearth_deposits(
        jnp.array([[38, 38]]), is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0), grid_size=GRID_SIZE, hearth_radius=3,
    )
    assert not bool(out_r3_far["attempted"][0]), "6 tiles away must still miss at radius=3"


def test_hearth_radius_toroidal_wrap():
    """The radius check must wrap the torus, same as every other distance
    check in this codebase -- an agent just past the grid edge from a
    hearth near the opposite edge should still be in range."""
    hearth_positions = jnp.array([[0, 0], [0, 96], [96, 0], [96, 96]])
    hearth_need = jnp.zeros((4,), dtype=jnp.int32)
    hearth_fill = jnp.zeros((4,), dtype=jnp.float32)
    hearth_credit = jnp.zeros((4, 1), dtype=jnp.float32)
    positions = jnp.array([[126, 2]])  # 2 tiles from (0,0) the wrapped way
    is_craft = jnp.array([True])
    inv_wood, inv_stone, inv_flint, inv_clay, inv_vine = _inv(1, wood=[1])

    out = resolve_hearth_deposits(
        positions, is_craft, inv_wood, inv_stone, inv_flint, inv_clay, inv_vine,
        hearth_positions, hearth_need, hearth_fill, hearth_credit,
        hearth_n_required=jnp.array(1), decay_factor=jnp.array(0.99),
        reroll_key=jax.random.PRNGKey(0), grid_size=GRID_SIZE, hearth_radius=3,
    )
    assert bool(out["deposited"][0]), "toroidal wrap must put this agent in range of hearth (0,0)"


if __name__ == "__main__":
    test_hearth_positions_are_one_per_quadrant_and_deterministic()
    test_solo_deposit_completes_a_stage0_n1_hearth()
    test_n2_requires_two_deposits_not_satisfiable_by_one()
    test_n2_completes_with_two_agents_same_step_and_splits_evenly()
    test_two_single_unit_deposits_one_step_apart_fall_just_short()
    test_decay_alone_can_prevent_completion()
    test_wrong_material_does_not_deposit()
    test_empty_handed_does_not_count_as_an_attempt()
    test_off_hearth_craft_is_neither_attempt_nor_deposit()
    test_need_rerolls_only_the_completed_hearth()
    test_max_pop_population_shape_runs_clean()
    test_hearth_radius_widens_the_footprint_without_changing_reward_math()
    test_hearth_radius_toroidal_wrap()
    print(
        "OK: resolve_hearth_deposits() -- solo N=1 bootstrap completes alone; N=2 needs two "
        "deposits, same-step or across steps before decay wins; decay can prevent completion; "
        "wrong-material/empty-handed/off-hearth are classified correctly; completion reward "
        "splits proportional to undecayed credit; need re-rolls only for the hearth that "
        "completed; runs clean at production population size."
    )
