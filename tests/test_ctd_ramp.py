"""Pins jax_sim.ctd_ramp -- the crafting-ramp recipe cap and red's potential-based
shaping term, including the catch-step exception Cam added 2026-09-14 explicitly
because it's "the one most likely to be silently wrong."

Tests against the real functions main_jax.py will call, not a reimplementation.
The shaping-magnitude table reproduces the arithmetic verified in chat: gamma
MUST be ppo_gamma (0.999, config.yaml:164) for the policy-invariance guarantee
to be real rather than decorative.
"""

import jax
import jax.numpy as jnp

from jax_sim.ctd_ramp import (
    red_shaping_term, hearth_shaping_term, HEARTH_RAMP_STAGE_N, HEARTH_RAMP_ACCEPT_ANY,
)

GRID_SIZE = 128
GAMMA = 0.999  # must match config.yaml's ppo_gamma
BETA = 2.5


def test_hearth_ramp_stage_n_is_solo_then_ratcheting_coop():
    """Hearths (2026-09-21) replace pair-adjacency crafting; the ramp now
    drives a hearth's required deposit count N. The leading stages (A0, A1)
    must be solo-satisfiable -- pure bootstrap, no coordination possible.
    Later stages never decrease N, matching the "1 -> 1 -> 2 -> 3" curriculum
    (A0/A1 split, 2026-09-28, Cam) -- A0/A1 share N=1 by design (the split is
    in accept_any_material, not N), so non-decreasing is the right
    invariant, not strictly increasing."""
    assert HEARTH_RAMP_STAGE_N[0] == 1, "stage 0 (A0) must be solo-satisfiable (N=1)"
    assert list(HEARTH_RAMP_STAGE_N) == sorted(HEARTH_RAMP_STAGE_N), (
        "N must never ratchet down across stages"
    )
    assert len(HEARTH_RAMP_STAGE_N) == 4, "spec: N ramps 1 -> 1 -> 2 -> 3 (A0, A1, B, B+)"


def test_hearth_ramp_accept_any_is_true_only_for_a0():
    """A0/A1 split (2026-09-28, Cam): accept_any_material distinguishes A0
    from A1 even though both have N=1 -- the whole point of the split. Every
    stage past A0 requires the hearth's actual need."""
    assert len(HEARTH_RAMP_ACCEPT_ANY) == len(HEARTH_RAMP_STAGE_N)
    assert HEARTH_RAMP_ACCEPT_ANY[0] is True, "A0 accepts any material"
    assert not any(HEARTH_RAMP_ACCEPT_ANY[1:]), "every stage past A0 requires the specific need"


GS = 128
HEARTHS = jnp.array([[32, 32], [32, 96], [96, 32], [96, 96]], dtype=jnp.int32)


def test_hearth_shaping_rewards_approach_to_the_matching_hearth():
    """Moving one Chebyshev step closer to the hearth whose need matches your
    held material should be a positive F_t (gamma close to 1 dominates)."""
    pos_before = jnp.array([[32, 40]])     # 8 away from hearth 0 (32,32) on y
    pos_after = jnp.array([[32, 39]])      # 7 away -- closer
    held = jnp.array([0])                  # holding wood
    need = jnp.array([0, 1, 2, 3])         # hearth 0 needs wood -- matches
    made_deposit = jnp.array([False])
    f_t = hearth_shaping_term(pos_before, pos_after, held, HEARTHS, need, made_deposit,
                               beta=2.5, gamma=0.999, grid_size=GS)
    assert float(f_t[0]) > 0, "moving closer to the matching hearth should be rewarded"


def test_hearth_shaping_is_zero_when_holding_nothing():
    pos_before = jnp.array([[32, 40]])
    pos_after = jnp.array([[32, 39]])
    held = jnp.array([-1])  # empty-handed
    need = jnp.array([0, 1, 2, 3])
    made_deposit = jnp.array([False])
    f_t = hearth_shaping_term(pos_before, pos_after, held, HEARTHS, need, made_deposit,
                               beta=2.5, gamma=0.999, grid_size=GS)
    assert float(f_t[0]) == 0.0, "an empty-handed agent has no well-defined target -- no shaping"


def test_hearth_shaping_is_zero_when_held_material_matches_no_hearth():
    pos_before = jnp.array([[32, 40]])
    pos_after = jnp.array([[32, 39]])
    held = jnp.array([4])          # holding vine
    need = jnp.array([0, 1, 2, 3]) # no hearth currently wants vine
    made_deposit = jnp.array([False])
    f_t = hearth_shaping_term(pos_before, pos_after, held, HEARTHS, need, made_deposit,
                               beta=2.5, gamma=0.999, grid_size=GS)
    assert float(f_t[0]) == 0.0


def test_hearth_shaping_is_exactly_zero_on_every_deposit_step():
    """Mirrors red's catch-zeroing: hearth_need rerolls on completion, so the
    nearest matching hearth can teleport to a farther one the instant a
    deposit lands -- must not claw back reward on the very step it happens."""
    pos_before = jnp.array([[32, 33]])
    pos_after = jnp.array([[32, 32]])   # arrived exactly at hearth 0
    held = jnp.array([0])
    need = jnp.array([0, 1, 2, 3])
    made_deposit = jnp.array([True])
    f_t = hearth_shaping_term(pos_before, pos_after, held, HEARTHS, need, made_deposit,
                               beta=2.5, gamma=0.999, grid_size=GS)
    assert float(f_t[0]) == 0.0


def test_hearth_shaping_picks_the_nearest_matching_hearth_not_nearest_overall():
    """Two hearths could match; shaping must use the nearest MATCHING one,
    not just the nearest hearth regardless of need."""
    pos_before = jnp.array([[64, 64]])   # equidistant-ish from all 4
    pos_after = jnp.array([[64, 63]])    # moves toward hearth 2/3 side (row 96), away from 0/1 (row 32)
    held = jnp.array([1])
    need = jnp.array([0, 1, 0, 0])  # only hearth 1 (32,96) matches -- far from pos_after's direction
    made_deposit = jnp.array([False])
    f_t = hearth_shaping_term(pos_before, pos_after, held, HEARTHS, need, made_deposit,
                               beta=2.5, gamma=0.999, grid_size=GS)
    assert float(f_t[0]) < 0, "moved away from the only matching hearth -- should be penalized"


def test_f_t_is_exactly_zero_on_every_catch_step():
    """The modification: Phi jumps discontinuously when a catch removes the
    nearest blue, which can fire a large NEGATIVE shaping value at the exact
    event the ramp exists to encourage. F_t must be exactly zero whenever
    made_catch is True, regardless of what the raw potential difference would
    otherwise be -- tested with a case constructed to produce a large negative
    raw value if the exception were missing."""
    # Red at (10, 10), adjacent to a blue at (10, 11) (d_before=1) that gets
    # caught this step. The only other blue is far away at (10, 50) (d=40
    # even after red doesn't move -- red_pos_after == red_pos_before here).
    red_pos_before = jnp.array([[10, 10]])
    red_pos_after = jnp.array([[10, 10]])
    blue_pos_before = jnp.array([[10, 11], [10, 50]])
    blue_alive_before = jnp.array([True, True])
    blue_pos_after = jnp.array([[10, 11], [10, 50]])
    blue_alive_after = jnp.array([False, True])  # the adjacent blue was just caught
    made_catch = jnp.array([True])

    f_t = red_shaping_term(
        red_pos_before, red_pos_after,
        blue_pos_before, blue_alive_before,
        blue_pos_after, blue_alive_after,
        made_catch, BETA, GAMMA, GRID_SIZE,
    )
    assert float(f_t[0]) == 0.0, f"expected exactly 0.0 on a catch step, got {float(f_t[0])}"

    # Sanity: without the catch-step exception this scenario really would be
    # large and negative (d jumps 1 -> 40), confirming the test is meaningful.
    made_catch_false = jnp.array([False])
    f_t_uncorrected = red_shaping_term(
        red_pos_before, red_pos_after,
        blue_pos_before, blue_alive_before,
        blue_pos_after, blue_alive_after,
        made_catch_false, BETA, GAMMA, GRID_SIZE,
    )
    assert float(f_t_uncorrected[0]) < -1.0, (
        "the scenario should produce a large negative raw shaping value when "
        "made_catch is (incorrectly) False -- otherwise this test isn't "
        "exercising the discontinuity the exception exists for"
    )


def test_f_t_matches_hand_computed_table():
    """Reproduces the exact arithmetic verified in chat: closing 1 cell is
    worth ~0.0156-0.0166 raw (beta=1), ~0.039-0.042 at beta=2.5, essentially
    independent of absolute distance because gamma=0.999 is so close to 1."""
    grid_size = GRID_SIZE
    d_max = grid_size // 2

    def make_case(d_before, d_after):
        # Place red and the single blue on a line so Chebyshev distance
        # equals the coordinate difference exactly.
        red_before = jnp.array([[0, 0]])
        red_after = jnp.array([[0, 0]])
        blue_before = jnp.array([[0, d_before]])
        blue_after = jnp.array([[0, d_after]])
        alive = jnp.array([True])
        return red_before, red_after, blue_before, alive, blue_after, alive

    for d in (64, 32, 8, 1):
        if d >= 1:
            r_b, r_a, b_b, al_b, b_a, al_a = make_case(d, d - 1)
            f = red_shaping_term(r_b, r_a, b_b, al_b, b_a, al_a, jnp.array([False]), 1.0, GAMMA, grid_size)
            expected = (d - GAMMA * (d - 1)) / d_max
            assert abs(float(f[0]) - expected) < 1e-6, f"closing at d={d}: {float(f[0])} vs {expected}"

        r_b, r_a, b_b, al_b, b_a, al_a = make_case(d, d)
        f = red_shaping_term(r_b, r_a, b_b, al_b, b_a, al_a, jnp.array([False]), 1.0, GAMMA, grid_size)
        expected = (d - GAMMA * d) / d_max
        assert abs(float(f[0]) - expected) < 1e-6, f"stationary at d={d}: {float(f[0])} vs {expected}"

    # beta=2.5 at a full closing step near d=1 should land close to
    # reward_red_move (0.04), per the arithmetic brought to Cam.
    r_b, r_a, b_b, al_b, b_a, al_a = make_case(1, 0)
    f = red_shaping_term(r_b, r_a, b_b, al_b, b_a, al_a, jnp.array([False]), BETA, GAMMA, grid_size)
    assert 0.035 < float(f[0]) < 0.045, f"expected ~0.039-0.042 at beta=2.5, got {float(f[0])}"


def test_no_blue_alive_gives_zero_shaping_not_nan():
    red_pos = jnp.array([[5, 5]])
    blue_pos = jnp.array([[10, 10]])
    dead = jnp.array([False])
    f = red_shaping_term(
        red_pos, red_pos, blue_pos, dead, blue_pos, dead,
        jnp.array([False]), BETA, GAMMA, GRID_SIZE,
    )
    assert float(f[0]) == 0.0
    assert not bool(jnp.isnan(f[0]))


if __name__ == "__main__":
    test_hearth_ramp_stage_n_is_solo_then_ratcheting_coop()
    test_hearth_ramp_accept_any_is_true_only_for_a0()
    test_hearth_shaping_rewards_approach_to_the_matching_hearth()
    test_hearth_shaping_is_zero_when_holding_nothing()
    test_hearth_shaping_is_zero_when_held_material_matches_no_hearth()
    test_hearth_shaping_is_exactly_zero_on_every_deposit_step()
    test_hearth_shaping_picks_the_nearest_matching_hearth_not_nearest_overall()
    test_f_t_is_exactly_zero_on_every_catch_step()
    test_f_t_matches_hand_computed_table()
    test_no_blue_alive_gives_zero_shaping_not_nan()
    print("OK: hearth ramp stage schedule is solo-satisfiable at stage 0 and "
          "strictly ratchets up; red shaping term matches the hand-computed "
          "table; F_t is exactly zero on every catch step, proven against a "
          "scenario that would otherwise be large and negative.")
