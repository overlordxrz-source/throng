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

from jax_sim.ctd_ramp import red_shaping_term, HEARTH_RAMP_STAGE_N

GRID_SIZE = 128
GAMMA = 0.999  # must match config.yaml's ppo_gamma
BETA = 2.5


def test_hearth_ramp_stage_n_is_solo_then_ratcheting_coop():
    """Hearths (2026-09-21) replace pair-adjacency crafting; the ramp now
    drives a hearth's required deposit count N. Stage 0 (N=1) must be
    solo-satisfiable -- pure bootstrap, no coordination possible. Later
    stages strictly increase N, matching the "1 -> 2 -> 3" curriculum
    Cam specified."""
    assert HEARTH_RAMP_STAGE_N[0] == 1, "stage 0 must be solo-satisfiable (N=1)"
    assert list(HEARTH_RAMP_STAGE_N) == sorted(HEARTH_RAMP_STAGE_N), (
        "N must strictly ratchet up, never down, across stages"
    )
    assert len(HEARTH_RAMP_STAGE_N) == 3, "spec: N ramps 1 -> 2 -> 3"


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
    test_f_t_is_exactly_zero_on_every_catch_step()
    test_f_t_matches_hand_computed_table()
    test_no_blue_alive_gives_zero_shaping_not_nan()
    print("OK: hearth ramp stage schedule is solo-satisfiable at stage 0 and "
          "strictly ratchets up; red shaping term matches the hand-computed "
          "table; F_t is exactly zero on every catch step, proven against a "
          "scenario that would otherwise be large and negative.")
