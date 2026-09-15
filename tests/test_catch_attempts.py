"""Catch-attempt instrumentation (Cam, 2026-09-14): "prove it produces catches
or prove why it can't, from instrumentation, not from reading the config."
Before this, apply_catches only ever surfaced the post-jitter outcome
(caught_b) -- there was no way to tell "red can't find blue" (zero attempts)
apart from "the catch path itself is broken" (attempts nonzero, conversions
zero) from a live rollout. catch_attempted fixes that: per-blue-agent, true
whenever at least one living red is within catch_radius of a living,
non-big-green blue this step, independent of the catch_prob jitter roll.
"""

import jax
import jax.numpy as jnp

from jax_sim.grid_jax import apply_catches


def test_attempt_fires_even_when_jitter_prevents_conversion():
    """A red adjacent to a living blue is an attempt regardless of whether
    the catch_prob roll actually converts it -- attempts and conversions
    must be able to disagree, or the instrumentation is redundant with
    caught_b."""
    b_pos = jnp.array([[5, 5]])
    b_alive = jnp.array([True])
    b_is_big_green = jnp.array([False])
    r_pos = jnp.array([[5, 5]])  # co-located: dist=0 <= catch_radius=1
    r_alive = jnp.array([True])
    r_actions = jnp.array([0])  # action irrelevant for small-blue catches

    _, caught_b, _, _, _, _, _, catch_attempted = apply_catches(
        b_pos, b_alive, b_is_big_green, r_pos, r_alive, r_actions,
        grid_size=20, step=0, coop_threshold_step=0,
        catch_radius=1, catch_prob=0.0,  # jitter forces zero conversions
        rng=jax.random.PRNGKey(0),
    )
    assert bool(catch_attempted[0]) is True, "red is adjacent to a living blue -- must count as an attempt"
    assert bool(caught_b[0]) is False, "catch_prob=0.0 must prevent conversion despite the attempt"


def test_no_attempt_when_red_out_of_range():
    b_pos = jnp.array([[5, 5]])
    b_alive = jnp.array([True])
    b_is_big_green = jnp.array([False])
    r_pos = jnp.array([[15, 15]])  # far away
    r_alive = jnp.array([True])
    r_actions = jnp.array([0])

    _, caught_b, _, _, _, _, _, catch_attempted = apply_catches(
        b_pos, b_alive, b_is_big_green, r_pos, r_alive, r_actions,
        grid_size=20, step=0, coop_threshold_step=0,
        catch_radius=1, catch_prob=1.0,
        rng=jax.random.PRNGKey(0),
    )
    assert bool(catch_attempted[0]) is False, "no red in range -- must not count as an attempt"
    assert bool(caught_b[0]) is False


def test_no_attempt_against_a_dead_blue():
    b_pos = jnp.array([[5, 5]])
    b_alive = jnp.array([False])  # already dead
    b_is_big_green = jnp.array([False])
    r_pos = jnp.array([[5, 5]])
    r_alive = jnp.array([True])
    r_actions = jnp.array([0])

    _, _, _, _, _, _, _, catch_attempted = apply_catches(
        b_pos, b_alive, b_is_big_green, r_pos, r_alive, r_actions,
        grid_size=20, step=0, coop_threshold_step=0,
        catch_radius=1, catch_prob=1.0,
        rng=jax.random.PRNGKey(0),
    )
    assert bool(catch_attempted[0]) is False, "a dead blue can't be the target of a live attempt"


if __name__ == "__main__":
    test_attempt_fires_even_when_jitter_prevents_conversion()
    test_no_attempt_when_red_out_of_range()
    test_no_attempt_against_a_dead_blue()
    print("OK: catch_attempted fires on red-blue proximity independent of the catch_prob "
          "jitter, stays false when no red is in range, and stays false against a dead blue.")
