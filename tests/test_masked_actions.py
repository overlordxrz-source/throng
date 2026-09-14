"""Pins the Sep 2026 restart pre-flight finding: Build (action 8) was masked
at red's rollout sampling but nowhere else -- not blue's rollout, not
imagination, not the (team-blind) PPO backward pass -- so blue could freely
build barriers, and red's own rollout/backward masks disagreed with each
other (a PPO log-prob-ratio corruption, mirroring AUDIT_SEP2026.md Finding 5).

barrier_sum=2938.9 after two real PPO updates on real ckpt-2763 was the
empirical evidence; this test pins the fix at the unit level so it can never
silently regress again. It asserts against `mask_disabled_actions`, the one
function jax_sim/main_jax.py (rollout, both teams), jax_sim/imagination_jax.py
(imagination), and jax_sim/rl_jax.py (PPO backward, team-blind) all actually
call -- not a reimplementation of the masking logic.

A passing test suite was not evidence before this fix landed: every existing
test passed while Build was live, because none of them checked index 8.
"""

import jax.numpy as jnp

from jax_sim.action_space import MASKED_ACTIONS, mask_disabled_actions


def test_masked_actions_constant_includes_build():
    """Sanity check on the constant itself: if this regresses to (6, 7) the
    rest of this file's assertions would pass while Build is live again."""
    assert 8 in MASKED_ACTIONS, (
        "MASKED_ACTIONS no longer includes Build(8) -- this is precisely the "
        "regression that let blue build barriers unmasked."
    )
    assert set(MASKED_ACTIONS) == {6, 7, 8}


def test_last_axis_masking_used_by_rollout_and_ppo_backward():
    """Rollout (main_jax.py, both b_action_logits and r_action_logits) and
    rl_jax.py's PPO backward pass both call mask_disabled_actions(x, axis=-1)
    on a (..., n_actions) logits tensor. One shape covers both teams since
    the function is team-blind by construction."""
    n_actions = 12
    N = 16
    logits = jnp.zeros((N, n_actions)) + 5.0  # nonzero so a bug can't hide as "already zero"
    masked = mask_disabled_actions(logits, axis=-1)

    for a in MASKED_ACTIONS:
        assert bool(jnp.all(masked[:, a] == -1e9)), (
            f"action {a} not masked to -1e9 along the last axis -- this is "
            "the exact tensor shape/axis rollout and the PPO backward pass use"
        )
    # Everything else must be untouched.
    for a in range(n_actions):
        if a not in MASKED_ACTIONS:
            assert bool(jnp.all(masked[:, a] == 5.0)), f"action {a} was masked but should not be"


def test_first_axis_masking_used_by_imagination():
    """imagination_jax.py's `scores` tensor is (n_actions, N) -- action axis
    first, the opposite convention from rollout/PPO logits."""
    n_actions = 12
    N = 16
    scores = jnp.zeros((n_actions, N)) + 5.0
    masked = mask_disabled_actions(scores, axis=0)

    for a in MASKED_ACTIONS:
        assert bool(jnp.all(masked[a, :] == -1e9)), (
            f"action {a} not masked to -1e9 along the first axis -- this is "
            "the exact tensor shape/axis imagination_jax.py uses"
        )
    for a in range(n_actions):
        if a not in MASKED_ACTIONS:
            assert bool(jnp.all(masked[a, :] == 5.0)), f"action {a} was masked but should not be"


def test_small_legacy_action_space_is_a_no_op_not_a_crash():
    """A legacy 5-action config (n_actions < 9) must not index out of bounds --
    mask_disabled_actions should no-op rather than raise."""
    logits = jnp.zeros((4, 5))
    masked = mask_disabled_actions(logits, axis=-1)
    assert jnp.array_equal(masked, logits)


if __name__ == "__main__":
    test_masked_actions_constant_includes_build()
    test_last_axis_masking_used_by_rollout_and_ppo_backward()
    test_first_axis_masking_used_by_imagination()
    test_small_legacy_action_space_is_a_no_op_not_a_crash()
    print("OK: Build(8) -- and Push(6)/Guard(7) -- are masked identically at "
          "both tensor conventions rollout/PPO-backward and imagination use, "
          "for both teams, off the single MASKED_ACTIONS constant.")
