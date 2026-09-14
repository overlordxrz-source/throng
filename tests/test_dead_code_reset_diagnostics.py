"""Cam's three offline diagnostics (2026-09-14) for the dead-code reset finding
against the real launch (ppo 2842-2861): codes_active collapsed to 1|3|3/64,
VQ loss=2.30e-09, resets firing on nearly every update including one update
resetting 81% of total codebook capacity (51/56/48 of 64 per slot).

Cam's diagnosis: this is not a reset-logic bug. Stage 0 (solo-satisfiable
crafting) removed all communication pressure -- futile_uncoordinated=0 across
all twenty updates means coordination isn't failing, it isn't required. The
codebook is faithfully reporting that the encoder stopped making distinctions.

Diagnostic 1: is the streak logic broken -- can a code with non-zero current-
window usage be reset? Tested directly against the real function
(dead_code_reset_codebook_params), not reasoned about.

Diagnostic 2 lives in a standalone calculation (reported in chat, not a test):
with ema_decay=0.8, 5 consecutive zero-usage updates only decay usage_ema to
32.768% of its prior value, so the thousands-scale EMAs observed at reset time
imply tens-of-thousands-scale usage_ema before the unused streak began --
consistent with heavily-used codes going completely silent, not with the math
being broken.

Diagnostic 3 requires a real checkpoint + real forward pass and lives in
scripts/diag_ze_variance.py, run against the actual stopped-run checkpoint.
"""

import jax
import jax.numpy as jnp

from jax_sim.network_jax import dead_code_reset_codebook_params


def test_a_code_used_this_window_is_never_reset_regardless_of_prior_streak():
    """Diagnostic 1 (Cam): if any code with non-zero current usage is being
    reset, the streak logic is broken. Construct the sharpest possible
    adversarial case -- a code one update away from the reset threshold
    (dead_streak=4) that IS used this window -- and confirm it survives."""
    vocab_size = 64
    key = jax.random.PRNGKey(0)

    embedding = jax.random.normal(key, (vocab_size, 12))
    usage_ema = jnp.zeros((vocab_size,), dtype=jnp.float32)
    dead_streak = jnp.zeros((vocab_size,), dtype=jnp.float32)

    # Code 0: heavy prior usage, then dead_streak already at 4 (one update
    # from reset) -- but IS used this window. Must survive.
    usage_ema = usage_ema.at[0].set(5000.0)
    dead_streak = dead_streak.at[0].set(4.0)

    # Code 1: identical prior state, but NOT used this window -- the true
    # negative control. Must be reset (proves the test can actually detect
    # a reset when the streak logic says it should happen).
    usage_ema = usage_ema.at[1].set(5000.0)
    dead_streak = dead_streak.at[1].set(4.0)

    params = {
        "codebook_0": {
            "embedding": embedding,
            "usage_ema": usage_ema,
            "dead_streak": dead_streak,
        }
    }

    # token_ids for this rollout window: agent(s) using code 0 (nonzero
    # current usage), nobody using code 1.
    token_ids = jnp.array([0, 0, 0])
    z_e = jnp.zeros((3, 12))

    new_params = dead_code_reset_codebook_params(
        params, token_ids, z_e, vocab_size, jax.random.PRNGKey(1),
        codebook_key="codebook_0",
        alive_pool_frac=1.0, min_pool_frac=0.25,
        dead_streak_window=5, ema_decay=0.8,
    )

    new_streak = new_params["codebook_0"]["dead_streak"]
    new_embedding = new_params["codebook_0"]["embedding"]

    # Code 0 (used this window): streak must reset to 0, embedding untouched.
    assert float(new_streak[0]) == 0.0, (
        f"code 0 was used this window but its streak is {float(new_streak[0])}, "
        "not reset to 0 -- streak logic is broken"
    )
    assert bool(jnp.allclose(new_embedding[0], embedding[0])), (
        "code 0 was used this window but its embedding changed -- it was reset "
        "despite non-zero current usage. Cam's diagnosis would be wrong: this "
        "is a streak-logic bug, not channel atrophy."
    )

    # Code 1 (not used, streak now 4+1=5 >= window): must be reset -- proves
    # the test setup is actually capable of triggering a reset at all.
    assert float(new_streak[1]) == 0.0, "code 1 should have reset (post-reset streak returns to 0)"
    assert not bool(jnp.allclose(new_embedding[1], embedding[1])), (
        "code 1 met every condition for reset (unused this window, streak was "
        "already 4) but was not reset -- the negative control failed, meaning "
        "this test cannot actually detect a broken streak logic"
    )


def test_pool_too_thin_touches_no_code_regardless_of_streak():
    """A too-thin alive pool must leave state untouched entirely -- including
    codes that would otherwise have met the reset bar -- so a noisy small-pool
    bincount can never corrupt persistent state."""
    vocab_size = 8
    embedding = jnp.ones((vocab_size, 4))
    usage_ema = jnp.full((vocab_size,), 999.0)
    dead_streak = jnp.full((vocab_size,), 10.0)  # already well past the window

    params = {"codebook_0": {"embedding": embedding, "usage_ema": usage_ema, "dead_streak": dead_streak}}
    token_ids = jnp.array([], dtype=jnp.int32)
    z_e = jnp.zeros((0, 4))

    new_params = dead_code_reset_codebook_params(
        params, token_ids, z_e, vocab_size, jax.random.PRNGKey(2),
        codebook_key="codebook_0",
        alive_pool_frac=0.1, min_pool_frac=0.25,  # below floor
        dead_streak_window=5, ema_decay=0.8,
    )
    assert bool(jnp.array_equal(new_params["codebook_0"]["dead_streak"], dead_streak))
    assert bool(jnp.allclose(new_params["codebook_0"]["embedding"], embedding))


if __name__ == "__main__":
    test_a_code_used_this_window_is_never_reset_regardless_of_prior_streak()
    test_pool_too_thin_touches_no_code_regardless_of_streak()
    print("OK (Diagnostic 1): a code with non-zero current-window usage is "
          "never reset, regardless of prior dead_streak -- proven against the "
          "real function with an adversarial case one update from the "
          "threshold, plus a negative control proving the test can detect a "
          "real reset. The streak logic is not broken.")
