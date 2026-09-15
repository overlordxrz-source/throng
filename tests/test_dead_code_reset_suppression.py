"""Dead-code reset arming delay (Cam, 2026-09-14): codes grafted fresh onto a
resumed checkpoint start with usage_ema=dead_streak=0, but updates 1-7 after a
resume are exactly when usage is LEAST stable (measured on this project's own
run: slot1 sampled 14|13|13|30|52 while still settling). A code transiently
unused during that window hits dead_streak_window at update 5-6 and gets
reset for a "death" that's really just settling noise -- and the reset
perturbs the codebook, stranding more codes, resetting more. This is a
cleaner explanation for the original collapse (100-157 resets/update, 1|3|3
within 20 updates) than gradual stage-0 atrophy.

Fix: dead_code_reset_codebook_params(..., suppress_reset=True) keeps
usage_ema/dead_streak updating normally (so the measurement stays live) but
skips the embedding reset itself, logging what WOULD have reset instead.
"""

import jax
import jax.numpy as jnp

from jax_sim.network_jax import dead_code_reset_codebook_params


def _base_params(vocab_size=8):
    embedding = jnp.arange(vocab_size * 4, dtype=jnp.float32).reshape(vocab_size, 4)
    usage_ema = jnp.zeros((vocab_size,), dtype=jnp.float32)
    dead_streak = jnp.full((vocab_size,), 4.0)  # one update from the threshold
    return {"codebook_0": {"embedding": embedding, "usage_ema": usage_ema, "dead_streak": dead_streak}}


def test_suppressed_reset_leaves_embedding_untouched_but_keeps_streak_live():
    vocab_size = 8
    params = _base_params(vocab_size)
    embedding_before = params["codebook_0"]["embedding"]

    # Nobody uses code 0 this window -- dead_streak was 4, crosses the
    # window=5 threshold this update.
    token_ids = jnp.array([1, 2, 3])
    z_e = jnp.ones((3, 4))

    new_params = dead_code_reset_codebook_params(
        params, token_ids, z_e, vocab_size, jax.random.PRNGKey(0),
        codebook_key="codebook_0",
        alive_pool_frac=1.0, min_pool_frac=0.25,
        dead_streak_window=5, ema_decay=0.8,
        suppress_reset=True,
    )

    # Embedding must be bitwise untouched -- no reset happened.
    assert bool(jnp.array_equal(new_params["codebook_0"]["embedding"], embedding_before)), (
        "suppress_reset=True must not touch the embedding even when a code "
        "crosses the dead_streak threshold"
    )
    # dead_streak must keep incrementing past the window (NOT reset to 0) --
    # this is what makes the measurement live: if it silently reset to 0 the
    # same as a real reset would, we couldn't tell suppressed resets apart
    # from real ones in the streak history.
    assert float(new_params["codebook_0"]["dead_streak"][0]) == 5.0, (
        "dead_streak must keep climbing under suppression, not reset to 0 -- "
        "otherwise the counter can't distinguish 'would have reset' from "
        "'actually reset'"
    )


def test_suppressed_reset_still_updates_usage_ema_for_used_codes():
    """Codes that WERE used this window must still get their usage_ema/
    dead_streak updated normally under suppression -- only the reset action
    itself is gated, not the underlying bookkeeping."""
    vocab_size = 8
    params = _base_params(vocab_size)

    token_ids = jnp.array([2, 2, 2])  # code 2 used heavily this window
    z_e = jnp.ones((3, 4))

    new_params = dead_code_reset_codebook_params(
        params, token_ids, z_e, vocab_size, jax.random.PRNGKey(0),
        codebook_key="codebook_0",
        alive_pool_frac=1.0, min_pool_frac=0.25,
        dead_streak_window=5, ema_decay=0.8,
        suppress_reset=True,
    )
    assert float(new_params["codebook_0"]["dead_streak"][2]) == 0.0, (
        "a code used this window must have its streak reset to 0 regardless "
        "of suppress_reset -- that bookkeeping is independent of whether the "
        "embedding reset itself is armed"
    )
    assert float(new_params["codebook_0"]["usage_ema"][2]) > 0.0, (
        "usage_ema must keep accumulating for used codes under suppression"
    )


def test_default_behavior_unchanged_when_suppress_reset_not_passed():
    """suppress_reset defaults to False -- every pre-existing call site (not
    yet updated to pass it) must behave exactly as before."""
    vocab_size = 8
    params = _base_params(vocab_size)
    embedding_before = params["codebook_0"]["embedding"]

    token_ids = jnp.array([1, 2, 3])
    z_e = jnp.ones((3, 4))

    new_params = dead_code_reset_codebook_params(
        params, token_ids, z_e, vocab_size, jax.random.PRNGKey(0),
        codebook_key="codebook_0",
        alive_pool_frac=1.0, min_pool_frac=0.25,
        dead_streak_window=5, ema_decay=0.8,
    )
    assert not bool(jnp.array_equal(new_params["codebook_0"]["embedding"], embedding_before)), (
        "without suppress_reset, code 0 (dead_streak crosses threshold this "
        "update) must actually reset, exactly as before this change"
    )
    assert float(new_params["codebook_0"]["dead_streak"][0]) == 0.0, (
        "a real reset must zero dead_streak (the existing grace-period "
        "behavior), unlike the suppressed case"
    )


if __name__ == "__main__":
    test_suppressed_reset_leaves_embedding_untouched_but_keeps_streak_live()
    test_suppressed_reset_still_updates_usage_ema_for_used_codes()
    test_default_behavior_unchanged_when_suppress_reset_not_passed()
    print(
        "OK: suppress_reset=True leaves the embedding untouched while dead_streak "
        "keeps climbing past the window (measurement stays live); used codes still "
        "get normal bookkeeping; suppress_reset=False (the default, every "
        "pre-existing call site) behaves exactly as before this change."
    )
