"""
Task 4 (docs/WILL_RESTART_SEP2026.md): the persistent usage-EMA dead-code
reset mechanism.

The original `dead_code_reset_codebook_params` reset any code with zero
usage in a SINGLE rollout window. When predation thins the population, the
surviving token pool shrinks, most codes read as transiently dead, and the
codebook gets scrambled precisely during the event the project is trying to
measure — the reset mechanism was destroying the measurement instrument
during the event it was meant to observe (codes_active 35|51|46 -> 7|11|14
at ppo=2578).

Fix: a code resets only after `dead_streak_window` CONSECUTIVE windows with
zero usage, tracked via a persistent per-code `dead_streak` + `usage_ema`
that lives alongside `embedding` (so it survives checkpointing via the
existing graft path), and the entire update is skipped when the alive-agent
pool is too thin to support a `vocab_size`-code usage estimate.

This is a fast, direct unit test of `dead_code_reset_codebook_params` and
`ensure_vq_usage_state` — no simulation, checkpoint, or GPU needed. It also
pins the exact bug a full end-to-end smoke test caught: `dead_streak` must
be float-typed, not int, because it lives inside the same params pytree
that PPO's `jax.grad` differentiates end to end — an integer leaf anywhere
in that tree makes JAX refuse to differentiate the WHOLE tree.
"""

import jax
import jax.numpy as jnp

from jax_sim.network_jax import (
    dead_code_reset_codebook_params,
    ensure_vq_usage_state,
    VQ_CODEBOOK_KEYS,
)

VOCAB = 16
DIM = 8


def _fresh_codebook_params():
    embedding = jax.random.normal(jax.random.PRNGKey(0), (VOCAB, DIM))
    params = {"codebook_0": {"embedding": embedding}, "head_action": {"kernel": jnp.zeros((4, 4))}}
    return ensure_vq_usage_state(params, VOCAB)


def test_ensure_vq_usage_state_initializes_fresh_fields():
    params = _fresh_codebook_params()
    cb = params["codebook_0"]
    assert cb["usage_ema"].shape == (VOCAB,)
    assert cb["dead_streak"].shape == (VOCAB,)
    assert cb["usage_ema"].dtype == jnp.float32
    assert cb["dead_streak"].dtype == jnp.float32, (
        "dead_streak must be float, not int — it lives in the same params "
        "pytree PPO's jax.grad differentiates, and an int leaf anywhere in "
        "that tree makes JAX refuse to differentiate the whole tree "
        "(TypeError: grad requires real- or complex-valued inputs)"
    )
    assert bool(jnp.all(cb["usage_ema"] == 0.0))
    assert bool(jnp.all(cb["dead_streak"] == 0.0))
    assert "head_action" in params, "ensure_vq_usage_state must not touch unrelated params"


def test_params_with_usage_state_survive_jax_grad():
    """Directly reproduces the bug the smoke test caught: embedding these
    fields must not break autodiff over the rest of the params tree."""
    params = _fresh_codebook_params()

    def loss_fn(p):
        return jnp.sum(p["codebook_0"]["embedding"] ** 2) + jnp.sum(p["head_action"]["kernel"] ** 2)

    grads = jax.grad(loss_fn)(params)
    assert grads["codebook_0"]["embedding"].shape == (VOCAB, DIM)
    # Unused leaves (usage_ema/dead_streak never appear in the loss) get an
    # exact zero gradient, not an error — this is what "must not crash" means.
    assert bool(jnp.all(grads["codebook_0"]["usage_ema"] == 0.0))
    assert bool(jnp.all(grads["codebook_0"]["dead_streak"] == 0.0))


def test_code_used_every_window_never_resets():
    params = _fresh_codebook_params()
    rng = jax.random.PRNGKey(1)
    # Every code gets used every window (token_ids covers the full vocab).
    token_ids = jnp.arange(VOCAB)
    z_e = jax.random.normal(jax.random.PRNGKey(2), (VOCAB, DIM))
    original_embedding = params["codebook_0"]["embedding"]

    for w in range(10):
        rng, key = jax.random.split(rng)
        params = dead_code_reset_codebook_params(
            params, token_ids, z_e, VOCAB, key, "codebook_0",
            alive_pool_frac=1.0, min_pool_frac=0.25, dead_streak_window=5,
        )
    assert bool(jnp.all(params["codebook_0"]["dead_streak"] == 0.0))
    assert bool(jnp.array_equal(params["codebook_0"]["embedding"], original_embedding)), (
        "a code used every window must never be reset"
    )


def test_code_unused_for_fewer_than_window_updates_is_not_reset():
    params = _fresh_codebook_params()
    rng = jax.random.PRNGKey(3)
    z_e = jax.random.normal(jax.random.PRNGKey(4), (VOCAB, DIM))
    original_embedding = params["codebook_0"]["embedding"]
    # Code 0 is NEVER in the used set; everything else is, every window.
    token_ids = jnp.arange(1, VOCAB)

    for w in range(4):  # < dead_streak_window=5
        rng, key = jax.random.split(rng)
        params = dead_code_reset_codebook_params(
            params, token_ids, z_e, VOCAB, key, "codebook_0",
            alive_pool_frac=1.0, min_pool_frac=0.25, dead_streak_window=5,
        )
    assert float(params["codebook_0"]["dead_streak"][0]) == 4.0
    assert bool(jnp.array_equal(params["codebook_0"]["embedding"][0], original_embedding[0])), (
        "a code unused for fewer than dead_streak_window consecutive updates must not reset yet"
    )


def test_code_unused_for_window_consecutive_updates_resets():
    params = _fresh_codebook_params()
    rng = jax.random.PRNGKey(5)
    z_e = jax.random.normal(jax.random.PRNGKey(6), (VOCAB, DIM))
    original_embedding = params["codebook_0"]["embedding"]
    token_ids = jnp.arange(1, VOCAB)  # code 0 never used

    for w in range(5):  # == dead_streak_window
        rng, key = jax.random.split(rng)
        params = dead_code_reset_codebook_params(
            params, token_ids, z_e, VOCAB, key, "codebook_0",
            alive_pool_frac=1.0, min_pool_frac=0.25, dead_streak_window=5,
        )
    assert not bool(jnp.array_equal(params["codebook_0"]["embedding"][0], original_embedding[0])), (
        "a code unused for dead_streak_window consecutive updates must be reset"
    )
    # Reset codes get a fresh grace period, not an instant re-trigger.
    assert float(params["codebook_0"]["dead_streak"][0]) == 0.0
    assert float(params["codebook_0"]["usage_ema"][0]) == 0.0


def test_single_zero_usage_window_alone_does_not_reset():
    """The regression this whole task exists to fix: one window of zero
    usage (e.g. a predation spike thinning the surviving token pool) must
    NOT be enough to reset a code by itself."""
    params = _fresh_codebook_params()
    z_e = jax.random.normal(jax.random.PRNGKey(7), (VOCAB, DIM))
    original_embedding = params["codebook_0"]["embedding"]
    token_ids = jnp.arange(1, VOCAB)  # code 0 unused this one window

    params = dead_code_reset_codebook_params(
        params, token_ids, z_e, VOCAB, jax.random.PRNGKey(8), "codebook_0",
        alive_pool_frac=1.0, min_pool_frac=0.25, dead_streak_window=5,
    )
    assert bool(jnp.array_equal(params["codebook_0"]["embedding"][0], original_embedding[0])), (
        "a single zero-usage window must not reset a code — this is exactly "
        "the bug that scrambled the codebook in sync with predation spikes"
    )


def test_thin_alive_pool_skips_the_entire_update():
    """A window where the alive-agent pool is below the floor must leave
    usage_ema/dead_streak completely untouched, not just skip the reset —
    a noisy small-pool bincount must not corrupt the persistent state."""
    params = _fresh_codebook_params()
    z_e = jax.random.normal(jax.random.PRNGKey(9), (VOCAB, DIM))
    token_ids = jnp.arange(1, VOCAB)  # would normally bump code 0's dead_streak

    before_ema = params["codebook_0"]["usage_ema"]
    before_streak = params["codebook_0"]["dead_streak"]
    before_embedding = params["codebook_0"]["embedding"]

    params = dead_code_reset_codebook_params(
        params, token_ids, z_e, VOCAB, jax.random.PRNGKey(10), "codebook_0",
        alive_pool_frac=0.1, min_pool_frac=0.25, dead_streak_window=5,
    )
    assert bool(jnp.array_equal(params["codebook_0"]["usage_ema"], before_ema))
    assert bool(jnp.array_equal(params["codebook_0"]["dead_streak"], before_streak))
    assert bool(jnp.array_equal(params["codebook_0"]["embedding"], before_embedding))


def test_thin_pool_cannot_be_bypassed_by_accumulating_many_updates():
    """A code that's genuinely unused should NOT reset just because many
    thin-pool windows passed, since none of them should count toward the
    streak at all."""
    params = _fresh_codebook_params()
    z_e = jax.random.normal(jax.random.PRNGKey(11), (VOCAB, DIM))
    original_embedding = params["codebook_0"]["embedding"]
    token_ids = jnp.arange(1, VOCAB)
    rng = jax.random.PRNGKey(12)

    for w in range(20):  # far more than dead_streak_window, but all thin-pool
        rng, key = jax.random.split(rng)
        params = dead_code_reset_codebook_params(
            params, token_ids, z_e, VOCAB, key, "codebook_0",
            alive_pool_frac=0.1, min_pool_frac=0.25, dead_streak_window=5,
        )
    assert bool(jnp.array_equal(params["codebook_0"]["embedding"][0], original_embedding[0]))
    assert float(params["codebook_0"]["dead_streak"][0]) == 0.0


if __name__ == "__main__":
    tests = [
        test_ensure_vq_usage_state_initializes_fresh_fields,
        test_params_with_usage_state_survive_jax_grad,
        test_code_used_every_window_never_resets,
        test_code_unused_for_fewer_than_window_updates_is_not_reset,
        test_code_unused_for_window_consecutive_updates_resets,
        test_single_zero_usage_window_alone_does_not_reset,
        test_thin_alive_pool_skips_the_entire_update,
        test_thin_pool_cannot_be_bypassed_by_accumulating_many_updates,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nAll {len(tests)} dead-code usage-EMA checks passed.")
