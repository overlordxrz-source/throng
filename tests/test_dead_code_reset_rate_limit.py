"""Cam's fix (2026-09-15) for the limit cycle measured on the 2026-09-20
relaunch: codes_active went 45|51|47 -> 15|13|16 -> mass reset of 31/33/19
codes at once -> overshoot to 47-59 (the reset reseeds from current z_e
samples, landing squarely in the encoder's occupied region) -> a small drift
then stranded a large fraction of the reseeded codes simultaneously -> 8|10|14,
tripwire HALT. Not decay (decay is monotone); a shock discharged all at once,
then recharged, then discharged again at higher amplitude -- structurally the
same error as the pre-warmup Adam spring on the comms unfreeze: an unbounded
correction applied in a single update.

Fix: dead_code_reset_codebook_params rate-limits to max_reset_per_update (2)
codes per codebook per update, longest dead_streak first; everything else
stays queued (its dead_streak keeps accumulating rather than being zeroed, so
it's first in line next update). Verified here against the real function:
more than max_reset_per_update codes eligible in one update only resets the
longest-streak ones and leaves the rest queued with their streak intact (not
reset to a fresh grace period), and the queue drains at a controlled rate
across repeated calls rather than reproducing the old one-shot mass reset.
"""

import jax
import jax.numpy as jnp

from jax_sim.network_jax import dead_code_reset_codebook_params


def test_more_than_the_limit_eligible_only_resets_the_limit_longest_streaks():
    vocab_size = 16
    embedding = jax.random.normal(jax.random.PRNGKey(0), (vocab_size, 4))
    usage_ema = jnp.zeros((vocab_size,), dtype=jnp.float32)
    # Codes 0-4 all already past the window (eligible this update), with
    # DISTINCT dead_streak values so the longest-streak-first rule is
    # unambiguous: code 4 has been dead longest, code 0 least long.
    dead_streak = jnp.zeros((vocab_size,), dtype=jnp.float32)
    dead_streak = dead_streak.at[0].set(5.0)
    dead_streak = dead_streak.at[1].set(6.0)
    dead_streak = dead_streak.at[2].set(7.0)
    dead_streak = dead_streak.at[3].set(8.0)
    dead_streak = dead_streak.at[4].set(9.0)  # longest -- must be selected

    params = {"codebook_0": {"embedding": embedding, "usage_ema": usage_ema, "dead_streak": dead_streak}}
    # Nobody uses codes 0-4 this window; some other code (5) is used so the
    # pool isn't empty.
    token_ids = jnp.array([5, 5, 5])
    z_e = jnp.ones((3, 4))

    new_params = dead_code_reset_codebook_params(
        params, token_ids, z_e, vocab_size, jax.random.PRNGKey(1),
        codebook_key="codebook_0",
        alive_pool_frac=1.0, min_pool_frac=0.25,
        dead_streak_window=5, ema_decay=0.8,
        max_reset_per_update=2,
    )
    new_embedding = new_params["codebook_0"]["embedding"]
    new_streak = new_params["codebook_0"]["dead_streak"]

    # Codes 3 and 4 (the two longest streaks, 8 and 9) must be reset.
    for idx in (3, 4):
        assert not bool(jnp.allclose(new_embedding[idx], embedding[idx])), (
            f"code {idx} has one of the two longest dead streaks among 5 eligible "
            f"codes but was not reset -- longest-streak-first selection is broken"
        )
        assert float(new_streak[idx]) == 0.0, f"code {idx} was reset but its streak wasn't zeroed"

    # Codes 0, 1, 2 (shorter streaks, eligible but not selected) must stay
    # queued: embedding untouched, AND streak keeps its accumulated value
    # (not zeroed) so they're first in line next update.
    for idx, expected_streak in ((0, 6.0), (1, 7.0), (2, 8.0)):
        assert bool(jnp.allclose(new_embedding[idx], embedding[idx])), (
            f"code {idx} was reset despite max_reset_per_update=2 and two codes "
            f"having longer streaks -- the rate limit isn't limiting anything"
        )
        assert float(new_streak[idx]) == expected_streak, (
            f"code {idx} is queued (not reset) so its streak should keep "
            f"accumulating to {expected_streak}, got {float(new_streak[idx])} -- "
            f"a queued code must not lose its place in line"
        )


def test_queue_drains_at_the_rate_limit_across_repeated_calls():
    """5 codes eligible, limit 2/update: after one call, 2 reset and 3 remain
    queued with their streak intact; a second call (nobody using any of them
    still) must reset 2 MORE of the still-queued ones -- not re-trigger a
    fresh 5-update grace period, and not dump the whole queue at once."""
    vocab_size = 16
    embedding = jax.random.normal(jax.random.PRNGKey(2), (vocab_size, 4))
    usage_ema = jnp.zeros((vocab_size,), dtype=jnp.float32)
    dead_streak = jnp.zeros((vocab_size,), dtype=jnp.float32)
    for i, streak in zip(range(5), [5.0, 6.0, 7.0, 8.0, 9.0]):
        dead_streak = dead_streak.at[i].set(streak)

    params = {"codebook_0": {"embedding": embedding, "usage_ema": usage_ema, "dead_streak": dead_streak}}
    token_ids = jnp.array([5, 5, 5])
    z_e = jnp.ones((3, 4))

    kwargs = dict(
        codebook_key="codebook_0", alive_pool_frac=1.0, min_pool_frac=0.25,
        dead_streak_window=5, ema_decay=0.8, max_reset_per_update=2,
    )

    after_call_1 = dead_code_reset_codebook_params(
        params, token_ids, z_e, vocab_size, jax.random.PRNGKey(3), **kwargs
    )
    n_reset_1 = sum(
        1 for i in range(5)
        if not bool(jnp.allclose(after_call_1["codebook_0"]["embedding"][i], embedding[i]))
    )
    assert n_reset_1 == 2, f"first call should reset exactly 2 of 5 eligible codes, reset {n_reset_1}"

    # Second call: same token_ids (nobody uses codes 0-4 again), starting
    # from after_call_1's state. The 3 still-queued codes' streaks kept
    # accumulating (not reset), so they're still eligible and should now be
    # the ones selected -- 2 more reset, 1 left queued.
    embedding_after_1 = after_call_1["codebook_0"]["embedding"]
    after_call_2 = dead_code_reset_codebook_params(
        after_call_1, token_ids, z_e, vocab_size, jax.random.PRNGKey(4), **kwargs
    )
    n_reset_2 = sum(
        1 for i in range(5)
        if not bool(jnp.allclose(after_call_2["codebook_0"]["embedding"][i], embedding_after_1[i]))
    )
    assert n_reset_2 == 2, (
        f"second call should reset 2 more of the still-queued codes, reset {n_reset_2} -- "
        f"the queue must drain at the rate limit across calls, not stall or dump"
    )

    # Across both calls, at most 4 of the original 5 codes have been reset --
    # never all 5 at once (which would reproduce the mass-reset shock).
    total_reset = sum(
        1 for i in range(5)
        if not bool(jnp.allclose(after_call_2["codebook_0"]["embedding"][i], embedding[i]))
    )
    assert total_reset == 4, f"expected exactly 4 of 5 codes reset after two rate-limited calls, got {total_reset}"


if __name__ == "__main__":
    test_more_than_the_limit_eligible_only_resets_the_limit_longest_streaks()
    test_queue_drains_at_the_rate_limit_across_repeated_calls()
    print(
        "OK: with 5 codes eligible in one update and max_reset_per_update=2, only "
        "the 2 longest dead streaks are actually reset; the other 3 stay queued "
        "with their streak intact (not zeroed, not restarted); across repeated "
        "calls the queue drains 2 at a time rather than dumping all 5 at once -- "
        "the mass-reset shock measured on 2026-09-20 cannot reproduce."
    )
