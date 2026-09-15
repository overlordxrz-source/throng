"""Cam's fix (2026-09-15) for the actual cause of both checkpoint losses that
night: `jax_sim/main_jax.py`'s startup guard (right after the resume target
resolves, before the first rollout) that refuses to launch into a checkpoint
directory holding any step numerically ahead of the resume target. Orbax's
`max_to_keep` retention prunes by step number, not save recency, so a
directory holding a fossil from an abandoned lineage silently deletes every
new checkpoint a lower-numbered (deliberately rolled-back) resume saves --
undetectable downstream, since `[CKPT] Saved` / `[CKPT] Committed` print
unconditionally regardless of whether the save survived retention.

This exercises the guard's actual logic (identical to the block in
`jax_sim/main_jax.py`, `ckpt_mngr.all_steps()` compared against the resume
target) against a real, local Orbax CheckpointManager -- not a full
simulation run, which the guard sits inside of and which is expensive to
construct just to test one precondition check.
"""

import tempfile

import jax.numpy as jnp
import orbax.checkpoint as ocp

from jax_sim.main_jax import find_fossil_checkpoints


def _make_manager(tmp_dir, steps):
    options = ocp.CheckpointManagerOptions(max_to_keep=10, create=True)
    mngr = ocp.CheckpointManager(tmp_dir, ocp.StandardCheckpointer(), options=options)
    for step in steps:
        mngr.save(step, items={"dummy": jnp.array([float(step)])})
        mngr.wait_until_finished()
    return mngr


def test_guard_fires_when_a_fossil_sits_ahead_of_a_rolled_back_resume():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Mirrors the real incident: an abandoned lineage left 2859/2862-style
        # higher steps in the directory; the live run rolls back to a lower one.
        mngr = _make_manager(tmp_dir, steps=[10, 20, 30])
        resume_target = 20  # deliberate rollback, 30 is the fossil
        fossils = find_fossil_checkpoints(mngr, resume_target)
        assert fossils == [30], (
            f"expected the guard to identify step 30 as a fossil ahead of "
            f"resume_target=20, got {fossils!r}"
        )


def test_guard_stays_quiet_on_a_clean_forward_resume():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mngr = _make_manager(tmp_dir, steps=[10, 20, 30])
        resume_target = 30  # latest -- nothing ahead of it
        fossils = find_fossil_checkpoints(mngr, resume_target)
        assert fossils == [], (
            f"guard flagged fossils on a clean resume to the latest step: {fossils!r}"
        )


def test_guard_stays_quiet_on_a_fresh_single_lineage_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # The actual fix in production: checkpoints_r2541/ seeded with only 2541.
        mngr = _make_manager(tmp_dir, steps=[2541])
        resume_target = 2541
        fossils = find_fossil_checkpoints(mngr, resume_target)
        assert fossils == [], (
            f"guard flagged a fossil in a single-checkpoint directory: {fossils!r}"
        )


def test_a_standing_resume_pin_would_fight_the_guard_on_its_own_next_restart():
    """Documents the exact hazard Cam caught (2026-09-15) and the reason
    resume_from_step was removed from config.yaml rather than left standing
    at 2541: a permanent pin and this guard actively contradict each other.
    Simulates one full cycle -- launch pinned to an old step, save real
    forward progress, restart still pinned to the same old step -- and
    shows the guard (correctly) refuses, reading the run's own progress as
    a fossil. This is not a bug in the guard: it is proof that a standing
    pin is the bug, and the fix is removing the pin (checkpoint_dir starts
    single-lineage instead), not weakening the guard."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # First launch: only 2541 present, pinned to 2541 -- fine, matches "latest."
        mngr = _make_manager(tmp_dir, steps=[2541])
        pinned_resume_target = 2541
        assert find_fossil_checkpoints(mngr, pinned_resume_target) == []

        # The run saves real forward progress: 2542.
        mngr.save(2542, items={"dummy": jnp.array([2542.0])})
        mngr.wait_until_finished()

        # Restart, with the pin left standing at 2541 (config.yaml unchanged,
        # the exact mistake this test exists to rule out): the guard now sees
        # 2542 -- the run's OWN real progress -- as a fossil ahead of the
        # (stale) pinned target, and correctly refuses.
        fossils_with_stale_pin = find_fossil_checkpoints(mngr, pinned_resume_target)
        assert fossils_with_stale_pin == [2542], (
            f"expected a standing pin to make the guard flag the run's own "
            f"progress (2542) as a fossil, got {fossils_with_stale_pin!r} -- "
            f"if this is empty, the guard is no longer enforcing the invariant "
            f"and a stale pin could silently roll back real progress instead."
        )

        # With the pin removed (this session's actual fix), the resume target
        # tracks the true latest checkpoint instead, and the guard is quiet.
        true_latest_target = mngr.latest_step()
        assert true_latest_target == 2542
        assert find_fossil_checkpoints(mngr, true_latest_target) == []


if __name__ == "__main__":
    test_guard_fires_when_a_fossil_sits_ahead_of_a_rolled_back_resume()
    test_guard_stays_quiet_on_a_clean_forward_resume()
    test_guard_stays_quiet_on_a_fresh_single_lineage_directory()
    test_a_standing_resume_pin_would_fight_the_guard_on_its_own_next_restart()
    print(
        "OK: the fossil guard correctly flags a higher-numbered checkpoint left over "
        "from an abandoned lineage when resuming below it (the exact mechanism that "
        "silently lost two real launches on 2026-09-15), stays quiet on a clean "
        "forward resume to the latest step, stays quiet on the new "
        "checkpoints_r2541/-style single-checkpoint directory this fix actually uses, "
        "and -- the reason resume_from_step was removed from config.yaml rather than "
        "left pointed at 2541 -- correctly refuses on a run's own next restart if a "
        "pin is left standing instead of tracking the true latest checkpoint."
    )
