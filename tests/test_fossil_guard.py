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


if __name__ == "__main__":
    test_guard_fires_when_a_fossil_sits_ahead_of_a_rolled_back_resume()
    test_guard_stays_quiet_on_a_clean_forward_resume()
    test_guard_stays_quiet_on_a_fresh_single_lineage_directory()
    print(
        "OK: the fossil guard correctly flags a higher-numbered checkpoint left over "
        "from an abandoned lineage when resuming below it (the exact mechanism that "
        "silently lost two real launches on 2026-09-15), stays quiet on a clean "
        "forward resume to the latest step, and stays quiet on the new "
        "checkpoints_r2541/-style single-checkpoint directory this fix actually uses."
    )
