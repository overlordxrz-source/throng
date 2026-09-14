"""Pins that CtD ramp progress and red curriculum state actually survive a
checkpoint round trip (Cam's correction, 2026-09-14): red_curriculum_idx/
red_sustain_count were plain Python ints, reset to zero on every process
start, never restored -- the same defect class as the rest of the Sep 2026
audit, just the mirror image of the confidence-head fossil (there, stale
state survived a reset it should have had; here, live state resets across a
boundary it should survive). Both ramp ratchets were built on that exact
unpersisted pattern before this fix.

This is a REAL round trip: state is saved to a fresh Orbax CheckpointManager
pointed at a temp directory, then read back by a SEPARATE, freshly
constructed CheckpointManager instance pointed at the same directory --
proving the actual on-disk serialization round-trips the data, not just that
a Python dict survives being kept in memory across two lines of the same
function.
"""

import shutil
import tempfile

import jax.numpy as jnp
import orbax.checkpoint as ocp


def test_training_state_survives_a_real_checkpoint_round_trip():
    tmp_dir = tempfile.mkdtemp(prefix="ctd_ramp_ckpt_test_")
    try:
        # Construct state mid-ramp: non-zero streaks, a red curriculum stage
        # already advanced past zero, one ramp already ratcheted off (active=
        # False) to prove that a *completed* ramp's history survives too, not
        # just an in-progress one.
        training_state = {
            "craft_ramp_active": jnp.array(True, dtype=jnp.bool_),
            "craft_ramp_start_step": jnp.array(1_414_656, dtype=jnp.int32),
            "craft_ramp_success_streak": jnp.array(7, dtype=jnp.int32),
            "red_ramp_active": jnp.array(False, dtype=jnp.bool_),  # already ratcheted off
            "red_ramp_start_step": jnp.array(1_414_656, dtype=jnp.int32),
            "red_ramp_catch_streak": jnp.array(10, dtype=jnp.int32),
            "red_curriculum_idx": jnp.array(2, dtype=jnp.int32),
            "red_sustain_count": jnp.array(3, dtype=jnp.int32),
        }
        # Dummy params trees alongside it, matching how main_jax.py's
        # ckpt_state is actually shaped ({"b_params", "r_params", "training_state"}).
        dummy_params = {"emb_own": {"kernel": jnp.zeros((4, 4)), "bias": jnp.zeros((4,))}}
        ckpt_state = {
            "b_params": dummy_params,
            "r_params": dummy_params,
            "training_state": training_state,
        }

        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        save_mngr = ocp.CheckpointManager(tmp_dir, ocp.StandardCheckpointer(), options=options)
        save_mngr.save(1, items=ckpt_state)
        save_mngr.wait_until_finished()
        del save_mngr  # ensure nothing is carried over except what's on disk

        # A genuinely separate CheckpointManager instance, as a resumed
        # process would construct -- not the same Python object.
        restore_mngr = ocp.CheckpointManager(tmp_dir, ocp.StandardCheckpointer(), options=options)
        latest = restore_mngr.latest_step()
        assert latest == 1
        restored = restore_mngr.restore(latest)

        assert "training_state" in restored, "training_state key did not survive the round trip"
        rts = restored["training_state"]

        assert bool(rts["craft_ramp_active"]) is True
        assert int(rts["craft_ramp_start_step"]) == 1_414_656
        assert int(rts["craft_ramp_success_streak"]) == 7
        assert bool(rts["red_ramp_active"]) is False
        assert int(rts["red_ramp_start_step"]) == 1_414_656
        assert int(rts["red_ramp_catch_streak"]) == 10
        assert int(rts["red_curriculum_idx"]) == 2
        assert int(rts["red_sustain_count"]) == 3
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_missing_training_state_is_distinguishable_from_present():
    """An old checkpoint (pre-persistence-fix) has no training_state key at
    all -- main_jax.py's restore code must be able to tell "no saved state,
    use fresh defaults" apart from "saved state with all-default values".
    This test just pins that the absence itself round-trips as absence."""
    tmp_dir = tempfile.mkdtemp(prefix="ctd_ramp_ckpt_test_old_")
    try:
        dummy_params = {"emb_own": {"kernel": jnp.zeros((4, 4)), "bias": jnp.zeros((4,))}}
        ckpt_state = {"b_params": dummy_params, "r_params": dummy_params}  # no training_state

        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        save_mngr = ocp.CheckpointManager(tmp_dir, ocp.StandardCheckpointer(), options=options)
        save_mngr.save(1, items=ckpt_state)
        save_mngr.wait_until_finished()
        del save_mngr

        restore_mngr = ocp.CheckpointManager(tmp_dir, ocp.StandardCheckpointer(), options=options)
        restored = restore_mngr.restore(restore_mngr.latest_step())
        assert "training_state" not in restored
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_training_state_survives_a_real_checkpoint_round_trip()
    test_missing_training_state_is_distinguishable_from_present()
    print("OK: training_state (CtD ramp progress + red curriculum) survives "
          "a real Orbax checkpoint round trip through a separate, freshly "
          "constructed CheckpointManager instance; absence in an old "
          "checkpoint round-trips as absence, not as zeros.")
