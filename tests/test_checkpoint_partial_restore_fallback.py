"""Regression test for the checkpoint-restore fallback bug found resuming
from step 2541 (2026-09-14): main_jax.py's structural-mismatch fallback
(`ckpt_mngr.restore(step, items=target_dict)`, reached when a checkpoint's
param tree doesn't match the current architecture 1:1 -- e.g. an old
checkpoint missing a head added since, or carrying dead_code_reset state
fields the current model.init() doesn't produce) had never been exercised
successfully in this project's history. Every prior resume was close enough
in architecture that the unconstrained restore one level up always
succeeded and this path was never reached.

It crashed hard: `items=target_dict` goes through CheckpointManager's
"default" item, which is bound to StandardCheckpointHandler.
StandardRestoreArgs's `strict=False` does NOT loosen a key-set mismatch
(only shape/dtype mismatches on keys present on both sides), and orbax's own
error message ("pass partial_restore=True") names a flag StandardRestore
doesn't expose -- it only exists on the lower-level PyTreeRestoreArgs, which
this CheckpointManager's handler registration refuses outright ("does not
match with any registered handler").

Fix: bypass the Standard-bound CheckpointManager for this one restore and go
straight to a PyTreeCheckpointer against the on-disk "default" item, which
does honor partial_restore, with restore_args built from target_dict's own
placement via `construct_restore_args` (needed on top of partial_restore
alone -- verified locally that partial_restore without explicit
restore_args still raises "Topology mismatch" / "sharding ... Got None"
against a checkpoint saved on a different device topology).

This test is a real Orbax round trip (save via CheckpointManager, matching
production's save path; restore via the fixed fallback path), with a
bidirectional mismatch: a key present in the checkpoint but not in the
current target (must be silently dropped), and a key present in the target
but not the checkpoint (must keep the target's own value, i.e. get grafted
by whatever runs next -- graft_missing_param_subtrees in production).
"""

import os
import shutil
import tempfile

import jax.numpy as jnp
import orbax.checkpoint as ocp


def test_partial_restore_fallback_survives_bidirectional_schema_mismatch():
    tmp_dir = tempfile.mkdtemp(prefix="ckpt_partial_restore_test_")
    try:
        # Simulates an old checkpoint: has a field the current architecture
        # no longer produces via model.init() (like codebook_N.usage_ema on
        # a pre-dead-code-reset-era checkpoint), missing a field the current
        # architecture expects (like a head added after this checkpoint was
        # saved).
        old_checkpoint = {
            "b_params": {
                "shared_layer": jnp.ones((4, 4)),
                "old_only_field": jnp.array(5.0),
            },
        }
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        save_mngr = ocp.CheckpointManager(tmp_dir, ocp.StandardCheckpointer(), options=options)
        save_mngr.save(1, items=old_checkpoint)
        save_mngr.wait_until_finished()
        del save_mngr

        # Current architecture's abstract target: missing old_only_field,
        # has a head that didn't exist when the checkpoint was saved.
        target_dict = {
            "b_params": {
                "shared_layer": jnp.zeros((4, 4)),
                "new_head": jnp.full((3,), -1.0),
            },
        }

        restore_mngr = ocp.CheckpointManager(tmp_dir, ocp.StandardCheckpointer(), options=options)
        latest = restore_mngr.latest_step()
        assert latest == 1

        # The strict path must actually fail first -- otherwise this test
        # would pass for the wrong reason (never touching the fallback).
        try:
            restore_mngr.restore(latest, items=target_dict)
            raise AssertionError(
                "expected the strict items= restore to raise on a structural "
                "mismatch -- if it didn't, this test is no longer exercising "
                "the fallback path at all"
            )
        except ValueError as exc:
            assert "do not match" in str(exc) or "Topology mismatch" in str(exc)

        # The fix under test.
        pytree_ckptr = ocp.PyTreeCheckpointer()
        raw_default_dir = os.path.join(tmp_dir, "1", "default")
        restore_args = ocp.checkpoint_utils.construct_restore_args(target_dict)
        restored = pytree_ckptr.restore(
            raw_default_dir,
            args=ocp.args.PyTreeRestore(
                item=target_dict, restore_args=restore_args, partial_restore=True
            ),
        )

        # Overlapping key: real value from the checkpoint, not the target's
        # placeholder.
        assert bool(jnp.array_equal(restored["b_params"]["shared_layer"], jnp.ones((4, 4)))), (
            "shared_layer should have been restored from the checkpoint, "
            "not left at the target's placeholder value"
        )
        # Present in checkpoint, absent from target: must be dropped, not
        # crash and not silently reappear.
        assert "old_only_field" not in restored["b_params"], (
            "old_only_field was in the checkpoint but not the target -- "
            "partial_restore should drop it, not carry it forward"
        )
        # Present in target, absent from checkpoint: keeps the target's own
        # (placeholder / freshly-initialized) value -- this is exactly the
        # slot graft_missing_param_subtrees is meant to fill in production.
        assert bool(jnp.array_equal(restored["b_params"]["new_head"], jnp.full((3,), -1.0))), (
            "new_head has no data in the checkpoint -- it should have kept "
            "target_dict's own value untouched, not been dropped or zeroed"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_partial_restore_fallback_survives_bidirectional_schema_mismatch()
    print(
        "OK: the checkpoint-restore fallback (PyTreeCheckpointer + "
        "partial_restore=True + construct_restore_args) survives a real "
        "bidirectional schema mismatch -- restores overlapping keys, drops "
        "checkpoint-only keys, keeps target-only keys at their own value. "
        "The strict items= path is confirmed to fail first, so the fallback "
        "is genuinely exercised, not bypassed by accident."
    )
