"""Checkpoint durability bug (Cam, 2026-09-14): `str(Path(ckpt_dir).resolve())`
in main_jax.py's checkpoint init silently resolved Modal's mounted volume
path (/mnt/throng-runs/checkpoints) to the volume's internal backing-store
path (/__modal/volumes/vo-.../checkpoints) -- which bypasses the FUSE
mount's write-tracking entirely. Every checkpoint save went to a path
Volume.commit() never saw; volume.commit() itself returned successfully
every time (nothing to report as broken), and the run's own logs showed
"[CKPT] Saved" / "[CKPT] Committed" with no error. Confirmed directly by
writing a probe file through the UNRESOLVED mount path from inside the
live container: it committed and became externally visible on the first
try, while five real periodic checkpoint saves through the RESOLVED path
never did.

Fix: os.path.abspath() instead of Path(...).resolve() for the path the
CheckpointManager actually writes through -- absolute, but does not follow
symlinks, so it can't be silently redirected off the FUSE-mounted volume.
The protected-backup guard (assert_checkpoint_dir_is_not_protected_backup)
still needs a genuinely resolved path to catch a checkpoint_dir that
reaches the backup via a symlink -- verified here that a LOCAL resolved
copy, computed only for that one check, still catches it.
"""

import os
import shutil
import tempfile
from pathlib import Path

from jax_sim.main_jax import assert_checkpoint_dir_is_not_protected_backup


def test_abspath_does_not_follow_symlinks_but_resolve_does():
    """The property the whole fix depends on: os.path.abspath() returns the
    symlink's own path unchanged; Path(...).resolve() follows it to the
    target. If this ever stops being true on some platform, the fix silently
    stops working."""
    # os.path.realpath() up front: on macOS, tempfile's own root (/tmp) is
    # itself a symlink to /private/tmp, which would otherwise make
    # Path.resolve() diverge from `real_target`'s literal string for a
    # reason that has nothing to do with the symlink this test constructs.
    tmp_root = os.path.realpath(tempfile.mkdtemp(prefix="ckpt_path_test_"))
    try:
        real_target = os.path.join(tmp_root, "real_target")
        os.makedirs(real_target)
        symlink_path = os.path.join(tmp_root, "mount_like_symlink")
        os.symlink(real_target, symlink_path)

        checkpoints_via_symlink = os.path.join(symlink_path, "checkpoints")

        abspath_result = os.path.abspath(checkpoints_via_symlink)
        resolve_result = str(Path(checkpoints_via_symlink).resolve())

        assert abspath_result == checkpoints_via_symlink, (
            "os.path.abspath() must not change an already-absolute path -- "
            "if it does, the 'no-op for production's already-absolute "
            "checkpoint_dir' assumption is wrong"
        )
        assert abspath_result.startswith(symlink_path), (
            "abspath must keep writes addressed through the symlink (the "
            "FUSE-mount-equivalent path in this test), not redirect them"
        )
        assert resolve_result.startswith(real_target), (
            "resolve() must follow the symlink to the real target -- this "
            "is the exact mechanism that silently bypassed Modal's "
            "Volume.commit() tracking in production"
        )
        assert abspath_result != resolve_result, (
            "the whole bug is that these two differ for a symlinked mount; "
            "if they're equal in this test setup, the test isn't exercising "
            "the real scenario"
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_protected_backup_guard_still_catches_a_symlinked_path():
    """The guard must still fire even though the CheckpointManager's own
    ckpt_dir is no longer resolved -- main_jax.py achieves this by resolving
    a SEPARATE local copy just for the guard call. Reproduce that pattern
    directly against the real guard function with a real symlink into a
    fake 'protected backup'."""
    tmp_root = os.path.realpath(tempfile.mkdtemp(prefix="ckpt_guard_symlink_test_"))
    try:
        fake_backup = os.path.join(tmp_root, "throng_backup")
        os.makedirs(fake_backup)
        symlink_to_backup = os.path.join(tmp_root, "sneaky_symlink")
        os.symlink(fake_backup, symlink_to_backup)

        unresolved_ckpt_dir = os.path.join(symlink_to_backup, "checkpoints")
        # Mirrors main_jax.py: ckpt_dir itself stays unresolved (abspath
        # only); a separate resolved copy is computed just for the guard.
        ckpt_dir = os.path.abspath(unresolved_ckpt_dir)
        resolved_for_guard = str(Path(ckpt_dir).resolve())

        assert resolved_for_guard.startswith(fake_backup), (
            "test setup sanity: the symlink must actually resolve into the "
            "fake backup, or this test proves nothing"
        )

        import jax_sim.main_jax as main_jax_mod
        original_expanduser = main_jax_mod.os.path.expanduser
        try:
            main_jax_mod.os.path.expanduser = (
                lambda p: fake_backup if p == "~/throng_backup" else original_expanduser(p)
            )
            raised = False
            try:
                assert_checkpoint_dir_is_not_protected_backup(resolved_for_guard)
            except ValueError:
                raised = True
            assert raised, (
                "the guard must still reject a checkpoint_dir that reaches "
                "the protected backup through a symlink -- ckpt_dir no "
                "longer being resolved must not defeat this check"
            )
        finally:
            main_jax_mod.os.path.expanduser = original_expanduser
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    test_abspath_does_not_follow_symlinks_but_resolve_does()
    test_protected_backup_guard_still_catches_a_symlinked_path()
    print(
        "OK: os.path.abspath() (what the CheckpointManager now writes through) "
        "does not follow symlinks the way Path(...).resolve() did -- the exact "
        "mechanism that silently redirected checkpoint writes off the mounted "
        "volume -- and the protected-backup guard still fires against a "
        "separately-resolved copy, so the safety check isn't weakened."
    )
