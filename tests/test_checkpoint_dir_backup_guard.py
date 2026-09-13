"""
Structural safeguard (Sep 2026 restart pre-flight, Cam's instruction): a
training run's checkpoint_dir must never resolve inside the local recovered
backup (~/throng_backup) — that directory is irreplaceable read-only source
material (the only copy of the corpus behind the Slot 2 ATE), and a training
run pointed at it would start writing new checkpoint steps into it.
"Remember not to do this" is exactly the failure mode Rule 13 exists to
eliminate; this test proves the guard raises rather than relying on anyone
remembering.
"""

import os
from pathlib import Path

from jax_sim.main_jax import assert_checkpoint_dir_is_not_protected_backup


def test_checkpoint_dir_inside_backup_raises():
    bad = str(Path(os.path.expanduser("~/throng_backup/checkpoints")).resolve())
    try:
        assert_checkpoint_dir_is_not_protected_backup(bad)
    except ValueError as e:
        assert "protected recovered-backup" in str(e)
    else:
        raise AssertionError("must raise when checkpoint_dir is inside ~/throng_backup")


def test_checkpoint_dir_equal_to_backup_root_raises():
    bad = str(Path(os.path.expanduser("~/throng_backup")).resolve())
    try:
        assert_checkpoint_dir_is_not_protected_backup(bad)
    except ValueError as e:
        assert "protected recovered-backup" in str(e)
    else:
        raise AssertionError("must raise when checkpoint_dir equals ~/throng_backup itself")


def test_checkpoint_dir_elsewhere_is_fine():
    good = str(Path("/tmp/some_other_run/checkpoints").resolve())
    assert_checkpoint_dir_is_not_protected_backup(good)  # must not raise


def test_sibling_directory_with_similar_prefix_is_not_falsely_flagged():
    """A directory like ~/throng_backup_v2 must NOT be caught by a naive
    string-prefix check — only exact equality or a real path separator
    boundary should trigger."""
    sibling = str(Path(os.path.expanduser("~/throng_backup_v2/checkpoints")).resolve())
    assert_checkpoint_dir_is_not_protected_backup(sibling)  # must not raise


if __name__ == "__main__":
    tests = [
        test_checkpoint_dir_inside_backup_raises,
        test_checkpoint_dir_equal_to_backup_root_raises,
        test_checkpoint_dir_elsewhere_is_fine,
        test_sibling_directory_with_similar_prefix_is_not_falsely_flagged,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("OK: checkpoint_dir backup guard fires correctly and doesn't over-trigger.")
