"""Pins scripts/durability_gate.py against a fake volume -- every branch,
including the two that matter most:

  * The historical silent-loss case (a write performed, nothing ever visible,
    NO tmp dir left behind -- the 7.5-hour loss and instance 6 both looked
    like this) MUST still be FATAL. An earlier draft of the 2026-09-22 fix
    softened "no new entry, no tmp" to a warning, which would have waved
    exactly those losses through; this test exists so that can't recur.
  * The 2026-09-22 04:26 EDT scenario (resumed at 2591, emergency save
    re-targeted 2591, Orbax declined, save() returned False) must NOT be FATAL
    -- and must not be counted as durability verified either.
"""

import pytest

from scripts.durability_gate import make_durability_gate


class _Entry:
    def __init__(self, path):
        self.path = path


class FakeVolume:
    def __init__(self, paths, raise_on_list=False):
        self.paths = set(paths)
        self.raise_on_list = raise_on_list
        self.commits = 0

    def commit(self):
        self.commits += 1

    def listdir(self, subdir):
        if self.raise_on_list:
            raise RuntimeError("listdir RPC failed")
        return [_Entry(p) for p in sorted(self.paths)]


SUB = "checkpoints_hearth"
BEFORE = {f"{SUB}/2589", f"{SUB}/2591"}


def _gate(vol):
    logs = []
    return make_durability_gate(vol, SUB, BEFORE, log=logs.append), logs


def test_real_write_that_lands_is_verified():
    vol = FakeVolume(BEFORE)
    gate, logs = _gate(vol)
    vol.paths.add(f"{SUB}/2592")
    gate(written=True)
    assert gate.state["verified"]
    assert "[DURABILITY-GATE] OK" in logs[-1]


def test_historical_silent_loss_is_still_fatal_without_a_tmp_dir():
    """A write was performed, nothing new visible, no tmp left behind --
    the shape of both real losses. Must halt."""
    vol = FakeVolume(BEFORE)
    gate, logs = _gate(vol)
    with pytest.raises(SystemExit):
        gate(written=True)
    assert "FATAL" in logs[-1] and "No tmp entry either" in logs[-1]
    assert not gate.state["verified"]


def test_orphaned_tmp_is_fatal_not_verified():
    """A tmp dir that appears after the write is NOT a checkpoint. The
    pre-extraction gate counted it as a new entry and would have reported a
    write that never finished as durability VERIFIED."""
    vol = FakeVolume(BEFORE)
    gate, logs = _gate(vol)
    vol.paths.add(f"{SUB}/2592.orbax-checkpoint-tmp-1789")
    with pytest.raises(SystemExit):
        gate(written=True)
    assert "Orphaned tmp entry present" in logs[-1]
    assert not gate.state["verified"]


def test_real_checkpoint_alongside_leftover_tmp_is_verified():
    vol = FakeVolume(BEFORE)
    gate, _ = _gate(vol)
    vol.paths |= {f"{SUB}/2592", f"{SUB}/2590.orbax-checkpoint-tmp-7"}
    gate(written=True)
    assert gate.state["verified"]


def test_declined_same_step_resave_is_not_fatal_and_not_verified():
    """The exact 2026-09-22 04:26 EDT scenario."""
    vol = FakeVolume(BEFORE)
    gate, logs = _gate(vol)
    gate(written=False)
    assert not gate.state["verified"], "a declined save proves nothing about durability"
    assert "no write was performed" in logs[-1]
    assert vol.commits == 1


def test_declined_then_real_write_still_gets_checked():
    vol = FakeVolume(BEFORE)
    gate, logs = _gate(vol)
    gate(written=False)
    with pytest.raises(SystemExit):
        gate(written=True)  # a real write that didn't land must still halt
    vol2 = FakeVolume(BEFORE)
    gate2, _ = _gate(vol2)
    gate2(written=False)
    vol2.paths.add(f"{SUB}/2594")
    gate2(written=True)
    assert gate2.state["verified"]


def test_listdir_failure_is_fatal():
    vol = FakeVolume(BEFORE, raise_on_list=True)
    gate, logs = _gate(vol)
    with pytest.raises(SystemExit):
        gate(written=True)
    assert "could not list" in logs[-1]


def test_after_verification_only_commits():
    vol = FakeVolume(BEFORE)
    gate, _ = _gate(vol)
    vol.paths.add(f"{SUB}/2592")
    gate(written=True)
    vol.paths.clear()  # later listings are never consulted once verified
    gate(written=True)
    assert vol.commits == 2
