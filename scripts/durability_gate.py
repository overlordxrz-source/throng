"""External checkpoint-durability gate, extracted from scripts/modal_app.py
(2026-09-22) so every branch can be proven against a fake volume instead of
only ever being exercised live on a paid GPU run -- which is how its one
false positive shipped.

Durability is a precondition for running, not a property to fix while
running: after the first checkpoint write, confirm via the volume's own
listdir RPC (not the container's FUSE view) that a new entry actually landed,
or halt. See docs/THE-ECOLOGY-NEVER-RAN.md instances 5 and 6 for the two
real losses this exists to catch.
"""

from __future__ import annotations

from typing import Callable, Iterable

FATAL_HISTORY = (
    "This is the exact failure mode that silently lost the 2026-09-14 19:12 EDT - "
    "2026-09-15 02:51 EDT run (7.5 hours, 44 PPO updates): '[CKPT] Committed' printed "
    "every time and nothing ever reached the volume. Halting now before burning more "
    "GPU on a run that cannot save its output."
)


def make_durability_gate(
    volume,
    ckpt_subdir: str,
    paths_before: Iterable[str],
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> Callable[..., None]:
    """Return the on_checkpoint_saved callback.

    `volume` needs .commit() and .listdir(path) -> iterable of objects with
    .path. The callback takes `written`: whether Orbax actually performed the
    save (CheckpointManager.save's own return value).

    2026-09-22 (Cam): "a gate that cries wolf will eventually be ignored,
    which is the only way it can hurt us." The 04:26 EDT relaunch resumed at
    2591, tripped on its first update, and its emergency save re-targeted
    2591 -- Orbax declined (save() returned False, no exception; verified
    locally and on the volume: mtime unchanged, no tmp dir), and this gate
    called "nothing new appeared" a durability failure. Knowing whether a
    write was attempted tells the two cases apart at the source:

      written=False -> no write attempted. Not a durability test: neither
        FATAL nor verified. Durability stays unproven until a real write.
      written=True, nothing new visible -> FATAL, unconditionally. Must NOT
        be softened by the absence of an orphaned tmp dir: the 7.5-hour loss
        and instance 6's retention pruning both left no tmp behind.
    """
    before = set(paths_before)
    state = {"verified": False}

    def gate(written: bool = True) -> None:
        volume.commit()
        if state["verified"]:
            return
        if not written:
            log(
                "[DURABILITY-GATE] no write was performed this save (Orbax declined, e.g. "
                "the step already exists) -- nothing to verify yet. Durability remains "
                "UNVERIFIED until a real write lands and is confirmed."
            )
            return
        # 2026-09-15: NOT volume.reload() -- that refreshes THIS container's
        # own local FUSE view, and fails outright ("there are open files
        # preventing the operation: path train.log is open") for the entire
        # run, since the Tee holds train.log open the whole time. listdir()
        # is a separate RPC against the backing store's committed state and
        # needs no reload() first. The first launch with reload() fired
        # FATAL for exactly this reason, not a real durability failure.
        try:
            after = {e.path for e in volume.listdir(ckpt_subdir)}
        except Exception as exc:
            log(
                f"[DURABILITY-GATE] FATAL: could not list {ckpt_subdir}/ via the volume RPC "
                f"after commit to verify durability: {exc!r}. Halting before burning more GPU."
            )
            raise SystemExit(1)
        # An orphaned Orbax tmp dir is not a checkpoint. Counting it as a
        # "new entry" (as the pre-extraction gate did) would report a write
        # that started and never finished as durability VERIFIED.
        tmp = sorted(p for p in after if "orbax-checkpoint-tmp-" in p)
        new = {p for p in after - before if "orbax-checkpoint-tmp-" not in p}
        if not new:
            note = (
                f" Orphaned tmp entry present ({tmp}): a write started and did not resolve."
                if tmp else
                " No tmp entry either: the write completed locally and never became visible "
                "(uncommitted, or pruned by retention -- instance 6)."
            )
            log(
                f"[DURABILITY-GATE] FATAL: a checkpoint write was performed, but no new entry "
                f"is visible under {ckpt_subdir}/ via volume.listdir() after commit.{note} "
                + FATAL_HISTORY
            )
            raise SystemExit(1)
        log(
            f"[DURABILITY-GATE] OK: {sorted(new)} confirmed visible under {ckpt_subdir}/ via "
            f"volume.listdir() after commit -- durability verified externally, not just "
            f"trusted from the container's own print."
        )
        state["verified"] = True

    gate.state = state  # exposed for tests
    return gate
