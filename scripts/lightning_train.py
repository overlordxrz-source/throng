"""Lightning AI launcher, mirroring scripts/modal_app.py's structure for a
different compute backend (2026-09-28: coolerthanyousix's Modal workspace was
paused mid-run -- see docs/THE-ECOLOGY-NEVER-RAN.md's status entries).

Key difference from Modal: a Lightning Studio's persistent filesystem
(/teamspace/studios/this_studio/) IS the durable store directly -- there is no
separate "commit to a network volume" step the way Modal's Volume needs
(scripts/durability_gate.py exists specifically because Modal's local
container view and its committed volume state can silently diverge; a
Studio's disk has no equivalent split). The durability check below is
adapted accordingly: it re-lists the checkpoint directory after save to
confirm the new step is visible on disk, which is the meaningful analogue
here, without inventing volume-commit semantics that don't exist on this
platform.

Run from inside the studio (this script assumes it, not from a laptop):
    cd /teamspace/studios/this_studio/throng && python scripts/lightning_train.py [--test] [--n-steps N]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STUDIO_ROOT = Path("/teamspace/studios/this_studio")
RUN_ROOT = STUDIO_ROOT / "throng-runs"
N_STEPS_FULL = 3_000_000


def build_cfg() -> dict:
    """Load config.yaml as-is; override only what's genuinely infra-specific.
    Mirrors scripts/modal_app.py's build_cfg() -- see that function's
    docstring for why this stays a thin override, not a hyperparameter
    shadow copy (AUDIT_SEP2026.md)."""
    import yaml

    with open(REPO / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["checkpoint_dir"] = str(RUN_ROOT / "checkpoints_hearth")
    cfg["corpus_dir"] = str(RUN_ROOT)
    cfg["corpus_filename"] = "signal_corpus_hearth.jsonl"
    cfg["corpus_filename_red"] = "signal_corpus_hearth_red.jsonl"
    return cfg


def _make_durability_check(ckpt_dir: str):
    """Re-lists ckpt_dir after each save to confirm the new step landed on
    disk -- the Studio-filesystem analogue of scripts/durability_gate.py's
    volume.listdir() check. No separate "commit" call exists on this
    platform for the check to distinguish "written" from "committed"; a
    fresh directory listing after the save call returns is the honest
    equivalent."""
    state = {"verified": False}

    def check(written: bool = True) -> None:
        if state["verified"] or not written:
            if not written:
                print(
                    "[DURABILITY] no write was performed this save -- nothing to verify yet.",
                    flush=True,
                )
            return
        try:
            steps = sorted(int(p.name) for p in Path(ckpt_dir).iterdir() if p.name.isdigit())
        except FileNotFoundError as exc:
            print(f"[DURABILITY] FATAL: could not list {ckpt_dir}: {exc!r}. Halting.", flush=True)
            raise SystemExit(1)
        if not steps:
            print(f"[DURABILITY] FATAL: {ckpt_dir} has no checkpoint steps after a save. Halting.", flush=True)
            raise SystemExit(1)
        print(f"[DURABILITY] OK: {ckpt_dir} shows steps up to {steps[-1]} after save.", flush=True)
        state["verified"] = True

    return check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="CPU preflight only, no GPU run")
    parser.add_argument("--n-steps", type=int, default=0)
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    print(f"[preflight] HEAD = {head}", flush=True)

    cfg = build_cfg()
    print(f"[preflight] build_cfg() OK, checkpoint_dir={cfg['checkpoint_dir']}", flush=True)

    sys.path.insert(0, str(REPO))

    if args.test:
        import orbax.checkpoint as ocp
        from jax_sim.main_jax import find_fossil_checkpoints

        os.environ["JAX_PLATFORMS"] = "cpu"
        mngr = ocp.CheckpointManager(
            cfg["checkpoint_dir"], ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(max_to_keep=10, create=False),
        )
        latest = mngr.latest_step()
        print(f"[preflight] checkpoint_dir latest step = {latest}", flush=True)
        if latest is not None:
            fossils = find_fossil_checkpoints(mngr, latest)
            if fossils:
                raise AssertionError(f"[preflight] FOSSIL-GUARD would refuse this launch: {fossils} ahead of {latest}")
            print(f"[preflight] fossil guard OK -- no checkpoint ahead of resume target {latest}", flush=True)
        print("[preflight] done.", flush=True)
        return

    log_path = RUN_ROOT / "train.log"

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self._streams:
                s.flush()

    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    n_steps = args.n_steps or N_STEPS_FULL
    gpu_name = "unknown"
    try:
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        pass
    print(
        f"lightning_train.py — THRONG | pinned HEAD {head} | n_steps={n_steps:_} | "
        f"GPU={gpu_name} (Lightning AI Studio, migrated off Modal's coolerthanyousix "
        "after its workspace was paused mid-run 2026-09-28 -- whatever GPU type is "
        "actually attached, detected live rather than assumed from the machine request)",
        flush=True,
    )

    from jax_sim.train_entry import run_simulation

    durability_check = _make_durability_check(cfg["checkpoint_dir"])
    run_simulation(cfg, seed=42, n_steps=n_steps, on_checkpoint_saved=durability_check)


if __name__ == "__main__":
    main()
