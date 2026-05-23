"""
utils/logging.py — Structured JSON run logging.

All log records are written as newline-delimited JSON (JSONL) to a per-run
log file.  JSONL is chosen over CSV because:
  - Schema-free: different record types (step_metrics, evo_event, mi_snapshot)
    can coexist in one file
  - Appendable: no need to hold a file handle open for the full run
  - Trivially parseable with Python's json module or jq

Log record types:
  "run_start"      — initial config snapshot
  "step_metrics"   — per-step population stats (written every N steps)
  "evo_event"      — evolution selection stats
  "mi_snapshot"    — mutual information matrix
  "checkpoint"     — checkpoint file path
  "run_end"        — final stats
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class RunLogger:
    """
    Thread-safe JSONL logger for one simulation run.

    Creates a new run directory under log_dir with a timestamp-based name
    (run_YYYYMMDD_HHMMSS) unless a specific run_id is provided.
    """

    def __init__(
        self,
        log_dir:  str,
        run_id:   Optional[str] = None,
        headless: bool          = False,
        use_wandb: bool         = False,
    ) -> None:
        self.headless = headless
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create run directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if run_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id    = f"run_{timestamp}"

        self.run_id  = run_id
        self.run_dir = log_path / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._log_file = self.run_dir / "events.jsonl"
        self._file_obj = open(self._log_file, "a", buffering=1)  # line-buffered

        # Write a separator if file already exists (e.g. resumed run)
        if self._log_file.stat().st_size > 0:
            self._write({"type": "resume", "ts": self._ts()})

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_run_start(self, config: Dict, step: int = 0) -> None:
        if self.use_wandb:
            wandb.init(
                project="throng",
                name=self.run_id,
                config=config,
                resume="allow",
                id=self.run_id
            )

        self._write({
            "type":   "run_start",
            "ts":     self._ts(),
            "step":   step,
            "run_id": self.run_id,
            "config": config,
        })

    def log_step_metrics(
        self,
        step:        int,
        population:  int,
        mean_fitness: float,
        max_fitness:  float,
        mean_energy:  float,
        evo_steps:    int,
        top_lineage_age: int,
    ) -> None:
        metrics = {
            "population":      population,
            "mean_fitness":    round(float(mean_fitness), 4),
            "max_fitness":     round(float(max_fitness),  4),
            "mean_energy":     round(float(mean_energy),  4),
            "evo_steps":       evo_steps,
            "top_lineage_age": top_lineage_age,
        }
        if self.use_wandb:
            wandb.log(metrics, step=step)

        self._write({
            "type":            "step_metrics",
            "ts":              self._ts(),
            "step":            step,
            **metrics
        })

    def log_evo_event(self, stats: Dict) -> None:
        self._write({"type": "evo_event", "ts": self._ts(), **stats})

    def log_mi_snapshot(self, step: int, snapshot) -> None:
        """
        Log an MISnapshot.  mi_matrix is serialised as a nested list so it
        can be read back without NumPy.
        """
        if self.use_wandb:
            wandb.log({"mi_max": float(snapshot.mi_matrix.max())}, step=step)

        self._write({
            "type":       "mi_snapshot",
            "ts":         self._ts(),
            "step":       step,
            "n_samples":  snapshot.n_samples,
            "mi_matrix":  snapshot.mi_matrix.tolist(),
        })

    def log_checkpoint(self, path: str, step: int) -> None:
        self._write({
            "type": "checkpoint",
            "ts":   self._ts(),
            "step": step,
            "path": str(path),
        })

    def log_run_end(self, step: int, reason: str = "user_exit") -> None:
        self._write({
            "type":   "run_end",
            "ts":     self._ts(),
            "step":   step,
            "reason": reason,
        })
        self._file_obj.flush()
        if self.use_wandb:
            wandb.finish()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write(self, record: Dict) -> None:
        try:
            self._file_obj.write(json.dumps(record) + "\n")
        except Exception as exc:
            # Logging must never crash the simulation
            print(f"[logger] WARNING: failed to write log record: {exc}")

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    def close(self) -> None:
        try:
            self._file_obj.flush()
            self._file_obj.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    @property
    def run_directory(self) -> Path:
        return self.run_dir
