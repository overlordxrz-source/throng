#!/usr/bin/env python3
"""Run JAX training on Modal (or any GPU host) outside a notebook kernel.

Usage (after git clone to /root/throng and volume at /mnt/throng-runs):

  nohup python -u /root/throng/scripts/modal_train.py > /mnt/throng-runs/train.log 2>&1 &
  tail -f /mnt/throng-runs/train.log
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# Let JAX allocate GPU memory on demand instead of pre-allocating a fixed fraction.
# This prevents the OOM-then-hardstuck failure mode.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/throng_jax_cache"
os.makedirs("/tmp/throng_jax_cache", exist_ok=True)

import yaml  # noqa: E402

from jax_sim.train_entry import run_simulation  # noqa: E402


def build_cfg() -> dict:
    """Load config.yaml as-is; override only what's genuinely Modal-infra-specific.

    This used to re-assert ~22 individual training/ecology hyperparameters on
    top of the loaded config.yaml (population sizes, PPO settings, red ecology
    params, n_actions, env_channels...). AUDIT_SEP2026.md flagged this as the
    exact mechanism that produced the historical `env_channels=12` bug
    (THRONG.md's Phase 18.7 headline): someone tunes a value in config.yaml,
    runs via this script, and the tuned value is silently overwritten back to
    whatever was hardcoded here. Verified line-by-line that all 22 previously
    matched config.yaml except two that had silently drifted apart:
    `red_population_size`/`max_pop_red` were hardcoded to 250 here but absent
    from config.yaml entirely, so loading config.yaml alone would have fallen
    back to `_normalize_config`'s default of 75 — a real ecology change, not a
    redundant duplicate. Fixed by adding both to config.yaml directly instead
    of perpetuating the shadow copy.
    """
    with open(REPO / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    # 2026-09-15 (Cam): checkpoints/ still holds a fossil (2862, from an
    # abandoned lineage) that Orbax's max_to_keep retention would prune
    # every new save against -- the actual mechanism behind two lost
    # launches tonight (docs/THE-ECOLOGY-NEVER-RAN.md instance 6). New
    # directory, new lineage, seeded with only the checkpoint we're
    # actually resuming from (2541, restored from ~/throng_backup). The old
    # checkpoints/ is left exactly as it was -- inert once nothing points
    # at it, and 2862 is now evidence, not live state.
    cfg["checkpoint_dir"] = "/mnt/throng-runs/checkpoints_r2541"
    return cfg


if __name__ == "__main__":
    print("modal_train.py: P10.5 Hard-Ceiling resume (hot ckpt on volume)", flush=True)
    run_simulation(build_cfg(), seed=42, n_steps=2_000_000)
