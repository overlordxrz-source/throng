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
    # 2026-09-21 (Hearths relaunch, Cam's volume layout): fresh, empty
    # workspace laid out from scratch to avoid inheriting the fossil/
    # self-deleting-run problems the pre-hearth checkpoints/ directory had
    # (docs/THE-ECOLOGY-NEVER-RAN.md instance 6). checkpoints_hearth/ is
    # the live run directory and holds ONLY the 2541 resume point, ever --
    # no fossils, no diagnostics. fossils/ (the six ladder checkpoints) and
    # archive/ (the old pair-craft-world corpora and train.log) are
    # read-only, never the run's target.
    cfg["checkpoint_dir"] = "/mnt/throng-runs/checkpoints_hearth"
    # Corpus durability fix (2026-09-21, Will, self-caught): the default
    # relative "runs/{run_name}" path resolves against the repo clone
    # (/root/throng), not the mounted volume, so it was never actually
    # durable -- see jax_sim/main_jax.py's corpus_dir comment. Written
    # directly at the volume root (same level as train.log), with a
    # filename distinct from the old signal_corpus.jsonl by construction --
    # Cam: "we spent real effort measuring a contamination boundary after
    # the fact; this time we prevent it by construction." This is also a
    # brand-new, empty volume, so there is no pre-existing corpus/log to
    # collide with regardless.
    cfg["corpus_dir"] = "/mnt/throng-runs"
    cfg["corpus_filename"] = "signal_corpus_hearth.jsonl"
    cfg["corpus_filename_red"] = "signal_corpus_hearth_red.jsonl"
    return cfg


if __name__ == "__main__":
    print("modal_train.py: P10.5 Hard-Ceiling resume (hot ckpt on volume)", flush=True)
    run_simulation(build_cfg(), seed=42, n_steps=2_000_000)
