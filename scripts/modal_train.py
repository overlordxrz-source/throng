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
    with open(REPO / "config_phase7.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["checkpoint_dir"] = "/mnt/throng-runs/checkpoints"
    # P10.5 Hard-Ceiling
    cfg["population_size"] = cfg["max_population"] = cfg["max_pop"] = 200
    cfg["min_population"] = 150
    cfg["ppo_gamma"] = 0.999
    # P10.4 Safety Bubble + ecology
    cfg["red_population_size"] = cfg["max_pop_red"] = 250
    cfg["min_red_population"] = 250
    cfg["red_curriculum_stages"] = [250]
    cfg["distill_enabled"] = False
    cfg["repro_energy_thresh"] = 0.95
    cfg["repro_energy_cost"] = 0.80
    cfg["red_catch_radius"] = 1
    cfg["red_catch_prob"] = 0.8
    cfg["red_detection_radius"] = 0
    cfg["resource_regen_rate"] = 0.0003
    cfg["resource_n_patches"] = 10
    cfg["resource_max"] = 0.5
    cfg["resource_spawn_boost"] = 0.1
    cfg["max_age"] = 1000
    cfg["vq_dead_code_reset"] = True
    cfg["ppo_rollout_steps"] = 512
    cfg["ppo_minibatch_size"] = 1024  # Phase 17: doubled from 512 (safe with dynamic GPU alloc)
    # Phase 17 — Gumbel-Softmax bottleneck + intrinsic entropy
    cfg["n_actions"] = 8        # N, S, E, W, Stay, Strike, Push, Guard
    cfg["env_channels"] = 10    # blue, red, wall, res, shelter, contested, scent, puzzle, blue_bg, barrier
    return cfg


if __name__ == "__main__":
    print("modal_train.py: P10.5 Hard-Ceiling resume (hot ckpt on volume)", flush=True)
    run_simulation(build_cfg(), seed=42, n_steps=2_000_000)
