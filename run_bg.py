#!/usr/bin/env python3
"""Background training entrypoint (Modal / nohup). Hot-resume from volume checkpoints.

  cd /root/throng
  nohup python -u run_bg.py > /mnt/throng-runs/train.log 2>&1 &
  tail -f /mnt/throng-runs/train.log

Config: config.yaml (the single active config). Resume restores weights only;
population/grid/curriculum start fresh. Does NOT wipe checkpoints, reward
structure, or VQ beta. See docs/ARCHITECTURE.md for the network/wire map and
docs/STRATEGIC_ROADMAP.md for the phase plan.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.modal_train import build_cfg, run_simulation  # noqa: E402

N_STEPS = 3_000_000

if __name__ == "__main__":
    print(
        "run_bg.py — THRONG | blue: 3-slot discrete VQ (12/8/12, 64-code) on a "
        "32-D effective wire, 12 actions, GWT-masked comms | red: pure ecological "
        f"pressure (VQ decoupled) | hot-resume from checkpoint | n_steps={N_STEPS:_}",
        flush=True,
    )
    run_simulation(build_cfg(), seed=42, n_steps=N_STEPS)
