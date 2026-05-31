#!/usr/bin/env python3
"""Background training entrypoint (Modal / nohup). Hot-resume from volume checkpoints.

  cd /root/throng
  nohup python -u run_bg.py > /mnt/throng-runs/train.log 2>&1 &
  tail -f /mnt/throng-runs/train.log

Does NOT wipe checkpoints. Does NOT touch reward structure or VQ beta (config_phase7.yaml).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.modal_train import build_cfg, run_simulation  # noqa: E402

if __name__ == "__main__":
    print("run_bg.py: P11.2 active imagination resume to 250k (latest Orbax ckpt on volume)", flush=True)
    run_simulation(build_cfg(), seed=42, n_steps=250_000)
