#!/usr/bin/env python3
"""Withdrawal test: run 2k steps with signals zeroed from checkpoint 50k, FORCE CPU."""
import sys
sys.path.insert(0, '.')

import torch
# Force CPU by disabling MPS availability
original_mps_available = torch.backends.mps.is_available
torch.backends.mps.is_available = lambda: False

from main import load_config, run

config = load_config('config_phase4.yaml')
config['log_dir'] = 'runs_withdrawal'
run(
    config,
    resume_path='runs_large/run_20260518_233539/checkpoint_50000.pkl',
    headless=True,
    withdrawal=True,
    max_steps=2000,
)
