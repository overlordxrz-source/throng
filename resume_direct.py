"""Resume the latest Phase 5 run directly (no subprocess — faster, no stdout issues).

Paste this into a Kaggle notebook cell and run:

    %run resume_direct.py

Or copy the contents into a cell.
"""
import os
import glob
import sys

# Ensure we are in the right directory
if os.path.exists('/kaggle/working/throng'):
    os.chdir('/kaggle/working/throng')

# Add the project root to sys.path so imports work
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Find the latest run directory
ckpt_dirs = sorted(glob.glob('runs_large/run_*'), key=os.path.getmtime, reverse=True)

if not ckpt_dirs:
    print("No run directories found in runs_large/", file=sys.stderr)
    sys.exit(1)

latest_dir = ckpt_dirs[0]
raw_ckpts = [f for f in glob.glob(f'{latest_dir}/checkpoint_*.pkl') if 'latest' not in f]

if not raw_ckpts:
    print(f"No numbered checkpoints found inside {latest_dir}!", file=sys.stderr)
    sys.exit(1)

ckpts = sorted(raw_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
latest_ckpt = ckpts[0]

print(f"Resuming from: {latest_ckpt}")

# Import and call main.py's run() function directly
from main import run, load_config

config = load_config("config_phase5.yaml")
run(config, resume_path=latest_ckpt, headless=True, fresh=False,
    withdrawal=False, blind=False, max_steps=0)
