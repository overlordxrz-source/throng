"""Resume the latest Phase 5 run from its most recent checkpoint.

Usage on Kaggle:
    !python resume_run.py

Or in a notebook cell:
    %run resume_run.py
"""
import os
import glob
import subprocess
import sys

# Ensure we are in the right directory
if os.path.exists('/kaggle/working/throng'):
    os.chdir('/kaggle/working/throng')

# Find the latest run directory
ckpt_dirs = sorted(glob.glob('runs_large/run_*'), key=os.path.getmtime, reverse=True)

if not ckpt_dirs:
    print("No run directories found in runs_large/", file=sys.stderr)
    sys.exit(1)

latest_dir = ckpt_dirs[0]

# Grab all .pkl files but ignore the 'latest' symlink to prevent the integer crash
raw_ckpts = [f for f in glob.glob(f'{latest_dir}/checkpoint_*.pkl') if 'latest' not in f]

if not raw_ckpts:
    print(f"No numbered checkpoints found inside {latest_dir}!", file=sys.stderr)
    sys.exit(1)

# Sort by the actual step number in the filename
ckpts = sorted(raw_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
latest_ckpt = ckpts[0]

print(f"Resuming from: {latest_ckpt}")

# Stream output line-by-line so it appears immediately in notebooks
process = subprocess.Popen(
    ["python", "-u", "main.py", "--headless", "--config", "config_phase5.yaml", "--resume", latest_ckpt],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in process.stdout:
    print(line, end='')

return_code = process.wait()
if return_code != 0:
    print(f"Process exited with code {return_code}", file=sys.stderr)
    sys.exit(return_code)
