"""Kaggle notebook cell: start training and tail logs in real time.

Paste this entire file into a Kaggle notebook cell and run.
"""
import os
import subprocess
import time
import threading

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Ensure output directory exists (prevents tee/no such file errors)
os.makedirs('/kaggle/working/throng/runs_large', exist_ok=True)

LOG_FILE = '/kaggle/working/throng/runs_large/train_out.txt'

def run_training():
    os.chdir('/kaggle/working/throng')
    with open(LOG_FILE, 'w') as f:
        proc = subprocess.Popen(
            ['python', '-u', 'main.py', '--config', 'config_phase5.yaml',
             '--headless', '--fresh'],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc.wait()

# Start training in a background thread
t = threading.Thread(target=run_training)
t.daemon = True
t.start()

print("🚀 Training started. Tailing logs every 30s...")
time.sleep(15)  # Let it get past CUDA JIT and create the log

while t.is_alive():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            if lines:
                # Print last 8 lines
                print(''.join(lines[-8:]), end='')
                print("─" * 60)
    else:
        print("(log file not created yet...)")
    time.sleep(30)

print("🏁 Training finished.")
