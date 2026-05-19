#!/usr/bin/env python3
"""Launch a fresh Throng run with unbuffered output for monitoring."""

import os
import subprocess
import sys
import time
from pathlib import Path

# Unbuffered stdout so we can see startup progress immediately
os.environ["PYTHONUNBUFFERED"] = "1"

LOG_DIR = Path("runs_large")
LOG_DIR.mkdir(exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"run_{timestamp}.log"

print(f"Launching Throng fresh run...")
print(f"Log: {log_path}")

with open(log_path, "w", buffering=1) as log_fh:
    proc = subprocess.Popen(
        [
            "python",
            "-u",  # unbuffered
            "main.py",
            "--config", "config_large.yaml",
            "--headless",
            "--fresh",
            "--max-steps", "200000",
        ],
        cwd=".",
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    print(f"PID: {proc.pid}")
    pid_file = LOG_DIR / f"run_{timestamp}.pid"
    pid_file.write_text(str(proc.pid))
    print(f"PID file: {pid_file}")
    print("Waiting 30s for MPS compilation...")

    try:
        for i in range(30):
            time.sleep(1)
            if proc.poll() is not None:
                print(f"\nProcess exited early with code {proc.poll()}")
                log_fh.flush()
                with open(log_path) as f:
                    tail = '\n'.join(f.read().splitlines()[-50:])
                print("--- Last 50 log lines ---")
                print(tail)
                sys.exit(1)
        print("\nProcess still running after 30s (MPS likely compiled).")
        print("Monitoring...")
        for i in range(30):
            time.sleep(1)
            if proc.poll() is not None:
                print(f"Process exited with code {proc.poll()}")
                log_fh.flush()
                with open(log_path) as f:
                    tail = '\n'.join(f.read().splitlines()[-50:])
                print("--- Last 50 log lines ---")
                print(tail)
                sys.exit(1)
        print("Process running after 60s total. Check log file for progress.")
    except KeyboardInterrupt:
        proc.terminate()
        print("Terminated.")
