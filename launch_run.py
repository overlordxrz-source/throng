#!/usr/bin/env python3
"""Launch a fresh Throng run with proper logging."""

import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path("runs_large")
LOG_DIR.mkdir(exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"run_{timestamp}.log"

print(f"Launching Throng run...")
print(f"Log: {log_path}")

proc = subprocess.Popen(
    [
        "python",
        "main.py",
        "--config", "config_large.yaml",
        "--headless",
        "--fresh",
        "--max-steps", "200000",
    ],
    cwd=".",
    stdout=open(log_path, "w"),
    stderr=subprocess.STDOUT,
)

print(f"PID: {proc.pid}")

# Wait a few seconds to see if it starts successfully
time.sleep(10)

if proc.poll() is not None:
    print(f"Process exited early with code {proc.poll()}")
    with open(log_path) as f:
        print(f.read())
    sys.exit(1)
else:
    print("Process running successfully.")
    # Write PID file
    pid_file = LOG_DIR / f"run_{timestamp}.pid"
    pid_file.write_text(str(proc.pid))
    print(f"PID file: {pid_file}")
