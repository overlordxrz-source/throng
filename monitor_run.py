#!/usr/bin/env python3
"""Monitor the current Throng run."""

import psutil
import time
from pathlib import Path

log_dir = Path("runs_large")

# Find most recent run directory by modification time
run_dirs = [p for p in log_dir.iterdir() if p.is_dir()]
if not run_dirs:
    print("No run directories found.")
    exit(1)

run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
print(f"Run: {run_dir.name}")

# Check process
pid_files = [p for p in log_dir.glob("run_*.pid")]
if pid_files:
    pid_file = max(pid_files, key=lambda p: p.stat().st_mtime)
    pid = int(pid_file.read_text().strip())
    try:
        proc = psutil.Process(pid)
        print(f"PID: {pid}")
        print(f"Status: {proc.status()}")
        print(f"CPU: {proc.cpu_percent(interval=1.0):.1f}%")
        print(f"Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
    except psutil.NoSuchProcess:
        print(f"PID {pid} not running.")
else:
    print("No PID file found.")

# Read wrapper log
wrapper_log = log_dir / f"{run_dir.name.replace('run_', 'run_')}.log"
# Actually the wrapper log naming doesn't match; find most recent .log in runs_large
log_files = [p for p in log_dir.glob("run_*.log")]
if log_files:
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    text = latest_log.read_text()
    lines = text.splitlines()
    print(f"\nWrapper log ({latest_log.name}): {len(lines)} lines")
    if lines:
        print("Last 15 lines:")
        for line in lines[-15:]:
            print(f"  {line}")

# Read science.log
sci_log = run_dir / "science.log"
if sci_log.exists():
    text = sci_log.read_text()
    lines = text.splitlines()
    print(f"\nScience log: {len(lines)} lines")
    if lines:
        print("Last 15 lines:")
        for line in lines[-15:]:
            print(f"  {line}")

# Read events.jsonl
events = run_dir / "events.jsonl"
if events.exists():
    with open(events) as f:
        lines = f.readlines()
    print(f"\nEvents: {len(lines)} lines")
    if lines:
        import json
        last = json.loads(lines[-1])
        print(f"Last event: {last.get('type')} step={last.get('step', 'N/A')}")
