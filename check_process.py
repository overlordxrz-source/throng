#!/usr/bin/env python3
"""Check if a Throng simulation is currently running."""

import psutil
import time
from pathlib import Path

found = False
for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
    try:
        cmdline = proc.info.get('cmdline', [])
        if cmdline and 'main.py' in ' '.join(cmdline):
            age = time.time() - proc.info['create_time']
            print(f"PID: {proc.info['pid']}")
            print(f"Age: {age:.0f}s ({age/60:.1f} min)")
            print(f"CMD: {' '.join(cmdline)}")
            print(f"CPU: {proc.cpu_percent(interval=1.0):.1f}%")
            print(f"Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
            found = True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not found:
    print("No main.py process found.")

# Also show most recent run dirs
print("\nRecent run directories:")
log_dir = Path("runs_large")
for d in sorted([p for p in log_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
    mtime = d.stat().st_mtime
    age = time.time() - mtime
    print(f"  {d.name} — modified {age:.0f}s ago")
    # Check for most recent checkpoint
    ckpts = list(d.glob("checkpoint_*.pkl"))
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
        print(f"    Latest checkpoint: {latest.name}")
