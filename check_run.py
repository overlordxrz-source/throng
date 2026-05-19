#!/usr/bin/env python3
"""Check status of the running Throng simulation."""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

LOG_DIR = Path("runs_large")


def get_main_py_pids():
    """Find PIDs of running main.py processes."""
    result = subprocess.run(
        ["pgrep", "-f", "main.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return [p for p in result.stdout.strip().split("\n") if p]


def get_process_info(pid):
    """Get basic info about a process without psutil."""
    try:
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "pid,etime,%cpu,vsz,rss,command"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) >= 2:
            return lines[1]  # Skip header
    except Exception:
        pass
    return None


def find_latest_run():
    """Find the most recent run directory."""
    runs = sorted(
        glob.glob(str(LOG_DIR / "run_*")),
        key=os.path.getmtime,
        reverse=True
    )
    return Path(runs[0]) if runs else None


def tail_science_log(run_dir, n=40):
    """Show last N lines of science.log with key metrics."""
    slog = run_dir / "science.log"
    if not slog.exists():
        return "No science.log"

    with open(slog) as f:
        lines = f.readlines()

    total = len(lines)
    out = [f"science.log: {total} lines"]

    # Find last occurrence of each metric type
    tags = [
        ("MI_max", "MI"),
        ("PPO#", "PPO"),
        ("ALARM", "Alarm"),
        ("NB_GAIN_SURV", "NB Gain"),
        ("CHAIN_DEPTH", "Chain"),
        ("GRANGER", "Granger"),
        ("CULTURE", "Culture"),
        ("CURRICULUM", "Curriculum"),
        ("SOCIAL_DIM_DOT", "Social Dot"),
        ("checkpoint", "Checkpoint"),
    ]

    for tag, label in tags:
        for line in reversed(lines):
            if tag in line:
                out.append(f"  [{label}] {line.strip()[:200]}")
                break

    out.append(f"\n--- Last {n} raw lines ---")
    for line in lines[-n:]:
        out.append(line.rstrip())

    return "\n".join(out)


def tail_events(run_dir):
    """Show last event from events.jsonl."""
    elog = run_dir / "events.jsonl"
    if not elog.exists():
        return "No events.jsonl"

    with open(elog) as f:
        lines = f.readlines()

    if not lines:
        return "events.jsonl empty"

    try:
        last = json.loads(lines[-1])
        return (
            f"events.jsonl: {len(lines)} lines\n"
            f"  Last: step={last.get('step')}, "
            f"pop={last.get('population')}, "
            f"surv={last.get('mean_energy', 0):.2f}"
        )
    except json.JSONDecodeError:
        return f"events.jsonl: {len(lines)} lines (parse error on last line)"


def main():
    print("=" * 60)
    print("THRONG Run Status Check")
    print("=" * 60)

    # Running processes
    pids = get_main_py_pids()
    print(f"\n[Processes] {len(pids)} main.py running")
    for pid in pids:
        info = get_process_info(pid)
        if info:
            print(f"  PID {pid}: {info}")
        else:
            print(f"  PID {pid}: (info unavailable)")

    # Latest run
    latest = find_latest_run()
    if not latest:
        print("\n[Run] No runs found in runs_large/")
        sys.exit(1)

    print(f"\n[Latest Run] {latest.name}")
    import time as _time
    mtime = os.path.getmtime(latest)
    age_mins = (_time.time() - mtime) / 60
    print(f"  Modified: {age_mins:.1f} minutes ago")

    # Checkpoints
    ckpts = sorted(
        glob.glob(str(latest / "checkpoint_*.pkl")),
        key=os.path.getmtime,
        reverse=True
    )
    if ckpts:
        print(f"  Latest checkpoint: {Path(ckpts[0]).name}")
    else:
        print("  No checkpoints yet")

    # Logs
    print(f"\n{tail_events(latest)}")
    print()
    print(tail_science_log(latest, n=30))
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
