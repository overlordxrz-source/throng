"""
tools/bearing_analysis.py — West-cluster bearing analysis.

Tests Cam's prediction: West-going scouts near reds have red_bearing
peaking at 60-120° (red is East of agent), confirming genuine avoidance
rather than topology artifact.

Usage:
    python tools/bearing_analysis.py runs/<run>/signal_corpus.jsonl
"""

from __future__ import annotations

import json
import sys
import numpy as np

ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY"}
A_WEST = 3


def load(path: str):
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    signals = np.array([r["sig"]      for r in records], dtype=np.float32)
    actions = np.array([r["action"]   for r in records], dtype=np.int32)
    scouts  = np.array([r["scout"]    for r in records], dtype=bool)
    dists   = np.array([r["red_dist"] for r in records], dtype=np.float32)
    bears   = np.array([r["red_bear"] for r in records], dtype=np.float32)
    return signals, actions, scouts, dists, bears, len(records)


def bearing_histogram(bears: np.ndarray, n_bins: int = 12) -> list:
    valid = bears[np.isfinite(bears)]
    counts = [0] * n_bins
    width  = 360.0 / n_bins
    for b in valid:
        counts[int(b / width) % n_bins] += 1
    return counts


def print_histogram(counts: list, title: str) -> None:
    total  = sum(counts)
    n_bins = len(counts)
    width  = 360 // n_bins
    print(f"\n{title}  (N={total})")
    print("  Bearing (° from East, CCW)  count   bar")
    bin_labels = [
        "  0- 30 (E→NE)",
        " 30- 60 (NE)",
        " 60- 90 (N→NE)",
        " 90-120 (N)",
        "120-150 (N→NW)",
        "150-180 (NW)",
        "180-210 (W→NW)",
        "210-240 (W)",
        "240-270 (SW→W)",
        "270-300 (SW)",
        "300-330 (S→SW)",
        "330-360 (S→SE)",
    ]
    for i, cnt in enumerate(counts):
        lbl = bin_labels[i] if i < len(bin_labels) else f"{i*width}-{(i+1)*width}"
        bar = "█" * int(40 * cnt / max(total, 1))
        pct = 100 * cnt / max(total, 1)
        print(f"  {lbl}: {cnt:4d} ({pct:4.1f}%)  {bar}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/run_20260516_210433/signal_corpus.jsonl"
    near_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    signals, actions, scouts, dists, bears, n_total = load(path)
    print(f"\nLoaded {n_total:,} records from {path}")

    near_red   = dists <= near_thresh
    west_scout = (actions == A_WEST) & scouts & near_red

    print(f"\n{'='*60}")
    print(f"  WEST-GOING SCOUTS NEAR REDS (dist ≤ {near_thresh:.0f} cells)")
    print(f"{'='*60}")
    print(f"  West-going scouts near red : {west_scout.sum():,}")
    print(f"  All scouts near red         : {(scouts & near_red).sum():,}")
    print(f"  All agents near red         : {near_red.sum():,}")

    print(f"\n  Action distribution — scouts within {near_thresh:.0f} cells of red:")
    sn = scouts & near_red
    for a, nm in enumerate(["N", "S", "E", "W", "STAY"]):
        cnt = int(((actions == a) & sn).sum())
        pct = 100 * cnt / max(sn.sum(), 1)
        print(f"    {nm}: {cnt:4d}  ({pct:5.1f}%)")

    west_bears = bears[west_scout]
    all_near_bears = bears[scouts & near_red]

    print_histogram(bearing_histogram(west_bears),
                    "West-going scouts (red bearing)")
    print_histogram(bearing_histogram(all_near_bears),
                    "All scouts near red (baseline bearing)")

    print(f"\n{'─'*60}")
    print(f"  SUMMARY STATS")
    print(f"{'─'*60}")
    wv = west_bears[np.isfinite(west_bears)]
    av = all_near_bears[np.isfinite(all_near_bears)]
    if len(wv):
        print(f"  West cluster bearing: mean={wv.mean():.1f}°  median={np.median(wv):.1f}°  std={wv.std():.1f}°")
        east_frac = float(((wv >= 45) & (wv <= 135)).sum()) / len(wv)
        print(f"  Fraction with red to East (45-135°): {east_frac:.1%}")
        print(f"  Cam's prediction: peak at 60-120° → {'CONFIRMED' if east_frac > 0.35 else 'NOT CONFIRMED'}")
    if len(av):
        print(f"  All-scout baseline: mean={av.mean():.1f}°  median={np.median(av):.1f}°")

    print(f"\n  Interpretation:")
    print(f"  If West cluster peaks at 60-120° → genuine directional avoidance (transfers to 64x64)")
    print(f"  If uniform/random → topology artifact or gradient coincidence\n")


if __name__ == "__main__":
    main()
