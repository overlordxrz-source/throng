"""
tools/dir_away.py — Directional awayΔ split for E/W binary validation.

For scouts, splits by which cardinal quadrant the red agent was in when the
signal was emitted (red_bearing), then computes flee rates for scouts and
blind agents in matching bearing ranges.

awayΔ = scout_flee_rate − blind_flee_rate (within that bearing quadrant)

If the E/W binary is genuinely directional, awayΔ should be larger for the
quadrant(s) whose bearing aligns with the dominant dim encoding directions.
If awayΔ is flat across all quadrants → encoding is not behaviorally load-bearing.

Bearing convention: degrees, 0=North, 90=East, 180=South, 270=West
(compass convention, as used in signal_corpus.jsonl red_bear field).

Usage:
    python tools/dir_away.py <corpus.jsonl> [--min-step N] [--max-step N]
    python tools/dir_away.py <corpus.jsonl> --min-step N --group-by crowding
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np


FLEE_ACTION = 1


def in_quadrant(bearing_deg: float, quad: str) -> bool:
    """Compass degrees: 0=N, 90=E, 180=S, 270=W. Each quadrant ±45° around cardinal."""
    b = bearing_deg % 360
    if quad == "N":
        return b < 45 or b >= 315
    elif quad == "E":
        return 45 <= b < 135
    elif quad == "S":
        return 135 <= b < 225
    elif quad == "W":
        return 225 <= b < 315
    return False


def load(path: str, min_step: int = 0, max_step: int = 0):
    actions, scouts, bearing, neighbors, steps = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = r.get("step", 0)
            if min_step > 0 and step < min_step:
                continue
            if max_step > 0 and step > max_step:
                continue
            actions.append(r.get("action", -1))
            scouts.append(bool(r.get("scout", False)))
            bearing.append(r.get("red_bear", float("nan")))
            neighbors.append(r.get("neighbors", float("nan")))
            steps.append(step)

    return (
        np.array(actions,   dtype=np.int32),
        np.array(scouts,    dtype=bool),
        np.array(bearing,   dtype=np.float64),
        np.array(neighbors, dtype=np.float64),
        np.array(steps,     dtype=np.int64),
    )


def _away_row(
    actions: np.ndarray,
    scouts: np.ndarray,
    mask: np.ndarray,
    overall_blind_flee: float,
) -> tuple:
    """Return (n_scouts, n_blind, scout_flee, blind_flee, awayΔ, note) for a sub-mask."""
    s_in = scouts & mask
    b_in = (~scouts) & mask
    n_scouts = int(s_in.sum())
    n_blind  = int(b_in.sum())
    if n_scouts < 5:
        return n_scouts, n_blind, float("nan"), float("nan"), float("nan"), "(insufficient scouts)"
    s_flee = float((actions[s_in] == FLEE_ACTION).mean())
    if n_blind < 5:
        b_flee = overall_blind_flee
        note = "(blind rate: global)"
    else:
        b_flee = float((actions[b_in] == FLEE_ACTION).mean())
        note = ""
    return n_scouts, n_blind, s_flee, b_flee, s_flee - b_flee, note


def away_delta(
    actions: np.ndarray,
    scouts: np.ndarray,
    bearing: np.ndarray,
) -> None:
    """Compute per-direction awayΔ and print results table."""

    finite_bear = np.isfinite(bearing)

    print(f"\n{'='*78}")
    print("  DIRECTIONAL awayΔ SPLIT — scouts vs blind by red bearing quadrant")
    print(f"  total records: {len(actions):,}   scouts: {scouts.sum():,}   blind: {(~scouts).sum():,}")
    print(f"{'='*78}")
    print(
        f"  {'dir':>4}  {'n_scouts':>9}  {'n_blind':>8}  "
        f"{'scout_flee%':>11}  {'blind_flee%':>11}  {'awayΔ':>8}  {'note'}"
    )
    print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*20}")

    overall_scout_flee = 0.0
    overall_blind_flee = 0.0
    if scouts.sum() > 0:
        overall_scout_flee = (actions[scouts] == FLEE_ACTION).mean()
    if (~scouts).sum() > 0:
        overall_blind_flee = (actions[~scouts] == FLEE_ACTION).mean()

    for quad in ["N", "E", "S", "W"]:
        quad_mask = np.zeros(len(bearing), dtype=bool)
        for i, b in enumerate(bearing):
            if np.isfinite(b):
                quad_mask[i] = in_quadrant(b, quad)

        s_in = scouts & quad_mask
        b_in = (~scouts) & quad_mask

        n_scouts = s_in.sum()
        n_blind  = b_in.sum()

        if n_scouts < 5:
            s_flee_rate = float("nan")
            away = float("nan")
            note = "(insufficient scouts)"
        else:
            s_flee_rate = (actions[s_in] == FLEE_ACTION).mean()
            if n_blind < 5:
                b_flee_rate = overall_blind_flee
                note = "(blind rate: global)"
            else:
                b_flee_rate = (actions[b_in] == FLEE_ACTION).mean()
                note = ""
            away = s_flee_rate - b_flee_rate

        def pct(x):
            if x != x:
                return "    nan"
            return f"{x*100:6.1f}%"

        def delta(x):
            if x != x:
                return "    nan"
            return f"{x:+.3f}"

        print(
            f"  {quad:>4}  {n_scouts:>9,}  {n_blind:>8,}  "
            f"{pct(s_flee_rate):>11}  {pct(b_flee_rate if n_scouts >= 5 else float('nan')):>11}  "
            f"{delta(away):>8}  {note}"
        )

    print(f"\n  Overall scout flee rate:  {overall_scout_flee*100:.1f}%")
    print(f"  Overall blind flee rate:  {overall_blind_flee*100:.1f}%")
    print(f"  Overall awayΔ:            {(overall_scout_flee - overall_blind_flee):+.3f}")
    print(f"\n  Interpretation:")
    print("    E/W binary load-bearing → awayΔ should be materially higher for E or W")
    print("    than for N/S. Flat awayΔ across all dirs → encoding is decorative.")
    print(f"{'='*78}\n")


def crowding_split(
    actions: np.ndarray,
    scouts: np.ndarray,
    neighbors: np.ndarray,
) -> None:
    """Split agents at median neighbor density and compute awayΔ for each half.

    Prediction: if social dims (29/30/31/27) are load-bearing:
      isolated (low-crowding) agents → higher awayΔ (flee more aggressively alone)
      clustered (high-crowding) agents → lower awayΔ (safety in numbers)
    """
    finite_nbr = np.isfinite(neighbors)
    median_nbr = float(np.median(neighbors[finite_nbr])) if finite_nbr.sum() > 0 else 0.0

    overall_blind_flee = 0.0
    if (~scouts).sum() > 0:
        overall_blind_flee = float((actions[~scouts] == FLEE_ACTION).mean())

    def pct(x):
        return "    nan" if x != x else f"{x*100:6.1f}%"

    def delta(x):
        return "    nan" if x != x else f"{x:+.3f}"

    print(f"\n{'='*78}")
    print("  SOCIAL DIM BEHAVIORAL SPLIT — awayΔ by crowding level")
    print(f"  median neighbors: {median_nbr:.4f}   total: {len(actions):,}")
    print(f"  scouts: {scouts.sum():,}   blind: {(~scouts).sum():,}")
    print(f"{'='*78}")
    print(
        f"  {'group':>12}  {'n_scouts':>9}  {'n_blind':>8}  "
        f"{'scout_flee%':>11}  {'blind_flee%':>11}  {'awayΔ':>8}  {'note'}"
    )
    print(f"  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*20}")

    groups = {
        "low-crowd": finite_nbr & (neighbors <= median_nbr),
        "high-crowd": finite_nbr & (neighbors > median_nbr),
        "overall": np.ones(len(actions), dtype=bool),
    }

    results = {}
    for label, mask in groups.items():
        ns, nb, sf, bf, aw, note = _away_row(actions, scouts, mask, overall_blind_flee)
        results[label] = aw
        print(
            f"  {label:>12}  {ns:>9,}  {nb:>8,}  "
            f"{pct(sf):>11}  {pct(bf):>11}  {delta(aw):>8}  {note}"
        )

    low  = results.get("low-crowd", float("nan"))
    high = results.get("high-crowd", float("nan"))
    if low == low and high == high:
        diff = low - high
        print(f"\n  awayΔ(isolated) − awayΔ(clustered) = {diff:+.3f}")
        if diff > 0.02:
            print("  → isolated agents flee more aggressively: social dim load-bearing ✅")
        elif diff < -0.02:
            print("  → clustered agents flee more aggressively: unexpected pattern")
        else:
            print("  → flat: social dim not yet behaviorally load-bearing")

    print(f"{'='*78}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Directional awayΔ split")
    ap.add_argument("corpus", help="Path to signal_corpus.jsonl (can be merged)")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=0)
    ap.add_argument(
        "--group-by", choices=["bearing", "crowding"], default="bearing",
        help="Split mode: 'bearing' (default) = per cardinal quadrant; "
             "'crowding' = high vs low neighbor density",
    )
    args = ap.parse_args()

    print(f"Loading {args.corpus} ...", end=" ", flush=True)
    actions, scouts, bearing, neighbors, steps = load(args.corpus, args.min_step, args.max_step)
    print(
        f"{len(actions):,} records"
        + (f", steps {steps.min()}–{steps.max()}" if len(steps) > 0 else "")
        + f"  ({np.isfinite(bearing).sum():,} with valid bearing)"
    )

    if args.group_by == "crowding":
        crowding_split(actions, scouts, neighbors)
    else:
        away_delta(actions, scouts, bearing)


if __name__ == "__main__":
    main()
