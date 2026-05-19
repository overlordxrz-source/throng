"""
tools/es_probe.py — Scout/blind dissociation probe for E/S-encoding dims.

For a given set of signal dims (default: the 10 E/S dims identified at step 20k),
compute per-dim Spearman correlations with own_energy and red_bearing
separately for scouts vs blind agents, then print a dissociation table.

Dissociation signature of real directional-danger encoding:
    scouts:  r(dim, red_bearing) high — dim tracks where the red is
    blind:   r(dim, energy)     high — dim influences agent's energy budget

If both populations show the same pattern → just global energy tracking.
If scouts show bearing correlation while blind show energy correlation → the
signal carries directional danger info and blind agents are using it to manage
energy (flee when signal is high, coast when not).

Usage:
    python tools/es_probe.py <corpus.jsonl> [--dims 2 5 6 7 9 10 11 21 30 31]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

import numpy as np
from scipy.stats import spearmanr


ES_DIMS_DEFAULT = [2, 5, 6, 7, 9, 10, 11, 21, 30, 31]


def load(path: str, min_step: int = 0, max_step: int = 0):
    signals, scouts, energy, bearing, neighbors, steps = [], [], [], [], [], []
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
            signals.append(r["sig"])
            scouts.append(r["scout"])
            energy.append(r.get("energy", float("nan")))
            bearing.append(r.get("red_bear", float("nan")))
            neighbors.append(r.get("neighbors", float("nan")))
            steps.append(r.get("step", 0))

    signals   = np.array(signals,   dtype=np.float32)
    scouts    = np.array(scouts,    dtype=bool)
    energy    = np.array(energy,    dtype=np.float32)
    bearing   = np.array(bearing,   dtype=np.float32)
    neighbors = np.array(neighbors, dtype=np.float32)
    steps     = np.array(steps,     dtype=np.int64)
    return signals, scouts, energy, bearing, neighbors, steps


def spear(x: np.ndarray, y: np.ndarray):
    """Spearman r, returning (r, p). NaN-safe: drops rows where either is nan."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


def dissociation_table(
    signals: np.ndarray,
    scouts: np.ndarray,
    energy: np.ndarray,
    bearing: np.ndarray,
    neighbors: np.ndarray,
    dims: List[int],
) -> None:
    s_mask = scouts
    b_mask = ~scouts

    print(f"\n{'='*100}")
    print("  E/S DIM DISSOCIATION PROBE — scouts vs blind (+ social crowding)")
    print(f"  scouts: {s_mask.sum():,}   blind: {b_mask.sum():,}")
    print(f"{'='*100}")
    print(
        f"  {'dim':>4}  "
        f"{'scout r(bear)':>13}  {'blind r(bear)':>13}  "
        f"{'scout r(nrg)':>12}  {'blind r(nrg)':>12}  "
        f"{'scout r(nbr)':>12}  {'blind r(nbr)':>12}  "
        f"{'verdict'}"
    )
    print(f"  {'-'*4}  {'-'*13}  {'-'*13}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*28}")

    for d in dims:
        sig = signals[:, d]

        sr_bear, sr_bear_p = spear(sig[s_mask], bearing[s_mask])
        br_bear, br_bear_p = spear(sig[b_mask], bearing[b_mask])
        sr_nrg,  sr_nrg_p  = spear(sig[s_mask], energy[s_mask])
        br_nrg,  br_nrg_p  = spear(sig[b_mask], energy[b_mask])
        sr_nbr,  sr_nbr_p  = spear(sig[s_mask], neighbors[s_mask])
        br_nbr,  br_nbr_p  = spear(sig[b_mask], neighbors[b_mask])

        def fmt(r, p):
            if r != r:  # nan
                return f"{'nan':>7}"
            star = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            return f"{r:+.3f}{star}"

        # Verdict
        scout_bear_sig = abs(sr_bear) > 0.05 and sr_bear_p < 0.05
        blind_nrg_sig  = abs(br_nrg)  > 0.05 and br_nrg_p  < 0.05
        scout_nrg_sig  = abs(sr_nrg)  > 0.05 and sr_nrg_p  < 0.05
        blind_bear_sig = abs(br_bear) > 0.05 and br_bear_p < 0.05
        scout_nbr_sig  = abs(sr_nbr)  > 0.05 and sr_nbr_p  < 0.05
        blind_nbr_sig  = abs(br_nbr)  > 0.05 and br_nbr_p  < 0.05

        if scout_bear_sig and blind_nrg_sig and not scout_nrg_sig:
            verdict = "✅ DISSOCIATED  (bear→scout, nrg→blind)"
        elif scout_bear_sig and blind_nrg_sig:
            verdict = "⚠  partial diss (bear>nrg in scout)"
        elif scout_bear_sig:
            verdict = "→  scout-bear only"
        elif blind_nrg_sig:
            verdict = "→  blind-nrg only"
        elif (scout_nbr_sig or blind_nbr_sig) and not scout_bear_sig and not blind_nrg_sig:
            verdict = "🔵 social/crowding signal (freed dim?)"
        else:
            verdict = "   weak / noise"

        print(
            f"  {d:>4}  "
            f"{fmt(sr_bear, sr_bear_p):>13}  {fmt(br_bear, br_bear_p):>13}  "
            f"{fmt(sr_nrg, sr_nrg_p):>12}  {fmt(br_nrg, br_nrg_p):>12}  "
            f"{fmt(sr_nbr, sr_nbr_p):>12}  {fmt(br_nbr, br_nbr_p):>12}  "
            f"{verdict}"
        )

    print(f"{'='*100}")
    print("  Dissociation key:")
    print("    ✅ DISSOCIATED      = scouts encode bearing, blind track energy → directional danger")
    print("    ⚠  partial         = pattern present but incomplete")
    print("    → one-sided        = only one half of dissociation present")
    print("    🔵 social/crowding  = dim freed from danger, picking up social structure (3rd domain?)")
    print(f"{'='*100}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scout/blind E/S dissociation probe")
    ap.add_argument("corpus", help="Path to signal_corpus.jsonl (can be merged)")
    ap.add_argument(
        "--dims", nargs="+", type=int, default=ES_DIMS_DEFAULT,
        help="Signal dims to probe (default: E/S dims from step 20k)",
    )
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading {args.corpus} ...", end=" ", flush=True)
    signals, scouts, energy, bearing, neighbors, steps = load(args.corpus, args.min_step, args.max_step)
    print(f"{len(signals):,} records, steps {steps.min()}–{steps.max()}")

    dissociation_table(signals, scouts, energy, bearing, neighbors, args.dims)


if __name__ == "__main__":
    main()
