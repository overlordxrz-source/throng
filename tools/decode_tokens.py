"""
tools/decode_tokens.py — Offline decoder for Phase 4/5 discrete token telemetry.

Usage:
    python tools/decode_tokens.py runs_large/run_XXXXXXXX_XXXXXX/events.jsonl --step 5000
    python tools/decode_tokens.py runs_large/run_XXXXXXXX_XXXXXX/events.jsonl --step 30000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY"}

# Context features available in token_telemetry (continuous variables)
CONTEXT_FEATURES = [
    "energy",
    "nearest_red_dist",
    "nearest_red_bear",
    "local_resource",
]


def load_telemetry(path: str, max_step: Optional[int] = None) -> List[dict]:
    """Load token_telemetry events from events.jsonl."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "token_telemetry":
                continue
            if max_step is not None and rec.get("step", 0) > max_step:
                continue
            records.append(rec)
    return records


def telemetry_to_arrays(records: List[dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Convert records to token_id array and context feature dict."""
    n = len(records)
    if n == 0:
        return np.array([], dtype=np.int32), {}

    token_ids = np.array([r["token_id"] for r in records], dtype=np.int32)
    ctx = {}
    for k in CONTEXT_FEATURES:
        vals = np.array([r.get(k, float("nan")) for r in records], dtype=np.float32)
        ctx[k] = vals

    actions = np.array([r.get("action", -1) for r in records], dtype=np.int32)
    ctx["action"] = actions

    # Received tokens: list of neighbor token IDs per agent
    received = [r.get("received_tokens", []) for r in records]
    ctx["received_tokens"] = received

    # Blind flag: agent cannot see reds beyond 5x5 patch (local_obs_radius=2)
    nearest_red = ctx.get("nearest_red_dist", np.full(n, float("nan")))
    ctx["is_blind"] = (nearest_red > 2.0) | ~np.isfinite(nearest_red)

    return token_ids, ctx


def token_frequency(token_ids: np.ndarray, vocab_size: int = 64) -> Tuple[np.ndarray, int]:
    """Return frequency counts and number of active tokens."""
    counts = np.bincount(token_ids, minlength=vocab_size)
    active = int((counts > 0).sum())
    return counts, active


def top_tokens_summary(token_ids: np.ndarray, counts: np.ndarray, n: int = 5) -> None:
    """Print top N most frequent tokens."""
    total = len(token_ids)
    top_idx = np.argsort(counts)[::-1][:n]
    print(f"\n  Top {n} most frequent tokens (total emissions: {total:,}):")
    print(f"  {'Rank':>4}  {'Token':>5}  {'Count':>7}  {'Pct':>6}  {'Bar':>20}")
    print(f"  {'-'*50}")
    for rank, tid in enumerate(top_idx, 1):
        cnt = int(counts[tid])
        pct = 100 * cnt / max(total, 1)
        bar = "█" * int(pct / 2)
        print(f"  {rank:>4}  {tid:>5}  {cnt:>7,}  {pct:>5.1f}%  {bar}")


def mean_context_by_token(
    token_ids: np.ndarray,
    ctx: Dict[str, np.ndarray],
    vocab_size: int = 64,
    min_count: int = 10,
) -> Dict[int, Dict[str, float]]:
    """Compute mean context per token (only for tokens with >= min_count emissions)."""
    profiles: Dict[int, Dict[str, float]] = {}
    counts = np.bincount(token_ids, minlength=vocab_size)

    for tid in range(vocab_size):
        if counts[tid] < min_count:
            continue
        mask = token_ids == tid
        prof = {}
        for k in CONTEXT_FEATURES:
            v = ctx[k][mask]
            v = v[np.isfinite(v)]
            prof[k] = float(v.mean()) if len(v) > 0 else float("nan")
        acts = ctx["action"][mask]
        acts = acts[acts >= 0]
        if len(acts) > 0:
            mode_act = int(np.bincount(acts).argmax())
            prof["dom_action"] = ACTION_NAMES.get(mode_act, str(mode_act))
            prof["action_mode_pct"] = float(np.bincount(acts).max() / len(acts))
        else:
            prof["dom_action"] = "?"
            prof["action_mode_pct"] = 0.0
        profiles[tid] = prof

    return profiles


def print_token_profiles(profiles: Dict[int, Dict[str, float]]) -> None:
    """Print a table of token context profiles."""
    if not profiles:
        print("  (no tokens with enough emissions for profiling)")
        return

    print(f"\n  Token context profiles (mean values per token, min_count=10):")
    header = f"  {'Token':>5}  {'N':>5}  {'Energy':>7}  {'RedDist':>8}  {'RedBear':>8}  {'Resource':>8}  {'DomAct':>6}  {'Act%':>5}"
    print(header)
    print(f"  {'-'*len(header)}")
    for tid in sorted(profiles.keys()):
        p = profiles[tid]
        print(f"  {tid:>5}  {'—':>5}  {p.get('energy', 0):>7.3f}  "
              f"{p.get('nearest_red_dist', 0):>8.1f}  {p.get('nearest_red_bear', 0):>8.1f}  "
              f"{p.get('local_resource', 0):>8.3f}  {p.get('dom_action', '?'):>6}  "
              f"{p.get('action_mode_pct', 0)*100:>4.0f}%")


# ── MI computation: direct discrete formula (no sklearn dependency issues) ──────

def _bin_feature(y: np.ndarray) -> Tuple[np.ndarray, int]:
    """Bin a continuous feature into 3 tertiles. Returns (binned, num_bins)."""
    valid = np.isfinite(y)
    if valid.sum() < 10 or y[valid].std() < 1e-8:
        return np.full(len(y), -1, dtype=np.int32), 0
    yv = y[valid]
    p33, p67 = np.percentile(yv, [33.3, 66.7])
    binned = np.full(len(y), -1, dtype=np.int32)
    binned[valid & (y < p33)] = 0
    binned[valid & (y >= p33) & (y < p67)] = 1
    binned[valid & (y >= p67)] = 2
    return binned, 3


def _compute_mi_binary(token_present: np.ndarray, feature_binned: np.ndarray) -> float:
    """Compute MI between a binary variable and a discrete feature."""
    n = len(token_present)
    joint = {}
    for t in [0, 1]:
        for f in range(3):
            if f < 0:
                continue
            cnt = ((token_present == t) & (feature_binned == f)).sum()
            if cnt > 0:
                joint[(t, f)] = cnt

    tok_count = token_present.sum()
    if tok_count == 0 or tok_count == n:
        return 0.0

    mi = 0.0
    for (t, f), cnt in joint.items():
        p_xy = cnt / n
        p_x = tok_count / n if t == 1 else (n - tok_count) / n
        p_y = (feature_binned == f).sum() / n
        mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def token_mi_analysis(
    token_ids: np.ndarray,
    ctx: Dict[str, np.ndarray],
    vocab_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Compute mutual information between each token (as one-hot) and each context feature.
    Returns dict: feature_name -> array of MI values per token.
    Uses direct discrete MI formula with percentile-based binning.
    """
    n = len(token_ids)
    if n < 50:
        return {}

    mi_results = {}
    for feat in CONTEXT_FEATURES:
        y = ctx[feat]
        y_binned, n_bins = _bin_feature(y)
        if n_bins == 0:
            continue

        mi_per_token = np.zeros(vocab_size, dtype=np.float32)
        for tid in range(vocab_size):
            tok_present = (token_ids == tid).astype(np.int32)
            mi_per_token[tid] = _compute_mi_binary(tok_present, y_binned)

        mi_results[feat] = mi_per_token

    return mi_results


def print_mi_vocabulary_map(
    mi_results: Dict[str, np.ndarray],
    top_n: int = 10,
    vocab_size: int = 64,
) -> None:
    """Print the top-N token vocabulary map: for each token, which context feature has highest MI."""
    if not mi_results:
        print("  (not enough data for MI analysis)")
        return

    feat_names = list(mi_results.keys())
    mi_matrix = np.column_stack([mi_results[f] for f in feat_names])  # (vocab_size, n_features)
    max_mi = mi_matrix.max(axis=1)
    best_feat_idx = mi_matrix.argmax(axis=1)

    # Sort tokens by max MI
    top_tokens = np.argsort(max_mi)[::-1][:top_n]

    print(f"\n  Vocabulary map (top {top_n} tokens by max MI):")
    print(f"  {'Rank':>4}  {'Token':>5}  {'MaxMI':>7}  {'BestFeature':>14}  {'MI_vals':>40}")
    print(f"  {'-'*80}")
    for rank, tid in enumerate(top_tokens, 1):
        best = feat_names[int(best_feat_idx[tid])]
        mi_vals_str = "  ".join(
            f"{f[:6]}={mi_results[f][tid]:.3f}" for f in feat_names
        )
        print(f"  {rank:>4}  {tid:>5}  {max_mi[tid]:>7.4f}  {best:>14}  {mi_vals_str}")


# ── ALARM metric: flee rate for blind agents that received token vs not ─────────

def compute_alarm_metric(
    token_ids: np.ndarray,
    ctx: Dict[str, np.ndarray],
    top_n: int = 10,
    vocab_size: int = 64,
) -> List[Tuple[int, float, float, float, str]]:
    """
    For each top-N token, compute:
      blind_flee_with    = flee rate among blind agents that RECEIVED this token
      blind_flee_without = flee rate among blind agents that did NOT receive it
      Δ = with - without

    Blind = nearest_red_dist > 2 (beyond 5x5 patch, must rely on signals).
    Flee  = action != STAY (agent is moving, not frozen).

    Returns list of (token_id, with_rate, without_rate, delta, annotation).
    """
    n = len(token_ids)
    if n < 50:
        return []

    is_blind = ctx.get("is_blind", np.zeros(n, dtype=bool))
    actions = ctx.get("action", np.full(n, -1, dtype=np.int32))
    received = ctx.get("received_tokens", [[] for _ in range(n)])
    is_flee = (actions >= 0) & (actions != 4)  # action != STAY

    counts = np.bincount(token_ids, minlength=vocab_size)
    top_tokens = np.argsort(counts)[::-1][:top_n]

    results = []
    for tid in top_tokens:
        # Which agents received this token?
        has_token = np.array([tid in recv for recv in received], dtype=bool)

        # Blind + received token + fled
        blind_with = is_blind & has_token
        n_blind_with = blind_with.sum()
        flee_with = (blind_with & is_flee).sum()
        rate_with = flee_with / n_blind_with if n_blind_with > 0 else 0.0

        # Blind + did NOT receive token + fled
        blind_without = is_blind & ~has_token
        n_blind_without = blind_without.sum()
        flee_without = (blind_without & is_flee).sum()
        rate_without = flee_without / n_blind_without if n_blind_without > 0 else 0.0

        delta = rate_with - rate_without

        if delta > 0.10:
            annot = "← danger signal?"
        elif delta < -0.10:
            annot = "← safety signal?"
        else:
            annot = "← neutral"

        results.append((tid, rate_with, rate_without, delta, annot))

    return results


def print_alarm_metric(results: List[Tuple[int, float, float, float, str]]) -> None:
    """Print ALARM metric results."""
    if not results:
        print("  (not enough data for ALARM analysis)")
        return

    print(f"\n  ALARM metric (flee rate of blind agents):")
    print(f"  {'Token':>6}  {'with':>8}  {'without':>8}  {'delta':>7}  {'Annotation':>16}")
    print(f"  {'-'*55}")
    for tid, rate_with, rate_without, delta, annot in results:
        print(f"  Token {tid:>2}:  {rate_with:>6.2f}  {rate_without:>8.2f}  {delta:>+6.2f}  {annot}")


# ── Watch point reports ────────────────────────────────────────────────────────

def watch_point_5k(records: List[dict]) -> bool:
    """Step 5k watch point."""
    print(f"\n{'='*70}")
    print(f"  WATCH POINT: Step 5k  (n_records={len(records):,})")
    print(f"{'='*70}")

    if not records:
        print("  No token telemetry records found yet.")
        return False

    token_ids, ctx = telemetry_to_arrays(records)
    counts, active = token_frequency(token_ids)

    energy = ctx["energy"]
    energy_valid = energy[np.isfinite(energy)]
    mean_energy = float(energy_valid.mean()) if len(energy_valid) > 0 else float("nan")
    print(f"\n  Mean energy: {mean_energy:.3f}")
    if mean_energy > 0.5:
        print("  ✅ Agents are successfully foraging (energy > 0.5)")
    elif mean_energy > 0.3:
        print("  ⚠️  Energy moderate — agents finding some resources")
    else:
        print("  🚨 Energy low (< 0.3) — foraging may be failing")

    surv_frac = float((energy_valid > 0.05).mean()) if len(energy_valid) > 0 else 0.0
    print(f"  Survival proxy (energy > 0.05): {surv_frac:.1%}")

    top_tokens_summary(token_ids, counts, n=5)

    total = len(token_ids)
    top_pct = 100 * counts.max() / max(total, 1)
    print(f"\n  Monoculture check: top token accounts for {top_pct:.1f}% of emissions")
    if top_pct > 40:
        print("  🚨 MONOCULTURE DETECTED — consider bumping entropy_coef")
        return True
    elif top_pct > 25:
        print("  ⚠️  Strong token bias — watch closely")
    else:
        print("  ✅ Token distribution looks healthy")

    return False


def watch_point_20k(records: List[dict]) -> None:
    """Step 20k watch point."""
    print(f"\n{'='*70}")
    print(f"  WATCH POINT: Step 20k  (n_records={len(records):,})")
    print(f"{'='*70}")

    if not records:
        print("  No token telemetry records found.")
        return

    token_ids, ctx = telemetry_to_arrays(records)
    counts, active = token_frequency(token_ids)
    total = len(token_ids)

    print(f"\n  Token vocabulary usage:")
    print(f"    Active tokens: {active}/64  ({100*active/64:.0f}%)")
    print(f"    Total emissions: {total:,}")

    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) > 0:
        print(f"    Mean emissions per active token: {nonzero_counts.mean():.0f}")
        print(f"    Std emissions per active token:  {nonzero_counts.std():.0f}")
        print(f"    Min/Max: {nonzero_counts.min()}/{nonzero_counts.max()}")
        cv = nonzero_counts.std() / (nonzero_counts.mean() + 1e-8)
        print(f"    Coefficient of variation: {cv:.2f}")

    probs = counts / max(total, 1)
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    max_entropy = np.log2(64)
    print(f"    Token entropy: {entropy:.2f} bits  (max={max_entropy:.2f}, uniform={entropy/max_entropy*100:.0f}%)")

    print(f"\n  Assessment:")
    if active < 8:
        print("  🚨 COLLAPSED — fewer than 8 tokens in use.")
    elif active < 15:
        print(f"  ⚠️  Narrow vocabulary — only {active}/64 tokens active")
    elif active <= 25:
        print(f"  ✅ Healthy emergence range — {active} tokens with frequency differentiation")
    elif active > 50 and entropy/max_entropy > 0.9:
        print("  ⚠️  Near-uniform usage — tokens may be emitted randomly")
    else:
        print(f"  ℹ️  Broad vocabulary — {active} tokens active")

    top_tokens_summary(token_ids, counts, n=10)


def watch_point_50k(records: List[dict]) -> None:
    """Step 50k watch point: full vocabulary map + ALARM."""
    print(f"\n{'='*70}")
    print(f"  WATCH POINT: Step 50k  (n_records={len(records):,})")
    print(f"  FIRST VOCABULARY MAP")
    print(f"{'='*70}")

    if not records:
        print("  No token telemetry records found.")
        return

    token_ids, ctx = telemetry_to_arrays(records)
    counts, active = token_frequency(token_ids)

    print(f"\n  Corpus stats:")
    print(f"    Active tokens: {active}/64")
    print(f"    Total emissions: {len(token_ids):,}")

    # MI analysis
    mi_results = token_mi_analysis(token_ids, ctx)
    print_mi_vocabulary_map(mi_results, top_n=10)

    # ALARM metric
    alarm_results = compute_alarm_metric(token_ids, ctx, top_n=10)
    print_alarm_metric(alarm_results)

    # Mean context profiles for top 10 tokens
    print(f"\n  Mean context profiles for top 10 tokens (by frequency):")
    top10 = np.argsort(counts)[::-1][:10]
    profiles = {}
    for tid in top10:
        mask = token_ids == tid
        if mask.sum() < 10:
            continue
        prof = {}
        for k in CONTEXT_FEATURES:
            v = ctx[k][mask]
            v = v[np.isfinite(v)]
            prof[k] = float(v.mean()) if len(v) > 0 else float("nan")
        acts = ctx["action"][mask]
        acts = acts[acts >= 0]
        if len(acts) > 0:
            mode_act = int(np.bincount(acts).argmax())
            prof["dom_action"] = ACTION_NAMES.get(mode_act, str(mode_act))
        profiles[tid] = prof

    if profiles:
        print(f"  {'Token':>5}  {'Energy':>7}  {'RedDist':>8}  {'RedBear':>8}  {'Resource':>8}  {'DomAct':>6}")
        print(f"  {'-'*55}")
        for tid in sorted(profiles.keys(), key=lambda t: counts[t], reverse=True):
            p = profiles[tid]
            print(f"  {tid:>5}  {p.get('energy', 0):>7.3f}  "
                  f"{p.get('nearest_red_dist', 0):>8.1f}  {p.get('nearest_red_bear', 0):>8.1f}  "
                  f"{p.get('local_resource', 0):>8.3f}  {p.get('dom_action', '?'):>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode Phase 4/5 discrete token telemetry")
    parser.add_argument("events_file", help="Path to events.jsonl")
    parser.add_argument("--step", type=int, default=None,
                        help="Filter records to step <= this value")
    parser.add_argument("--watch", choices=["5k", "20k", "50k"], default=None,
                        help="Run a specific watch-point report")
    args = parser.parse_args()

    path = Path(args.events_file)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    max_step = args.step if args.step else None
    if args.watch == "5k":
        max_step = 5000
    elif args.watch == "20k":
        max_step = 20000
    elif args.watch == "50k":
        max_step = 50000

    records = load_telemetry(str(path), max_step=max_step)

    if args.watch == "5k":
        intervention = watch_point_5k(records)
        sys.exit(1 if intervention else 0)
    elif args.watch == "20k":
        watch_point_20k(records)
    elif args.watch == "50k":
        watch_point_50k(records)
    else:
        print(f"Token telemetry decoder — {len(records):,} records")
        if len(records) >= 100:
            watch_point_5k([r for r in records if r.get("step", 0) <= 5000])
        if len(records) >= 500:
            watch_point_20k([r for r in records if r.get("step", 0) <= 20000])
        if len(records) >= 1000:
            watch_point_50k(records)


if __name__ == "__main__":
    main()
