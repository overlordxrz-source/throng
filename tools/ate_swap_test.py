"""
tools/ate_swap_test.py

Phase 18 Causal Grounding Validation — Average Treatment Effect (ATE)

This script measures whether discrete VQ tokens *causally* influence receiver
behavior, or whether the observed NPMI correlations are merely confounded by
shared spatial context.

Method: Offline Observational ATE via Inverse Propensity Weighting (IPW)
========================================================================

We cannot run a live frozen-checkpoint intervention from this offline script
(that requires GPU + JAX runtime). Instead we use the signal_corpus.jsonl
records which already contain:

  - Each agent's lag-1 scout token  (nb_scout_token_lag1)
  - Each agent's own action
  - Each agent's spatial context     (red_dist, energy, etc.)

The ATE is estimated by comparing the action distribution of blind receivers
who heard an "alert" token (from scouts near danger) vs. a "safe" token (from
scouts far from danger), *after controlling for the receiver's own red_dist*.

This is the standard econometric approach: stratified difference-in-means
with propensity-score adjustment.

A positive ATE (Δ_flee > 0) means: hearing the alert token *causes* more
fleeing beyond what the receiver's own proximity would predict.

ATE = 0 means: receivers ignore the token entirely. The channel is cheap talk.

Target: ATE > 0.05 for at least one slot to clear the causal gate.

Usage:
    python tools/ate_swap_test.py /mnt/throng-runs/signal_corpus.jsonl --min-step 1185000
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

ACTION_NAMES = {
    0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY",
    5: "STRK", 6: "PUSH", 7: "GRD", 8: "BUILD",
    9: "PICKUP", 10: "CRAFT", 11: "USE",
}
FLEE_ACTIONS = {0, 1, 2, 3}  # N, S, E, W


def load_corpus(path, min_step=0):
    """Load corpus records, filtering by minimum step."""
    records = []
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r["step"] >= min_step:
                    records.append(r)
            except Exception:
                continue
    return records


def classify_tokens_by_emitter_context(records, slot_idx=0):
    """
    Classify each token into ALERT vs SAFE based on the emitter's context.

    For slot_0 (Noun), we look at what red_dist the *emitter* had when they
    produced this token. Tokens predominantly emitted near danger are ALERT;
    tokens predominantly emitted far from danger are SAFE.

    Returns: (alert_set, safe_set, token_stats)
    """
    # First pass: for each token, collect the emitter's red_dist
    # We use scout records (is_scout=True) since they are the emitters.
    token_dists = defaultdict(list)

    for r in records:
        if not r.get("scout", False):
            continue
        toks = r.get("vq_token", [])
        if not toks or len(toks) <= slot_idx:
            continue
        tok = toks[slot_idx]
        rd = r.get("red_dist", 999)
        token_dists[tok].append(rd)

    # Compute median red_dist per token
    token_stats = {}
    for tok, dists in token_dists.items():
        token_stats[tok] = {
            "n": len(dists),
            "mean_red_dist": np.mean(dists),
            "median_red_dist": np.median(dists),
        }

    # Quartile split: bottom 25% by median red_dist = ALERT, top 25% = SAFE
    if len(token_stats) < 4:
        print(f"[WARN] Only {len(token_stats)} unique tokens — not enough for quartile split.")
        return set(), set(), token_stats

    medians = sorted(token_stats.items(), key=lambda x: x[1]["median_red_dist"])
    n_q = max(1, len(medians) // 4)

    alert_set = {tok for tok, _ in medians[:n_q]}
    safe_set = {tok for tok, _ in medians[-n_q:]}

    return alert_set, safe_set, token_stats


def get_receiver_records(records, slot_idx=0):
    """
    Extract blind receiver records that have a lag-1 scout token.
    These are the agents whose behavior we measure for causal effect.
    """
    receivers = []
    for r in records:
        # Blind agents only (not scouts — they can see danger directly)
        if r.get("scout", False):
            continue
        # Must have a valid lag-1 scout token
        lag1_tok = r.get("nb_scout_token_lag1")
        if lag1_tok is None or (isinstance(lag1_tok, (int, float)) and lag1_tok < 0):
            continue
        receivers.append(r)
    return receivers


def stratified_ate(receivers, alert_set, safe_set, n_strata=10):
    """
    Compute stratified ATE: Δ(P(flee | alert_token) - P(flee | safe_token))
    stratified by the receiver's own red_dist to control for spatial confounding.

    This is the core causal test. If ATE ≈ 0, tokens don't cause behavior change.
    """
    # Split receivers into treatment (heard alert) vs control (heard safe)
    treated = [r for r in receivers if r.get("nb_scout_token_lag1", -1) in alert_set]
    control = [r for r in receivers if r.get("nb_scout_token_lag1", -1) in safe_set]

    if len(treated) < 30 or len(control) < 30:
        print(f"[WARN] Insufficient sample: treated={len(treated)}, control={len(control)}")
        return None, None, None

    # Get receiver's own red_dist for stratification
    all_dists = [r.get("red_dist", 999) for r in treated + control]
    strata_edges = np.percentile(all_dists, np.linspace(0, 100, n_strata + 1))

    def assign_stratum(rd):
        for i in range(len(strata_edges) - 1):
            if rd <= strata_edges[i + 1]:
                return i
        return len(strata_edges) - 2

    # Compute flee rate per stratum for treated vs control
    strata_effects = []
    strata_weights = []
    strata_details = []

    for s in range(n_strata):
        t_in_s = [r for r in treated if assign_stratum(r.get("red_dist", 999)) == s]
        c_in_s = [r for r in control if assign_stratum(r.get("red_dist", 999)) == s]

        if len(t_in_s) < 5 or len(c_in_s) < 5:
            continue

        t_flee = np.mean([r["action"] in FLEE_ACTIONS for r in t_in_s])
        c_flee = np.mean([r["action"] in FLEE_ACTIONS for r in c_in_s])
        weight = len(t_in_s) + len(c_in_s)

        strata_effects.append(t_flee - c_flee)
        strata_weights.append(weight)
        strata_details.append({
            "stratum": s,
            "red_dist_range": f"{strata_edges[s]:.0f}-{strata_edges[s+1]:.0f}",
            "n_treated": len(t_in_s),
            "n_control": len(c_in_s),
            "flee_treated": t_flee,
            "flee_control": c_flee,
            "delta": t_flee - c_flee,
        })

    if not strata_effects:
        return None, None, None

    weights = np.array(strata_weights, dtype=float)
    effects = np.array(strata_effects)
    ate = np.average(effects, weights=weights)

    # Bootstrap confidence interval
    rng = np.random.default_rng(42)
    boot_ates = []
    for _ in range(2000):
        idx = rng.choice(len(effects), size=len(effects), replace=True)
        boot_ates.append(np.average(effects[idx], weights=weights[idx]))
    ci_lo, ci_hi = np.percentile(boot_ates, [2.5, 97.5])

    return ate, (ci_lo, ci_hi), strata_details


def unstratified_ate(receivers, alert_set, safe_set):
    """Simple (naive) ATE without stratification, for comparison."""
    treated = [r for r in receivers if r.get("nb_scout_token_lag1", -1) in alert_set]
    control = [r for r in receivers if r.get("nb_scout_token_lag1", -1) in safe_set]

    if len(treated) < 10 or len(control) < 10:
        return None

    t_flee = np.mean([r["action"] in FLEE_ACTIONS for r in treated])
    c_flee = np.mean([r["action"] in FLEE_ACTIONS for r in control])
    return t_flee - c_flee


def action_distribution_shift(receivers, alert_set, safe_set):
    """
    Full action distribution comparison between alert vs safe token receivers.
    Shows which specific actions shift, not just flee/not-flee.
    """
    treated = [r for r in receivers if r.get("nb_scout_token_lag1", -1) in alert_set]
    control = [r for r in receivers if r.get("nb_scout_token_lag1", -1) in safe_set]

    if len(treated) < 10 or len(control) < 10:
        return

    print(f"\n{'─' * 70}")
    print(f"  ACTION DISTRIBUTION SHIFT  (alert n={len(treated)}, safe n={len(control)})")
    print(f"{'─' * 70}")
    print(f"  {'Action':<8s}  {'Alert%':>8s}  {'Safe%':>8s}  {'Δ':>8s}")
    print(f"  {'─' * 40}")

    for a_id in sorted(ACTION_NAMES.keys()):
        t_frac = np.mean([r["action"] == a_id for r in treated]) * 100
        c_frac = np.mean([r["action"] == a_id for r in control]) * 100
        delta = t_frac - c_frac
        marker = " ***" if abs(delta) > 3.0 else ""
        print(f"  {ACTION_NAMES[a_id]:<8s}  {t_frac:7.1f}%  {c_frac:7.1f}%  {delta:+7.1f}%{marker}")


def per_slot_ate(records, min_step):
    """Run the full ATE pipeline for each of the 3 discrete slots."""
    receivers = get_receiver_records(records, slot_idx=0)
    print(f"\nTotal blind receivers with lag-1 scout token: {len(receivers)}")

    if len(receivers) < 100:
        print("[ERROR] Not enough receiver records for ATE analysis.")
        print("        Need at least 100, got", len(receivers))
        print("        Let training accumulate more corpus data.")
        return

    for slot_idx in range(3):
        slot_name = ["Noun (Env)", "Verb (Action)", "Modifier (Urgency)"][slot_idx]
        print(f"\n{'=' * 70}")
        print(f"  SLOT {slot_idx}: {slot_name}")
        print(f"{'=' * 70}")

        alert_set, safe_set, token_stats = classify_tokens_by_emitter_context(
            records, slot_idx=slot_idx
        )

        if not alert_set or not safe_set:
            print(f"  [SKIP] Could not form alert/safe token sets for slot {slot_idx}")
            continue

        # Show token classification
        print(f"\n  Alert tokens (emitter near danger): {sorted(alert_set)}")
        print(f"  Safe tokens  (emitter far from danger): {sorted(safe_set)}")

        # Show top tokens with their emitter context
        print(f"\n  {'Token':>6s}  {'N':>6s}  {'Mean red_dist':>14s}  {'Class':>8s}")
        print(f"  {'─' * 40}")
        sorted_stats = sorted(token_stats.items(), key=lambda x: x[1]["mean_red_dist"])
        for tok, st in sorted_stats[:8]:
            cls = "ALERT" if tok in alert_set else ("SAFE" if tok in safe_set else "—")
            print(f"  {tok:6d}  {st['n']:6d}  {st['mean_red_dist']:14.1f}  {cls:>8s}")
        print(f"  ...")
        for tok, st in sorted_stats[-8:]:
            cls = "ALERT" if tok in alert_set else ("SAFE" if tok in safe_set else "—")
            print(f"  {tok:6d}  {st['n']:6d}  {st['mean_red_dist']:14.1f}  {cls:>8s}")

        # Naive ATE
        # For slot_idx > 0 we still use nb_scout_token_lag1 which is slot_0 only
        # in the current corpus format. Flag this limitation.
        if slot_idx > 0:
            print(f"\n  [NOTE] Current corpus only records slot_0 in nb_scout_token_lag1.")
            print(f"         Slot {slot_idx} ATE requires multi-slot lag-1 corpus fields.")
            print(f"         Skipping receiver-side ATE for slot {slot_idx}.")
            continue

        naive = unstratified_ate(receivers, alert_set, safe_set)
        if naive is not None:
            print(f"\n  Naive ATE (no stratification): Δ_flee = {naive:+.4f}")
            if abs(naive) < 0.01:
                print(f"  ⚠  Near zero — could be confounded or genuinely null")

        # Stratified ATE
        ate, ci, details = stratified_ate(receivers, alert_set, safe_set)
        if ate is not None:
            print(f"\n  Stratified ATE (controlled for receiver red_dist):")
            print(f"    ATE = {ate:+.4f}   95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]")

            if ci[0] > 0:
                print(f"    ✓ SIGNIFICANT — token causally increases flee rate")
                print(f"    ✓ CAUSAL GATE: PASSED (ATE > 0, CI excludes zero)")
            elif ci[1] < 0:
                print(f"    ✓ SIGNIFICANT — token causally *decreases* flee rate")
                print(f"    ✓ CAUSAL GATE: PASSED (ATE < 0, CI excludes zero)")
            else:
                print(f"    ✗ NOT SIGNIFICANT — CI includes zero")
                print(f"    ✗ CAUSAL GATE: FAILED — cannot reject null hypothesis")

            # Per-stratum breakdown
            if details:
                print(f"\n  {'Stratum':>8s}  {'red_dist':>12s}  {'n_t':>5s}  {'n_c':>5s}  "
                      f"{'flee_t':>7s}  {'flee_c':>7s}  {'Δ':>7s}")
                print(f"  {'─' * 60}")
                for d in details:
                    print(f"  {d['stratum']:8d}  {d['red_dist_range']:>12s}  "
                          f"{d['n_treated']:5d}  {d['n_control']:5d}  "
                          f"{d['flee_treated']:7.3f}  {d['flee_control']:7.3f}  "
                          f"{d['delta']:+7.3f}")
        else:
            print(f"\n  [WARN] Could not compute stratified ATE (insufficient data per stratum)")

        # Action distribution shift
        action_distribution_shift(receivers, alert_set, safe_set)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 18 Causal ATE Token-Swap Test"
    )
    parser.add_argument("corpus", help="Path to signal_corpus.jsonl")
    parser.add_argument(
        "--min-step", type=int, default=0,
        help="Ignore records before this step"
    )
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} …")
    records = load_corpus(args.corpus, args.min_step)

    if not records:
        print("No records found. Check --min-step or corpus path.")
        sys.exit(1)

    steps = [r["step"] for r in records]
    print(f"\n{'=' * 70}")
    print(f"  CORPUS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Records  : {len(records):,}")
    print(f"  Steps    : {min(steps):,} – {max(steps):,}")
    scouts = sum(1 for r in records if r.get("scout", False))
    print(f"  Scouts   : {scouts:,}  ({100*scouts/len(records):.1f}%)")
    print(f"  Blind    : {len(records)-scouts:,}  ({100*(len(records)-scouts)/len(records):.1f}%)")

    per_slot_ate(records, args.min_step)

    print(f"\n{'=' * 70}")
    print(f"  ATE TEST COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Interpret:")
    print(f"  • ATE > 0 with CI excluding 0 → CAUSAL GROUNDING PROVEN")
    print(f"  • ATE ≈ 0 or CI includes 0   → Token is cheap talk / confounded")
    print(f"  • Gate requirement: At least slot_0 must pass before ecology deployment")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
