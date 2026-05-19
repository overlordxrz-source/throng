#!/usr/bin/env python3
"""Quick decoder pass for Phase 4 token telemetry.

Computes MI between top-N tokens and context features from events.jsonl.
"""
import json
import math
import collections
import sys
from pathlib import Path

EVENTS_PATH = Path("runs_large/run_20260518_233539/events.jsonl")
TOP_N = 10


def bin_energy(e):
    if e is None:
        return "none"
    if e < 0.5:
        return "low"
    if e < 0.9:
        return "med"
    return "high"


def bin_resource(v):
    if v is None:
        return "none"
    if v < 0.3:
        return "low"
    if v < 0.7:
        return "med"
    return "high"


def bin_dist(d):
    if d is None:
        return "none"
    if d < 4:
        return "near"
    if d < 8:
        return "med"
    return "far"


def compute_mi(records, token_id, feature_name, binner):
    joint = collections.Counter()
    tok_total = 0
    feat_counts = collections.Counter()

    for r in records:
        feat_val = binner(r.get(feature_name))
        feat_counts[feat_val] += 1
        if r["token_id"] == token_id:
            joint[feat_val] += 1
            tok_total += 1

    if tok_total == 0:
        return 0.0

    N = len(records)
    mi = 0.0
    for feat_val, joint_cnt in joint.items():
        p_xy = joint_cnt / N
        p_x = tok_total / N
        p_y = feat_counts[feat_val] / N
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))

    return mi


def main():
    records = []
    with open(EVENTS_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "token_telemetry":
                records.append(d)

    print(f"Loaded {len(records)} token telemetry records")

    tok_counts = collections.Counter(r["token_id"] for r in records)
    total = len(records)
    top10 = tok_counts.most_common(TOP_N)

    print(f"\nTop {TOP_N} tokens:")
    for tok, cnt in top10:
        print(f"  Token {tok:>2}: {cnt:>4} ({cnt / total * 100:.1f}%)")

    features = [
        ("local_resource", bin_resource),
        ("on_wall", lambda v: "yes" if v == 1.0 else "no"),
        ("nearest_red_dist", bin_dist),
        ("energy", bin_energy),
    ]

    print("\n" + "=" * 70)
    print("DECODER RESULTS")
    print("=" * 70 + "\n")

    strong = []
    moderate = []
    weak = []
    noise = []

    for tok, cnt in top10:
        pct = cnt / total * 100
        results = []
        for feat_name, binner in features:
            mi = compute_mi(records, tok, feat_name, binner)
            results.append((feat_name, mi))

        results.sort(key=lambda x: x[1], reverse=True)
        best_feat, best_mi = results[0]

        if best_mi > 0.15:
            annot = "*** STRONG ***"
            strong.append((tok, best_feat, best_mi, pct))
        elif best_mi > 0.08:
            annot = "** MODERATE **"
            moderate.append((tok, best_feat, best_mi, pct))
        elif best_mi > 0.03:
            annot = "* WEAK *"
            weak.append((tok, best_feat, best_mi, pct))
        else:
            annot = "noise"
            noise.append((tok, best_feat, best_mi, pct))

        print(f"Token {tok:>2} ({pct:>4.1f}%): {best_feat:<18} MI={best_mi:.4f}  {annot}")
        for feat_name, mi in results[1:]:
            print(f"                {feat_name:<18} MI={mi:.4f}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Strong (MI>0.15):    {len(strong)} tokens")
    if strong:
        for tok, feat, mi, pct in strong:
            print(f"    Token {tok} ({pct:.1f}%) → {feat}  MI={mi:.4f}")
    print(f"  Moderate (MI>0.08):  {len(moderate)} tokens")
    if moderate:
        for tok, feat, mi, pct in moderate:
            print(f"    Token {tok} ({pct:.1f}%) → {feat}  MI={mi:.4f}")
    print(f"  Weak (MI>0.03):      {len(weak)} tokens")
    if weak:
        for tok, feat, mi, pct in weak:
            print(f"    Token {tok} ({pct:.1f}%) → {feat}  MI={mi:.4f}")
    print(f"  Noise (MI<0.03):     {len(noise)} tokens")
    if noise:
        for tok, feat, mi, pct in noise:
            print(f"    Token {tok} ({pct:.1f}%) → {feat}  MI={mi:.4f}")

    # Phase 3 reference: Phase A run hit MI~0.82 for own_energy on continuous signals.
    # Phase 4 discrete tokens with tau still soft may need lower tau for comparable MI.
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if strong:
        print("Tokens show MEANINGFUL semantic correlation. Continue to 100k.")
    elif moderate:
        print("Tokens show moderate context correlation. Likely to sharpen as tau hardens.")
    elif weak:
        print("Weak correlations only. Vocabulary structure exists but may be social grooming.")
    else:
        print("NO semantic correlation detected. Tokens are likely social grooming noise.")
        print("Recommendation: lower signal_entropy_coef_end to 0.002 and resume.")


if __name__ == "__main__":
    main()
