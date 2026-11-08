"""Registered analysis: step B live-wire branch diagnostic (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_stepB.py <branch run .jsonl> ...  (tags STEPB_NONE / STEPB_FROZEN_LISTENER / STEPB_STOPGRAD; seeds
140-142; each resumed from the T3 full-state snapshot at 1,750). Readouts at 2,000, per seed and arm:
  continuity vs the elder's map at 1,750; reliance (empirical swap, 10 pairs: drop, paired SE, listener deliveries
  over updates 1,951-2,000); copy accuracy (exact triple and per slot) vs the elder's (obs, token) pairs.
Registered reading (Cam): (ii) holds the code -> the listener's re-learning drives the drift; (ii) still drifts ->
the sender's own learning does it; (iii) loses reliance -> listeners abandon an imperfect code; anything else is
reported. "Holds the code" is read as: continuity at 2,000 above arm (i)'s in >= 2 of 3 seeds AND above the
per-slot copy level measured right after imitation at 1,750 (Cam's wording: "above the copy level").
"""
import json
import sys
from collections import defaultdict

import numpy as np


def main(paths):
    D = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            D[(r["tag"], r["seed"])][r["phase"]].append(r)
    arms = ("STEPB_NONE", "STEPB_FROZEN_LISTENER", "STEPB_STOPGRAD")
    seeds = sorted({k[1] for k in D})
    out = {}
    for arm in arms:
        print(f"== {arm}")
        for s in seeds:
            k = (arm, s)
            if k not in D:
                print(f"   s{s}: missing"); continue
            tv = [r for r in D[k]["turnover"] if r["update"] == 2000]
            cont = tv[-1].get("continuity", float("nan")) if tv else float("nan")
            P = [p for p in D[k]["swap"] if p["update"] == 2000 and p["mode"] == "empirical"]
            c = np.array([p["listener_dep_control"] for p in P]); sw = np.array([p["listener_dep_swap"] for p in P])
            d = c - sw; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            drop = 1 - sw.mean() / max(c.mean(), 1e-9)
            tr = [r for r in D[k]["train"] if 1950 < r["update"] <= 2000]
            ldep = float(np.mean([r["listener_dep_per_world"] for r in tr]))
            ca = D[k]["copy_acc"][-1] if D[k]["copy_acc"] else {}
            out[k] = dict(cont=cont, drop=drop, ldep=ldep, copy=ca.get("copy_acc"), copy_slot=ca.get("copy_acc_slot"))
            print(f"   s{s}: continuity {cont:.3f} | reliance drop {drop:+.0%} (paired {d.mean():.2f} SE {se:.2f}) | L dep "
                  f"{ldep:.2f} | copy acc exact {ca.get('copy_acc', float('nan')):.3f} per-slot {ca.get('copy_acc_slot', float('nan')):.3f}")
    print("\n== comparisons (continuity vs arm (i); reliance >= 30%)")
    for arm in arms[1:]:
        above = [out[(arm, s)]["cont"] > out[("STEPB_NONE", s)]["cont"] for s in seeds if (arm, s) in out and ("STEPB_NONE", s) in out]
        rel = [out[(arm, s)]["drop"] >= 0.30 for s in seeds if (arm, s) in out]
        print(f"   {arm}: continuity above (i) in {sum(above)}/{len(above)} seeds | reliance >= 30% in {sum(rel)}/{len(rel)}")


if __name__ == "__main__":
    main(sys.argv[1:])
