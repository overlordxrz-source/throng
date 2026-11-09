"""Registered analysis: rung-3 rerun at the calibrated erosion strength (Cam Nov 9 ~07:00 (c), ~07:55; protocol).

usage: analyze_rung3.py <run_dir> <mi_dir>
  run_dir: r3_{base,sg,off}_s*.jsonl (plan rung3); mi_dir: mi_r3_off_s{seed}_u3000.jsonl (comm_toy.py --same-spot
  --mi-null 100 on each wire-off seed's u3000 params).
Per wire-on seed and readout u in {2500, 3000} (empirical swap, 10 pairs):
  reliance = drop >= 30% AND paired drop > 2 SE AND listener deliveries (training window, 50 updates before u) >= the
             wire-off mean over the same window (as C1 / c1rep);
  gain     = the same listener window >= wire-off mean + 1.0 per world (Cam ~02:55: misdirection alone can't pass).
  A seed HOLDS iff reliance and gain hold at BOTH 2,500 and 3,000.
Per arm (base = C1 baseline, sg = stop-grad): PASS iff >= 3 of 5 seeds hold.
Wire-off bar: shuffle-corrected joint MI (MM - permutation-null mean) < 0.05 bits at u3000 in ALL wire-off seeds.
Stop-grad is ADOPTED unless it does worse than baseline, i.e. unless (# sg seeds holding) < (# base seeds holding).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main(run_dir, mi_dir):
    runs = defaultdict(dict)
    for p in glob.glob(os.path.join(run_dir, "r3_*_s*.jsonl")):
        b = os.path.basename(p)[3:-6]; arm, s = b.split("_s"); runs[arm][int(s)] = load(p)
    win = lambda R, u: float(np.mean([r["listener_dep_per_world"] for r in R if r["phase"] == "train" and u - 50 < r["update"] <= u]))  # noqa: E731
    holds = {}
    for u in (2500, 3000):
        nos = float(np.mean([win(R, u) for R in runs["off"].values()]))
        print(f"\n== u{u}: wire-off listener window mean (NOSIG level) {nos:.2f}; gain bar {nos + 1.0:.2f}")
        for arm in ("base", "sg"):
            for s, R in sorted(runs[arm].items()):
                P = [r for r in R if r["phase"] == "swap" and r["update"] == u and r["mode"] == "empirical"]
                c = np.array([p["listener_dep_control"] for p in P]); w = np.array([p["listener_dep_swap"] for p in P])
                d = c - w; se = d.std(ddof=1) / np.sqrt(len(d)); drop = 1 - w.mean() / max(c.mean(), 1e-9); L = win(R, u)
                rel = bool(drop >= 0.30 and d.mean() > 2 * se and L >= nos); gain = bool(L >= nos + 1.0)
                holds.setdefault((arm, s), []).append(rel and gain)
                print(f"   {arm} s{s}: drop {drop:+.0%} paired {d.mean():.2f} SE {se:.2f} | L window {L:.2f} -> reliance "
                      f"{'ok' if rel else 'x'} gain {'ok' if gain else 'x'}")
    n = {arm: sum(all(holds[(arm, s)]) for s in runs[arm]) for arm in ("base", "sg")}
    for arm in ("base", "sg"):
        print(f"\n== {arm}: {n[arm]}/{len(runs[arm])} seeds hold at both readouts -> {'PASS' if n[arm] >= 3 else 'FAIL'}")
    mi = {}
    for p in glob.glob(os.path.join(mi_dir, "mi_r3_off_s*_u3000.jsonl")):
        s = int(os.path.basename(p).split("_s")[1].split("_")[0])
        r = [x for x in load(p) if x["phase"] == "mi_null"][-1]; mi[s] = r["joint_mi_mm"] - r["null_mm_mean"]
    print(f"\n== wire-off shuffle-corrected joint MI at u3000: { {s: round(v, 3) for s, v in sorted(mi.items())} } -> "
          f"{'PASS' if mi and len(mi) == len(runs['off']) and all(v < 0.05 for v in mi.values()) else 'FAIL'} (all < 0.05)")
    print(f"== stop-grad {'ADOPTED' if n['sg'] >= n['base'] else 'NOT adopted (does worse than baseline)'} "
          f"(sg {n['sg']} vs base {n['base']})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
