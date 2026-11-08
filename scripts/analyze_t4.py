"""Registered analysis: T4 (MAPPO-style centralized critic for message credit) under the erosion protocol with the
reliance-first bar (docs/RESEARCH_PROTOCOL.md, 2026-11-08). usage: analyze_t4.py <run.jsonl[.gz]> ...
tags T4A_ON / T4A_OFF (centralized V_shared baseline), T4B_ON / T4B_OFF (centralized Q_shared counterfactual).
Per arm, at update 2,000:
  wire-on seed passes the reliance bar: empirical-swap deposit drop >= 30%, paired drop > 2 SE, and listener deposits
    (50 updates before) >= the arm's wire-off mean over the same window (the NOSIG level).
  PASS = reliance in >= 2 of 3 wire-on AND wire-off MI (Miller-Madow, mean over slots) < 0.5 bit in both wire-off.
  Secondary: wire-on MI >= 0.5; reliance retention vs update 1,500.
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

U = 2000


def rows(paths):
    out = []
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        out += [json.loads(l) for l in op(p, "rt") if l.strip()]
    return out


def main(paths):
    D = defaultdict(lambda: defaultdict(list))
    for r in rows(paths):
        D[(r["tag"], r["seed"])][r["phase"]].append(r)
    win = lambda k, u, n=50: float(np.mean([r["listener_dep_per_world"] for r in D[k]["train"] if u - n < r["update"] <= u]))  # noqa: E731
    mi_at = lambda k, u: float(np.mean([r["mi_bits"] for r in D[k]["mi"] if r["update"] == u]))  # noqa: E731

    def drop(k, u):
        P = [p for p in D[k]["swap"] if p["update"] == u and p["mode"] == "empirical"]
        c = np.array([p["listener_dep_control"] for p in P]); s = np.array([p["listener_dep_swap"] for p in P])
        d = c - s
        return 1 - s.mean() / max(c.mean(), 1e-9), d.mean(), d.std(ddof=1) / np.sqrt(len(d))

    for arm in ("T4A", "T4B"):
        on = sorted(k for k in D if k[0] == f"{arm}_ON"); off = sorted(k for k in D if k[0] == f"{arm}_OFF")
        if not (on or off):
            continue
        print(f"== {arm} ({'centralized V_shared baseline' if arm == 'T4A' else 'centralized Q_shared counterfactual'})")
        offmi = [mi_at(k, U) for k in off]
        for k, m in zip(off, offmi):
            print(f"   wire-off s{k[1]}: MI@1500 {mi_at(k, 1500):.3f} -> MI@2000 {m:.3f}")
        erodes = len(off) == 2 and all(m < 0.5 for m in offmi)
        res = []
        for k in on:
            dr, pd, se = drop(k, U); d15 = drop(k, 1500)[0]
            nos = float(np.mean([win(o, U) for o in off]))
            ok = dr >= 0.30 and pd > 2 * se and win(k, U) >= nos
            res.append(ok)
            print(f"   wire-on s{k[1]}: drop@1500 {d15:+.0%} -> @2000 {dr:+.0%} (paired {pd:.2f} SE {se:.2f}; retention "
                  f"{dr / d15 if d15 > 0 else float('nan'):.2f}) | L dep {win(k, U):.2f} vs NOSIG {nos:.2f} | MI@2000 "
                  f"{mi_at(k, U):.3f} (secondary >= 0.5: {mi_at(k, U) >= 0.5}) -> {'pass' if ok else 'fail'}")
        print(f"   wire-off erodes (both < 0.5): {erodes}")
        print(f"   {arm}: {'PASS' if erodes and len(res) == 3 and sum(res) >= 2 else 'FAIL'} (reliance {sum(res)}/{len(res)})\n")


if __name__ == "__main__":
    main(sys.argv[1:])
