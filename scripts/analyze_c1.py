"""Registered analysis: C1, the wire-blind message head from the start (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_c1.py <run.jsonl[.gz]> ...   tags C1T3_ON / C1T3_OFF (T3 protocol), C1E_ON / C1E_OFF (erosion protocol)
Reliance at update u (empirical swap, 10 pairs): drop >= 30%, paired drop > 2 SE, and listener deliveries over the 50
updates before u >= the same protocol's wire-off mean over that window (the NOSIG level).
FORMATION GATE (each protocol): reliance at 1,500 in >= 2 of 3 wire-on seeds; otherwise stop and report.
C1-T3 READING ("the coupling is the destabiliser"): in >= 2 of 3 wire-on seeds, continuity across the first sender
  reset (at 2,000 vs the map at 1,750) >= 0.5 AND exact-triple copy accuracy on the elder's inputs at 2,000 >= half
  its post-imitation value at 1,750 (same 4,096 sampled rows).
C1-E (rung-3 bar, unchanged): reliance at 2,000 in >= 2 of 3 wire-on AND wire-off MI (Miller-Madow, mean over
  slots) < 0.5 bit at 2,000 in both wire-off seeds.
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np


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
    keys = lambda t: sorted(k for k in D if k[0] == t)  # noqa: E731
    win = lambda k, u, n=50: float(np.mean([r["listener_dep_per_world"] for r in D[k]["train"] if u - n < r["update"] <= u]))  # noqa: E731
    mi = lambda k, u: float(np.mean([r["mi_bits"] for r in D[k]["mi"] if r["update"] == u]))  # noqa: E731

    def rel(k, u, off):
        P = [p for p in D[k]["swap"] if p["update"] == u and p["mode"] == "empirical"]
        c = np.array([p["listener_dep_control"] for p in P]); s = np.array([p["listener_dep_swap"] for p in P])
        d = c - s; se = d.std(ddof=1) / np.sqrt(len(d)); drop = 1 - s.mean() / max(c.mean(), 1e-9)
        nos = float(np.mean([win(o, u) for o in off]))
        return drop, d.mean(), se, win(k, u), nos, bool(drop >= 0.30 and d.mean() > 2 * se and win(k, u) >= nos)

    gates = {}
    for proto in ("C1T3", "C1E"):
        on, off = keys(f"{proto}_ON"), keys(f"{proto}_OFF")
        if not on:
            continue
        print(f"== {proto}: formation gate at 1,500")
        g = []
        for k in on:
            dr, pd, se, ld, nos, ok = rel(k, 1500, off); g.append(ok)
            print(f"   s{k[1]}: drop {dr:+.0%} paired {pd:.2f} SE {se:.2f} | L dep {ld:.2f} vs NOSIG {nos:.2f} -> {'ok' if ok else 'x'}"
                  f" | MI@1500 {mi(k, 1500):.3f}")
        gates[proto] = len(g) == 3 and sum(g) >= 2
        print(f"   formation gate: {'PASS' if gates[proto] else 'FAIL -> stop and report'} ({sum(g)}/{len(g)})")

    on, off = keys("C1T3_ON"), keys("C1T3_OFF")
    if on:
        print("\n== C1-T3 reading (continuity across the first reset >= 0.5 AND exact copy at 2,000 >= half post-imitation)")
        res = []
        for k in on:
            tv = {r["update"]: r for r in D[k]["turnover"]}
            cont = tv.get(2000, {}).get("continuity", float("nan"))
            imit = [r for r in D[k]["reset"] if r.get("kind") == "sender" and "imitation_acc_after" in r]
            post = imit[0]["imitation_acc_after"] if imit else float("nan")
            ca = [r for r in D[k]["copy_acc"] if r["update"] == 2000]
            cp = ca[-1]["copy_acc"] if ca else float("nan")
            ok = cont >= 0.5 and cp >= 0.5 * post
            res.append(ok)
            dr, pd, se, ld, nos, rok = rel(k, 2000, off)
            print(f"   s{k[1]}: continuity@2000 {cont:.3f} | copy exact post-imitation {post:.3f} -> @2000 {cp:.3f} -> "
                  f"{'yes' if ok else 'no'} | (reported) reliance@2000 {dr:+.0%}, MI@2000 {mi(k, 2000):.3f}")
        for k in off:
            tv = {r["update"]: r for r in D[k]["turnover"]}
            print(f"   wire-off s{k[1]}: continuity@2000 {tv.get(2000, {}).get('continuity', float('nan')):.3f} | MI@2000 {mi(k, 2000):.3f}")
        print(f"   C1-T3: coupling {'CONFIRMED as the destabiliser' if len(res) == 3 and sum(res) >= 2 else 'NOT confirmed'} "
              f"({sum(res)}/{len(res)}){'' if gates.get('C1T3') else '  [formation gate failed: reading is moot]'}")

    on, off = keys("C1E_ON"), keys("C1E_OFF")
    if on:
        print("\n== C1-E rung-3 bar at 2,000")
        offmi = [mi(k, 2000) for k in off]
        res = []
        for k in on:
            dr, pd, se, ld, nos, ok = rel(k, 2000, off); res.append(ok)
            print(f"   wire-on s{k[1]}: drop {dr:+.0%} paired {pd:.2f} SE {se:.2f} | L dep {ld:.2f} vs NOSIG {nos:.2f} | MI@2000 "
                  f"{mi(k, 2000):.3f} -> {'pass' if ok else 'fail'}")
        for k, m in zip(off, offmi):
            print(f"   wire-off s{k[1]}: MI@2000 {m:.3f}")
        erodes = len(off) == 2 and all(m < 0.5 for m in offmi)
        print(f"   C1-E: {'PASS' if erodes and len(res) == 3 and sum(res) >= 2 else 'FAIL'} (reliance {sum(res)}/{len(res)}; wire-off "
              f"erodes {erodes}){'' if gates.get('C1E') else '  [formation gate failed: reading is moot]'}")


if __name__ == "__main__":
    main(sys.argv[1:])
