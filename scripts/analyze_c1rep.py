"""Registered analysis: C1-E replication (Cam's item 4, 2026-11-08 night; docs/RESEARCH_PROTOCOL.md).

usage: analyze_c1rep.py <run.jsonl[.gz]> ...   tags C1REP_ON (wire-blind head, erosion protocol; 5 fresh seeds) / C1REP_OFF (2)
Reliance per seed at u (empirical swap, 10 pairs): drop >= 30%, paired drop > 2 SE, listener deliveries over the 50 updates
  before u >= the wire-off mean over the same window (the NOSIG level).  [as analyze_c1.py]
PASS (at 2,000) iff reliance in >= 3 of 5 wire-on seeds AND pooled drop >= 30% (1 - sum of swap deliveries / sum of control
  deliveries over all 50 wire-on empirical pairs) AND wire-off MI (Miller-Madow, mean over slots) < 0.5 bit in both.
Reported, not gated: redirection per seed (Crafts at T', swap >= 2x control and paired increase > 2 SE), the mean of the
  per-seed drops, the formation check at 1,500, and -- if a continuation's rows are present -- the same readouts at 2,500 /
  3,000 (the continuation's bar is Cam's to set).
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
    on = sorted(k for k in D if k[0] == "C1REP_ON"); off = sorted(k for k in D if k[0] == "C1REP_OFF")
    win = lambda k, u, n=50: float(np.mean([r["listener_dep_per_world"] for r in D[k]["train"] if u - n < r["update"] <= u]))  # noqa: E731
    mi = lambda k, u: float(np.mean([r["mi_bits"] for r in D[k]["mi"] if r["update"] == u]))  # noqa: E731
    us = sorted({p["update"] for k in on for p in D[k]["swap"]})
    for u in us:
        nos = float(np.mean([win(o, u) for o in off])) if all(any(r["update"] == u for r in D[o]["train"]) for o in off) else float("nan")
        print(f"\n== update {u} (NOSIG level {nos:.2f})")
        rel, red, drops, C, S = [], [], [], [], []
        for k in on:
            P = [p for p in D[k]["swap"] if p["update"] == u and p["mode"] == "empirical"]
            if not P:
                continue
            c = np.array([p["listener_dep_control"] for p in P]); s = np.array([p["listener_dep_swap"] for p in P])
            d = c - s; se = d.std(ddof=1) / np.sqrt(len(d)); drop = 1 - s.mean() / max(c.mean(), 1e-9)
            rc = np.array([p["listener_crafts_at_Tprime_control"] for p in P]); rs = np.array([p["listener_crafts_at_Tprime_swap"] for p in P])
            rd = rs - rc; rse = rd.std(ddof=1) / np.sqrt(len(rd))
            ok = bool(drop >= 0.30 and d.mean() > 2 * se and win(k, u) >= nos); okr = bool(rs.mean() >= 2 * rc.mean() and rd.mean() > 2 * rse)
            rel.append(ok); red.append(okr); drops.append(drop); C += c.tolist(); S += s.tolist()
            print(f"   wire-on s{k[1]}: drop {drop:+.0%} paired {d.mean():.2f} SE {se:.2f} | L dep {win(k, u):.2f} -> reliance "
                  f"{'ok' if ok else 'x'} | crafts@T' {rc.mean():.2f} -> {rs.mean():.2f} redirection {'ok' if okr else 'x'} | MI {mi(k, u):.3f}")
        offmi = [mi(o, u) for o in off if D[o]["mi"] and any(r["update"] == u for r in D[o]["mi"])]
        for o, m in zip(off, offmi):
            print(f"   wire-off s{o[1]}: MI {m:.3f}")
        pooled = 1 - np.sum(S) / max(np.sum(C), 1e-9)
        print(f"   reliance {sum(rel)}/{len(rel)} | redirection {sum(red)}/{len(red)} | pooled drop {pooled:+.0%} | mean per-seed drop "
              f"{np.mean(drops):+.0%} | wire-off MI < 0.5 in both: {len(offmi) == 2 and all(m < 0.5 for m in offmi)}")
        if u == 2000:
            ok = len(rel) == 5 and sum(rel) >= 3 and pooled >= 0.30 and len(offmi) == 2 and all(m < 0.5 for m in offmi)
            print(f"\n== C1-E REPLICATION (registered, at 2,000): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1:])
