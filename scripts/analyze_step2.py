"""Registered analysis: build-order step 2, channel A/B (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_step2.py <run.jsonl[.gz]> ...   tags S2_VQ, S2_SYM, S2_NOSIG (all with the wire-blind message head)
At update 2,000 (empirical swap, 10 pairs):
  reliance per seed: drop >= 30%, paired drop > 2 SE, listener deliveries (last 50) >= the NOSIG mean (same window).
  redirection per seed: listener Crafts at the swapped fire, swap >= 2x control, paired increase > 2 SE.
  SYM ADOPTED iff SYM reliance passes in >= 4 of 5 seeds AND mean SYM drop >= mean VQ drop - 0.10. Otherwise VQ stays.
  If both arms fail their own reliance bar (< 4 of 5 each), the A/B is uninformative. Redirection co-reported (4 of 5).
  Secondary: MI at 2,000, formation speed (first update with plug-in MI > 1 bit), listener deliveries, sender
  message entropy.
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
    keys = lambda t: sorted(k for k in D if k[0] == t)  # noqa: E731
    win = lambda k, n=50: float(np.mean([r["listener_dep_per_world"] for r in D[k]["train"] if U - n < r["update"] <= U]))  # noqa: E731
    nos = float(np.mean([win(k) for k in keys("S2_NOSIG")]))
    print(f"== NOSIG listener deliveries/world (last 50): {[round(win(k), 2) for k in keys('S2_NOSIG')]} mean {nos:.2f}")
    summ = {}
    for arm in ("S2_VQ", "S2_SYM"):
        print(f"== {arm}")
        rel, red, drops = [], [], []
        for k in keys(arm):
            P = [p for p in D[k]["swap"] if p["update"] == U and p["mode"] == "empirical"]
            c = np.array([p["listener_dep_control"] for p in P]); s = np.array([p["listener_dep_swap"] for p in P])
            d = c - s; se = d.std(ddof=1) / np.sqrt(len(d)); drop = 1 - s.mean() / max(c.mean(), 1e-9)
            rc = np.array([p["listener_crafts_at_Tprime_control"] for p in P]); rs = np.array([p["listener_crafts_at_Tprime_swap"] for p in P])
            rd = rs - rc; rse = rd.std(ddof=1) / np.sqrt(len(rd))
            ok_rel = drop >= 0.30 and d.mean() > 2 * se and win(k) >= nos
            ok_red = rs.mean() >= 2 * rc.mean() and rd.mean() > 2 * rse
            rel.append(ok_rel); red.append(ok_red); drops.append(drop)
            tr = sorted(D[k]["train"], key=lambda r: r["update"])
            form = next((r["update"] for r in tr if r.get("mi_plugin", 0) > 1.0), None)
            mi = float(np.mean([r["mi_bits"] for r in D[k]["mi"] if r["update"] == U]))
            ent = float(np.mean([r["token_entropy_sender"] for r in tr[-50:]]))
            print(f"   s{k[1]}: drop {drop:+.0%} (paired {d.mean():.2f} SE {se:.2f}) L dep {win(k):.2f} -> reliance {'ok' if ok_rel else 'x'} | "
                  f"crafts@T' {rc.mean():.2f} -> {rs.mean():.2f} -> redirection {'ok' if ok_red else 'x'} | MI {mi:.3f} | "
                  f"MI>1 bit at update {form} | sender msg entropy {ent:.2f}")
        summ[arm] = dict(rel=sum(rel), n=len(rel), red=sum(red), mean_drop=float(np.mean(drops)) if drops else float("nan"))
        print(f"   {arm}: reliance {sum(rel)}/{len(rel)} | redirection {sum(red)}/{len(red)} | mean drop {summ[arm]['mean_drop']:+.0%}")
    v, s = summ.get("S2_VQ"), summ.get("S2_SYM")
    if v and s:
        if v["rel"] < 4 and s["rel"] < 4:
            verdict = "UNINFORMATIVE (both arms fail their own reliance bar)"
        elif s["rel"] >= 4 and s["mean_drop"] >= v["mean_drop"] - 0.10:
            verdict = "SYM ADOPTED"
        else:
            verdict = "VQ STAYS"
        print(f"\n== STEP 2: {verdict} (SYM reliance {s['rel']}/{s['n']}, mean drop {s['mean_drop']:+.0%} vs VQ {v['mean_drop']:+.0%} - 0.10)")


if __name__ == "__main__":
    main(sys.argv[1:])
