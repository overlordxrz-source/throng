"""Registered analysis: build-order step 3 -- no assigned sender (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_step3.py <run.jsonl[.gz]> ...   tags PROX (wire on, knowledge within radius 2), ASSIGNED (today's
sender / listener), PROX_NOSIG (wire off).
At update 2,000 (empirical swap, 10 pairs):
  PROX redirection per seed: Crafts at the swapped fire T' by agents uninformed at that step, per world:
    mean(swap) >= 2 x mean(control) AND paired increase > 2 SE.
  PASS: redirection in >= 4 of 5 PROX seeds AND the PROX mean redirection ratio >= 0.5 x the ASSIGNED mean ratio
    (ASSIGNED ratio = listener Crafts at T', swap / control).
  Co-reported: reliance (deposit drop under the swap) and deliveries vs the PROX_NOSIG level; the informed fraction;
  MI of informed agents' tokens.
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

    def sw(k, u):
        P = [p for p in D[k]["swap"] if p["update"] == u and p.get("mode", "empirical") == "empirical"]
        c = np.array([p["listener_crafts_at_Tprime_control"] for p in P]); s = np.array([p["listener_crafts_at_Tprime_swap"] for p in P])
        dc = np.array([p["listener_dep_control"] for p in P]); ds = np.array([p["listener_dep_swap"] for p in P])
        d = s - c
        return dict(c=c.mean(), s=s.mean(), ratio=s.mean() / max(c.mean(), 1e-9), inc=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)),
                    drop=1 - ds.mean() / max(dc.mean(), 1e-9), dep_c=dc.mean())

    def deps(k, u, n=50):
        tr = [r for r in D[k]["train"] if u - n < r["update"] <= u]
        return float(np.mean([r["listener_dep_per_world"] + r["sender_dep_per_world"] for r in tr]))

    mi = lambda k, u: float(np.mean([r["mi_bits"] for r in D[k]["mi"] if r["update"] == u]))  # noqa: E731
    nos = [deps(k, U) for k in keys("PROX_NOSIG")]
    print(f"== PROX_NOSIG total deliveries/world (last 50): {[round(x, 2) for x in nos]}")
    res, ratios = [], []
    for k in keys("PROX"):
        r = sw(k, U)
        ok = r["s"] >= 2 * r["c"] and r["inc"] > 2 * r["se"]
        res.append(ok); ratios.append(r["ratio"])
        inf = np.mean([t.get("informed_frac") or 0 for t in D[k]["train"][-50:]])
        print(f"   PROX s{k[1]}: crafts@T' (uninformed) {r['c']:.2f} -> {r['s']:.2f} (ratio {r['ratio']:.2f}, paired +{r['inc']:.2f} "
              f"SE {r['se']:.2f}) -> {'pass' if ok else 'fail'} | reliance drop {r['drop']:+.0%} | deliveries {deps(k, U):.2f} "
              f"vs NOSIG {np.mean(nos) if nos else float('nan'):.2f} | informed frac {inf:.3f} | MI {mi(k, U):.3f}")
    a_ratios = []
    for k in keys("ASSIGNED"):
        r = sw(k, U); a_ratios.append(r["ratio"])
        print(f"   ASSIGNED s{k[1]}: crafts@T' {r['c']:.2f} -> {r['s']:.2f} (ratio {r['ratio']:.2f}) | reliance drop {r['drop']:+.0%}")
    par = bool(ratios and a_ratios and np.mean(ratios) >= 0.5 * np.mean(a_ratios))
    print(f"\n   PROX mean ratio {np.mean(ratios):.2f} vs 0.5 x ASSIGNED mean {0.5 * np.mean(a_ratios):.2f} -> on a par: {par}")
    print(f"== STEP 3: {'PASS' if len(res) == 5 and sum(res) >= 4 and par else 'FAIL'} (PROX redirection {sum(res)}/{len(res)})")


if __name__ == "__main__":
    main(sys.argv[1:])
