"""Registered analysis: build-order step 6, compositionality pilot (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_step6.py <run.jsonl[.gz]> ...   (tag S6_COMP; 3,000 updates)
GATE 1 (topographic similarity): at 3,000, Spearman rho between meaning distance (Hamming over (fire, material)) and
  message distance (Hamming over the 3 slots of each training meaning's modal message) > the 95th percentile of a
  1,000-permutation null over the message-to-meaning assignment, in >= 4 of 5 seeds.
GATE 2 (held-out generalisation): at 3,000, in the frozen held-out test (only the 2 held-out pairs active), listener
  success per Craft attempt at a fire (both parts right) is above chance 1/16 (one-sided binomial p < 0.01) AND above
  the shuffled-message null (one-sided two-proportion z-test p < 0.01), in >= 3 of 5 seeds.
PASS = both gates.
"""
import gzip
import json
import math
import sys
from collections import defaultdict

from scipy.stats import binomtest

U = 3000


def rows(paths):
    out = []
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        out += [json.loads(l) for l in op(p, "rt") if l.strip()]
    return out


def ztest_greater(s1, n1, s2, n2):
    p1, p2 = s1 / max(n1, 1), s2 / max(n2, 1)
    p = (s1 + s2) / max(n1 + n2, 1)
    se = math.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-18))
    z = (p1 - p2) / se
    return 0.5 * math.erfc(z / math.sqrt(2))


def main(paths):
    D = defaultdict(lambda: defaultdict(list))
    for r in rows(paths):
        D[(r["tag"], r["seed"])][r["phase"]].append(r)
    g1, g2 = [], []
    for k in sorted(D):
        ts = {r["update"]: r for r in D[k]["topsim"]}; ho = {r["update"]: r for r in D[k]["heldout"]}
        for u in sorted(ts):
            t, h = ts[u], ho.get(u, {})
            print(f"   s{k[1]} u{u}: topsim rho {t['rho']:.3f} (null p95 {t['null_p95']:.3f}; {t['distinct_messages']} distinct "
                  f"messages / {t['n_meanings']} meanings) | held-out success real {h.get('rate_real', float('nan')):.4f} "
                  f"({h.get('succ_real')}/{h.get('att_real')}) vs shuffled {h.get('rate_shuf', float('nan')):.4f} | training-pair "
                  f"success {h.get('train_rate', float('nan')):.4f}")
        t, h = ts.get(U), ho.get(U)
        ok1 = bool(t and t["rho"] > t["null_p95"])
        if h:
            p_bin = binomtest(h["succ_real"], max(h["att_real"], 1), 1 / 16, alternative="greater").pvalue
            p_shuf = ztest_greater(h["succ_real"], h["att_real"], h["succ_shuf"], h["att_shuf"])
            ok2 = p_bin < 0.01 and p_shuf < 0.01
        else:
            p_bin = p_shuf = float("nan"); ok2 = False
        g1.append(ok1); g2.append(ok2)
        print(f"   s{k[1]} @3000: gate 1 {'ok' if ok1 else 'x'} | gate 2 {'ok' if ok2 else 'x'} (binomial vs 1/16 p {p_bin:.2g}; vs "
              f"shuffled p {p_shuf:.2g})")
    P1, P2 = sum(g1) >= 4, sum(g2) >= 3
    print(f"\n== STEP 6: gate 1 (topsim) {'PASS' if P1 else 'FAIL'} ({sum(g1)}/{len(g1)}) | gate 2 (held-out) "
          f"{'PASS' if P2 else 'FAIL'} ({sum(g2)}/{len(g2)}) -> {'PASS' if P1 and P2 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1:])
