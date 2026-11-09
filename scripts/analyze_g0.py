"""Registered analysis: separate-brains word-bias gate G0 (Cam Nov 9; docs/RESEARCH_PROTOCOL.md).

usage: analyze_g0.py <dictionary.json> ...   (comm_toy.py --export-m on each G0 run's u1500 params; meta.checkpoint names
                                             the run; tags G0_N260 seeds 280 / 281, G0_N261 seeds 282 / 283)
Words = each run's per-slot modal token for each fire (4 fires x 3 slots = 12 cells).
Agreement(i, j) = share of the 12 cells where runs i and j use the same token.
  Same-start pairs (positive control): (280, 281) and (282, 283).
  Different-start pairs: the 4 cross pairs.
Null: for each different-start pair, agreement after relabelling run j's fires by each of the 23 non-identity
permutations of 4 fires; the null pool is all 4 x 23 values.
PASS iff the mean different-start agreement <= the null pool's 95th percentile.
Reported: same-start agreement (expected high, as the ORACLE0-start seeds were), per-pair values.
"""
import itertools
import json
import sys

import numpy as np


def main(paths):
    W = {}
    for p in paths:
        d = json.load(open(p)); ck = d["meta"]["checkpoint"]
        seed = int(ck.split("_s")[-1].split("_")[0]) if "_s" in ck else None
        W[seed] = np.array([r["slot_words"] for r in d["dictionary"]])          # (4 fires, 3 slots)
    agree = lambda a, b: float(np.mean(a == b))  # noqa: E731
    same = [(280, 281), (282, 283)]; cross = [(i, j) for i in (280, 281) for j in (282, 283)]
    perms = [p for p in itertools.permutations(range(4)) if p != (0, 1, 2, 3)]
    null = [agree(W[i], W[j][list(p)]) for i, j in cross for p in perms]
    s_vals = [agree(W[i], W[j]) for i, j in same]; c_vals = [agree(W[i], W[j]) for i, j in cross]
    p95 = float(np.percentile(null, 95))
    for (i, j), v in zip(same, s_vals):
        print(f"same start   {i}-{j}: agreement {v:.3f}")
    for (i, j), v in zip(cross, c_vals):
        print(f"different    {i}-{j}: agreement {v:.3f}")
    print(f"null (fire-relabelled cross pairs, n={len(null)}): mean {np.mean(null):.3f}, p95 {p95:.3f}")
    ok = np.mean(c_vals) <= p95
    print(f"\n== G0: mean different-start agreement {np.mean(c_vals):.3f} vs null p95 {p95:.3f} -> {'PASS' if ok else 'FAIL'} "
          f"(same-start mean {np.mean(s_vals):.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])
