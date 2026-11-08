"""Unregistered descriptive check (2026-11-07): per run, mean swap-pair movement (toward T' and T) and listener
deposits per world in control vs swap rollouts, plus a few training-trajectory points. usage: swap_detail.py <run.jsonl>..."""
import gzip
import json
import sys

import numpy as np

for p in sys.argv[1:]:
    op = gzip.open if p.endswith(".gz") else open
    rows = [json.loads(l) for l in op(p, "rt")]
    S = [r for r in rows if r["phase"] == "swap"]
    if not S:
        continue
    f = lambda k: np.mean([s[k] for s in S])  # noqa: E731
    print(p.split("/")[-1], "muT' ctrl %.3f swap %.3f | muT ctrl %.3f swap %.3f | L dep ctrl %.2f swap %.2f" % (
        f("mu_Tprime_control"), f("mu_Tprime_swap"), f("mu_T_control"), f("mu_T_swap"), f("listener_dep_control"),
        f("listener_dep_swap")))
