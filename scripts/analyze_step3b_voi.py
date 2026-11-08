"""Registered analysis: step 3b value-of-information check (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_step3b_voi.py <run.jsonl[.gz]> ...   tags S3B_ORACLE (--lookout-oracle) and S3B_NOSIG; 2,000 updates, PS off
World: G 24 torus, fires clustered at (3,3) (3,9) (9,3) (9,9), lookout = Chebyshev 1 around (18,18), lit duration 10.
Deliveries = deliveries by either agent per world per episode (agent_dep_per_world), mean over updates 1,801-2,000.
QUALIFIES iff the ORACLE mean >= 1.5x the NOSIG mean AND every ORACLE seed > every NOSIG seed AND the ORACLE mean
  >= 0.5 deliveries per world (floor: below it neither arm has learned the world and the ratio is uninformative).
Otherwise: retune the world (distances / lit duration), never the agents.
Reported, not gated: lookout occupancy (share of agent-steps at the lookout; share of world-steps with either agent
there), informed fraction, and the same windows at 500 / 1,000 / 1,500.
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
    D = defaultdict(list)
    for r in rows(paths):
        if r["phase"] == "train":
            D[(r["tag"], r["seed"])].append(r)
    w = lambda k, key, hi=2000, n=200: float(np.mean([r[key] for r in D[k] if hi - n < r["update"] <= hi]))  # noqa: E731
    means = {}
    for tag in ("S3B_ORACLE", "S3B_NOSIG"):
        ks = sorted(k for k in D if k[0] == tag); vals = []
        print(f"== {tag}")
        for k in ks:
            traj = " ".join(f"@{u} {w(k, 'agent_dep_per_world', u):.2f}" for u in (500, 1000, 1500, 2000))
            print(f"   s{k[1]}: deliveries/world {traj} | lookout occupancy (agent-steps) {w(k, 'lookout_occ'):.4f}, either agent "
                  f"{w(k, 'lookout_any'):.4f} | informed {w(k, 'informed_frac'):.4f}")
            vals.append(w(k, "agent_dep_per_world"))
        means[tag] = vals
    o, n = means["S3B_ORACLE"], means["S3B_NOSIG"]
    mo, mn = float(np.mean(o)), float(np.mean(n))
    q = mo >= 1.5 * mn and min(o) > max(n) and mo >= 0.5
    print(f"\n== STEP 3b VOI: ORACLE {mo:.2f} vs NOSIG {mn:.2f} (ratio {mo / max(mn, 1e-9):.2f}; every ORACLE > every NOSIG "
          f"{min(o) > max(n)}; floor {mo >= 0.5}) -> {'QUALIFIES' if q else 'DOES NOT QUALIFY (retune the world)'}")


if __name__ == "__main__":
    main(sys.argv[1:])
