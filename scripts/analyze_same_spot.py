"""Registered analysis (Cam, Nov 9 ~02:40, check (a)): same-spot swap control on the C1-E replication's u3000 params.

usage: analyze_same_spot.py <seed.jsonl> ...   (comm_toy.py --same-spot output, one file per wire-on seed)
Per seed (10 pairs): cross-spot drop = 1 - sum(dep_cross) / sum(dep_control); same-spot drop = 1 - sum(dep_same) /
sum(dep_control). The verdict ("toy self-sustaining code MET, FRAGILE") is WITHDRAWN iff same-spot drop >= 0.5 x
cross-spot drop in >= 2 of the 3 seeds. Also printed: paired per-pair SEs and the mi_joint rows (check (c), report-only).
"""
import json
import sys

import numpy as np


def main(paths):
    flags = []
    for p in paths:
        R = [json.loads(l) for l in open(p) if l.strip()]
        P = [r for r in R if r["phase"] == "same_spot"]
        c = np.array([r["listener_dep_control"] for r in P])
        x = np.array([r["listener_dep_cross"] for r in P]); s = np.array([r["listener_dep_same"] for r in P])
        dx, ds = 1 - x.sum() / c.sum(), 1 - s.sum() / c.sum()
        se = lambda d: d.std(ddof=1) / np.sqrt(len(d))  # noqa: E731
        f = bool(ds >= 0.5 * dx); flags.append(f)
        print(f"{p.split('/')[-1]}: control {c.mean():.2f} | cross-spot {x.mean():.2f} (drop {dx:+.0%}, paired {np.mean(c - x):.2f} SE "
              f"{se(c - x):.2f}) | same-spot {s.mean():.2f} (drop {ds:+.0%}, paired {np.mean(c - s):.2f} SE {se(c - s):.2f}) | "
              f"same >= 0.5 x cross: {f} | swapped triple differs: cross {np.mean([r['swapped_differs_cross'] for r in P]):.3f} "
              f"same {np.mean([r['swapped_differs_same'] for r in P]):.3f}")
        for r in R:
            if r["phase"] == "mi_joint":
                print(f"   MI: slots (MM) {[round(v, 3) for v in r['slot_mi_mm']]} | joint plug-in {r['joint_mi_plugin']:.3f} MM "
                      f"{r['joint_mi_mm']:.3f} ({r['distinct_triples']} triples, n {r['n']}) | pre-Craft slots "
                      f"{[round(v, 3) for v in r['pre_craft_slot_mi_mm']]} joint plug-in {r['pre_craft_joint_mi_plugin']:.3f} MM "
                      f"{r['pre_craft_joint_mi_mm']:.3f} (n {r['pre_craft_n']})")
    print(f"\n== SAME-SPOT CONTROL: same-spot drop >= half the cross-spot drop in {sum(flags)}/{len(flags)} seeds -> verdict "
          f"{'WITHDRAWN' if sum(flags) >= 2 else 'STANDS'}")


if __name__ == "__main__":
    main(sys.argv[1:])
