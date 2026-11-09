"""Registered analysis: erosion calibration (Cam Nov 9 ~02:55 item 4 / ~07:00 (b); docs/RESEARCH_PROTOCOL.md).

usage: analyze_erocal.py <dir>   (dir holds mi_wd{wd}_s{seed}_u{u}.jsonl from `comm_toy.py --same-spot --mi-null 100` on
                                 each saved params file of plan ero_cal; wire-off seeds 270 / 271 / 272)
Shuffle-corrected joint MI = Miller-Madow joint MI over the 3 slots - the mean of 100 spot-label permutations (same N).
PICK = the smallest weight decay at which EVERY wire-off seed is below 0.05 bits by u2500 (i.e. at u2500).
If none qualifies: report, don't extend.
"""
import glob
import json
import os
import re
import sys


def main(d):
    T = {}
    for p in glob.glob(os.path.join(d, "mi_wd*_s*_u*.jsonl")):
        m = re.search(r"mi_wd([0-9e.-]+)_s(\d+)_u(\d+)\.jsonl", os.path.basename(p))
        wd, s, u = m.group(1), int(m.group(2)), int(m.group(3))
        r = [json.loads(l) for l in open(p) if '"mi_null"' in l][-1]
        T[(wd, s, u)] = r["joint_mi_mm"] - r["null_mm_mean"]
    wds = sorted({k[0] for k in T}, key=float); seeds = sorted({k[1] for k in T}); us = sorted({k[2] for k in T})
    print("shuffle-corrected joint MI (bits) by strength, seed, update:")
    for wd in wds:
        for s in seeds:
            print(f"  wd {wd} s{s}: " + "  ".join(f"u{u} {T.get((wd, s, u), float('nan')):+.3f}" for u in us))
    ok = [wd for wd in wds if all(T.get((wd, s, 2500), 9) < 0.05 for s in seeds)]
    print(f"\n== ERO CALIBRATION PICK: {ok[0] if ok else 'NONE QUALIFIES (report; do not extend)'}"
          f"  (qualifying strengths: {ok or '-'})")


if __name__ == "__main__":
    main(sys.argv[1])
