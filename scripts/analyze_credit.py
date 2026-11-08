"""Registered analysis: credit fixes A / B under the erosion protocol (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_credit.py <run.jsonl[.gz]> ...   (tags CREDIT_A_ON / CREDIT_A_OFF / CREDIT_B_ON / CREDIT_B_OFF)
Per fix, at update 2,000 (500 updates of weight decay 3e-4 on the message head, nudge at 0):
  wire-on seed passes: MI (Miller-Madow, mean over slots, updates 1,991-2,000) >= 1.0 bit AND reliance (empirical
    swap at 2,000: deposit drop >= 30%, paired drop > 2 SE, listener deposits over the last 50 updates >= the
    fix's wire-off mean).
  wire-off still erodes: MI < 0.5 bit at 2,000 in BOTH wire-off seeds. If either is >= 0.5, the fix maintains a
    code nobody listens to -> an artifact -> FAIL of that fix (Cam).
  PASS = >= 2 of 3 wire-on seeds pass AND wire-off still erodes.
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
    tr, sw, mi = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows(paths):
        k = (r["tag"], r["seed"])
        if r["phase"] in ("train", "swap", "mi"):
            {"train": tr, "swap": sw, "mi": mi}[r["phase"]][k].append(r)
    for k in tr:
        tr[k].sort(key=lambda r: r["update"])
    last = lambda k, f, n=50: float(np.mean([r[f] for r in tr[k][-n:]]))  # noqa: E731
    mi_at = lambda k, u: float(np.mean([r["mi_bits"] for r in mi[k] if r["update"] == u]))  # noqa: E731
    for fix in ("A", "B"):
        on = sorted(k for k in tr if k[0] == f"CREDIT_{fix}_ON")
        off = sorted(k for k in tr if k[0] == f"CREDIT_{fix}_OFF")
        if not on and not off:
            continue
        print(f"== fix {fix}")
        off_mi = [mi_at(k, U) for k in off]
        off_dep = float(np.mean([last(k, "listener_dep_per_world") for k in off])) if off else float("nan")
        for k, m in zip(off, off_mi):
            print(f"   wire-off s{k[1]}: MI@1500 {mi_at(k, 1500):.3f} -> MI@2000 {m:.3f} | L dep {last(k, 'listener_dep_per_world'):.2f}")
        erodes = len(off) == 2 and all(m < 0.5 for m in off_mi)
        print(f"   wire-off still erodes (both < 0.5): {erodes}")
        res = []
        for k in on:
            m = mi_at(k, U)
            P = [p for p in sw[k] if p["update"] == U and p["mode"] == "empirical"]
            c = np.array([p["listener_dep_control"] for p in P]); s_ = np.array([p["listener_dep_swap"] for p in P])
            d = c - s_; se = d.std(ddof=1) / np.sqrt(len(d))
            drop = 1 - s_.mean() / max(c.mean(), 1e-9)
            P15 = [p for p in sw[k] if p["update"] == 1500 and p["mode"] == "empirical"]
            drop15 = 1 - np.mean([p["listener_dep_swap"] for p in P15]) / max(np.mean([p["listener_dep_control"] for p in P15]), 1e-9)
            ldep = last(k, "listener_dep_per_world")
            ok = m >= 1.0 and drop >= 0.30 and d.mean() > 2 * se and ldep >= off_dep
            res.append(ok)
            print(f"   wire-on s{k[1]}: MI@1500 {mi_at(k, 1500):.3f} -> MI@2000 {m:.3f} | reliance@1500 {drop15:+.0%} -> "
                  f"@2000 {drop:+.0%} (paired {d.mean():.2f} SE {se:.2f}) | L dep {ldep:.2f} vs off {off_dep:.2f} -> "
                  f"{'pass' if ok else 'fail'}")
        verdict = "PASS" if erodes and len(res) == 3 and sum(res) >= 2 else "FAIL"
        why = "" if erodes else " (wire-off code maintained or missing -> artifact)"
        print(f"   FIX {fix}: {verdict}{why} (wire-on passing {sum(res)}/{len(res)})\n")


if __name__ == "__main__":
    main(sys.argv[1:])
