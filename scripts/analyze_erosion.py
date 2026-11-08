"""Registered analysis: erosion test in the CPU toy (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_erosion.py <run.jsonl[.gz]> ...   (tags EROSION_ON = wire on, EROSION_OFF = wire off)
At update 2,000 (500 updates after the nudge reached 0 and weight decay 3e-4 on the message head started):
  MI = Miller-Madow MI(token; active hearth), mean over the 3 slots, sender tokens of updates 1,991-2,000.
  Manipulation check: wire-off MI < 0.5 bit in >= 2 of 3 seeds; otherwise the test is UNINFORMATIVE (neither pass
  nor fail).
  Wire-on seed passes: MI >= 1.0 bit AND reliance (empirical swap at 2,000: deposit drop >= 30%, paired drop > 2 SE,
  listener deposits over the last 50 updates >= the wire-off arm's mean).
  PASS: manipulation check holds and >= 2 of 3 wire-on seeds pass. FAIL: manipulation holds and < 2 pass.
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
        {"train": tr, "swap": sw, "mi": mi}[r["phase"]][k].append(r)
    for k in tr:
        tr[k].sort(key=lambda r: r["update"])
    seeds = lambda t: sorted(s for (x, s) in tr if x == t)  # noqa: E731
    last = lambda k, f, n=50: float(np.mean([r[f] for r in tr[k][-n:]]))  # noqa: E731
    mi_at = lambda k, u: float(np.mean([r["mi_bits"] for r in mi[k] if r["update"] == u]))  # noqa: E731
    trace = lambda k: " ".join(f"{u}:{tr[k][u - 1]['mi_plugin']:.2f}" for u in (1000, 1500, 1600, 1750, 2000)  # noqa: E731
                               if len(tr[k]) >= u)

    off = [("EROSION_OFF", s) for s in seeds("EROSION_OFF")]
    on = [("EROSION_ON", s) for s in seeds("EROSION_ON")]
    off_dep = float(np.mean([last(k, "listener_dep_per_world") for k in off]))
    print("== wire OFF (manipulation check: MI < 0.5 bit at 2,000 in >= 2 of 3)")
    man = []
    for k in off:
        m = mi_at(k, U); man.append(m < 0.5)
        print(f"   s{k[1]}: MI@2000 {m:.3f} | MI@1500 {mi_at(k, 1500):.3f} | plug-in trace {trace(k)} | L dep "
              f"{last(k, 'listener_dep_per_world'):.2f} | wd from {tr[k][-1].get('update')}")
    manip = len(man) == 3 and sum(man) >= 2
    print(f"   manipulation check: {'HOLDS' if manip else 'FAILS'} ({sum(man)}/{len(man)}); wire-off listener dep mean "
          f"{off_dep:.2f}")

    print("\n== wire ON (seed passes: MI >= 1.0 and reliance)")
    res = []
    for k in on:
        m = mi_at(k, U)
        P = [p for p in sw[k] if p["update"] == U and p["mode"] == "empirical"]
        c = np.array([p["listener_dep_control"] for p in P]); s_ = np.array([p["listener_dep_swap"] for p in P])
        d = c - s_; se = d.std(ddof=1) / np.sqrt(len(d))
        drop = 1 - s_.mean() / max(c.mean(), 1e-9)
        ldep = last(k, "listener_dep_per_world")
        ok = m >= 1.0 and drop >= 0.30 and d.mean() > 2 * se and ldep >= off_dep
        res.append(ok)
        print(f"   s{k[1]}: MI@2000 {m:.3f} (MI@1500 {mi_at(k, 1500):.3f}) | trace {trace(k)} | drop {drop:+.0%} paired "
              f"{d.mean():.2f} SE {se:.2f} | L dep {ldep:.2f} vs wire-off {off_dep:.2f} -> {'pass' if ok else 'fail'}")
    if not manip:
        verdict = "UNINFORMATIVE (manipulation check failed)"
    else:
        verdict = "PASS" if len(res) == 3 and sum(res) >= 2 else "FAIL"
    print(f"\n== EROSION TEST: {verdict} (wire-on seeds passing: {sum(res)}/{len(res)})")


if __name__ == "__main__":
    main(sys.argv[1:])
