"""Registered analysis: T1 (credit B on the reliance bar), T2 (generational turnover), T3 (turnover + newborn
imitation). docs/RESEARCH_PROTOCOL.md, 2026-11-08.

usage: analyze_turnover.py <run.jsonl[.gz]> ...   tags: T1_ON/T1_OFF, T2_ON/T2_OFF, T3_ON/T3_OFF
Reliance at update u (empirical swap, 10 pairs): deposit drop >= 30%, paired drop > 2 SE, and listener deposits over
the 50 updates before u >= the same arm's wire-off mean over the same window (the NOSIG level).
T1 PASS: reliance at 2,000 in >= 4 of 5 wire-on seeds AND wire-off MI (Miller-Madow, mean over slots) < 0.5 at 2,000
  in both wire-off seeds. Secondary: wire-on MI >= 0.5.
T2 / T3 seed passes: before each of the last 3 resets (updates 2,250 / 2,500 / 2,750): reliance as above AND
  code continuity >= 0.8 [pre-data amendment 2026-11-08: reliance retention (drop / drop at 1,500) is reported as a
  secondary, not gated] (per-slot token -> fire decoding with
  the mapping recorded at the previous eval point, i.e. across the reset in between; chance 25%; unseen tokens miss).
  Arm PASS: >= 2 of 3 wire-on seeds. Manipulation check: wire-off MI < 0.5 at 2,000 (after the first sender reset
  at 1,750) in both wire-off seeds -- a gate for T2; reported but not a gate for T3 (registered in advance: newborn
  imitation transmits the code whether or not anyone listens, so a maintained wire-off code is expected there).
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

LAST3 = (2250, 2500, 2750)


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
    for k in D:
        D[k]["train"].sort(key=lambda r: r["update"])

    def win_dep(k, u, n=50):
        return float(np.mean([r["listener_dep_per_world"] for r in D[k]["train"] if u - n < r["update"] <= u]))

    def mi_at(k, u):
        m = [r["mi_bits"] for r in D[k]["mi"] if r["update"] == u]
        return float(np.mean(m)) if m else float("nan")

    def reliance(k, u, off_keys):
        P = [p for p in D[k]["swap"] if p["update"] == u and p["mode"] == "empirical"]
        if not P:
            return None
        c = np.array([p["listener_dep_control"] for p in P]); s = np.array([p["listener_dep_swap"] for p in P])
        d = c - s; se = d.std(ddof=1) / np.sqrt(len(d)); drop = 1 - s.mean() / max(c.mean(), 1e-9)
        nos = float(np.mean([win_dep(o, u) for o in off_keys])) if off_keys else 0.0
        ok = drop >= 0.30 and d.mean() > 2 * se and win_dep(k, u) >= nos
        return dict(drop=drop, paired=d.mean(), se=se, ldep=win_dep(k, u), nosig=nos, ok=ok)

    def keys(tag):
        return sorted(k for k in D if k[0] == tag)

    # ---- T1
    on, off = keys("T1_ON"), keys("T1_OFF")
    if on or off:
        print("== T1: credit B, wd 3e-4, reliance primary")
        offmi = [mi_at(k, 2000) for k in off]
        for k, m in zip(off, offmi):
            print(f"   wire-off s{k[1]}: MI@2000 {m:.3f}")
        res = []
        for k in on:
            r = reliance(k, 2000, off)
            res.append(bool(r and r["ok"]))
            print(f"   wire-on s{k[1]}: drop {r['drop']:+.0%} paired {r['paired']:.2f} SE {r['se']:.2f} | L dep {r['ldep']:.2f} vs"
                  f" NOSIG {r['nosig']:.2f} -> {'pass' if res[-1] else 'fail'} | MI@2000 {mi_at(k, 2000):.3f} (secondary >= 0.5:"
                  f" {mi_at(k, 2000) >= 0.5})")
        erodes = len(off) == 2 and all(m < 0.5 for m in offmi)
        print(f"   T1: {'PASS' if erodes and len(res) == 5 and sum(res) >= 4 else 'FAIL'} (reliance {sum(res)}/{len(res)}; "
              f"wire-off erodes: {erodes})\n")

    # ---- T2 / T3
    for arm in ("T2", "T3"):
        on, off = keys(f"{arm}_ON"), keys(f"{arm}_OFF")
        if not (on or off):
            continue
        print(f"== {arm}: generational turnover{' + newborn imitation' if arm == 'T3' else ''}")
        for k in off:
            tv = {r["update"]: r for r in D[k]["turnover"]}
            print(f"   wire-off s{k[1]}: MI@1500 {mi_at(k, 1500):.3f} MI@2000 {mi_at(k, 2000):.3f} | MI by eval point: "
                  + " ".join(f"{u}:{mi_at(k, u):.2f}" for u in sorted(tv)))
        manip = len(off) == 2 and all(mi_at(k, 2000) < 0.5 for k in off)
        print(f"   manipulation check (wire-off MI < 0.5 at 2,000 in both): {manip}"
              + ("" if arm == "T2" else "  [not a gate for T3, registered in advance]"))
        res = []
        for k in on:
            tv = {r["update"]: r for r in D[k]["turnover"]}
            r15 = reliance(k, 1500, off)
            ok_seed = True; parts = []
            for u in LAST3:
                r = reliance(k, u, off); c = tv.get(u, {}).get("continuity", float("nan"))
                ret = (r["drop"] / r15["drop"]) if (r and r15 and r15["drop"] > 0) else float("nan")
                # pre-data amendment (Cam, 2026-11-08): the gate is the absolute rung-3 reliance bar + continuity >= 0.8;
                # retention (drop / drop at 1,500) is reported as a secondary, not gated
                ok_u = bool(r and r["ok"] and c >= 0.8)
                ok_seed &= ok_u
                parts.append(f"{u}: drop {r['drop']:+.0%} (ret {ret:.2f}) cont {c:.2f} Ldep {r['ldep']:.2f}/{r['nosig']:.2f}"
                             f" {'ok' if ok_u else 'x'}")
            res.append(ok_seed)
            imit = [x for x in D[k]["reset"] if "imitation_acc_after" in x]
            print(f"   wire-on s{k[1]}: @1500 drop {r15['drop']:+.0%} | " + " | ".join(parts) + f" -> {'pass' if ok_seed else 'fail'}")
            print("      all eval points: " + " ".join(f"{u}:cont {tv[u].get('continuity', float('nan')):.2f}/MI {tv[u]['mi_plugin']:.2f}"
                                               for u in sorted(tv)))
            if imit:
                print("      imitation acc after: " + " ".join(f"{x['update']}:{x['imitation_acc_after']:.2f}" for x in imit))
        gate = manip if arm == "T2" else True
        verdict = "PASS" if gate and len(res) == 3 and sum(res) >= 2 else "FAIL"
        print(f"   {arm}: {verdict} (wire-on passing {sum(res)}/{len(res)}"
              + ("" if arm == "T3" or manip else "; manipulation check failed") + ")\n")


if __name__ == "__main__":
    main(sys.argv[1:])
