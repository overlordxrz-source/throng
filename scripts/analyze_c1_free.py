"""Cam's free checks on C1 (2026-11-08 night), exploratory, $0.

usage: analyze_c1_free.py   (reads toy_out_modal/c1/*.jsonl and the eval-only re-probes in toy_out_modal/c1_free/)
1. C1-E wire-on redirection at 2,000 (in-run empirical swap): listener Crafts at T', swap vs control (step 2's rule:
   swap >= 2x control and paired increase > 2 SE).
2. Wire-on slopes over 1,800-2,000: per-update plug-in MI and listener deliveries / wire sensitivity (OLS slope per 100
   updates), plus reliance at 1,500 / 1,750 / 2,000 from eval-only re-probes of the saved actor params.
3. C1-T3 at 2,000: the in-run probe ran AFTER the listener wire-row reset at 2,000 (evaluate(params, a.updates) runs
   after the loop's last do_reset). Re-probe of the pre-reset u2000 params (saved before the reset).
Eval-only re-probes: controls and constructed swaps are bit-identical to the in-run probe; the empirical replacement
triple is a different draw (the eval RNG is seeded with the update number and in-run MI consumes it first).
"""
import glob
import json
from collections import defaultdict

import numpy as np

R = "toy_out_modal/"


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def rel(P, mode="empirical"):
    P = [p for p in P if p["mode"] == mode]
    c = np.array([p["listener_dep_control"] for p in P]); s = np.array([p["listener_dep_swap"] for p in P])
    d = c - s; se = d.std(ddof=1) / np.sqrt(len(d))
    rc = np.array([p["listener_crafts_at_Tprime_control"] for p in P]); rs = np.array([p["listener_crafts_at_Tprime_swap"] for p in P])
    rd = rs - rc; rse = rd.std(ddof=1) / np.sqrt(len(rd))
    return dict(drop=1 - s.mean() / max(c.mean(), 1e-9), pd=d.mean(), se=se, ok_stat=bool(1 - s.mean() / c.mean() >= .3 and d.mean() > 2 * se),
                rc=rc.mean(), rs=rs.mean(), red=bool(rs.mean() >= 2 * rc.mean() and rd.mean() > 2 * rse))


def slope(rows, key, lo=1800, hi=2000):
    x = np.array([r["update"] for r in rows if lo <= r["update"] <= hi]); y = np.array([r[key] for r in rows if lo <= r["update"] <= hi])
    return 100 * np.polyfit(x, y, 1)[0], y[:50].mean(), y[-50:].mean()


def main():
    run = {}
    files = {}
    for p in sorted(glob.glob(R + "c1/*.jsonl")):          # prefer the restart-cleaned file where one exists
        nm = p.split("/")[-1].split(".")[0]
        if nm not in files or p.endswith(".clean.jsonl"):
            files[nm] = p
    for p in files.values():
        n = p.split("/")[-1].split(".")[0]; D = defaultdict(list)
        for r in load(p):
            D[r["phase"]].append(r)
        run[n] = D
    win = lambda n, u: float(np.mean([r["listener_dep_per_world"] for r in run[n]["train"] if u - 50 < r["update"] <= u]))  # noqa: E731
    for proto, on, off in (("C1E", (185, 186, 187), (188, 189)), ("C1T3", (180, 181, 182), (183, 184))):
        pre = proto.lower()
        print(f"\n===== {proto}")
        for s in on:
            n = f"{pre}_on_s{s}"; tr = sorted(run[n]["train"], key=lambda r: r["update"])
            m = slope(tr, "mi_plugin"); l = slope(tr, "listener_dep_per_world"); t = slope(tr, "listener_sens_tv")
            mi = {u: np.mean([r["mi_bits"] for r in run[n]["mi"] if r["update"] == u]) for u in (1500, 2000)}
            print(f"  s{s}: MI(MM) 1500 {mi[1500]:.3f} -> 2000 {mi[2000]:.3f} | 1800-2000 slopes per 100 upd: plug-in MI {m[0]:+.3f} "
                  f"({m[1]:.3f} -> {m[2]:.3f}), L dep {l[0]:+.3f} ({l[1]:.2f} -> {l[2]:.2f}), L wire-sens TV {t[0]:+.4f} ({t[1]:.4f} -> {t[2]:.4f})")
            for u in (1500, 2000):
                ir = rel([r for r in run[n]["swap"] if r["update"] == u])
                nos = np.mean([win(f"{pre}_off_s{o}", u) for o in off])
                tag = " (in-run, AFTER the 2,000 listener reset)" if proto == "C1T3" and u == 2000 else " (in-run)"
                print(f"     @{u}{tag}: drop {ir['drop']:+.0%} paired {ir['pd']:.2f} SE {ir['se']:.2f} | L dep {win(n, u):.2f} vs NOSIG {nos:.2f}"
                      f" -> reliance {'ok' if ir['ok_stat'] and win(n, u) >= nos else 'x'} | crafts@T' {ir['rc']:.2f} -> {ir['rs']:.2f} "
                      f"redirection {'ok' if ir['red'] else 'x'}")
            for u in (1500, 1750, 2000):
                p = f"{R}c1_free/{pre}_on_s{s}_u{u}.jsonl"
                if not glob.glob(p):
                    continue
                e = rel([r for r in load(p) if r["phase"] == "swap"])
                nos = np.mean([win(f"{pre}_off_s{o}", u) for o in off])
                print(f"     @{u} (eval-only re-probe of saved params{', PRE-reset' if proto == 'C1T3' and u == 2000 else ''}): drop "
                      f"{e['drop']:+.0%} paired {e['pd']:.2f} SE {e['se']:.2f} | L dep {win(n, u):.2f} vs NOSIG {nos:.2f} -> reliance "
                      f"{'ok' if e['ok_stat'] and win(n, u) >= nos else 'x'} | crafts@T' {e['rc']:.2f} -> {e['rs']:.2f} redirection "
                      f"{'ok' if e['red'] else 'x'}")
        for s in off:
            n = f"{pre}_off_s{s}"; tr = sorted(run[n]["train"], key=lambda r: r["update"]); m = slope(tr, "mi_plugin")
            print(f"  wire-off s{s}: plug-in MI slope {m[0]:+.3f}/100 ({m[1]:.3f} -> {m[2]:.3f}) | MI(MM)@2000 "
                  f"{np.mean([r['mi_bits'] for r in run[n]['mi'] if r['update'] == 2000]):.3f}")


if __name__ == "__main__":
    main()
