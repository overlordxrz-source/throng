"""Registered analysis for the CPU comm toy (docs/RESEARCH_PROTOCOL.md, toy registration 2026-11-07).

usage: analyze_toy.py <oracle.jsonl> <runs.jsonl> [<runs2.jsonl> ...]
Arms are keyed by (arm, ps>0, pl>0): NOSIG, BASE (COMM, no aux), PS, PL, BOTH.
"""
import json
import sys
from collections import defaultdict

import numpy as np

R_DEP, COST = 2.7, 0.45


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def arm_name(r):
    if r["arm"] != "COMM":
        return r["arm"]
    return {(False, False): "BASE", (True, False): "PS", (False, True): "PL", (True, True): "BOTH"}[(r["ps"] > 0, r["pl"] > 0)]


def main(oracle_path, run_paths):
    orc = [r for r in load([oracle_path]) if r["phase"] == "train"]
    rows = load(run_paths)
    tr = defaultdict(list); sw = defaultdict(list); mi = defaultdict(list)
    for r in rows:
        k = (arm_name(r), r["seed"])
        {"train": tr, "swap": sw, "mi": mi}[r["phase"]][k].append(r)
    for d in (tr,):
        for k in d:
            d[k].sort(key=lambda r: r["update"])
    last = lambda v, f, n=50: float(np.mean([r[f] for r in v[-n:]]))  # noqa: E731
    first = lambda v, f, n=5: float(np.mean([r[f] for r in v[:n]]))  # noqa: E731
    seeds = lambda arm: sorted(s for (a, s) in tr if a == arm)  # noqa: E731

    print("== per run (last 50 updates)")
    for k in sorted(tr):
        v = tr[k]
        print(f"  {k[0]:6s} s{k[1]} n={len(v)}: L dep/world {last(v,'listener_dep_per_world'):.2f} | S dep/world "
              f"{last(v,'sender_dep_per_world'):.2f} | L crafts/upd {last(v,'listener_crafts'):.0f} | H_send "
              f"{first(v,'token_entropy_sender'):.2f}->{last(v,'token_entropy_sender'):.2f} | H_list "
              f"{first(v,'token_entropy_listener'):.2f}->{last(v,'token_entropy_listener'):.2f} | swap_diff "
              f"{first(v,'swap_diff'):.3f}->{last(v,'swap_diff'):.3f} | L sens TV {first(v,'listener_sens_tv'):.4f}->"
              f"{last(v,'listener_sens_tv'):.4f}")

    # 1. channel-value precondition
    o_l = float(np.mean([r["listener_dep_per_world"] for r in orc[-50:]]))
    n_l = float(np.mean([last(tr[("NOSIG", s)], "listener_dep_per_world") for s in seeds("NOSIG")]))
    pre = o_l >= 2 * n_l
    print(f"\n== PRECONDITION (channel value): ORACLE listener dep/world {o_l:.2f} >= 2 x NOSIG {n_l:.2f} -> "
          f"{'PASS' if pre else 'FAIL'}")
    # craft-cost check (reported, never retuned)
    for s in seeds("NOSIG"):
        v = tr[("NOSIG", s)][-50:]
        crafts = sum(r["listener_crafts"] for r in v)
        p = sum(r["listener_dep_per_world"] * 64 for r in v) / max(crafts, 1)
        be = p * R_DEP / (1 - p) if p < 1 else float("inf")
        print(f"   craft cost check NOSIG s{s}: listener p(deposit | Craft) {p:.3f} -> break-even {be:.2f} vs cost "
              f"{COST} -> blind search {'UNPROFITABLE' if COST > be else 'still PROFITABLE'}")

    # 2. gate: live-wire collapse reproduced
    ratio = lambda k: last(tr[k], "token_entropy_sender") / first(tr[k], "token_entropy_sender")  # noqa: E731
    cr = [ratio(("BASE", s)) for s in seeds("BASE")]
    nr = [ratio(("NOSIG", s)) for s in seeds("NOSIG")]
    gate = bool(cr and nr and np.mean(cr) <= 0.6 and min(nr) >= 0.9 and max(cr) < min(nr))
    print(f"\n== GATE (sender token entropy, last-50 / first-5): BASE {[round(x, 3) for x in cr]} (mean "
          f"{np.mean(cr) if cr else float('nan'):.3f}, need <= 0.6) | NOSIG {[round(x, 3) for x in nr]} (each >= 0.9) "
          f"| max BASE < min NOSIG -> {'PASS' if gate else 'FAIL'}")
    if not (pre and gate):
        print("   -> registered stop: the toy is not representative (precondition or gate failed); aux arms not run")

    # 3. primary: token swap per arm, all 3 seeds
    print("\n== PRIMARY (token swap, Delta = mu_Tprime_swap - mu_Tprime_control over 10 pairs; per seed Delta >= 0.05"
          " and > 2 SE; arm passes iff all seeds pass)")
    for arm in ("BASE", "PS", "PL", "BOTH"):
        res = []
        for s in seeds(arm):
            d = np.array([p["mu_Tprime_swap"] - p["mu_Tprime_control"] for p in sw[(arm, s)]])
            if len(d) == 0:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            ok = d.mean() >= 0.05 and d.mean() > 2 * se
            dif = np.mean([p["swapped_token_differs_frac"] for p in sw[(arm, s)]])
            res.append(ok)
            print(f"   {arm:5s} s{s}: Delta {d.mean():+.4f} (SE {se:.4f}, n={len(d)}) -> {'pass' if ok else 'fail'} | "
                  f"swapped token differs on {dif:.1%} of sender steps")
        if res:
            print(f"   {arm:5s}: {'PASS' if all(res) and len(res) >= 3 else 'FAIL'}")

    # secondaries
    print("\n== SECONDARY: listener dep/world (last 50) vs NOSIG; MI(token; T) vs a plain (non-block) permutation null -- overstates significance under autocorrelation")
    for arm in ("BASE", "PS", "PL", "BOTH"):
        if seeds(arm):
            print(f"   {arm:5s} listener dep/world {[round(last(tr[(arm, s)], 'listener_dep_per_world'), 2) for s in seeds(arm)]}"
                  f" vs NOSIG {[round(last(tr[('NOSIG', s)], 'listener_dep_per_world'), 2) for s in seeds('NOSIG')]}")
            for s in seeds(arm):
                m = sorted(mi[(arm, s)], key=lambda r: r["slot"])
                print("     s%d MI bits: %s" % (s, " | ".join(f"slot{r['slot']} {r['mi_bits']:.4f} (null p99 "
                                                            f"{r['null_p99']:.4f})" for r in m)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
