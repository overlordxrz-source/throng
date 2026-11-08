"""Registered analysis for the PS replication in the CPU comm toy (docs/RESEARCH_PROTOCOL.md, 2026-11-07).

usage: analyze_rep.py <run.jsonl[.gz]> ...
Arms: PS (ps > 0, no anneal), PSA (ps > 0 with anneal; identified by tag "PSA"), NOSIG.
P1 reliance (all 5 PS seeds): under the EMPIRICAL swap at the final update, listener deposits drop >= 30%
   (1 - mean swap / mean control over the 10 pairs), the paired drop is > 2 SE, and the seed's listener deposits
   (last 50 training updates) are >= the NOSIG mean (the channel must help, not just be relied on).
P2 meaning (>= 4 of 5 PS seeds): Delta = mean(mu_T'_swap - mu_T'_control), empirical swap, >= +0.05 and > 2 SE.
P3 self-sustaining (>= 2 of 3 PSA seeds): at the final update (500 updates at c_ps = 0) mean-over-slots MI >= 1.0
   bit and P1's reliance conditions hold.
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

R_DEP, COST = 2.7, 0.45


def rows(paths):
    out = []
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        out += [json.loads(l) for l in op(p, "rt") if l.strip()]
    return out


def arm(r):
    if r["arm"] == "NOSIG":
        return "NOSIG"
    if r.get("tag") == "PSA":
        return "PSA"
    return "PS" if r["ps"] > 0 else "BASE"


def main(paths):
    R = rows(paths)
    tr, sw, mi = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in R:
        k = (arm(r), r["seed"])
        {"train": tr, "swap": sw, "mi": mi}[r["phase"]][k].append(r)
    for k in tr:
        tr[k].sort(key=lambda r: r["update"])
    final_u = {k: v[-1]["update"] for k, v in tr.items()}
    last = lambda k, f, n=50: float(np.mean([r[f] for r in tr[k][-n:]]))  # noqa: E731
    seeds = lambda a: sorted(s for (x, s) in tr if x == a)  # noqa: E731
    nosig = float(np.mean([last(("NOSIG", s), "listener_dep_per_world") for s in seeds("NOSIG")]))

    def swap_stats(k, u, mode):
        P = [p for p in sw[k] if p["update"] == u and p["mode"] == mode]
        if not P:
            return None
        c = np.array([p["listener_dep_control"] for p in P]); s = np.array([p["listener_dep_swap"] for p in P])
        d = c - s
        dl = np.array([p["mu_Tprime_swap"] - p["mu_Tprime_control"] for p in P])
        return dict(n=len(P), drop=1 - s.mean() / max(c.mean(), 1e-9), d=d.mean(), d_se=d.std(ddof=1) / np.sqrt(len(P)),
                    delta=dl.mean(), delta_se=dl.std(ddof=1) / np.sqrt(len(P)), ctrl=c.mean(), swp=s.mean(),
                    seenTp=np.mean([p["swap_token_seen_for_Tprime"] for p in P]),
                    seenAny=np.mean([p["swap_token_seen_at_all"] for p in P]),
                    redir_c=np.mean([p["listener_crafts_at_Tprime_control"] for p in P]),
                    redir_s=np.mean([p["listener_crafts_at_Tprime_swap"] for p in P]),
                    redir_d=np.mean([p["listener_crafts_at_Tprime_swap"] - p["listener_crafts_at_Tprime_control"] for p in P]),
                    redir_se=np.std([p["listener_crafts_at_Tprime_swap"] - p["listener_crafts_at_Tprime_control"] for p in P],
                                    ddof=1) / np.sqrt(len(P)))

    def mi_mean(k, u):
        m = [r["mi_bits"] for r in mi[k] if r["update"] == u]
        return float(np.mean(m)) if m else float("nan")

    def reliance(k, st):
        return (st is not None and st["drop"] >= 0.30 and st["d"] > 2 * st["d_se"]
                and last(k, "listener_dep_per_world") >= nosig)

    print(f"== NOSIG listener deposits/world (last 50): {[round(last(('NOSIG', s), 'listener_dep_per_world'), 2) for s in seeds('NOSIG')]}"
          f" mean {nosig:.2f}")
    for s in seeds("NOSIG"):
        v = tr[("NOSIG", s)][-50:]
        p = sum(r["listener_dep_per_world"] * 64 for r in v) / max(sum(r["listener_crafts"] for r in v), 1)
        print(f"   known limit -- blind-search break-even cost NOSIG s{s}: p {p:.3f} -> {p * R_DEP / (1 - p):.2f} vs {COST}")

    for a in ("PS", "PSA"):
        print(f"\n== {a} per seed")
        for s in seeds(a):
            k = (a, s)
            for u in sorted({p["update"] for p in sw[k]}):
                for mode in ("empirical", "constructed"):
                    st = swap_stats(k, u, mode)
                    if st:
                        print(f"   s{s} u{u} {mode:11s}: L dep ctrl {st['ctrl']:.2f} swap {st['swp']:.2f} (drop "
                              f"{st['drop']:+.0%}, paired {st['d']:.2f} SE {st['d_se']:.2f}) | Delta {st['delta']:+.4f} "
                              f"(SE {st['delta_se']:.4f}) | swap token seen for T' {st['seenTp']:.1%}, at all "
                              f"{st['seenAny']:.1%}")
                print(f"   s{s} u{u} MI mean over slots {mi_mean(k, u):.3f} bits")
            print(f"   s{s} listener dep/world last 50 {last(k, 'listener_dep_per_world'):.2f} | sender H "
                  f"{last(k, 'token_entropy_sender'):.2f} | c_ps at end {tr[k][-1]['ps_coef']:.3f}")

    print("\n== P1 reliance (all 5 PS seeds; empirical swap, final update)")
    p1 = []
    for s in seeds("PS"):
        k = ("PS", s); st = swap_stats(k, final_u[k], "empirical"); ok = reliance(k, st); p1.append(ok)
        print(f"   PS s{s}: drop {st['drop']:+.0%} (>= 30%), paired {st['d']:.2f} > 2 x {st['d_se']:.2f}, L dep "
              f"{last(k, 'listener_dep_per_world'):.2f} >= NOSIG {nosig:.2f} -> {'pass' if ok else 'fail'}")
    P1 = len(p1) == 5 and all(p1)
    print(f"   P1: {'PASS' if P1 else 'FAIL'} ({sum(p1)}/{len(p1)})")

    print("\n== P2 meaning (>= 4 of 5 PS seeds; empirical swap; threshold set after a 2/3 outcome, fresh seeds)")
    p2 = []
    for s in seeds("PS"):
        k = ("PS", s); st = swap_stats(k, final_u[k], "empirical")
        ok = st["delta"] >= 0.05 and st["delta"] > 2 * st["delta_se"]; p2.append(ok)
        print(f"   PS s{s}: Delta {st['delta']:+.4f} (SE {st['delta_se']:.4f}) -> {'pass' if ok else 'fail'}")
    P2 = len(p2) == 5 and sum(p2) >= 4
    print(f"   P2: {'PASS' if P2 else 'FAIL'} ({sum(p2)}/{len(p2)})")

    print("\n== SECONDARY (registered, not decisive) redirection: listener Crafts at the swapped hearth T' per world,"
          " swap >= 2x control and paired increase > 2 SE, in >= 4 of 5 PS seeds")
    rd = []
    for s in seeds("PS"):
        k = ("PS", s); st = swap_stats(k, final_u[k], "empirical")
        ok = st["redir_s"] >= 2 * st["redir_c"] and st["redir_d"] > 2 * st["redir_se"]; rd.append(ok)
        print(f"   PS s{s}: Crafts at T' control {st['redir_c']:.2f} swap {st['redir_s']:.2f} (paired +{st['redir_d']:.2f}, SE "
              f"{st['redir_se']:.2f}) -> {'pass' if ok else 'fail'}")
    print(f"   redirection: {'PASS' if len(rd) == 5 and sum(rd) >= 4 else 'FAIL'} ({sum(rd)}/{len(rd)})")

    print("\n== P3 self-sustaining (>= 2 of 3 PSA seeds; final update, after 500 updates at c_ps = 0)")
    p3 = []
    for s in seeds("PSA"):
        k = ("PSA", s); st = swap_stats(k, final_u[k], "empirical"); m = mi_mean(k, final_u[k])
        ok = m >= 1.0 and reliance(k, st); p3.append(ok)
        print(f"   PSA s{s}: MI {m:.3f} (>= 1.0) | drop {st['drop']:+.0%} paired {st['d']:.2f} SE {st['d_se']:.2f} | L dep "
              f"{last(k, 'listener_dep_per_world'):.2f} vs NOSIG {nosig:.2f} -> {'pass' if ok else 'fail'}")
    P3 = len(p3) == 3 and sum(p3) >= 2
    print(f"   P3: {'PASS' if P3 else 'FAIL'} ({sum(p3)}/{len(p3)})")
    print(f"\n== DECISION: {'P1 and P2 pass -> draft the real-network PS confirmation (nothing launches)' if P1 and P2 else 'report; nothing goes to Modal'}"
          f" | P3 {'PASS -> include an anneal arm' if P3 else 'FAIL -> no anneal arm'}")


if __name__ == "__main__":
    main(sys.argv[1:])
