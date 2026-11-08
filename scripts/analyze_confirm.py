"""Registered analysis: real-network PS confirmation (docs/RESEARCH_PROTOCOL.md, 2026-11-08).

usage: analyze_confirm.py <run_seed_a.jsonl> <run_seed_b.jsonl>
PRIMARY (both seeds): redirection on the mean of the probes at updates 350, 400, 450 (empirical swap, 10 pairs
each, pooled to 30 pairs): listener Crafts at the swapped hearth T' per world, mean(swap) >= 2 x mean(control)
AND the paired increase (swap - control) > 2 SE over the 30 pairs.
Reported (not decisive): per probe -- reliance (deposit drop under the swap), listener and sender deliveries
(incl. sender crashes: per-update minimum in each 50-update window), movement Delta toward T', MI vs null, swap
token membership; halts.
"""
import json
import sys

import numpy as np

PRIMARY_PROBES = (350, 400, 450)


def main(paths):
    verdicts = []
    for p in paths:
        R = [json.loads(l) for l in open(p) if l.strip()]
        seed = R[0]["seed"]
        tr = sorted([r for r in R if r["phase"] == "train"], key=lambda r: r["update"])
        sw = [r for r in R if r["phase"] == "swap" and r.get("mode", "empirical") == "empirical"]
        halts = [r for r in R if r["phase"] in ("halt", "mi_probe_halt") or (r["phase"] in ("apparatus", "mi_check")
                                                                              and not r.get("ok", True))]
        print(f"== seed {seed}: {len(tr)} updates; halts: {[(h['phase'], h['update']) for h in halts] or 'none'}")
        for r in R:
            if r["phase"] in ("apparatus", "mi_check"):
                print(f"   {r['phase']} @ {r['update']}: " + ", ".join(f"{k}={v}" for k, v in r.items()
                                                                     if k in ("ok", "mi_bits", "floor",
                                                                              "ps_grad_message_heads",
                                                                              "distinct_sender_messages",
                                                                              "listener_wire_nonzero_frac")))
        mi_by = {r["update"]: r for r in tr}
        for u in sorted({r["update"] for r in sw}):
            P = [r for r in sw if r["update"] == u]
            c = np.array([r["listener_crafts_at_Tprime_control"] for r in P])
            s = np.array([r["listener_crafts_at_Tprime_swap"] for r in P])
            dc = np.array([r["listener_dep_control"] for r in P]); ds = np.array([r["listener_dep_swap"] for r in P])
            dl = np.mean([r["mu_Tprime_swap"] - r["mu_Tprime_control"] for r in P])
            win = [r for r in tr if u - 50 < r["update"] <= u]
            print(f"   probe {u}: crafts@T' {c.mean():.2f} -> {s.mean():.2f} | deposits {dc.mean():.2f} -> {ds.mean():.2f} "
                  f"(reliance {1 - ds.mean() / max(dc.mean(), 1e-9):+.0%}) | Delta {dl:+.3f} | L dep/world (train, window)"
                  f" {np.mean([r['listener_dep_per_world'] for r in win]):.2f} | S dep min/mean "
                  f"{min(r['sender_dep_per_world'] for r in win):.1f}/{np.mean([r['sender_dep_per_world'] for r in win]):.1f}"
                  f" | MI {mi_by[u]['mi_bits']:.3f} (null p95 {mi_by[u]['mi_null_p95']:.3f}) | swap token seen "
                  f"{np.mean([r['swap_token_seen_for_Tprime'] for r in P]):.1%}")
        P = [r for r in sw if r["update"] in PRIMARY_PROBES]
        have = sorted({r["update"] for r in P})
        if have != list(PRIMARY_PROBES):
            print(f"   PRIMARY: probes {have} present, need {PRIMARY_PROBES} -> NOT EVALUABLE (run halted or short)")
            verdicts.append(None)
            continue
        c = np.array([r["listener_crafts_at_Tprime_control"] for r in P])
        s = np.array([r["listener_crafts_at_Tprime_swap"] for r in P])
        d = s - c; se = d.std(ddof=1) / np.sqrt(len(d))
        ok = s.mean() >= 2 * c.mean() and d.mean() > 2 * se
        verdicts.append(ok)
        print(f"   PRIMARY (probes 350/400/450, {len(d)} pairs): crafts@T' control {c.mean():.2f} swap {s.mean():.2f} "
              f"(ratio {s.mean() / max(c.mean(), 1e-9):.2f}, need >= 2) | paired +{d.mean():.2f} SE {se:.2f} (need > 2 SE)"
              f" -> {'pass' if ok else 'fail'}")
    if any(v is None for v in verdicts):
        print("\n== PRIMARY: NOT EVALUABLE in at least one seed")
    else:
        print(f"\n== PRIMARY: {'PASS' if len(verdicts) == 2 and all(verdicts) else 'FAIL'} ({sum(verdicts)}/{len(verdicts)})")


if __name__ == "__main__":
    main(sys.argv[1:])
