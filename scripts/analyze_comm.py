"""Registered analysis for the testbed communication experiment (docs/RESEARCH_PROTOCOL.md, comm registration).

    python scripts/analyze_comm.py testbed_comm_results.jsonl [oracle_source.jsonl]
    (final run: ORACLE is not re-run; its apparatus result comes from attempt 2's file, the warm-start source)

PRIMARY (per COMM seed): 10 paired eval rollouts; Delta_k = m.u toward T' (swap) - (control), listener moves only.
  A seed passes iff mean Delta >= 0.05 AND mean Delta > 2 x SE (SE = sd(Delta_k) / sqrt(n_pairs)).
  The primary passes iff BOTH COMM seeds pass.
APPARATUS (ORACLE): listener correct-hearth fraction over updates N-9..N >= 0.50 with >= 50 listener attempts in
  the window; if it fails, the comm result is uninterpretable.
SECONDARY: listener correct fraction and deposits/world, COMM vs NOSIG (last 10 updates); MI(token; active) per slot
  vs its block-permutation null p99 (COMM seeds). NOSIG correct fraction > 0.25 is logged (memory / search order).
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np


ORACLE0_LISTENER_ATTEMPTS = 435.0  # attempt 2 ORACLE0, mean listener attempts per update over 111-120
CAMPING_BAR = 2 * ORACLE0_LISTENER_ATTEMPTS


def main(path, oracle_path=None):
    halts = [json.loads(l) for l in open(path) if '"phase": "halt"' in l]
    if halts:
        print("== AUTO-HALT fired:")
        for h in halts:
            print(f"  {h['arm']}{h['seed']} at update {h['update']}: sender dep/world {h['sender_dep']:.2f}, "
                  f"token entropy {h['token_entropy']:.2f}")
    train = collections.defaultdict(dict)
    mi, swap = collections.defaultdict(list), collections.defaultdict(list)
    lines = list(open(path)) + ([l for l in open(oracle_path) if '"arm": "ORACLE"' in l] if oracle_path else [])
    for line in lines:
        r = json.loads(line)
        if r.get("phase") == "halt":
            continue
        k = (r["arm"], r["seed"])
        if r.get("phase") == "train":
            train[k][r["update"]] = r
        elif r.get("phase") == "mi":
            mi[k].append(r)
        elif r.get("phase") == "swap":
            swap[k].append(r)
    N = max(max(us) for us in train.values())
    W = range(N - 9, N + 1)
    for k, us in train.items():
        assert all(u in us for u in range(1, N + 1)), f"{k} incomplete"

    print(f"== last 10 updates ({W.start}-{W.stop - 1})")
    summ = {}
    for k in sorted(train):
        us = train[k]
        att = sum(us[u]["listener_attempts"] for u in W)
        ok = sum(us[u]["listener_correct_frac"] * us[u]["listener_attempts"] for u in W)
        frac = ok / att if att else float("nan")
        ldep = float(np.mean([us[u]["listener_dep_per_world"] for u in W]))
        sdep = float(np.mean([us[u]["sender_dep_per_world"] for u in W]))
        summ[k] = (frac, att, ldep)
        print(f"  {k[0]}{k[1]} [{us[N]['gpu']}]: listener correct {frac:.3f} ({att} attempts) | listener dep/world "
              f"{ldep:.2f} | sender dep/world {sdep:.2f} | token entropy {us[N]['token_entropy']:.2f} | EV {us[N]['ev']:+.2f}")

    print("\n== CAMPING CHECK (NOSIG listener attempts/update, last 10, <= 2x ORACLE0's 435)")
    for k in sorted(train):
        if k[0] in ("NOSIG", "COMM"):
            us = train[k]
            att = float(np.mean([us[u]["listener_attempts"] for u in W]))
            dep = float(np.mean([us[u]["listener_dep_per_world"] for u in W])) * 32
            flag = f" -> {'CLOSED' if att <= CAMPING_BAR else 'OPEN'}" if k[0] == "NOSIG" else ""
            print(f"  {k[0]}{k[1]}: {att:.0f} attempts/update, {att / max(dep, 1e-9):.1f} per listener deposit{flag}")

    print("\n== APPARATUS (ORACLE)")
    oracle = [v for k, v in summ.items() if k[0] == "ORACLE"]
    app_ok = bool(oracle) and all(f >= 0.50 and a >= 50 for f, a, _ in oracle)
    print(f"  ORACLE listener correct >= 0.50 with >= 50 attempts: {app_ok} ({[(round(f, 3), a) for f, a, _ in oracle]})")

    print("\n== PRIMARY (token swap, COMM seeds)")
    seeds_ok = []
    for k in sorted(swap):
        d = np.array([p["mu_Tprime_swap"] - p["mu_Tprime_control"] for p in swap[k]])
        mirror = np.array([p["mu_T_swap"] - p["mu_T_control"] for p in swap[k]])
        diff = np.mean([p["swapped_token_differs_frac"] for p in swap[k]])
        se = d.std(ddof=1) / np.sqrt(len(d))
        ok = d.mean() >= 0.05 and d.mean() > 2 * se
        seeds_ok.append(ok)
        print(f"  {k[0]}{k[1]}: Delta {d.mean():+.4f} (SE {se:.4f}, n={len(d)}) -> {'PASS' if ok else 'FAIL'} | mirror "
              f"(toward true T) {mirror.mean():+.4f} | swapped token differs on {diff:.1%} of sender steps")
    primary = len(seeds_ok) == 2 and all(seeds_ok)
    print(f"  PRIMARY: {'PASS' if primary else 'FAIL'}" + ("" if app_ok else "  [apparatus failed -> uninterpretable]"))

    print("\n== SECONDARY")
    for arm in ("COMM", "NOSIG"):
        vals = [v for k, v in summ.items() if k[0] == arm]
        print(f"  {arm}: listener correct {[round(f, 3) for f, _, _ in vals]} | listener dep/world "
              f"{[round(x, 2) for _, _, x in vals]}")
    c = [v[0] for k, v in summ.items() if k[0] == "COMM"]
    n = [v[0] for k, v in summ.items() if k[0] == "NOSIG"]
    if c and n:
        print(f"  COMM correct > NOSIG outside seed spread: {min(c) > max(n)}")
    print(f"  NOSIG correct > 0.25 (logged, memory/search, not the channel): {[f > 0.25 for f in n]}")
    for k in sorted(mi):
        if k[0] != "COMM":
            continue
        rows = sorted(mi[k], key=lambda r: r["slot"])
        print(f"  MI {k[0]}{k[1]}: " + " | ".join(
            f"slot{r['slot']} {r['mi_bits']:.4f} bits vs null p99 {r['null_p99']:.4f}"
            f"{' *' if r['mi_bits'] > r['null_p99'] else ''}" for r in rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
