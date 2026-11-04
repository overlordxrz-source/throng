"""Registered analysis for the learner testbed (docs/RESEARCH_PROTOCOL.md, Step 2 registration + addendum).

Per run: decline = mean(mean_dist, updates 1-3) - mean(mean_dist, last 3 updates).
An arm LEARNS iff all 3 seeds have decline >= 0.5 AND min(arm) > max(E).
"X faster than Y" iff min(X) > max(Y); "X ~ Y" if the seed ranges overlap.
Critic bar (C, D): mean EV over the last third of updates >= 0.2 in every seed; if failed, C/D are not read.

    python scripts/analyze_testbed.py testbed_results.jsonl
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

BAR = 0.5
EV_BAR = 0.2


def main(path: str) -> None:
    runs = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        runs[(r["arm"], r["seed"])][r["update"]] = r  # last occurrence wins per update
    N = max(max(u) for u in runs.values())
    first, last = [1, 2, 3], [N - 2, N - 1, N]
    third = list(range(N - N // 3 + 1, N + 1))

    dec, dep, ev = collections.defaultdict(dict), collections.defaultdict(dict), collections.defaultdict(dict)
    for (arm, seed), us in sorted(runs.items()):
        missing = [u for u in range(1, N + 1) if u not in us]
        assert not missing, f"{arm}{seed} missing updates {missing[:5]}"
        d = lambda ws, k: float(np.mean([us[u][k] for u in ws]))
        dec[arm][seed] = d(first, "mean_dist") - d(last, "mean_dist")
        dep[arm][seed] = (d(first, "deposits_per_env"), d(last, "deposits_per_env"))
        ev[arm][seed] = d(third, "ev")
        print(f"{arm}{seed}: dist {d(first,'mean_dist'):.3f} -> {d(last,'mean_dist'):.3f}  decline {dec[arm][seed]:+.3f}"
              f" | deposits/env {dep[arm][seed][0]:.2f} -> {dep[arm][seed][1]:.2f}"
              f" | EV(last third) {ev[arm][seed]:+.3f} | toward-frac {d(last,'frac_moves_toward'):.3f}"
              f" | entropy {d(last,'entropy'):.3f}")

    e_max = max(dec["E"].values())
    print(f"\nE (frozen) declines: {[round(v, 3) for v in dec['E'].values()]}  max {e_max:+.3f}")
    for arm in "ABCD":
        v = list(dec[arm].values())
        learns = all(x >= BAR for x in v) and min(v) > e_max
        crit = ""
        if arm in "CD":
            ok = all(x >= EV_BAR for x in ev[arm].values())
            crit = f" | critic bar {'PASS' if ok else 'FAIL -> not read'} (EV {[round(x, 3) for x in ev[arm].values()]})"
        print(f"{arm}: declines {[round(x, 3) for x in v]} -> LEARNS={learns}{crit}")

    def cmp(x, y):
        lo, hi = min(dec[x].values()), max(dec[y].values())
        if lo > hi:
            return f"{x} faster than {y}"
        if min(dec[y].values()) > max(dec[x].values()):
            return f"{y} faster than {x}"
        return f"{x} ~ {y} (seed ranges overlap)"

    print("\nB vs A:", cmp("B", "A"), "  [Cam: B ~ A; correction: B faster]")
    print("C vs A:", cmp("C", "A"), "  [predicted C faster]")
    print("D vs C:", cmp("D", "C"), "  [predicted D faster]")


if __name__ == "__main__":
    main(sys.argv[1])
