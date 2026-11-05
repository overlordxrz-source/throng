"""Registered analyses for Step 0 (critic fix) and Step 1 (map-size ladder) -- docs/RESEARCH_PROTOCOL.md.

    python scripts/analyze_ladder.py step0  testbed_step0_results.jsonl
    python scripts/analyze_ladder.py cretest testbed_cretest_results.jsonl
    python scripts/analyze_ladder.py ladder testbed_ladder_results.jsonl
    python scripts/analyze_ladder.py stage3 testbed_stage3_results.jsonl
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

OLD_A_MIN_DECLINE = 0.966  # smallest Step 2 arm-A decline (docs/data/testbed_2026-11-04_analysis.txt)
STEP2_E_MAX_DECLINE = -0.006  # largest Step 2 arm-E (frozen) decline, same file


def load(path):
    runs = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if "mean_dist" not in r:      # timing-probe records carry no metrics
            continue
        runs[(r["arm"], r["seed"], r["grid"], r.get("vf_coef"))][r["update"]] = r
    for k, us in runs.items():
        missing = [u for u in range(1, 81) if u not in us]
        assert not missing, f"{k} missing updates {missing[:5]}"
    return runs


def win(us, ws, key):
    return float(np.mean([us[u][key] for u in ws]))


def step0(path):
    runs = load(path)
    ok_all = True
    for (arm, seed, grid, _vf), us in sorted(runs.items()):
        ev = win(us, range(36, 41), "ev")
        dec = win(us, (1, 2, 3), "mean_dist") - win(us, (78, 79, 80), "mean_dist")
        ok = ev >= 0.2 and dec >= OLD_A_MIN_DECLINE
        ok_all &= ok
        print(f"{arm}{seed} g{grid} [{us[80]['gpu']}]: EV(36-40) {ev:+.3f} (bar 0.2) | decline {dec:+.3f} "
              f"(bar {OLD_A_MIN_DECLINE}) | deposits/env last3 {win(us, (78, 79, 80), 'deposits_per_env'):.2f} "
              f"| EV(78-80) {win(us, (78, 79, 80), 'ev'):+.3f} -> {'PASS' if ok else 'FAIL'}")
    print(f"\nStep 0: {'PASS -> fix lands in production' if ok_all else 'FAIL'}")


def cretest(path):
    """Cam's C' re-test: per vf, every seed needs EV(36-40) >= 0.2, decline >= 0.966 and > every Step 2 E seed.
    Both vf pass -> take 0.1; one passes -> that one; neither -> stop and report."""
    runs = load(path)
    passed = {}
    for vf in sorted({k[3] for k in runs}):
        ok_vf = True
        for (arm, seed, grid, v), us in sorted(runs.items()):
            if v != vf:
                continue
            ev = win(us, range(36, 41), "ev")
            dec = win(us, (1, 2, 3), "mean_dist") - win(us, (78, 79, 80), "mean_dist")
            ok = ev >= 0.2 and dec >= OLD_A_MIN_DECLINE and dec > STEP2_E_MAX_DECLINE
            ok_vf &= ok
            print(f"C'{seed} vf={vf} [{us[80]['gpu']}]: EV(36-40) {ev:+.3f} | EV(78-80) {win(us, (78, 79, 80), 'ev'):+.3f}"
                  f" | decline {dec:+.3f} | deposits/env last3 {win(us, (78, 79, 80), 'deposits_per_env'):.2f}"
                  f" -> {'PASS' if ok else 'FAIL'}")
        passed[vf] = ok_vf
    print(f"\nper-vf pass: {passed}")
    winners = [v for v in sorted(passed) if passed[v]]
    print(f"cretest: {'PASS -> take vf=' + format(winners[0], 'g') if winners else 'FAIL -> stop and report'}")


def ladder(path):
    runs = load(path)
    W = range(71, 81)
    for grid in sorted({k[2] for k in runs}):
        e_rates = [win(us, W, "deposits_per_env") for (a, s, g, _v), us in runs.items() if a == "E" and g == grid]
        floor = 0.5 / (max(len(e_rates), 1) * 10 * 32)  # 0.5 deposits over E's pooled window
        e = max(float(np.mean(e_rates)), floor) if e_rates else float("nan")
        print(f"\n== {grid}x{grid}: E rate {e:.4f} (seeds {[round(x, 4) for x in e_rates]})")
        for arm in "AC":
            rows = sorted(((s, us) for (a, s, g, _v), us in runs.items() if a == arm and g == grid), key=lambda t: t[0])
            ratios = []
            for s, us in rows:
                r = win(us, W, "deposits_per_env")
                ratios.append(r / e)
                print(f"  {arm}{s} [{us[80]['gpu']}]: rate {r:.4f} = {r / e:.2f}x E | EV(71-80) {win(us, W, 'ev'):+.3f}"
                      f" | dist {win(us, (1, 2, 3), 'mean_dist'):.2f} -> {win(us, (78, 79, 80), 'mean_dist'):.2f}")
            learns = len(ratios) == 2 and all(x >= 2.0 for x in ratios)
            print(f"  {arm}{chr(39) if arm == 'C' else ''} LEARNS at {grid}: {learns}")
    critic_check(runs)


def critic_check(runs):
    """Cam's free check on the EV collapse (registered with the ladder amendment). Per C' run: Var(target) and
    Var(target - V), ramp window (updates 11-20) vs plateau window (71-80).
    artifact   : Var(target) plateau/ramp < 0.5 AND Var(resid) plateau/ramp <= 2  (target variance collapses,
                 residual flat -- EV blows up on small V jitter)
    diverging  : Var(resid) plateau/ramp > 2                                         (V really drifts -> PopArt first)
    otherwise  : neither"""
    print("\n== critic check (C' runs): Var(target), Var(target - V), ramp 11-20 vs plateau 71-80")
    for (arm, seed, grid, _v), us in sorted(runs.items()):
        if arm != "C" or "var_target" not in us[80]:
            continue
        vt_r, vt_p = win(us, range(11, 21), "var_target"), win(us, range(71, 81), "var_target")
        vr_r, vr_p = win(us, range(11, 21), "var_resid"), win(us, range(71, 81), "var_resid")
        t_ratio, r_ratio = vt_p / max(vt_r, 1e-12), vr_p / max(vr_r, 1e-12)
        verdict = "diverging" if r_ratio > 2 else ("artifact" if t_ratio < 0.5 else "neither")
        print(f"  C'{seed} g{grid}: Var(target) {vt_r:.4g} -> {vt_p:.4g} (x{t_ratio:.2f}) | Var(resid) {vr_r:.4g} -> "
              f"{vr_p:.4g} (x{r_ratio:.2f}) | mean V {win(us, range(71, 81), 'v_mean'):.3f} vs target "
              f"{win(us, range(71, 81), 'target_mean'):.3f} -> {verdict}")


LADDER_E = {32: 0.1469, 64: 0.0250, 128: 0.0094}  # ladder E rates, updates 71-80 (testbed_ladder_2026-11-05)


def stage3(path):
    """Cam's stage 3. Transfer: the frozen final C'@32 policy, 10 eval rollouts at 64 and 128; rate = mean
    deposits/env over all 10; transfers at a size iff both seeds >= 2x the ladder's E rate there. A seed whose
    C'@32 training run did not itself reach >= 2x the ladder E rate at 32 (updates 71-80) voids its transfer
    reading. PopArt: critic fixed iff Var(resid) plateau(71-80)/ramp(11-20) <= 2 in both 32 seeds; extends reach
    iff both 64 seeds reach >= 2x the ladder E rate at 64."""
    train, ev = collections.defaultdict(dict), collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if "mean_dist" not in r:
            continue
        if r.get("phase") == "eval":
            ev[(r["seed"], r["grid"])].append(r["deposits_per_env"])
        else:
            train[("popart" if r.get("popart") else "fix", r["seed"], r["grid"])][r["update"]] = r
    for k, us in train.items():
        assert all(u in us for u in range(1, 81)), f"{k} incomplete"

    print("== C'@32 (critic fix, vf 0.5) re-run with saving -- training sanity")
    valid = {}
    for seed in (0, 1):
        us = train[("fix", seed, 32)]
        r = win(us, range(71, 81), "deposits_per_env")
        valid[seed] = r >= 2 * LADDER_E[32]
        print(f"  C'{seed} [{us[80]['gpu']}]: rate {r:.3f} = {r / LADDER_E[32]:.1f}x ladder E -> "
              f"{'valid' if valid[seed] else 'VOID (did not learn at 32)'}")
    print("== Transfer (frozen C'@32 policy)")
    for G in (64, 128):
        ratios = []
        for seed in (0, 1):
            xs = ev[(seed, G)]
            assert len(xs) == 10, (seed, G, len(xs))
            ratios.append(float(np.mean(xs)) / LADDER_E[G])
            print(f"  seed {seed} @ {G}: {float(np.mean(xs)):.4f}/env-rollout = {ratios[-1]:.2f}x ladder E")
        ok = all(valid.values()) and all(x >= 2 for x in ratios)
        print(f"  TRANSFERS at {G}: {ok}" + ("" if all(valid.values()) else " (void seed present)"))
    print("== PopArt (C', vf 0.5)")
    fixed = []
    for seed in (0, 1):
        us = train[("popart", seed, 32)]
        vr = win(us, range(71, 81), "var_resid") / max(win(us, range(11, 21), "var_resid"), 1e-12)
        vt = win(us, range(71, 81), "var_target") / max(win(us, range(11, 21), "var_target"), 1e-12)
        fixed.append(vr <= 2)
        print(f"  PopArt{seed} @32 [{us[80]['gpu']}]: Var(resid) x{vr:.2f}, Var(target) x{vt:.2f}, EV(71-80) "
              f"{win(us, range(71, 81), 'ev'):+.3f}, rate {win(us, range(71, 81), 'deposits_per_env'):.3f} = "
              f"{win(us, range(71, 81), 'deposits_per_env') / LADDER_E[32]:.1f}x E")
    print(f"  CRITIC FIXED (both 32 seeds Var(resid) <= 2x): {all(fixed)}")
    reach = []
    for seed in (0, 1):
        us = train[("popart", seed, 64)]
        r = win(us, range(71, 81), "deposits_per_env")
        reach.append(r >= 2 * LADDER_E[64])
        print(f"  PopArt{seed} @64 [{us[80]['gpu']}]: rate {r:.4f} = {r / LADDER_E[64]:.2f}x ladder E, EV(71-80) "
              f"{win(us, range(71, 81), 'ev'):+.3f}")
    print(f"  EXTENDS REACH to 64 (both seeds >= 2x E): {all(reach)}")


if __name__ == "__main__":
    {"step0": step0, "cretest": cretest, "ladder": ladder, "stage3": stage3}[sys.argv[1]](sys.argv[2])
