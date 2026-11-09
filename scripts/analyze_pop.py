"""Registered analysis: separate-brains population (Cam Nov 9 ~14:45, item 2; docs/RESEARCH_PROTOCOL.md).

usage: analyze_pop.py <dir>   (pop_*.jsonl from scripts/pop_toy.py, tags POP / POPCTL, plus their .u{u}.tok.npz dumps)
PRIMARY (ladder rung 3) at u7500, per POP seed (10 probe pairs; all gen-4 brains; before fire 5 is lit):
  reliance: cross-spot drop = 1 - sum(dep_cross) / sum(dep_control) >= 30% AND paired (control - cross) > 2 SE;
  gain:     the seed's true-message deliveries (probe control, listener per world) >= the control population's mean
            (same probe, its 3 seeds) + 1.0;
  same-spot clean: same-spot drop < 0.5 x cross-spot drop.
  PASS iff all three hold in >= 3 of 5 POP seeds.
SECONDARY (use keeps it alive): shuffle-corrected joint MI (Miller-Madow over the 3 slots, all brains' sender steps
  pooled, minus the mean of 100 fire-label permutations) at u7500: POP median >= 2 x control median AND >= control
  median + 0.2 bits. Reported per seed and per generation (u1500 / 3500 / 5500 / 7500) in both arms.
FIRE 5 at u9000 (secondary, reported): redirection to fire 5 = listener Crafts at fire 5 when every sender's message is
  replaced by its own fire-5 word, swap >= 2 x control AND paired increase > 2 SE, in >= 3 of 5 POP seeds; plus which
  age group coins the word (brain ages at 7,500 and their fire-5 word specificity at 9,000).
Also writes per-generation dictionaries (per brain, per fire, the modal slot words) to <dir>/pop_dictionaries.json.
"""
import glob
import json
import os
import sys

import numpy as np


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def joint_mi(tok, lab, ncls):
    V = 64; e = (tok[:, 0] * V + tok[:, 1]) * V + tok[:, 2]
    _, ei = np.unique(e, return_inverse=True)
    J = np.zeros((ei.max() + 1, ncls)); np.add.at(J, (ei, lab), 1); P = J / J.sum()
    px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True); nz = P > 0
    mi = float((P[nz] * np.log2(P[nz] / (px @ py)[nz])).sum())
    return mi - (nz.sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2 * len(lab) * np.log(2))


def corrected_mi(npz, seed=0, perms=100):
    d = np.load(npz); tok, lab = d["tok"], d["lab"]; ncls = int(lab.max()) + 1
    r = np.random.default_rng(seed)
    null = np.mean([joint_mi(tok, r.permutation(lab), ncls) for _ in range(perms)])
    return joint_mi(tok, lab, ncls) - null


def main(d):
    runs = {}
    for p in glob.glob(os.path.join(d, "pop_*.jsonl")):
        R = load(p); runs[p] = R
    pop = {p: R for p, R in runs.items() if R[0]["tag"] == "POP"}; ctl = {p: R for p, R in runs.items() if R[0]["tag"] == "POPCTL"}
    sw = lambda R, u: [r for r in R if r["phase"] == "pop_swap" and r["update"] == u]  # noqa: E731
    ctl_dep = float(np.mean([np.mean([r["dep_control"] for r in sw(R, 7500)]) for R in ctl.values()]))
    print(f"== PRIMARY at u7500 (control-population true-message deliveries {ctl_dep:.2f}; gain bar {ctl_dep + 1.0:.2f})")
    ok_n = 0
    for p, R in sorted(pop.items()):
        P = sw(R, 7500)
        c = np.array([r["dep_control"] for r in P]); x = np.array([r["dep_cross"] for r in P]); s = np.array([r["dep_same"] for r in P])
        dx = 1 - x.sum() / c.sum(); ds = 1 - s.sum() / c.sum(); pd = c - x; se = pd.std(ddof=1) / np.sqrt(len(pd))
        rel = dx >= 0.30 and pd.mean() > 2 * se; gain = c.mean() >= ctl_dep + 1.0; clean = ds < 0.5 * dx
        ok = rel and gain and clean; ok_n += ok
        print(f"   seed {R[0]['seed']}: cross drop {dx:+.0%} (paired {pd.mean():.2f} SE {se:.2f}) | true-message {c.mean():.2f} | "
              f"same-spot {ds:+.0%} -> reliance {'ok' if rel else 'x'} gain {'ok' if gain else 'x'} clean {'ok' if clean else 'x'}")
    print(f"== RUNG 3 (population): {ok_n}/{len(pop)} -> {'PASS' if ok_n >= 3 else 'FAIL'}")
    print("\n== SECONDARY: shuffle-corrected joint MI (bits) per generation")
    mi = {}
    for arm, D in (("POP", pop), ("CTL", ctl)):
        for p, R in sorted(D.items()):
            vals = []
            for u in (1500, 3500, 5500, 7500):
                f = f"{p}.u{u}.tok.npz"
                v = corrected_mi(f) if os.path.exists(f) else float("nan"); vals.append(v)
                mi[(arm, R[0]["seed"], u)] = v
            print(f"   {arm} seed {R[0]['seed']}: " + "  ".join(f"u{u} {v:+.3f}" for u, v in zip((1500, 3500, 5500, 7500), vals)))
    pm = np.median([v for (a, _, u), v in mi.items() if a == "POP" and u == 7500])
    cm = np.median([v for (a, _, u), v in mi.items() if a == "CTL" and u == 7500])
    print(f"== SECONDARY at u7500: POP median {pm:.3f} vs control median {cm:.3f} -> "
          f"{'PASS' if pm >= 2 * cm and pm >= cm + 0.2 else 'FAIL'} (>= 2x and >= +0.2)")
    print("\n== FIRE 5 at u9000")
    f5 = 0
    for p, R in sorted(pop.items()):
        P = sw(R, 9000)
        if not P or "crafts_last_to5" not in P[0]:
            print(f"   seed {R[0]['seed']}: no fire-5 probe"); continue
        c = np.array([r["crafts_last_control"] for r in P]); t = np.array([r["crafts_last_to5"] for r in P]); dd = t - c
        se = dd.std(ddof=1) / np.sqrt(len(dd)); ok = t.mean() >= 2 * c.mean() and dd.mean() > 2 * se; f5 += ok
        births = [r for r in R if r["phase"] == "birth"]
        age7500 = [7500 - max([0] + [b["update"] for b in births if b["slot"] == s and b["update"] <= 7500]) for s in range(4)]
        print(f"   seed {R[0]['seed']}: crafts at fire 5 control {c.mean():.2f} -> told fire 5 {t.mean():.2f} (paired {dd.mean():.2f} SE "
              f"{se:.2f}) -> {'ok' if ok else 'x'} | brain ages at 7,500: {age7500}")
    print(f"== FIRE 5: {f5}/{len(pop)} -> {'PASS' if f5 >= 3 else 'FAIL'}")
    dic = {}
    for p, R in sorted(pop.items()):
        dic[R[0]["seed"]] = {r["update"]: dict(words=r["words"], gen=r["gen"], nav_of=r["nav_of"])
                             for r in R if r["phase"] == "pop_words" and r["update"] in (1500, 3500, 5500, 7500, 9000)}
    json.dump(dic, open(os.path.join(d, "pop_dictionaries.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
