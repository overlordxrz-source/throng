"""Dose-response read (Cam Nov 10 ~18:20, registered decision). Per arm (pooled over seeds) and per seed: eligible lives,
eligibility, pass at death, pass at each snapshot age, ever-pass vs at-death, pass vs cumulative gradient steps, median
lifespan. Decision: any arm >= 50% at death (>= 50 eligible) -> report; R rises with dose (age-1000 pass R128 > R32 > the
all-on 10.4%, two-proportion z > 2 per step) -> compute-limited; R flat and J2000 rises -> experience-limited; all flat
-> the learner is the limit. usage: dose_summary.py <dir>"""
import glob, json, math, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "."
BASE_1000 = (69, 661)          # all-on (replay 8) pooled age-1000 pass, from learn_arms


def z2(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12)); return (k2 / n2 - k1 / n1) / se


A = {}
for arm in ("r32", "r128", "j2000"):
    lives_all, el_rates, seed_rows = [], [], []
    for f in sorted(glob.glob(f"{D}/dose_{arm}_s*_bd6f4af.json")):
        J = json.load(open(f)); S = J["summary"]; h = S["heard25"]
        L = [r for r in J["lives"] if r["scorable"]]; lives_all += L; el_rates.append(h["eligibility_rate"])
        seed_rows.append((f.split("/")[-1][:14], h["eligible"], h["eligibility_rate"], h["learner_pass_frac_eligible"], S["median_lifespan"], S["median_grad_steps_per_life"]))
    n = len(lives_all); kd = sum(r["passed"] for r in lives_all)
    ages = sorted({int(k) for r in lives_all for k in (r.get("snap_pass") or {})})
    by = {a: [r["snap_pass"][str(a)] for r in lives_all if str(a) in (r.get("snap_pass") or {})] for a in ages}
    ever = sum(r["passed"] or any((r.get("snap_pass") or {}).values()) for r in lives_all)
    A[arm] = dict(n=n, k_death=kd, by={a: (sum(v), len(v)) for a, v in by.items()})
    print(f"== {arm}: eligible {n} (mean eligibility {np.mean(el_rates):.0%}); PASS AT DEATH {kd}/{n} = {kd / max(n, 1):.3f}; "
          f"ever {ever / max(n, 1):.3f}")
    for row in seed_rows:
        print(f"   {row[0]}: eligible {row[1]} ({row[2]:.0%}) pass {row[3]:.3f} | median lifespan {row[4]:.0f} | grad steps/life {row[5]:.0f}")
    print("   by age: " + ", ".join(f"{a}: {sum(by[a])}/{len(by[a])} = {sum(by[a]) / len(by[a]):.3f}" for a in ages))
    for a, b in zip(ages, ages[1:]):       # PAIRED: lives with both snapshots (McNemar-style sign test on discordant pairs)
        pr = [(r["snap_pass"][str(a)], r["snap_pass"][str(b)]) for r in lives_all if str(a) in r.get("snap_pass", {}) and str(b) in r.get("snap_pass", {})]
        up = sum((not x) and y for x, y in pr); dn = sum(x and (not y) for x, y in pr)
        zm = (up - dn) / math.sqrt(up + dn) if up + dn else 0.0
        print(f"   paired {a} -> {b}: n {len(pr)}, gained {up}, lost {dn}, McNemar z {zm:+.2f}")
    gs = [(g, r["snap_pass"][k]) for r in lives_all for k, g in (r.get("snap_gsteps") or {}).items() if g is not None and k in r.get("snap_pass", {})]
    if gs:
        g = np.array([x for x, _ in gs]); p = np.array([y for _, y in gs], float)
        qs = np.quantile(g, [0, .25, .5, .75, 1.0])
        print("   pass vs cumulative gradient steps (quartile bins): " + ", ".join(f"[{qs[i]:.0f}-{qs[i + 1]:.0f}] {p[(g >= qs[i]) & (g <= qs[i + 1])].mean():.3f}" for i in range(4)))
print("\n== DECISION (registered)")
passing = [k for k, v in A.items() if v["n"] >= 50 and v["k_death"] / v["n"] >= 0.5]
print("any arm >= 50% at death:", passing or "none")
k32, n32 = A["r32"]["by"].get(1000, (0, 1)); k128, n128 = A["r128"]["by"].get(1000, (0, 1))
z_32 = z2(*BASE_1000, k32, n32); z_128 = z2(k32, n32, k128, n128)
rising_R = (k128 / n128 > k32 / n32 > BASE_1000[0] / BASE_1000[1]) and z_32 > 2 and z_128 > 2
print(f"age-1000 pass: base R8 {BASE_1000[0]}/{BASE_1000[1]} = {BASE_1000[0] / BASE_1000[1]:.3f} | R32 {k32}/{n32} = {k32 / n32:.3f} (z vs R8 {z_32:+.2f}) | "
      f"R128 {k128}/{n128} = {k128 / n128:.3f} (z vs R32 {z_128:+.2f}) -> R rises with dose: {rising_R}")
kj, nj = A["j2000"]["by"].get(1000, (0, 1)); zj = z2(*BASE_1000, kj, nj)
print(f"J2000 age-1000 pass {kj}/{nj} = {kj / nj:.3f} (z vs R8 {zj:+.2f}); J2000 by age {A['j2000']['by']}")
