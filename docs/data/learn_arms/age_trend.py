"""Registered decision-tree read (Cam Nov 10): pooled all-on reliance by snapshot age, two-proportion z between
consecutive ages (the later set is a subset of the earlier; flagged deviation from 'paired'), and the age at which a
fit would reach 30%. usage: age_trend.py <dir>"""
import glob, json, math, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "."
for arm in ("allon", "nolev"):
    K, N = {}, {}
    for f in sorted(glob.glob(f"{D}/learn_{arm}_s*_62b6758.json")):
        h = json.load(open(f))["summary"]["heard25"]
        for k, v in h["by_age"].items():
            if v.get("n"):
                K[k] = K.get(k, 0) + round(v["pass_frac"] * v["n"]); N[k] = N.get(k, 0) + v["n"]
    ages = sorted(int(k) for k in K)
    p = {a: K[str(a)] / N[str(a)] for a in ages}
    print(f"== {arm}: " + ", ".join(f"age {a}: {K[str(a)]}/{N[str(a)]} = {p[a]:.3f}" for a in ages))
    zs = []
    for a, b in zip(ages, ages[1:]):
        k1, n1, k2, n2 = K[str(a)], N[str(a)], K[str(b)], N[str(b)]
        pp = (k1 + k2) / (n1 + n2); se = math.sqrt(max(pp * (1 - pp) * (1 / n1 + 1 / n2), 1e-12))
        z = (k2 / n2 - k1 / n1) / se; zs.append(z); print(f"   {a} -> {b}: z {z:+.2f}")
    rising = len(zs) > 0 and all(z > 2 for z in zs)
    print(f"   RISING (every step z > 2): {rising}")
    if arm == "allon" and p[ages[-1]] > p[ages[0]]:
        x = np.array(ages, float); y = np.array([p[a] for a in ages]); w = np.array([N[str(a)] for a in ages], float)
        b1, b0 = np.polyfit(x, y, 1, w=np.sqrt(w)); print(f"   linear fit: {b1 * 1000:.3f} per 1,000 steps -> 30% at age ~{(0.30 - b0) / b1:,.0f}")
        c1, c0 = np.polyfit(np.log(x), y, 1, w=np.sqrt(w)); print(f"   log-age fit: {c1:.3f} per e-fold -> 30% at age ~{math.exp((0.30 - c0) / c1):,.0f}")
