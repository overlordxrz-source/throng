"""Cam Nov 13 ~11:00, registered BEFORE the data: the v2 decision at 50M (and the same PASS clause at 100M).
usage: python decide_50m.py evo2_<sha>.json [--at 5e7 --prev 2.5e7]
ratio = hungry memory-kept food true/deranged at the milestone; forage = adult-food ratio vs the navigator.
PASS: ratio >= 1.5 AND forage >= 0.70 AND ratio >= ratio(prev).  CLIMBING: 1.1 <= ratio < 1.5 AND ratio > ratio(prev).
FAIL: otherwise."""
import json, sys
a = sys.argv; f = a[1]; at = float(a[a.index("--at") + 1]) if "--at" in a else 5e7; prev = float(a[a.index("--prev") + 1]) if "--prev" in a else 2.5e7
R = json.load(open(f))["summary"]["readouts"]
def pick(x):
    c = [r for r in R if abs(float(r["tag"]) - x) / x < 1e-6]
    return c[-1] if c else None
def ratio(r):
    t = r["tests"]["hungry_memory_carried"]; return t["eaten_true"] / t["eaten_swap"] if t.get("eaten_swap") else None
def forage(r):
    return r["forage"]["check"]["ratio_adult_food_per_step"]["point"]
r1, r0 = pick(at), pick(prev)
assert r1 is not None and r0 is not None, "milestone readouts missing"
q1, q0, fo = ratio(r1), ratio(r0), forage(r1)
v = "PASS" if (q1 >= 1.5 and fo is not None and fo >= 0.70 and q1 >= q0) else ("CLIMBING" if (1.1 <= q1 < 1.5 and q1 > q0) else "FAIL")
t1 = r1["tests"]["hungry_memory_carried"]
print(json.dumps(dict(verdict=v, at=at, ratio=q1, ratio_prev=q0, forage=fo, pass_frac=t1["pass_frac"], ci95=t1["ci95"], eligible=t1["eligible"],
                      ti=r1.get("ti_fixed"))))
