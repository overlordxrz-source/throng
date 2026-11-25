"""v3 metric checks (Cam Nov 24 ~23:55Z rule v3), eval only.
(1) IDENTITY: every key the old code (abed942) wrote is bit-equal in the new code's JSON (traj and rl2 evals, each mode).
(2) REBUILD: the new hungry_reach_2nd_half_nd (+ bouts, + by sector) equals an offline rebuild from the traj npz:
    bouts closing hungry (hung) at age >= A_max/2 whose HEARING step was not demonstrated (dem at s0).
(3) NEGATIVE: the nd metric differs from the all-bouts hungry_reach_2nd_half when demonstrated bouts exist.
Usage: check_v3_metric.py OLD_traj.json NEW_traj.json NEW_traj_m0.npz [NEW_traj_m3.npz] -- OLD_rl2.json NEW_rl2.json"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1/d1"); import d1_anatomy as D  # noqa: E402
import traj_analysis as TA  # noqa: E402
a = sys.argv[1:]; cut = a.index("--"); T, R = a[:cut], a[cut + 1:]
NEWK = ("hungry_reach_2nd_half_nd", "hungry_bouts_2nd_half_nd", "hungry_reach_2nd_half_nd_bysec", "hungry_bouts_2nd_half_nd_bysec")
out = dict(identity={}, rebuild={}, negative={})
for nm, (fo, fn) in (("traj", T[:2]), ("rl2", R[:2])):
    o, n = json.load(open(fo)), json.load(open(fn))
    for m in o["modes"]:
        diff = [k for k in o["modes"][m] if o["modes"][m][k] != n["modes"][m].get(k)]
        extra = sorted(set(n["modes"][m]) - set(o["modes"][m]))
        out["identity"][f"{nm}_m{m}"] = dict(old_keys=len(o["modes"][m]), differing=diff, new_keys=extra, PASS=not diff and extra == sorted(NEWK))
        e = n["modes"][m]; out["negative"][f"{nm}_m{m}"] = dict(all=e["hungry_reach_2nd_half"], nd=e["hungry_reach_2nd_half_nd"], n_all=e["hungry_bouts_2nd_half"], n_nd=e["hungry_bouts_2nd_half_nd"],
                                                                 PASS=e["hungry_bouts_2nd_half_nd"] < e["hungry_bouts_2nd_half"])
J = json.load(open(T[1]))["modes"]
for f in T[2:]:
    m = f.split("_m")[-1].split(".")[0]; z = dict(np.load(f)); age = D.ages(z); A = np.zeros(8); C = np.zeros(8)
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
        if z["hung"][w, s1, n] and age[w, s1, n] >= D.HALF and not z["dem"][w, s0, n]:
            s = int(z["sec"][w, s0, n]); C[s] += 1; A[s] += bool(z["cl_item"][w, s1, n])
    e = J[m]; rb = A.sum() / C.sum(); rbs = [A[s] / C[s] for s in range(8)]
    out["rebuild"][f"m{m}"] = dict(rebuilt=rb, eval=e["hungry_reach_2nd_half_nd"], n_rebuilt=float(C.sum()), n_eval=e["hungry_bouts_2nd_half_nd"],
                                   bysec_maxdiff=float(max(abs(x - y) for x, y in zip(rbs, e["hungry_reach_2nd_half_nd_bysec"]))),
                                   PASS=abs(rb - e["hungry_reach_2nd_half_nd"]) < 1e-6 and C.sum() == e["hungry_bouts_2nd_half_nd"] and list(C) == e["hungry_bouts_2nd_half_nd_bysec"])
out["ALL_PASS"] = all(v["PASS"] for g in ("identity", "rebuild", "negative") for v in out[g].values())
print(json.dumps(out, indent=1))
