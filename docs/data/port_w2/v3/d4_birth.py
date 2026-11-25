"""D4 (Cam Nov 24 ~23:55Z item 6): why does first-demonstration failure double in reborn lives (9.8% -> ~20%)?
From a traj npz with --birth-audit (+ --mem-audit): life 0 (env.reset: uniform random positions, food = capacity) vs
reborn lives (respawn at the predecessor's last position). "Hearth" = the young's BIRTH position (no hearth in the code).
Per birth: world food within 12 cells (Chebyshev) and energy. Per FIRST-hearing (demonstrated) bout: food within 12 cells
and energy at the bout open, age, outcome, sated at close, generic food eaten, and the demonstrator's mean / max Chebyshev
distance from the young during the bout. Then first-demonstration failure by food12 tercile (pooled cuts), life 0 vs
reborn, to see whether food at the bout open carries the difference. Usage: d4_birth.py FILE_m0.npz"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1/d1"); import d1_anatomy as D  # noqa: E402
import traj_analysis as TA  # noqa: E402
z = dict(np.load(sys.argv[1])); lid = D.lives(z); age = D.ages(z); W, T, N = z["dead"].shape
births = {"life0": [], "reborn": []}
for w in range(W):
    for n in range(N):
        starts = [0] + [t + 1 for t in np.flatnonzero(z["dead"][w, :, n]) if t + 1 < T]
        for t in starts:
            births["life0" if t == 0 else "reborn"].append((float(z["ba_food12"][w, t, n]), float(z["ba_E"][w, t, n])))
F = []
for b in TA.bouts(z):
    w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
    if not z["first"][w, s0, n]:
        continue
    dd = z["ba_demd"][w, s0:s1 + 1, n].astype(float); dd = dd[dd >= 0]
    F.append(dict(g="life0" if lid[w, s0, n] == 0 else "reborn", food=float(z["ba_food12"][w, s0, n]), E=float(z["ba_E"][w, s0, n]), age=int(age[w, s0, n]),
                  fail=not bool(z["cl_item"][w, s1, n]), sat=not bool(z["hung"][w, s1, n]), gen=bool(z["gen"][w, s1, n]),
                  demd_mean=float(dd.mean()) if len(dd) else None, demd_max=float(dd.max()) if len(dd) else None))
cuts = np.percentile([f["food"] for f in F], [100 / 3, 200 / 3])
out = dict(file=sys.argv[1].split("/")[-1], food12_tercile_cuts=[round(float(c), 2) for c in cuts], births={}, first_bouts={}, fail_by_food_tercile={})
for g, B in births.items():
    a = np.array(B); out["births"][g] = dict(n=len(B), food12_median=round(float(np.median(a[:, 0])), 2), food12_mean=round(float(a[:, 0].mean()), 2), energy_mean=round(float(a[:, 1].mean()), 4))
for g in ("life0", "reborn"):
    G = [f for f in F if f["g"] == g]; fl = [f for f in G if f["fail"]]; ok = [f for f in G if not f["fail"]]
    med = lambda L, k: round(float(np.median([x[k] for x in L if x[k] is not None])), 3) if L else None  # noqa: E731
    out["first_bouts"][g] = dict(n=len(G), fail=round(len(fl) / len(G), 4), food12_median=med(G, "food"), food12_median_failed=med(fl, "food"), food12_median_ok=med(ok, "food"),
                                 energy_open_mean=round(float(np.mean([f["E"] for f in G])), 4), energy_open_mean_failed=round(float(np.mean([f["E"] for f in fl])), 4) if fl else None,
                                 energy_open_mean_ok=round(float(np.mean([f["E"] for f in ok])), 4), age_median=med(G, "age"),
                                 demd_mean_failed=med(fl, "demd_mean"), demd_mean_ok=med(ok, "demd_mean"), demd_max_failed=med(fl, "demd_max"), demd_max_ok=med(ok, "demd_max"),
                                 failed_sated=sum(f["sat"] for f in fl), failed_gen=sum(f["gen"] for f in fl))
    out["fail_by_food_tercile"][g] = {f"t{i}": (lambda L: dict(n=len(L), fail=round(sum(f["fail"] for f in L) / len(L), 4) if L else None))(
        [f for f in G if (i == 0 and f["food"] <= cuts[0]) or (i == 1 and cuts[0] < f["food"] <= cuts[1]) or (i == 2 and f["food"] > cuts[1])]) for i in range(3)}
print(json.dumps(out, indent=1))
