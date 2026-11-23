"""Cam Nov 23 ~05:10Z item 3 (eval only): seed 9 at 1e8, d = 12, LATER bouts (not demonstrated, not a first hearing).
At the hearing step: the head's full softmax (hpf, --head-probs). Reports:
(1) reach | head right, by head CONFIDENCE tercile (max prob) and by ORBIT (cardinal sectors 0/2/4/6, diagonal 1/3/5/7);
(2) the head's probability mass on the two NEIGHBOURS of the true sector (sec +- 1) and on the true sector, by orbit and by
    outcome (ate / not), for right-head bouts;
(3) for right-head bouts that FAILED vs succeeded: the share of moves in the bout that point into the true sector vs into a
    neighbouring sector (4-way moves; a move points into a sector if within 45 deg).
The question: is the diagonal walk loss MOTOR, or a SOFT READ that splits the walk between neighbours?"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2"); import traj_analysis as TA  # noqa: E402
z = dict(np.load(sys.argv[1])); rows = []
for b in TA.bouts(z):
    w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
    if z["dem"][w, s0, n] or z["first"][w, s0, n]:
        continue
    sec = int(z["sec"][w, s0, n]); p = z["hpf"][w, s0, n].astype(np.float64); hl = int(z["hl"][w, s0, n]); ate = bool(z["cl_item"][w, s1, n])
    A = z["a"][w, s0:s1 + 1, n].astype(int); mv = [a for a in A if 1 <= a <= 4]
    into_true = np.mean([TA.into(a, sec) for a in mv]) if mv else np.nan
    into_nb = np.mean([TA.into(a, (sec + 1) % 8) or TA.into(a, (sec - 1) % 8) for a in mv]) if mv else np.nan
    rows.append(dict(sec=sec, orbit="diag" if sec % 2 else "card", right=hl == sec, ate=ate, pmax=float(p.max()), ptrue=float(p[sec]),
                     pnb=float(p[(sec + 1) % 8] + p[(sec - 1) % 8]), into_true=into_true, into_nb=into_nb))
R = [r for r in rows if r["right"]]; q = np.quantile([r["pmax"] for r in R], [1 / 3, 2 / 3])
def terc(x): return 0 if x <= q[0] else 1 if x <= q[1] else 2
out = dict(file=sys.argv[1], n_later=len(rows), n_right=len(R), p_right=round(len(R) / max(len(rows), 1), 3), tercile_cuts=[round(float(v), 3) for v in q], by_conf_orbit={}, mass={}, moves={})
for o in ("card", "diag"):
    for t in range(3):
        S = [r for r in R if r["orbit"] == o and terc(r["pmax"]) == t]
        out["by_conf_orbit"][f"{o}_t{t}"] = dict(n=len(S), reach_right=round(np.mean([r["ate"] for r in S]), 3) if S else None, pmax_mean=round(np.mean([r["pmax"] for r in S]), 3) if S else None)
    for ate in (True, False):
        S = [r for r in R if r["orbit"] == o and r["ate"] == ate]
        k = f"{o}_{'ate' if ate else 'failed'}"
        out["mass"][k] = dict(n=len(S), p_true=round(np.mean([r["ptrue"] for r in S]), 3) if S else None, p_neighbours=round(np.mean([r["pnb"] for r in S]), 3) if S else None)
        out["moves"][k] = dict(into_true=round(np.nanmean([r["into_true"] for r in S]), 3) if S else None, into_neighbour=round(np.nanmean([r["into_nb"] for r in S]), 3) if S else None)
print(json.dumps(out, indent=1))
