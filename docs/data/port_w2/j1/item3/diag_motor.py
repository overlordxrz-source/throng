"""Item 3, part 2: the MOTOR signature of seed 9's right-head DIAGONAL bouts at d = 12 (later bouts), ate vs failed.
In 4-way a diagonal item needs both axes: per bout, the moves split into toward-x, toward-y (the item's signs), away
(either axis), stay, eat; the axis balance |nx - ny| / (nx + ny) at the bout end; and, for timed-out bouts, the item's
offset from the agent at the end decomposed into ALONG-track (toward the ray) and CROSS-track error, plus the Chebyshev
distance. Also p(stay) / p(eat) shares. Confidence split replaced by p < 1 vs p == 1 (float16; the head saturates)."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2"); import traj_analysis as TA  # noqa: E402
z = dict(np.load(sys.argv[1])); MV = TA.MOVES; out = {}
for b in TA.bouts(z):
    w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
    if z["dem"][w, s0, n] or z["first"][w, s0, n]:
        continue
    sec = int(z["sec"][w, s0, n])
    if int(z["hl"][w, s0, n]) != sec:
        continue
    ate = bool(z["cl_item"][w, s1, n]); tmax = bool(z["cl_tmax"][w, s1, n]); orb = "diag" if sec % 2 else "card"
    st = z["start"][w, s0, n].astype(int); ip = z["ipos"][w, s0, n].astype(int); tgt = TA.torus(ip - st); sx, sy = np.sign(tgt)
    A = z["a"][w, s0:s1 + 1, n].astype(int); nx = ny = away = stay = eat = 0
    for a in A:
        if a == 0: stay += 1
        elif a == 5: eat += 1
        else:
            m = MV[a]
            if m[0] != 0 and np.sign(m[0]) == sx: nx += 1
            elif m[1] != 0 and np.sign(m[1]) == sy: ny += 1
            else: away += 1
    L = max(len(A), 1); e = out.setdefault(f"{orb}_{'ate' if ate else ('timeout' if tmax else 'other')}", dict(n=0, x=0., y=0., away=0., stay=0., eat=0., imb=0., along=[], cross=[], cheb=[], psat=0))
    e["n"] += 1; e["x"] += nx / L; e["y"] += ny / L; e["away"] += away / L; e["stay"] += stay / L; e["eat"] += eat / L
    e["imb"] += abs(nx - ny) / max(nx + ny, 1); e["psat"] += int(float(z["hpf"][w, s0, n][sec]) >= 1.0)
    if tmax:
        pend = z["pos"][w, s1, n].astype(int); rel = TA.torus(ip - pend); u = TA.U[sec]
        e["along"].append(float(rel @ u)); e["cross"].append(float(abs(rel[0] * u[1] - rel[1] * u[0]))); e["cheb"].append(int(np.abs(rel).max()))
res = {}
for k, e in sorted(out.items()):
    nn = max(e["n"], 1)
    res[k] = dict(n=e["n"], share_toward_x=round(e["x"] / nn, 3), share_toward_y=round(e["y"] / nn, 3), share_away=round(e["away"] / nn, 3),
                  share_stay=round(e["stay"] / nn, 3), share_eat=round(e["eat"] / nn, 3), axis_imbalance=round(e["imb"] / nn, 3), head_p_saturated=round(e["psat"] / nn, 3),
                  **({"end_along_mean": round(np.mean(e["along"]), 2), "end_cross_mean": round(np.mean(e["cross"]), 2), "end_cheb_mean": round(np.mean(e["cheb"]), 2), "end_cheb_le1": round(np.mean(np.array(e["cheb"]) <= 1), 3)} if e["along"] else {}))
print(json.dumps(res, indent=1))
