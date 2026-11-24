"""D1 / D2 follow-ups on a mode-0 traj (24,000-step eval = 8 lives per agent):
(1) by LIFE (life 0 starts from env.reset; later lives are reborn after an old-age death): first-hearing (demonstrated)
    bout failure, the median age at first hearings, and for later bouts P(head right) and reach. The 3,000-step evals
    (decomp_d12, head_later) see life 0 only.
(2) first-hearing bouts by outcome x generic (world) food eaten during the bout x sated at the close (sat = energy
    reached H during the bout; the eval's `hung` = closed & ~dead & ~sat), for life 0 and reborn lives.
(3) right-head LATER bouts by orbit: eaten / failed sated / failed hungry (the bar counts hungry bouts only).
Usage: life_sat.py FILE_m0.npz"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1/d1"); import d1_anatomy as D  # noqa: E402
import traj_analysis as TA  # noqa: E402
z = dict(np.load(sys.argv[1])); lid = D.lives(z); age = D.ages(z)
L1, F2, R3 = {}, {}, {}
for b in TA.bouts(z):
    w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; sec = int(z["sec"][w, s0, n]); ate = bool(z["cl_item"][w, s1, n])
    gen = bool(z["gen"][w, s1, n]); sat = not bool(z["hung"][w, s1, n]); li = int(lid[w, s0, n])
    e = L1.setdefault(str(min(li, 3)), dict(first=0, first_fail=0, ages=[], later=0, right=0, reach=0))
    if z["first"][w, s0, n]:
        e["first"] += 1; e["first_fail"] += int(not ate); e["ages"].append(int(age[w, s0, n]))
        k = ("ate" if ate else "fail") + ("_gen" if gen else "_nogen") + ("" if ate else ("_sat" if sat else "_hungry"))
        g = F2.setdefault("life0" if li == 0 else "reborn", {}); g[k] = g.get(k, 0) + 1
    elif not z["dem"][w, s0, n]:
        e["later"] += 1; ok = int(z["hl"][w, s0, n]) == sec; e["right"] += int(ok); e["reach"] += int(ate)
        if ok:
            k = "ate" if ate else ("fail_sat" if sat else "fail_hungry")
            g = R3.setdefault("diag" if sec % 2 else "card", {}); g[k] = g.get(k, 0) + 1
out = dict(file=sys.argv[1].split("/")[-1],
           by_life={k: dict(first=e["first"], first_fail=round(e["first_fail"] / e["first"], 3), first_age_median=float(np.median(e["ages"])),
                            later=e["later"], p_right=round(e["right"] / e["later"], 3), reach=round(e["reach"] / e["later"], 3)) for k, e in sorted(L1.items())},
           first_bouts=F2,
           right_later={o: dict(g, reach=round(g.get("ate", 0) / sum(g.values()), 3), reach_excl_sated=round(g.get("ate", 0) / (g.get("ate", 0) + g.get("fail_hungry", 0)), 3))
                        for o, g in R3.items()})
print(json.dumps(out))
