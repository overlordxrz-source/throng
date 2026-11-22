"""Cam Nov 21 ~21:20Z item 3: the d = 12 decomposition, per sector, from a learner traj npz (learner_w2_rl_traj.py,
--d-fixed 12, mode 0): over LATER bouts (not demonstrated, not a first hearing),
    reach = P(head right) x reach | head right + P(head wrong) x reach | head wrong,
where head right = the head's argmax at the hearing step == the item's sector. Also all bouts (incl. first hearings and
demonstrated) for reference. Usage: decomp_d12.py file.npz [...] -> one JSON line per file."""
import sys, json, numpy as np
sys.path.insert(0, "/root/tt/w2"); import traj_analysis as TA  # noqa: E402
for f in sys.argv[1:]:
    z = dict(np.load(f)); S = {s: dict(n=0, nr=0, ar=0, nw=0, aw=0) for s in range(8)}; allb = [0, 0]
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; sec = int(z["sec"][w, s0, n]); ate = bool(z["cl_item"][w, s1, n])
        allb[0] += ate; allb[1] += 1
        if z["dem"][w, s0, n] or z["first"][w, s0, n]:
            continue
        e = S[sec]; e["n"] += 1
        if int(z["hl"][w, s0, n]) == sec:
            e["nr"] += 1; e["ar"] += ate
        else:
            e["nw"] += 1; e["aw"] += ate
    out = {}
    for s, e in S.items():
        pr = e["nr"] / max(e["n"], 1); rr = e["ar"] / max(e["nr"], 1); rw = e["aw"] / max(e["nw"], 1)
        out[s] = dict(n=e["n"], p_right=round(pr, 3), reach_right=round(rr, 3), reach_wrong=round(rw, 3), reach=round(pr * rr + (1 - pr) * rw, 3))
    tot = {k: sum(e[k] for e in S.values()) for k in ("n", "nr", "ar", "nw", "aw")}
    out["later_all"] = dict(n=tot["n"], p_right=round(tot["nr"] / max(tot["n"], 1), 3), reach_right=round(tot["ar"] / max(tot["nr"], 1), 3),
                            reach_wrong=round(tot["aw"] / max(tot["nw"], 1), 3), reach=round((tot["ar"] + tot["aw"]) / max(tot["n"], 1), 3))
    out["all_bouts_reach"] = round(allb[0] / max(allb[1], 1), 3)
    print(json.dumps({f.split("/")[-1]: out}))
