"""Seed 8 at 1e8: how later right-head bouts fail, by item sector (d = 3 / 12 traj npz). Per sector: reach; among the
moves in those bouts, the share that are diagonal (actions 6..9), cardinal (1..4), stay (0), eat (5); for TIMED-OUT
bouts, the min Chebyshev distance to the item and the share that came within visibility (<= 1) without eating."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2"); import traj_analysis as TA  # noqa: E402
out = {}
for f in sys.argv[1:]:
    z = dict(np.load(f)); S = {s: dict(n=0, ate=0, mv=np.zeros(10), to=0, vis_noeat=0, mind=[]) for s in range(8)}
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; sec = int(z["sec"][w, s0, n])
        if z["dem"][w, s0, n] or z["first"][w, s0, n] or int(z["hl"][w, s0, n]) != sec:
            continue
        e = S[sec]; e["n"] += 1; ate = bool(z["cl_item"][w, s1, n]); e["ate"] += ate
        A = z["a"][w, s0:s1 + 1, n].astype(int); e["mv"] += np.bincount(A, minlength=10)[:10]
        if z["cl_tmax"][w, s1, n]:
            ip = z["ipos"][w, s0, n].astype(int); md = min(int(np.abs(TA.torus(ip - z["pos"][w, s, n].astype(int))).max()) for s in range(s0, s1 + 1))
            e["to"] += 1; e["mind"].append(md); e["vis_noeat"] += int(md <= 1)
    r = {}
    for s, e in S.items():
        t = e["mv"].sum()
        r[s] = dict(n=e["n"], reach=round(e["ate"] / max(e["n"], 1), 3), diag=round(e["mv"][6:].sum() / t, 3), card=round(e["mv"][1:5].sum() / t, 3),
                    stay=round(e["mv"][0] / t, 3), eat=round(e["mv"][5] / t, 3), timeouts=e["to"], to_vis_noeat=round(e["vis_noeat"] / max(e["to"], 1), 3),
                    to_mind=round(float(np.mean(e["mind"])) if e["mind"] else -1, 2))
    out[f.split("/")[-1]] = r
print(json.dumps(out, indent=0))
