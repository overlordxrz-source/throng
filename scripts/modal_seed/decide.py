"""Chain verdict for one oracle arm (Cam Nov 16 ~17:25Z PASS; ~17:35Z secondary). Reach per sector = item eaten / bouts closed, all
bouts, from the arm's d = 3 and d = 12 trajectory evals; frontier from its train JSON. Prints one JSON line."""
import json, sys
import numpy as np
sys.path.insert(0, "/root/tt/w2"); import traj_analysis as TA
arm, d3, d12, tr = sys.argv[1:5]
def bysec(f):
    z = dict(np.load(f)); e = np.zeros((8, 2))
    for b in TA.bouts(z):
        s = int(z["sec"][b["w"], b["s0"], b["n"]]); e[s, 0] += bool(z["cl_item"][b["w"], b["s1"], b["n"]]); e[s, 1] += 1
    return (e[:, 0] / np.maximum(e[:, 1], 1)).round(3).tolist(), e[:, 1].astype(int).tolist()
r3, n3 = bysec(d3); r12, n12 = bysec(d12); fr = json.load(open(tr))["frontier"]
dead = sorted(set(i for i in range(8) if r3[i] < 0.30 or r12[i] < 0.30))
PASS = min(r3) >= 0.85 and min(r12) >= 0.75 and fr >= 22
SEC = len(dead) == 0 and min(r3) >= 0.75 and fr >= 22
print(json.dumps(dict(arm=arm, frontier=fr, d3=r3, d12=r12, n3=n3, n12=n12, min3=min(r3), min12=min(r12), dead_lt030=dead,
                      dead_le010=[i for i in range(8) if r3[i] <= 0.10 or r12[i] <= 0.10], PASS=PASS, SECONDARY=SEC, QUALIFIES=PASS or SEC)))
