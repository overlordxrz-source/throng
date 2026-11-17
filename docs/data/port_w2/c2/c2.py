"""Cam Nov 17 ~14:10Z C2. Per sector, from a traj eval with d ~ the run's training distribution at the checkpoint
(uniform 1..F, the oracle has no lookahead) and a d = 21 eval: HUNGRY bouts (closed, not by death, never sated);
success = item eaten; extra move energy = d_move (.003) x the number of move actions in the bout (the energy above
standing still); net payoff = .1 x eaten - extra move energy (item energy only: generic intake is not recorded per
bout; the share of bouts with any generic meal is given). usage: c2.py LABEL mixed.npz d21.npz moves8(0/1) dying_sectors"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2"); import traj_analysis as TA
lab, fm, f21, m8, dying = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1", [int(x) for x in sys.argv[5].split(",")]
MV = {1, 2, 3, 4, 6, 7, 8, 9} if m8 else {1, 2, 3, 4}
def rows(f):
    z = dict(np.load(f)); R = []
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
        if not z["hung"][w, s1, n]:
            continue
        A = z["a"][w, s0:s1 + 1, n]; nm = int(sum(int(x) in MV for x in A))
        R.append((int(z["sec"][w, s0, n]), bool(z["cl_item"][w, s1, n]), nm, bool(z["gen"][w, s1, n]), int(z["d"][w, s0, n])))
    return np.array(R)
M = rows(fm); D = rows(f21); out = dict(label=lab, dying=dying, by_sector={})
for s in range(8):
    m = M[M[:, 0] == s]; d = D[D[:, 0] == s]; ex = 0.003 * m[:, 2]; net = 0.1 * m[:, 1] - ex
    lo, hi = m[m[:, 4] <= 12], m[m[:, 4] > 12]
    out["by_sector"][s] = dict(n=len(m), success=round(m[:, 1].mean(), 3), moves=round(m[:, 2].mean(), 2), extra_move_E=round(ex.mean(), 4), net_payoff=round(net.mean(), 4),
                               success_d_le12=round(lo[:, 1].mean(), 3) if len(lo) else None, success_d_gt12=round(hi[:, 1].mean(), 3) if len(hi) else None,
                               gen_share=round(m[:, 3].mean(), 3), n21=len(d), reach21=round(d[:, 1].mean(), 3) if len(d) else None)
bs = out["by_sector"]; low_net = min(bs, key=lambda s: bs[s]["net_payoff"]); low21 = min(bs, key=lambda s: bs[s]["reach21"])
out.update(lowest_net=low_net, lowest_reach21=low21, hit=bool(low_net in dying or low21 in dying), d_max_seen=int(M[:, 4].max()))
print(json.dumps(out))
