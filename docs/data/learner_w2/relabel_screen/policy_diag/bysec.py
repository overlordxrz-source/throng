import sys, numpy as np
sys.path.insert(0, "w2"); import traj_analysis as TA
def run(f, oracle=False):
    z = dict(np.load(f)); R = {}; prev = {}; H = {}
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; g = lambda k, s: z[k][w, s, n]
        sec = int(g("sec", s0)); ate = bool(g("cl_item", s1)); dem = bool(g("dem", s0)); first = bool(g("first", s0))
        cat = "oracle" if oracle else ("dem" if dem else ("first_nodem" if first else ("later_right" if int(g("hl", s0)) == sec else "later_wrong")))
        e = R.setdefault(cat, np.zeros((8, 2), int)); e[sec, 0] += ate; e[sec, 1] += 1
        if not oracle and not dem and not first:   # D5 follow-up: the previous bout of the same sector in this life (same token: code fixed per life)
            # find previous closed bout of same sector in this life
            key = (w, n); pv = prev.get(key, {}).get(sec)
            lab = "none" if pv is None else ("prev_solved" if pv else "prev_unsolved")
            h = H.setdefault(lab, [0, 0]); h[0] += int(int(g("hl", s0)) == sec); h[1] += 1
        if not oracle:
            if z["dead"][w, s0:s1 + 1, n].any(): prev[(w, n)] = {}
            else: prev.setdefault((w, n), {})[sec] = ate
    print("==", f.split("/")[-1])
    for c, e in R.items(): print(f"  {c:12s} reach by sector", " ".join(f"{e[s,0]/max(e[s,1],1):.2f}" for s in range(8)), " n", e[:, 1].tolist())
    if H: print("  head (later) by previous same-sector bout:", {k: f"{v[0]/v[1]:.3f} (n{v[1]})" for k, v in H.items()})
for f in sys.argv[1:]: run(f, oracle="/o_" in f)
