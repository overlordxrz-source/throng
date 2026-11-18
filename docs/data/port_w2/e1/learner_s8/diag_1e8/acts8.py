"""Per-action counts (0 stay, 1 -x, 2 +x, 3 +y, 4 -y, 5 eat, 6 (+1,+1), 7 (-1,+1), 8 (-1,-1), 9 (+1,-1)) in later right-head
bouts, by item sector, from a traj npz; also over ALL steps of the file."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2"); import traj_analysis as TA  # noqa: E402
for f in sys.argv[1:]:
    z = dict(np.load(f)); C = {s: np.zeros(10, int) for s in range(8)}
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; sec = int(z["sec"][w, s0, n])
        if z["dem"][w, s0, n] or z["first"][w, s0, n] or int(z["hl"][w, s0, n]) != sec:
            continue
        C[sec] += np.bincount(z["a"][w, s0:s1 + 1, n].astype(int), minlength=10)[:10]
    al = np.bincount(z["a"].astype(int).ravel(), minlength=10)[:10]
    print("==", f.split("/")[-1], "ALL steps share:", (al / al.sum()).round(3).tolist())
    for s in (0, 2, 4, 6, 1):
        c = C[s]; print("  sector", s, "share", (c / max(c.sum(), 1)).round(3).tolist(), "n", int(c.sum()))
