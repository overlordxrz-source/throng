import sys, numpy as np
sys.path.insert(0, "w2"); import traj_analysis as TA
NAMES = ["stay", "-x", "+x", "+y", "-y", "eat"]
def run(f, oracle=False):
    z = dict(np.load(f)); C = {}
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; g = lambda k, s: z[k][w, s, n]
        sec = int(g("sec", s0)); dem = bool(g("dem", s0)); first = bool(g("first", s0))
        cat = "oracle" if oracle else ("dem" if dem else ("later_right" if (not first and int(g("hl", s0)) == sec) else None))
        if cat is None: continue
        A = z["a"][w, s0:s1 + 1, n]
        c = C.setdefault((cat, sec), np.zeros(6)); c += np.bincount(A, minlength=6)
    # out-of-bout actions too
    inb = z["inb"]; A = z["a"][~inb]; out = np.bincount(A, minlength=6) / len(A)
    print("==", f.split("/")[-1], " outside bouts:", " ".join(f"{NAMES[i]} {out[i]:.2f}" for i in range(6)))
    for (cat, sec), c in sorted(C.items()):
        p = c / c.sum(); print(f"  {cat:12s} sec {sec}:", " ".join(f"{NAMES[i]} {p[i]:.2f}" for i in range(6)))
for f in sys.argv[1:]: run(f, oracle="/o_" in f)
