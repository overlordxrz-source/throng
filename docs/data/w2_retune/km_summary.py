"""W2 KM re-read + sweep (c) (Cam Nov 10 ~05:15; registered before data, RESEARCH_PROTOCOL.md). Kaplan-Meier medians
(right-censored at the horizon): random over all lives; blind / oracle over lives born in the eval.
Gates: P1m blind >= 3x random; P2 oracle >= 2x blind; P4 blind >= THR (1,100 for sweep (c); the re-read also shows
1,000, the old bar). A KM median whose survival never reaches 0.5 is reported '>' and counts as passing P4.
Candidates: sweep (c) only (d0 0.0012 / 0.0013, P8, rel 0.001). The old pick and runner-up are the re-read: reported, not
picked from. Pick: all three on the 4-seed median AND in >= 3/4 seeds; the largest P2 margin among passers."""
import json, os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CF = [("old pick (re-read, reference)", "d0.0015_P8_rel0.001", False), ("runner-up (re-read)", "d0.0018_P10_rel0.0008", False),
      ("sweep (c) d0 0.0012", "d0.0012_P8_rel0.001", True), ("sweep (c) d0 0.0013", "d0.0013_P8_rel0.001", True)]
def vals(p):
    R = json.load(open(p))["results"]; r, b, o = R["random"]["all_lives"], R["blind"]["offspring_lives"], R["oracle"]["offspring_lives"]
    return (r["km_median_life"], b["km_median_life"], o["km_median_life"], b["km_reached"], o["km_reached"],
            b["median_life"], o["median_life"], b["lives"], o["lives"], b["censored"], o["censored"])
def gate(r, b, o, thr): return (b >= 3 * r, o >= 2 * b, b >= thr)
print(__doc__)
passers = []
for name, c, cand in CF:
    files = [f"k_{c}_s{s}.json" for s in range(4) if os.path.exists(f"k_{c}_s{s}.json")]
    print(f"== {name} {c}: {len(files)} seeds")
    V = []; npass = {1000: 0, 1100: 0}
    for f in files:
        r, b, o, br, orr, bn, on, nb, no, cb, co = vals(f); V.append((r, b, o))
        for thr in npass: npass[thr] += all(gate(r, b, o, thr))
        g = gate(r, b, o, 1100)
        print(f"  {f:32s} KM random {r:4.0f} | blind {'>' if not br else ' '}{b:5.0f} (naive {bn:5.0f}; n {nb}, cens {cb}) | oracle "
              f"{'>' if not orr else ' '}{o:5.0f} (naive {on:5.0f}; n {no}, cens {co}) | P1m {g[0]!s:5s} P2 {g[1]!s:5s} ({o/b:.2f}x) P4@1100 {g[2]!s:5s}")
    if len(files) < 4:
        print("  incomplete"); continue
    m = np.median(np.array(V), 0)
    for thr in (1000, 1100):
        gm = gate(*m, thr); ok = all(gm) and npass[thr] >= 3
        print(f"  4-seed median: random {m[0]:.0f} | blind {m[1]:.0f} | oracle {m[2]:.0f} | P1m {gm[0]} P2 {gm[1]} ({m[2]/m[1]:.2f}x) "
              f"P4@{thr} {gm[2]}; seeds passing all three {npass[thr]}/4 -> {'passes' if ok else 'fails'} at P4 >= {thr}")
        if thr == 1100 and ok and cand:
            passers.append((m[2] / m[1], name, c))
print("\n== PICK (sweep (c) candidates, P4 >= 1,100):", f"{max(passers)[1]} ({max(passers)[0]:.2f}x)" if passers else "NONE -> report")
