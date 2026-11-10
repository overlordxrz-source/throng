"""W2 pick confirmation (Cam Nov 10 ~04:35). Same basis as summary_v3: random = median of all lives; blind / oracle =
median of lives born in the eval. Gates: P1m blind >= 3x random; P2 oracle >= 2x blind; P4 blind >= 1,000.
A config stands iff all three pass on the MEDIAN across its 4 seeds AND in >= 3/4 seeds."""
import json, glob, os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CF = [("pick", "d0.0015_P8_rel0.001"), ("runner-up", "d0.0018_P10_rel0.0008")]
def vals(p):
    R = json.load(open(p))["results"]
    return R["random"]["all_lives"]["median_life"], R["blind"]["offspring_lives"]["median_life"], R["oracle"]["offspring_lives"]["median_life"], R["blind"]["offspring_lives"]["lives"], R["oracle"]["offspring_lives"]["lives"]
gate = lambda r, b, o: (b >= 3 * r, o >= 2 * b, b >= 1000)
print(__doc__)
verdict = {}
for name, c in CF:
    files = [f"r2_{c}.json"] + sorted(glob.glob(f"c_{c}_s*.json"))
    files = [f for f in files if os.path.exists(f)]
    print(f"== {name} {c}: {len(files)} seeds")
    V = []; npass = 0
    for f in files:
        r, b, o, nb, no = vals(f); g = gate(r, b, o); npass += all(g); V.append((r, b, o))
        print(f"  {f:38s} random {r:4.0f} | blind {b:5.0f} (n {nb}) | oracle {o:5.0f} (n {no}) | P1m {g[0]!s:5s} P2 {g[1]!s:5s} ({o/b:.2f}x) P4 {g[2]!s:5s} -> {'all' if all(g) else 'x'}")
    m = np.median(np.array(V), 0); gm = gate(*m)
    ok = len(files) == 4 and all(gm) and npass >= 3
    verdict[name] = ok
    print(f"  median across seeds: random {m[0]:.0f} | blind {m[1]:.0f} | oracle {m[2]:.0f} | P1m {gm[0]} P2 {gm[1]} ({m[2]/m[1]:.2f}x) P4 {gm[2]}; seeds passing all three {npass}/{len(files)}"
          f" -> {'STANDS' if ok else ('FAILS' if len(files) == 4 else 'incomplete')}")
print("\n== DECISION:", "pick stands" if verdict["pick"] else ("runner-up stands" if verdict["runner-up"] else "neither stands -> report"))
