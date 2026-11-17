"""Cam Nov 17 ~16:40Z E1 gate (absolute), per seed, from its five verdict JSONs (1e7..5e7, oldest first):
  frontier 24 in the final table; QUALIFIES by decide3 (decide.py thresholds on the mean of the 3e7/4e7/5e7 tables, final-table
  sector < .30 at d3 or d12 = FAIL); 0 DEAD (d3 sector < .30 in any table after the frontier reached 24; d12 < .30 reported too);
  <= 1 SECTOR EVENT (events_v2 literal rule: drop >= .3, or < .30 after >= .60, while the mean d3 drops < .10).
Outcomes: (a) both seeds pass; (c) any DEAD in either seed; (b) sector events, 0 dead, not (a); otherwise 'unregistered'.
Usage: e1_outcome.py s0_1 .. s0_5 s1_1 .. s1_5"""
import json, sys, subprocess, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
F = sys.argv[1:11]; res = {}
for sd, fs in (("s0", F[:5]), ("s1", F[5:])):
    V = [json.load(open(f)) for f in fs]
    T = [dict(t=f"{i+1}e7", frontier=v["frontier"], d3=v["d3"]) for i, v in enumerate(V)]
    tmp = f"/tmp/e1_ev_{os.getpid()}_{sd}.json"; json.dump({sd: T}, open(tmp, "w"))
    ev = json.loads(subprocess.check_output([sys.executable, f"{H}/events_v2.py", tmp]).decode().strip())[sd]
    q = json.loads(subprocess.check_output([sys.executable, f"{H}/../d1/decide3.py", *fs[2:5]]).decode())
    reached = False; d12dead = []
    for v, t in zip(V, T):
        reached = reached or v["frontier"] >= 24
        if reached:
            d12dead += [(t["t"], s) for s in range(8) if v["d12"][s] < 0.30]
    ok = V[-1]["frontier"] >= 24 and q["QUALIFIES"] and ev["n_dead"] == 0 and ev["n_events"] <= 1
    res[sd] = dict(frontier_final=V[-1]["frontier"], frontiers=[v["frontier"] for v in V], rule3=q, events=ev, d12_below030_after24=d12dead, PASS=bool(ok))
anyd = any(res[s]["events"]["n_dead"] > 0 for s in res); nev = sum(res[s]["events"]["n_events"] for s in res)
out = "a" if all(res[s]["PASS"] for s in res) else "c" if anyd else "b" if nev > 0 else "unregistered"
print(json.dumps(dict(outcome=out, n_events=nev, any_dead=anyd, per_seed=res)))
