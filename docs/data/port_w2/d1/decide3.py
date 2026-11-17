"""Cam Nov 17 ~14:10Z MEASUREMENT RULE: a verdict uses the last three 1e7 tables; reach = the mean of the three; a sector
< .30 in the FINAL table = FAIL whatever the mean. Applies decide.py's thresholds to the per-sector means (frontier from the
final table). Inputs: three verdict JSONs, oldest first."""
import json, sys, numpy as np
V = [json.load(open(f)) for f in sys.argv[1:4]]
m3 = np.mean([v["d3"] for v in V], 0).round(3).tolist(); m12 = np.mean([v["d12"] for v in V], 0).round(3).tolist(); fr = V[-1]["frontier"]
final_dead = V[-1]["dead_lt030"]; dead = sorted(set(i for i in range(8) if m3[i] < 0.30 or m12[i] < 0.30) | set(final_dead))
PASS = min(m3) >= 0.85 and min(m12) >= 0.75 and fr >= 22 and not final_dead
SEC = len(dead) == 0 and min(m3) >= 0.75 and fr >= 22
print(json.dumps(dict(tables=[v["arm"] for v in V], mean_d3=m3, mean_d12=m12, frontier=fr, final_dead=final_dead, dead=dead, PASS=PASS, SECONDARY=SEC, QUALIFIES=PASS or SEC)))
