"""Cam Nov 17 ~14:10Z D1 outcome. Inputs: verdict JSONs (decide.py) for seeds 0 and 1 at 1e7, 2e7, 3e7, in that order.
EVENT (counted on the d = 3 tables, as the controls' 3 events were): a sector drops >= .30 between consecutive tables, or
falls < .30 after having been >= .60 in an earlier table. DEAD = decide.py dead_lt030 at 3e7 (d = 3 or 12 < .30).
(a) 0 events and both qualify at 3e7; (b) <= 1 event and a seed does not qualify; (c) >= 2 events or a dead sector;
otherwise 'unregistered' (1 event, both qualify): report. Events on d = 12 are listed for information."""
import json, sys
V = [json.load(open(f)) for f in sys.argv[1:7]]
def events(tabs, key):
    ev = []
    for s in range(8):
        r = [t[key][s] for t in tabs]
        for i in range(1, len(r)):
            if r[i - 1] - r[i] >= 0.30 or (r[i] < 0.30 and max(r[:i]) >= 0.60):
                ev.append(dict(sector=s, between=[i, i + 1], vals=[r[i - 1], r[i]]))
    return ev
out = {}
for sd, tabs in ((0, V[0:3]), (1, V[3:6])):
    out[f"s{sd}"] = dict(events_d3=events(tabs, "d3"), events_d12=events(tabs, "d12"), dead=tabs[-1]["dead_lt030"], qualifies=tabs[-1]["QUALIFIES"],
                         d3=[t["d3"] for t in tabs], d12=[t["d12"] for t in tabs])
ne = sum(len(out[k]["events_d3"]) for k in ("s0", "s1")); dead = any(out[k]["dead"] for k in ("s0", "s1")); both = all(out[k]["qualifies"] for k in ("s0", "s1"))
cls = "c" if (ne >= 2 or dead) else "a" if (ne == 0 and both) else "b" if (ne <= 1 and not both) else "unregistered"
out.update(n_events_d3=ne, any_dead=dead, both_qualify=both, outcome=cls)
print(json.dumps(out))
