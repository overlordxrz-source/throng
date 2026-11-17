"""Cam Nov 17 ~16:40Z refined EVENT RULE (registered; applied equally to old and new tables), on consecutive d = 3 tables:
BROAD dip   = mean d3 drops >= .10, or >= 5 of 8 sectors drop >= .10. Logged, not counted.
SECTOR EVENT= a sector drops >= .3 (or falls < .30 after >= .60) while the mean d3 drops < .10. Counted.
              (literal reading; 'strict' also excludes transitions that are BROAD by the 5-of-8 clause; both reported)
DEAD        = a sector < .30 in a table taken after the frontier reached 24.
Input: JSON {arm: [{"t": label, "frontier": f, "d3": [8]}, ...]} (tables oldest first). Prints one JSON line per arm."""
import json, sys, numpy as np
D = json.load(open(sys.argv[1])); out = {}
for arm, T in D.items():
    broad, ev, ev_strict, dead = [], [], [], []; reached = False
    for i, t in enumerate(T):
        reached = reached or t["frontier"] >= 24
        if reached:
            dead += [(t["t"], s) for s in range(8) if t["d3"][s] < 0.30]
        if i == 0:
            continue
        a, b = np.array(T[i - 1]["d3"]), np.array(t["d3"]); dm = a.mean() - b.mean(); n10 = int(((a - b) >= 0.10).sum())
        isb = dm >= 0.10 or n10 >= 5
        if isb:
            broad.append(dict(at=f'{T[i-1]["t"]}->{t["t"]}', mean_drop=round(float(dm), 3), n_sectors_drop10=n10))
        for s in range(8):
            if (a[s] - b[s]) >= 0.3 or (b[s] < 0.30 and a[s] >= 0.60):
                e = dict(at=f'{T[i-1]["t"]}->{t["t"]}', sector=s, frm=float(a[s]), to=float(b[s]), mean_drop=round(float(dm), 3), n10=n10)
                if dm < 0.10:
                    ev.append(e)
                    if n10 < 5:
                        ev_strict.append(e)
    out[arm] = dict(broad=broad, sector_events=ev, n_events=len(ev), n_events_strict=len(ev_strict), dead=dead, n_dead=len(dead))
    print(json.dumps({arm: out[arm]}))
