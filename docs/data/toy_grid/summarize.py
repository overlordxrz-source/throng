"""Toy-grid table: one row per cell (d, visible/hidden, lambda), one column group per learner. Cell = true / shuffled /
silence reach (%), with '*' when it LEARNS (true >= 80% AND >= 2x shuffled AND >= +30 points).
usage: python summarize.py DIR TIER"""
import glob, json, os, sys
D, TIER = sys.argv[1], sys.argv[2]
rows = {}
for f in sorted(glob.glob(os.path.join(D, f"{TIER}_*.json"))):
    d = json.load(open(f)); c = d["cfg"]; e = d["eval"]
    key = (c["d"], "hidden" if c["hidden"] else "visible", c["lam"]); lrn = f"{c['learner']}" + (f" KL{c['kl']}" if c["learner"] == "L1" else "")
    rows.setdefault(key, {})[lrn] = f"{round(100*e['true'])}/{round(100*e['shuffled'])}/{round(100*e['silence'])}" + ("*" if d["learns"] else "")
cols = sorted({k for r in rows.values() for k in r})
print(f"| d | food | lambda | " + " | ".join(cols) + " |"); print("|" + "---|" * (3 + len(cols)))
for k in sorted(rows):
    print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(rows[k].get(c_, "-") for c_ in cols) + " |")
