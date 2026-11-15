"""Toy round 2 table: one row per (learner, move cost, distance mode, seed): max d reached by the curriculum, the d = 12
hidden test (true / shuffled reach %), and PASS (>= 80% true, >= 2x shuffled, +30 points). usage: python summarize_r2.py DIR"""
import glob, json, os, sys
print("| learner | move cost | distance | seed | max d | d=12 true | d=12 shuffled | pass | train s |"); print("|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f)); c = d["cfg"]; t = d["test"]
    print(f"| {c['learner']} | {c['move_cost']} | {'curriculum' if c['curriculum'] else 'fixed 12'} | {c['seed']} | {d['d_max_reached']} | "
          f"{round(100*t['true'])} | {round(100*t['shuffled'])} | {'PASS' if d['test_pass'] else '-'} | {d.get('train_seconds')} |")
