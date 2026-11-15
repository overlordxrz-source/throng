"""T3 round 2 table: per run (learner, arm, seed): max d; at d = 12 and at its own max d: true 1st->2nd half, within-life
shuffled 2nd half, wiped 2nd half; and the head's first-step accuracy on the 1st vs 2nd+ hearing of a token (true code).
usage: python summarize_t3r2.py DIR"""
import glob, json, os, re, sys
D = sys.argv[1]
print("| run | max d | eval d | true 1st->2nd | shuffled 2nd | wiped 2nd | head acc 1st / 2nd+ hearing | bar |"); print("|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(D, "*_train.json"))):
    tag = os.path.basename(f)[:-11]; dm = json.load(open(f))["d_max_reached"]
    for g in sorted(glob.glob(os.path.join(D, f"{tag}_eval_d*.json")), key=lambda z: int(re.findall(r"_d(\d+)", z)[-1])):
        e = json.load(open(g))["eval_modes"]; dd = re.findall(r"_d(\d+)", g)[-1]
        t, sh, wp = e["0"], e["3"], e["4"]; h = t.get("head_acc_by_hearing") or {}
        s2, sh2 = t["second_half"], sh["second_half"]
        bar = "PASS" if (s2 >= 0.9 and s2 >= sh2 + 0.30 and s2 >= 2 * sh2) else "-"
        hf = h.get("first", [None])[0]; hl = h.get("later", [None])[0]
        print(f"| {tag} | {dm} | {dd} | {round(100*t['first_half'])}->{round(100*s2)} | {round(100*sh2)} | {round(100*wp['second_half'])} | "
              f"{'-' if hf is None else round(100*hf)} / {'-' if hl is None else round(100*hl)} | {bar if dd == '12' else ''} |")
