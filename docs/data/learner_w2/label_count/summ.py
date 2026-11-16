import json, glob, sys, os
O = sys.argv[1]; rows = []
for f in sorted(glob.glob(O + "/s1_*.json")) + sorted(glob.glob(O + "/pilot_*.json")) + sys.argv[2:]:
    d = json.load(open(f)); b = d["by_d"]; H = sum(b["hear"]); S3 = lambda k: sum(b[k][3:])  # noqa: E731
    per = lambda n: 1e5 * n / max(H, 1)  # noqa: E731
    rows.append(dict(run=os.path.basename(f)[:-5], steps=d["steps_at_ckpt"], frontier=d["frontier"], hearings=H, hear_per_1e6_steps=round(1e6 * H / d["rollout_steps"]),
                     frac_hear_d3plus=round(sum(b["hear"][3:]) / max(H, 1), 3), later_d3plus_unseen_per100k=round(per(S3("later_unseen"))),
                     own_labels_per100k=round(per(sum(b["own"]))), own_label_frac_of_hearings=round(sum(b["own"]) / max(H, 1), 3),
                     MEMONLY_own_per100k=round(per(S3("own_later_unseen"))), MEMONLY_privileged_per100k=round(per(S3("later_unseen"))),
                     own_rate_later_d3plus=round(S3("own_later_unseen") / max(S3("later_unseen"), 1), 3),
                     own_by_d={k + 1: v for k, v in enumerate(b["own"][1:13]) if v}, later_unseen_by_d={k + 1: v for k, v in enumerate(b["later_unseen"][1:13]) if v},
                     own_later_unseen_by_d={k + 1: v for k, v in enumerate(b["own_later_unseen"][1:13]) if v}))
json.dump(rows, open(O + "/summary.json", "w"), indent=1)
print("| run | steps | frontier | hearings /1e6 steps | share of hearings at d>=3 | own-meal labels /100k hearings | MEMORY-ONLY own-meal /100k | MEMORY-ONLY privileged /100k | own-meal rate on memory-only bouts |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['run']} | {r['steps']/1e6:.1f}M | {r['frontier']:.0f} | {r['hear_per_1e6_steps']:,} | {r['frac_hear_d3plus']} | {r['own_labels_per100k']:,} | {r['MEMONLY_own_per100k']:,} | {r['MEMONLY_privileged_per100k']:,} | {r['own_rate_later_d3plus']} |")
