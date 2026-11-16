import json, glob, os, sys
P = sys.argv[1]; rows = []
print("| ckpt | demo hearings K | lookahead | frontier | hearings | memory-only bouts (later, d>=3) /100k | own-meal (seq-masked, training mask) /100k | own-meal (bout) /100k | + observed STRICT /100k | own + STRICT /100k | + observed LOOSE /100k | own + LOOSE /100k | first-hearing STRICT-only /100k |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(P + "/yield/*.json")):
    d = json.load(open(f)); b = d["by_d"]; H = sum(b["hear"]); g = lambda k: sum(b[k][3:]); p = lambda v: round(1e5 * v / max(H, 1))  # noqa: E731
    nm = os.path.basename(f)[:-5]; S, K, la = nm.split("_")
    r = dict(run=nm, frontier=d["frontier"], H=H, mem=p(g("b_later")), own_seq=p(g("own_later_unseen")), own_b=p(g("b_later_ate")), obs_s=p(g("b_later_obs_s")), obs_l=p(g("b_later_obs_l")),
             first_s=p(sum(b["b_first_obs_s"])), by_d={k: [b[k][i] for i in range(1, 13)] for k in ("b_later", "b_later_ate", "b_later_obs_s", "b_later_obs_l", "own_later_unseen")})
    rows.append(r)
    print(f"| {S} | {K[1:]} | {la[2:]} | {r['frontier']:.0f} | {H:,} | {r['mem']:,} | {r['own_seq']:,} | {r['own_b']:,} | {r['obs_s']:,} | {r['own_b'] + r['obs_s']:,} | {r['obs_l']:,} | {r['own_b'] + r['obs_l']:,} | {r['first_s']:,} |")
json.dump(rows, open(P + "/yield/summary.json", "w"), indent=1)
