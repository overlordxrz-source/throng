"""(e) population: fetch each seed's JSONL + token dumps from the volume and keep the LAST COMPLETE segment (train rows
1..9000) of each JSONL (runs restart from u1 on preemption and the relaunch appends). usage: fetch_dedupe.py <dir> [tags...]"""
import json, os, subprocess, sys
D = sys.argv[1]; os.makedirs(D, exist_ok=True)
TAGS = sys.argv[2:] or [f"pop_on_s{s}" for s in range(320, 325)] + [f"pop_ctl_s{s}" for s in range(325, 328)]
M = "/Users/overlord/.venv/bin/modal"; env = dict(os.environ, MODAL_PROFILE="roadtoagi111")
for t in TAGS:
    base = f"{t}_5c5b07a.jsonl"
    for f in [base] + [f"{base}.u{u}.tok.npz" for u in (1500, 3500, 5500, 7500, 9000)]:
        subprocess.run([M, "volume", "get", "--force", "throng-runs", f"testbed/pop/{f}", D + "/"], env=env, capture_output=True)
    p = f"{D}/{base}"
    if not os.path.exists(p):
        print(t, "MISSING"); continue
    R = []
    for l in open(p):
        try: R.append(json.loads(l))
        except Exception: pass
    segs, cur = [], []
    for r in R:
        if r.get("phase") == "train" and r.get("update") == 1 and any(x.get("phase") == "train" for x in cur):
            segs.append(cur); cur = []
        cur.append(r)
    segs.append(cur)
    mx = [max([x["update"] for x in s if x.get("phase") == "train"] or [0]) for s in segs]
    comp = [i for i, m in enumerate(mx) if m == 9000]
    use = segs[comp[-1]] if comp else segs[-1]
    os.replace(p, p + ".raw")
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in use))
    print(f"{t}: {len(segs)} segments, max updates {mx}; using segment {comp[-1] if comp else len(segs) - 1} ({'complete' if comp else 'INCOMPLETE'})")
