"""Free diagnostic (Cam Nov 10): from the existing (e) dumps, per generation, speaker MI per brain and listener reliance
over time. Q1: did ANY newborn (gen >= 2) ever acquire the founders' code? Q2: did the founders' listeners rely at u1500
(a working channel to pass on)? usage: transmission_diag.py <dir of deduped pop_*.jsonl + .tok.npz>"""
import glob, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../throng-toy/scripts"))
from analyze_pop import joint_mi  # noqa: E402

D = sys.argv[1]; US = (1500, 3500, 5500, 7500, 9000)


def cmi(tok, lab, ncls, r, perms=30):
    if len(lab) < 50:
        return float("nan")
    return joint_mi(tok, lab, ncls) - np.mean([joint_mi(tok, r.permutation(lab), ncls) for _ in range(perms)])


def words(tok, lab, ncls):
    """modal 3-slot word per fire"""
    out = {}
    for c in range(ncls):
        m = lab == c
        if m.sum():
            out[c] = Counter(map(tuple, tok[m])).most_common(1)[0][0]
    return out


summary = []
for p in sorted(glob.glob(f"{D}/pop_*.jsonl")):
    R = [json.loads(l) for l in open(p)]; tag, seed = R[0]["tag"], R[0]["seed"]
    births = [r for r in R if r["phase"] == "birth"]
    def gen_age(b, u):
        bb = [x for x in births if x["slot"] == b and x["update"] <= u]
        return (bb[-1]["gen"], u - bb[-1]["update"]) if bb else (1, u)
    sw = {u: [r for r in R if r["phase"] == "pop_swap" and r["update"] == u and "dep_cross" in r] for u in US}
    rel = {u: (1 - sum(r["dep_cross"] for r in sw[u]) / max(sum(r["dep_control"] for r in sw[u]), 1e-9)) if sw[u] else None for u in US}
    r_ = np.random.default_rng(0); founders_w = None; rows = []
    for u in US:
        f = f"{p}.u{u}.tok.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f); tok, lab, br = z["tok"], z["lab"], z["brain"]; ncls = int(lab.max()) + 1
        per = []
        for b in np.unique(br):
            m = br == b; g, age = gen_age(int(b), u)
            w = words(tok[m], lab[m], ncls)
            per.append(dict(brain=int(b), gen=g, age=age, mi=cmi(tok[m], lab[m], ncls, r_), words=w))
        if u == 1500:     # the founders' dictionary: modal word per fire over the gen-1 brains' pooled sender steps
            fm = np.isin(br, [x["brain"] for x in per if x["gen"] == 1])
            founders_w = words(tok[fm], lab[fm], ncls)
        for x in per:
            x["agree_founders"] = (float(np.mean([x["words"].get(c) == founders_w.get(c) for c in founders_w]))
                                   if founders_w is not None else None)
        rows.append((u, per))
    print(f"== {tag} seed {seed}: population cross-spot reliance by update: " +
          ", ".join(f"u{u} {rel[u]:+.0%}" for u in US if rel[u] is not None))
    for u, per in rows:
        print(f"   u{u}: " + " | ".join(f"b{x['brain']} gen{x['gen']} age{x['age']} MI {x['mi']:+.2f} agree {x['agree_founders']:.2f}" for x in per))
    newborn = [x for u, per in rows for x in per if x["gen"] >= 2]
    best = max(newborn, key=lambda x: x["mi"]) if newborn else None
    summary.append(dict(tag=tag, seed=seed, reliance_u1500=rel[1500],
                        founders_mi_u1500=float(np.mean([x["mi"] for u, per in rows if u == 1500 for x in per if x["gen"] == 1])),
                        newborn_max_mi=best["mi"] if best else None,
                        newborn_max_agree=float(max(x["agree_founders"] for x in newborn)) if newborn else None,
                        newborns_with_mi_ge_0_5=int(sum(x["mi"] >= 0.5 for x in newborn)), newborn_obs=len(newborn)))
print("\n== SUMMARY"); [print(json.dumps(s)) for s in summary]
json.dump(summary, open(f"{os.path.dirname(os.path.abspath(__file__))}/transmission_diag_summary.json", "w"), indent=1)
