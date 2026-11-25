"""bar3.py under MEASUREMENT RULE v3 (Cam Nov 24 ~23:55Z): every REACH criterion is read on NON-DEMONSTRATED bouts
(hungry, 2nd half of life, d = 12, 24k-step horizon, hearing index > K for that word in that life; the eval's
hungry_reach_2nd_half_nd keys, code throng-toy v3-k2). The reference is read on the same class (its _nd key). The LIFE
criterion is unchanged: the default-length (3000-step) eval vs the twin of the same world (per_world, bootstrap LB).
usage: bar3v3.py learner_eval_default.json learner_eval_long24k.json ref.json twin_eval.json"""
import json, sys, numpy as np
sd, sl, o3, tw = (json.load(open(f)) for f in sys.argv[1:5])
import os  # noqa: E402
assert sl["cfg"]["eval_steps"] == int(os.environ.get("BAR_ES", "24000")), "v3: the reach eval is the 24k horizon (BAR_ES: smoke only)"
m = sl["modes"]; K = "hungry_reach_2nd_half_nd"
t, sh, wp = (m[k][K] for k in ("0", "3", "4")); nb = {k: m[k]["hungry_bouts_2nd_half_nd"] for k in ("0", "3", "4")}
ref = o3["modes"]["0"][K]; c1 = t >= 0.9 * ref; c2 = t >= sh + 0.30; c3 = t >= 2 * sh
ps, pt = np.array(sd["modes"]["0"]["per_world"]), np.array(tw["modes"]["0"]["per_world"])
life = lambda P, ix: P[ix, 0].sum() / P[ix, 1].sum()  # noqa: E731
W = len(ps); ix0 = np.arange(W); ratio = life(ps, ix0) / life(pt, ix0); rb = np.random.default_rng(17)
lb = float(np.percentile([life(ps, ix) / life(pt, ix) for ix in (rb.integers(0, W, W) for _ in range(2000))], 2.5))
c4 = bool(ratio >= 1.3 and lb > 1.0)
out = dict(rule="v3", true_nd=t, shuffled_nd=sh, wiped_nd=wp, bouts_nd=nb, true_nd_bysec=m["0"]["hungry_reach_2nd_half_nd_bysec"],
           bouts_nd_bysec=m["0"]["hungry_bouts_2nd_half_nd_bysec"], all_bouts_true=m["0"]["hungry_reach_2nd_half"], reach_eval_steps=sl["cfg"]["eval_steps"],
           demo_hearings=sl["cfg"].get("demo_hearings"), oracle_ref_nd=ref, oracle_ref_bouts_nd=o3["modes"]["0"].get("hungry_bouts_2nd_half_nd"), need=0.9 * ref,
           c_reach=bool(c1), c_shuf30=bool(c2), c_2x=bool(c3), life=life(ps, ix0), twin_life=life(pt, ix0), life_ratio=float(ratio), life_lb=lb, c_life=c4,
           life_eval_steps=sd["cfg"]["eval_steps"], twin_demo_hearings=tw["cfg"].get("demo_hearings"), twin_demonstrator=tw["cfg"].get("demonstrator"),
           head_first=sd["modes"]["0"]["head_acc_first"], head_later=sd["modes"]["0"]["head_acc_later"], PASS=bool(c1 and c2 and c3 and c4))
print(json.dumps(out, indent=1))
