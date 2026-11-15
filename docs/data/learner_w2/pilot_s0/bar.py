"""Registered W2 learner bar (Nov 15 ~05:50) for one seed's d=12 eval (modes 0,3,4), vs oracle #2 and the deaf twin in the same evaluator."""
import json, sys, numpy as np
seed_eval, o2, tw = (json.load(open(f)) for f in sys.argv[1:4])
m = seed_eval["modes"]; t, sh, wp = (m[k]["reach_2nd_half"] for k in ("0", "3", "4"))
o2r = o2["modes"]["0"]["reach_2nd_half"]; tl = tw["modes"]["0"]
c1 = t >= 0.9 * o2r; c2 = t >= sh + 0.30; c3 = t >= 2 * sh
pw_s = np.array(m["0"]["per_world"]); pw_t = np.array(tl["per_world"])          # [alive, lives, ate2, closed2, paid]
life = lambda P, ix: P[ix, 0].sum() / P[ix, 1].sum()  # noqa: E731
W = len(pw_s); rb = np.random.default_rng(13); ix0 = np.arange(W)
ratio = life(pw_s, ix0) / life(pw_t, ix0)
boots = [life(pw_s, ix) / life(pw_t, ix) for ix in (rb.integers(0, W, W) for _ in range(2000))]
lb = float(np.percentile(boots, 2.5)); c4 = ratio >= 1.3 and lb > 1.0
out = dict(true_2nd=t, shuffled_2nd=sh, wiped_2nd=wp, oracle2_2nd=o2r, need=0.9 * o2r, c_reach=c1, c_shuf30=c2, c_2x=c3,
           life=life(pw_s, ix0), twin_life=life(pw_t, ix0), life_ratio=ratio, life_lb=lb, c_life=c4,
           head_first=m["0"]["head_acc_first"], head_later=m["0"]["head_acc_later"], PASS=bool(c1 and c2 and c3 and c4))
print(json.dumps(out, indent=1))
