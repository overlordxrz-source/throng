"""bar2.py with the eval-power fix (Nov 16): the REACH criteria use the lengthened eval (>= 2000 2nd-half hungry bouts, true and
shuffled; Cam 08:25Z / 10:15Z), the LIFE criterion uses the default-length eval (3000 steps = A_max, as registered and as the twin's
reference was made). Reason: the evaluator's life count assumes an old-age death can only fall on the final step, which is false
when eval_steps > A_max. Same thresholds as bar2.py.
usage: bar3.py learner_eval_default.json learner_eval_long.json(or the same file) oracle_ref.json twin_eval.json"""
import json, sys, numpy as np
sd, sl, o3, tw = (json.load(open(f)) for f in sys.argv[1:5])
m = sl["modes"]; t, sh, wp = (m[k]["hungry_reach_2nd_half"] for k in ("0", "3", "4")); nb = {k: m[k]["hungry_bouts_2nd_half"] for k in ("0", "3", "4")}
ref = o3["modes"]["0"]["hungry_reach_2nd_half"]; c1 = t >= 0.9 * ref; c2 = t >= sh + 0.30; c3 = t >= 2 * sh
ps, pt = np.array(sd["modes"]["0"]["per_world"]), np.array(tw["modes"]["0"]["per_world"])
life = lambda P, ix: P[ix, 0].sum() / P[ix, 1].sum()  # noqa: E731
W = len(ps); ix0 = np.arange(W); ratio = life(ps, ix0) / life(pt, ix0); rb = np.random.default_rng(17)
lb = float(np.percentile([life(ps, ix) / life(pt, ix) for ix in (rb.integers(0, W, W) for _ in range(2000))], 2.5))
c4 = bool(ratio >= 1.3 and lb > 1.0)
out = dict(hungry_2nd_true=t, hungry_2nd_shuffled=sh, hungry_2nd_wiped=wp, bouts_2nd_half=nb, reach_eval_steps=sl["cfg"]["eval_steps"], oracle_ref=ref,
           oracle_ref_bouts=o3["modes"]["0"]["hungry_bouts_2nd_half"], need=0.9 * ref, c_reach=bool(c1), c_shuf30=bool(c2), c_2x=bool(c3),
           life=life(ps, ix0), twin_life=life(pt, ix0), life_ratio=float(ratio), life_lb=lb, c_life=c4, life_eval_steps=sd["cfg"]["eval_steps"],
           head_first=sd["modes"]["0"]["head_acc_first"], head_later=sd["modes"]["0"]["head_acc_later"], PASS=bool(c1 and c2 and c3 and c4))
print(json.dumps(out, indent=1))
