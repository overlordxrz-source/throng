"""Registered W2 learner bar v2 (Cam Nov 15 ~13:30): at D = 12 hidden, 2nd-half HUNGRY-bout reach >= 0.9 x the new oracle's,
>= shuffled + 0.30, >= 2x shuffled; mean life >= 1.3x the new twin's, bootstrap over worlds LB > 1.0x."""
import json, sys, numpy as np
seed_eval, o3, tw = (json.load(open(f)) for f in sys.argv[1:4])
m = seed_eval["modes"]; t, sh, wp = (m[k]["hungry_reach_2nd_half"] for k in ("0", "3", "4"))
ref = o3["modes"]["0"]["hungry_reach_2nd_half"]; tl = tw["modes"]["0"]
c1 = t >= 0.9 * ref; c2 = t >= sh + 0.30; c3 = t >= 2 * sh
ps, pt = np.array(m["0"]["per_world"]), np.array(tl["per_world"])
life = lambda P, ix: P[ix, 0].sum() / P[ix, 1].sum()  # noqa: E731
W = len(ps); ix0 = np.arange(W); ratio = life(ps, ix0) / life(pt, ix0); rb = np.random.default_rng(17)
lb = float(np.percentile([life(ps, ix) / life(pt, ix) for ix in (rb.integers(0, W, W) for _ in range(2000))], 2.5))
c4 = bool(ratio >= 1.3 and lb > 1.0)
out = dict(hungry_2nd_true=t, hungry_2nd_shuffled=sh, hungry_2nd_wiped=wp, oracle_ref=ref, need=0.9 * ref, c_reach=bool(c1), c_shuf30=bool(c2), c_2x=bool(c3),
           life=life(ps, ix0), twin_life=life(pt, ix0), life_ratio=float(ratio), life_lb=lb, c_life=c4,
           head_first=m["0"]["head_acc_first"], head_later=m["0"]["head_acc_later"], PASS=bool(c1 and c2 and c3 and c4))
print(json.dumps(out, indent=1))
