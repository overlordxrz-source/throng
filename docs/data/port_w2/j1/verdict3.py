"""Cam Nov 21 ~21:20Z item 3: the learner's 3-TABLE verdict. Inputs: three bar3.py JSONs (the 8e7 / 9e7 / 1e8 checkpoints,
each from its own default + lengthened d = 12 evals vs the J1 reference and twin), then the 1e8 bysec.txt.
Reach criteria on the MEANS over the three: mean true >= .9 x ref; mean true >= mean shuffled + .30; mean true >= 2 x mean
shuffled. Life: the mean life ratio >= 1.3 AND every checkpoint's bootstrap LB > 1.0 (registered reading). Final-table rule:
any sector of the 1e8 later right-head reach at d = 3 or d = 12 < .30 -> FAIL."""
import json, sys, numpy as np
B = [json.load(open(f)) for f in sys.argv[1:4]]; bys = open(sys.argv[4]).read().splitlines()
lr = [l for l in bys if "later_right" in l]
d3, d12 = ([float(v) for v in l.split("sector")[1].split("n [")[0].split()] for l in lr[:2])
mt, ms, mw = (float(np.mean([b[k] for b in B])) for k in ("hungry_2nd_true", "hungry_2nd_shuffled", "hungry_2nd_wiped"))
need = B[0]["need"]; c1 = mt >= need; c2 = mt >= ms + 0.30; c3 = mt >= 2 * ms
rat = float(np.mean([b["life_ratio"] for b in B])); lbs = [b["life_lb"] for b in B]; c4 = rat >= 1.3 and min(lbs) > 1.0
fdead = [s for s in range(8) if d3[s] < 0.30 or d12[s] < 0.30]
print(json.dumps(dict(files=sys.argv[1:4], per_ckpt=[{k: b[k] for k in ("hungry_2nd_true", "hungry_2nd_shuffled", "hungry_2nd_wiped", "life_ratio", "life_lb", "head_later", "bouts_2nd_half")} for b in B],
                      mean_true=mt, mean_shuffled=ms, mean_wiped=mw, need=need, c_reach=bool(c1), c_shuf30=bool(c2), c_2x=bool(c3),
                      mean_life_ratio=rat, life_lbs=lbs, c_life=bool(c4), final_later_right_d3=d3, final_later_right_d12=d12, final_dead=fdead,
                      PASS=bool(c1 and c2 and c3 and c4 and not fdead))))
