"""verdict3.py under MEASUREMENT RULE v3 (Cam Nov 24 ~23:55Z). Inputs: three bar3v3.py JSONs (8e7 / 9e7 / 1e8).
Reach criteria on the MEANS over the three, all on non-demonstrated bouts (hungry, 2nd half, d12, 24k): mean true >=
.9 x ref (same class); mean true >= mean shuffled + .30; mean true >= 2 x mean shuffled. Life: mean ratio >= 1.3 AND every
checkpoint's LB > 1.0 (vs the twin of the same world). FINAL-SECTOR rule (registered reading of v3): any sector of the
1e8 table's non-demonstrated d12 reach (all heads) < .30 -> FAIL. Usage: verdict3v3.py bar_8e7.json bar_9e7.json bar_10e7.json"""
import json, sys, numpy as np
B = [json.load(open(f)) for f in sys.argv[1:4]]
assert all(b.get("rule") == "v3" for b in B)
mt, ms, mw = (float(np.mean([b[k] for b in B])) for k in ("true_nd", "shuffled_nd", "wiped_nd"))
need = B[0]["need"]; c1 = mt >= need; c2 = mt >= ms + 0.30; c3 = mt >= 2 * ms
rat = float(np.mean([b["life_ratio"] for b in B])); lbs = [b["life_lb"] for b in B]; c4 = rat >= 1.3 and min(lbs) > 1.0
fs = B[2]["true_nd_bysec"]; fdead = [s for s in range(8) if fs[s] < 0.30]
print(json.dumps(dict(rule="v3", files=sys.argv[1:4], per_ckpt=[{k: b[k] for k in ("true_nd", "shuffled_nd", "wiped_nd", "bouts_nd", "life_ratio", "life_lb", "head_later", "all_bouts_true")} for b in B],
                      mean_true=mt, mean_shuffled=ms, mean_wiped=mw, need=need, c_reach=bool(c1), c_shuf30=bool(c2), c_2x=bool(c3),
                      mean_life_ratio=rat, life_lbs=lbs, c_life=bool(c4), final_nd_bysec_d12=fs, final_dead=fdead,
                      PASS=bool(c1 and c2 and c3 and c4 and not fdead))))
