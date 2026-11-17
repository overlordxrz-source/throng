"""Cam Nov 17 G3: per-sector head accuracy and mean head confidence (max softmax) at LATER hearings (not demonstrated,
not a token's first hearing), from learner_w2_rl_traj.py records (eval-only, no new rollouts)."""
import sys, json
import numpy as np
out = {}
for f in sys.argv[2:]:
    z = np.load(f); h = z["hear"] & ~z["dem"] & ~z["first"]
    sec = z["sec"][h].astype(int); hl = z["hl"][h].astype(int); hp = z["hp"][h].astype(np.float64)
    row = {}
    for k in range(8):
        m = sec == k; ok = hl[m] == k
        row[k] = dict(n=int(m.sum()), acc=round(float(ok.mean()), 3), conf=round(float(hp[m].mean()), 3),
                      conf_right=round(float(hp[m][ok].mean()), 3) if ok.any() else None, conf_wrong=round(float(hp[m][~ok].mean()), 3) if (~ok).any() else None,
                      wrong_to=np.bincount(hl[m][~ok], minlength=8).tolist())
    c = [row[k] for k in (0, 2, 4, 6)]; dg = [row[k] for k in (1, 3, 5, 7)]
    summ = dict(acc_card=round(np.mean([r["acc"] for r in c]), 3), acc_diag=round(np.mean([r["acc"] for r in dg]), 3),
                conf_card=round(np.mean([r["conf"] for r in c]), 3), conf_diag=round(np.mean([r["conf"] for r in dg]), 3))
    out[f] = dict(by_sector=row, summary=summ)
    print("==", f.split("/")[-1], summ)
    for k in range(8):
        r = row[k]; print(f"  sec {k}: n {r['n']:4d} acc {r['acc']:.3f} conf {r['conf']:.3f} (right {r['conf_right']}, wrong {r['conf_wrong']}) wrong->{r['wrong_to']}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
