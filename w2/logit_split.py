"""Cam Nov 16 ~17:25Z item 2: the hearing-step logit split, frozen vs working sectors (definitions registered in the protocol).

Input: learner_w2_rl_traj.py --logits npz. Bouts: later (non-demonstrated, not a first hearing) right-head bouts for
the learner; every bout for the oracle (--oracle). Sector classes from this file's own reach given a right head:
FROZEN <= 0.10, WORKING >= 0.5. At the hearing step: D_stay = logit(stay) - logit(best toward move) and D_eat the same
for eat, where the toward move is the move within 45 deg of the item sector with the highest TOTAL logit; each D is
split into trunk (current params) + side (total - trunk); the pretrained navigator's D is given for reference.
"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_analysis as TA  # noqa: E402

STAY, EAT = 0, 5


def run(f, oracle):
    z = dict(np.load(f)); rows = []
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
        sec = int(z["sec"][w, s0, n])
        if not oracle and (z["dem"][w, s0, n] or z["first"][w, s0, n] or int(z["hl"][w, s0, n]) != sec):
            continue
        lg, lgn, lgp = (z[k][w, s0, n].astype(np.float64) for k in ("lg", "lgn", "lgp"))
        tw = [m for m in (1, 2, 3, 4) if TA.into(m, sec)]; m = max(tw, key=lambda q: lg[q])
        rows.append(dict(sec=sec, ate=bool(z["cl_item"][w, s1, n]),
                         **{f"{c}_{nm}": float(v[o] - v[m]) for c, v in (("tot", lg), ("trunk", lgn), ("side", lg - lgn), ("pre", lgp))
                            for nm, o in (("stay", STAY), ("eat", EAT))},
                         p_stay=float(np.exp(lg[STAY] - np.logaddexp.reduce(lg))), p_eat=float(np.exp(lg[EAT] - np.logaddexp.reduce(lg))),
                         p_toward=float(sum(np.exp(lg[q] - np.logaddexp.reduce(lg)) for q in tw))))
    reach = {s: np.mean([r["ate"] for r in rows if r["sec"] == s]) for s in range(8) if any(r["sec"] == s for r in rows)}
    cls = {s: ("frozen" if v <= 0.10 else "working" if v >= 0.5 else "mid") for s, v in reach.items()}
    out = dict(file=f, reach_by_sector={s: round(v, 3) for s, v in reach.items()}, classes=cls, by_class={}, by_sector={})
    keys = [k for k in rows[0] if k not in ("sec", "ate")]
    for c in ("frozen", "working", "mid"):
        R = [r for r in rows if cls[r["sec"]] == c]
        if R:
            out["by_class"][c] = dict(n=len(R), **{k: round(float(np.mean([r[k] for r in R])), 3) for k in keys})
    for s in range(8):
        R = [r for r in rows if r["sec"] == s]
        if R:
            out["by_sector"][s] = dict(n=len(R), **{k: round(float(np.mean([r[k] for r in R])), 3) for k in keys})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--oracle-files", nargs="*", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [run(f, False) for f in a.files] + [run(f, True) for f in a.oracle_files]
    json.dump(res, open(a.out, "w"), indent=1, default=float); print("WROTE", a.out)
