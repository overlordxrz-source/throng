"""D1/D2 on the BAR's own population (bar3: hungry bouts, closed & ~dead & ~sat, closing at age >= A_max/2), from a long
(24,000-step) mode-0 traj with --mem-audit: share and reach of first (demonstrated) hearings, and of later hearings split
by memory state (a solved record for the word / failed records only) x head right / wrong x orbit.
Usage: bar_pop.py FILE_m0.npz EVAL_LOG (the EVAL_MODE 0 line gives hungry_reach_2nd_half for the control)."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1/d1"); import d1_anatomy as D  # noqa: E402
import traj_analysis as TA  # noqa: E402
z = dict(np.load(sys.argv[1])); age = D.ages(z); T = {}
for b in TA.bouts(z):
    w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
    if not (z["hung"][w, s1, n] and age[w, s1, n] >= D.HALF):
        continue
    sec = int(z["sec"][w, s0, n]); orb = "diag" if sec % 2 else "card"; ate = bool(z["cl_item"][w, s1, n])
    if z["first"][w, s0, n]:
        k = "first_dem"
    elif z["dem"][w, s0, n]:                                      # a demonstrated REPEAT hearing (only with --demo-hearings > 1)
        k = "repeat_dem"
    else:
        st = "solved" if z["ma_ns"][w, s0, n].sum() > 0 else "failed_only"
        k = f"later_{st}_{'right' if int(z['hl'][w, s0, n]) == sec else 'wrong'}_{orb}"
    e = T.setdefault(k, [0, 0]); e[0] += ate; e[1] += 1
N = sum(v[1] for v in T.values()); A = sum(v[0] for v in T.values())
J = [json.loads(l.split(" ", 2)[2]) for l in open(sys.argv[2]) if l.startswith("EVAL_MODE 0 ")][0]
print(json.dumps(dict(file=sys.argv[1].split("/")[-1], bouts=N, reach=round(A / N, 4), CTL_eval_hungry_reach_2nd_half=round(J["hungry_reach_2nd_half"], 4),
                      CTL_eval_bouts=J["hungry_bouts_2nd_half"],
                      split={k: dict(n=n, share=round(n / N, 4), reach=round(a / n, 3), lost=round((n - a) / N, 4)) for k, (a, n) in sorted(T.items())})))
