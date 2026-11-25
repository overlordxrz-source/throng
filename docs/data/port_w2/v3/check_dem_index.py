"""v3 check: `dem` (the parent demonstrates this bout) <=> hearing index <= K for that word (token) in that life, and
`dem` is constant from the hearing step to the close step (the eval counts at the close). From a traj npz made with
--mem-audit (ma_tok = the heard token) and --demo-hearings K. Usage: check_dem_index.py FILE.npz K"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1/d1"); import d1_anatomy as D  # noqa: E402
import traj_analysis as TA  # noqa: E402
z = dict(np.load(sys.argv[1])); K = int(sys.argv[2]); lid = D.lives(z); cnt = {}; T = dict(n=0, agree=0, dem_le=0, dem_gt=0, nodem_le=0, nodem_gt=0, dem_changed_in_bout=0)
for b in sorted(TA.bouts(z), key=lambda b: (b["w"], b["n"], b["s0"])):
    w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; key = (w, n, int(lid[w, s0, n]), int(z["ma_tok"][w, s0, n]))
    cnt[key] = cnt.get(key, 0) + 1; le = cnt[key] <= K; dm = bool(z["dem"][w, s0, n])
    T["n"] += 1; T["agree"] += int(dm == le); T[("dem_" if dm else "nodem_") + ("le" if le else "gt")] += 1
    T["dem_changed_in_bout"] += int(bool(z["dem"][w, s1, n]) != dm)
T["PASS"] = T["agree"] == T["n"] and T["dem_changed_in_bout"] == 0
print(json.dumps(dict(file=sys.argv[1].split("/")[-1], K=K, **T)))
