"""Cam Nov 17 ~18:35Z seed-8 watch: head accuracy at LATER hearings (not demonstrated, not a first hearing) from
learner_w2_rl_traj.py --traj-out npz files: share of later hearing steps where the head's argmax (hl) == the item's sector.
Usage: head_later.py file_d3.npz [file_d12.npz ...]  -> one JSON line."""
import sys, json, numpy as np
out = {}
for f in sys.argv[1:]:
    z = np.load(f); m = z["hear"].astype(bool) & ~z["dem"].astype(bool) & ~z["first"].astype(bool)
    ok = (z["hl"].astype(int) == z["sec"].astype(int)) & m
    out[f.split("/")[-1]] = dict(acc=round(float(ok.sum() / max(m.sum(), 1)), 3), n=int(m.sum()))
print(json.dumps(out))
