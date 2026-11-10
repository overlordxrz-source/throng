"""W2 navigators (Cam Nov 9 ~17:50): independent seeds, learned in W2 at k = 0, no comm. They are the starting
weights of every W2 newborn. Gate (as N1): each navigator's frozen-policy lifespan >= 0.7x the best one's.
usage: python w2/navigators.py [--n 8] [--updates 400] [--out w2_out/nav]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import flax.serialization

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env, learn  # noqa: E402
from w2.worlds import world_kw  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--out", default="w2_out/nav")
    ap.add_argument("--only", default="", help="comma-separated navigator indices to train (for parallel runs)")
    ap.add_argument("--world", default="proto", help='"proto" or "d0,P,relocate" (non-proto: trained as retune.py trains its blind learner, juvenile 500)')
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = []
    todo = [int(x) for x in a.only.split(",")] if a.only else range(a.n)
    for i in todo:
        if a.world == "proto":
            p, net = learn.train(False, a.updates, seed=1000 + i)              # blind (no oracle), k = 0 (own reward)
            life, se = learn.eval_lifespan(p, net, False)
        else:
            wc = env.Cfg(L=96, N=64, patch_sigma=1.0, reproduce=False, A_max=100_000, **world_kw(a.world))
            p, net = learn.train(False, a.updates, seed=1000 + i, juvenile=500, cfg=wc)
            # Cam Nov 10 (pick confirmation): the retune evaluation (setpoint reproduction, parent-paid provisioning J 500,
            # 16 worlds x 64 slots, 32 founders, 6,000 steps); KM median lifespan of the lives born in the eval
            from w2 import retune
            ecfg = env.Cfg(L=96, N=64, patch_sigma=1.0, reproduce=True, rep_mode="setpoint", juvenile=500, A_max=100_000, **world_kw(a.world))
            ev = retune.evaluate(ecfg, p, net, False, "policy")["offspring_lives"]
            life, se = ev["km_median_life"], None
            Path(f"{a.out}/eval_s{1000 + i}.json").write_text(json.dumps(ev))
        Path(f"{a.out}/w2nav_s{1000 + i}.msgpack").write_bytes(flax.serialization.to_bytes(p))
        res.append(dict(seed=1000 + i, lifespan=life, se=se)); print(res[-1], flush=True)
    if a.only:
        sys.exit(0)
    best = max(r["lifespan"] for r in res)
    for r in res:
        r["gate"] = r["lifespan"] >= 0.7 * best
    json.dump(dict(best=best, navigators=res, n_pass=sum(r["gate"] for r in res)), open(f"{a.out}/gate.json", "w"), indent=1)
    print(f"gate (>= 0.7 x best {best:.1f}): {sum(r['gate'] for r in res)}/{len(res)} pass")
