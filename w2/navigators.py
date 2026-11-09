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
from w2 import learn  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--out", default="w2_out/nav")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = []
    for i in range(a.n):
        p, net = learn.train(False, a.updates, seed=1000 + i)                  # blind (no oracle), k = 0 (own reward)
        life, se = learn.eval_lifespan(p, net, False)
        Path(f"{a.out}/w2nav_s{1000 + i}.msgpack").write_bytes(flax.serialization.to_bytes(p))
        res.append(dict(seed=1000 + i, lifespan=life, se=se)); print(res[-1], flush=True)
    best = max(r["lifespan"] for r in res)
    for r in res:
        r["gate"] = r["lifespan"] >= 0.7 * best
    json.dump(dict(best=best, navigators=res, n_pass=sum(r["gate"] for r in res)), open(f"{a.out}/gate.json", "w"), indent=1)
    print(f"gate (>= 0.7 x best {best:.1f}): {sum(r['gate'] for r in res)}/{len(res)} pass")
