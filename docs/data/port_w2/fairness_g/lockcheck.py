"""Registered late-lock classes (protocol, Nov 17): base = the 3e7 verdict, tgt = the later one (decide.py JSON).
LOCK = >= 1 diagonal < .30 at d = 3 or d = 12, OR the diagonal mean at d = 12 drops >= .20 from base;
KEEP = 0 sectors < .30, diagonal mean at d = 12 >= base - .10, min d = 3 >= .85; else PARTIAL."""
import json, sys
import numpy as np
def classify(b, t):
    D = [1, 3, 5, 7]; dead = [k for k in range(8) if t["d3"][k] < 0.30 or t["d12"][k] < 0.30]
    mb = float(np.mean([b["d12"][k] for k in D])); mt = float(np.mean([t["d12"][k] for k in D]))
    lock = any(k in D for k in dead) or mt <= mb - 0.20
    keep = not dead and mt >= mb - 0.10 and min(t["d3"]) >= 0.85
    return dict(cls="LOCK" if lock else "KEEP" if keep else "PARTIAL", dead=dead, diag_d12_base=round(mb, 4), diag_d12_now=round(mt, 4), min_d3=min(t["d3"]))
if __name__ == "__main__":
    print(json.dumps(classify(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))))
