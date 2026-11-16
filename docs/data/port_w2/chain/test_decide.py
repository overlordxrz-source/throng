"""Cam Nov 16 ~19:15Z: synthetic controls for chain/decide.py (run UNCHANGED, as a subprocess, on fabricated trajectory files).
Each synthetic file holds, per sector, n bouts of 2 steps (hear, then close) with a chosen number eaten."""
import json, os, subprocess, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(D, "tests")

def fake(path, reach, n=200):
    steps = []
    for s, r in enumerate(reach):
        k = int(round(r * n))
        for i in range(n):
            steps.append((s, True, False)); steps.append((s, False, i < k))
    L = len(steps); z = {k: np.zeros((1, L, 1), bool) for k in ("hear", "cl_item", "cl_tmax", "cl_death")}; z["sec"] = np.zeros((1, L, 1), np.int8)
    for t, (s, h, ate) in enumerate(steps):
        z["sec"][0, t, 0] = s; z["hear"][0, t, 0] = h
        if not h:
            z["cl_item"][0, t, 0] = ate; z["cl_tmax"][0, t, 0] = not ate
    np.savez_compressed(path, **z)

def verdict(name, r3, r12, frontier):
    a, b, tr = (os.path.join(T, f"{name}_{x}") for x in ("d3.npz", "d12.npz", "train.json"))
    fake(a, r3); fake(b, r12); json.dump(dict(frontier=frontier), open(tr, "w"))
    out = subprocess.run([sys.executable, os.path.join(D, "decide.py"), name, a, b, tr], capture_output=True, text=True, cwd="/Users/overlord/CascadeProjects/throng-toy")
    return json.loads(out.stdout)

CASES = [  # name, d3, d12, frontier, expected (PASS, SECONDARY, QUALIFIES, dead_lt030)
    ("must_pass", [0.95, 0.9, 0.86, 0.9, 0.95, 0.88, 0.9, 0.97], [0.9, 0.8, 0.76, 0.85, 0.9, 0.8, 0.85, 0.9], 24, (True, True, True, [])),
    ("secondary_only", [0.95, 0.8, 0.9, 0.76, 0.95, 0.9, 0.9, 0.9], [0.9, 0.5, 0.85, 0.35, 0.9, 0.8, 0.85, 0.9], 22, (False, True, True, [])),
    ("one_dead_fails", [0.99, 0.99, 0.99, 0.05, 0.99, 0.99, 0.99, 0.99], [0.95, 0.95, 0.95, 0.02, 0.95, 0.95, 0.95, 0.95], 24, (False, False, False, [3])),
    ("pass_but_frontier_21", [0.95] * 8, [0.9] * 8, 21, (False, False, False, [])),
    ("dead_only_at_d12", [0.9] * 8, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.25], 24, (False, False, False, [7])),
]
ok = 0
for name, r3, r12, fr, exp in CASES:
    v = verdict(name, r3, r12, fr); got = (v["PASS"], v["SECONDARY"], v["QUALIFIES"], v["dead_lt030"])
    good = got == exp; ok += good
    print(("PASS " if good else "FAIL ") + name, "got", got, "expected", exp, "| d3", v["d3"], "d12", v["d12"])
print(f"{ok}/{len(CASES)} cases as expected")
sys.exit(0 if ok == len(CASES) else 1)
