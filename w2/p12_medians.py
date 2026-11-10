"""Cam Nov 10: re-report W2 P1 / P2 on MEDIANS with juvenile provisioning on (J = 500; no parents in this setting,
so nobody pays). Same learner and budget as w2/learn.py (400 updates, seed 0); lifespan test 16 worlds x 64 agents x
3,000 steps, juvenile on in training and test. P1: learned blind median > random median by > 2 bootstrap SE.
P2: learned oracle median >= 2x learned blind median."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from w2 import learn
J = 500
pb, nb = learn.train(False, 400, 0, juvenile=J); po, no = learn.train(True, 400, 0, juvenile=J)
rnd = learn.eval_lifespan(pb, nb, False, juvenile=J, random_policy=True, stat="median")
bl = learn.eval_lifespan(pb, nb, False, juvenile=J, stat="median")
orc = learn.eval_lifespan(po, no, True, juvenile=J, stat="median")
print(f"MEDIANS with juvenile {J}: random {rnd[0]:.0f} (SE {rnd[1]:.0f}) | learned blind {bl[0]:.0f} (SE {bl[1]:.0f}) | learned oracle {orc[0]:.0f} (SE {orc[1]:.0f})")
print(f"P1 (blind > random by > 2 SE): {'PASS' if bl[0] - rnd[0] > 2 * np.hypot(bl[1], rnd[1]) else 'FAIL'} | "
      f"P2 (oracle >= 2x blind): {orc[0] / bl[0]:.2f}x -> {'PASS' if orc[0] >= 2 * bl[0] else 'FAIL'}")
