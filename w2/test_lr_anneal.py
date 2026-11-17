"""G2 unit test of w2/lr_anneal.py (incl. the frontier-triggered start) + a mutant negative control (min for max)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_anneal as LA  # noqa: E402
A, L, L0, LE = 5e7, 5e7, 3e-4, 3e-5
MID = L0 + (LE - L0) * 0.5
CASES = [  # (step, t24, expected)
    (0, None, L0), (9e7, None, L0), (2e8, None, L0),                     # frontier never reached 24: no anneal
    (4.9e7, 2e7, L0), (5e7, 2e7, L0), (7.5e7, 2e7, MID), (1e8, 2e7, LE), (1.3e8, 2e7, LE),     # t24 early -> start 5e7
    (5.9e7, 6e7, L0), (6e7, 6e7, L0), (8.5e7, 6e7, MID), (1.1e8, 6e7, LE), (1.5e8, 6e7, LE),   # t24 late -> start t24
    (5e7, 5e7, L0), (7.5e7, 5e7, MID)]


def check(f):
    bad = [(s, t, e, f(s, t)) for s, t, e in CASES if abs(f(s, t) - e) > 1e-12]
    return bad


def mutant(step, t24, lr0=L0, lr_end=LE, after=A, length=L):
    if t24 is None or step < min(after, t24):
        return lr0
    return lr0 + (lr_end - lr0) * min(1.0, (step - min(after, t24)) / length)


if __name__ == "__main__":
    b = check(lambda s, t: LA.lr_at(s, t, L0, LE, A, L)); print("schedule cases:", "PASS" if not b else f"FAIL {b}")
    cv = [dict(steps=1e6 * i, frontier=min(24, 2 + i)) for i in range(40)]
    t = LA.t24_from_curve(cv); ok_c = t == 22e6 and LA.t24_from_curve(cv[:10]) is None; print("t24_from_curve:", "PASS" if ok_c else f"FAIL {t}")
    bm = check(mutant); ok_m = len(bm) > 0; print("mutant caught:", "PASS" if ok_m else "FAIL (the test cannot see min-for-max)", len(bm))
    print("ALL PASS" if (not b and ok_c and ok_m) else "SOME FAIL")
