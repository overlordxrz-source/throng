"""Unit tests for w2/relabel.py (Cam Nov 16 ~10:15Z check (a)). Run: python w2/test_relabel.py"""
import inspect, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import relabel as RL  # noqa: E402
FAIL = []
S = 8


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAIL.append(name)


def rec(tok, sec, ate, d=5):
    r = np.zeros(11); r[tok] = 1; ang = sec * np.pi / 4; r[8:10] = np.round([d * np.cos(ang), d * np.sin(ang)]) / 24.0; r[10] = float(ate); return r


def seq(T=12):
    return dict(hear=np.zeros((1, T), bool), tok=np.zeros((1, T, 8)), mem0=np.zeros((1, S * 11)), ptr0=np.zeros(1), wflag=np.zeros((1, T), bool),
                wrec=np.zeros((1, T, 11)), dead=np.zeros((1, T), bool))


def run(q):
    return RL.relabel(q["hear"], q["tok"], q["mem0"], q["ptr0"], q["wflag"], q["wrec"], q["dead"], S)


def hearing(q, t, tok):
    q["hear"][0, t] = True; q["tok"][0, t, tok] = 1


def write(q, t, r):
    q["wflag"][0, t] = True; q["wrec"][0, t] = r


# 1. sector_of matches the placement for every sector and distance 1..24
ok = all(RL.sector_of(np.round([d * np.cos(s_ * np.pi / 4), d * np.sin(s_ * np.pi / 4)])) == s_ for s_ in range(8) for d in range(1, 25))
check("sector_of = the placement's sector (8 sectors x d 1..24)", ok)
# 2. masked when X has no solved record (an unsolved record of X, and a solved record of ANOTHER token)
q = seq(); hearing(q, 5, 3); write(q, 2, rec(3, 6, 0)); write(q, 3, rec(4, 1, 1)); t, m, s, _ = run(q)
check("masked when the token has no solved record (unsolved X, solved other token)", not m[0, 5])
# 3. an earlier solved record labels a later hearing; the most recent one wins
q = seq(); write(q, 1, rec(3, 6, 1)); write(q, 4, rec(3, 6, 1)); hearing(q, 7, 3); t, m, s, st = run(q)
check("earlier solved record -> label = its sector (source earlier)", m[0, 7] and t[0, 7] == 6 and s[0, 7] == 1 and st["n_disagree"] == 0)
q = seq(); write(q, 1, rec(3, 2, 1)); write(q, 4, rec(3, 6, 1)); hearing(q, 7, 3); t, m, s, st = run(q)
check("two disagreeing earlier records -> the MOST RECENT, counted as a disagreement", t[0, 7] == 6 and st["n_disagree"] == 1)
# 4. no earlier -> the earliest LATER solved record (incl. this bout's own close)
q = seq(); hearing(q, 2, 5); write(q, 6, rec(5, 1, 1)); write(q, 9, rec(5, 3, 1)); t, m, s, _ = run(q)
check("no earlier -> earliest later solved record (source later)", m[0, 2] and t[0, 2] == 1 and s[0, 2] == 2)
# 5. the death boundary: an earlier record from before a death does not label; a later one after a death does not either
q = seq(); write(q, 1, rec(3, 6, 1)); q["dead"][0, 3] = True; hearing(q, 5, 3); t, m, s, _ = run(q)
check("death boundary (earlier): a record from before a death does not label", not m[0, 5])
q = seq(); hearing(q, 2, 3); q["dead"][0, 4] = True; write(q, 8, rec(3, 6, 1)); t, m, s, _ = run(q)
check("death boundary (later): a record after a death does not label", not m[0, 2])
# 6. the record buffer at the sequence start counts (ring order: most recent = slot ptr - 1)
q = seq(); mem = np.zeros((S, 11)); mem[0] = rec(2, 1, 1); mem[1] = rec(2, 5, 1); q["mem0"][0] = mem.reshape(-1); q["ptr0"][0] = 2; hearing(q, 3, 2)
t, m, s, st = run(q)
check("start-of-sequence buffer: most recent slot (ptr - 1) wins", m[0, 3] and t[0, 3] == 5 and st["n_disagree"] == 1)
q["ptr0"][0] = 1; mem2 = mem.copy(); mem2[1] = 0; mem2[7] = rec(2, 5, 1); q["mem0"][0] = mem2.reshape(-1); t, m, s, _ = run(q)
check("start-of-sequence buffer: ring wrap order (slot 0 newer than slot 7 when ptr = 1)", t[0, 3] == 1)
# 7. a write at step t is available from t + 1 (not to a hearing at the same step); first hearing labelled only by a later record
q = seq(); write(q, 4, rec(3, 6, 1)); hearing(q, 4, 3); t, m, s, _ = run(q)
check("a record written at step t labels a hearing at t only as 'later' (its own bout)", m[0, 4] and s[0, 4] == 2)
# 8. never reads the code table or the item's sector: the signature has neither
sig = list(inspect.signature(RL.relabel).parameters)
check(f"relabel's inputs are only {sig} (no code table, no item sector)", sig == ["hear", "tok_onehot", "mem0", "ptr0", "wflag", "wrec", "dead", "S"])
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"); sys.exit(1 if FAIL else 0)
