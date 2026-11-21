"""Cam Nov 21 ~21:25Z fallback (3) --offset-proj checks (function level; the run-level flag-off identity is a separate
script, run only when the laptop is free).
(1) flag-off: named_item.hear is untouched and is the function called when the flag is off (the code diff only adds the
    flag);
(2) within 9 cells (every offset with ||off||_inf <= 9), hear_proj == NI.hear BITWISE (offset columns and token / flag);
(3) D4 equivariance of the new input: proj(M off) == M proj(off) for every g and every |off| <= 47 (bitwise), and the tie's
    input tables (w2/d4.tables) map the proj input of a world to the proj input of the turned world;
(4) the G2b follower (test_offset_wide.reach_table: 4-way, steers by the offset input alone, decoded as -h * 9) reaches
    8/8 sectors at every d = 3..24 on the new input; NEGATIVE control: on the old clipped input it fails diagonals at long d;
(5) MUTANTS caught: (a) an L2 denominator max(9, ||off||_2) breaks (2); (b) an x-only denominator max(9, |dx|) breaks (3)."""
import sys, os, types, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
from w2 import named_item as NI, offset_proj as OP, d4 as D
import w2.test_offset_wide as TW
L = 96; ok_all = True


def rep(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok); print(("PASS " if ok else "FAIL ") + name, info)


here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
diff = subprocess.run(["git", "-C", here, "diff", "comm-toy", "--", "w2/named_item.py"], capture_output=True, text=True).stdout
rep("(1) named_item.hear untouched vs comm-toy", diff == "")


def hear_at(fn, offs):
    B = len(offs); st = np.full((B, 2), 48); pos = (st - np.asarray(offs)) % L     # torus(start - pos) = off
    item = dict(NI.new_item(B), sec=jnp.zeros(B, jnp.int32), start=jnp.array(st, jnp.int32), act=jnp.ones(B, bool))
    return np.asarray(fn(types.SimpleNamespace(pos=jnp.array(pos, jnp.int32)), item, jnp.tile(jnp.arange(8)[None], (B, 1)), L))


g9 = np.array([(i, j) for i in range(-9, 10) for j in range(-9, 10)])
a, b = hear_at(NI.hear, g9), hear_at(OP.hear_proj, g9)
rep("(2) within 9 cells: hear_proj == NI.hear bitwise (all 361 offsets, all 11 columns)", (a == b).all() and (a.view(np.uint32) == b.view(np.uint32)).all())
g47 = np.array([(i, j) for i in range(-47, 48, 2) for j in range(-47, 48, 3)])
P = np.asarray(OP.proj(jnp.array(g47)))
eq = all((np.asarray(OP.proj(jnp.array(g47 @ M.T))) == P @ M.T).all() for M in D.MATS)
rep("(3a) proj(M off) == M proj(off), all 8 g, |off| <= 47 (bitwise)", eq)
Pt, St, _, _ = D.tables(6); x = np.zeros((len(g47), 126)); x[:, 109:111] = P
eq2 = all(np.array_equal((x[:, Pt[g]] * St[g])[:, 109:111], np.asarray(OP.proj(jnp.array(g47 @ D.MATS[g].T)))) for g in range(8))
rep("(3b) the tie's input tables carry the proj input of a world to that of the turned world", eq2)
R, ds = TW.reach_table(OP.hear_proj, 9.0)
rep("(4) G2b follower on the NEW input: 8/8 at every d = 3..24", R.all(), f"reached per sector {R.sum(1).tolist()} of {len(ds)}")
O, _ = TW.reach_table(NI.hear, 9.0); fails = [(int(k), int(ds[j])) for k, j in zip(*np.nonzero(~O))]
rep("(4-neg) on the OLD clipped input it fails diagonals (cardinals all pass)", (not O.all()) and O[[0, 2, 4, 6]].all() and all(k % 2 == 1 for k, _ in fails),
    f"failing (sector, d) n = {len(fails)}; min failing d = {min(d for _, d in fails) if fails else None}; {fails[:10]}")


def mut_l2(s, item, code, L):
    N = s.pos.shape[0]; tok = code[jnp.arange(N), item["sec"]]; off = NI.torus(item["start"] - s.pos, L).astype(jnp.float32)
    p = off / jnp.maximum(9.0, jnp.sqrt((off ** 2).sum(-1, keepdims=True)))
    return jnp.concatenate([jax.nn.one_hot(tok, 8), jnp.ones((N, 1)), p], 1) * item["act"][:, None]


rep("(5a) MUTANT L2 denominator caught by (2)", not (hear_at(mut_l2, g9) == a).all())
px = lambda o: o.astype(jnp.float32) / jnp.maximum(9.0, jnp.abs(o[..., :1]).astype(jnp.float32))  # noqa: E731
Px = np.asarray(px(jnp.array(g47)))
rep("(5b) MUTANT x-only denominator caught by (3)", not all((np.asarray(px(jnp.array(g47 @ M.T))) == Px @ M.T).all() for M in D.MATS))
print("ALL PASS" if ok_all else "SOME FAIL")
