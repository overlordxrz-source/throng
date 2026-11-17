"""Cam Nov 17 ~14:10Z D1 check: Cfg.d_move = 0 changes the energy of MOVE steps only (stay / eat / every non-move
quantity identical), for 4-way and 8-way; with the world's d_move a move costs d0 + 0.003, a stay d0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from w2 import env
from w2.worlds import world_kw
L = 96
for m8 in (False, True):
    na = 10 if m8 else 6
    base = env.Cfg(L=L, N=na, patch_sigma=1.0, reproduce=False, A_max=3000, moves8=m8, **world_kw("0.0013,8,0.001"))
    zero = base._replace(d_move=0.0)
    s = env.reset(base, jax.random.PRNGKey(0)); s = s._replace(pos=jnp.full((na, 2), 50, jnp.int32), energy=jnp.full(na, 0.6))
    acts = jnp.arange(na)
    a, _, ia = env.step(base, s, acts, jax.random.PRNGKey(1)); b, _, ib = env.step(zero, s, acts, jax.random.PRNGKey(1))
    moved = np.asarray(ia["moved"]); dE = np.asarray(a.energy) - np.asarray(b.energy)
    assert (np.asarray(a.pos) == np.asarray(b.pos)).all() and (np.asarray(a.food) == np.asarray(b.food)).all()
    assert np.allclose(dE[moved], -0.003) and np.allclose(dE[~moved], 0.0), dE
    cost0 = 0.6 - np.asarray(b.energy); assert np.allclose(cost0[moved], cost0[~moved][0]), cost0
    print(f"{'8-way' if m8 else '4-way'}: moved actions {np.flatnonzero(moved).tolist()}; d_move 0 removes exactly .003 on them, 0 elsewhere; "
          f"world: move {0.6 - float(np.asarray(a.energy)[moved][0]):.4f} vs stay {0.6 - float(np.asarray(a.energy)[0]):.4f}; d_move 0: all {float(cost0[0]):.4f}")
print("ALL PASS")
