"""Cam Nov 18 ~16:15Z --d4-side checks (unit level; flag-off identity and speed are run checks).
(1) group tables: MATS closed, sector / action perms are homomorphisms, grid perm is a bijection;
(2) EQUIVARIANCE over all 8 g, random params and inputs, through a recurrent sequence (T steps from h = 0): with the
    trunk logits / value zeroed, lg(g . x_t)[pi_g(a)] == lg(x_t)[a] and v(g . x_t) == v(x_t); max err < 1e-5;
    4-way and 8-way;
(3) a MUTANT (one wrong action permutation for one g) is caught;
(4) for one-hot oracle inputs (everything else 0), the side-path stay / eat logits are equal across each orbit
    (cardinals 0, 2, 4, 6; diagonals 1, 3, 5, 7)."""
import sys, os
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import d4 as D  # noqa: E402

ok_all = True


def rep(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok); print(("PASS " if ok else "FAIL ") + name, info)


# (1) group tables
idx = lambda M: next(i for i, N in enumerate(D.MATS) if (N == M).all())  # noqa: E731
closed = all(any((N == A @ B).all() for N in D.MATS) for A in D.MATS for B in D.MATS)
rep("D4 closed under composition (8 distinct)", closed and len({tuple(M.ravel()) for M in D.MATS}) == 8)
for n_act in (6, 10):
    P, S, PI, SIG = D.tables(n_act)
    hom = all((PI[idx(D.MATS[g] @ D.MATS[h])] == PI[g][PI[h]]).all() and (SIG[idx(D.MATS[g] @ D.MATS[h])] == SIG[g][SIG[h]]).all()
              for g in range(8) for h in range(8))
    rep(f"action / sector perms are homomorphisms (n_act {n_act})", hom)
    rep(f"stay / eat fixed by every g (n_act {n_act})", (PI[:, 0] == 0).all() and (PI[:, 5] == 5).all())
    rep(f"grid perm bijective, identity at g = 0 (n_act {n_act})", all(len(set(P[g, :49])) == 49 for g in range(8)) and (P[0] == np.arange(P.shape[1])).all() and (S[0] == 1).all())


def make(n_act, mutant, key):
    net = D.RecAgentD4(128, n_act=n_act, mutant=mutant); Dx = 126 + (4 if n_act == 10 else 0)
    p = net.init(key, jnp.zeros((8 * 128,)), jnp.zeros((Dx,)))
    leaves, tree = jax.tree_util.tree_flatten(p); ks = jax.random.split(jax.random.fold_in(key, 7), len(leaves))
    p = jax.tree_util.tree_unflatten(tree, [jax.random.normal(k, l.shape) * 0.5 for k, l in zip(ks, leaves)])   # zero-init heads made non-zero
    pp = jax.tree_util.tree_map(lambda v: v, p); pr = dict(pp["params"])
    pr["Dense_2"] = jax.tree_util.tree_map(jnp.zeros_like, pr["Dense_2"]); pr["Dense_3"] = jax.tree_util.tree_map(jnp.zeros_like, pr["Dense_3"])
    return net, {"params": pr}, Dx


def run_seq(net, p, xs):
    h = jnp.zeros((8 * 128,)); L, V = [], []
    for x in xs:
        h, lg, v, _ = net.apply(p, h, x); L.append(lg); V.append(v)
    return jnp.stack(L), jnp.stack(V)


def equiv_err(n_act, mutant, seed=0, T=6):
    key = jax.random.PRNGKey(seed); net, p, Dx = make(n_act, mutant, key)
    P, S, PI, _ = D.tables(n_act)
    xs = jax.random.normal(jax.random.fold_in(key, 3), (T, Dx))
    L0, V0 = run_seq(net, p, xs); err = 0.0
    for g in range(8):
        xg = xs[:, P[g]] * S[g]
        Lg, Vg = run_seq(net, p, xg)
        err = max(err, float(jnp.abs(Lg[:, PI[g]] - L0).max()), float(jnp.abs(Vg - V0).max()))
    return err, float(jnp.abs(L0).max())


for n_act in (6, 10):
    e, m = equiv_err(n_act, False); rep(f"equivariance, all 8 g, T = 6 recurrent steps (n_act {n_act})", e < 1e-5, f"max err {e:.2e} (max |logit| {m:.2f})")
    e2, _ = equiv_err(n_act, False, seed=11); rep(f"equivariance, second random draw (n_act {n_act})", e2 < 1e-5, f"max err {e2:.2e}")
    em, _ = equiv_err(n_act, True); rep(f"MUTANT (g = 3: actions 1 / 2 swapped) is caught (n_act {n_act})", em > 1e-3, f"mutant err {em:.2e}")

for n_act in (6, 10):
    net, p, Dx = make(n_act, False, jax.random.PRNGKey(5)); SE = {}
    for k in range(8):
        x = jnp.zeros((Dx,)).at[118 + k].set(1.0)
        _, lg, _, _ = net.apply(p, jnp.zeros((8 * 128,)), x); SE[k] = (float(lg[0]), float(lg[5]))
    spread = max(max(abs(SE[k][i] - SE[o[0]][i]) for k in o for i in (0, 1)) for o in ((0, 2, 4, 6), (1, 3, 5, 7)))
    diff = max(abs(SE[0][i] - SE[1][i]) for i in (0, 1))
    rep(f"one-hot oracle: stay / eat equal across each orbit (n_act {n_act})", spread < 1e-5, f"max within-orbit spread {spread:.2e}; cardinal vs diagonal differ by {diff:.3f} (allowed)")

# (5) SEMANTICS: each transform matches rotating the world by M (not just a consistent permutation)
P, S, PI, SIG = D.tables(10); sem = True
for g, M in enumerate(D.MATS):
    for (i, j) in [(0, 0), (1, 5), (6, 2), (3, 4)]:                       # food at window offset v -> at M v after the turn
        x = np.zeros(130); x[i * 7 + j] = 1.0; xg = x[P[g]] * S[g]; v = M @ np.array([i - 3, j - 3])
        sem &= xg[(v[0] + 3) * 7 + (v[1] + 3)] == 1.0 and xg[:49].sum() == 1.0
    for off in [(0.3, -0.7), (-1.0, 0.2)]:
        x = np.zeros(130); x[109:111] = off; xg = x[P[g]] * S[g]; sem &= np.allclose(xg[109:111], M @ np.array(off))
    for a in range(10):
        if D.MOVEV[a] is not None:
            sem &= tuple(M @ np.array(D.MOVEV[a])) == D.MOVEV[PI[g][a]]
    for k in range(8):
        x = np.zeros(130); x[118 + k] = 1.0; xg = x[P[g]] * S[g]; sem &= xg[118 + SIG[g][k]] == 1.0 and tuple(M @ np.array(D.DIRS[k])) == D.DIRS[SIG[g][k]]
rep("semantics: view cell, offset, move and sector each follow the world turned by M", sem)

# (6) POSITIVE CONTROL for (4): the untied RecAgent with random params does NOT have equal stay / eat across an orbit
from w2.learnability import RecAgent  # noqa: E402
net0 = RecAgent(128, n_act=6); p0 = net0.init(jax.random.PRNGKey(5), jnp.zeros((128,)), jnp.zeros((126,)))
l0, t0 = jax.tree_util.tree_flatten(p0); k0 = jax.random.split(jax.random.PRNGKey(9), len(l0))
p0 = jax.tree_util.tree_unflatten(t0, [jax.random.normal(k, l.shape) * 0.5 for k, l in zip(k0, l0)])
SE0 = {k: net0.apply(p0, jnp.zeros((128,)), jnp.zeros((126,)).at[118 + k].set(1.0))[1] for k in range(8)}
sp0 = max(abs(float(SE0[k][i] - SE0[0][i])) for k in (2, 4, 6) for i in (0, 5))
rep("positive control: the UNTIED agent's stay / eat differ across the cardinal orbit", sp0 > 1e-3, f"spread {sp0:.3f}")

print("ALL PASS" if ok_all else "SOME FAIL")
