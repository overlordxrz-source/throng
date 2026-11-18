"""--d4-side for the ep learner (w2/d4.py EpAgentDevD4): equivariance over all 8 g, random params / inputs / memory
records, through a recurrent sequence: with the trunk logits / value zeroed, lg(g . x)[pi_g(a)] == lg(x)[a],
v(g . x) == v(x), hl(g . x)[sigma_g(k)] == hl(x)[k] (the memory records turned by g too); max err < 1e-5; a MUTANT is
caught; the kv / step_kv path equals __call__; shared_tok on and off; 4-way and 8-way."""
import sys, os
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import d4 as D  # noqa: E402
ok_all = True


def rep(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok); print(("PASS " if ok else "FAIL ") + name, info)


def make(n_act, shared, mutant, key, S=16):
    net = D.EpAgentDevD4(S=S, shared_tok=shared, n_act=n_act, mutant=mutant); Dx = 126 + (4 if n_act == 10 else 0); Hd = 8 * 128 + S * 11 + 1
    p = net.init(key, jnp.zeros((Hd,)), jnp.zeros((Dx,)))
    l, t = jax.tree_util.tree_flatten(p); ks = jax.random.split(jax.random.fold_in(key, 7), len(l))
    p = jax.tree_util.tree_unflatten(t, [jax.random.normal(k, x.shape) * 0.5 for k, x in zip(ks, l)])
    pr = dict(p["params"]); pr["Dense_2"] = jax.tree_util.tree_map(jnp.zeros_like, pr["Dense_2"]); pr["Dense_3"] = jax.tree_util.tree_map(jnp.zeros_like, pr["Dense_3"])
    return net, {"params": pr}, Dx, Hd, S


def run(net, p, h0, xs):
    h = h0; L, V, HL = [], [], []
    for x in xs:
        h, lg, v, (hl, _) = net.apply(p, h, x); L.append(lg); V.append(v); HL.append(hl)
    return jnp.stack(L), jnp.stack(V), jnp.stack(HL)


def err(n_act, shared, mutant, seed=0, T=5):
    key = jax.random.PRNGKey(seed); net, p, Dx, Hd, S = make(n_act, shared, mutant, key)
    P, Sg, PI, SIG = D.tables(n_act); PR, SR = D.rec_tables()
    xs = jax.random.normal(jax.random.fold_in(key, 3), (T, Dx))
    mem = jax.random.normal(jax.random.fold_in(key, 4), (S, 11)); mem = mem.at[:, :8].set(jax.nn.one_hot(jax.random.randint(jax.random.fold_in(key, 5), (S,), 0, 8), 8))
    mem = mem.at[S - 3:].set(0.0)                                           # 3 empty slots
    h0 = jnp.concatenate([jnp.zeros(8 * 128), mem.ravel(), jnp.zeros(1)])
    L0, V0, H0 = run(net, p, h0, xs); e = 0.0
    for g in range(8):
        h0g = jnp.concatenate([jnp.zeros(8 * 128), (mem[:, PR[g]] * SR[g]).ravel(), jnp.zeros(1)])
        Lg, Vg, Hg = run(net, p, h0g, xs[:, P[g]] * Sg[g])
        e = max(e, float(jnp.abs(Lg[:, PI[g]] - L0).max()), float(jnp.abs(Vg - V0).max()), float(jnp.abs(Hg[:, SIG[g]] - H0).max()))
    # kv / step_kv == __call__
    K, Vv = net.apply(p, mem, method=D.EpAgentDevD4.kv); filled = mem[:, :8].sum(-1) > 0
    _, lgk, vk, hlk = net.apply(p, jnp.zeros(8 * 128), xs[0], K, Vv, filled, method=D.EpAgentDevD4.step_kv)
    kvd = max(float(jnp.abs(lgk - L0[0]).max()), float(jnp.abs(vk - V0[0]).max()), float(jnp.abs(hlk - H0[0]).max()))
    return e, kvd, float(jnp.abs(L0).max())


for n_act in (6, 10):
    for shared in (True, False):
        e, kvd, m = err(n_act, shared, False); rep(f"learner equivariance (lg, v, hl), 8 g, T = 5, memory turned (n_act {n_act}, shared_tok {shared})", e < 1e-5, f"max err {e:.2e} (max |logit| {m:.2f})")
        rep(f"kv / step_kv == __call__ (n_act {n_act}, shared_tok {shared})", kvd < 1e-6, f"{kvd:.1e}")
    em, _, _ = err(n_act, True, True); rep(f"learner MUTANT caught (n_act {n_act})", em > 1e-3, f"mutant err {em:.2e}")
PR, SR = D.rec_tables(); sem = all(np.allclose((np.r_[np.zeros(8), 0.3, -0.6, 1.0][PR[g]] * SR[g])[8:10], M @ np.array([0.3, -0.6])) for g, M in enumerate(D.MATS))
rep("record disp turns with the world (M . disp); token / ate fixed", sem and all((PR[g][:8] == np.arange(8)).all() and PR[g][10] == 10 for g in range(8)))
print("ALL PASS" if ok_all else "SOME FAIL")
