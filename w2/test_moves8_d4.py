"""Cam Nov 23 ~05:10Z item 5 (code only): 8-WAY + D4 (moves8 + --d4-side [+ --offset-proj]). Function checks:
(1) flag-off identity: this branch adds only this test file on top of offset-proj (no trainer change: --moves8 + --d4-side
    already wire n_act = 10 into RecAgentD4 / EpAgentDevD4 and the 8-way x layout into the D4 tables);
(2) equivariance over all 8 g with 10 actions (oracle RecAgentD4 and learner EpAgentDevD4), through recurrent steps;
(3) the STABILISER property Carbon's (2) rests on: for a one-hot oracle sector with every other input D4-invariant (zeros),
    the tied side path gives EQUAL logits to the two moves that the reflection fixing that ray swaps:
    cardinal sector k -> its two neighbouring diagonals; diagonal sector k -> its two neighbouring cardinals.
    POSITIVE CONTROL: the untied RecAgent (10 actions) breaks it;
(4) a G2b-style follower (test_moves8's sector + offset-only follower) reaches 8/8 at every d = 3..24 with 8-way moves on
    the --offset-proj input, and on the old clip input; NEGATIVE control: 4-way on the clip input fails diagonals;
(5) a MUTANT (a wrong 10-action permutation for one g) is caught by (2);
(6) a real 8-way observation (env obs + hear_proj + prev action incl. the diagonal columns + oracle) runs through the tie.
No training runs."""
import sys, os, types, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
from w2 import d4 as D, env, named_item as NI, offset_proj as OP
from w2.learnability import RecAgent
import w2.test_moves8 as TM
ok_all = True


def rep(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok); print(("PASS " if ok else "FAIL ") + name, info)


here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st = subprocess.run(["git", "-C", here, "diff", "--name-only", "offset-proj"], capture_output=True, text=True).stdout.split()
un = subprocess.run(["git", "-C", here, "status", "--porcelain"], capture_output=True, text=True).stdout
rep("(1) only this test file differs from offset-proj (no trainer change)", st == [] and all("test_moves8_d4.py" in l or not l.strip() for l in un.splitlines()), f"diff {st}; untracked/modified {un.strip()!r}")


def rand(net, h0, x0, key):
    p = net.init(key, h0, x0); l, t = jax.tree_util.tree_flatten(p); ks = jax.random.split(jax.random.fold_in(key, 7), len(l))
    p = jax.tree_util.tree_unflatten(t, [jax.random.normal(k, v.shape) * 0.5 for k, v in zip(ks, l)]); pr = dict(p["params"])
    for k in ("Dense_2", "Dense_3"):
        pr[k] = jax.tree_util.tree_map(jnp.zeros_like, pr[k])
    return {"params": pr}


P, S, PI, SIG = D.tables(10)
for mut in (False, True):
    net = D.RecAgentD4(128, n_act=10, mutant=mut); p = rand(net, jnp.zeros(1024), jnp.zeros(130), jax.random.PRNGKey(3))
    xs = jax.random.normal(jax.random.PRNGKey(4), (5, 130)); err = 0.0
    def run(xseq):
        h = jnp.zeros(1024); L = []
        for x in xseq:
            h, lg, v, _ = net.apply(p, h, x); L.append(lg)
        return jnp.stack(L)
    L0 = run(xs)
    for g in range(8):
        err = max(err, float(jnp.abs(run(xs[:, P[g]] * S[g])[:, PI[g]] - L0).max()))
    if mut:
        rep("(5) MUTANT (one wrong 10-action permutation) caught by equivariance", err > 1e-3, f"err {err:.2e}")
    else:
        rep("(2) oracle tie equivariance, all 8 g, 10 actions, 5 recurrent steps", err < 1e-5, f"max err {err:.2e}")
r = subprocess.run([sys.executable, os.path.join(here, "w2", "test_d4_learner.py")], capture_output=True, text=True).stdout
rep("(2) learner tie equivariance incl. 10 actions (test_d4_learner)", "ALL PASS" in r and "n_act 10" in r)

# (3) stabiliser: cardinal k -> neighbouring diagonals equal; diagonal k -> neighbouring cardinals equal
DIAG_OF = {0: (6, 9), 2: (6, 7), 4: (7, 8), 6: (8, 9)}     # sector -> its two neighbouring diagonal actions
CARD_OF = {1: (2, 3), 3: (1, 3), 5: (1, 4), 7: (2, 4)}     # diagonal sector -> its two neighbouring cardinal actions
def stab(net, p, h0):
    m = 0.0
    for k, (a1, a2) in {**DIAG_OF, **CARD_OF}.items():
        _, lg, _, _ = net.apply(p, h0, jnp.zeros(130).at[118 + k].set(1.0)); m = max(m, abs(float(lg[a1] - lg[a2])))
    return m
net = D.RecAgentD4(128, n_act=10); p = rand(net, jnp.zeros(1024), jnp.zeros(130), jax.random.PRNGKey(5))
m_tied = stab(net, p, jnp.zeros(1024))
un = RecAgent(128, n_act=10); pu = rand(un, jnp.zeros(128), jnp.zeros(130), jax.random.PRNGKey(5)); m_un = stab(un, pu, jnp.zeros(128))
rep("(3) stabiliser: the reflection-swapped pair of moves gets EQUAL tied logits (all 8 sectors)", m_tied < 1e-5, f"max |diff| {m_tied:.2e}")
rep("(3) positive control: the UNTIED agent breaks it", m_un > 1e-2, f"max |diff| {m_un:.3f}")
for k, (a1, a2) in DIAG_OF.items():                       # the table check behind it: the fixing reflection swaps the pair
    g = next(g for g in range(8) if g >= 4 and SIG[g][k] == k)
    assert PI[g][a1] == a2 and PI[g][a2] == a1, (k, g)
rep("(3) table check: for each cardinal ray, the reflection fixing it swaps its two neighbouring diagonals", True)

# (4) follower with 8-way moves on the proj input
def reach(moves, hear_fn):
    ds = np.arange(3, 25); Sx, Dd = np.meshgrid(np.arange(8), ds, indexing="ij"); Sx = Sx.ravel(); Dd = Dd.ravel(); B = len(Sx)
    stt = np.full((B, 2), 40); ang = Sx * np.pi / 4
    ipos = (stt + np.stack([np.round(Dd * np.cos(ang)), np.round(Dd * np.sin(ang))], -1).astype(int)) % TM.L
    code = jnp.tile(jnp.arange(8)[None], (B, 1)); item = dict(NI.new_item(B), sec=jnp.array(Sx, jnp.int32), start=jnp.array(stt, jnp.int32), act=jnp.ones(B, bool))
    pos = stt.copy(); done = np.zeros(B, bool)
    for t in range(1, 4 * 24 + 1):
        hr = np.asarray(hear_fn(types.SimpleNamespace(pos=jnp.array(pos, jnp.int32)), item, code, TM.L))
        a = TM.follower(hr, pos, ipos, Sx, moves); live = ~done & (t <= 4 * Dd)
        done |= live & np.all(pos == ipos, -1) & (a == TM.EAT); pos = np.where(done[:, None] | (t > 4 * Dd)[:, None], pos, (pos + np.asarray(moves)[a]) % TM.L)
    return done.reshape(8, len(ds)), ds
Rp, ds = reach(TM.M8, OP.hear_proj); Rc, _ = reach(TM.M8, NI.hear); R4, _ = reach(np.asarray(env.MOVES), NI.hear)
rep("(4) 8-way follower on the --offset-proj input: 8/8 at every d = 3..24", Rp.all(), f"{Rp.sum(1).tolist()} of {len(ds)}")
rep("(4) 8-way follower on the old clip input: 8/8 (as test_moves8)", Rc.all())
rep("(4) NEGATIVE control: 4-way on the clip input fails diagonals", (not R4.all()) and R4[[0, 2, 4, 6]].all(), f"failing cells {int((~R4).sum())}")

# (6) a real 8-way observation through the tie
cfg = env.Cfg(L=96, N=4, patch_sigma=1.0, reproduce=False, A_max=3000, moves8=True); s = env.reset(cfg, jax.random.PRNGKey(1))
item = dict(NI.new_item(4), sec=jnp.array([0, 1, 2, 3], jnp.int32), start=s.pos, act=jnp.ones(4, bool)); pa = jnp.array([0, 2, 7, 9])
x = jnp.concatenate([NI.observe(cfg, s, item), OP.hear_proj(s, item, jnp.tile(jnp.arange(8)[None], (4, 1)), 96), jax.nn.one_hot(pa, 6), jnp.zeros((4, 1)), NI.oracle(item, True), jax.nn.one_hot(pa, 10)[:, 6:]], 1)
outs = [net.apply(p, jnp.zeros(1024), x[i]) for i in range(4)]
rep("(6) a real 8-way + proj observation runs through the tie (Dx 130, 10 logits, finite)", x.shape[1] == 130 and all(o[1].shape == (10,) and bool(jnp.isfinite(o[1]).all()) for o in outs))
print("ALL PASS" if ok_all else "SOME FAIL")
