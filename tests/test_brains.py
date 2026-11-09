"""Unit tests for scripts/brains.py (separate-brains toy mechanics, Cam Nov 9). Each test pairs the property with a mutant
that must fail it. Run: python -m pytest tests/test_brains.py --noconftest -q"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import brains  # noqa: E402
from comm_toy import ToyNet  # noqa: E402

H, D, B = 64, 59, 4


@pytest.fixture(scope="module")
def net():
    m = ToyNet(H, wire_blind=1)
    ps = [m.init(jax.random.PRNGKey(10 + b), jnp.zeros((1, H)), jnp.zeros((1, D))) for b in range(B)]
    return m, brains.stack(ps), ps


def vec(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(tree)])


def test_pair_schedule_balanced_and_reshuffled():
    rng = np.random.default_rng(0)
    s1, l1 = brains.pair_schedule(48, B, rng)
    s2, l2 = brains.pair_schedule(48, B, rng)
    assert np.all(s1 != l1)
    pairs, counts = np.unique(np.stack([s1, l1], 1), axis=0, return_counts=True)
    assert len(pairs) == B * (B - 1) and np.all(counts == 48 // (B * (B - 1)))
    for b in range(B):   # every brain speaks and listens equally
        assert (s1 == b).sum() == (l1 == b).sum() == 48 // B
    assert not (np.array_equal(s1, s2) and np.array_equal(l1, l2))   # rotates
    with pytest.raises(ValueError):
        brains.pair_schedule(50, B, rng)


def test_agent_brain_map_layout():
    m = brains.agent_brain_map(np.array([0, 2]), np.array([1, 3]))
    assert m.tolist() == [0, 1, 2, 3]


def test_apply_rows_forward_matches_each_brain(net):
    m, S, ps = net
    rng = np.random.default_rng(1)
    ids = rng.integers(0, B, 16); obs = jnp.asarray(rng.normal(size=(16, D)), jnp.float32); car = jnp.zeros((16, H))
    _, out = brains.apply_rows(m.apply, S, ids, car, obs)
    for r in range(16):
        _, o1 = m.apply(ps[ids[r]], car[r:r + 1], obs[r:r + 1])
        assert np.allclose(np.asarray(out.action_logits[r]), np.asarray(o1.action_logits[0]), atol=1e-5)


def test_gradients_stay_inside_a_brain(net):
    m, S, _ = net
    rng = np.random.default_rng(2)
    ids = np.array([0, 0, 1, 2, 3, 0, 1, 2]); obs = jnp.asarray(rng.normal(size=(8, D)), jnp.float32); car = jnp.zeros((8, H))
    w0 = jnp.asarray(ids == 0, jnp.float32)

    def loss(S_, apply_ids):
        _, o = brains.apply_rows(m.apply, S_, apply_ids, car, obs)
        return jnp.sum(w0[:, None] * o.action_logits ** 2)       # a loss on brain 0's rows only
    g = jax.grad(loss)(S, ids)
    for b in (1, 2, 3):
        assert np.all(vec(brains.take(g, b)) == 0.0), f"brain-0 rows leaked gradient into brain {b}"
    assert np.abs(vec(brains.take(g, 0))).sum() > 0
    # mutant: a shared brain (every row runs brain 1's params) -> brain-0 rows put gradient on brain 1: must be caught
    gm = jax.grad(loss)(S, np.ones_like(ids))
    assert np.abs(vec(brains.take(gm, 1))).sum() > 0


def test_clip_is_per_brain(net):
    _, S, _ = net
    g = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 1e-4, S)
    g = {"params": dict(g["params"])}
    g["params"]["enc_0"] = jax.tree_util.tree_map(lambda x: x.at[2].set(100.0), g["params"]["enc_0"])   # brain 2 explodes
    keys = list(g["params"])
    groups = [["head_action"], ["head_value"], [k for k in keys if k not in ("head_action", "head_value")]]
    c, norms = brains.clip_per_brain(g, groups)
    for b in (0, 1, 3):   # untouched by brain 2's clip
        assert np.array_equal(vec(brains.take(c, b)), vec(brains.take(g, b)))
    assert np.linalg.norm(vec({k: brains.take(c, 2)["params"][k] for k in groups[2]})) <= 2.0 + 1e-4
    # mutant: one global clip over all brains would shrink brains 0 / 1 / 3 as well
    flat = {k: g["params"][k] for k in groups[2]}
    gn = np.sqrt(sum(float(jnp.sum(x ** 2)) for x in jax.tree_util.tree_leaves(flat)))
    assert gn > 2.0   # so a global clip would have scaled every brain


def test_plasticity_schedule():
    p = brains.plasticity([0, 999, 1000, 1500, 2000, 5000], sensitive=1000, decay=1000, floor=0.1)
    assert np.allclose(p, [1, 1, 1, 0.55, 0.1, 0.1])


def test_scale_updates_applies_speed_and_age(net):
    _, S, _ = net
    u = jax.tree_util.tree_map(jnp.ones_like, S)
    lr = np.array([0.5, 1.0, 1.5, 2.0]); mm = np.array([1.0, 0.1, 1.0, 0.5])
    out = brains.scale_updates(u, lr, mm)
    for b in range(B):
        hv = vec(brains.take({"params": {"head_value": out["params"]["head_value"]}}, b))
        ms = vec(brains.take({"params": {"gwt_comms_1": out["params"]["gwt_comms_1"]}}, b))
        assert np.allclose(hv, lr[b]) and np.allclose(ms, lr[b] * mm[b])
    # mutant: ignoring the age multiplier leaves brain 1's message head at lr only -> detected
    mut = brains.scale_updates(u, lr, np.ones(B))
    assert not np.allclose(vec(brains.take({"params": {"gwt_comms_1": mut["params"]["gwt_comms_1"]}}, 1)), lr[1] * mm[1])


def test_vmapped_adam_equals_independent_adam(net):
    _, S, ps = net
    opt = optax.adam(3e-4)
    st = brains.opt_init_per_brain(opt, S)
    rng = np.random.default_rng(3)
    gs = [jax.tree_util.tree_map(lambda x: jnp.asarray(rng.normal(size=x.shape), x.dtype), p) for p in ps]
    sts = [opt.init(p) for p in ps]
    upd, st = brains.opt_update_per_brain(opt, brains.stack(gs), st, S)
    for _ in range(2):   # a few steps so the bias correction (step count) matters
        upd, st = brains.opt_update_per_brain(opt, brains.stack(gs), st, S)
    for b in range(B):
        ub, sb = None, sts[b]
        for _ in range(3):
            ub, sb = opt.update(gs[b], sb, ps[b])
        assert np.allclose(vec(brains.take(upd, b)), vec(ub), atol=1e-7)


def test_newborn_reset_touches_only_that_brain(net):
    m, S, ps = net
    opt = optax.adam(3e-4)
    st = brains.opt_init_per_brain(opt, S)
    g = jax.tree_util.tree_map(jnp.ones_like, S)
    _, st = brains.opt_update_per_brain(opt, g, st, S)            # non-trivial state everywhere
    new = m.init(jax.random.PRNGKey(99), jnp.zeros((1, H)), jnp.zeros((1, D)))
    S2, st2 = brains.newborn_reset(S, st, 2, new, opt.init(new))
    for b in (0, 1, 3):
        assert np.array_equal(vec(brains.take(S2, b)), vec(brains.take(S, b)))
        assert np.array_equal(vec(brains.take(st2, b)), vec(brains.take(st, b)))
    assert np.array_equal(vec(brains.take(S2, 2)), vec(new))
    assert np.array_equal(vec(brains.take(st2, 2)), vec(opt.init(new)))


def test_imitation_only_from_elder_senders():
    ids = np.array([0, 1, 2, 3, 0, 2]); snd = np.array([1, 0, 1, 0, 0, 1], bool); ages = np.array([5000, 3000, 10, 800])
    rows = brains.elder_rows(ids, snd, ages, newborn=2)
    assert rows.tolist() == [True, False, False, False, False, False]   # brain 0 (older) sender only; not brain 2 itself
    rows3 = brains.elder_rows(ids, snd, ages, newborn=3)
    assert rows3.tolist() == [True, False, False, False, False, False]  # brain 2 is younger than brain 3: excluded


def test_msg_grad_mask():
    snd = np.array([1, 0, 1, 0], bool)
    assert brains.msg_grad_mask(snd, "all").tolist() == [1, 1, 1, 1]
    assert brains.msg_grad_mask(snd, "sender").tolist() == [1, 0, 1, 0]
