"""Cam Nov 13 GAE/death audit: hand-built trajectories; each property has a mutant that must fail.
python -m pytest tests/test_gae.py --noconftest -q"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2.gae import gae  # noqa: E402

G, L = 0.99, 0.95
rng = np.random.default_rng(0)


def traj(T=10, death=None):
    r = rng.normal(0, 0.1, (1, T)).astype(np.float32); v = rng.normal(-1, 0.3, (1, T)).astype(np.float32)
    dn = np.zeros((1, T), np.float32)
    if death is not None:
        dn[0, death] = 1.0
    return r, v, dn


def no_leak_and_terminal(**kw):
    """a death at t=3, then a NEWBORN's rewards and values in the same slot (t >= 4)"""
    r, v, dn = traj(death=3); a1 = gae(r, v, dn, G, L, **kw)
    r2, v2 = r.copy(), v.copy(); r2[0, 4:] += 5.0; v2[0, 4:] -= 7.0     # change only the newborn's part
    a2 = gae(r2, v2, dn, G, L, **kw)
    leak_free = np.allclose(a1[0, :4], a2[0, :4])
    terminal = np.isclose(a1[0, 3], r[0, 3] - v[0, 3])                 # the death step's target is r alone
    return leak_free and terminal


def bootstrapped_end(**kw):
    """a live agent at the rollout end: the last step bootstraps (not terminal)"""
    r, v, dn = traj(); vl = np.array([-1.7], np.float32); a1 = gae(r, v, dn, G, L, v_last=vl, **kw)
    return np.isclose(a1[0, -1], r[0, -1] + G * vl[0] - v[0, -1])


def test_death_no_leak_terminal():
    assert no_leak_and_terminal()


def test_death_mutant_fails():
    assert not no_leak_and_terminal(ignore_done=True)


def test_rollout_end_bootstrapped():
    assert bootstrapped_end()


def test_rollout_end_mutant_fails():
    assert not bootstrapped_end(end_terminal=True)


def test_matches_inline_trainer():
    """the factored function equals the trainer's former inline loop exactly (production call: v_last = v[:, T-1])"""
    r, v, dn = traj(T=64, death=40); r = np.repeat(r, 3, 0); v = np.repeat(v, 3, 0); dn = np.repeat(dn, 3, 0); dn[1] = 0
    adv = np.zeros_like(r); last = 0.0; T = r.shape[1]
    for t in reversed(range(T)):
        nv = v[:, t + 1] if t + 1 < T else v[:, t]; nt = 1.0 - dn[:, t]
        last = r[:, t] + G * nv * nt - v[:, t] + G * L * nt * last; adv[:, t] = last
    assert np.array_equal(adv, gae(r, v, dn, G, L))
