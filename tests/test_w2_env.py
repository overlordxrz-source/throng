"""Unit tests for the W2 prototype world core (w2/env.py). Run: python -m pytest tests/test_w2_env.py --noconftest -q"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402


def mk(cfg, pos, energy=None, food=None, alive=None, age=None):
    N = cfg.N
    centres = jnp.array([[10, 10]] * cfg.P)
    cap = env.capacity(cfg, centres)
    return env.State(cap if food is None else food, cap, centres, jnp.asarray(pos, jnp.int32),
                     jnp.full(N, 0.5) if energy is None else jnp.asarray(energy),
                     jnp.zeros(N, jnp.int32) if age is None else jnp.asarray(age, jnp.int32),
                     jnp.ones(N, bool) if alive is None else jnp.asarray(alive))


def test_energy_accounting_and_food_conservation():
    cfg = env.Cfg(L=16, N=2, P=1, regrow=0.0)
    s = mk(cfg, [[10, 10], [3, 3]])
    s2, r, info = env.step(cfg, s, jnp.array([env.EAT, 2]), jax.random.PRNGKey(0))
    take = float(info["take"][0])
    assert take > 0 and float(info["take"][1]) == 0
    assert np.isclose(float(s2.energy[0]), 0.5 + cfg.eff * take - cfg.d0)               # ate, didn't move
    assert np.isclose(float(s2.energy[1]), 0.5 - cfg.d0 - cfg.d_move)                     # moved, didn't eat
    assert np.isclose(float(s.food.sum() - s2.food.sum()), take, atol=1e-6)               # food eaten = food gone
    assert s2.pos[1].tolist() == [4, 3]


def test_death_at_zero_energy_and_dead_agents_frozen():
    cfg = env.Cfg(L=16, N=2, P=1)
    s = mk(cfg, [[1, 1], [5, 5]], energy=[0.004, 0.5])
    s2, r, info = env.step(cfg, s, jnp.array([1, 1]), jax.random.PRNGKey(0))
    assert bool(info["dead"][0]) and not bool(s2.alive[0]) and bool(s2.alive[1])
    s3, r3, _ = env.step(cfg, s2, jnp.array([2, 0]), jax.random.PRNGKey(1))
    assert s3.pos[0].tolist() == s2.pos[0].tolist() and float(r3[0]) == 0.0              # dead: no move, no reward


def test_birth_only_above_threshold_into_free_slot():
    cfg = env.Cfg(L=16, N=3, P=1, reproduce=True, A_min=0, rep_mode="threshold")
    s = mk(cfg, [[5, 5], [8, 8], [0, 0]], energy=[0.99, 0.6, 0.5], alive=[True, True, False])
    s2, _, info = env.step(cfg, s, jnp.array([0, 0, 0]), jax.random.PRNGKey(0))
    assert info["births"].tolist() == [False, False, True]                                # slot 2 was free
    assert bool(s2.alive[2]) and int(s2.age[2]) == 0
    assert np.isclose(float(s2.energy[0]), 0.99 - cfg.d0 - cfg.c_rep)                     # parent paid
    d = np.abs(np.array(s2.pos[2]) - np.array(s2.pos[0])); d = np.minimum(d, 16 - d)
    assert d.sum() == 1                                                                   # adjacent to the parent
    # nobody above E_rep -> no births
    s = mk(cfg, [[5, 5], [8, 8], [0, 0]], energy=[0.9, 0.6, 0.5], alive=[True, True, False])
    _, _, info = env.step(cfg, s, jnp.array([0, 0, 0]), jax.random.PRNGKey(0))
    assert not bool(info["births"].any())


def test_window_is_absolute_same_scene_any_world_size():
    """the same local scene gives the same observation at L = 64 and L = 256 (no division by L anywhere)"""
    obs = []
    for L in (64, 256):
        cfg = env.Cfg(L=L, N=1, P=1)
        food = jnp.zeros((L, L)).at[12, 11].set(0.7).at[9, 10].set(0.3)
        s = mk(cfg, [[10, 10]], food=food)
        s = s._replace(centres=jnp.array([[14, 10]]))
        obs.append(np.asarray(env.observe(cfg, s, oracle=True)))
    assert np.array_equal(obs[0], obs[1])
    # mutant: a G-scaled position sense (the toy's pos / G) differs between the two sizes -> would be caught
    assert 10 / 64 != 10 / 256


def test_torus_wrap_in_window_and_moves():
    cfg = env.Cfg(L=16, N=1, P=1, r=1)
    food = jnp.zeros((16, 16)).at[15, 0].set(1.0)
    s = mk(cfg, [[0, 0]], food=food)
    w = np.asarray(env.window(cfg, s.food, s.pos, 1))[0]
    assert w[0, 1] == 1.0                                                                 # (-1, 0) wraps to x = 15
    s2, _, _ = env.step(cfg, s, jnp.array([1]), jax.random.PRNGKey(0))
    assert s2.pos[0].tolist() == [15, 0]


def test_homeostatic_reward_sign():
    cfg = env.Cfg(L=16, N=2, P=1, regrow=0.0)
    s = mk(cfg, [[10, 10], [3, 3]], energy=[0.5, 0.5])
    _, r, _ = env.step(cfg, s, jnp.array([env.EAT, 0]), jax.random.PRNGKey(0))
    assert float(r[0]) > 0 > float(r[1])                                                  # toward the setpoint is rewarded


def full(cfg, pos, energy, alive, fed=None, rel=None, parent=None):
    s = mk(cfg, pos, energy=energy, alive=alive)
    N = cfg.N
    return s._replace(fed=jnp.zeros(N, jnp.int32) if fed is None else jnp.asarray(fed, jnp.int32),
                      rel=(0.5 * (1 - jnp.eye(N))) if rel is None else jnp.asarray(rel),
                      parent=jnp.full(N, -1, jnp.int32) if parent is None else jnp.asarray(parent, jnp.int32))


def test_setpoint_reproduction_after_hold():
    cfg = env.Cfg(L=16, N=3, P=1, reproduce=True, A_min=0, rep_hold=50, regrow=0.0, d0=0.0, d_move=0.0)
    s = full(cfg, [[5, 5], [8, 8], [0, 0]], [0.85, 0.6, 0.5], [True, True, False], fed=[49, 0, 0])
    s2, _, info = env.step(cfg, s, jnp.array([0, 0, 0]), jax.random.PRNGKey(0))
    assert info["births"].tolist() == [False, False, True]                       # 50th step at >= 0.8 -> birth
    assert np.isclose(float(s2.energy[0]), 0.85 - 0.3) and np.isclose(float(s2.energy[2]), 0.5)
    assert int(s2.fed[0]) == 0 and int(s2.parent[2]) == 0
    s = full(cfg, [[5, 5], [8, 8], [0, 0]], [0.85, 0.6, 0.5], [True, True, False], fed=[10, 0, 0])
    _, _, info = env.step(cfg, s, jnp.array([0, 0, 0]), jax.random.PRNGKey(0))
    assert not bool(info["births"].any())                                         # not held long enough


def test_relatedness_lineage():
    cfg = env.Cfg(L=16, N=5, P=1, reproduce=True, A_min=0, rep_hold=1, regrow=0.0, d0=0.0, d_move=0.0)
    rel = np.zeros((5, 5)); rel[0, 1] = rel[1, 0] = 0.5                           # 0 and 1 are founders (0.5)
    s = full(cfg, [[2, 2], [9, 9], [0, 0], [4, 4], [6, 6]], [0.9, 0.5, 0.5, 0.5, 0.5], [True, True, False, False, False],
             fed=[5, 0, 0, 0, 0], rel=rel)
    s, _, info = env.step(cfg, s, jnp.zeros(5, jnp.int32), jax.random.PRNGKey(0))  # 0 has a child in slot 2
    R_ = np.asarray(s.rel)
    assert info["births"].tolist()[2] and R_[2, 0] == 0.5 and R_[0, 2] == 0.5     # parent-offspring 0.5
    assert np.isclose(R_[2, 1], 0.25)                                             # parent's kin at 0.5 x
    s = s._replace(energy=s.energy.at[0].set(0.9), fed=s.fed.at[0].set(5))
    s, _, info = env.step(cfg, s, jnp.zeros(5, jnp.int32), jax.random.PRNGKey(1))  # 0 has a second child (slot 3)
    R_ = np.asarray(s.rel)
    assert info["births"].tolist()[3] and R_[3, 2] == 0.5 and R_[2, 3] == 0.5     # siblings 0.5
    assert np.allclose(R_, R_.T) and np.all(np.diag(R_) == 0)
    s = s._replace(energy=s.energy.at[1].set(0.0))
    s, _, _ = env.step(cfg, s, jnp.ones(5, jnp.int32), jax.random.PRNGKey(2))        # 1 dies -> leaves the kin table
    R_ = np.asarray(s.rel)
    assert np.all(R_[1] == 0) and np.all(R_[:, 1] == 0)


def test_inclusive_reward():
    for inc in (False, True):
        cfg = env.Cfg(L=16, N=2, P=1, regrow=0.0, inclusive=inc)
        s = full(cfg, [[10, 10], [3, 3]], [0.5, 0.5], [True, True])
        _, r, info = env.step(cfg, s, jnp.array([env.EAT, 0]), jax.random.PRNGKey(0))
        own0 = float(env.drive(cfg, 0.5) - env.drive(cfg, 0.5 + cfg.eff * float(info["take"][0]) - cfg.d0))
        own1 = float(env.drive(cfg, 0.5) - env.drive(cfg, 0.5 - cfg.d0))
        if not inc:                                                               # k = 0 control: own reward only
            assert np.isclose(float(r[0]), own0) and np.isclose(float(r[1]), own1)
        else:                                                                     # + 0.5 x the kin's reduction
            assert np.isclose(float(r[0]), own0 + 0.5 * own1, atol=1e-6) and np.isclose(float(r[1]), own1 + 0.5 * own0, atol=1e-6)


def test_parent_pays_juvenile_provisioning():
    cfg = env.Cfg(L=16, N=2, P=1, reproduce=True, juvenile=10, regrow=0.0, d0=0.01, d_move=0.0, rep_hold=10_000)
    s = full(cfg, [[2, 2], [9, 9]], [0.9, 0.3], [True, True], parent=[-1, 0])
    s = s._replace(age=jnp.array([200, 3], jnp.int32))
    s2, _, _ = env.step(cfg, s, jnp.zeros(2, jnp.int32), jax.random.PRNGKey(0))
    assert np.isclose(float(s2.energy[1]), cfg.E0)                                # young topped up to E0
    paid = cfg.E0 - (0.3 - 0.01)
    assert np.isclose(float(s2.energy[0]), 0.9 - 0.01 - paid, atol=1e-6)          # the parent paid exactly that
    s = s._replace(age=jnp.array([200, 50], jnp.int32))                            # past the juvenile period: no top-up
    s3, _, _ = env.step(cfg, s, jnp.zeros(2, jnp.int32), jax.random.PRNGKey(0))
    assert np.isclose(float(s3.energy[1]), 0.3 - 0.01)


def test_food_relocates():
    cfg = env.Cfg(L=32, N=1, P=4, relocate=1.0)
    s = env.reset(cfg, jax.random.PRNGKey(0))
    s2, _, _ = env.step(cfg, s, jnp.zeros(1, jnp.int32), jax.random.PRNGKey(1))
    assert not np.array_equal(np.asarray(s2.centres), np.asarray(s.centres))
    assert float(jnp.max(s2.food - s2.cap)) <= 1e-6                                # food never exceeds the new capacity
