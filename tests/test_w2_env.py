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
    cfg = env.Cfg(L=16, N=3, P=1, reproduce=True, A_min=0)
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
