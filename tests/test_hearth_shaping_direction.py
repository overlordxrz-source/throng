"""Direction (sign) tests for hearth_shaping_term, at production shapes.

2026-09-30 (Cam): the beta=24 window came back as "no approach" with shaping at
~55-75% of reward variance. A flat/rising distance is exactly what a *sign or
position* error in the shaping would produce, and the earlier Monte Carlo only
validated the shaping's magnitude. These tests assert the sign against movement:
toward the matching hearth -> F > 0, away -> F < 0, across the torus wrap
boundary too, using the real apply_moves for the positions.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jax_sim.ctd_ramp import hearth_shaping_term
from jax_sim.grid_jax import apply_moves, generate_hearth_positions

GS = 128          # production grid
N = 200           # production max_population
BETA = 24.0       # the value the failed window ran at
GAMMA = 0.999
HEARTHS = np.asarray(generate_hearth_positions(GS))   # (4, 2): (32,32) (32,96) (96,32) (96,96)

# action -> delta table copied by *using* the real apply_moves, not re-typed:
# probe it once so the tests can't drift from the environment's convention.
ACT = {"N": 1, "S": 2, "E": 3, "W": 4}


def _move(pos_xy, action):
    """Real apply_moves on a single agent (production grid, no walls)."""
    pos = jnp.asarray([pos_xy], dtype=jnp.int32)
    walls = jnp.zeros((GS, GS), dtype=bool)
    barrier = jnp.zeros((GS, GS), dtype=jnp.float32)
    new_pos, _ = apply_moves(pos, jnp.asarray([action], dtype=jnp.int32),
                             jnp.asarray([True]), GS, walls, barrier, False)
    return tuple(int(v) for v in np.asarray(new_pos)[0])


def _F(before, after, held, need, deposit=False):
    f = hearth_shaping_term(
        jnp.asarray([before], jnp.int32), jnp.asarray([after], jnp.int32),
        jnp.asarray([held], jnp.int32), jnp.asarray(HEARTHS, jnp.int32),
        jnp.asarray(need, jnp.int32), jnp.asarray([deposit]), BETA, GAMMA, GS)
    return float(np.asarray(f)[0])


def _torus_cheb(p, h):
    dx = abs(p[0] - h[0]); dy = abs(p[1] - h[1])
    return max(min(dx, GS - dx), min(dy, GS - dy))


def _np_min_match_dist(p, held, need):
    ds = [_torus_cheb(p, HEARTHS[i]) for i in range(4) if need[i] == held]
    return min(ds) if ds else None


def _all_moves(pos):
    return {k: _move(pos, a) for k, a in ACT.items()}


def test_toward_positive_away_negative_plain():
    need = [2, 0, 0, 0]          # only hearth 0 at (32,32) wants material 2
    before = (40, 32)
    moves = _all_moves(before)
    # exactly one of E/W approaches along x; find them from geometry, not from names
    toward = [k for k, p in moves.items() if _torus_cheb(p, HEARTHS[0]) < _torus_cheb(before, HEARTHS[0])]
    away = [k for k, p in moves.items() if _torus_cheb(p, HEARTHS[0]) > _torus_cheb(before, HEARTHS[0])]
    assert len(toward) == 1 and len(away) == 1, (moves, toward, away)
    assert _F(before, moves[toward[0]], held=2, need=need) > 0.5 * BETA / (GS // 2)
    assert _F(before, moves[away[0]], held=2, need=need) < -0.5 * BETA / (GS // 2)


def test_torus_wrap_x_toward_is_positive():
    need = [2, 0, 0, 0]                        # hearth 0 at (32,32)
    before = (127, 32)                         # direct dist 95, wrapped dist 33
    assert _torus_cheb(before, HEARTHS[0]) == 33
    moves = _all_moves(before)
    crossing = [k for k, p in moves.items() if p == (0, 32)]
    assert crossing, f"real apply_moves did not wrap x: {moves}"       # proves the env wraps
    assert _F(before, moves[crossing[0]], held=2, need=need) > 0
    leaving = [k for k, p in moves.items() if p == (126, 32)]
    assert leaving and _F(before, moves[leaving[0]], held=2, need=need) < 0


def test_torus_wrap_y_and_other_hearth():
    need = [0, 0, 3, 0]                        # hearth 2 at (96,32)
    before = (96, 2)                           # y: direct 30, wrapped 30 -> pick a wrap-critical point
    before = (96, 126)                         # y: direct 94, wrapped 34
    assert _torus_cheb(before, HEARTHS[2]) == 34
    moves = _all_moves(before)
    wrap = [k for k, p in moves.items() if p == (96, 127) or p == (96, 0)]
    for k, p in moves.items():
        d0, d1 = _torus_cheb(before, HEARTHS[2]), _torus_cheb(p, HEARTHS[2])
        f = _F(before, p, held=3, need=need)
        if d1 < d0: assert f > 0, (k, p, f)
        if d1 > d0: assert f < 0, (k, p, f)
    assert wrap


def test_property_random_positions_actions_needs():
    rng = np.random.default_rng(0)
    checked = wrapped = 0
    for _ in range(1500):
        pos = tuple(int(v) for v in rng.integers(0, GS, size=2))
        held = int(rng.integers(0, 5))
        need = [int(v) for v in rng.integers(0, 5, size=4)]
        d0 = _np_min_match_dist(pos, held, need)
        if d0 is None:
            assert _F(pos, pos, held, need) == 0.0        # no target -> no shaping
            continue
        a = int(rng.integers(1, 5))
        new = _move(pos, a)
        if abs(new[0] - pos[0]) > 1 or abs(new[1] - pos[1]) > 1:
            wrapped += 1
        d1 = _np_min_match_dist(new, held, need)
        f = _F(pos, new, held, need)
        if d1 < d0:
            assert f > 0, (pos, new, held, need, f)
        elif d1 > d0:
            assert f < 0, (pos, new, held, need, f)
        checked += 1
    assert checked > 800 and wrapped > 5


def test_empty_handed_and_deposit_are_zero():
    need = [2, 2, 2, 2]
    assert _F((40, 32), (39, 32), held=-1, need=need) == 0.0
    assert _F((40, 32), (39, 32), held=2, need=need, deposit=True) == 0.0


def test_batched_production_shape_matches_single():
    """N=200 batch through the real function == one-at-a-time (no batch/axis mix-up)."""
    rng = np.random.default_rng(1)
    pos = rng.integers(0, GS, size=(N, 2)); held = rng.integers(-1, 5, size=N)
    need = np.asarray([2, 1, 4, 0])
    new = np.stack([np.asarray(_move(tuple(int(v) for v in p), int(rng.integers(1, 5)))) for p in pos])
    fb = np.asarray(hearth_shaping_term(
        jnp.asarray(pos, jnp.int32), jnp.asarray(new, jnp.int32), jnp.asarray(held, jnp.int32),
        jnp.asarray(HEARTHS, jnp.int32), jnp.asarray(need, jnp.int32), jnp.zeros(N, bool), BETA, GAMMA, GS))
    for i in range(0, N, 7):
        assert abs(fb[i] - _F(tuple(int(v) for v in pos[i]), tuple(int(v) for v in new[i]), int(held[i]), need)) < 1e-6
