"""Cam Nov 16 ~17:25Z O-T: tests of the topographic tie in RecAgent (w2/learnability.py)."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2.learnability import RecAgent, TOPO_P  # noqa: E402
from w2 import env  # noqa: E402


def test_P_values():
    P = np.asarray(TOPO_P); MV = np.asarray(env.MOVES)[1:5]
    for m in range(4):
        for k in range(8):
            u = np.array([np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)])
            assert abs(P[m, k] - max(0.0, MV[m] @ u)) < 1e-6, (m, k)   # moves are unit vectors: cos = dot product
    assert P.shape == (4, 8) and P.min() >= 0
    assert abs(P[1, 0] - 1) < 1e-6 and abs(P[0, 4] - 1) < 1e-6 and abs(P[1, 1] - np.sqrt(0.5)) < 1e-6 and P[1, 4] == 0


def _setup():
    x = jax.random.normal(jax.random.PRNGKey(3), (126,)); x = x.at[118:126].set(jax.nn.one_hot(5, 8)); h = jnp.zeros(128)
    off, on = RecAgent(128), RecAgent(128, topo=True)
    po = off.init(jax.random.PRNGKey(0), h, x); pt = on.init(jax.random.PRNGKey(0), h, x)
    return x, h, off, on, po, pt


def test_param_tree():
    x, h, off, on, po, pt = _setup()
    assert "topo_g" not in po["params"] and float(pt["params"]["topo_g"]) == 0.0
    assert set(pt["params"]) - set(po["params"]) == {"topo_g"}


def test_g0_identical_and_gdelta():
    x, h, off, on, po, pt = _setup()
    pt = {"params": dict(po["params"], topo_g=jnp.float32(0.0))}
    _, lo, vo, _ = off.apply(po, h, x); _, lt, vt, _ = on.apply(pt, h, x)
    assert np.array_equal(np.asarray(lo), np.asarray(lt)) and np.array_equal(np.asarray(vo), np.asarray(vt))
    pt = {"params": dict(po["params"], topo_g=jnp.float32(0.7))}
    _, lt, vt, _ = on.apply(pt, h, x); d = np.asarray(lt - lo)
    want = 0.7 * np.concatenate([[0.0], np.asarray(TOPO_P) @ np.asarray(jax.nn.one_hot(5, 8)), [0.0]])
    assert np.allclose(d, want, atol=1e-6) and d[0] == 0 and d[5] == 0 and np.array_equal(np.asarray(vo), np.asarray(vt))
    # sector 5 (225 deg) favours -x and -y equally
    assert abs(d[1] - d[4]) < 1e-6 and d[1] > 0 and abs(d[2]) < 1e-6 and abs(d[3]) < 1e-6


def test_no_signal_no_change():
    x, h, off, on, po, pt = _setup(); x = x.at[118:126].set(0.0)
    pt = {"params": dict(po["params"], topo_g=jnp.float32(3.0))}
    assert np.allclose(np.asarray(off.apply(po, h, x)[1]), np.asarray(on.apply(pt, h, x)[1]))


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("PASS", k)
    print(f"{n}/{n} passed")
