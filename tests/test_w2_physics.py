"""Tests for the W2 generative physics prototype. Run: python -m pytest tests/test_w2_physics.py --noconftest -q"""
import inspect
import itertools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import physics as ph  # noqa: E402

BASE = jnp.array([[0.1, 0.3, 0.7, 0.2, 0.6],      # soft, wet, nutritious
                  [0.0, 0.5, 0.0, 0.9, 0.0],      # hard, dry, inert
                  [0.2, 0.2, 0.3, 0.3, 0.9]])     # watery


def test_closure_products_are_materials():
    for a, b in itertools.product(BASE, BASE):
        c = ph.combine(a, b)
        assert c.shape == (ph.NPROP,) and bool(jnp.all((c >= 0) & (c <= 1)))
    for m in BASE:
        for f in (ph.burn(m, 0.9), ph.grow(m), ph.decay(m), *ph.break_(m, 0.95)):
            assert f.shape == (ph.NPROP,) and bool(jnp.all((f >= 0) & (f <= 1)))


def test_combinations_compound():
    """depth-2 products include materials no depth-1 product is (products are materials, so they combine again)"""
    d1 = [ph.combine(a, b) for a, b in itertools.product(BASE, BASE)]
    s1 = {int(ph.signature(m)) for m in d1}
    d2 = [ph.combine(a, b) for a, b in itertools.product(d1, list(BASE) + d1)]
    d2 += [ph.burn(m, 0.9) for m in d1]
    s2 = {int(ph.signature(m)) for m in d2}
    assert len(s2 - s1 - {int(ph.signature(m)) for m in BASE}) > 0


def test_rules_have_effects():
    wet_nut = BASE[0]
    assert float(ph.grow(wet_nut)[ph.MASS]) > float(wet_nut[ph.MASS])
    assert float(ph.decay(wet_nut)[ph.NUTR]) < float(wet_nut[ph.NUTR])
    assert float(ph.burn(wet_nut, 0.2)[ph.MOIST]) == float(wet_nut[ph.MOIST])          # below ignition: nothing
    assert float(ph.burn(wet_nut, 0.95)[ph.MOIST]) == 0.0                              # above: dried
    p, q = ph.break_(BASE[1], 0.5)                                                     # force < hardness
    assert float(q.sum()) == 0.0 and float(p[ph.MASS]) == float(BASE[1][ph.MASS])
    p, q = ph.break_(BASE[1], 0.95)
    assert np.isclose(float(p[ph.MASS] + q[ph.MASS]), float(BASE[1][ph.MASS]))         # mass conserved


def test_vectorised_and_jittable():
    A = jax.random.uniform(jax.random.PRNGKey(0), (1000, ph.NPROP)); Bm = A[::-1]
    out = jax.jit(jax.vmap(ph.combine))(A, Bm)
    assert out.shape == (1000, ph.NPROP)


def test_no_named_objects_in_agent_facing_code():
    """the module exposes properties and rules only: no object-name vocabulary in its code"""
    src = inspect.getsource(ph).lower()
    for w in ("wood", "stone", "fire", "bread", "water", "tree", "apple", "meat", "tool"):
        assert w not in src.split('"""', 2)[-1], w     # outside the module docstring
