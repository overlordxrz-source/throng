"""W2 generative physics prototype (Cam, Nov 9 ~08:00, open-endedness item 1).

A material is a property vector, never a name:
    [heat, mass, nutrition, hardness, moisture]   each in [0, 1]
A small set of rules acts on vectors and returns vectors (products are materials again, so combinations compound):
    combine(a, b)   mass-weighted mix, plus an interaction: hard + wet softens, hot + wet steams off moisture, and
                    nutrients bind when the two parts are similar in hardness (a "dough"-like product, unnamed)
    burn(m, heat)   if heat >= ignition (drier burns lower): moisture -> 0, nutrition up then charred, mass down
    grow(m)         nutrition and moisture together add mass (only if nutrition > 0.2 and moisture > 0.2)
    decay(m)        nutrition and moisture fall each step, faster when warm
    break_(m, f)    if force > hardness: two pieces of half mass, the same composition
No code path that agents see uses a name; the observation of a material is its vector (or a noisy reading of it).
Novelty metric: distinct quantised vectors produced per life (`signature`).
All functions are pure jnp, so they jit / vmap over cells and agents.
"""
from __future__ import annotations

import jax.numpy as jnp

HEAT, MASS, NUTR, HARD, MOIST = range(5)
NPROP = 5


def _clip(m):
    return jnp.clip(m, 0.0, 1.0)


def combine(a, b):
    wa, wb = a[..., MASS:MASS + 1], b[..., MASS:MASS + 1]
    mix = (a * wa + b * wb) / jnp.maximum(wa + wb, 1e-6)
    mix = mix.at[..., MASS].set(jnp.minimum(a[..., MASS] + b[..., MASS], 1.0))
    wet = mix[..., MOIST]; hard = mix[..., HARD]; hot = mix[..., HEAT]
    mix = mix.at[..., HARD].add(-0.5 * hard * wet)                                   # wet softens
    mix = mix.at[..., MOIST].add(-0.5 * hot * wet)                                   # hot dries
    similar = 1.0 - jnp.abs(a[..., HARD] - b[..., HARD])
    mix = mix.at[..., NUTR].add(0.2 * similar * jnp.minimum(a[..., NUTR], b[..., NUTR]))   # binding gain
    return _clip(mix)


def burn(m, heat):
    ignition = 0.4 + 0.5 * m[..., MOIST]
    on = (heat >= ignition).astype(m.dtype)
    out = m.at[..., MOIST].set(m[..., MOIST] * (1 - on))
    nut = m[..., NUTR]
    cooked = jnp.where(nut < 0.6, nut + 0.3, nut - 0.5)                            # mild heat helps, too much chars
    out = out.at[..., NUTR].set(jnp.where(on > 0, cooked, nut))
    out = out.at[..., MASS].set(m[..., MASS] * (1 - 0.3 * on))
    out = out.at[..., HEAT].set(jnp.maximum(m[..., HEAT], heat * on))
    return _clip(out)


def grow(m, rate=0.05):
    ok = ((m[..., NUTR] > 0.2) & (m[..., MOIST] > 0.2)).astype(m.dtype)
    return _clip(m.at[..., MASS].add(rate * ok * m[..., NUTR] * m[..., MOIST]))


def decay(m, rate=0.01):
    f = rate * (1 + m[..., HEAT])
    return _clip(m.at[..., NUTR].add(-f).at[..., MOIST].add(-0.5 * f).at[..., HEAT].multiply(0.95))


def break_(m, force):
    ok = (force > m[..., HARD]).astype(m.dtype)
    piece = m.at[..., MASS].set(m[..., MASS] * jnp.where(ok > 0, 0.5, 1.0))
    return piece, piece * ok[..., None]                                             # (piece, second piece or zeros)


def signature(m, bins=8):
    """quantised vector -> one integer; distinct signatures = novelty (distinct materials made)"""
    q = jnp.clip((m * bins).astype(jnp.int32), 0, bins - 1)
    return sum(q[..., i] * bins ** i for i in range(NPROP))
