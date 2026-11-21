"""Cam Nov 21 ~21:25Z fallback (3), flag --offset-proj (code only; launches only on Cam's ruling after seed 9's verdict).

PROJECTIVE in-bout offset input: off / max(9, ||off||_inf) per agent, where off = torus(start - pos).
  - Within 9 cells (||off||_inf <= 9) the denominator is exactly 9.0, so it is BITWISE equal to named_item.hear's
    clip(off / 9, +-1) (the same float division).
  - Beyond 9 cells it keeps the DIRECTION (the dx : dy ratio) instead of saturating each axis, so a diagonal walker can
    still see cross-track drift (G2b). Range stays [-1, 1].
  - D4-equivariant: ||M off||_inf = ||off||_inf for every signed permutation M, so proj(M off) = M proj(off).
Token and flag unchanged; Dx unchanged. (Carbon's proposal, carbon.md Finding 1C.)"""
import jax
import jax.numpy as jnp

from w2 import named_item as NI


def proj(off):
    off = off.astype(jnp.float32)
    return off / jnp.maximum(jnp.float32(9.0), jnp.max(jnp.abs(off), -1, keepdims=True))


def hear_proj(s, item, code, L):
    N = s.pos.shape[0]; tok = code[jnp.arange(N), item["sec"]]
    hr = jnp.concatenate([jax.nn.one_hot(tok, 8), jnp.ones((N, 1)), proj(NI.torus(item["start"] - s.pos, L))], 1)
    return hr * item["act"][:, None]
