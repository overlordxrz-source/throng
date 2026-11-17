"""Cam Nov 17 ~07:55Z, L4 (a world / sense fix, ADOPTED): the WIDE in-bout offset input, flag --offset-wide.

The original (named_item.hear) gives the offset to the bout start as clip(torus(start - pos) / 9, +-1) per axis, which
saturates past 9 cells: on a diagonal, beyond d ~ 13 the agent cannot sense x-vs-y drift (G2b). The wide input is
clip(torus(start - pos), +-48) / 24 per axis: the memory record's / 24 scale, saturating only past half the torus
(torus() already returns -48..47 on L = 96, so the clip only guards other L). Token and flag are unchanged."""
import jax
import jax.numpy as jnp

from w2 import named_item as NI


def hear_wide(s, item, code, L):
    N = s.pos.shape[0]; tok = code[jnp.arange(N), item["sec"]]
    hr = jnp.concatenate([jax.nn.one_hot(tok, 8), jnp.ones((N, 1)), jnp.clip(NI.torus(item["start"] - s.pos, L), -48, 48) / 24.0], 1)
    return hr * item["act"][:, None]
