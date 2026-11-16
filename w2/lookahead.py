"""Cam Nov 16 ~01:00: the LOOKAHEAD curriculum, one shared sampler for port_w2 (gate controls) and learner_w2_la.

Each bout: with probability p, d is drawn uniformly from frontier+1 .. d_la (a LOOKAHEAD bout); otherwise from
1 .. frontier exactly as before (the same key, so p = 0 reproduces the old sampler bit for bit). When the frontier is
already >= d_la there are no lookahead bouts. The gate scores only IN-RANGE bouts (d <= frontier), so lookahead bouts
never block or fake an advance."""
import jax
import jax.numpy as jnp


def sampler(dfront, N, p=0.0, d_la=12):
    def f(k):
        din = jnp.floor(jax.random.uniform(k, (N,)) * dfront) + 1.0        # the existing in-range draw, same key
        if p <= 0:
            return din
        n_out = jnp.maximum(d_la - dfront, 0.0)
        dout = dfront + 1.0 + jnp.floor(jax.random.uniform(jax.random.fold_in(k, 11), (N,)) * n_out)
        use = (jax.random.uniform(jax.random.fold_in(k, 12), (N,)) < p) & (n_out > 0)
        return jnp.where(use, dout, din)
    return f


def in_range(d, dfront):
    """Mask of bouts the gate may score (d is the closing bout's distance)."""
    return d <= dfront
