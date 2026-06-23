"""Gradient-flow regression test for the VQ communication bottleneck (hardening H2).

THRONG has repeatedly been bitten by "severance-class" bugs — a learning signal
silently disconnected from the tensor it was meant to train (VQ severed from the
autodiff tape in Phase 18.4; the red-index bug in 18.5). These are invisible to a
loss curve: the number looks fine while the gradient goes nowhere.

This test pins the gradient wiring of `vector_quantize_signals` directly, with no
network / obs / GPU dependency, so it runs in milliseconds and would have caught
both bugs in seconds instead of thousands of PPO updates:

  1. The commitment loss must push gradient into the ENCODER output `z_e`
     (otherwise the speaker never learns to commit to codes).
  2. The codebook loss must push gradient into the CODEBOOK on used rows
     (otherwise the vocabulary never moves — the dead-code spiral).
  3. The straight-through estimator must carry gradient from the broadcast wire
     `signal_out` back to `z_e` (otherwise the policy can't shape what it says).
  4. The broadcast must NOT leak gradient into the codebook (it uses a frozen
     `stop_gradient(codebook)` by design — codebook moves only via the VQ loss).

Run:  PYTHONPATH=. python tests/test_vq_gradient_flow.py
"""

import jax
import jax.numpy as jnp

from jax_sim.network_jax import vector_quantize_signals


M = 32          # agents
D = 12          # slot width (e.g. slot 0)
V = 64          # vocab
BETA = 0.25


def _fixtures():
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    z_e = jax.random.normal(k1, (M, D))
    codebook = jax.random.normal(k2, (V, D))
    return z_e, codebook


def _nonzero(x) -> bool:
    return bool(jnp.isfinite(x).all()) and float(jnp.linalg.norm(x)) > 1e-8


def test_commitment_loss_trains_the_encoder():
    z_e, cb = _fixtures()
    g = jax.grad(lambda ze: vector_quantize_signals(ze, cb, beta=BETA)[2].sum())(z_e)
    assert _nonzero(g), "loss_vq has no gradient w.r.t. z_e — encoder is severed from the VQ loss"


def test_codebook_loss_trains_the_codebook():
    z_e, cb = _fixtures()
    g = jax.grad(lambda c: vector_quantize_signals(z_e, c, beta=BETA)[2].sum())(cb)
    assert _nonzero(g), "loss_vq has no gradient w.r.t. the codebook — vocabulary cannot move"


def test_ste_passes_gradient_from_wire_to_encoder():
    z_e, cb = _fixtures()
    g = jax.grad(lambda ze: vector_quantize_signals(ze, cb, beta=BETA)[0].sum())(z_e)
    assert _nonzero(g), "signal_out has no gradient to z_e — straight-through estimator is broken"


def test_broadcast_does_not_leak_gradient_into_codebook():
    z_e, cb = _fixtures()
    g = jax.grad(lambda c: vector_quantize_signals(z_e, c, beta=BETA)[0].sum())(cb)
    # signal_out = y @ stop_gradient(codebook): by design the broadcast must be a
    # frozen-codebook read, so its gradient to the codebook is exactly zero.
    assert float(jnp.linalg.norm(g)) < 1e-8, "broadcast leaks gradient into the codebook (frozen-read broken)"


if __name__ == "__main__":
    test_commitment_loss_trains_the_encoder()
    test_codebook_loss_trains_the_codebook()
    test_ste_passes_gradient_from_wire_to_encoder()
    test_broadcast_does_not_leak_gradient_into_codebook()
    print("OK: VQ gradient wiring intact (encoder<-commitment, codebook<-vq, wire<-STE, codebook frozen on broadcast).")
