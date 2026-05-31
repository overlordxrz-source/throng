"""
jax_sim/imagination_jax.py — Phase 11.3 K-step mental rollout for epistemic gating.

Rolls carry forward K steps with frozen head_fwd_dyn, scores discounted head_value(carry_k).
Returns imagined argmax actions for gating (low conf_pred → imagined, high → reactive).
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from jax_sim.network_jax import AgentNetworkJax, params_apply_variables


def make_imagination_fn(
    model: AgentNetworkJax,
    K: int = 5,
    gamma: float = 0.999,
) -> Callable:
    """Return jitted (params, carries, action_logits, alive) -> (imagined, gain, agree_greedy)."""

    def _carry_dyn(params, carry, action_oh):
        return model.apply(
            params_apply_variables(params),
            carry,
            action_oh,
            method=model.carry_forward_dynamics,
        )

    def _value(params, carry):
        return model.apply(
            params_apply_variables(params),
            carry,
            method=model.value_from_carry,
        )

    @jax.jit
    def imagine(
        params,
        carries: jnp.ndarray,
        action_logits: jnp.ndarray,
        alive: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Frozen carry-dynamics rollouts for action selection / telemetry.
        imagined: argmax over 5 actions of discounted value sum.
        gain: best imagined return − greedy(logits) imagined return.
        agree_greedy: 1 if imagined == argmax(logits).
        """
        n_agents = carries.shape[0]
        greedy = jnp.argmax(action_logits, axis=-1)
        action_oh_table = jnp.eye(5, dtype=carries.dtype)

        def score_action(a: jnp.ndarray) -> jnp.ndarray:
            action_oh = jnp.broadcast_to(action_oh_table[a], (n_agents, 5))

            def scan_body(carry, _k):
                v = _value(params, carry)
                next_carry = _carry_dyn(params, carry, action_oh)
                return next_carry, v

            _, values = lax.scan(scan_body, carries, jnp.arange(K))
            discounts = (gamma ** jnp.arange(K))[:, None]
            return (values * discounts).sum(axis=0)

        scores = jax.vmap(score_action)(jnp.arange(5))
        imagined = jnp.argmax(scores, axis=0)

        greedy_scores = jnp.take_along_axis(scores, greedy[None, :], axis=0).squeeze(0)
        best_scores = scores.max(axis=0)
        gain = best_scores - greedy_scores
        agree_greedy = (imagined == greedy).astype(jnp.float32)

        alive_f = alive.astype(jnp.float32)
        gain = gain * alive_f
        agree_greedy = agree_greedy * alive_f
        imagined = jnp.where(alive, imagined, 0)
        return imagined, gain, agree_greedy

    return imagine
