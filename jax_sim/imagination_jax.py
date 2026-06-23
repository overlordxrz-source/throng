"""
jax_sim/imagination_jax.py — Phase 11.3 K-step mental rollout for epistemic gating.

Rolls carry forward K steps with frozen head_fwd_dyn, scores discounted head_value(carry_k).
Returns imagined argmax actions for gating (conf_pred below batch-relative τ → imagined).
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
    n_imagine_actions: int | None = None,
) -> Callable:
    """Return jitted (params, carries, action_logits, alive) -> (imagined, gain, agree_greedy).

    `n_imagine_actions` controls how many of the action head's logits the K-step
    mental rollout actually scores. Historically this was hard-coded to 5 (Stay +
    N/S/E/W), so the epistemic gate was blind to Strike/Push/Guard and every
    Phase-18 tool action (8-11) — System-2 could not deliberate about the
    combinatorial behaviours the policy can take. Default None preserves the legacy
    5-action behaviour; set it to `model.n_actions` (12) to imagine the full space.
    """
    _n_imag = 5 if n_imagine_actions is None else int(n_imagine_actions)
    _n_imag = max(1, min(_n_imag, int(model.n_actions)))

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
        imagined: argmax over the first `_n_imag` actions of discounted value sum.
        gain: best imagined return − greedy(logits) imagined return.
        agree_greedy: 1 if imagined == argmax(logits).
        """
        n_agents = carries.shape[0]
        greedy = jnp.argmax(action_logits, axis=-1)
        action_oh_table = jnp.eye(model.n_actions, dtype=carries.dtype)

        def score_action(a: jnp.ndarray) -> jnp.ndarray:
            action_oh = jnp.broadcast_to(action_oh_table[a], (n_agents, model.n_actions))

            def scan_body(carry, _k):
                v = _value(params, carry)
                next_carry = _carry_dyn(params, carry, action_oh)
                return next_carry, v

            _, values = lax.scan(scan_body, carries, jnp.arange(K))
            discounts = (gamma ** jnp.arange(K))[:, None]
            return (values * discounts).sum(axis=0)

        scores = jax.vmap(score_action)(jnp.arange(_n_imag))
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
