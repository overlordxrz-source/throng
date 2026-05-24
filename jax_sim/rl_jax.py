"""
jax_sim/rl_jax.py — PPO loss + GAE + value/grad functions (pure JAX).

All functions are jittable.  The update step uses Optax.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
import optax
from typing import Dict, Tuple, Any


def compute_gae(
    rewards: jnp.ndarray,   # (T, N)
    values:  jnp.ndarray,   # (T, N)
    dones:   jnp.ndarray,   # (T, N)  1.0 = terminal
    gamma: float = 0.99,
    lam:   float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generalized Advantage Estimation.
    Returns (advantages, returns) both (T, N).
    """
    T, N = rewards.shape

    def _step(carry, t):
        next_adv, next_val = carry
        # If done, bootstrap from 0 (episode end)
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        adv = delta + gamma * lam * (1.0 - dones[t]) * next_adv
        return (adv, values[t]), adv

    # Scan backwards
    init = (jnp.zeros(N), jnp.zeros(N))
    _, advantages = lax.scan(_step, init, jnp.arange(T - 1, -1, -1))
    advantages = advantages[::-1]  # reverse back to (T, N)

    returns = advantages + values
    return advantages, returns


def ppo_loss(
    params: Dict,
    apply_fn: Any,  # model.apply
    obs: jnp.ndarray,          # (T, N, obs_dim)
    actions: jnp.ndarray,      # (T, N) int
    old_log_probs: jnp.ndarray, # (T, N)
    advantages: jnp.ndarray,    # (T, N)
    returns: jnp.ndarray,       # (T, N)
    carries: jnp.ndarray,       # (T+1, N, hidden_dim)
    n_layers: int,
    clip_eps: float = 0.2,
    vf_coef: float = 0.25,
    ent_coef: float = 0.05,
    alive: jnp.ndarray = None,  # (T, N) bool-ish
) -> Tuple[jnp.ndarray, Dict]:
    """
    Full PPO loss (policy + value + entropy).
    Returns (loss_scalar, metrics_dict).
    """
    T, N = obs.shape[:2]

    # Forward pass all timesteps
    def _forward_step(carry_t, inp_t):
        c, o = carry_t, inp_t
        new_c, outs = apply_fn(params, c, o, n_layers)
        return new_c, outs

    init_carries = carries[0]  # (N, hidden_dim)
    final_carries, outputs = lax.scan(_forward_step, init_carries, obs)

    # Unpack outputs
    action_logits = outputs[0]      # (T, N, 5)
    signal_logits = outputs[1]      # (T, N, vocab_size)
    symbol_write = outputs[2]       # (T, N, sym_dim)
    values_pred = outputs[3]        # (T, N)
    tom_logits = outputs[4]         # (T, N, K, 5)
    token_ids = outputs[5]         # (T, N)
    signal_probs = outputs[6]      # (T, N, vocab_size)
    culture_fast = outputs[7]      # (T, N, sym_dim)
    culture_slow = outputs[8]      # (T, N, sym_dim)

    # Action log probs
    action_log_probs = jax.nn.log_softmax(action_logits, axis=-1)
    log_probs_taken = jnp.take_along_axis(
        action_log_probs, actions[..., None], axis=-1
    ).squeeze(-1)  # (T, N)

    # Probability ratio
    ratio = jnp.exp(log_probs_taken - old_log_probs)

    # Clipped surrogate objective
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss1 = ratio * advantages
    pg_loss2 = clipped_ratio * advantages
    pg_loss = -jnp.minimum(pg_loss1, pg_loss2)

    # Value loss (MSE)
    vf_loss = jnp.square(values_pred - returns)

    # Entropy bonus
    action_probs = jax.nn.softmax(action_logits, axis=-1)
    entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-10), axis=-1)

    # Signal entropy (discrete vocab)
    signal_entropy = -jnp.sum(signal_probs * jnp.log(signal_probs + 1e-10), axis=-1)

    # Mask dead agents
    if alive is not None:
        mask = alive.astype(jnp.float32)
        pg_loss = pg_loss * mask
        vf_loss = vf_loss * mask
        entropy = entropy * mask
        signal_entropy = signal_entropy * mask
        denom = mask.sum() + 1e-8
    else:
        denom = float(T * N)

    # Aggregate
    loss_pg = pg_loss.sum() / denom
    loss_vf = vf_loss.sum() / denom
    loss_ent = -ent_coef * entropy.sum() / denom
    loss_sig_ent = -0.01 * signal_entropy.sum() / denom  # small signal entropy bonus

    total_loss = loss_pg + vf_coef * loss_vf + loss_ent + loss_sig_ent

    # Metrics returned as JAX arrays (can't call float() inside JIT)
    metrics = {
        "ppo_pg_loss":  loss_pg,
        "ppo_vf_loss":  loss_vf,
        "ppo_entropy":  entropy.sum() / denom,
        "ppo_clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
        "signal_entropy": signal_entropy.sum() / denom,
    }

    return total_loss, metrics


def create_optimizer(lr: float = 3e-4, max_grad_norm: float = 2.0) -> optax.GradientTransformation:
    """Adam + gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )


def ppo_update(
    params: Dict,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    apply_fn: Any,
    batch: Dict[str, jnp.ndarray],
    n_layers: int,
    key: jax.Array,
) -> Tuple[Dict, Any, Dict]:
    """
    Single gradient update step.
    Returns (new_params, new_opt_state, metrics).
    """
    obs = batch["obs"]              # (T, N, obs_dim)
    actions = batch["actions"]      # (T, N)
    old_log_probs = batch["log_probs"]  # (T, N)
    rewards = batch["rewards"]      # (T, N)
    dones = batch["dones"]          # (T, N)
    values = batch["values"]        # (T, N)
    carries = batch["carries"]      # (T+1, N, hidden_dim)
    alive = batch.get("alive")      # (T, N)

    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones)
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    # Loss + grad
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        params, apply_fn, obs, actions, old_log_probs,
        advantages, returns, carries, n_layers,
        alive=alive,
    )

    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    metrics["total_loss"] = loss
    return new_params, new_opt_state, metrics
