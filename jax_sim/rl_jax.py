from __future__ import annotations

"""
jax_sim/rl_jax.py — PPO loss + GAE + value/grad functions (pure JAX).

All functions are jittable.  The update step uses Optax.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

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
    Pure discounted returns + advantage = returns - values.
    NO bootstrapping from predicted values to break feedback loop.
    For short rollouts (32 steps) this is standard and stable.
    Returns (advantages, returns) both (T, N).
    """
    T, N = rewards.shape

    # Compute pure discounted returns independent of value predictions
    def _returns_step(carry, t):
        ret = rewards[t] + gamma * carry * (1.0 - dones[t])
        return ret, ret

    _, returns = lax.scan(_returns_step, jnp.zeros(N), jnp.arange(T - 1, -1, -1))
    returns = returns[::-1]  # (T, N)

    # Advantages = returns - predicted values
    advantages = returns - values

    # Normalize advantages only (values learn raw returns scale)
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    advantages = (advantages - adv_mean) / adv_std
    advantages = jnp.clip(advantages, -10.0, 10.0)

    return advantages, returns


def ppo_loss(
    params: Dict,
    apply_fn: Any,  # model.apply
    obs: jnp.ndarray,          # (T, N, obs_dim)
    actions: jnp.ndarray,      # (T, N) int
    old_log_probs: jnp.ndarray, # (T, N)
    advantages: jnp.ndarray,    # (T, N)
    returns: jnp.ndarray,       # (T, N)
    carries: jnp.ndarray,       # (T, N, hidden_dim) — exact historical carries from rollout
    n_layers: int,
    old_values: jnp.ndarray,     # (T, N) for value clipping
    clip_eps: float = 0.2,
    vf_coef: float = 0.25,
    ent_coef: float = 0.05,
    alive: jnp.ndarray = None,  # (T, N) bool-ish
) -> Tuple[jnp.ndarray, Dict]:
    """
    PPO loss evaluated with exact historical carries per timestep.
    Using updated params but the same recurrent state that generated the rollout.
    Returns (loss_scalar, metrics_dict).
    """
    T, N = obs.shape[:2]

    # Evaluate each timestep with its exact historical carry (ignore new_carry output)
    @jax.remat
    def _eval_step(t, _):
        c = jax.lax.stop_gradient(carries[t])
        _, outs = apply_fn(params, c, obs[t], n_layers, detach_value=True)
        return t + 1, outs

    _, outputs = lax.scan(_eval_step, 0, None, length=T)

    # Unpack outputs
    action_logits = outputs[0]      # (T, N, 5)
    values_pred = outputs[3]        # (T, N)

    # Action log probs
    action_log_probs = jax.nn.log_softmax(action_logits, axis=-1)
    log_probs_taken = jnp.take_along_axis(
        action_log_probs, actions[..., None], axis=-1
    ).squeeze(-1)

    # Probability ratio
    ratio = jnp.exp(log_probs_taken - old_log_probs)
    ratio = jnp.clip(ratio, 0.0, 10.0)

    # Clipped surrogate objective
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    pg_loss = -jnp.minimum(surr1, surr2)

    # Value loss: Huber (smooth L1) instead of MSE to prevent gradient explosion
    # from large errors while still being quadratic near the target.
    delta = 0.5
    err = jnp.abs(values_pred - returns)
    vf_loss = jnp.where(err < delta, 0.5 * jnp.square(err), delta * (err - 0.5 * delta))

    # Entropy bonus with floor to prevent deterministic policy collapse
    action_probs = jax.nn.softmax(action_logits, axis=-1)
    entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-10), axis=-1)
    # Entropy floor: penalize dropping below 0.5 (for 5 actions, this keeps some exploration)
    min_entropy = 0.5
    entropy_penalty = jnp.where(entropy < min_entropy, 2.0 * jnp.square(entropy - min_entropy), 0.0)

    # Mask dead agents
    if alive is not None:
        mask = alive.astype(jnp.float32)
        pg_loss = pg_loss * mask
        vf_loss = vf_loss * mask
        entropy = entropy * mask
        entropy_penalty = entropy_penalty * mask
        denom = mask.sum() + 1e-8
    else:
        denom = float(T * N)

    # Aggregate
    loss_pg = pg_loss.sum() / denom
    loss_vf = vf_loss.sum() / denom
    loss_ent = -ent_coef * entropy.sum() / denom

    loss_ent_penalty = entropy_penalty.sum() / denom
    total_loss = loss_pg + vf_coef * loss_vf + loss_ent + loss_ent_penalty

    metrics = {
        "ppo_pg_loss":  loss_pg,
        "ppo_vf_loss":  loss_vf,
        "ppo_entropy":  entropy.sum() / denom,
        "ppo_ent_penalty": loss_ent_penalty,
        "ppo_clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
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
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[Dict, Any, Dict]:
    """
    Single gradient update step.
    Returns (new_params, new_opt_state, metrics).
    """
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    values = batch["values"]
    carries = batch["carries"]
    alive = batch.get("alive")

    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones)

    # Debug
    print(f"  [DEBUG] rewards mean={float(rewards.mean()):.4f} std={float(rewards.std()):.4f}")
    print(f"  [DEBUG] values  mean={float(values.mean()):.4f} std={float(values.std()):.4f}")
    print(f"  [DEBUG] adv     mean={float(advantages.mean()):.4f} std={float(advantages.std()):.4f}")
    print(f"  [DEBUG] returns mean={float(returns.mean()):.4f} std={float(returns.std()):.4f}")

    old_values = batch["values"]  # (T, N)
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        params, apply_fn, obs, actions, old_log_probs,
        advantages, returns, carries, n_layers,
        old_values,
        clip_eps, vf_coef, ent_coef,
        alive=alive,
    )

    # Debug: gradient norms
    def _head_grad_norm(grads_tree, head_name):
        norms = []
        for path, g in jax.tree_util.tree_flatten_with_path(grads_tree)[0]:
            path_str = "/".join(str(p.key) for p in path)
            if head_name in path_str:
                norms.append(jnp.sum(g**2))
        return jnp.sqrt(jnp.sum(jnp.array(norms))) if norms else jnp.array(0.0)

    vf_grad_norm = _head_grad_norm(grads, "head_value")
    act_grad_norm = _head_grad_norm(grads, "head_action")
    total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    print(f"    [DEBUG] grad_norms total={float(total_grad_norm):.4f} vf={float(vf_grad_norm):.4f} act={float(act_grad_norm):.4f}")

    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    metrics["total_loss"] = loss
    return new_params, new_opt_state, metrics
