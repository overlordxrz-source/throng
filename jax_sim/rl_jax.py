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
    Generalized Advantage Estimation.
    Returns (advantages, returns) both (T, N).
    """
    T, N = rewards.shape

    # Compute pure discounted returns (independent of value predictions)
    # This prevents value bootstrap feedback loop
    def _returns_step(carry, t):
        ret = rewards[t] + gamma * carry * (1.0 - dones[t])
        return ret, ret

    _, returns = lax.scan(_returns_step, jnp.zeros(N), jnp.arange(T - 1, -1, -1))
    returns = returns[::-1]  # (T, N)

    # Advantages = returns - predicted values
    advantages = returns - values

    # Normalize advantages (but NOT returns — values must learn raw scale)
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

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

    # Forward pass all timesteps with gradient checkpointing
    # NOTE: stop_gradient through carry_t to prevent exploding BPTT gradients
    # from corrupting the shared policy/value backbone.
    @jax.remat
    def _forward_step(carry_t, inp_t):
        c, o = jax.lax.stop_gradient(carry_t), inp_t
        new_c, outs = apply_fn(params, c, o, n_layers, detach_value=True)
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

    # Value loss: plain MSE (value clipping breaks with decoupled GAE)
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

    # NaN protection: if any NaN, zero out loss to prevent param corruption
    has_nan = jnp.isnan(total_loss) | jnp.isnan(loss_pg) | jnp.isnan(loss_vf)
    total_loss = jnp.where(has_nan, 0.0, total_loss)

    # NaN debug: check individual components
    nan_action_logits = jnp.isnan(action_logits).any()
    nan_values_pred = jnp.isnan(values_pred).any()
    nan_old_log_probs = jnp.isnan(old_log_probs).any()
    nan_advantages = jnp.isnan(advantages).any()
    nan_ratio = jnp.isnan(ratio).any()

    # Metrics returned as JAX arrays (can't call float() inside JIT)
    metrics = {
        "ppo_pg_loss":  jnp.where(has_nan, 0.0, loss_pg),
        "ppo_vf_loss":  jnp.where(has_nan, 0.0, loss_vf),
        "ppo_entropy":  jnp.where(has_nan, 0.0, entropy.sum() / denom),
        "ppo_clip_frac": jnp.where(has_nan, 0.0, jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)),
        "signal_entropy": jnp.where(has_nan, 0.0, signal_entropy.sum() / denom),
        "has_nan": has_nan.astype(jnp.float32),
        "nan_action_logits": nan_action_logits.astype(jnp.float32),
        "nan_values_pred": nan_values_pred.astype(jnp.float32),
        "nan_old_log_probs": nan_old_log_probs.astype(jnp.float32),
        "nan_advantages": nan_advantages.astype(jnp.float32),
        "nan_ratio": nan_ratio.astype(jnp.float32),
        "_values_pred": values_pred,  # for debugging: compare with rollout values
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

    # Debug: print statistics before normalization
    print(f"  [DEBUG] rewards mean={float(rewards.mean()):.4f} std={float(rewards.std()):.4f} min={float(rewards.min()):.4f} max={float(rewards.max()):.4f}")
    print(f"  [DEBUG] values  mean={float(values.mean()):.4f} std={float(values.std()):.4f} min={float(values.min()):.4f} max={float(values.max()):.4f}")
    print(f"  [DEBUG] adv raw mean={float(advantages.mean()):.4f} std={float(advantages.std()):.4f} min={float(advantages.min()):.4f} max={float(advantages.max()):.4f}")
    print(f"  [DEBUG] returns mean={float(returns.mean()):.4f} std={float(returns.std()):.4f} min={float(returns.min()):.4f} max={float(returns.max()):.4f}")

    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std
    # Clip advantages to prevent gradient explosion
    advantages = jnp.clip(advantages, -10.0, 10.0)

    print(f"  [DEBUG] adv norm mean={float(advantages.mean()):.4f} std={float(advantages.std()):.4f} min={float(advantages.min()):.4f} max={float(advantages.max()):.4f}")

    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        params, apply_fn, obs, actions, old_log_probs,
        advantages, returns, carries, n_layers,
        clip_eps, vf_coef, ent_coef,
        alive=alive,
    )

    # Debug: compare ppo_loss forward pass values with rollout values
    values_pred = metrics["_values_pred"]
    val_diff = float(jnp.abs(values_pred - values).max())
    print(f"    [DEBUG] ppo_loss values_pred mean={float(values_pred.mean()):.4f} rollout values mean={float(values.mean()):.4f} max_diff={val_diff:.4f}")

    # Debug: inspect gradient norms for value and action heads
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

    # Debug: value head bias before/after update
    old_bias = None
    for path, p in jax.tree_util.tree_flatten_with_path(params)[0]:
        path_str = "/".join(str(p.key) for p in path)
        if "head_value" in path_str and "bias" in path_str:
            old_bias = float(p.mean())
            print(f"    [DEBUG] value_bias_old={old_bias:.6f}")
            break

    for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
        path_str = "/".join(str(p.key) for p in path)
        if "head_value" in path_str and "bias" in path_str:
            print(f"    [DEBUG] value_bias_grad mean={float(g.mean()):.6f} std={float(g.std()):.6f}")
            break

    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Debug: value head bias after update
    for path, p in jax.tree_util.tree_flatten_with_path(new_params)[0]:
        path_str = "/".join(str(p.key) for p in path)
        if "head_value" in path_str and "bias" in path_str:
            new_bias = float(p.mean())
            if old_bias is not None:
                print(f"    [DEBUG] value_bias_new={new_bias:.6f} delta={new_bias - old_bias:+.6f}")
            break

    metrics["total_loss"] = loss
    return new_params, new_opt_state, metrics
