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
import numpy as np
import optax
import functools
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
    obs: jnp.ndarray,          # (M, obs_dim)
    actions: jnp.ndarray,      # (M,) int
    old_log_probs: jnp.ndarray, # (M,)
    advantages: jnp.ndarray,    # (M,)
    returns: jnp.ndarray,       # (M,)
    carries: jnp.ndarray,       # (M, hidden_dim) — exact historical carries from rollout
    n_layers: int,
    old_values: jnp.ndarray,     # (M,) for value clipping
    clip_eps: float = 0.2,
    vf_coef: float = 0.25,
    ent_coef: float = 0.05,
    alive: jnp.ndarray = None,  # (M,) bool-ish
) -> Tuple[jnp.ndarray, Dict]:
    """
    PPO loss evaluated with exact historical carries per timestep.
    Using updated params but the same recurrent state that generated the rollout.
    Returns (loss_scalar, metrics_dict).
    """
    M = obs.shape[0]

    # Explicitly stop gradients on inputs
    obs = jax.lax.stop_gradient(obs)
    carries = jax.lax.stop_gradient(carries)

    # Evaluate minibatch directly (no scan needed because samples are independent)
    _, outs = apply_fn(params, carries, obs, n_layers, detach_value=False)

    # Unpack outputs
    action_logits = outs[0]      # (M, 5)
    values_pred = outs[3]        # (M,)

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

    # Value loss with PPO clipping to safely allow backbone learning
    v_clipped = old_values + jnp.clip(values_pred - old_values, -clip_eps, clip_eps)
    err = jnp.abs(values_pred - returns)
    err_clipped = jnp.abs(v_clipped - returns)
    
    delta = 0.5
    def vf_huber(e):
        return jnp.where(e < delta, 0.5 * jnp.square(e), delta * (e - 0.5 * delta))
        
    vf_loss = jnp.maximum(vf_huber(err), vf_huber(err_clipped))

    # Entropy bonus
    action_probs = jax.nn.softmax(action_logits, axis=-1)
    entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-10), axis=-1)
    
    # L2 penalty on logits to prevent vanishing entropy gradients when deterministic
    logit_penalty = 0.01 * jnp.mean(jnp.square(action_logits), axis=-1)

    # Mask dead agents
    if alive is not None:
        mask = alive.astype(jnp.float32)
        pg_loss = pg_loss * mask
        vf_loss = vf_loss * mask
        entropy = entropy * mask
        logit_penalty = logit_penalty * mask
        denom = mask.sum() + 1e-8
    else:
        denom = float(M)

    # Aggregate
    loss_pg = pg_loss.sum() / denom
    loss_vf = vf_loss.sum() / denom
    loss_ent = -ent_coef * entropy.sum() / denom

    loss_logit_penalty = logit_penalty.sum() / denom
    total_loss = loss_pg + vf_coef * loss_vf + loss_ent + loss_logit_penalty

    metrics = {
        "ppo_pg_loss":  loss_pg,
        "ppo_vf_loss":  loss_vf,
        "ppo_entropy":  entropy.sum() / denom,
        "ppo_logit_pen": loss_logit_penalty,
        "ppo_clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
    }

    return total_loss, metrics


def create_optimizer(lr: float = 3e-4, max_grad_norm: float = 2.0) -> optax.GradientTransformation:
    """Adam + gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )


@functools.partial(jax.jit, static_argnames=("apply_fn", "optimizer", "n_layers"))
def _minibatch_step(
    params, opt_state, apply_fn, optimizer,
    obs, actions, old_log_probs, advantages, returns, carries,
    n_layers, old_values, clip_eps, vf_coef, ent_coef, alive
):
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        params, apply_fn, obs, actions, old_log_probs,
        advantages, returns, carries, n_layers,
        old_values, clip_eps, vf_coef, ent_coef, alive=alive
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    metrics["total_loss"] = loss
    return new_params, new_opt_state, metrics, grads


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
    minibatch_size: int = 512,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[Dict, Any, Dict]:
    """
    Single gradient update step using minibatches.
    Offloads rollout data to CPU; only one minibatch lives on GPU at a time.
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

    advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

    # Debug
    print(f"  [DEBUG] rewards mean={float(rewards.mean()):.4f} std={float(rewards.std()):.4f}")
    print(f"  [DEBUG] values  mean={float(values.mean()):.4f} std={float(values.std()):.4f}")
    print(f"  [DEBUG] adv     mean={float(advantages.mean()):.4f} std={float(advantages.std()):.4f}")
    print(f"  [DEBUG] returns mean={float(returns.mean()):.4f} std={float(returns.std()):.4f}")

    # Flatten (T, N, ...) -> (M, ...) and move to CPU (numpy) to free GPU VRAM.
    # The full rollout obs alone is ~2.3GB on GPU — must be freed before backward pass.
    T, N = obs.shape[:2]
    M = T * N

    def flatten_to_cpu(x):
        if x is None: return None
        return np.asarray(x.reshape((M,) + x.shape[2:]))

    flat_obs = flatten_to_cpu(obs)
    flat_actions = flatten_to_cpu(actions)
    flat_log_probs = flatten_to_cpu(old_log_probs)
    flat_adv = flatten_to_cpu(advantages)
    flat_ret = flatten_to_cpu(returns)
    flat_carries = flatten_to_cpu(carries)
    flat_values = flatten_to_cpu(values)
    flat_alive = flatten_to_cpu(alive)

    # Delete GPU references so XLA can reclaim VRAM
    del obs, actions, old_log_probs, advantages, returns, carries, values, alive
    del batch

    # Shuffle on CPU (no GPU allocation for permutation array)
    rng = np.random.RandomState(int(jax.random.bits(key, dtype=jnp.uint32)))
    perm = rng.permutation(M)

    # Minibatch loop — GPU only holds params + opt_state + one minibatch + backward workspace
    minibatch_size = int(minibatch_size)
    if minibatch_size <= 0:
        raise ValueError(f"ppo_minibatch_size must be positive, got {minibatch_size}")
    n_minibatches = M // minibatch_size

    # Accumulate metrics as Python floats (not 500 JAX scalar dicts)
    metric_sums = {}
    final_grads = None

    for i in range(n_minibatches):
        idx = perm[i * minibatch_size : (i + 1) * minibatch_size]

        # Transfer just this minibatch to GPU
        mb_obs = jnp.array(flat_obs[idx])
        mb_act = jnp.array(flat_actions[idx])
        mb_lp = jnp.array(flat_log_probs[idx])
        mb_adv = jnp.array(flat_adv[idx])
        mb_ret = jnp.array(flat_ret[idx])
        mb_c = jnp.array(flat_carries[idx])
        mb_v = jnp.array(flat_values[idx])
        mb_al = jnp.array(flat_alive[idx]) if flat_alive is not None else None

        params, opt_state, mb_mets, grads = _minibatch_step(
            params, opt_state, apply_fn, optimizer,
            mb_obs, mb_act, mb_lp, mb_adv, mb_ret, mb_c,
            n_layers, mb_v, clip_eps, vf_coef, ent_coef, mb_al
        )

        # Accumulate as Python floats to avoid holding 500 JAX arrays
        for k, v in mb_mets.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
        final_grads = grads

    # Average metrics
    metrics = {k: v / n_minibatches for k, v in metric_sums.items()}

    # Debug: gradient norms (using the last minibatch's gradients)
    def _head_grad_norm(grads_tree, head_name):
        norms = []
        for path, g in jax.tree_util.tree_flatten_with_path(grads_tree)[0]:
            path_str = "/".join(str(p.key) for p in path)
            if head_name in path_str:
                norms.append(jnp.sum(g**2))
        return jnp.sqrt(jnp.sum(jnp.array(norms))) if norms else jnp.array(0.0)

    if final_grads is not None:
        vf_grad_norm = _head_grad_norm(final_grads, "head_value")
        act_grad_norm = _head_grad_norm(final_grads, "head_action")
        total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(final_grads)))
        print(f"    [DEBUG] grad_norms (last mb) total={float(total_grad_norm):.4f} vf={float(vf_grad_norm):.4f} act={float(act_grad_norm):.4f}")

    return params, opt_state, metrics


# ── Auxiliary Losses: Forward Dynamics + Self-Prediction (Phase 9.1 + 9.2) ───

@functools.partial(jax.jit, static_argnames=("aux_apply_fn", "optimizer", "fwd_coef", "self_pred_coef"))
def _aux_minibatch_step(
    params, opt_state, aux_apply_fn, optimizer,
    carry_t, action_oh, carry_tp1, action_tp1_oh, alive_mask,
    fwd_coef, self_pred_coef,
):
    """
    Single minibatch gradient step combining:
      - Forward dynamics loss: predict carry_{t+1} from (carry_t, action_t)
      - Self-prediction loss:  predict action_{t+1} from carry_t
    """
    def loss_fn(p):
        carry_pred, self_pred_logits = aux_apply_fn(p, carry_t, action_oh)

        # Forward dynamics — stop_gradient on target is CRITICAL.
        # Without it the model learns carry_pred = 0.9 * carry_t trivially.
        target = jax.lax.stop_gradient(carry_tp1)
        fwd_err = jnp.square(carry_pred - target).mean(axis=-1)          # (M,)
        fwd_loss = (fwd_err * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        # Self-prediction cross-entropy
        log_probs = jax.nn.log_softmax(self_pred_logits, axis=-1)
        ce = -jnp.sum(action_tp1_oh * log_probs, axis=-1)                # (M,)
        sp_loss = (ce * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        total = fwd_coef * fwd_loss + self_pred_coef * sp_loss
        return total, (fwd_loss, sp_loss, self_pred_logits)

    (_, (fwd_l, sp_l, sp_logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Self-prediction accuracy (random baseline = 0.20 for 5 actions)
    sp_acc = (
        (jnp.argmax(sp_logits, axis=-1) == jnp.argmax(action_tp1_oh, axis=-1)).astype(jnp.float32)
        * alive_mask
    ).sum() / (alive_mask.sum() + 1e-8)

    return new_params, new_opt_state, fwd_l, sp_l, sp_acc


def auxiliary_update(
    params,
    opt_state,
    optimizer,
    aux_apply_fn,         # model.apply bound to auxiliary_heads method
    carries_np,           # (T, N, hidden_dim) numpy — full rollout carries
    actions_np,           # (T, N) numpy int    — full rollout actions
    alive_np,             # (T, N) numpy float, or None
    key: jax.Array,
    minibatch_size: int = 1024,
    fwd_coef: float = 0.05,
    self_pred_coef: float = 0.1,
) -> Tuple[Dict, Any, float, float, float]:
    """
    Compute forward dynamics + self-prediction auxiliary losses using
    sequential (t, t+1) carry pairs from the rollout.

    Returns: (params, opt_state, avg_fwd_loss, avg_sp_loss, avg_sp_acc)
    Self-prediction accuracy starts at ~0.20 (random) and rises if agents
    build internal models of their own behaviour.
    """
    T, N, hidden_dim = carries_np.shape

    carry_t      = carries_np[:-1].reshape((T - 1) * N, hidden_dim)
    carry_tp1    = carries_np[1:].reshape((T - 1) * N, hidden_dim)
    action_t     = actions_np[:-1].reshape((T - 1) * N)
    action_tp1   = actions_np[1:].reshape((T - 1) * N)
    action_oh    = np.eye(5, dtype=np.float32)[action_t]    # ((T-1)*N, 5)
    action_tp1_oh = np.eye(5, dtype=np.float32)[action_tp1] # ((T-1)*N, 5)

    if alive_np is not None:
        alive_t = alive_np[:-1].reshape((T - 1) * N).astype(np.float32)
    else:
        alive_t = np.ones((T - 1) * N, dtype=np.float32)

    M    = carry_t.shape[0]
    rng  = np.random.RandomState(int(jax.random.bits(key, dtype=jnp.uint32)))
    perm = rng.permutation(M)
    n_mb = max(1, M // minibatch_size)

    fwd_sum = sp_loss_sum = sp_acc_sum = 0.0

    for i in range(n_mb):
        idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
        params, opt_state, fwd_l, sp_l, sp_a = _aux_minibatch_step(
            params, opt_state, aux_apply_fn, optimizer,
            jnp.array(carry_t[idx]),
            jnp.array(action_oh[idx]),
            jnp.array(carry_tp1[idx]),
            jnp.array(action_tp1_oh[idx]),
            jnp.array(alive_t[idx]),
            fwd_coef, self_pred_coef,
        )
        fwd_sum      += float(fwd_l)
        sp_loss_sum  += float(sp_l)
        sp_acc_sum   += float(sp_a)

    return params, opt_state, fwd_sum / n_mb, sp_loss_sum / n_mb, sp_acc_sum / n_mb


# Keep old name as alias for backward compatibility with any external callers
def fwd_dynamics_update(params, opt_state, optimizer, fwd_apply_fn,
                        carries_np, actions_np, alive_np, key,
                        minibatch_size=1024, fwd_coef=0.05):
    """Deprecated: use auxiliary_update instead."""
    import functools as _ft
    aux_fn = _ft.partial(
        lambda apply, p, ct, ao: apply(p, ct, ao)[0],  # extract carry_pred only
        fwd_apply_fn,
    )
    p, o, fl, _, _ = auxiliary_update(
        params, opt_state, optimizer, fwd_apply_fn,
        carries_np, actions_np, alive_np, key,
        minibatch_size=minibatch_size, fwd_coef=fwd_coef, self_pred_coef=0.0,
    )
    return p, o, fl
