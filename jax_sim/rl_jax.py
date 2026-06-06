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
    rewards,   # (T, N) — numpy or JAX
    values,    # (T, N)
    dones,     # (T, N)  1.0 = terminal
    gamma: float = 0.99,
    lam:   float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure discounted returns + advantage = returns - values (CPU numpy loop).

    Rollout batches are offloaded to CPU before PPO; a JAX lax.scan here would
    index numpy arrays with traced indices and raise TracerArrayConversionError.
    """
    del lam  # kept for API compat; no value bootstrapping in this rollout setup
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    T, N = rewards.shape

    returns = np.empty((T, N), dtype=np.float32)
    carry = np.zeros(N, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        carry = rewards[t] + gamma * carry * (1.0 - dones[t])
        returns[t] = carry

    advantages = returns - values
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std
    advantages = np.clip(advantages, -10.0, 10.0)

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
    vq_coef: float = 0.1,
    loss_vq_rollout: jnp.ndarray = None,  # (M,) per-agent VQ loss from rollout
    alive: jnp.ndarray = None,  # (M,) bool-ish
    rng_key: jax.Array = None,  # Added for noise
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
    kwargs = {}
    if rng_key is not None:
        kwargs["rngs"] = {"dropout": rng_key}
    _, outs = apply_fn(params, carries, obs, n_layers, detach_value=False, **kwargs)

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

    loss_vq_mean = jnp.array(0.0)
    if loss_vq_rollout is not None:
        vq = loss_vq_rollout
        if alive is not None:
            vq = vq * mask
        loss_vq_mean = vq.sum() / denom
        total_loss = total_loss + vq_coef * loss_vq_mean

    metrics = {
        "ppo_pg_loss":  loss_pg,
        "ppo_vf_loss":  loss_vf,
        "ppo_entropy":  entropy.sum() / denom,
        "ppo_logit_pen": loss_logit_penalty,
        "ppo_clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
        "ppo_vq_loss": loss_vq_mean,
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
    n_layers, old_values, clip_eps, vf_coef, ent_coef, vq_coef, loss_vq, alive, rng_key
):
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        params, apply_fn, obs, actions, old_log_probs,
        advantages, returns, carries, n_layers,
        old_values, clip_eps, vf_coef, ent_coef, vq_coef, loss_vq, alive=alive, rng_key=rng_key
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
    vq_coef: float = 0.1,
    minibatch_size: int = 512,
    gamma: float = 0.99,
    lam: float = 0.95,
    team: str = "blue",
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
    loss_vq = batch.get("loss_vq")

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
    flat_loss_vq = flatten_to_cpu(loss_vq)

    # Delete GPU references so XLA can reclaim VRAM
    del obs, actions, old_log_probs, advantages, returns, carries, values, alive, loss_vq
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

        if i == 0:
            print(
                f"  [JAX] {team} PPO minibatch 1/{n_minibatches} "
                f"(M={M}, mb={minibatch_size}) — H2D + backward...",
                flush=True,
            )

        # Transfer just this minibatch to GPU
        mb_obs = jnp.array(flat_obs[idx])
        mb_act = jnp.array(flat_actions[idx])
        mb_lp = jnp.array(flat_log_probs[idx])
        mb_adv = jnp.array(flat_adv[idx])
        mb_ret = jnp.array(flat_ret[idx])
        mb_c = jnp.array(flat_carries[idx])
        mb_v = jnp.array(flat_values[idx])
        mb_al = jnp.array(flat_alive[idx]) if flat_alive is not None else None
        mb_vq = jnp.array(flat_loss_vq[idx]) if flat_loss_vq is not None else None

        key, mb_key = jax.random.split(key)
        params, opt_state, mb_mets, grads = _minibatch_step(
            params, opt_state, apply_fn, optimizer,
            mb_obs, mb_act, mb_lp, mb_adv, mb_ret, mb_c,
            n_layers, mb_v, clip_eps, vf_coef, ent_coef, vq_coef, mb_vq, mb_al, mb_key
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

@functools.partial(
    jax.jit,
    static_argnames=(
        "aux_apply_fn", "optimizer",
        "fwd_coef", "carry_fwd_coef", "self_pred_coef", "conf_coef", "proprio_coef",
    ),
)
def _aux_minibatch_step(
    params, opt_state, aux_apply_fn, optimizer,
    carry_t, action_oh, loc_env_tp1, carry_tp1, action_tp1_oh, energy_tp1, alive_mask,
    fwd_coef, carry_fwd_coef, self_pred_coef, conf_coef, proprio_coef,
):
    """
    Single minibatch gradient step combining:
      - Forward dynamics (loc_env): predict flat loc_env_{t+1} from (carry_t, action_t)
      - Latent forward dynamics: predict carry_{t+1} from (carry_t, action_t)
      - Self-prediction loss: predict action_{t+1} from carry_t
    """
    def loss_fn(p):
        env_pred, self_pred_logits, carry_pred, conf_pred, energy_pred = aux_apply_fn(
            p, carry_t, action_oh
        )

        # External world target — stop_gradient on loc_env_{t+1}
        env_target = jax.lax.stop_gradient(loc_env_tp1)
        fwd_err = jnp.mean(jnp.square(env_pred - env_target), axis=-1)
        fwd_loss = (fwd_err * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        # Latent carry target — stop_gradient or network learns identity (0.9*carry_t)
        carry_target = jax.lax.stop_gradient(carry_tp1)
        carry_err = jnp.square(carry_pred - carry_target)
        carry_mse = jnp.mean(carry_err, axis=-1)
        carry_fwd_loss = (carry_mse * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        # Phase 9.1 — confidence predicts detached per-agent carry MSE
        conf_target = jax.lax.stop_gradient(carry_mse)
        conf_sq = jnp.square(conf_pred - conf_target)
        conf_loss = (conf_sq * alive_mask).sum() / (alive_mask.sum() + 1e-8)
        conf_pred_mean = (conf_pred * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        log_probs = jax.nn.log_softmax(self_pred_logits, axis=-1)
        ce = -jnp.sum(action_tp1_oh * log_probs, axis=-1)
        sp_loss = (ce * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        energy_target = jax.lax.stop_gradient(energy_tp1)
        proprio_per = jnp.square(energy_pred - energy_target)
        proprio_loss = (proprio_per * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        total = (
            fwd_coef * fwd_loss
            + carry_fwd_coef * carry_fwd_loss
            + self_pred_coef * sp_loss
            + conf_coef * conf_loss
            + proprio_coef * proprio_loss
        )
        return total, (
            fwd_loss,
            carry_fwd_loss,
            sp_loss,
            self_pred_logits,
            conf_loss,
            conf_pred_mean,
            proprio_loss,
        )

    (_, (fwd_l, carry_fwd_l, sp_l, sp_logits, conf_l, conf_pred_m, proprio_l)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    sp_acc = (
        (jnp.argmax(sp_logits, axis=-1) == jnp.argmax(action_tp1_oh, axis=-1)).astype(jnp.float32)
        * alive_mask
    ).sum() / (alive_mask.sum() + 1e-8)

    return new_params, new_opt_state, fwd_l, carry_fwd_l, sp_l, sp_acc, conf_l, conf_pred_m, proprio_l


def auxiliary_update(
    params,
    opt_state,
    optimizer,
    aux_apply_fn,         # model.apply bound to auxiliary_heads method
    carries_np,           # (T, N, hidden_dim) numpy — full rollout carries
    actions_np,           # (T, N) numpy int    — full rollout actions
    obs_np,               # (T, N, obs_dim) numpy — for loc_env_{t+1} target
    loc_env_start: int,
    loc_env_end: int,
    alive_np=None,        # (T, N) numpy float, or None
    key: jax.Array = None,
    minibatch_size: int = 1024,
    fwd_coef: float = 0.05,
    carry_fwd_coef: float = 0.05,
    self_pred_coef: float = 0.1,
    conf_coef: float = 0.0,
    energy_np: np.ndarray = None,
    proprio_coef: float = 0.0,
) -> Tuple[Dict, Any, float, float, float, float, float, float, float]:
    """
    Compute forward dynamics + latent carry dynamics + self-prediction aux losses.

    Forward dynamics target: obs[t+1, loc_env_start:loc_env_end] (flat loc_env).
    Carry dynamics target: carries[t+1] with stop_gradient (Phase 11 / 9.2).

    Returns: (params, opt_state, avg_fwd_loss, avg_carry_fwd_loss, avg_sp_loss, avg_sp_acc,
              avg_conf_loss, avg_conf_pred, avg_proprio_loss)
    """
    T, N, hidden_dim = carries_np.shape

    carry_t = carries_np[:-1].reshape((T - 1) * N, hidden_dim)
    carry_tp1 = carries_np[1:].reshape((T - 1) * N, hidden_dim)
    action_t = actions_np[:-1].reshape((T - 1) * N)
    action_tp1 = actions_np[1:].reshape((T - 1) * N)
    loc_env_tp1 = obs_np[1:].reshape((T - 1) * N, obs_np.shape[-1])[:, loc_env_start:loc_env_end]
    if energy_np is not None:
        energy_tp1 = np.asarray(energy_np[1:]).reshape((T - 1) * N).astype(np.float32)
    else:
        energy_tp1 = np.zeros((T - 1) * N, dtype=np.float32)
    action_oh = np.eye(5, dtype=np.float32)[action_t]
    action_tp1_oh = np.eye(5, dtype=np.float32)[action_tp1]

    if alive_np is not None:
        alive_t = alive_np[:-1].reshape((T - 1) * N).astype(np.float32)
    else:
        alive_t = np.ones((T - 1) * N, dtype=np.float32)

    M = carry_t.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)
    rng = np.random.RandomState(int(jax.random.bits(key, dtype=jnp.uint32)))
    perm = rng.permutation(M)
    n_mb = max(1, M // minibatch_size)

    fwd_sum = carry_fwd_sum = sp_loss_sum = sp_acc_sum = conf_sum = conf_pred_sum = proprio_sum = 0.0

    for i in range(n_mb):
        idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
        params, opt_state, fwd_l, carry_fwd_l, sp_l, sp_a, conf_l, conf_p, proprio_l = _aux_minibatch_step(
            params, opt_state, aux_apply_fn, optimizer,
            jnp.array(carry_t[idx]),
            jnp.array(action_oh[idx]),
            jnp.array(loc_env_tp1[idx]),
            jnp.array(carry_tp1[idx]),
            jnp.array(action_tp1_oh[idx]),
            jnp.array(energy_tp1[idx]),
            jnp.array(alive_t[idx]),
            fwd_coef, carry_fwd_coef, self_pred_coef, conf_coef, proprio_coef,
        )
        fwd_sum += float(fwd_l)
        carry_fwd_sum += float(carry_fwd_l)
        sp_loss_sum += float(sp_l)
        sp_acc_sum += float(sp_a)
        conf_sum += float(conf_l)
        conf_pred_sum += float(conf_p)
        proprio_sum += float(proprio_l)

    return (
        params,
        opt_state,
        fwd_sum / n_mb,
        carry_fwd_sum / n_mb,
        sp_loss_sum / n_mb,
        sp_acc_sum / n_mb,
        conf_sum / n_mb,
        conf_pred_sum / n_mb,
        proprio_sum / n_mb,
    )


@functools.partial(
    jax.jit,
    static_argnames=("red_aux_apply_fn", "optimizer", "proprio_coef", "retention_coef"),
)
def _red_minibatch_step(
    params,
    opt_state,
    red_aux_apply_fn,
    optimizer,
    carry_t,
    obs_seq,
    energy_tp1,
    nb_sigs_target,
    alive_mask,
    proprio_coef,
    retention_coef,
):
    """Red predator minibatch step with SRL BPTT and Proprio (Phase 15.4)."""

    def loss_fn(p):
        energy_pred, retention_pred = red_aux_apply_fn(p, carry_t, obs_seq)
        
        # Proprio (Energy) Loss
        energy_target = jax.lax.stop_gradient(energy_tp1)
        proprio_per = jnp.square(energy_pred - energy_target)
        proprio_loss = (proprio_per * alive_mask).sum() / (alive_mask.sum() + 1e-8)
        
        # SRL (Semantic Retention Loss)
        retention_target = jax.lax.stop_gradient(nb_sigs_target)
        retention_per = jnp.square(retention_pred - retention_target).sum(axis=-1)
        retention_loss = (retention_per * alive_mask).sum() / (alive_mask.sum() + 1e-8)
        
        total_loss = proprio_coef * proprio_loss + retention_coef * retention_loss
        return total_loss, (proprio_loss, retention_loss)

    (_, (proprio_l, retention_l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, proprio_l, retention_l


def red_auxiliary_update(
    params,
    opt_state,
    optimizer,
    red_aux_apply_fn,
    carries_np: np.ndarray,
    obs_seq_np: np.ndarray,
    energy_np: np.ndarray,
    nb_sigs_target_np: np.ndarray,
    alive_np: np.ndarray = None,
    key: jax.Array = None,
    minibatch_size: int = 1024,
    proprio_coef: float = 0.05,
    retention_coef: float = 0.1,
) -> Tuple[Dict, Any, float, float]:
    """Energy and Semantic Retention predictions (Phase 15.4 BPTT)."""
    T_lag, N, hidden_dim = carries_np.shape
    carry_t = carries_np.reshape(T_lag * N, hidden_dim)
    
    # obs_seq_np shape: (T_lag, N, lag, obs_dim)
    # Transpose to (lag, T_lag*N, obs_dim) for lax.scan
    lag = obs_seq_np.shape[2]
    obs_dim = obs_seq_np.shape[3]
    obs_seq = obs_seq_np.transpose((2, 0, 1, 3)).reshape(lag, T_lag * N, obs_dim)
    
    energy_tp1 = np.asarray(energy_np).reshape(T_lag * N).astype(np.float32)
    nb_sigs_target = np.asarray(nb_sigs_target_np).reshape(T_lag * N, -1).astype(np.float32)
    if alive_np is not None:
        alive_t = alive_np.reshape(T_lag * N).astype(np.float32)
    else:
        alive_t = np.ones(T_lag * N, dtype=np.float32)

    M = carry_t.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)
    rng = np.random.RandomState(int(jax.random.bits(key, dtype=jnp.uint32)))
    perm = rng.permutation(M)
    n_mb = max(1, M // int(minibatch_size))
    proprio_sum = 0.0
    retention_sum = 0.0
    for i in range(n_mb):
        idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
        params, opt_state, proprio_l, retention_l = _red_minibatch_step(
            params,
            opt_state,
            red_aux_apply_fn,
            optimizer,
            jnp.array(carry_t[idx]),
            jnp.array(obs_seq[:, idx, :]),
            jnp.array(energy_tp1[idx]),
            jnp.array(nb_sigs_target[idx]),
            jnp.array(alive_t[idx]),
            proprio_coef,
            retention_coef,
        )
        proprio_sum += float(proprio_l)
        retention_sum += float(retention_l)
    return params, opt_state, proprio_sum / n_mb, retention_sum / n_mb


# Keep old name as alias for backward compatibility with any external callers
def fwd_dynamics_update(params, opt_state, optimizer, fwd_apply_fn,
                        carries_np, actions_np, alive_np, key,
                        minibatch_size=1024, fwd_coef=0.05):
    """Deprecated: use auxiliary_update instead."""
    import functools as _ft
    raise NotImplementedError("fwd_dynamics_update requires obs_np and loc_env bounds; use auxiliary_update")
    return p, o, fl


# ── Phase 14.1b: VQEL monologue (reconstruction + information bottleneck) ───

VQEL_FROZEN_TOP_KEYS = frozenset({
    "head_action",
    "head_value",
    "head_fwd_dyn_1",
    "head_fwd_dyn_2",
    "head_fwd_1",
    "head_fwd_2",
    "head_self_pred",
    "head_confidence_1",
    "head_confidence_2",
    "head_symbol",
    "head_tom",
    "head_culture_fast",
    "head_culture_slow",
})


def _vqel_trainable_mask(params: Dict) -> Dict:
    """Zero gradients for policy / aux heads; allow transformer + VQ + monologue decoder."""
    def _mask_at_path(path, leaf):
        if path:
            top = path[0].key if hasattr(path[0], "key") else str(path[0])
            if top in VQEL_FROZEN_TOP_KEYS:
                return jnp.zeros_like(leaf)
        return jnp.ones_like(leaf)

    return jax.tree_util.tree_map_with_path(_mask_at_path, params)


def compute_vqel_losses(
    spatial_hat: jnp.ndarray,
    spatial_ego: jnp.ndarray,
    loss_vq: jnp.ndarray,
    z_e: jnp.ndarray,
    z_q: jnp.ndarray,
    alive: jnp.ndarray = None,
    recon_coef: float = 1.0,
    hash_penalty_coef: float = 0.5,
    vq_coef: float = 1.0,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Phase 14.1b combined monologue loss.

    L_recon: MSE(decode(z_q), stop_grad(spatial_ego))
    L_hash: commitment-style ||z_e - z_q||^2 (Voronoi squeeze), scaled by hash_penalty_coef
    L_vq:  standard VQ codebook + commitment from vector_quantize_signals
    """
    target = jax.lax.stop_gradient(spatial_ego)
    recon_per = jnp.mean(jnp.square(spatial_hat - target), axis=-1)

    z_q_sg = jax.lax.stop_gradient(z_q)
    hash_per = jnp.sum(jnp.square(z_e - z_q_sg), axis=-1)

    if alive is not None:
        mask = alive.astype(jnp.float32)
        denom = mask.sum() + 1e-8
        vqel_recon_mse = (recon_per * mask).sum() / denom
        vqel_hash_penalty = (hash_per * mask).sum() / denom
        vqel_vq_loss = (loss_vq * mask).sum() / denom
    else:
        denom = float(spatial_hat.shape[0])
        vqel_recon_mse = jnp.mean(recon_per)
        vqel_hash_penalty = jnp.mean(hash_per)
        vqel_vq_loss = jnp.mean(loss_vq)

    total = (
        recon_coef * vqel_recon_mse
        + hash_penalty_coef * vqel_hash_penalty
        + vq_coef * vqel_vq_loss
    )
    metrics = {
        "vqel_recon_mse": vqel_recon_mse,
        "vqel_hash_penalty": vqel_hash_penalty,
        "vqel_vq_loss": vqel_vq_loss,
        "vqel_total_loss": total,
    }
    return total, metrics


def vqel_monologue_loss(
    params: Dict,
    monologue_apply_fn: Any,
    carries: jnp.ndarray,
    obs: jnp.ndarray,
    n_layers: int,
    alive: jnp.ndarray = None,
    recon_coef: float = 1.0,
    hash_penalty_coef: float = 0.5,
    vq_coef: float = 1.0,
) -> Tuple[jnp.ndarray, Dict]:
    """Evaluate monologue forward + VQEL losses (differentiable)."""
    carries = jax.lax.stop_gradient(carries)
    obs = jax.lax.stop_gradient(obs)

    _z_q, _tok, spatial_hat, spatial_ego, loss_vq, z_e = monologue_apply_fn(
        params, carries, obs, n_layers
    )
    z_q = z_e + jax.lax.stop_gradient(_z_q - z_e)

    return compute_vqel_losses(
        spatial_hat,
        spatial_ego,
        loss_vq,
        z_e,
        z_q,
        alive=alive,
        recon_coef=recon_coef,
        hash_penalty_coef=hash_penalty_coef,
        vq_coef=vq_coef,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "monologue_apply_fn",
        "optimizer",
        "n_layers",
        "recon_coef",
        "hash_penalty_coef",
        "vq_coef",
    ),
)
def _vqel_monologue_minibatch_step(
    params,
    opt_state,
    monologue_apply_fn,
    optimizer,
    carries,
    obs,
    n_layers,
    alive,
    recon_coef,
    hash_penalty_coef,
    vq_coef,
):
    grad_fn = jax.value_and_grad(vqel_monologue_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        params,
        monologue_apply_fn,
        carries,
        obs,
        n_layers,
        alive=alive,
        recon_coef=recon_coef,
        hash_penalty_coef=hash_penalty_coef,
        vq_coef=vq_coef,
    )
    mask = _vqel_trainable_mask(params)
    grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, mask)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    metrics = dict(metrics)
    metrics["vqel_total_loss"] = loss
    return new_params, new_opt_state, metrics


def vqel_monologue_update(
    params: Dict,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    monologue_apply_fn: Any,
    obs_np: np.ndarray,
    carries_np: np.ndarray,
    n_layers: int,
    alive_np: np.ndarray = None,
    key: jax.Array = None,
    minibatch_size: int = 512,
    recon_coef: float = 1.0,
    hash_penalty_coef: float = 0.5,
    vq_coef: float = 1.0,
) -> Tuple[Dict, Any, Dict]:
    """
    Minibatched VQEL monologue update with masked gradients (policy heads frozen).

    Returns (new_params, new_opt_state, metrics) with vqel_recon_mse and vqel_hash_penalty.
    """
    T, N = obs_np.shape[:2]
    M = T * N
    hidden_dim = carries_np.shape[-1]

    flat_obs = np.asarray(obs_np.reshape(M, obs_np.shape[-1]))
    flat_carries = np.asarray(carries_np.reshape(M, hidden_dim))
    if alive_np is not None:
        flat_alive = np.asarray(alive_np.reshape(M)).astype(np.float32)
    else:
        flat_alive = np.ones(M, dtype=np.float32)

    if key is None:
        key = jax.random.PRNGKey(0)
    rng = np.random.RandomState(int(jax.random.bits(key, dtype=jnp.uint32)))
    perm = rng.permutation(M)
    minibatch_size = int(minibatch_size)
    n_mb = max(1, M // minibatch_size)

    metric_sums: Dict[str, float] = {}
    for i in range(n_mb):
        idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
        params, opt_state, mb_mets = _vqel_monologue_minibatch_step(
            params,
            opt_state,
            monologue_apply_fn,
            optimizer,
            jnp.array(flat_carries[idx]),
            jnp.array(flat_obs[idx]),
            n_layers,
            jnp.array(flat_alive[idx]),
            recon_coef,
            hash_penalty_coef,
            vq_coef,
        )
        for k, v in mb_mets.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

    metrics = {k: v / n_mb for k, v in metric_sums.items()}
    return params, opt_state, metrics
