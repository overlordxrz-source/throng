"""
agents/rl.py — MAPPO rollout buffer, GAE, and PPO update step.

Key design decisions:
  - Loss function is jit(value_and_grad(f)) — gradients INSIDE the JIT boundary.
  - Cached per (model, n_layers) via lru_cache so recompilation only happens on brain growth.
  - Optimizer chain (clip + adam) handles gradient clipping; no double-clipping.
  - Full-batch minibatches only (skip partial last batch) to keep shapes static.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Collects T steps × N agents of (obs, action, log_prob, value, reward, done, alive, warmup_ok).
    warmup_ok: 1.0 for agents whose age >= ppo_warmup_steps, 0.0 otherwise.
    Policy/entropy loss is zeroed for cold-started (floor-injected) agents;
    value loss still learns from them so the critic stays calibrated.
    """

    def __init__(self, rollout_steps: int, max_pop: int, obs_dim: int,
                 neighbor_k: int = 0, hidden_dim: int = 0) -> None:
        self.T          = rollout_steps
        self.N          = max_pop
        self.hidden_dim = hidden_dim
        self._t         = 0

        self.obs        = np.zeros((rollout_steps, max_pop, obs_dim), dtype=np.float32)
        self.actions    = np.zeros((rollout_steps, max_pop),          dtype=np.int32)
        self.log_probs  = np.zeros((rollout_steps, max_pop),          dtype=np.float32)
        self.values     = np.zeros((rollout_steps, max_pop),          dtype=np.float32)
        self.rewards    = np.zeros((rollout_steps, max_pop),          dtype=np.float32)
        self.dones      = np.zeros((rollout_steps, max_pop),          dtype=np.float32)
        self.alive      = np.zeros((rollout_steps, max_pop),          dtype=np.float32)
        self.warmup_ok  = np.ones((rollout_steps, max_pop),           dtype=np.float32)
        self.tom_targets = (
            np.full((rollout_steps, max_pop, neighbor_k), -1, dtype=np.int64)
            if neighbor_k > 0 else None
        )
        # Recurrent state captured BEFORE forward at step t — i.e. the carry
        # that produced obs[t]'s logits. Used by recurrent PPO replay so that
        # the policy gradient is computed on the same (carry, obs) joint state
        # the rollout actually executed, not on a zero-carry surrogate.
        self.carries = (
            np.zeros((rollout_steps, max_pop, hidden_dim), dtype=np.float32)
            if hidden_dim > 0 else None
        )

    def push(
        self,
        obs:         np.ndarray,
        actions:     np.ndarray,
        log_probs:   np.ndarray,
        values:      np.ndarray,
        rewards:     np.ndarray,
        dones:       np.ndarray,
        alive:       np.ndarray,
        warmup_ok:   np.ndarray,   # 1.0 if agent age >= warmup_steps else 0.0
        tom_targets: np.ndarray = None,  # (N, K) int32 neighbour action targets
        carries:     np.ndarray = None,  # (N, hidden_dim) carry BEFORE forward
    ) -> None:
        t = self._t % self.T
        self.obs[t]        = obs
        self.actions[t]    = actions
        self.log_probs[t]  = log_probs
        self.values[t]     = values
        self.rewards[t]    = rewards
        self.dones[t]      = dones
        self.alive[t]      = alive
        self.warmup_ok[t]  = warmup_ok
        if self.tom_targets is not None and tom_targets is not None:
            self.tom_targets[t] = tom_targets
        if self.carries is not None and carries is not None:
            self.carries[t] = carries
        self._t           += 1

    @property
    def full(self) -> bool:
        return self._t >= self.T

    def reset(self) -> None:
        self._t = 0

    def get(self) -> Dict[str, np.ndarray]:
        d = {
            "obs":        self.obs.copy(),
            "actions":    self.actions.copy(),
            "log_probs":  self.log_probs.copy(),
            "values":     self.values.copy(),
            "rewards":    self.rewards.copy(),
            "dones":      self.dones.copy(),
            "alive":      self.alive.copy(),
            "warmup_ok":  self.warmup_ok.copy(),
        }
        if self.tom_targets is not None:
            d["tom_targets"] = self.tom_targets.copy()
        if self.carries is not None:
            d["carries"] = self.carries.copy()
        return d


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards:    np.ndarray,   # (T, N)
    values:     np.ndarray,   # (T, N)
    dones:      np.ndarray,   # (T, N)
    alive:      np.ndarray,   # (T, N)
    last_value: np.ndarray,   # (N,)
    gamma:      float,
    lam:        float,
) -> Tuple[np.ndarray, np.ndarray]:
    T, N       = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae        = np.zeros(N,       dtype=np.float32)
    next_val   = last_value.copy()
    next_done  = np.zeros(N,       dtype=np.float32)

    for t in reversed(range(T)):
        delta    = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        gae      = (delta + gamma * lam * (1.0 - next_done) * gae) * alive[t]
        advantages[t] = gae
        next_val  = values[t]
        next_done = dones[t]

    return advantages, advantages + values


# ── PPO loss (cached, jit+grad compiled once per brain depth) ─────────────────

@lru_cache(maxsize=8)
def _get_loss_fn(model, n_layers: int, clip: float, value_coef: float, entropy_coef: float):
    """
    Build and return jit(value_and_grad(ppo_loss)) for this brain config.
    Cached by (model, n_layers, ...) so recompilation only on brain growth.
    Gradients are computed INSIDE the JIT boundary for efficiency.
    """
    hidden = model.hidden_dim

    def ppo_loss(
        params,
        obs:           jnp.ndarray,   # (B, obs_dim)
        actions:       jnp.ndarray,   # (B,) int32
        old_log_probs: jnp.ndarray,   # (B,)
        advantages:    jnp.ndarray,   # (B,)
        returns:       jnp.ndarray,   # (B,)
        alive_mask:    jnp.ndarray,   # (B,) float32
        warmup_mask:   jnp.ndarray,   # (B,) float32 — 0 for cold-started agents
    ):
        B = obs.shape[0]
        dummy_carry = jnp.zeros((B, hidden))

        def _agent_fwd(carry, x):
            _, (logits, _sig, _sym, val) = model.apply(params, carry, x, n_layers)
            return logits, val

        logits, values = jax.vmap(_agent_fwd)(dummy_carry, obs)   # (B,5), (B,)

        log_probs_all = jax.nn.log_softmax(logits, axis=-1)       # (B, 5)
        act_lp        = log_probs_all[jnp.arange(B), actions]     # (B,)

        ratio     = jnp.exp(act_lp - old_log_probs)
        adv_norm  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg1       = -adv_norm * ratio
        pg2       = -adv_norm * jnp.clip(ratio, 1.0 - clip, 1.0 + clip)

        # policy + entropy: only from warmed-up agents
        policy_mask = alive_mask * warmup_mask
        n_policy  = policy_mask.sum() + 1e-8
        n_alive   = alive_mask.sum() + 1e-8
        pg_loss   = (jnp.where(policy_mask, jnp.maximum(pg1, pg2), 0.0)).sum() / n_policy
        vf_loss   = (jnp.where(alive_mask,  (values - returns) ** 2, 0.0)).sum() / n_alive

        probs     = jax.nn.softmax(logits, axis=-1)
        entropy   = -(probs * log_probs_all).sum(axis=-1)
        ent_loss  = -(jnp.where(policy_mask, entropy, 0.0)).sum() / n_policy

        total = pg_loss + value_coef * vf_loss + entropy_coef * ent_loss
        return total, {"pg": pg_loss, "vf": vf_loss, "ent": -ent_loss}

    return jax.jit(jax.value_and_grad(ppo_loss, has_aux=True))


# ── PPO update step ───────────────────────────────────────────────────────────

def ppo_update_step(
    params,
    opt_state,
    optimizer:    optax.GradientTransformation,
    buffer_data:  Dict[str, np.ndarray],
    last_value:   np.ndarray,                   # (N,) bootstrap values
    model,
    n_layers:     int,
    config:       Dict,
    rng:          np.random.Generator,
) -> Tuple:
    gamma     = float(config["ppo_gamma"])
    lam       = float(config["ppo_gae_lam"])
    clip      = float(config["ppo_clip"])
    epochs    = int(config["ppo_epochs"])
    mb_size   = int(config["ppo_minibatch_size"])
    val_coef  = float(config["ppo_value_coef"])
    ent_coef  = float(config.get("action_entropy_coef", config["ppo_entropy_coef"]))

    rewards = buffer_data["rewards"]
    values  = buffer_data["values"]
    dones   = buffer_data["dones"]
    alive   = buffer_data["alive"]

    advantages, returns = compute_gae(
        rewards, values, dones, alive, last_value, gamma, lam
    )

    T, N = rewards.shape
    flat_obs      = buffer_data["obs"].reshape(T * N, -1)
    flat_act      = buffer_data["actions"].reshape(T * N)
    flat_lp       = buffer_data["log_probs"].reshape(T * N)
    flat_adv      = advantages.reshape(T * N)
    flat_ret      = returns.reshape(T * N)
    flat_alive    = alive.reshape(T * N)
    flat_warmup   = buffer_data.get(
        "warmup_ok", np.ones((T, N), dtype=np.float32)
    ).reshape(T * N)

    max_norm  = float(config["ppo_max_grad_norm"])
    loss_fn   = _get_loss_fn(model, n_layers, clip, val_coef, ent_coef)
    total_n   = T * N

    pg_acc = vf_acc = ent_acc = n_updates = clip_count = 0.0

    for _epoch in range(epochs):
        perm = rng.permutation(total_n)
        for start in range(0, total_n - mb_size + 1, mb_size):
            mb = perm[start : start + mb_size]

            obs_j     = jnp.array(flat_obs[mb])
            act_j     = jnp.array(flat_act[mb])
            lp_j      = jnp.array(flat_lp[mb])
            adv_j     = jnp.array(flat_adv[mb])
            ret_j     = jnp.array(flat_ret[mb])
            alive_j   = jnp.array(flat_alive[mb])
            warmup_j  = jnp.array(flat_warmup[mb])

            (_, aux), grads = loss_fn(
                params, obs_j, act_j, lp_j, adv_j, ret_j, alive_j, warmup_j
            )

            grad_norm = float(optax.global_norm(grads))
            clip_count += float(grad_norm > max_norm)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            pg_acc  += float(aux["pg"])
            vf_acc  += float(aux["vf"])
            ent_acc += float(aux["ent"])
            n_updates += 1

    n = max(n_updates, 1)
    stats = {
        "ppo_pg_loss":   pg_acc  / n,
        "ppo_vf_loss":   vf_acc  / n,
        "ppo_entropy":   ent_acc / n,
        "ppo_updates":   int(n_updates),
        "ppo_clip_frac": clip_count / n,   # fraction of updates that hit the clip
    }
    return params, opt_state, stats
