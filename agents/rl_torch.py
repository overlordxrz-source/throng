"""
agents/rl_torch.py — PPO update using PyTorch (MPS/CUDA/CPU).

GAE advantage estimation: pure numpy (fast, no autograd needed).
PPO loss: torch autograd on MPS.

Warmup mask still supported but no longer required once inject_offspring
replaces inject_random_agent (Priority 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from agents.network_torch import TorchBrain


# ── GAE (stays in numpy — fast, no GPU needed) ───────────────────────────────

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
        delta     = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        gae       = (delta + gamma * lam * (1.0 - next_done) * gae) * alive[t]
        advantages[t] = gae
        next_val  = values[t]
        next_done = dones[t]

    return advantages, advantages + values


# ── PPO loss (torch, runs on MPS) ─────────────────────────────────────────────

def _ppo_loss_batch(
    brain:        "TorchBrain",
    carries:      torch.Tensor,      # (B, hidden_dim) — real recurrent state
    obs:          torch.Tensor,      # (B, obs_dim)
    actions:      torch.Tensor,      # (B,) int64
    old_log_probs: torch.Tensor,     # (B,)
    advantages:   torch.Tensor,      # (B,)
    returns:      torch.Tensor,      # (B,)
    alive_mask:   torch.Tensor,      # (B,) float32
    warmup_mask:  torch.Tensor,      # (B,) float32
    n_layers:     int,
    clip:         float,
    val_coef:     float,
    ent_coef:     float,
    sig_ent_coef: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    B = obs.shape[0]
    device = brain.device

    # Real carries from rollout (stateful PPO replay): the policy is
    # evaluated under the same recurrent context that produced the action.
    _, (logits, _, _, values, _, _, sig_logits) = brain.model(carries, obs, n_layers)

    log_probs_all = F.log_softmax(logits, dim=-1)               # (B, 5)
    act_lp        = log_probs_all[torch.arange(B, device=device), actions]

    ratio = torch.exp(act_lp - old_log_probs)

    # advantages already normalised over alive-only samples upstream
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)

    policy_mask = alive_mask * warmup_mask
    n_policy    = policy_mask.sum() + 1e-8
    n_alive     = alive_mask.sum() + 1e-8

    pg_loss  = (torch.where(policy_mask.bool(), torch.maximum(pg1, pg2),
                            torch.zeros_like(pg1))).sum() / n_policy
    vf_loss  = (torch.where(alive_mask.bool(), (values - returns) ** 2,
                            torch.zeros_like(values))).sum() / n_alive

    probs    = F.softmax(logits, dim=-1)
    entropy  = -(probs * log_probs_all).sum(dim=-1)
    ent_loss = -(torch.where(policy_mask.bool(), entropy,
                             torch.zeros_like(entropy))).sum() / n_policy

    # Signal head entropy bonus — prevent token monoculture
    sig_ent_loss = torch.tensor(0.0, device=device)
    if sig_ent_coef > 0 and sig_logits is not None:
        sig_probs      = F.softmax(sig_logits, dim=-1)
        sig_log_probs  = F.log_softmax(sig_logits, dim=-1)
        sig_entropy    = -(sig_probs * sig_log_probs).sum(dim=-1)   # (B,)
        sig_ent_loss   = -(torch.where(policy_mask.bool(), sig_entropy,
                                       torch.zeros_like(sig_entropy))).sum() / n_policy

    total = pg_loss + val_coef * vf_loss + ent_coef * ent_loss + sig_ent_coef * sig_ent_loss

    # Clip ratio stats for logging
    clip_count = ((ratio.detach() - 1.0).abs() > clip).float().mean()

    aux = {
        "pg":   pg_loss.detach(),
        "vf":   vf_loss.detach(),
        "ent":  (-ent_loss).detach(),
        "clip": clip_count.detach(),
        "sig_ent": (-sig_ent_loss).detach(),
    }
    return total, aux


def _tom_loss_batch(
    brain:       "TorchBrain",
    carries:     torch.Tensor,   # (B, hidden_dim)
    obs:         torch.Tensor,   # (B, obs_dim)
    tom_targets: torch.Tensor,   # (B, K) int64
    n_layers:    int,
) -> torch.Tensor:
    """Cross-entropy loss for the ToM prediction head.

    Uses the same real carries as the PPO loss so the predictor can rely on
    the recurrent context built up over previous steps.
    """
    B      = obs.shape[0]
    _, (_, _, _, _, tom_logits, _, _) = brain.model(carries, obs, n_layers)
    # tom_logits: (B, K, 5)
    K = tom_logits.shape[1]
    loss = F.cross_entropy(
        tom_logits.reshape(B * K, 5),
        tom_targets.reshape(B * K),
        ignore_index=-1,
        reduction="mean",
    )
    return loss, tom_logits.detach()


# ── PPO update step ───────────────────────────────────────────────────────────

def ppo_update_torch(
    brain:        "TorchBrain",
    buffer_data:  Dict[str, np.ndarray],
    last_value:   np.ndarray,   # (N,) bootstrap
    n_layers:     int,
    rng:          np.random.Generator,
) -> dict:
    config   = brain.config
    gamma    = float(config["ppo_gamma"])
    lam      = float(config["ppo_gae_lam"])
    clip     = float(config["ppo_clip"])
    epochs   = int(config["ppo_epochs"])
    mb_size  = int(config["ppo_minibatch_size"])
    val_coef = float(config["ppo_value_coef"])
    ent_coef     = float(config.get("action_entropy_coef", config["ppo_entropy_coef"]))
    sig_ent_coef = float(config.get("signal_entropy_coef", 0.0))
    max_norm     = float(config["ppo_max_grad_norm"])
    device       = brain.device

    rewards = buffer_data["rewards"]
    values  = buffer_data["values"]
    dones   = buffer_data["dones"]
    alive   = buffer_data["alive"]

    advantages, returns = compute_gae(
        rewards, values, dones, alive, last_value, gamma, lam
    )

    T, N = rewards.shape
    hidden_dim   = brain.model.hidden_dim
    flat_obs     = buffer_data["obs"].reshape(T * N, -1)
    flat_act     = buffer_data["actions"].reshape(T * N)
    flat_lp      = buffer_data["log_probs"].reshape(T * N)
    flat_adv     = advantages.reshape(T * N)
    flat_ret     = returns.reshape(T * N)
    flat_alive   = alive.reshape(T * N)
    flat_warmup  = buffer_data.get(
        "warmup_ok", np.ones((T, N), dtype=np.float32)
    ).reshape(T * N)
    flat_carry   = buffer_data.get("carries")
    if flat_carry is not None:
        flat_carry = flat_carry.reshape(T * N, hidden_dim)
    else:
        # Backward-compat: legacy buffer without carries falls back to zeros.
        flat_carry = np.zeros((T * N, hidden_dim), dtype=np.float32)

    total_n   = T * N
    n_updates = 0
    pg_acc_t = vf_acc_t = ent_acc_t = clip_acc_t = sig_ent_acc_t = tom_acc_t = None

    # Normalise advantages over alive samples only (MAPPO standard).
    # Including the ~92% dead-agent zeros in the normalisation heavily biases
    # the mean toward zero and inflates alive-agent norms ~4×, breaking pg sign.
    alive_mask_np = alive.reshape(T * N).astype(bool)
    if alive_mask_np.sum() > 1:
        _adv_alive = flat_adv[alive_mask_np]
        flat_adv   = (flat_adv - _adv_alive.mean()) / (_adv_alive.std() + 1e-8)

    flat_tom = buffer_data.get("tom_targets")  # (T, N, K) or None
    use_tom  = flat_tom is not None
    if use_tom:
        flat_tom = flat_tom.reshape(T * N, -1)  # (T*N, K)

    # ── Compact to alive-only rows before GPU transfer ─────────────────────────
    # With 30/400 alive the full (T*N) buffer is ~13x larger than needed.
    # Compacting gives 13x smaller GPU tensors and 13x more real gradient signal
    # per minibatch (each 2048-sample minibatch now contains 2048 alive samples
    # instead of ~154 alive + 1894 dead zeros).
    flat_obs    = flat_obs[alive_mask_np]
    flat_act    = flat_act[alive_mask_np]
    flat_lp     = flat_lp[alive_mask_np]
    flat_adv    = flat_adv[alive_mask_np]
    flat_ret    = flat_ret[alive_mask_np]
    flat_warmup = flat_warmup[alive_mask_np]
    flat_carry  = flat_carry[alive_mask_np]
    flat_alive  = np.ones(alive_mask_np.sum(), dtype=np.float32)  # all alive by construction
    if use_tom:
        flat_tom = flat_tom[alive_mask_np]  # (n_alive, K)
    total_n = int(alive_mask_np.sum())

    # Transfer compact alive-only buffer to GPU once
    g_obs    = torch.tensor(flat_obs,    dtype=torch.float32, device=device)
    g_act    = torch.tensor(flat_act,    dtype=torch.int64,   device=device)
    g_lp     = torch.tensor(flat_lp,     dtype=torch.float32, device=device)
    g_adv    = torch.tensor(flat_adv,    dtype=torch.float32, device=device)
    g_ret    = torch.tensor(flat_ret,    dtype=torch.float32, device=device)
    g_alive  = torch.tensor(flat_alive,  dtype=torch.float32, device=device)
    g_warmup = torch.tensor(flat_warmup, dtype=torch.float32, device=device)
    g_carry  = torch.tensor(flat_carry,  dtype=torch.float32, device=device)
    g_tom    = (torch.tensor(flat_tom, dtype=torch.int64, device=device)
                if use_tom else None)

    # Per-action ToM accuracy counters — GPU tensors to avoid .item() sync inside loop
    _tom_action_correct = torch.zeros(5, dtype=torch.int64, device=device)
    _tom_action_total   = torch.zeros(5, dtype=torch.int64, device=device)

    brain.model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(total_n, device=device)
        for start in range(0, total_n - mb_size + 1, mb_size):
            mb = perm[start: start + mb_size]

            obs_j    = g_obs[mb]
            act_j    = g_act[mb]
            lp_j     = g_lp[mb]
            adv_j    = g_adv[mb]
            ret_j    = g_ret[mb]
            alive_j  = g_alive[mb]
            warmup_j = g_warmup[mb]
            carry_j  = g_carry[mb]

            # ── Policy + value + entropy loss ─────────────────────────────
            brain.optimizer.zero_grad()
            loss, aux = _ppo_loss_batch(
                brain, carry_j, obs_j, act_j, lp_j, adv_j, ret_j,
                alive_j, warmup_j, n_layers, clip, val_coef, ent_coef,
                sig_ent_coef,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in brain.model.named_parameters()
                 if not n.startswith("tom_head")],
                max_norm,
            )
            brain.optimizer.step()

            # Accumulate stats as GPU tensors — no .item() sync inside loop
            if pg_acc_t is None:
                pg_acc_t       = aux["pg"].clone()
                vf_acc_t       = aux["vf"].clone()
                ent_acc_t      = aux["ent"].clone()
                clip_acc_t     = aux["clip"].clone()
                sig_ent_acc_t  = aux["sig_ent"].clone()
            else:
                pg_acc_t       += aux["pg"]
                vf_acc_t       += aux["vf"]
                ent_acc_t      += aux["ent"]
                clip_acc_t     += aux["clip"]
                sig_ent_acc_t  += aux["sig_ent"]
            n_updates += 1

            # ── ToM auxiliary loss (separate optimizer, lower lr) ───────────
            if use_tom:
                tom_j = g_tom[mb]   # (B, K)
                brain.tom_optimizer.zero_grad()
                tom_loss, _tom_logits = _tom_loss_batch(brain, carry_j, obs_j, tom_j, n_layers)
                tom_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    brain.model.tom_head.parameters(), max_norm
                )
                brain.tom_optimizer.step()
                if tom_acc_t is None:
                    tom_acc_t = tom_loss.detach().clone()
                else:
                    tom_acc_t += tom_loss.detach()

                # Per-action accuracy — reuse logits from _tom_loss_batch (no extra forward)
                K   = _tom_logits.shape[1]
                B_mb = _tom_logits.shape[0]
                _pred  = _tom_logits.argmax(dim=-1)          # (B, K)
                _tgt   = tom_j.reshape(B_mb, K)
                _valid = (_tgt >= 0)                          # mask out -1 padding
                _correct = (_pred == _tgt) & _valid
                for _a in range(5):
                    _mask = (_tgt == _a) & _valid
                    _tom_action_total[_a]   += _mask.sum()
                    _tom_action_correct[_a] += (_correct & _mask).sum()

    brain.model.eval()
    n = max(n_updates, 1)
    # Single GPU→CPU sync here — after all epochs complete
    result = {
        "ppo_pg_loss":   (pg_acc_t   / n).item() if pg_acc_t   is not None else 0.0,
        "ppo_vf_loss":   (vf_acc_t   / n).item() if vf_acc_t   is not None else 0.0,
        "ppo_entropy":   (ent_acc_t  / n).item() if ent_acc_t  is not None else 0.0,
        "ppo_updates":   int(n_updates),
        "ppo_clip_frac": (clip_acc_t / n).item() if clip_acc_t is not None else 0.0,
        "ppo_sig_ent":  (sig_ent_acc_t / n).item() if sig_ent_acc_t is not None else 0.0,
    }
    if use_tom:
        result["tom_loss"] = (tom_acc_t / n).item() if tom_acc_t is not None else 0.0
        _total   = _tom_action_total.cpu().numpy().astype(np.float64)
        _correct = _tom_action_correct.cpu().numpy().astype(np.float64)
        result["tom_acc_per_action"] = [
            float(_correct[a] / max(_total[a], 1)) for a in range(5)
        ]
    return result
