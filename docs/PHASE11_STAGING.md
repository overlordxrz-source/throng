# Phase 11 Staging — Latent Carry Forward Dynamics

**Branch:** `feature/phase11-imagination`  
**Status:** Staged only — **do not merge to `master` until P10.6 decode @ ≥100k env steps.**

## What this adds

Predict **`carry_{t+1}`** (256-d) from **`[carry_t, onehot(action_t)]`** via `head_fwd_dyn_1/2`, alongside existing:

- `head_fwd_1/2` → flat **loc_env** MSE (Phase 9.2 external world model)
- `head_self_pred` → own next-action CE (Phase 9.1)

## Critical implementation detail

Carry update in `network_jax.py`:

```python
new_carries = 0.9 * carries + 0.1 * pooled
```

Without **`jax.lax.stop_gradient(carry_{t+1})`** on the MSE target, the network trivially learns the identity / 0.9 mixing — not real dynamics.

## Files changed (feature branch)

| File | Change |
|------|--------|
| `jax_sim/network_jax.py` | `head_fwd_dyn_1/2`, `carry_forward_dynamics()`, extended `auxiliary_heads` |
| `jax_sim/rl_jax.py` | `carry_fwd_loss` in `_aux_minibatch_step` / `auxiliary_update` |
| `jax_sim/main_jax.py` | Dashboard: `carry_fwd`, `carry_rank`, `carry_H` |
| `config_phase7.yaml` | `carry_fwd_coef: 0.05` (branch only) |

## Dashboard (when enabled)

```text
AuxLoss: fwd_env=... | carry_fwd=... (↓0.05–0.1) | self_pred_acc=... | carry_rank=... | carry_H=...
```

**Success criterion (Phase 9 canvas):** `carry_fwd` drops from ~0.5 (random) toward **0.05–0.1** over 50k–200k steps. If plateau > 0.3, try `ppo_gamma: 0.995`.

## Merge gate

1. P10.6 run completes **≥100k** env steps on **`master`**
2. `decode_signals.py` passes scouts %, lag-1 LRT, VQ token χ²
3. User/Cam approve merge + optional fresh checkpoint lineage

## Next after carry fwd converges

- Phase 11.1: GPU-resident rollouts (drop CPU offload on B200)
- Phase 9.3: Dreamer imagination loop (K-step mental rollout using carry fwd head)
