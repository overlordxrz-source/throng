# Phase 9 Canvas — Cross-Attention Receiver (9.4)

**Branch:** `feature/phase9-canvas` (from `master` 150k baseline)

## Goal

Cure solipsistic world-models by giving the network a dedicated **Other** pathway: neighbor proto-language signals are read via cross-attention from the agent's self+carry query, instead of only flat per-neighbor tokens.

## Architecture

| Role | Tensor |
|------|--------|
| **Query** | `LayerNorm(emb_own(own_state) + carry)` → `(N, 1, d)` |
| **Key/Value** | `emb_nb(nb_signals)` → `(N, K, d)` |
| **Output** | `processed_signals = self_encoded + attention_out` → `(N, d)` |
| **Transformer** | Single **Other** token `(N, 1, d)` when enabled; else legacy `(N, K, d)` |

Module: `NeighborCrossAttention` in `jax_sim/network_jax.py`.

## Config (`config_phase7.yaml`)

```yaml
phase9_canvas:
  cross_attn_enabled: false   # default off — new params; enable to compile 9.4 path
  cross_attn_num_heads: 4
```

## Status

- [x] Flax module + residual cross-attention scaffold

## Phase 9.1 — Confidence head (staged)

| Item | Detail |
|------|--------|
| **Input** | `[carry_t, action_t]` (same projection as `head_fwd_dyn`) |
| **Output** | `conf_pred` — scalar per agent (softplus, ≥ 0) |
| **Target** | Per-agent carry MSE = `mean((carry_pred - carry_tp1)²)` with **`stop_gradient`** on target |
| **Loss** | `confidence_coef * MSE(conf_pred, target)` — does not backprop into carry dynamics via target |
| **Dashboard** | `conf_loss`, `conf_pred` on AuxLoss line when `confidence_enabled: true` |

```yaml
phase9_canvas:
  confidence_enabled: false   # set true after pull on Modal
  confidence_coef: 0.05
```
- [x] Checkpoint graft — `graft_missing_param_subtrees` + `ensure_aux_head_params` for `nb_cross_attn`
- [x] **9.1 Confidence head** — `head_confidence_1/2` predicts carry_fwd MSE; `conf_loss` / `conf_pred` on dashboard
- [ ] Enable `confidence_enabled: true` after P9.4 run + `git pull` (grafts new heads)
- [ ] Full training run with `cross_attn_enabled: true` on ckpt **390**
- [ ] Resume from 150k/390 with `cross_attn_enabled: true`
