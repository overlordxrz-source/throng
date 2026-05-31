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
- [ ] Confidence head (9.1) — epistemic uncertainty on `head_fwd_dyn`
- [ ] Full training / checkpoint merge policy for new params
- [ ] Resume from 150k/390 with `cross_attn_enabled: true`
