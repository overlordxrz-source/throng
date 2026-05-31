# Phase 12.0 — Adversarial Co-Evolution (Red Comms)

**Branch:** `feature/phase12-red-coevolution`  
**Flag:** `phase12_coevolution.red_comms_enabled` (default **false**)

## Goal

Give Red agents the same **communication substrate** as Blue (32-d wire + 64-code VQ + neighbor cross-attention) on a **separate** `PredatorNetworkJax` and `red_codebook`, so catch pressure can evolve coordinated trap language without contaminating Blue’s codebook.

## Architecture

| Component | Blue | Red (Phase 12) |
|-----------|------|----------------|
| Module | `AgentNetworkJax` | `PredatorNetworkJax` |
| `hidden_dim` | 256 | 128 (`red_hidden_dim`) |
| Codebook | `codebook` | `red_codebook` |
| Cross-attn | `nb_cross_attn` | `red_nb_cross_attn` |
| Aux / P11 | carry_fwd, conf, imagination | **stripped** |
| PPO aux update | on | **off** (catch + VQ only) |

## Observations

`build_observations_jax(r_pop, …)` → `nb_sigs` from **K nearest alive reds** (same flat layout as blue). Blues do not read red signals in this step.

## Checkpoint policy

Resuming a **Phase 11.3** checkpoint:

- **`b_params`**: load from Orbax (full blue stack + graft).
- **`r_params`**: **always fresh** `init_predator_params` — legacy `r_params` in ckpt were a blue clone and must **not** be loaded.

Restore uses `items={"b_params": …}` only when `red_comms_enabled`.

## Config

```yaml
phase12_coevolution:
  red_comms_enabled: false
  red_cross_attn_enabled: true
  red_vocab_size: 64
```

## Dashboard

When enabled:

- `Actions (red):` — macro movement distribution (pincer / trap signature).
- `RedVQ: loss=… | red_codes_active=X/64 | red_entropy=…`

## Gradient path (no comm shaping)

`reward_red_catch` → Red PPO → policy/value heads → `red_nb_cross_attn` → neighbor `emb_nb(nb_sigs)` → prior step red `signal_out` (STE VQ) → `red_codebook` / `head_signal`. No extra `stop_gradient` on the signal path (value head may still detach pooled when `detach_value=True`; rollout PPO uses `detach_value=False`).

## Launch (new run only)

Do **not** flip on the live P11.3 production job. Start a **new** Modal run from this branch with `red_comms_enabled: true` after Cam approves.
