# Phase 11.2 — K-Step Imagination (Active Override)

**Branch:** `feature/phase11-2-imagination`  
**Status:** Active override — imagined action drives env step + PPO buffer.

## What this adds

Each rollout step for **blues**:

1. Forward pass → `action_logits`, pre-action `carries`
2. For each candidate action `a ∈ {STAY, N, S, E, W}`: K-step frozen `head_fwd_dyn` rollforward
3. Score `Σ γ^k · head_value(carry_k)`; **`b_actions = argmax`**
4. Environment step + rollout buffer use **`b_actions`**
5. **`log_probs_taken = log π(b_actions | logits)`** for PPO actor loss on executed actions

**Reds:** unchanged (`categorical(logits)`).

**Not changed:** PPO update loop structure, rewards, VQ, ecology, aux training (`head_fwd_dyn` still trained via `auxiliary_update`).

## Science questions

- Does the converged world model produce imagined trajectories that **agree** with the greedy policy?
- Does higher **`imagination_gain`** correlate with survival after active override?

## Files

| File | Change |
|------|--------|
| `jax_sim/imagination_jax.py` | JIT kernel: `(actions, gain, agree)` |
| `jax_sim/network_jax.py` | `value_from_carry()` for latent value readout |
| `jax_sim/main_jax.py` | Imagination override when `imagination_enabled` |
| `config_phase7.yaml` | `imagination_enabled`, `imagination_k`, `imagination_gamma` |

## Config

```yaml
imagination_enabled: true   # feature branch: overrides blue actions
imagination_k: 5
imagination_gamma: 0.999
```

## Dashboard

```text
Imagination: gain=0.0123 | agree=67.3% (vs greedy; K=5)
```

| Metric | Meaning |
|--------|---------|
| `imagination_gain` | Mean imagined return − greedy-action imagined return (alive agents) |
| `imagination_agree` | % rollout-mean: imagined argmax == `argmax(logits)` |

## Observed metrics (200k extension, metrics-only era, B200)

| step | ppo | steps/sec | `imagination_gain` | `imagination_agree` |
|------|-----|-----------|-------------------|---------------------|
| 198144 | 387 | 5 | 0.2290 | 2.0% |
| 198656 | 388 | 5 | 0.1255 | 0.9% |
| 199168 | 389 | 5 | 0.2257 | 3.4% |
| 199680 | 390 | 5 | 0.0826 | 15.6% |

Decode @ 149504+ on `signal_corpus-2.jsonl` verified continuous comms; Cam authorized active override.

## Param names (checkpoint)

Must exist in restored checkpoint:

- `head_fwd_dyn_1`, `head_fwd_dyn_2` — carry rollforward
- `head_value` — latent value on imagined carries

## Merge gate

1. **Throughput:** ≥ **2 steps/sec** on B200 with `imagination_enabled: true`
2. **Metrics-only baseline:** **5 steps/sec** (steps 198144–199680)

## Rollback

Set `imagination_enabled: false` — blues revert to stochastic `categorical(logits)` like `master`.
