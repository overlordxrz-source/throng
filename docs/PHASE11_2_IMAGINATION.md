# Phase 11.2 — K-Step Imagination (Metrics Only)

**Branch:** `feature/phase11-2-imagination`  
**Status:** **FROZEN (concluded May 2026).** Metrics-only at `181b98c`. Active override `6cf965a` reverted.

**Conclusion:** Active imagination achieves policy distillation (agree ~27%) but triggers solipsistic value-exploitation (Stay≈99%). Experiment concluded. Moving to Phase 9.

## What this adds

Each rollout step, **in parallel** with stochastic action sampling:

1. For each candidate action `a ∈ {STAY, N, S, E, W}`:
2. Roll carry forward **K=5** steps with **frozen** `head_fwd_dyn_1/2`:
   `carry ← f_dyn(carry, onehot(a))` (same `a` each imagined step)
3. Score trajectory: `Σ_{k=0}^{K-1} γ^k · head_value(carry_k)` with **γ=0.999**
4. Log **`imagination_gain`** and **`imagination_agree`** to dashboard

**Blues still execute `categorical(logits)` actions** — imagination does **not** override behavior. PPO log-probs match executed actions (on-policy).

**Not changed:** PPO loop, rewards, VQ, ecology, aux training (`head_fwd_dyn` still trained via `auxiliary_update`).

## Science questions (without corrupting training)

- Does the converged world model produce imagined trajectories that **agree** with the greedy policy?
- Does higher **`imagination_gain`** correlate with survival?

## Files

| File | Change |
|------|--------|
| `jax_sim/imagination_jax.py` | JIT kernel: metrics only `(gain, agree)` |
| `jax_sim/network_jax.py` | `value_from_carry()` for latent value readout |
| `jax_sim/main_jax.py` | Parallel metrics; stochastic action selection unchanged |
| `config_phase7.yaml` | `imagination_enabled`, `imagination_k`, `imagination_gamma` |

## Config

```yaml
imagination_enabled: true   # feature branch only until merge gate
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

## Observed metrics (200k extension, B200)

| step | ppo | steps/sec | `imagination_gain` | `imagination_agree` |
|------|-----|-----------|-------------------|---------------------|
| 198144 | 387 | 5 | 0.2290 | 2.0% |
| 198656 | 388 | 5 | 0.1255 | 0.9% |
| 199168 | 389 | 5 | 0.2257 | 3.4% |
| 199680 | 390 | 5 | 0.0826 | 15.6% |

Git: **`aebe131`**. Resume ckpt **292**. `carry_fwd` **0.0001–0.0002** in same window.

## Param names (checkpoint)

Frozen for metrics pass — must exist in restored checkpoint:

- `head_fwd_dyn_1`, `head_fwd_dyn_2` — carry rollforward
- `head_value` — latent value on imagined carries

## Merge gate

1. **Throughput:** ≥ **2 steps/sec** on B200 with `imagination_enabled: true`
2. **Observed (200k extension, steps 198144–199680):** **5 steps/sec**

## Rollback

Set `imagination_enabled: false` — zero behavior change vs `master`.
