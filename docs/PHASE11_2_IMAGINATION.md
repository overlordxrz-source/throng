# Phase 11.2 — K-Step Imagination (Metrics Only)

**Branch:** `feature/phase11-2-imagination`  
**Status:** Staging — **not on `master`** until throughput benchmark passes.

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
| `imagination_agree` | % steps imagined argmax == `argmax(logits)` (greedy policy) |

## Param names (checkpoint)

Frozen for metrics pass — must exist in restored checkpoint:

- `head_fwd_dyn_1`, `head_fwd_dyn_2` — carry rollforward
- `head_value` — latent value on imagined carries

## Merge gate

1. **Throughput:** ≥ **2 steps/sec** on B200 with `imagination_enabled: true` (baseline ~6 without)
2. First compile of `imagine()` may take minutes — steady-state matters

## Rollback

Set `imagination_enabled: false` — zero behavior change vs `master`.
