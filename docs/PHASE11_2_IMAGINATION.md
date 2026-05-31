# Phase 11.2 — K-Step Imagination (Inference)

**Branch:** `feature/phase11-2-imagination`  
**Status:** Staging — **not on `master`** until throughput benchmark passes.

## What this adds

At **blue action selection** (inside `sim_step`, outside PPO update):

1. For each candidate action `a ∈ {STAY, N, S, E, W}`:
2. Roll carry forward **K=5** steps with **frozen** `head_fwd_dyn_1/2`:
   `carry ← f_dyn(carry, onehot(a))` (same `a` each imagined step)
3. Score trajectory: `Σ_{k=0}^{K-1} γ^k · head_value(carry_k)` with **γ=0.999**
4. Execute `argmax_a` score (alive agents only)

**Not changed:** PPO loop, rewards, VQ, ecology, aux training (`head_fwd_dyn` still trained via `auxiliary_update`).

## Files

| File | Change |
|------|--------|
| `jax_sim/imagination_jax.py` | JIT kernel: vectorized over 5 actions × K scan × N agents |
| `jax_sim/network_jax.py` | `value_from_carry()` for latent value readout |
| `jax_sim/main_jax.py` | Wire imagination at blue action pick; dashboard metrics |
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
| `imagination_agree` | % steps imagination picks same action as `argmax(logits)` |

## Param names (checkpoint)

Frozen at inference — must exist in restored checkpoint:

- `head_fwd_dyn_1`, `head_fwd_dyn_2` — carry rollforward
- `head_value` — latent value on imagined carries

Verified via `AUX_HEAD_KEYS` in `network_jax.py`.

## Merge gate

1. **Throughput:** ≥ **2 steps/sec** on B200 with `imagination_enabled: true` (baseline ~6 without)
2. First compile of `imagine()` may take minutes — steady-state matters
3. If &lt;2 steps/sec: vectorize further or reduce K / agent subset before merge

## Rollback

Set `imagination_enabled: false` — zero behavior change vs `master`.
