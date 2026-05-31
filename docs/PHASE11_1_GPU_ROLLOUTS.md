# Phase 11.1 — GPU-Resident Rollouts

**Status:** Merged to `master` after Phase 11.0 `carry_fwd` converged (< 0.05).

## Change

Replaces `8077a12` CPU rollout offload in `ppo_update()` (`rl_jax.py`).

| Mode | `gpu_resident_rollouts` | Behavior |
|------|-------------------------|----------|
| **B200 default** | `true` | Rollout `(T=512, N, …)` stays on device; minibatch slice in-place |
| **A100 fallback** | `false` | Legacy CPU numpy offload + H2D per minibatch |

## Config

```yaml
gpu_resident_rollouts: true
```

## Startup log

```text
[JAX] Phase11.1 PPO: GPU-resident rollouts (no CPU offload / H2D)
```

PPO logs show `GPU-resident backward` instead of `H2D + backward`.

## Expected

Throughput **15+ steps/sec** on B200 (vs ~6 with H2D bottleneck).

## Rollback

Set `gpu_resident_rollouts: false` in `config_phase7.yaml` if OOM on smaller GPUs.
