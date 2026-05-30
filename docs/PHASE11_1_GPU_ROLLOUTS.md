# Phase 11.1 — GPU-Resident Rollouts (staging)

**Branch:** `feature/phase11-1-gpu-rollouts`  
**Status:** Staging — **do not merge to `master` until Phase 11.0 `carry_fwd` ↓ ~0.05**

## Problem

`8077a12` offloads rollout tensors to CPU before PPO to fit A100 15GB VRAM. On B200 (192GB), PPO logs `H2D + backward...` ~40s while `lax.scan` ~17s — PCIe bottleneck caps ~6 steps/sec.

## Target

Keep `(T=512, N, …)` rollout batch **on device** through GAE + PPO minibatches. Expect **15+ steps/sec** on B200.

## Files to change

| File | Change |
|------|--------|
| `jax_sim/rl_jax.py` | Remove CPU numpy flatten/offload in `ppo_update`; keep arrays as JAX on GPU |
| `jax_sim/main_jax.py` | Stop `np.asarray` carry save if aux can read device arrays; optional config flag |
| `config_phase7.yaml` | `gpu_resident_rollouts: true` (branch only until merge) |

## Merge gate

1. Phase 11.0 training shows **`carry_fwd` ~0.05–0.1** on dashboard
2. User/Cam approve merge
3. Verify no OOM at `MEM_FRACTION=0.80` on B200 with full rollout on device

## Rollback

Re-enable CPU offload path if OOM — keep both paths behind config flag during staging.
