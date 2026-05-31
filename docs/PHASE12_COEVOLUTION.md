# Phase 12.0 — Adversarial Co-Evolution (Red Comms)

**Branch:** `feature/phase12-red-coevolution`  
**Flag:** `phase12_coevolution.red_comms_enabled` (default **true** on `feature/phase12-red-coevolution` — matches 128-d checkpoints)

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
  red_comms_enabled: true
  red_cross_attn_enabled: true
  red_vocab_size: 64
  red_corpus_enabled: true       # wiretap — signal_corpus_red.jsonl
  hunt_scout_range: 8
```

## Dashboard

When enabled:

- `Actions (red):` — macro movement distribution (pincer / trap signature).
- `RedVQ: loss=… | red_codes_active=X/64 | red_entropy=…`

## Gradient path (no comm shaping)

`reward_red_catch` → Red PPO → policy/value heads → `red_nb_cross_attn` → neighbor `emb_nb(nb_sigs)` → prior step red `signal_out` (STE VQ) → `red_codebook` / `head_signal`. No extra `stop_gradient` on the signal path (value head may still detach pooled when `detach_value=True`; rollout PPO uses `detach_value=False`).

## Phase 12.1 — Red corpus logging

**Flags:** `red_corpus_enabled: true` (requires `red_comms_enabled: true`)

| File | Path on Modal |
|------|----------------|
| Blue (unchanged) | `/mnt/throng-runs/signal_corpus.jsonl` |
| Red (new) | `/mnt/throng-runs/signal_corpus_red.jsonl` |

**Red record fields:** `team`, `sig`, `vq_token`, `action`, `hunter`, `blue_dist`, `blue_bear`, `nb_hunter_sig_lag1`, `nb_hunter_dist_lag1`, `nb_hunter_token_lag1`.

- **`hunter`** = `blue_dist <= hunt_scout_range` (default **8**, symmetric with `alarm_scout_range`).
- **Lag-1:** hunter reds at T−1 within Chebyshev range → listener red at T (pincer decode in 12.2).
- Red corpus runs **even when blue population is locally zero** (post-kill search vocabulary).

CPU-only post-rollout; no `sim_step` / XLA changes.

## Phase 12.2 — Red decode (`tools/decode_signals.py --red`)

After ~20k steps of red corpus:

```bash
python3 tools/decode_signals.py --red /mnt/throng-runs/signal_corpus_red.jsonl --min-step <12.1_restart> --k 16
```

**Red VQ pincer test (χ²):** tokens with mean emitter `blue_dist ≤ 2` (Chase) vs `> 5` (Search); compares receiver **lag-1** N/S/E/W pursuit mix among non-hunters with `nb_hunter_token_lag1`. Significant χ² → **spatial coordination** (conditional geometry), not scalar proximity alone.

### First decode @ ~30k wiretap (May 2026) — **FAIL (expected)**

| Test | Result |
|------|--------|
| Lag-1 omnibus LRT | **p = 0.9767** ❌ |
| Pincer χ² (Chase vs Search) | ❌ — receivers not shifting pursuit vectors |
| Verdict | **Babbling** — VQ alive, semantics absent; ~30k red comms steps too early |

**Holding pattern:** keep Modal training; re-decode at +20k–50k steps. **Do not branch Phase 13** until pincer χ² **p < 0.05**.

Blue decode unchanged: `python3 tools/decode_signals.py signal_corpus.jsonl`.

## Launch (new run only)

On this branch, **`red_comms_enabled: true`** and **`red_corpus_enabled: true`** by default (128-d `r_params` + `signal_corpus_red.jsonl` wiretap). No notebook `sed` required. Blue decode stays on `signal_corpus.jsonl` only.

**Epistemic gate (P12.1):** `phase9_canvas.confidence_multiplier: 1.0` — stateless batch-relative gate: `use_imagination = conf_pred < mean(conf|alive) * mult` (no EMA; scan carry unchanged).
