# THRONG Architecture — Code-Grounded Reference

> **Purpose:** the single source of truth for *what the code actually does*, written from a source review of the JAX path. When the narrative prose in `THRONG.md` or a phase log disagrees with this file, **trust this file** (or re-verify against the code). Last verified: 2026-06-23 against `jax_sim/`, `tools/`, `communication/`.
>
> **Why this exists:** THRONG has been bitten repeatedly by drift between documentation and implementation (the "severance-class" bugs). This file is deliberately literal and cites `file:line` so future Cam/Will reboots can re-anchor in minutes.

---

## 0. One-paragraph orientation

THRONG trains a shared-policy population of **blue prey** against **red predators** on a 128×128 toroidal grid (`jax_sim/`, JAX/Flax). Blues are blind to distant predators and must survive on local perception + neighbor signals passed through a **discrete vector-quantized (VQ) communication bottleneck**. The scientific question is whether a *grounded, compositional proto-language* emerges from lethal selection pressure. The entry point is `train_entry.run_simulation()` → `main_jax._run_simulation_impl()`, which runs a `lax.scan` rollout (T=512) then a CPU-offloaded PPO update for each team. Config is the single `config.yaml`.

---

## 1. Pipeline

```
run_bg.py
  └─ scripts.modal_train.build_cfg()        # loads config.yaml, applies P10.5 hard-ceiling overrides
  └─ jax_sim.train_entry.run_simulation()   # import shim — evicts stale modules after git pull
       └─ jax_sim.main_jax._run_simulation_impl()
            ├─ lax.scan(sim_step, T=512)     # GPU rollout: perception → network → action → env step
            │    ├─ NeighborCrossAttention (9.4): K neighbor signals → 1 "Other" token
            │    └─ epistemic gate (12.1b): if conf_pred < batch-relative τ → K-step imagined action
            ├─ ppo_update(blue)              # CPU-offload (H2D) minibatch 512; per-group grad clip
            ├─ ppo_update(red)               # red VQ DECOUPLED (red_vq_loss_coef=0.0) since 18.6
            ├─ auxiliary_update(blue)        # fwd_env + carry_fwd + self-pred + confidence + proprio
            ├─ dead_code_reset               # replace unused VQ codes with live z_e slices
            └─ corpus flush + dashboard
```

| File | Role |
|------|------|
| `jax_sim/train_entry.py` | **Always import here** — evicts stale modules after `git pull` |
| `jax_sim/main_jax.py` | Training loop, ecology, dashboard, checkpoints, corpus (~2.5k lines) |
| `jax_sim/network_jax.py` | `AgentNetworkJax` (blue) + `PredatorNetworkJax` (red) + VQ |
| `jax_sim/imagination_jax.py` | K-step mental rollout for the epistemic gate |
| `jax_sim/rl_jax.py` | PPO loss, GAE, minibatch step, aux update |
| `jax_sim/observations_jax.py` | Obs builder (`red_sense_api=v2`) |
| `jax_sim/grid_jax.py` | Catches, resources, shelter, barrier physics |
| `jax_sim/population_jax.py` | Agent state, inventory, reproduction |
| `communication/analysis.py` | Corpus writers (`maybe_record`, `maybe_record_red`) — shared with legacy |
| `tools/causal_intervention.py` | **Live** ATE instrument (frozen-ckpt token swap) |
| `tools/decode_signals.py` | Offline corpus decode (MI, LRT, χ², PosDis, topsim, NPMI) |
| `agents/network_torch.py` | **Kept** only for `compute_obs_dim_torch`/`compute_fwd_env_dim` helpers used by tools |

---

## 2. Blue network — `AgentNetworkJax` (`network_jax.py`)

256-d, 4-layer pre-norm transformer (up to 6 blocks allocated). Token sequence (cross-attn on): own_state, the attended neighbor "Other" token, local symbols (5×5), local env (5×5), own signal, optional memory slots, cultural fast/slow grids → mean-pool → `final_norm` → `pooled`.

### Two pathways (a Global-Workspace split)
- **`h_policy` = pooled** → action, value, symbol, ToM, culture heads.
- **`h_comms` = `gwt_comms_1(ReLU(exteroceptive_obs))`** where `exteroceptive_obs = obs.at[:, :4].set(0.0)` zeroes age(0), mat(1), **energy(2)**, layers(3). The GWT router structurally blinds the *communication* heads to interoceptive/metabolic state, forcing exteroceptive grounding. `h_comms` → the 3 VQ slots + alarm.

### Carry / recurrence
Blue uses a **soft EMA**: `new_carries = 0.9*carries + 0.1*pooled`. (Note: `THRONG.md` references a "GRUCell pivot" — that landed on the **red** network's `carry_gru`, not blue. Verify before relying on blue having gated recurrent memory.)

### Output tuple (11 elements) — memorize this order
```
(action_logits, signal_out, symbol_write, values, tom_logits,
 token_ids, alarm_out, loss_vq, z_e, culture_fast, culture_slow)
   0             1           2            3       4
   5            6          7         8     9            10
```
`loss_vq` is at index **7** for blue (because `alarm_out` occupies 6). PPO reads it via `vq_loss_idx`.

### Heads
| Head | In → out | Notes |
|------|----------|-------|
| `head_action` | pooled → (N, n_actions) logits / temp 2 | 12 actions |
| `head_value` | pooled → (N,) | critic (stop-grad if `detach_value`) |
| `head_signal_slot0/1/2` + `codebook_0/1/2` | h_comms → VQ | 3 slots, widths 12/8/12, 64 codes each |
| `head_alarm` | h_comms → (N,2) | 1-bit alarm; **vestigial** (ATE=0 since 17.5) but kept |
| `head_proprio` | carry → energy | disentangles metabolic state (coef 0.15) |
| `head_fwd_dyn_1/2` | [carry, action_oh] → carry_{t+1} | world model; `carry_fwd≈0.0001` |
| `head_fwd_1/2` | [carry, action_oh] → loc_env_{t+1} | anti-cheat aux |
| `head_self_pred` | carry → next action | self-model (`self_pred_acc≈0.4`) |
| `head_confidence_1/2` | [carry, action_oh] → predicted carry-fwd MSE | drives epistemic gate |
| `head_symbol`, `head_culture_fast/slow` | pooled → grid writes | persistence substrate (Phase 19 hooks) |
| `head_tom` | pooled → (N,K,n_actions) | **vestigial — no ToM reward in the JAX loop** |
| `head_vqel_recon_1/2` | z_q → spatial ego | VQEL monologue decoder (optional) |
| `head_signal` (Dense 8) | — | **defined but never called** (Phase 18.1 bypass amputation) |

---

## 3. Red network — `PredatorNetworkJax` (`network_jax.py`)

128-d transformer, **real `carry_gru`** (not EMA). Has its own dual pathway (`h_policy` + GWT-masked `h_comms` with `red_nb_cross_attn`).

- **Bottleneck differs from blue:** **DCVQ** (4 subspaces, 64 codes each, STE) → **SimVQ** linear reparam `tanh(clip(W) @ z_q) * scale`. `token_ids` exports only subspace 0.
- **No alarm head, no fwd_dyn/confidence/self_pred/imagination.** It has `head_proprio` + `head_retention` (aux only).
- **Output tuple (10 elements, no alarm):**
```
(action_logits, signal_out, symbol_write, values, tom_logits,
 token_ids, loss_vq, z_e, culture_fast, culture_slow)
                          6        7
```
`loss_vq` is at index **6** for red. The blue/red index divergence (7 vs 6) is the canonical severance foot-gun — see §7.
- **Phase 18.6 status:** red VQ is **decoupled** (`red_vq_loss_coef: 0.0`). Red policy/value + proprio/retention aux still train on `reward_red_catch`; its comms channel is mute. `RedVQ(decoupled)` on the dashboard is diagnostic only (a collapsed/huge value is expected and harmless).

---

## 4. The communication bottleneck (the heart of the experiment)

### Wire layout — 40 dims, 32 effective
| Segment | Dims | Source |
|---------|------|--------|
| Continuous bypass | 8 | **Hard-zeroed** (Phase 18.1 amputation): `z_e_cont = jnp.zeros((N,8))` |
| Slot 0 | 12 | `head_signal_slot0` → `codebook_0` (64×12) |
| Slot 1 | 8 | `head_signal_slot1` → `codebook_1` (64×8) |
| Slot 2 | 12 | `head_signal_slot2` → `codebook_2` (64×12) |

`signal_out = concat([zeros_8, z_q_slot0, z_q_slot1, z_q_slot2])`. The first 8 dims always carry zero; **all information is in the 3 discrete slots.**

### VQ mechanics (`vector_quantize_signals`)
Per slot: distances to a stop-grad codebook → hard argmin token + straight-through estimator (`y = y_soft + stop_grad(y_hard - y_soft)`); broadcast uses `y @ stop_grad(codebook)` (frozen-codebook gradient). `loss_vq` = codebook term `‖sg(z_e) − z_q‖²` + commitment `β·‖z_e − sg(z_q)‖²` (β=0.25). Total `loss_vq` = sum over the 3 slots. **`dead_code_reset`** runs *post-rollout* (`dead_code_reset_codebook_params`), replacing unused codes with random live-agent `z_e` slices — this is what keeps `codes_active` high.

### Dialogue mode
`dialogue_signal_mode: hard` — receivers read hard `z_q` centroids from the codebook (VQEL graduated). `monologue_enabled: false`.

---

## 5. Cognition: world model + epistemic gate

- **World model:** `head_fwd_dyn` predicts `carry_{t+1}` (MSE vs stop-grad target); `value_from_carry` reads a value directly off carry without a full forward pass.
- **K-step imagination (`imagination_jax.py`):** rolls carry forward K=5 steps with frozen dynamics, scores discounted `value_from_carry`, picks the best action.
  - **Action coverage (knob):** `imagination_n_actions` (config `phase9_canvas`, default **5** = Stay + N/S/E/W). Historically hard-coded to 5, so the gate is **blind to Strike/Push/Guard and the tool actions**. Set to 12 to imagine the full action space. *Behavioural change — A/B before adopting.*
- **Epistemic gate (12.1b, stateless, batch-relative):** sample reactive action → `conf_pred` → `τ = mean(conf_pred|alive)·confidence_multiplier` → if `conf_pred < τ` use the imagined action, else reactive. Gating fires for ~55–60% of agents; each gated "think" costs `imagination_metabolic_delta·K` energy (Phase 13 metabolic tax).

---

## 6. Ecology (`grid_jax.py`, `main_jax.py`, `population_jax.py`)

| System | Mechanics |
|--------|-----------|
| **Movement** | Actions 1–4 = N/S/E/W (1 cell/step light-cone). 0=Stay. |
| **Combat** | 5=Strike (red catch logic). **6=Push / 7=Guard have NO env mechanics in `jax_sim/` — they are no-op moves.** |
| **Crafting (Phase 18)** | 9=PickUp (wood west `x<grid/2`, stone east; capacity 1), 10=Craft (adjacent agent w/ wood+stone → both get axe), 11=UseTool (2× resource gain with axe), 8=Build (+3 barrier HP, energy cost). Failed craft/use → **−0.20** futile penalty. |
| **Predation** | Chebyshev ≤ `red_catch_radius` (1); `red_catch_prob` 0.8 jitter; shelter blocks catches; Big-Green prey need ≥2 strikers after `coop_threshold_step`. |
| **Metabolism** | blue `energy_decay` 0.001/step; red `red_energy_decay` 0.0001 (asymmetry → apex predators); starvation < 0.1. |
| **Population** | blue `max_pop` 200 / `min_pop` 150; red locked at 250. Reproduction: `repro_energy_thresh` 0.95, cost 0.80. |
| **Selection signal** | `NB_GAIN↔surv` is `nan` while everyone survives at the cap — the recurring "no evolutionary pressure on signal benefit" failure mode. |

**Open-endedness: none implemented.** `build_cfg` is fully static; no POET / quality-diversity / environment-genome code exists in the JAX path. The roadmap's open-ended tier is greenfield (see `STRATEGIC_ROADMAP.md` §8).

---

## 7. Measurement instruments (what can actually be proven)

| Tool | Type | Metric | Pass bar |
|------|------|--------|----------|
| `tools/causal_intervention.py` | **Live** token swap on frozen ckpt | ΔP(action) on blind receivers, paired t-test | `p<0.05` AND `|Δ|>0.05`. Supports full 3-slot tokens. **This is the real ATE instrument.** |
| `tools/ate_swap_test.py` | Offline stratified | Δ flee-rate alert vs safe | 95% CI excludes 0. **Slot-0 only** — corpus collapses the lag-1 token to slot 0, so per-slot offline ATE is *not* measurable today. |
| `tools/decode_signals.py` | Offline | MI/Spearman, lag-1 LRT, χ² pincer, **PosDis** (real), **topographic similarity** (real), **NPMI** (real, needs `adj_*` fields) | per-test `p<0.05`. **TRE here is a Ridge-R² proxy**, not true tree-reconstruction error — caveat any report. |

**Corpus (`communication/analysis.py`):** blue `signal_corpus.jsonl` + red `signal_corpus_red.jsonl`, sampled `corpus_sample_frac` of alive agents every `corpus_every_n_steps`, fsync'd each PPO update. The blue writer *can* log a 3-slot `vq_token` list, but `decode_signals.load_corpus` collapses the lag-1 neighbor token to slot 0 — fixing this is the prerequisite for per-slot compositional ATE.

---

## 8. Known divergences, vestigial parts, and foot-guns

| Item | Reality | Action |
|------|---------|--------|
| **Blue/red `loss_vq` index** | 7 (blue) vs 6 (red) — divergent output tuples | Threaded via `vq_loss_idx`; long-term fix = named `NetworkOutputs` dataclass (roadmap H1) |
| **Imagination action coverage** | was hard-coded to 5 actions | now `imagination_n_actions` knob (default 5) |
| **`head_tom` / `tom_logits`** | computed, **no reward** | vestigial; leave for checkpoint compat, do not trust as ToM |
| **Push (6) / Guard (7)** | no env mechanics | no-op moves; any "trapping" is movement |
| **Blue carry** | EMA, not GRU | verify before relying on gated memory for blue |
| **`head_signal` (Dense 8)** | defined, never called | kept for checkpoint grafting; the 8-D bypass is zeroed |
| **`RedVQ(decoupled)` telemetry** | huge/collapsed value | expected & harmless since 18.6; not a training-health signal |
| **`NB_GAIN↔surv: nan`** | population at cap | recurring — no selection pressure on signal benefit while everyone lives |

**Standing discipline:** a learning signal that is not gradient-checked + telemetry-gated is assumed broken. Do not remove network heads casually — Orbax restore of the live ~1.25M-step checkpoint depends on the parameter tree shape; grafting is preferred over cold restart.

---

## 9. Operational quick reference

- **Launch:** `python -u run_bg.py` (config `config.yaml`). Hot-resume restores **weights only**; population/grid/curriculum start fresh.
- **Env:** `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, `JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache`.
- **Restart is the highest-memory moment** (recompiles both backward graphs); if a restart dies with a truncated/no-Python traceback, capture the real error by running to a logfile (`> train.log 2>&1`) and checking the exit code rather than the notebook cell.
- **Checkpoints:** Orbax on the Modal volume `/mnt/throng-runs/checkpoints/`; folder N ≈ PPO update N ≈ `N×512` env steps.
