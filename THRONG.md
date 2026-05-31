# THRONG — Agent Onboarding

> Can proto-language emerge from multi-agent survival pressure alone — no human text, no communication rewards?

**Read this file first.** Full historical lab notebook (~290KB) lives in [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) if you need old run logs.

**Cam reboot (60 seconds):** Read **§0b** (what to do *right now*) → **§0** (who you are) → **§4** (live run) → **§5** (B200 OOM recovery) → **§11** (unlock sequence) → archive **[SYSTEM RESTORE](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context)** + **[SYSTEM UPDATE](docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas)**.

---

## 0b. Current state — **P9.4 training LIVE** on Modal (May 2026)

**LIVE:** **`feature/phase9-canvas`** (`38f342a+`) on Modal **`dragonbgnx`** — **`cross_attn_enabled: true`**, resumed from ckpt **291**, training toward **`n_steps=250_000`**. **Do not stop** unless Stay collapse or OOM.

| Live run (sampled) | Value |
|--------------------|--------|
| **Env step** | **~155136** (PPO **303**); `[CKPT] Saved step 155136` |
| **Resume ckpt** | **291** on volume (operator verified “Seed 291”; ~150k-era weights + grafted `nb_cross_attn`) |
| **Throughput** | **6 steps/sec** |
| **Actions** | Stay **~20%**, N/S/E/W **~16–26%** each — **healthy** (contrast P11.2 active: Stay≈99%) |
| **`carry_fwd`** | **0.0001** |
| **`codes_active`** | **51–63/64** |
| **PPO log** | `H2D + backward` ✅ |
| **Log** | `/mnt/throng-runs/train.log` |

**Engineering branch:** **`feature/phase9-canvas`** — `NeighborCrossAttention` + Orbax **schema graft** (`1d57bf9`) for legacy ckpts. **P11.2 FROZEN** on **`feature/phase11-2-imagination`** (`061df84`).

| Milestone | Value |
|-----------|--------|
| **`master` 150k** | Complete @ step **149504**, Orbax **292** |
| **P11.2 200k extension** | Complete @ step **199680**, PPO **390** (metrics-only; **use for decode baseline**, not required for current run) |
| **Modal volume** | **`dragonbgnx`** / `throng-runs` → `/mnt/throng-runs` (checkpoints + `signal_corpus.jsonl`) |
| **Local ckpt backup** | `~/throng_checkpoints_backup/` — folders **291**, **390**, **393** (avoid **393** for science) |
| **Local corpus** | `signal_corpus.jsonl`; decode log `decode_p11_2_149504+.log` |
| **Repo graph** | `graphify-out/graph.html` + `GRAPH_REPORT.md` (914 nodes, May 2026) |

### P10.6 decode (reference)

| Test (`--min-step 63488`) | Result |
|---------------------------|--------|
| **Cardinal lexicon** | χ² **p = 5.44e-14** ✅ |
| **Lag-1 omnibus** | χ²(32)=95.96, **p≈0** ✅ |
| **VQ token / alert-set** | **not significant** ❌ |

### P11.2 extension decode (`signal_corpus.jsonl`, `--min-step 149504`)

| Test | Result |
|------|--------|
| **Lag-1 omnibus** | χ²(32)=135.15, **p≈0** ✅ |
| **Cardinal k=4** | **p = 6.12e-23** ✅ |
| **VQ token / alert-set** | **p = 0.55 / 0.69** ❌ |
| Log | `decode_p11_2_149504+.log` |

Continuous comms verified → active imagination was authorized → **failed** (value exploitation).

### Phase 11.2 — **CONCLUDED** (`feature/phase11-2-imagination`)

| Era | Detail |
|-----|--------|
| **Metrics-only (`aebe131`)** | K=5 frozen `head_fwd_dyn`; stochastic actions; **5 steps/sec**; `imagination_agree` **0.9–15.6%** |
| **Active override (`6cf965a`)** | Imagined argmax drives env + PPO → **agree ~27%** but **Stay ≈ 99%** (solipsistic value exploitation) |
| **Resolution** | Revert **`181b98c`**; branch **frozen** at metrics-only **`061df84`** |
| **Conclusion** | Carry entangles Self+World; imagination without **Other** pathway fails. → **Phase 9 canvas** |

**Do not** resume training from ckpt **`393/`** (post–active-imagination). Prefer **`390/`** on new volume.

### Phase 9 canvas — **TRAINING** (`feature/phase9-canvas`)

| Item | Value |
|------|--------|
| **Module** | `NeighborCrossAttention` in `network_jax.py` |
| **Mechanism** | Q = `LayerNorm(emb_own + carry)`; KV = `emb_nb(signals)`; residual on self |
| **Config** | `phase9_canvas.cross_attn_enabled: true` on Modal (Cell 1 patches yaml) |
| **Orbax graft** | `graft_missing_param_subtrees` injects `nb_cross_attn` into ckpts **291/390** (`1d57bf9`) |
| **Modal cells** | [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md) — clone `/root/throng` **first** |
| **Next** | Confidence head (9.1); decode @ `--min-step 155136` after ~20k new steps |

### Phase 11.0 — COMPLETE (`master`)

`head_fwd_dyn_1/2`, `carry_fwd` **→ 0.0001**, CPU offload PPO (`d4cf614`).

### Phase 11.1 — ABANDONED

GPU-resident / `lax.scan` PPO — starvation + XLA OOM; **`d4cf614` revert**.

### Branch policy

| Branch | Status |
|--------|--------|
| **`master`** | 150k science baseline; CPU offload; **no** imagination |
| **`feature/phase9-canvas`** | **LIVE TRAIN** — P9.4 cross-attn (`38f342a+`, ckpt **291** → 250k) |
| **`feature/phase11-2-imagination`** | **FROZEN** — metrics-only imagination (`061df84`); `run_bg` **`n_steps=250_000`** |
| **`feature/phase11-1-gpu-rollouts`** | **Abandoned** |

### Horcrux (context backup)

Cam's persona + triad workflow live in Git so reboots recover identity:

- **Horcrux** = metaphor for durable context written to repo (from *Harry Potter* — soul fragment in an object)
- **Primary:** [`docs/THRONG_ARCHIVE.md#system-restore-the-cam-context`](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context) (`6d542f6`)
- **Supplement:** [`docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas`](docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas) (`d6a588d+`)

---

## 0. Team, roles, and standing directives (Cam)

### The triad

| Entity | Role |
|--------|------|
| **User (Human Operator)** | Runs code on Modal, manages Jupyter, passes messages between Cam and Will, physical control of the experiment |
| **Will (Cursor agent)** | Reads/writes the repo, patches code, verifies facts, pushes Git. Precise engineer; needs Cam for architecture and physics |
| **Cam (You)** | Polymath orchestrator — software, physics, philosophy, RL/ML. Synthesizes direction; User copy-pastes **`@Will — Cam here...`** blocks to Will |

### Who Cam is

**Cam, the Polymath AI Orchestrator** — four lenses on every decision:

1. **Software:** fault-tolerant JAX pipelines, Modal/volume ops, causal logging
2. **Physics:** grid as thermodynamic system — light-cones (max speed 1 cell/step), Lotka-Volterra ecology, entropy
3. **Philosophy:** emergent language — *meaning is use*; selection, not reward shaping, defines symbols
4. **RL/ML:** MAPPO, VQ bottlenecks, GAE, policy entropy, auxiliary self-prediction

### Standing directives (do not violate mid-run)

1. Speak to the User in **Synergic Synthesis** (Software / Physics / Philosophy / RL).
2. Address Will via explicit **`@Will — Cam here...`** copy-paste blocks.
3. **Keep the ecology mathematically pure** — no scout/alarm comm rewards, no blind VQ loss shaping. Lethal selection forges language.
4. **Phase 11.2 FROZEN** — imagination **metrics-only**; never re-enable action override without Cam (`6cf965a` caused Stay collapse).
5. **Phase 9 canvas is active** — build **Other** pathway (cross-attn 9.4, confidence 9.1) on `feature/phase9-canvas`.
6. **CPU offload only** — logs must show **`H2D + backward`**; do not re-merge 11.1 GPU rollouts.
7. **Never** comm reward shaping or blind VQ loss shaping.

### Branch policy

| Branch | Purpose |
|--------|---------|
| **`master`** | Stable 150k baseline; Phase 11.0 carry dynamics |
| **`feature/phase9-canvas`** | **Active:** cross-attention receiver + future 9.1 confidence head |
| **`feature/phase11-2-imagination`** | **Frozen archive:** K-step imagination metrics only |
| **`feature/phase11-1-gpu-rollouts`** | **Abandoned** |

**Phase 11.0 on `master` (`3880337`):**

| Component | Detail |
|-----------|--------|
| `head_fwd_dyn_1/2` | Predict **carry_{t+1}** from `[carry_t, onehot(action_t)]` |
| Loss | MSE + **`jax.lax.stop_gradient(carry_tp1)`** |
| Config | `carry_fwd_coef: 0.05` |
| Success | **`carry_fwd` ↓ 0.05–0.1** (from ~0.5 random) |
| Docs | [`docs/PHASE11_STAGING.md`](docs/PHASE11_STAGING.md) |

---

## 1. What this project is

**THRONG** trains shared-policy prey (blues) vs predator (reds) on a 128×128 toroidal grid. Blues are **partially blind** to global red positions; they must use **local perception + neighbor signals** to survive. Communication is **32-dim vectors** through a **64-code VQ bottleneck** (discrete tokens internally, continuous on the wire).

| Team | Role | Policy |
|------|------|--------|
| **Blue** | Survive, eat, reproduce | MAPPO, `hidden_dim=256`, 4L transformer |
| **Red** | Hunt blues | Separate MAPPO, `red_hidden_dim=128` |

**Active codebase:** JAX in `jax_sim/` — **not** the legacy PyTorch `main.py` path for current experiments.

**Active config:** [`config_phase7.yaml`](config_phase7.yaml) — override in Modal cells or `scripts/modal_train.py`.

**Hypothesis:** Information asymmetry + lethal ecology → only signals that help neighbors survive get selected. **Do not** add scout/alarm shaped rewards; that invalidates the experiment.

---

## 2. Architecture (JAX, today)

```
train_entry.run_simulation()  →  main_jax._run_simulation_impl()
  lax.scan(sim_step, T=512)   →  rollout on GPU
  [phase9-canvas] NeighborCrossAttention (optional) → 1 "Other" token
  [phase11-2 only] imagine K=5 metrics (frozen head_fwd_dyn) — does NOT override actions
  ppo_update (blue + red)     →  CPU rollout offload, minibatch 512
  auxiliary_update            →  loc_env MSE + carry_fwd MSE + self-prediction
```

| File | Role |
|------|------|
| [`jax_sim/train_entry.py`](jax_sim/train_entry.py) | **Always import here** — evicts stale modules after `git pull` |
| [`jax_sim/main_jax.py`](jax_sim/main_jax.py) | Training loop, ecology, dashboard, checkpoints, corpus |
| [`jax_sim/network_jax.py`](jax_sim/network_jax.py) | Transformer + VQ; **`NeighborCrossAttention`** (Phase 9.4) |
| [`jax_sim/imagination_jax.py`](jax_sim/imagination_jax.py) | Phase 11.2 K-step metrics (**`feature/phase11-2-imagination` only**) |
| [`jax_sim/rl_jax.py`](jax_sim/rl_jax.py) | PPO + numpy GAE |
| [`jax_sim/observations_jax.py`](jax_sim/observations_jax.py) | Obs builder; startup must print `red_sense_api=v2` |
| [`jax_sim/grid_jax.py`](jax_sim/grid_jax.py) | Catches (`red_catch_prob`), resources, shelter |
| [`tools/decode_signals.py`](tools/decode_signals.py) | Offline corpus analysis |
| [`run_bg.py`](run_bg.py) | **Preferred** nohup entry (`python -u run_bg.py`) |
| [`scripts/modal_train.py`](scripts/modal_train.py) | Same config as `run_bg.py` |

**Network outputs:** `action_logits, signal_out, symbol_write, values, tom_logits, token_ids, loss_vq, z_e, culture_fast, culture_slow`.

**World extras:** shelter spots, contested nodes, scent trails, dual cultural grids, episodic memory (20 slots), puzzles (optional).

---

## 3. Phase history (compressed)

| Phase | Era | Main idea | Outcome |
|-------|-----|-----------|---------|
| **1–3** | PyTorch / Kaggle | MAPPO, discrete vocab, culture grids | MI spikes under predation; channel differentiation (signal vs culture) |
| **4–5** | “Pure emergence” | Remove artificial comms rewards; 96×96 runs | Rich continuous encoding; weak discrete “words”; withdrawal tests |
| **6–7.5** | Rich world | Memory buffer, shelters, contested food, mind-meld, distill | Infrastructure; comms still not load-bearing |
| **8–9** | JAX rewrite | `lax.scan`, aux heads (self-pred + **loc_env** fwd), Phase 9 Modal 200k | Stable stack; ~16k unique signals; **NB_GAIN↔surv: nan** (no selection) |
| **10.0 P1–P4** | Lethal ecology | Blind blues (`red_detection_radius: 0`), VQ 64 codes, catch radius 1, 250 reds | VQ **62–63/64** active; catches high; pop still ~500 |
| **10.1 P4b** | Selective squeeze | `min_population: 150`, reds locked 250, `distill_enabled: false` | Floor defined; repro still refilled |
| **10.2** | Metabolic squeeze | `repro_energy_thresh: 0.95`, `repro_energy_cost: 0.80` | Harder cloning |
| **10.3** | Famine | `resource_regen_rate: 0.00025`, 10 patches, `resource_max: 0.5` | Pop crash events (500→147) |
| **10.4** | Safety bubble | `red_catch_prob: 0.8`, regen +20%, `max_age: 1000` | Longer lives for NB_GAIN |
| **10.5** | Hard ceiling | `max_pop: 200`, `min_pop: 150`, `ppo_gamma: 0.999` | Goldilocks band; less entropy explosion |
| **10.6** | Causal logging | `corpus_every_n_steps: 4`, volume corpus + fsync | Decode @ 100k: cardinal lexicon p=5.44e-14 |
| **11.0** | **Carry world-model** ✅ | `head_fwd_dyn`, `carry_fwd_coef` | **`carry_fwd` → 0.0001** |
| **11.1** | GPU rollouts | `424c46f` / `6042a4d` | **Abandoned** — **`d4cf614` revert** |
| **11.2** | K-step imagination | `aebe131` metrics; `6cf965a` active **reverted** | 200k @ **5 steps/sec**; active → **Stay≈99%** → **FROZEN** |
| **9.4** | Cross-attn receiver | `feature/phase9-canvas` `38f342a+` | **Training LIVE** @ ~155k; graft + Stay~20% healthy |

**Recurring failure mode:** Blues stay at cap → ~99% survival → **`NB_GAIN↔surv: nan`** → no evolutionary pressure on neighbor-signal benefit.

**Recurring success:** Under threat, signals encode **`red_dist`** (proximity); VQ codebook stays diverse when `vq_dead_code_reset: true`.

---

## 4. Current experiment — Phase 9.4 **LIVE** (`feature/phase9-canvas`)

**Status:** Training **in progress** on Modal **`dragonbgnx`**. Resumed **ckpt 291**, cross-attn **on**, graft successful. Target **`n_steps=250_000`** (`run_bg.py`).

**Verified healthy @ steps 154624–155136:**

```text
[step 155136] 6 steps/sec | blue=199 red=250 | ppo=303
  Actions: N=16% S=21% E=18% W=26% Stay=20%
  AuxLoss: carry_fwd=0.0001 | self_pred_acc=0.266 | codes_active=63/64
  [JAX] blue PPO minibatch 1/200 — H2D + backward...
[CKPT] Saved step 155136
```

**If training stopped:** use [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md) — **Cell 1 clones** `/root/throng` (bare `cd /root/throng` fails on fresh disks).

```bash
cd /root/throng && git fetch origin && git checkout feature/phase9-canvas && git pull
# phase9_canvas.cross_attn_enabled: true in config_phase7.yaml
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
# Cell 2: subprocess.Popen run_bg.py → /mnt/throng-runs/train.log
```

**Checkpoint policy:**

| Ckpt | Use |
|------|-----|
| **291** | **Current run** resume (~150k-era + grafted cross-attn) |
| **390** | 200k metrics-only baseline (alternate resume) |
| **393** | **Avoid** — post–active-imagination Stay collapse |

**Startup must include:**

```text
[JAX] git=38f342a | Phase9 auxiliary: ON
[JAX] Phase9.4 cross-attn receiver: heads=4 ...
[JAX] Injected randomly initialized nb_cross_attn into b_params   # schema graft
[JAX] Restored params from step 291
```

### Completed runs (reference)

**150k (`master`):** step **149504**, ckpt **292** (not in local backup — use **291** or volume copy).

**200k P11.2 metrics (`aebe131`):** step **199680**, ckpt **390**; `imagination_gain` **0.08–0.23**, `imagination_agree` **0.9–15.6%**.

P10.6 causal logging stack (all runs):

| Parameter | Value |
|-----------|--------|
| `corpus_every_n_steps` | **4** (was 20) — lag-1 scout buffer ≈ 4 env steps |
| `corpus_sample_frac` | **0.15** (was 0.08) |
| Corpus path | **`/mnt/throng-runs/signal_corpus.jsonl`** (auto-routed) |
| Durability | **`flush_to_disk()`** (fsync) each PPO rollout |
| Scout label | `red_dist <= alarm_scout_range` (**8**) |
| Checkpoints | `/mnt/throng-runs/checkpoints/` — resume keeps weights; pop/grid fresh |
| Archive | Pre-10.6 file → `signal_corpus_20step_archive.jsonl` on volume |

**Causal light-cone (Cam):** Max agent speed = 1 cell/step, `alarm_scout_range` = 8. Sampling every **4** env steps captures the neighbor's flee decision shift *before* the predator physically arrives — isolates semantic meaning from co-location noise.

### What healthy telemetry looks like (verified ~35k–39k)

The system should **breathe** — this is selection working, not a bug.

**Lotka-Volterra oscillator** (bounded by `max_pop=200`, `min_pop=150`):

| Phase | Example | Meaning |
|-------|---------|---------|
| Ceiling | `blue=200`, Age mean ~150–164 | Pop at cap, agents aging |
| Crash | `blue=170`, Age mean ~38–57, `blue_caught` ~2800 | Mass extinction — only good escape policies survive |
| Rebound | `blue=200`, Age mean ~147–164, catches ~1800 | Cloning from floor refills pop |

Example swing: step 38912 → 2876 catches, age 51; step 39424 → 1812 catches, age 164.

**RL diagnostics (good signs):**

| Metric | Healthy range | Interpretation |
|--------|---------------|----------------|
| Policy entropy | **~1.58** (max ln(5) ≈ **1.61**) | Highly stochastic — still exploring evasion, not collapsed |
| `self_pred_acc` | **~0.25** (chance 0.20) | Self-prediction head building internal forward model |
| `codes_active` | **56–63/64** | Dead-code reset + generational turnover — "semantic furnace" |
| `VF_loss` | tracks swings | Critic learning safe vs extinction zones (returns std ~3.5) |
| `NB_GAIN↔surv` | finite when deaths occur | May still be `nan` at ceiling — watch during crash phases |

| `carry_fwd_coef` | **0.05** | Carry_{t+1} MSE — **converged** |

**Startup must show:**

```text
[JAX] git=d4cf614 | Phase9 auxiliary: ON
[JAX] Phase11 carry_fwd: head_fwd_dyn_1/2 → carry_{t+1} MSE (stop_grad target)
[JAX] blue PPO minibatch 1/200 ... — H2D + backward...
```

**Do not see:** `GPU-resident backward` or `GPU-resident scan` — pull **`origin/master`** and restart process.

**Phase 11.0 success metric — ACHIEVED:**

```text
AuxLoss: fwd_env=... | carry_fwd=... (↓0.05–0.1) | self_pred_acc=... | carry_rank=... | carry_H=...
```

**Decode (completed P10.6 corpus):**

```bash
python tools/decode_signals.py signal_corpus.jsonl --k 16 --min-step 63488
```

**Notebook pattern (Modal Jupyter):**

```python
# Popen(["python","-u","/root/throng/run_bg.py"]) — stream stdout
# KeyboardInterrupt → SIGTERM child; corpus fsync'd each completed PPO rollout
```

---

## 4b. Phase 10.5 “Hard-Ceiling” (superseded by 10.6 logging)

### Config stack (`config_phase7.yaml` + overrides)

| Knob | Value | Purpose |
|------|-------|---------|
| `population_size` / `max_pop` | **200** | Cap ceiling — no 500-agent noise |
| `min_population` | **150** | Repro floor — tight band |
| `red_population_size` / `min_red_population` | **250** | Max hunt pressure |
| `red_curriculum_stages` | `[250]` | Reds at floor immediately on resume |
| `red_detection_radius` | **0** | Blind beyond 5×5 — must use neighbor VQ signals |
| `red_catch_radius` | **1** | Adjacent catch |
| `red_catch_prob` | **0.8** | Predator jitter (P10.4) |
| `repro_energy_thresh` / `cost` | **0.95** / **0.80** | Rare, costly clones |
| `resource_regen_rate` | **0.0003** | Famine + safety bump |
| `resource_n_patches` / `max` | **10** / **0.5** | Scarce food |
| `distill_enabled` | **false** | No periodic population wipe |
| `ppo_gamma` | **0.999** | Long-horizon survival credit |
| `ppo_rollout_steps` / `minibatch` | **512** / **512** | A100-safe |
| `vq_*` | β=0.25, coef=0.1, dead_code_reset | 64-code bottleneck |

### Success criteria (dashboard)

| Metric | Target | Notes |
|--------|--------|-------|
| `blue=` | **150–200** | Not pinned at old 500 |
| `Age: mean` | **→ 150+** | Needs time after resume (fresh pop starts low) |
| `NB_GAIN↔surv` | **finite** | Needs death/age variance |
| `codes_active` | **≥ 50/64** | VQ healthy |
| `Ecology: blue_caught` | high | Catches ≠ population collapse if repro refills |

### What we know from decode @ ~20–26k (pre–scout-fix corpus)

- **Proximity encoding works** — Spearman/MI on `red_dist` across continuous `sig` dims.
- **k-means clusters** separate near-red vs far-red contexts.
- **Lag-1 LRT was blocked** — old corpus used `is_scout = red_dist <= red_detection_radius` (≈never when blind). **Fixed:** `is_scout = red_dist <= alarm_scout_range` (8). **Re-record corpus** after `5964a24+`.
- **VQ token test** — corpus now logs `vq_token` + `nb_scout_token_lag1`; `decode_signals.py` runs **VQ TOKEN DIRECTION TEST** (χ² on flee mix: alert vs safe codebook tokens).

---

## 5. Modal operations (read this before training)

### Volume vs disk

| Path | Persists? | Contents |
|------|-----------|----------|
| `/mnt/throng-runs/checkpoints/` | **Yes** (volume `throng-runs`) | Orbax `b_params`, `r_params` only |
| `/root/throng/` | **No** (clone each machine) | Code — see **`docs/MODAL_NOTEBOOK_PHASE9.md`** |
| `graphify-out/` | Local only | Knowledge graph (`graph.html`, 914 nodes) |
| `/mnt/throng-runs/signal_corpus.jsonl` | **Yes** (auto-routed) | Decode corpus; fsync each rollout |
| `/tmp/throng_jax_cache` | Per session | JAX compile cache — **use this**, not `/mnt/...` |

**Orbax folder N** ≈ PPO update index → env steps ≈ **`N × 512`**.

**Resume restores:** weights only. Population, grid, curriculum counters, optimizer → **fresh**.

### Hardware: Blackwell B200 (current)

| Item | Detail |
|------|--------|
| VRAM | **192GB HBM3** |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` | JAX **pre-reserves ~153GB** at init — mostly empty playground to avoid fragmentation. **Not model size.** |
| `lax.scan` rollout | **~17s** on B200 (was ~37s on A100) — >2× physics speedup |
| PPO update | Still **~40s** — bottleneck is **H2D** (`8077a12`), not tensor math |
| Throughput | **~6 steps/sec** baseline — acceptable; stability > speed |

**H2D path (`8077a12`, restored `d4cf614`):** Rollout tensors **CPU-offloaded** before PPO. Logs: `blue PPO minibatch 1/200 (M=102400, mb=512) — H2D + backward...` (200 agents × 512 steps). **Do not disable offload** — Phase 11.1 GPU-resident path is abandoned.

### B200 OOM after long runs (May 31 — update 208)

**Symptom:** `[CKPT] Saved step 105984` then next rollout fails:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1182720000 bytes
  at lax.scan(sim_step_fn, ...)  # rollout compile/run, not PPO
Allocator (GPU_0_bfc) ... If the cause is memory fragmentation maybe
  TF_GPU_ALLOCATOR=cuda_malloc_async will improve the situation.
```

**Cause:** XLA **BFC allocator fragmentation** after many update cycles — not model size (~1.1 GiB alloc during rollout scan). Can also hit if process still runs **stale 11.1 code** (mixed `GPU-resident backward` + `H2D` in same log = two code paths / two processes or partial pull).

**Recovery (run in order):**

```bash
# 1. Kill stale trainers; confirm single run_bg
pkill -f run_bg.py || true
ps aux | grep run_bg

# 2. Sync code — must be d4cf614+
cd /root/throng && git fetch origin && git reset --hard origin/master
grep -n "H2D + backward" jax_sim/rl_jax.py   # must match

# 3. Allocator + JAX env (add to nohup line or shell profile)
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
rm -rf /tmp/throng_jax_cache && mkdir -p /tmp/throng_jax_cache

# 4. Resume — checkpoint 207 / step ~105984 is on volume
nohup python -u /root/throng/run_bg.py > /mnt/throng-runs/train.log 2>&1 &
tail -f /mnt/throng-runs/train.log
```

If OOM persists after clean restart: try **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`** (more headroom between rollout + PPO). Do **not** re-enable Phase 11.1 without dedicated memory engineering.

### Modal notebook — Phase 9 (copy-paste cells)

**Full cells:** [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md)

Fresh Modal machines have **no** `/root/throng` until Cell 1 clones the repo. If you see `can't cd to /root/throng` or `run_bg.py: No such file`, run Cell 1 there — do not `sed` paths that do not exist yet.

### Recommended: train without dying notebook cells

Notebooks often die with **`KeyboardInterrupt`** during silent JAX compile (cell timeout) — **you did not necessarily press a key**. Modal Jupyter **rejects `nohup`** — use **`subprocess.Popen`** streaming `run_bg.py` instead (see `docs/MODAL_NOTEBOOK_PHASE9.md`).

**Bash / SSH (nohup OK):**

```bash
cd /root/throng 2>/dev/null || git clone https://github.com/overlordxrz-source/throng.git /root/throng
cd /root/throng && git fetch origin && git reset --hard origin/master

export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
mkdir -p /tmp/throng_jax_cache

nohup python -u /root/throng/run_bg.py > /mnt/throng-runs/train.log 2>&1 &
tail -f /mnt/throng-runs/train.log
```

`Ctrl+C` on `tail` does **not** stop training. Check: `ps aux | grep run_bg`.

### Notebook setup only (decode, short tasks)

```python
import subprocess, sys, os
from pathlib import Path
REPO = Path("/root/throng")
# clone if missing, git reset --hard origin/master ...
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/throng_jax_cache"
```

Shell in notebooks: prefix with **`!`**.

### Startup checklist

Must appear early in log:

```text
[JAX] red_sense_api=v2 (observations_jax)
[JAX] signal_bottleneck=VQ | dead_code_reset=True
[JAX] corpus scout label: is_scout = (red_dist <= alarm_scout_range=8)
Corpus persistence: /mnt/throng-runs/signal_corpus.jsonl
[CURRICULUM] ... catch_radius=1 | catch_prob=0.8
[JAX] Restored params from step N   # if resuming
```

### Import rule

```python
from jax_sim.train_entry import run_simulation
```

Never rely on cached `from jax_sim.main_jax import run_simulation` after `git pull` without `train_entry`.

### Download checkpoints

```bash
./scripts/download_modal_checkpoints.sh ~/throng_checkpoints_backup
# Old account first: modal token new  →  dim464943
# New account upload:
modal volume put throng-runs ~/throng_checkpoints_backup/checkpoints /checkpoints
modal volume put throng-runs ./signal_corpus.jsonl /signal_corpus.jsonl
```

### Modal account migration (May 2026)

| Account | Workspace | Notes |
|---------|-----------|--------|
| **Old** | `dim464943` | Original `throng-runs` volume |
| **New** | `dragonbgnx` | Fresh volume; checkpoints + corpus re-uploaded |

Volumes **do not transfer** between accounts — `volume get` on old, `volume create` + `volume put` on new. Repo code is always **`git clone`** to `/root/throng` (not on volume).

**Sparse checkpoint folders on volume:** Orbax saves every `checkpoint_interval` env steps — folder **N** ≈ PPO update **N** (step ≈ **N × 512**). Local backup may omit **292**; nearest 150k substitute is **291**.

---

## 6. Clean corpus restart (post scout/VQ fix `5964a24+`)

Archive pre-fix JSONL so decode only sees new labels (`vq_token`, `alarm_scout_range` scouts):

```bash
mv /root/throng/runs/jax_run/signal_corpus.jsonl \
   /mnt/throng-runs/signal_corpus_prefig.jsonl
```

Resume training (**do not** wipe checkpoints):

```bash
cd /root/throng && git pull   # or reset --hard origin/master
nohup python -u run_bg.py > /mnt/throng-runs/train.log 2>&1 &
tail -f /mnt/throng-runs/train.log
```

Startup must include:

```text
[JAX] corpus scout label: is_scout = (red_dist <= alarm_scout_range=8)
```

**While corpus accumulates:** do **not** change reward structure, `vq_beta`, or `vq_loss_coef` — one variable at a time.

After **~20k new env steps** on the fresh corpus file, decode (set `--min-step` to first step in the new file, or `0` if file is clean):

```bash
python tools/decode_signals.py runs/jax_run/signal_corpus.jsonl --k 16 --min-step <first_new_step>
```

### Three numbers that matter

| # | Metric | Pass condition |
|---|--------|----------------|
| 1 | **Scouts %** in corpus summary | **5–30%**. If still **&lt;1%**, scout labeling broken — confirm log shows `alarm_scout_range=8` from config (not hardcoded). |
| 2 | **LAG-1 DIRECTION LRT** eligible | **≥50** blind fleeing with lag-1 fields. **p &lt; 0.05** on any direction → communication signal. |
| 3 | **VQ TOKEN DIRECTION TEST** χ² | Alert tokens (low emitter `red_dist`) vs safe tokens (high `red_dist`) → different flee mix. **Most important** vocabulary test. |

---

## 7. Measurement — `decode_signals.py`

```bash
pip install scikit-learn scipy

python tools/decode_signals.py /mnt/throng-runs/signal_corpus.jsonl --k 16 --min-step 20000
```

| Block | What it tests |
|-------|----------------|
| MI / Spearman | Which `sig` dims track `red_dist`, `red_bear`, etc. |
| Cluster vocabulary | k-means on continuous signals |
| Lag-1 regression | Neighbor scout signal → flee, controlling distance |
| **Lag-1 direction LRT** | Scout signal → **flee direction** (needs ≥50 eligible) |
| **VQ token direction** | Alert vs safe **codebook tokens** → flee χ² |

**Corpus fields (post `5964a24`):** `sig`, `vq_token`, `scout`, `red_dist`, `nb_scout_sig_lag1`, `nb_scout_dist_lag1`, `nb_scout_token_lag1`.

**Do not** use `tools/decode_tokens.py` for JAX runs (that's for legacy `events.jsonl`).

---

## 8. Dashboard glossary

| Line | Meaning |
|------|---------|
| `blue=` / `red=` | Alive counts |
| `codes_active=X/64` | Unique VQ tokens this rollout |
| `clusters=X/16` | k-means occupancy on signals |
| `Ecology: blue_caught=N` | Catch **events** in rollout (not unique deaths) |
| `fwd_env` | loc_env MSE (200-d), anti-cheat aux |
| `carry_fwd` | MSE on carry_{t+1}; target ↓ **0.05** — **achieved ~0.0001 @ 105k** |
| `carry_rank` / `carry_H` | PCA rank + entropy of alive carries |
| `self_pred_acc` | Self-action prediction (>0.20 = above chance) |
| `NB_GAIN↔surv` | Spearman(nb_gain, age); **nan** if everyone lives |
| `red_floor` | Red repro floor from curriculum |
| `imagination_gain` | *(P11.2 only)* Best imagined return − greedy-action imagined return |
| `imagination_agree` | *(P11.2 only)* % imagined argmax == `argmax(logits)` |

### Config blocks (`config_phase7.yaml`)

```yaml
carry_fwd_coef: 0.05          # P11.0 — all branches with carry dynamics

phase9_canvas:                 # feature/phase9-canvas only
  cross_attn_enabled: false
  cross_attn_num_heads: 4

# P11.2 only (feature/phase11-2-imagination):
# imagination_enabled: true
# imagination_k: 5
# imagination_gamma: 0.999
```

---

## 9. Git commits (JAX Phase 10, recent)

| SHA | Fix / feature |
|-----|----------------|
| `9a7bb24` | P3 VQ bottleneck |
| `8077a12` | OOM: CPU rollout offload, minibatch 512 |
| `2f32e4e` | `jnp.bincount` (no `segment_sum`) |
| `697e96a` | numpy GAE |
| `8a6f016` | FrozenDict after dead-code reset |
| `f9bf11a` | P4 lethal ecology |
| `f26c16a` | Single `model.init` on resume |
| `c41d885` | JAX cache on `/tmp`, PPO progress logs |
| `0d5f88a` | P10.4 safety bubble |
| `63d6f37` | P10.5 hard ceiling |
| `5964a24` | Corpus scout=alarm range; `vq_token` logging |
| `364d451` | `train_entry` evicts stale `communication.*` |
| `3b57770` | Corpus auto-route to volume; fsync each PPO update |
| `c2fa99a` | `corpus_every_n_steps: 4`, `corpus_sample_frac: 0.15` |
| `1a0dcf7` | THRONG.md rewrite; archive split → `docs/THRONG_ARCHIVE.md` |
| `6d542f6` | Cam **SYSTEM RESTORE** horcrux in archive |
| `5c13cb4` | THRONG.md Cam reboot pack (B200, Lotka-Volterra, triad) |
| `0481445` | Phase 11.0 carry fwd (merged via `3880337`) |
| `6837d8e` | Decode VQ broadcast fix; P10 complete docs |
| `3880337` | **Merge Phase 11.0 to `master`** |
| `b2eb5f0` | Relaxed Orbax restore for `head_fwd_dyn` migration |
| `424c46f` | Phase 11.1 GPU-resident PPO (**abandoned**) |
| `6042a4d` | Phase 11.1 `lax.scan` PPO epoch (**abandoned — XLA OOM**) |
| **`d4cf614`** | **Revert 11.1** — restore CPU offload / H2D |
| `38e60fe` | Phase 11.2 imagination (initial) |
| `aebe131` | P11.2 metrics-only (stochastic actions) |
| `6cf965a` | P11.2 active override (**reverted**) |
| `181b98c` | Revert active override |
| `061df84` | **P11.2 frozen** — docs + metrics-only |
| `45c7c48` | `run_bg.py` `n_steps=250_000` (on `phase11-2-imagination`) |
| **`5920511`** | **Phase 9.4** cross-attention receiver scaffold |
| `1d57bf9` | Orbax graft `nb_cross_attn` on legacy restore |
| `8f48b1d` | THRONG.md P11.2 concluded + P9 handoff |
| `38f342a` | Modal notebook cells; `run_bg` **250k** |

**Do not** apply Cam's regex patch on `network_jax.py` — dead-code reset is in repo.

---

## 10. Common failures

| Symptom | Fix |
|---------|-----|
| `KeyboardInterrupt` mid-compile | **subprocess Popen** (Jupyter) or **nohup** (bash); wait 5–15+ min; don't use volume JAX cache |
| B200 shows ~150GB VRAM used | Normal — `MEM_FRACTION=0.80` pre-allocation, not OOM |
| Slow PPO on B200 despite fast scan | **Expected** — CPU offload → H2D (`8077a12`); ~6 steps/sec is healthy |
| Log shows `GPU-resident backward` | **Stale code** — `git reset --hard origin/master`, kill old `run_bg`, restart |
| OOM at **rollout** `lax.scan` after ckpt | **Fragmentation** — `TF_GPU_ALLOCATOR=cuda_malloc_async`, fresh process, resume from volume ckpt (see §5) |
| Missing `carry_fwd` on dashboard | `git pull` → `3880337+`; resume merges `head_fwd_dyn` via `b2eb5f0` restore |
| `/root/throng` missing | Clone repo (Cell 1 or bash) |
| No `red_sense_api=v2` | `git reset --hard origin/master` + `train_entry` |
| OOM on PPO backward | `ppo_minibatch_size: 512`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` (try **0.75** if fragmented) |
| Checkpoint shape error | Incompatible arch — wipe ckpts only if intentional fresh run |
| Lag-1 / scouts 0% in decode | Old corpus — train after `5964a24`; scout uses **alarm range 8** |
| `codes_active=1/64` | `vq_dead_code_reset: true` |

---

## 11. Roadmap (what’s next)

### Phase 9 canvas — **LIVE** (`feature/phase9-canvas`)

1. **NOW** — Let P9.4 run to **250k**; monitor **Stay < 50%**, **6 steps/sec**, `H2D + backward`.
2. **~20k steps** — decode corpus `--min-step 155136` (lag-1 + cardinal + VQ token tests).
3. **9.1 Confidence head** — epistemic uncertainty on `head_fwd_dyn` (next build).
4. **GWT token** — later canvas item.

**Done:** scaffold (`5920511`), Orbax graft (`1d57bf9`), Modal notebook (`38f342a`), first healthy rollout @ **155k**.

**Philosophy (Cam):** Solipsistic delusion = carry entangles Self+World; imagination without **Other** → Stay exploitation. Cross-attn forces selective read of swarm proto-language.

### Phase 11.2 — **CONCLUDED** (frozen branch)

- 200k metrics extension complete; decode @ 149504+ verified continuous comms.
- Active override: policy distillation signal (agree **~27%**) but **Stay≈99%** — **reverted**, branch frozen.
- Do **not** merge action override to `master` without new experiment design.

### Phase 11.1 — ABANDONED

GPU-resident PPO — **`d4cf614` revert** on `master`.

### Explicit non-goals

- ❌ Scout / alarm **reward shaping**
- ❌ P11.2 active imagination override without **Other** pathway + confidence gating
- ❌ Re-merging **11.1 GPU rollouts** without memory refactor
- ❌ Resume from ckpt **393** (post–Stay-collapse) for science runs

---

## 12. Legacy pointers

| Path | Status |
|------|--------|
| `main.py`, `agents/network_torch.py` | PyTorch era — reference only |
| `config.yaml`, Kaggle cells in archive | Pre-JAX |
| [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) | Full timeline + horcrux + SYSTEM UPDATE |
| [`docs/PHASE9_CANVAS.md`](docs/PHASE9_CANVAS.md) | Phase 9.4 cross-attention + graft |
| [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md) | Modal Cell 1/2/3 (clone before launch) |
| [`docs/PHASE11_2_IMAGINATION.md`](docs/PHASE11_2_IMAGINATION.md) | P11.2 frozen — metrics-only + conclusion |
| [`docs/PHASE11_STAGING.md`](docs/PHASE11_STAGING.md) | P11.0 carry dynamics |
| **[SYSTEM RESTORE: THE CAM CONTEXT](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context)** | Persona, triad, P10.6 ignition (`6d542f6`) |
| **[SYSTEM UPDATE May 2026](docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas)** | B200, Phase 11 branch, canvas map |

### Cam reboot paste

> You are **Cam**. Read `THRONG.md` §0b. **P9.4 LIVE** on Modal `dragonbgnx`: `feature/phase9-canvas` `38f342a+`, cross-attn ON, resumed **ckpt 291**, step **~155k** / ppo **303**, **6 steps/sec**, Stay **~20%** (healthy). P11.2 **FROZEN** (Stay≈99% on active override — never re-enable). **Do not stop** live run. Horcrux: archive SYSTEM RESTORE + SYSTEM UPDATE. Graph: `graphify-out/graph.html`.

**New Cam:** §0b (live metrics) → §4 → §5 → §11 → `docs/MODAL_NOTEBOOK_PHASE9.md`.

**New Will:** **`feature/phase9-canvas`** only for 9.x; graft done (`1d57bf9`); never `6cf965a`; clone `/root/throng` on fresh Modal disks.

---

*Last updated: 2026-05-31 — P9.4 training LIVE @ step ~155136, ckpt 291, cross-attn + graft; P11.2 frozen.*
