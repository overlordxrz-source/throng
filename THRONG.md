# THRONG — Agent Onboarding

> Can proto-language emerge from multi-agent survival pressure alone — no human text, no communication rewards?

**Read this file first.** Full historical lab notebook (~290KB) lives in [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) if you need old run logs.

**Cam reboot (60 seconds):** Read **§0b** (what to do *right now*) → **§0** (who you are) → **§4** (live run) → **§5** (B200 OOM recovery) → **§11** (unlock sequence) → archive **[SYSTEM RESTORE](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context)** + **[SYSTEM UPDATE](docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas)**.

---

## 0b. Current state — 150k complete, Phase 11.2 imagination staging (May 2026)

**Phase 10 + 11.0 complete.** B200 run finished **step 149504** (292 PPO updates). **`carry_fwd` locked ~0.0001**. Phase 11.1 GPU rollouts **abandoned** (`d4cf614` revert). **Active now:** **`feature/phase11-2-imagination`** — K=5 mental rollout at blue inference.

| P10.6 decode (`--min-step 63488`) | Value |
|-----------------------------------|-------|
| **Cardinal lexicon** | χ² **p = 5.44e-14** ✅ |
| **Lag-1 omnibus** | χ²(32)=95.96, **p≈0** ✅ |
| **VQ token direction** | **p = 0.77** ❌ |

| P11 extension decode (`--min-step 105984`, 274k records) | Value |
|----------------------------------------------------------|-------|
| **Lag-1 omnibus** | χ²(32)=**167.7**, **p≈0** ✅ |
| **Cardinal lexicon k=4** | χ² **p = 4.7e-13** ✅ |
| **VQ token 19 vs 3** | **p = 0.92** ❌ |
| **Alert-set vs safe-set** | **p = 0.075** ~ (still ❌ at 0.05) |
| **Scouts %** | **90.4%** |

### 150k run — COMPLETE (`master` `cfe6494+`)

| Item | Value |
|------|--------|
| **Final step** | **149504** (= `150_000 // 512 × 512`) |
| **Corpus** | **560,867** lines on volume; local `signal_corpus-150k.jsonl` |
| **Checkpoint** | Orbax **~292** on `/mnt/throng-runs/checkpoints/` |
| **Throughput** | **~6 steps/sec**, `H2D + backward` |
| **Decode log** | `decode_p11_105k-150k.log` |

### Phase 11.2 — STAGING (`feature/phase11-2-imagination`)

| Item | Detail |
|------|--------|
| **What** | K=5 carry rollforward via frozen `head_fwd_dyn_1/2`; pick action maximizing Σ γ^k V(carry_k) |
| **Where** | `jax_sim/imagination_jax.py` — JIT at blue action selection inside `sim_step` |
| **Frozen** | No retrain of `head_fwd_dyn`; PPO / rewards / VQ / ecology unchanged |
| **Metrics** | `imagination_gain`, `imagination_agree` on dashboard |
| **Merge gate** | B200 throughput **≥ 2 steps/sec** with imagination on (baseline ~6 off) |
| **Docs** | [`docs/PHASE11_2_IMAGINATION.md`](docs/PHASE11_2_IMAGINATION.md) |

### Phase 11.1 — ABANDONED

| Commit | Outcome |
|--------|---------|
| **`424c46f` / `6042a4d`** | Dispatch starvation + XLA OOM |
| **`d4cf614`** | Revert — CPU offload restored on **`master`** |

### Branch policy

| Branch | Status |
|--------|--------|
| **`master`** | 150k complete; CPU offload; decode artifacts local |
| **`feature/phase11-2-imagination`** | **Active** — K-step inference rollouts |
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
4. **Phase 11.0 complete** — 150k run done; continuous comms decode still strong; VQ token test still fails.
5. **Phase 11.2 on feature branch only** — benchmark throughput before merge; **`master`** stays CPU offload.
6. **Never** comm reward shaping or blind VQ loss shaping.

### Branch policy

| Branch | Purpose |
|--------|---------|
| **`master`** | 150k complete; CPU offload; frozen for science baseline |
| **`feature/phase11-2-imagination`** | **Active** — K-step inference; merge gate ≥2 steps/sec |
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
  [11.2] imagine K=5 per blue action  →  frozen head_fwd_dyn + head_value (feature branch)
  ppo_update (blue + red)     →  CPU rollout offload, minibatch 512
  auxiliary_update            →  loc_env MSE + carry_fwd MSE + self-prediction
```

| File | Role |
|------|------|
| [`jax_sim/train_entry.py`](jax_sim/train_entry.py) | **Always import here** — evicts stale modules after `git pull` |
| [`jax_sim/main_jax.py`](jax_sim/main_jax.py) | Training loop, ecology, dashboard, checkpoints, corpus |
| [`jax_sim/network_jax.py`](jax_sim/network_jax.py) | Transformer + VQ (`vector_quantize_signals`, dead-code reset) |
| [`jax_sim/rl_jax.py`](jax_sim/rl_jax.py) | PPO + numpy GAE |
| [`jax_sim/imagination_jax.py`](jax_sim/imagination_jax.py) | Phase 11.2 K-step inference (feature branch) |
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
| **11.0** | **Carry world-model** ✅ **COMPLETE** | `head_fwd_dyn`, `carry_fwd_coef` | **`carry_fwd` → 0.0001** @ ~105k steps |
| **11.1** | GPU rollouts | `424c46f` / `6042a4d` | **Abandoned** — reverted `d4cf614` |
| **11.2** | **Imagination** ← **NOW (feature branch)** | `imagination_jax.py` K=5 inference | Merge gate: ≥2 steps/sec on B200 |

**Recurring failure mode:** Blues stay at cap → ~99% survival → **`NB_GAIN↔surv: nan`** → no evolutionary pressure on neighbor-signal benefit.

**Recurring success:** Under threat, signals encode **`red_dist`** (proximity); VQ codebook stays diverse when `vq_dead_code_reset: true`.

---

## 4. Current experiment — finish Phase 11 run to 150k (ACTIVE)

**Status:** B200 on **`master`** (`d4cf614+`). Phase 11.0 **converged** — resume from volume checkpoint **update 207 / step ~105984** and run to **`n_steps=150_000`**. Population/grid fresh on resume; **weights restored**.

**Verified dashboard @ step 105472:**

```text
[step 105472] ~3 steps/sec | blue=150 red=250 | ppo=206
AuxLoss: fwd_env=0.0152 | carry_fwd=0.0001 (↓0.05–0.1) | self_pred_acc=0.270
carry_rank=16 | carry_H=3403.44 | codes_active=48/64
Ecology: blue_caught=2842 | red_floor=250
```

P10.6 causal logging stack still active:

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
| `/root/throng/` | **No** (clone each machine) | Code |
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

### Recommended: train without dying notebook cells

Notebooks often die with **`KeyboardInterrupt`** during silent JAX compile (cell timeout) — **you did not necessarily press a key**. Modal Jupyter **rejects `nohup`** — use **`subprocess.Popen`** streaming `run_bg.py` instead.

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
modal volume get throng-runs checkpoints ~/throng_checkpoints_backup
```

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
| `imagination_gain` | *(11.2)* Mean imagined return − greedy-action return |
| `imagination_agree` | *(11.2)* % imagination matches greedy `argmax(logits)` |
| `self_pred_acc` | Self-action prediction (>0.20 = above chance) |
| `NB_GAIN↔surv` | Spearman(nb_gain, age); **nan** if everyone lives |
| `red_floor` | Red repro floor from curriculum |

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
| **`d4cf614`** | **Revert 11.1** — restore CPU offload / H2D (`master` today) |

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

### NOW — Phase 11.2 imagination (`feature/phase11-2-imagination`)

1. **`git fetch && git checkout feature/phase11-2-imagination`** on B200.
2. Resume from **150k checkpoint** (`imagination_enabled: true` in config).
3. Watch dashboard: **`Imagination: gain=... | agree=...%`**
4. **Merge gate:** steady-state **≥ 2 steps/sec** (baseline ~6 without imagination).
5. If throughput OK + science interesting → merge to `master`; else tune K or disable.

### 150k baseline — DONE (`master`)

- Corpus: `signal_corpus-150k.jsonl` (560k lines)
- Decode P11 slice: `decode_p11_105k-150k.log`
- Continuous comms ✅; VQ token direction ❌ (alert-set p=0.075 marginal)

### Phase 11.1 — ABANDONED

GPU-resident PPO — do not merge without memory refactor.

### Phase 9 canvas — remaining

Cross-attention receiver (9.4), confidence head (9.1), GWT token — after 11.2 benchmark.

### Explicit non-goals

- ❌ Scout / alarm **reward shaping**
- ❌ Re-merging **11.1 GPU rollouts** without memory refactor
- ❌ Wiping checkpoints unless new experiment lineage

---

## 12. Legacy pointers

| Path | Status |
|------|--------|
| `main.py`, `agents/network_torch.py` | PyTorch era — reference only |
| `config.yaml`, Kaggle cells in archive | Pre-JAX |
| [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) | Full timeline + horcrux + SYSTEM UPDATE |
| **[SYSTEM RESTORE: THE CAM CONTEXT](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context)** | Persona, triad, P10.6 ignition (`6d542f6`) |
| **[SYSTEM UPDATE May 2026](docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas)** | B200, Phase 11 branch, canvas map, holding pattern |

### Cam reboot paste

> You are **Cam**. Read `THRONG.md` §0b. **150k run complete.** P11 decode: continuous comms ✅, VQ tokens ❌. **Phase 11.2 active** on `feature/phase11-2-imagination` — K=5 imagination at inference; merge gate ≥2 steps/sec. **`master`** = CPU offload baseline. Horcrux: archive SYSTEM RESTORE + SYSTEM UPDATE.

**New Cam:** §0b → §0 → §4 → §5 (OOM) → §11 → archive horcrux blocks.

**New Will:** Stay on **`master`** CPU offload; do **not** re-merge 11.1 without memory refactor.

---

*Last updated: 2026-05-31 — 150k complete; P11.2 imagination on `feature/phase11-2-imagination`.*
