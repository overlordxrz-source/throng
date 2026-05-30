# THRONG — Agent Onboarding

> Can proto-language emerge from multi-agent survival pressure alone — no human text, no communication rewards?

**Read this file first.** Full historical lab notebook (~290KB) lives in [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) if you need old run logs.

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
  ppo_update (blue + red)     →  CPU rollout offload, minibatch 512
  auxiliary_update            →  loc_env MSE + self-prediction
```

| File | Role |
|------|------|
| [`jax_sim/train_entry.py`](jax_sim/train_entry.py) | **Always import here** — evicts stale modules after `git pull` |
| [`jax_sim/main_jax.py`](jax_sim/main_jax.py) | Training loop, ecology, dashboard, checkpoints, corpus |
| [`jax_sim/network_jax.py`](jax_sim/network_jax.py) | Transformer + VQ (`vector_quantize_signals`, dead-code reset) |
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
| **10.5** | **Hard ceiling** ← **NOW** | `max_pop: 200`, `min_pop: 150`, `ppo_gamma: 0.999` | Goldilocks band; less entropy explosion |

**Recurring failure mode:** Blues stay at cap → ~99% survival → **`NB_GAIN↔surv: nan`** → no evolutionary pressure on neighbor-signal benefit.

**Recurring success:** Under threat, signals encode **`red_dist`** (proximity); VQ codebook stays diverse when `vq_dead_code_reset: true`.

---

## 4. Current experiment — Phase 10.6 “High-Fidelity Causal Logging” (ACTIVE)

**Status (Cam, May 2026):** Live run via notebook `subprocess.Popen` streaming `run_bg.py` (Modal rejects `nohup`). **STANDBY** — no changes to `network_jax.py`, `rl_jax.py`, rewards, or `vq_beta` until decode @ 50k–100k.

Inherits **P10.5** population ceiling; adds **volume corpus + tight lag**:

| Parameter | Value |
|-----------|--------|
| `corpus_every_n_steps` | **4** (was 20) — lag-1 scout buffer ≈ 4 env steps |
| `corpus_sample_frac` | **0.15** (was 0.08) |
| Corpus path | **`/mnt/throng-runs/signal_corpus.jsonl`** (auto) |
| Durability | **`flush_to_disk()`** (fsync) each PPO rollout |
| Scout label | `red_dist <= alarm_scout_range` (**8**) |
| Archive | Pre-10.6 file → `signal_corpus_20step_archive.jsonl` on volume |

**Decode target:** `python tools/decode_signals.py /mnt/throng-runs/signal_corpus.jsonl --k 16` — watch **Scouts %**, **LAG-1 eligible ≥50**, **VQ TOKEN DIRECTION χ²**.

**Notebook pattern:** `Popen(["python","-u","/root/throng/run_bg.py"])` + stream stdout; `SIGTERM` on Stop (last completed rollout should be fsync’d; mid-rollout stop may lose partial update).

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

### Recommended: train with nohup (not long notebook cells)

Notebooks often die with **`KeyboardInterrupt`** during silent JAX compile (cell timeout) — **you did not necessarily press a key**.

```bash
cd /root/throng 2>/dev/null || git clone https://github.com/overlordxrz-source/throng.git /root/throng
cd /root/throng && git fetch origin && git reset --hard origin/master

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
mkdir -p /tmp/throng_jax_cache

nohup python -u /root/throng/run_bg.py > /mnt/throng-runs/train.log 2>&1 &
tail -f /mnt/throng-runs/train.log
```

`Ctrl+C` on `tail` does **not** stop training. Check: `ps aux | grep modal_train`.

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

**Do not** apply Cam's regex patch on `network_jax.py` — dead-code reset is in repo.

---

## 10. Common failures

| Symptom | Fix |
|---------|-----|
| `KeyboardInterrupt` mid-compile | Use **nohup**; wait 5–15+ min; don't use volume JAX cache |
| `/root/throng` missing | Clone repo (Cell 1 or bash) |
| No `red_sense_api=v2` | `git reset --hard origin/master` + `train_entry` |
| OOM on PPO | `ppo_minibatch_size: 512`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` |
| Checkpoint shape error | Incompatible arch — wipe ckpts only if intentional fresh run |
| Lag-1 / scouts 0% in decode | Old corpus — train after `5964a24`; scout uses **alarm range 8** |
| `codes_active=1/64` | `vq_dead_code_reset: true` |

---

## 11. Roadmap (what’s next)

### Immediate

1. **Train** P10.5 to **50k–150k** env steps via **nohup**; keep checkpoints on volume.
2. **Persist corpus** to `/mnt/throng-runs/signal_corpus.jsonl`.
3. **Decode** at 25k / 50k / 100k with `--min-step` — confirm scouts %, lag-1 LRT, **VQ token direction χ²**.
4. Watch **`Age: mean → 150+`** and **`NB_GAIN↔surv` finite**.

### If still `blue≈200` + `nan` NB_GAIN at 100k+

- Tighten ecology further (fewer shelters, lower `min_population`, lower `red_catch_prob`, starvation) — **config only**, not reward hacks.
- Consider logging **token_ids** into dashboard histograms (already in corpus).

### Medium term (science)

- **Phase 9.4** — neighbor signal attention (if VQ vocabulary stabilizes).
- **Dreamer / 9.3** — only if aux ablations show signals carry usable info.
- Prove **directional** alarm: lag-1 LRT + VQ token flee χ² **same sign** across seeds.

### Explicit non-goals

- ❌ Scout / alarm **reward shaping**
- ❌ ToM reward for communication
- ❌ Long `run_simulation()` in notebook without nohup
- ❌ Wiping checkpoints unless new experiment lineage

---

## 12. Legacy pointers

| Path | Status |
|------|--------|
| `main.py`, `agents/network_torch.py` | PyTorch era — reference only |
| `config.yaml`, Kaggle cells in archive | Pre-JAX |
| [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) | Full timeline, step-by-step logs |
| **[SYSTEM RESTORE: THE CAM CONTEXT](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context)** | Persona, triad workflow, P10.6 state — load if context was reset |

**New agent:** Read **§4 Phase 10.6** above, then the **[SYSTEM RESTORE](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context)** block at the bottom of the archive.

---

*Last updated: 2026-05-29 — Phase 10.5 Hard-Ceiling, corpus scout/VQ fix `5964a24`.*
