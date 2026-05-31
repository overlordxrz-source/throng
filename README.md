# THRONG — Emergent Communication & Cumulative Culture from Survival Pressure

THRONG is a multi-agent artificial-life simulation in which two populations of
neural agents — **blue** (prey) and **red** (predator) — co-evolve in a rich
2D world. Blues must learn, from scratch and with no human-supplied rewards,
to **survive, signal, and pass knowledge on**. The goal is not "an agent that
plays a game well." The goal is **emergence**: language, culture, and proto-
cognition arising purely from selection pressure.

**Current state (May 2026):** **Phase 12 co-evolution LIVE** on Modal B200 —
branch **`feature/phase12-red-coevolution`**. Dual brain: blue P11.3 stack +
`PredatorNetworkJax` (red VQ comms). **Holding pattern:** first red decode @
~30k wiretap steps **failed** (lag-1 LRT **p=0.9767** — babbling); Modal run
continues until red **pincer χ²** passes. **Phase 13 blocked** until then.

**Full ops / decode / roadmap:** [THRONG.md](THRONG.md) §0b (read first).

**Framework:** JAX + Flax (`lax.scan` rollout, CPU-offload PPO on B200)
**Active config:** `config_phase7.yaml`
**Active branch:** `feature/phase12-red-coevolution` (not `master` for live train)
**Working files:** `jax_sim/` (PyTorch in `agents/`, `main.py` is legacy).

For the full research log, theory, philosophy, and per-phase post-mortems see
[THRONG.md](THRONG.md) (agent onboarding). Phase 12 co-evolution:
[docs/PHASE12_COEVOLUTION.md](docs/PHASE12_COEVOLUTION.md). Historical log:
[docs/THRONG_ARCHIVE.md](docs/THRONG_ARCHIVE.md).

### Decode (offline)

```bash
# Blue alarm / flee (214k reference: decode_p11_3_214k.log)
python3 tools/decode_signals.py signal_corpus.jsonl --k 16 --min-step 149500

# Red pincer / pursuit (re-run as red corpus grows)
python3 tools/decode_signals.py --red /mnt/throng-runs/signal_corpus_red.jsonl --k 16 --min-step <wiretap_restart>
```

**Pass bar (red):** RED VQ PINCER TEST χ² **p < 0.05** (Chase vs Search tokens → receiver lag-1 N/S/E/W). First run @ ~30k: **FAIL** (LRT p=0.9767).

---

## What THRONG Is Trying to Do

Most "intelligent agent" projects work *top-down*: define a task, define a
reward, build an architecture, get it to score well. THRONG works *bottom-up*:
build a world hard enough that **survival itself selects for the things we
care about** — communication, cooperation, planning, cumulative culture — and
get out of the way.

The bet is that grounded, embodied, multi-agent pressure produces something
qualitatively different from text-trained LLMs:

- LLMs are **mirrors** of human concepts.
- THRONG agents are **seeds** that grow their own concepts from physical
  survival outcomes — concepts that may or may not align with anything a human
  would name.

We try to falsify this. Every claim ("agents are communicating", "they are
forming a vocabulary", "they are planning") has a *number* we watch in the
dashboard. If the number doesn't move, the claim is wrong, and the design is
incomplete.

---

## Quick Start (Colab / Kaggle T4)

```python
# Cell 1 — setup
!git clone https://github.com/overlordxrz-source/throng.git /content/throng 2>/dev/null || true
%cd /content/throng
!git reset --hard
!git pull origin master
!pip install -q jax[cuda12] flax optax pyyaml wandb orbax-checkpoint

# Cell 2 — fresh run with aggressive overrides for free T4
import sys, os, shutil
sys.path.insert(0, "/content/throng")
os.chdir("/content/throng")

# Clear prior checkpoints if doing a fresh run after major code change
shutil.rmtree("runs/jax_run/checkpoints", ignore_errors=True)

import yaml
from jax_sim.main_jax import run_simulation

with open("config_phase7.yaml") as f:
    cfg = yaml.safe_load(f)

# T4-friendly overrides (do NOT use these on an A100 / H100; scale back up)
cfg["population_size"]    = 100
cfg["red_population_size"] = 75
cfg["grid_size"]          = 64
cfg["n_layers"]           = 2     # brain_vote may expand this up to brain_max_layers=6
cfg["ppo_rollout_steps"]  = 128
cfg["ppo_minibatch_size"] = 256
cfg["memory_buffer_size"] = 5
cfg["mind_meld_enabled"]  = False
cfg["use_pmap"]           = False

final_params, metrics = run_simulation(cfg, seed=42, n_steps=1_000_000)
```

**Local (Linux + CUDA 12):**

```bash
pip install jax[cuda12] flax optax orbax-checkpoint pyyaml wandb
python -c "
import yaml
from jax_sim.main_jax import run_simulation
cfg = yaml.safe_load(open('config_phase7.yaml'))
run_simulation(cfg, seed=42, n_steps=1_000_000)
"
```

**Legacy PyTorch (frozen, not maintained):**

```bash
pip install -r requirements.txt
python main.py --config config_phase7.yaml --headless --fresh
```

---

## Architecture (At a Glance)

### The World

A toroidal grid (default 128×128, runnable at 64×64 for low-VRAM) with eight
overlapping environmental channels — every cell carries:

| Channel | What it is | Why it matters |
|---|---|---|
| Resources | Gaussian food patches that regenerate | Energy economy |
| Walls | Procedural cave-like barriers | Navigation challenge |
| Shelter spots | Safe zones that double red detection range | Strategic geography |
| Contested nodes | High-value hotspots agents fight over | Competition gradient |
| Scent trails | Fading traces left by reds | Indirect danger signal |
| Symbols | Persistent "graffiti" agents write to the ground | Long-lived communication |
| Cultural Fast (decay 0.90) | Recent danger / coordination traces | "Red was here" memory |
| Cultural Slow (decay 0.995) | Long-term landmarks | "This valley is safe" memory |
| Puzzle grid | Co-op lock-and-key mechanic | Cooperation pressure |

### The Agent Brain

Each agent is a **Flax transformer** (`jax_sim/network_jax.py`):

- 2–6 attention layers (`n_layers`), expanded dynamically by **brain-vote**.
- 128-dim tokens through multi-head self-attention.
- 256-dim recurrent **carry** persisting across the agent's lifetime.
- Eight output heads (action, signal, symbol, culture-fast, culture-slow,
  value, theory-of-mind, gain).
- Observation dimension currently **1,800** at `n_layers=2, neighbor_k=6,
  memory_buffer_size=5, env_ch=8`.

### Learning

- **MAPPO** (Multi-Agent PPO) — one shared policy per team. All blues update
  one network; all reds update another.
- **GAE** advantage normalisation, value clipping, gradient clipping at norm 2.
- **Survival-only reward**: +small for staying alive each step, large negative
  on death. We deliberately do **not** reward communicating, cooperating, or
  exploring.
- **Red curriculum**: floor stages `[6, 15, 30, 75]`. Reds graduate when blues
  sustain ≥ `curriculum_survival_threshold` (0.80 by default) for
  `curriculum_sustain_updates` (5) consecutive PPO updates.
- **Capacity-based brain vote**: `n_layers` grows when *signal entropy
  plateaus* + *signal diversity plateaus* + *VF loss stays high* + *survival
  is under pressure*. Translation: "agents have squeezed everything they can
  out of their current brain — give them more." Not "agents are dying" (the
  old trigger).

### What Information Flows Between Agents

Four channels, each with a different temporal and spatial scale:

1. **Signals** (per-step, 32-dim continuous) — 6 nearest neighbours hear what
   each agent broadcasts. **Live as of May 28 2026.**
2. **Cultural Fast grid** (decay 0.90, ~10 step memory) — danger traces.
3. **Cultural Slow grid** (decay 0.995, ~200 step memory) — stable landmarks.
4. **Parameter sharing (MAPPO gradient)** — what one blue learns, every blue
   learns. The "evolutionary memory" channel.

(A fifth channel — **mind-meld** — is implemented but currently disabled for
performance. It directly blends carries between adjacent old/young agents.)

---

## What the Dashboard Means

Every `T = ppo_rollout_steps` simulation steps you'll see a panel like:

```
======================================================================
[step    5120] 2 steps/sec | blue=100 red=75 | ppo=40
  Actions: N=15% S=30% E=15% W=27% Stay=14%
  Energy:  mean=0.622 std=0.101 | Age: mean=62 max=345
  Values:  mean=1.6022 | VF_loss=0.8093 | Clip=0.044
  Reward:  mean=0.1037 | Entropy: 1.5030
  Signals: 3135 unique | NB_GAIN↔surv: nan
  Curriculum: red_floor=75 sustain=0/5 | brain=2L
======================================================================
```

| Field | Meaning |
|---|---|
| `step` | Simulation steps since launch. |
| `steps/sec` | Throughput. T4 free tier sustains ~2 steps/sec at N=100. |
| `blue=… red=…` | Currently alive populations. |
| `ppo=` | PPO update count. |
| `Actions: …%` | Action distribution this rollout. Healthy = no single action above ~50%. |
| `Energy` | Per-agent energy. Should stabilise around `repro_energy_thresh` once foraging is solved. |
| `Age max=…` | Lifespan of the oldest agent — proxy for "real elders exist". |
| `Values mean / VF_loss / Clip` | Critic stats. Loss should fall as predation pressure stabilises. Clip is PPO fraction clipped. |
| `Reward` | Mean per-step reward — survival-only, so close to 0.10 ≈ 100% survival. |
| `Entropy` | Action policy entropy. `ln(5)=1.609` is fully uniform. Below ~1.0 means a strong preference has formed. |
| `Signals: N unique` | **Distinct broadcast vectors among alive agents** this rollout. Pre-May-28: stuck at 1 (channel was dead). Now: ~N_alive, which means broadcasts are non-trivial but **not yet compressed into a vocabulary**. |
| `NB_GAIN↔surv` | Spearman correlation between "did I weight neighbour signals?" and "did I survive?". `nan` until there's enough death variance. **Positive = listening is selected for.** |
| `Curriculum red_floor / sustain` | Current red population minimum, and how many consecutive PPO updates met the survival threshold. |
| `brain=NL` | Current `n_layers` of the blue transformer. Grows via brain-vote up to `brain_max_layers`. |

The `[DEBUG]` block that prints **before** the first `step 512` summary is a
one-shot sanity panel that verifies (a) parameters initialised without NaN,
(b) action logits and entropy are at sensible defaults, (c) the JIT-compiled
rollout produced no NaNs, (d) PPO inputs (rewards / values / advantages /
returns) have reasonable means and stds, (e) gradient norms are nonzero and
below clip, and (f) PPO updates didn't blow anything up. After the first
cycle, the same block prints once per minibatch so you can see exactly which
PPO update introduced a numerical event, should one occur.

---

## Project Layout

```
throng/
├── config_phase7.yaml         # Active hyperparameters
├── jax_sim/                   # ★ Active code
│   ├── main_jax.py            # Outer loop, PPO orchestration, telemetry,
│   │                          #   brain-vote, curriculum, checkpointing
│   ├── network_jax.py         # AgentNetworkJax — Flax transformer + heads
│   ├── rl_jax.py              # PPO + GAE + minibatch updates
│   ├── grid_jax.py            # GridState + obs builder + world generation
│   ├── population_jax.py      # PopState + reproduction + mind-meld
│   └── debug_metrics.py       # Sanity panel
├── agents/, environment/, communication/, utils/   # Legacy PyTorch
├── main.py                    # Legacy CLI
├── THRONG.md                  # Full research log + theory + roadmap
└── README.md                  # You are here
```

---

## Configuration Cheatsheet

All knobs live in `config_phase7.yaml`. The ones you actually touch:

| Knob | What it controls | Default |
|---|---|---|
| `population_size` | Blue agents (max alive) | 500 |
| `red_population_size` | Red agents (max alive) | 75 |
| `grid_size` | World edge length | 128 |
| `n_layers` | Initial transformer depth | 2–4 |
| `brain_max_layers` | Cap for brain-vote expansion | 6 |
| `brain_token_dim` | Transformer hidden | 128 |
| `signal_dim` | Continuous signal embed | 32 |
| `signal_vocab_size` | Discrete signal vocab | 64 |
| `symbol_dim` | Symbol / cultural vector dim | 16 |
| `neighbor_k` | Visible neighbours | 6 |
| `local_obs_radius` | Half-width of local patch (2 → 5×5) | 2 |
| `memory_buffer_size` | Episodic memory slots | 20 |
| `culture_fast_decay` | 0.90 | recent-danger half-life ≈ 7 steps |
| `culture_slow_decay` | 0.995 | landmark half-life ≈ 140 steps |
| `ppo_rollout_steps` | Steps between PPO updates | 512 |
| `ppo_minibatch_size` | PPO minibatch (touch this for OOM) | 2048 |
| `repro_energy_thresh` / `_cost` | High-energy self-cloning rule | 0.80 / 0.40 |
| `min_population` | Floor-enforced respawn | 200 |
| `curriculum_survival_threshold` | Sustain blue surv to advance reds | 0.80 |
| `curriculum_sustain_updates` | Consecutive updates required | 5 |
| `brain_vote_interval` | Steps between capacity checks | 5000 |
| `brain_vote_survival_threshold` | Survival level that gates a layer add | 0.55 |
| `signal_gate_prob` | Fraction of self-obs randomly masked | 0.5 |
| `resource_obs_noise` | Gaussian σ on resource readings | 0.2 |
| `red_starvation_steps` | Steps without a catch before a red starves | 400 |

---

## Roadmap

**Authoritative roadmap:** [THRONG.md §11](THRONG.md#11-roadmap-whats-next).

| Phase | Status |
|-------|--------|
| **9.x–11.3** | Merged to `master` — cross-attn, confidence, carry fwd, epistemic gate |
| **12.0–12.2** | **LIVE** on `feature/phase12-red-coevolution` — red VQ + wiretap + `--red` decode |
| **12.2 decode @ ~30k** | **FAIL** — babbling (LRT p=0.9767); **holding pattern** |
| **13.0+** | **BLOCKED** — metabolic execution tax, inscription, proprio aux — **after** red pincer χ² passes |

Do **not** branch `feature/phase13-thermodynamics` until red spatial coordination is proven.

---

## Historical roadmap (Phase 9 plan — superseded)

The sections below document the original Phase 9 design intent. Most are **done**
on `master`. See THRONG.md for current work.

- Transformer brains, capacity-based brain-vote (2L → 6L).
- Eight environment channels, dual cultural grids, scent trails, contested
  nodes, shelter, puzzles.
- Discrete + continuous signals; **signal propagation fixed and live (May 28
  2026)**.
- Red curriculum with survival-gated graduation `[6, 15, 30, 75]`.
- Episodic memory buffer per agent (20 slots).
- Theory-of-Mind head predicting neighbour actions.
- MAPPO + GAE + value clipping + grad clipping; numerically stable.
- Orbax checkpointing (params-only; population restarts on resume).
- Distillation pass (currently age-based; "human among apes" version queued).
- Sanity-DEBUG panel that surfaces NaN / gradient anomalies the moment they
  happen.

### 🎯 Next, in order — Phase 9

These are organised by *cost and risk*, so the cheap wins land first.

**Phase 9.1 — Self-Model + Metacognition** (1–2 days)
- `head_self_action(h_t)` predicting the agent's own next action.
- Auxiliary cross-entropy loss against the actually sampled action.
- Confidence head + low-confidence → "help" signal mode.
- Gated mind-meld (only blend carries when both confident).

**Phase 9.2 — Forward Dynamics Head** (~2 days)
- `head_fwd(h_t, a_t, pooled_neighbour_signals) → predicted_obs_{t+1}`.
- MSE auxiliary loss; coefficient ramps 0.05 → 0.2 over 50k steps.
- **The signal channel is included in the input**, so signals become
  load-bearing for prediction — a sharper selection pressure than survival.
- Falsifiable test: ablating signals from `head_fwd` should *increase*
  prediction MSE. If it doesn't, signals are still cosmetic and the
  environment needs to be tightened further.

**Phase 9.3 — Dreamer / Imagination Loop** (gated on 9.2 working)
- At inference, use `head_fwd` + critic to score each candidate action by
  imagined-2-step return.
- Replace `argmax(action_logits)` with `argmax_a vf(fwd(h, a))`.
- Fixed branching factor (5 actions × depth K) — JIT-friendly.

**Phase 9.4 — Communication Upgrades** (post-9.3)
- Multi-head attention over neighbour signals (replace mean-pool).
- Variable-length message sequences (2–5 tokens / step).
- Adversarial reds that can mimic blue signals.

### 📊 Dashboard metrics queued

- **Signal clusters (k-means, k=16)** — effective cluster count + occupancy
  entropy. When this drops from ≈ N_alive toward a small fixed number with
  survival still ≥80%, **a vocabulary is forming**.
- **Self-prediction accuracy** (after Phase 9.1).
- **Forward-dynamics MSE + signal-ablation Δ** (after Phase 9.2). This Δ is
  the single sharpest test of whether communication is real.

### 🗂️ Deprioritised / on ice

- Hierarchical RL (redundant with carry + value head).
- Crafting / construction (high engineering, weak hypothesis).
- Generic "scale up before adding architecture" — the current 64×64 / N=100
  setup is sufficient to observe vocabulary formation if the architecture is
  right; scale is not the bottleneck.

---

## Philosophy in One Paragraph

Language and intelligence in biological organisms were not designed. They
were *grown* under information asymmetry — animals that needed to coordinate
to survive, in environments that did not hand them concepts. THRONG is the
hypothesis that this growth process can be replicated computationally if four
ingredients are present: (1) partial observability sharp enough to make
silence costly, (2) coordination pressure that makes cooperation pay, (3) a
communication channel that is load-bearing for at least one downstream task,
and (4) cumulative memory so that a discovery by one agent can be inherited
by the next generation. We've built (1), (2), and (4). Phase 9.2's
signal-conditioned forward-dynamics head is what makes (3) *provably* the
case for the first time.

For the full version, with citations, see [THRONG.md](THRONG.md) — sections
"Philosophical Foundations" and "The Architecture of Thought".

---

## Running Tips

- **OOM on P100 / T4?** Drop `ppo_minibatch_size` to 256, `n_layers` to 2,
  `grid_size` to 64.
- **First run takes 5–10 minutes to start producing steps** — XLA is
  compiling the entire `lax.scan` rollout into one kernel. After that,
  steps/sec is constant.
- **Checkpoints resume populations from scratch** but restore params — this
  is intentional. Long-run training is robust to interrupted populations
  because MAPPO learns from the shared policy, not from individual lifelines.
- **If `Signals: 1 unique` persists for more than 5k steps after a fresh
  pull**, your build is from before commit `82a1c7f` and is still suffering
  from the signal-propagation bug. `git pull && rm -rf runs/jax_run/checkpoints`.

---

## License & Citation

Research project, MIT-style permissive license (see repo root). If you build
on THRONG, please cite the repository and the relevant phase in `THRONG.md`.

---

*"Give them a rich enough world and a reason to talk, then get out of the way."*
