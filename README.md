# THRONG v2 — Emergent Communication & Cultural Memory

THRONG is a multi-agent artificial life simulation where populations of neural agents
evolves **language, cooperation, and shared knowledge from scratch** — no human-defined
rewards, no pre-programmed behaviours. Agents learn to survive, communicate, and build
collective memory through natural selection pressure.

**Current Phase:** 7.5 — Dual Cultural Memory Grids + Red Co-evolution  
**Framework:** JAX + Flax (fully JIT-compiled simulation)  
**Obs Dim:** 2,310 per agent  
**Device:** GPU (T4 on Kaggle)

---

## Quick Start

### JAX Simulation (current — recommended)

```bash
# 1. Install dependencies
pip install jax[cuda12] flax optax pyyaml wandb

# 2. Run from Python
from jax_sim.main_jax import run_simulation
import yaml
with open("config_phase7.yaml") as f:
    cfg = yaml.safe_load(f)
final_params, metrics = run_simulation(cfg, seed=42, n_steps=1_000_000)
```

### Legacy PyTorch (still available)

```bash
pip install -r requirements.txt
python main.py --config config_phase7.yaml --headless --fresh
```

### Kaggle Notebook (recommended for long runs)

```python
# Cell 1 — setup
!git clone https://github.com/overlordxrz-source/throng.git /kaggle/working/throng 2>/dev/null || true
%cd /kaggle/working/throng
!git reset --hard
!git pull origin master
!pip install -q jax[cuda12] flax optax pyyaml wandb

# Cell 2 — run
import sys
sys.path.insert(0, "/kaggle/working/throng")
import yaml
from jax_sim.main_jax import run_simulation

with open("/kaggle/working/throng/config_phase7.yaml", "r") as f:
    cfg = yaml.safe_load(f)
cfg["ppo_minibatch_size"] = 512  # prevent P100 OOM
cfg["use_pmap"] = False          # single GPU mode

total_steps = 1_000_000
final_params, metrics = run_simulation(cfg, seed=42, n_steps=total_steps)
```

---

## Architecture Overview

### The World

A 128×128 toroidal grid with rich environmental layers:

| Layer | What it is | Why it matters |
|-------|-----------|----------------|
| **Resources** | Gaussian food patches | Agents need energy to survive |
| **Walls** | Procedural cave-like barriers | Navigation challenge |
| **Shelter spots** | Safe zones (double red detection radius) | Strategic locations |
| **Contested nodes** | High-value resources worth fighting over | Competition hotspots |
| **Scent trails** | Fading traces left by red predators | Danger detection |
| **Symbols** | Graffiti agents draw on the ground | Persistent communication |
| **Cultural Fast** (0.90 decay) | Recent danger / coordination traces | "Red was here 5 steps ago" |
| **Cultural Slow** (0.995 decay) | Stable landmarks / long-term memory | "This valley is safe" |

### The Agent Brain

Each agent has a **transformer-based neural network** (not a GRU anymore) with:

- **4-6 attention layers** (curriculum expands from 4 → 6)
- **128-dim tokens** processed through multi-head self-attention
- **Recurrent carry state** (256-dim) passed between steps

### Input Tokens (what the agent "sees")

Every step, the agent receives a flat observation vector (2,285 numbers) split into semantic tokens:

| Token Group | Dimensions | What it encodes |
|-------------|-----------|-----------------|
| Own state | 6 | Energy, age, neighbor count, etc. |
| Neighbor signals | 6 × 32 = 192 | What nearby agents are saying |
| Local symbols | 25 × 16 = 400 | Graffiti in 5×5 window |
| Environment | 25 × 7 = 175 | Walls, food, shelter, contested, scent, presence |
| Own signal | 32 | What I just broadcast |
| Episodic memory | 20 × 34 = 680 | My diary of last 20 events |
| Cultural Fast | 25 × 16 = 400 | Recent danger traces around me |
| Cultural Slow | 25 × 16 = 400 | Long-term landmarks around me |

### Output Heads (what the agent "decides")

| Head | Output | Purpose |
|------|--------|---------|
| `head_action` | 5 logits | Move N/S/E/W/Stay |
| `head_signal` | 256 vocab logits | Which "word" to broadcast |
| `head_symbol` | 16-dim tanh | What graffiti to draw |
| `head_culture_fast` | 16-dim tanh | Danger trace to deposit |
| `head_culture_slow` | 16-dim tanh | Landmark to record |
| `head_value` | scalar | How good is my situation? |
| `tom_head` | 5 logits × 6 neighbors | Predict what each neighbor will do |

### Learning: MAPPO + Survival Pressure

- **MAPPO (Multi-Agent PPO)** — reinforcement learning where the reward is simply *"did you survive?"*
- **Theory-of-Mind head** — learns to predict neighbor actions (trained on actual neighbor moves)
- **Signal entropy bonus** — encourages exploration of the communication space
- **Curriculum** — reds start at 6 agents, ramp to 75 as blues prove they can survive

---

## Project Layout

```
throng/
├── config_phase7.yaml           # All hyperparameters
├── jax_sim/                     # ★ Active — JAX implementation
│   ├── main_jax.py              # Simulation loop + rollout (full @jax.jit via lax.scan)
│   ├── network_jax.py           # AgentNetworkJax (Flax transformer)
│   ├── rl_jax.py                # PPO loss, GAE, minibatch updates
│   ├── grid_jax.py              # Grid state, obs building, puzzle mechanics
│   └── population_jax.py        # PopState, reproduction, mind-meld
├── main.py                      # Legacy PyTorch CLI
├── environment/                 # Legacy PyTorch grid
├── agents/                      # Legacy PyTorch network + RL
├── communication/
│   ├── channel.py               # Signal aggregation
│   └── analysis.py              # MI, compositionality, culture metrics
├── utils/
│   ├── logging.py               # Structured JSON logs
│   └── checkpointing.py         # Full state save/load
└── README.md                    # You are here
```

---

## What the Metrics Mean

### Mutual Information (MI)

Every ~1,000 steps, THRONG runs a logistic regression from each of the 32 signal
embedding dimensions against environmental features. **Higher MI = that dimension
correlates with something real in the world.**

Example: `dim15/'own_energy'=0.12` means signal dimension 15 is weakly encoding
how much energy the agent has. When MI reaches ~0.3+, it's a reliable signal.

### Culture Metrics

| Metric | Meaning |
|--------|---------|
| `H` | Entropy of symbol distribution (higher = more diversity) |
| `surv_corr` | Correlation between symbol patterns and survival |
| `r@1, r@5, r@10` | Recall: how well can we predict red positions from cultural traces? |

### Chain Depth

How many hops does a signal travel? `max=0` means no relay chains yet. When
`hops>1 > 0`, agents are repeating messages they heard from others — **rumour chains**.

### NB_GAIN_SURV

Spearman correlation between "did I listen to neighbors?" (signal gain) and
"did I survive?" A value near 1.0 means **listening to others is strongly
selected for** — the population has discovered communication is useful.

---

## Configuration

All parameters live in `config_phase7.yaml`. Key knobs:

| Parameter | What it does | Current |
|-----------|-------------|---------|
| `population_size` | Blue agents | 500 |
| `red_population_size` | Red predators | 75 |
| `n_layers` | Transformer layers (expands via curriculum) | 4 |
| `brain_token_dim` | Transformer hidden size | 128 |
| `signal_dim` | Continuous signal embedding size | 32 |
| `symbol_dim` | Symbol/cultural vector size | 16 |
| `neighbor_k` | How many neighbors each agent observes | 6 |
| `local_obs_radius` | Window size for grid patches | 2 (5×5) |
| `memory_buffer_size` | Episodic memory slots per agent | 20 |
| `culture_fast_decay` | How fast recent danger fades | 0.90 |
| `culture_slow_decay` | How fast landmarks fade | 0.995 |
| `ppo_rollout_steps` | Steps between PPO updates | 512 |
| `ppo_minibatch_size` | GPU batch size for PPO | 2048 |
| `min_population` | Blue population floor (auto-respawn) | 200 |
| `repro_energy_thresh` | Energy needed to self-clone | 0.80 |
| `repro_energy_cost` | Energy cost of cloning | 0.40 |
| `curriculum_survival_threshold` | Blue survival rate to advance red stage | 0.80 |
| `curriculum_sustain_updates` | Consecutive PPO updates above threshold | 5 |

---

## Current State & Roadmap

### ✅ What Works Now

- [x] Transformer-based agent brains (4-6 layers)
- [x] Discrete vocab communication (256 tokens)
- [x] Episodic memory buffer per agent (20 slots)
- [x] Theory-of-Mind prediction head
- [x] Dual cultural memory grids (fast danger + slow landmarks)
- [x] Red co-evolution with curriculum (6 → 75 agents)
- [x] MAPPO training with survival-only reward
- [x] Shelter, contested resources, scent trails
- [x] Checkpoint save/load with full state
- [x] MI, compositionality, culture, chain-depth metrics

### 🔄 In Progress

- [ ] First successful long run on Kaggle (1M+ steps) with JAX
- [ ] Measure actual steps/sec on T4
- [ ] Verify cultural memory emergence and signal-to-survival correlation

### ✅ Completed: JAX Rewrite

The entire simulation is now JAX-native. Key facts:
- **`sim_step`** decorated with `@jax.jit`, executed via `jax.lax.scan` — one compiled XLA program
- **Everything inside JIT**: obs building, forward pass, movement, wall collision, resource consumption, energy decay, reproduction, mind-meld, catch detection, puzzles, rewards, culture writes/decay
- **Outside JIT** (intentional): red curriculum reproduction (dynamic `min_pop`), PPO gradient updates (own JIT), telemetry
- Red curriculum stages: `[6, 15, 30, 75]` — advances when blues sustain ≥80% survival

### 🎯 Next

- **Richer signals** — continuous 32-dim vectors instead of discrete tokens
- **Scale up** — 256×256 grid, 1000+ agents, once performance is confirmed
- **Deception / tool use** — can agents learn to lie or use objects?

---

## GPU Notes

Current target: **T4 GPU** (Kaggle free tier). With the JAX rewrite, the entire
simulation step is compiled to a single XLA kernel — no Python loop bottleneck.

For JAX GPU install:
```bash
pip install jax[cuda12] flax optax
```

---

## Checkpoints

**JAX:** Saved via Orbax every N updates to `runs/main_run/checkpoints/`.
Contains: grid state, both populations, both param trees, optimizer states, carries.
Automatically resumes from latest checkpoint on restart.

**Legacy PyTorch:** Saved as `checkpoint_{step}.pkl` under `runs/run_XXX/`.
Resume: `python main.py --resume runs/run_XXX/checkpoint_latest.pkl`

---

## Design Philosophy

> "Give them a rich enough world and a reason to talk, then get out of the way."

THRONG is built on the hypothesis that **intelligence and language emerge from
environmental pressure**, not architectural tricks. We don't:
- Pre-define what signals mean
- Reward agents for communicating
- Hard-code cooperative behaviours

Instead, we create a world where survival is genuinely hard, give agents the
capacity to sense, remember, and broadcast, and let selection do the rest.

The cultural memory grids (Phase 7.5) are the newest addition — a shared
writable layer that persists across agent lifetimes. It's the first step toward
true cumulative culture: one agent's hard-won knowledge can outlive them and
benefit the tribe.
