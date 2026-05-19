# THRONG — Emergent Artificial Life Simulation

THRONG is a continuous artificial life simulation where a population of neural agents
evolve communication, spatial strategy, and cooperative behaviour **from scratch** — no
human-defined fitness function, no reward shaping.  Emergent structure is measured via
**mutual information (MI)** between agent signal dimensions and environmental features.
Rising MI curves mean language is appearing.

---

## Quick Start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies  (JAX CPU build shown; swap jaxlib for GPU if needed)
pip install -r requirements.txt

# 3. Run with default config
python main.py

# 4. Run headless (no pygame, logging only — good for servers)
python main.py --headless

# 5. Custom config
python main.py --config myconfig.yaml

# 6. Resume from checkpoint
python main.py --resume runs/run_001/checkpoint_50000.pkl

# 7. Open analytics dashboard on a saved run
python main.py --analyze runs/run_001/
```

---

## Project Layout

```
throng/
├── main.py                  # CLI entry point + main simulation loop
├── config.yaml              # All hyperparameters (edit here, not in code)
├── environment/
│   ├── grid.py              # 2D toroidal resource grid
│   └── resource.py          # Gaussian-cluster patch generation + regeneration
├── agents/
│   ├── genome.py            # Param flatten/unflatten, mutation, crossover
│   ├── network.py           # Flax GRU agent brain (vmappable)
│   └── population.py        # Full population state + lifecycle management
├── evolution/
│   ├── selection.py         # Tournament + elitism selection
│   └── metrics.py           # Fitness tracking, lineage trees
├── communication/
│   ├── channel.py           # k-NN signal aggregation
│   └── analysis.py          # Mutual information (sklearn) + k-means clustering
├── visualization/
│   ├── renderer.py          # Pygame real-time grid render
│   └── dashboard.py         # Matplotlib analytics (MI curves, population stats)
└── utils/
    ├── logging.py            # Structured JSON run logs
    └── checkpointing.py      # Save/load full simulation state
```

---

## What the MI Curves Mean

Each agent broadcasts a 16-dimensional **signal vector** (tanh-bounded) to its 6
nearest neighbours.  Every 1000 simulation steps, THRONG computes the
**mutual information** between each of those 16 signal dimensions and four
environmental features:

| Feature | What it captures |
|---|---|
| `local_resource` | Is this signal dimension encoding food availability? |
| `neighbor_count` | Is it encoding crowding / competition? |
| `own_energy` | Is it encoding urgency / health? |
| `dist_to_red` | Is it encoding threat proximity? (post-generation 500) |

A **flat MI curve** means random noise — agents aren't communicating anything useful.
A **rising MI curve** means a signal dimension has become correlated with a real
environmental variable — proto-language is emerging.
When multiple agents converge on the same signal pattern for the same context (visible
in the UMAP cluster view), discrete "words" have crystallised.

---

## Reading the Visualisation

**Pygame window (left):**
- **Green intensity** → resource level of each grid cell
- **Blue dots** → main population agents; brightness = energy
- **Red dots** → competitor population (spawns at evolution step 500)
- **Thin lines** → drawn between agents whose latest signal vectors have
  cosine similarity > 0.8 (i.e. they are "saying the same thing")
- **HUD** (top-left) → step, population count, mean energy, oldest lineage age

**Matplotlib dashboard (right, updates every 30 s):**
- Population count over time
- Mean fitness over time
- MI curves per signal dimension (all 4 env features, 4 subplots)
- Lineage tree of top-10 longest-surviving lineages
- 2D UMAP of recent signal vectors coloured by context

---

## Configuration

All parameters live in `config.yaml`.  Key knobs:

- `mutation_sigma_small / mutation_sigma_large` — evolutionary pressure
- `resource_regen_rate` — environmental richness (lower = more competition)
- `neighbor_k` — communication radius
- `evolution_interval` — how often selection pressure applies
- `competitor_spawn_generation` — when Red population arrives

---

## GPU / Accelerator Support

JAX will automatically use any available GPU/TPU.  For GPU:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Checkpoints

Checkpoints are saved every `checkpoint_interval` steps (default 10 000) as
pickle files under `runs/run_XXX/`.  They contain the full simulation state
including all agent parameters, GRU hidden states, grid state, and metrics
history — enough to resume identically.
