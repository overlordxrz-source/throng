"""Quick smoke test — runs 10 simulation steps headless, no pygame."""
import sys
sys.path.insert(0, ".")

import yaml
import jax
import jax.numpy as jnp
import numpy as np

from agents.network import AgentNetwork, make_forward_fn, INPUT_DIM
from agents.population import create_population, kill_agent
from environment.grid import ToroidalGrid
from environment.resource import ResourceManager
from evolution.selection import run_evolution_step, compute_fitness
from evolution.metrics import MetricsTracker
from communication.channel import aggregate_neighbour_signals

# ── Config ────────────────────────────────────────────────────────────────────
config = yaml.safe_load(open("config.yaml"))
config["population_size"] = 32
config["max_population"]  = 50
config["min_population"]  = 10
config["log_dir"]         = "/tmp/throng_test"

# ── Model ─────────────────────────────────────────────────────────────────────
model      = AgentNetwork(hidden_dim=64, signal_dim=16, n_actions=6)
forward_fn = make_forward_fn(model)

# ── Population ────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(42)
pop = create_population(config, model, key, config["grid_size"])
print(f"[OK] Population created — {pop.alive.sum()} alive agents")

# ── Grid + Resources ──────────────────────────────────────────────────────────
grid = ToroidalGrid(size=128)
rm   = ResourceManager(
    grid_size=128, n_clusters=8, regen_rate=0.002, diffuse_sigma=0.3
)
rng = np.random.default_rng(0)
rm.initialise(rng)
for _ in range(100):
    rm.regenerate(grid.resources)
print(f"[OK] Grid initialised — resource range [{grid.resources.min():.3f}, {grid.resources.max():.3f}]")

# ── Simulate 10 steps ────────────────────────────────────────────────────────
print("\nRunning 10 simulation steps...")
for step in range(1, 11):
    # Presence maps
    blue_map, red_map = grid.build_presence_maps(
        pop.positions, pop.alive, pop.team
    )

    # Observations (vectorised)
    obs = grid.get_all_local_views(
        pop.positions, blue_map, red_map, radius=2
    )  # (max_pop, 50)

    # Neighbour signal aggregation
    recv = aggregate_neighbour_signals(
        pop.positions, pop.signals, pop.alive, k=6, grid_size=128
    )  # (max_pop, 16)

    # Own state
    own_s = np.stack([
        pop.energies  / 200.0,
        pop.ages      / 5000.0,
        pop.cooldowns / 50.0,
    ], axis=1).astype(np.float32)  # (max_pop, 3)

    # Forward pass (vmap over full population)
    inputs = jnp.array(np.concatenate([obs, recv, own_s], axis=1))
    new_carries, (logits, signals_out, repro_out) = forward_fn(
        pop.params, pop.carries, inputs
    )

    pop.carries = new_carries
    pop.signals = np.asarray(signals_out, dtype=np.float32)

    # Apply existence cost and age
    pop.energies[pop.alive] -= config["energy_exist_cost"]
    pop.ages[pop.alive]     += 1

    # Kill depleted
    dead = pop.alive & (pop.energies <= 0)
    for idx in np.where(dead)[0]:
        kill_agent(pop, int(idx))

    # Resource regen
    rm.regenerate(grid.resources)

    n_alive    = int(pop.alive.sum())
    mean_e     = float(pop.energies[pop.alive].mean()) if n_alive else 0.0
    action_dist = np.bincount(np.asarray(jnp.argmax(logits, axis=-1)), minlength=6)
    print(
        f"  step {step:>2}  alive={n_alive:>3}  mean_energy={mean_e:.2f}"
        f"  actions={action_dist.tolist()}"
    )

# ── Evolution step ────────────────────────────────────────────────────────────
print("\nRunning evolution step...")
key, evo_key = jax.random.split(key)
pop, key, evo_stats = run_evolution_step(pop, config, evo_key, model, 128, evo_count=0)
print(f"[OK] Evolution: {evo_stats}")

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics = MetricsTracker()
metrics.record_step(10, pop.alive, pop.energies, pop.ages, pop.total_consumed, pop.lineage_ids)
print(f"[OK] Metrics: pop_history={metrics.population_history}")

print("\n✓  Smoke test passed — THRONG is ready.")
