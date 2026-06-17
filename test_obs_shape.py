import jax
import jax.numpy as jnp
from jax_sim.observations_jax import build_observations_jax
from jax_sim.grid_jax import GridState
from jax_sim.population_jax import init_population

gs = 50
W = 25
N = 10
grid = GridState(gs, 16)
grid = grid.replace(walls=jnp.zeros((gs, gs), dtype=jnp.bool_))

key = jax.random.PRNGKey(42)
pop = init_population(N, 128, 32, gs, 0, key, N, memory_slots=0)

config = {
    "grid_size": gs,
    "neighbor_k": 6,
    "local_obs_radius": 2,
    "max_age": 100,
    "signal_dim": 32,
    "symbol_dim": 16,
    "env_channels": 9
}

obs = build_observations_jax(pop, grid, jnp.zeros((gs, gs), dtype=jnp.bool_), jnp.zeros((gs, gs), dtype=jnp.bool_), config, 0, blue_bg_map=jnp.zeros((gs, gs), dtype=jnp.bool_))
print("obs shape:", obs.shape)
