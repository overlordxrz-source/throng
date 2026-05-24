"""Debug script to check PPO metrics on Kaggle."""
import jax
import jax.numpy as jnp
from jax_sim.rl_jax import ppo_loss, compute_gae
from jax_sim.network_jax import AgentNetworkJax
from jax_sim.main_jax import build_observations, init_population, init_grid
import yaml

with open("config_phase7.yaml") as f:
    cfg = yaml.safe_load(f)

cfg = {
    "grid_size": cfg["grid_size"],
    "neighbor_k": cfg["neighbor_k"],
    "local_obs_radius": cfg["local_obs_radius"],
    "symbol_dim": cfg["symbol_dim"],
    "signal_dim": cfg["signal_dim"],
    "max_age": cfg["max_age"],
    "agent_hidden_dim": cfg["agent_hidden_dim"],
    "vocab_size": cfg.get("vocab_size", 32),
    "memory_slots": cfg.get("memory_slots", 20),
    "token_dim": cfg.get("brain_token_dim", 128),
    "n_layers": cfg["n_layers"],
}

gs = cfg["grid_size"]
key = jax.random.PRNGKey(42)

# Init grid + populations
grid = init_grid(gs, key)
b_pop = init_population(cfg["grid_size"], 500, key, 256, 20)
r_pop = init_population(cfg["grid_size"], 75, key, 256, 20)

# Build obs
blue_map = jnp.zeros((gs, gs), dtype=jnp.int32)
red_map = jnp.zeros((gs, gs), dtype=jnp.int32)
b_obs = build_observations(b_pop, grid, blue_map, red_map, cfg, step=0)

# Init model
model = AgentNetworkJax(
    obs_dim=b_obs.shape[-1],
    hidden_dim=cfg["agent_hidden_dim"],
    vocab_size=cfg["vocab_size"],
    sym_dim=cfg["symbol_dim"],
    n_heads=cfg.get("brain_n_heads", 4),
)
key, init_key = jax.random.split(key)
dummy_carry = jnp.zeros((500, cfg["agent_hidden_dim"]))
params = model.init(init_key, dummy_carry, b_obs, n_layers=cfg["n_layers"])

# Forward pass
key, act_key = jax.random.split(key)
action_logits, signal_logits, sym_w, values, tom, tok, sig_p, c_f, c_s, new_carry = model.apply(
    params, dummy_carry, b_obs, n_layers=cfg["n_layers"]
)

# Sample actions
action_keys = jax.random.split(act_key, 500)
actions = jax.vmap(jax.random.categorical)(action_keys, action_logits)

# Compute log probs
log_probs = jax.nn.log_softmax(action_logits, axis=-1)
log_probs_taken = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

# Fake rollout data (T=2, N=500)
T = 2
obs = jnp.stack([b_obs, b_obs])
carries = jnp.stack([dummy_carry, new_carry, new_carry])
actions_batch = jnp.stack([actions, actions])
old_log_probs = jnp.stack([log_probs_taken, log_probs_taken])
rewards = jnp.ones((T, 500)) * 0.05  # constant reward
values = jnp.stack([values, values])
dones = jnp.zeros((T, 500))
alive = jnp.ones((T, 500))

# GAE
advantages, returns = compute_gae(rewards, values, dones)
print(f"adv_mean: {advantages.mean():.6f}, adv_std: {advantages.std():.6f}")
print(f"adv_min: {advantages.min():.6f}, adv_max: {advantages.max():.6f}")

# PPO loss
loss, metrics = ppo_loss(
    params, model.apply, obs, actions_batch, old_log_probs,
    advantages, returns, carries, cfg["n_layers"], alive=alive,
)

print(f"\nloss: {loss:.6f}")
for k, v in metrics.items():
    print(f"  {k}: {v:.6f}")

# Check for NaN
print(f"\nAny NaN in action_logits: {jnp.isnan(action_logits).any()}")
print(f"Any NaN in values: {jnp.isnan(values).any()}")
print(f"Any NaN in log_probs: {jnp.isnan(old_log_probs).any()}")
print(f"Any NaN in advantages: {jnp.isnan(advantages).any()}")
print(f"Any NaN in returns: {jnp.isnan(returns).any()}")

# Raw entropy check
probs = jax.nn.softmax(action_logits, axis=-1)
entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
print(f"\nRaw entropy mean: {entropy.mean():.6f}")
print(f"Raw entropy min: {entropy.min():.6f}")
print(f"Raw entropy max: {entropy.max():.6f}")
