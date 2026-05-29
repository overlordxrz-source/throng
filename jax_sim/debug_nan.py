"""Debug NaN issue in JAX model."""
import jax
import jax.numpy as jnp
import yaml
from jax_sim.observations_jax import build_observations_jax
from jax_sim.main_jax import init_population, init_grid
from jax_sim.network_jax import AgentNetworkJax

with open('config_phase7.yaml') as f:
    cfg = yaml.safe_load(f)

cfg_jax = {
    'grid_size': cfg['grid_size'],
    'neighbor_k': cfg['neighbor_k'],
    'local_obs_radius': cfg['local_obs_radius'],
    'symbol_dim': cfg['symbol_dim'],
    'signal_dim': cfg['signal_dim'],
    'max_age': cfg['max_age'],
    'agent_hidden_dim': cfg['agent_hidden_dim'],
    'vocab_size': cfg.get('vocab_size', 32),
    'memory_slots': cfg.get('memory_slots', 20),
    'token_dim': cfg.get('brain_token_dim', 128),
    'n_layers': cfg['n_layers'],
}

key = jax.random.PRNGKey(42)
gs = cfg_jax['grid_size']

grid = init_grid(gs, key)
b_pop = init_population(gs, 500, key, 256, 20)
r_pop = init_population(gs, 75, key, 256, 20)

blue_map = jnp.zeros((gs, gs), dtype=jnp.int32)
red_map = jnp.zeros((gs, gs), dtype=jnp.int32)

b_obs = build_observations_jax(b_pop, grid, blue_map, red_map, cfg_jax, 0)
print('obs NaN:', jnp.isnan(b_obs).any(), 'obs Inf:', jnp.isinf(b_obs).any())
print('obs min:', float(b_obs.min()), 'obs max:', float(b_obs.max()))
print('obs shape:', b_obs.shape)

model = AgentNetworkJax(
    obs_dim=b_obs.shape[-1],
    hidden_dim=cfg_jax['agent_hidden_dim'],
    vocab_size=cfg_jax['vocab_size'],
    sym_dim=cfg_jax['symbol_dim'],
    n_heads=cfg.get('brain_n_heads', 4),
)

key, init_key = jax.random.split(key)
dummy_carry = jnp.zeros((500, cfg_jax['agent_hidden_dim']))
print('Initializing model...')
params = model.init(init_key, dummy_carry, b_obs, n_layers=cfg_jax['n_layers'])
print('Model initialized')

# Check params for NaN
flat_params = jax.tree_util.tree_leaves(params)
has_nan_params = any(jnp.isnan(p).any() for p in flat_params)
print('Params NaN:', has_nan_params)

# Forward pass
print('Forward pass...')
new_carry, outs = model.apply(params, dummy_carry, b_obs, n_layers=cfg_jax['n_layers'])
action_logits, signal_out, sym_w, vals, tom, tok, loss_vq, _z_e, c_f, c_s = outs

print('action_logits NaN:', bool(jnp.isnan(action_logits).any()), 'max:', float(jnp.max(jnp.abs(action_logits))))
print('values NaN:', bool(jnp.isnan(vals).any()), 'max:', float(jnp.max(jnp.abs(vals))))
print('new_carry NaN:', bool(jnp.isnan(new_carry).any()), 'max:', float(jnp.max(jnp.abs(new_carry))))

# Check which agents have NaN
nan_agents = jnp.isnan(action_logits).any(axis=-1)
print('NaN agents:', int(nan_agents.sum()), 'out of', len(nan_agents))
if nan_agents.any():
    nan_idx = jnp.where(nan_agents)[0][0]
    print('First NaN agent obs has NaN:', bool(jnp.isnan(b_obs[nan_idx]).any()))
    print('First NaN agent alive:', bool(b_pop.alive[nan_idx]))
    print('First NaN agent carry has NaN:', bool(jnp.isnan(dummy_carry[nan_idx]).any()))
