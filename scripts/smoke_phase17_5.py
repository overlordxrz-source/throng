import yaml
import jax
import jax.numpy as jnp
from pprint import pprint

from jax_sim.grid_jax import GridState
from jax_sim.population_jax import init_population
from jax_sim.main_jax import run_simulation
from jax_sim.observations_jax import build_observations_jax
from jax_sim.network_jax import AgentNetworkJax, PredatorNetworkJax
from jax_sim.obs_layout import make_obs_layout

def smoke_test_phase_17_5():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    config['n_steps'] = 2
    config['gui'] = False
    config['use_wandb'] = False
    config['grid_size'] = 32
    config['pop_size'] = 8
    config['red_pop_size'] = 2
    config['signal_dim'] = 32

    rng = jax.random.PRNGKey(42)
    k_grid, k_pop, k_model = jax.random.split(rng, 3)
    
    grid = GridState(config['grid_size'], symbol_dim=config.get('symbol_dim', 8))
    
    blue_pop = init_population(
        config['pop_size'], config.get('agent_hidden_dim', 256), config['signal_dim'], 
        config['grid_size'], team_id=0, key=k_pop, n_agents=config['pop_size']
    )
    red_pop = init_population(
        config['red_pop_size'], config.get('agent_hidden_dim', 256), config['signal_dim'], 
        config['grid_size'], team_id=1, key=k_pop, n_agents=config['red_pop_size']
    )

    assert blue_pop.alarms.shape == (config['pop_size'], 2), f"Expected {(config['pop_size'], 2)}, got {blue_pop.alarms.shape}"
    assert red_pop.alarms.shape == (config['red_pop_size'], 2)

    _layout = make_obs_layout(
        signal_dim=config["signal_dim"],
        symbol_dim=config.get("symbol_dim", 8),
        memory_slots=config.get("memory_slots", 0),
        neighbor_k=config["neighbor_k"],
        local_cells=(2 * config["local_obs_radius"] + 1)**2,
        env_channels=int(config.get("env_channels", 10)),
        own_state_dim=int(config.get("own_state_dim", 10)),
    )
    obs_dim = _layout.total_dim

    model_blue = AgentNetworkJax(
        hidden_dim=config.get('agent_hidden_dim', 256),
        n_heads=config.get("n_heads", 4),
        n_layers=1,
        obs_dim=obs_dim,
        signal_dim=config["signal_dim"],
        symbol_dim=config.get("symbol_dim", 8),
        vocab_size=config.get("vocab_size", 64),
        vq_beta=0.25,
        vq_dead_code_reset=True,
        memory_slots=config.get("memory_slots", 0),
        fwd_env_dim=_layout.loc_env_end - _layout.loc_env_start,
        cross_attn_enabled=False,
        cross_attn_num_heads=config.get("n_heads", 4),
        env_channels=int(config.get("env_channels", 10)),
        own_state_dim=int(config.get("own_state_dim", 10)),
        n_actions=8,
        local_cells=(2 * config['local_obs_radius'] + 1)**2,
        neighbor_k=config['neighbor_k'],
    )

    b_obs = build_observations_jax(
        pop=blue_pop,
        grid=grid,
        blue_map=jnp.zeros((config['grid_size'], config['grid_size'])),
        red_map=jnp.zeros((config['grid_size'], config['grid_size'])),
        config=config,
        step=0,
        key=k_pop,
        limit_red_sensing=False
    )
    
    print(f"[TEST] Actual built observation shape: {b_obs.shape}")
    assert b_obs.shape[-1] == obs_dim, f"FATAL: Built obs_dim {b_obs.shape[-1]} != network obs_dim {obs_dim}"

    b_params = model_blue.init(k_model, blue_pop.carries, b_obs, 1)
    
    new_b_carry, b_outs = model_blue.apply(
        b_params,
        blue_pop.carries,
        b_obs,
        1,
        jnp.ones(config['pop_size']),
        False, False,
        rngs={"dropout": rng}
    )

    (action_logits, signal_out, symbol_write, values, tom_logits, token_ids, alarm_out, loss_vq, z_e, culture_fast, culture_slow) = b_outs

    assert signal_out.shape == (config['pop_size'], config['signal_dim'])
    assert alarm_out.shape == (config['pop_size'], 2)

    # Hacky way to extract internal state representation for nb_combined
    # We will slice b_obs identically to the network to explicitly assert it
    N, K = config['pop_size'], config['neighbor_k']
    nb_sigs = b_obs[:, _layout.nb_sigs_start : _layout.nb_sigs_end].reshape(N, K, config['signal_dim'])
    nb_alarms = b_obs[:, _layout.nb_alarms_start : _layout.nb_alarms_end].reshape(N, K, 2)
    nb_combined = jnp.concatenate([nb_sigs, nb_alarms], axis=-1)
    
    print(f"[TEST] nb_combined explicit shape extracted: {nb_combined.shape}")
    assert nb_combined.shape == (N, K, 34), f"Expected {(N, K, 34)}, got {nb_combined.shape}"

    print("[TEST] All Phase 17.5 explicit shape assertions passed!")
    print("[TEST] Launching main lax.scan pipeline to confirm no hidden XLA bounds errors...")

    run_simulation(config)
    print("✓ PASS: Smoke test completed a full end-to-end scan.")

if __name__ == "__main__":
    smoke_test_phase_17_5()
