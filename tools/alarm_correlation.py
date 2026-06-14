#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint._src.metadata import sharding as orbax_sharding

def mock_to_jax_sharding(self):
    return jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('x',)), jax.sharding.PartitionSpec())
orbax_sharding.NamedShardingMetadata.to_jax_sharding = mock_to_jax_sharding
orbax_sharding.SingleDeviceShardingMetadata.to_jax_sharding = mock_to_jax_sharding

import jax_sim.observations_jax as _obs_mod
original_build_obs = _obs_mod.build_observations_jax
def patched_build_obs(*args, **kwargs):
    obs = original_build_obs(*args, **kwargs)
    N = obs.shape[0]
    import yaml
    with open(ROOT / "config_phase7.yaml") as f:
        cfg = yaml.safe_load(f)
    env_channels = int(cfg.get("env_channels", 10))
    if env_channels == 9 and obs.shape[1] >= (598 + 250):
        part1 = obs[:, :598]
        loc_env = obs[:, 598:598+250].reshape((N, 25, 10))
        loc_env_9 = loc_env[:, :, :9].reshape((N, 225))
        part3 = obs[:, 598+250:]
        return jnp.concatenate([part1, loc_env_9, part3], axis=1)
    return obs
_obs_mod.build_observations_jax = patched_build_obs

from agents.network_torch import compute_obs_dim_torch, compute_fwd_env_dim
from jax_sim.grid_jax import GridState
from jax_sim.main_jax import _normalize_config, make_sim_step
from jax_sim.network_jax import AgentNetworkJax, PredatorNetworkJax, make_model_apply
from jax_sim.population_jax import init_population
from flax.core.frozen_dict import unfreeze, freeze

def get_red_dists(b_pos, b_alive, r_pos, r_alive, gs):
    dy = np.abs(b_pos[:, None, 0] - r_pos[None, :, 0])
    dx = np.abs(b_pos[:, None, 1] - r_pos[None, :, 1])
    dy = np.minimum(dy, gs - dy)
    dx = np.minimum(dx, gs - dx)
    dist = np.maximum(dy, dx)
    # Mask out dead reds
    dist = np.where(r_alive[None, :], dist, 9999)
    min_dist = np.min(dist, axis=1)
    return min_dist

def run_npmi_scan(checkpoint_dir: str, steps: int = 500, scout_range: int = 8):
    with open(ROOT / "config_phase7.yaml") as f:
        config = yaml.safe_load(f)
    config = _normalize_config(config)
    config["ppo_rollout_steps"] = 1

    gs = int(config["grid_size"])
    max_pop = int(config["max_pop"])
    max_pop_red = int(config["max_pop_red"])
    hidden_d = int(config["hidden_dim"])
    n_layers = int(config["n_layers"])
    sig_d = int(config["signal_dim"])
    obs_dim = compute_obs_dim_torch(config)
    fwd_env_dim = compute_fwd_env_dim(config)

    model = AgentNetworkJax(
        hidden_dim=hidden_d,
        n_heads=int(config["n_heads"]),
        n_layers=n_layers,
        obs_dim=0,
        signal_dim=sig_d,
        symbol_dim=int(config["symbol_dim"]),
        vocab_size=int(config["vocab_size"]),
        memory_slots=0,
        fwd_env_dim=fwd_env_dim,
        cross_attn_enabled=config.get("phase9_canvas", {}).get("cross_attn_enabled", False),
        neighbor_k=int(config["neighbor_k"]),
        n_actions=int(config.get("n_actions", 9)),
        env_channels=int(config.get("env_channels", 10)),
    )
    model_apply = make_model_apply(model)

    _p14t = config.get("phase14_transcendental", {})
    model_red = PredatorNetworkJax(
        hidden_dim=int(_p14t.get("hidden_dim", 128)),
        neighbor_k=int(config["neighbor_k"]),
        local_obs_radius=int(config["local_obs_radius"]),
        n_heads=int(config["n_heads"]),
        n_layers=n_layers,
        signal_dim=sig_d,
        symbol_dim=int(config["symbol_dim"]),
        vocab_size=int(_p14t.get("red_vocab_size", 64)),
        vq_beta=float(config.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(config.get("vq_dead_code_reset", True)),
        cross_attn_enabled=bool(_p14t.get("cross_attn_enabled", False)),
        cross_attn_num_heads=int(config.get("phase9_canvas", {}).get("cross_attn_num_heads", 4)),
        n_actions=int(config.get("n_actions", 9)),
        env_channels=int(config.get("env_channels", 10)),
    )
    r_model_apply = make_model_apply(model_red)

    ckpt_mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer())
    _ckpt_latest = ckpt_mngr.latest_step()
    if _ckpt_latest is None:
        sys.exit(f"No valid checkpoint found in {checkpoint_dir}")
    raw_restored = ckpt_mngr.restore(_ckpt_latest)
    b_params = freeze(unfreeze(raw_restored)["b_params"])
    r_params = freeze(unfreeze(raw_restored)["r_params"])

    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 10)
    grid = GridState(gs, symbol_dim=int(config["symbol_dim"]))
    grid = grid.replace(walls=jnp.zeros((gs, gs), dtype=jnp.bool_))

    b_pop = init_population(max_pop, hidden_d, sig_d, gs, 0, keys[0], max_pop, 20)
    r_pop = init_population(max_pop_red, int(_p14t.get("hidden_dim", 128)), sig_d, gs, 1, keys[1], max_pop_red, 20)
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, int(_p14t.get("hidden_dim", 128))))

    sim_step = make_sim_step(config, model, model_apply, r_model_apply=r_model_apply)

    all_alarms = []
    all_near = []

    print(f"Running NPMI Scan on checkpoint {_ckpt_latest} for {steps} steps...")
    carry = (grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params)
    
    for i in range(steps):
        step_key = jax.random.fold_in(keys[2], i)
        carry, rollout = sim_step(carry, (step_key, i))
        
        # We want to extract the alarms chosen AT this step.
        # However, pop.alarms is updated at the END of the step.
        b_pop_out = carry[1]
        r_pop_out = carry[2]
        
        alive = np.asarray(b_pop_out.alive)
        alarms = np.asarray(b_pop_out.alarms)
        alarm_fired = alarms[:, 1] > 0.5
        
        b_pos = np.asarray(b_pop_out.positions)
        r_pos = np.asarray(r_pop_out.positions)
        r_alive = np.asarray(r_pop_out.alive)
        
        red_dists = get_red_dists(b_pos, alive, r_pos, r_alive, gs)
        is_near = red_dists <= scout_range
        
        all_alarms.extend(alarm_fired[alive])
        all_near.extend(is_near[alive])
        
        if (i+1) % 50 == 0:
            print(f"  Step {i+1}/{steps}...")

    all_alarms = np.array(all_alarms, dtype=bool)
    all_near = np.array(all_near, dtype=bool)

    # NPMI Calculation
    p_alarm = np.mean(all_alarms)
    p_near = np.mean(all_near)
    p_both = np.mean(all_alarms & all_near)

    print("\n" + "="*50)
    print(" ALARM NPMI SCAN RESULTS")
    print("="*50)
    print(f"Total agent-steps: {len(all_alarms)}")
    print(f"P(Alarm=1):        {p_alarm:.4f}")
    print(f"P(Red Near):       {p_near:.4f}")
    print(f"P(Alarm=1 & Near): {p_both:.4f}")
    
    if p_both == 0 or p_alarm == 0 or p_near == 0:
        print("NPMI: Undefined (Zero probabilities)")
    else:
        pmi = np.log(p_both / (p_alarm * p_near))
        npmi = pmi / -np.log(p_both)
        print(f"PMI:               {pmi:.4f}")
        print(f"NPMI:              {npmi:.4f}")
        
        if npmi > 0.1:
            print("\n[CONCLUSION] POSITIVE CORRELATION. Alarm firing is significantly correlated with predator proximity.")
        else:
            print("\n[CONCLUSION] NO MEANINGFUL CORRELATION. The alarm is firing randomly with respect to predators.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scout-range", type=int, default=8)
    args = parser.parse_args()
    run_npmi_scan(args.checkpoint, args.steps, args.scout_range)
