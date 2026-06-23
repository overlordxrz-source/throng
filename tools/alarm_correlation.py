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
    with open(ROOT / "config.yaml") as f:
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
    with open(ROOT / "config.yaml") as f:
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
        n_actions=int(config.get("n_actions", 8)),
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
        n_actions=int(config.get("n_actions", 8)),
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
    all_near_8 = []
    all_near_3 = []
    all_near_2 = []
    all_low_energy = []
    all_crowded = []
    all_res_at_pos = []
    all_res_adj = []
    all_puz_adj = []
    all_c_res_at_pos = []
    all_c_res_adj = []

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
        
        b_energy = np.asarray(b_pop_out.energy)
        
        # Calculate neighbor counts
        dy = np.abs(b_pos[:, None, 0] - b_pos[None, :, 0])
        dx = np.abs(b_pos[:, None, 1] - b_pos[None, :, 1])
        dy = np.minimum(dy, gs - dy)
        dx = np.minimum(dx, gs - dx)
        b_dist = np.maximum(dy, dx)
        b_dist = np.where(alive[None, :], b_dist, 9999)
        neighbors = np.sum(b_dist <= 1, axis=1) - 1
        
        red_dists = get_red_dists(b_pos, alive, r_pos, r_alive, gs)
        
        grid_state = carry[0]
        res_grid = np.asarray(grid_state.resources)
        res_at_pos = res_grid[b_pos[:, 0], b_pos[:, 1]] > 0.01
        
        res_y, res_x = np.nonzero(res_grid > 0.01)
        if len(res_y) > 0:
            res_coords = np.stack([res_y, res_x], axis=1)
            dy_res = np.abs(b_pos[:, None, 0] - res_coords[None, :, 0])
            dx_res = np.abs(b_pos[:, None, 1] - res_coords[None, :, 1])
            dy_res = np.minimum(dy_res, gs - dy_res)
            dx_res = np.minimum(dx_res, gs - dx_res)
            dist_res = np.maximum(dy_res, dx_res)
            min_dist_res = np.min(dist_res, axis=1)
        else:
            min_dist_res = np.full(b_pos.shape[0], 9999)
            
        puz_grid = np.asarray(grid_state.puzzle_grid)
        puz_y, puz_x = np.nonzero(puz_grid > 0.01)
        if len(puz_y) > 0:
            puz_coords = np.stack([puz_y, puz_x], axis=1)
            dy_puz = np.abs(b_pos[:, None, 0] - puz_coords[None, :, 0])
            dx_puz = np.abs(b_pos[:, None, 1] - puz_coords[None, :, 1])
            dy_puz = np.minimum(dy_puz, gs - dy_puz)
            dx_puz = np.minimum(dx_puz, gs - dx_puz)
            dist_puz = np.maximum(dy_puz, dx_puz)
            min_dist_puz = np.min(dist_puz, axis=1)
        else:
            min_dist_puz = np.full(b_pos.shape[0], 9999)
            
        c_res_grid = np.asarray(grid_state.contested_res)
        if i == 0:
            print(f"Max contested res on grid: {np.max(c_res_grid)}")
        c_res_at_pos = c_res_grid[b_pos[:, 0], b_pos[:, 1]] > 0.01
        
        c_res_y, c_res_x = np.nonzero(c_res_grid > 0.01)
        if len(c_res_y) > 0:
            c_res_coords = np.stack([c_res_y, c_res_x], axis=1)
            dy_c = np.abs(b_pos[:, None, 0] - c_res_coords[None, :, 0])
            dx_c = np.abs(b_pos[:, None, 1] - c_res_coords[None, :, 1])
            dy_c = np.minimum(dy_c, gs - dy_c)
            dx_c = np.minimum(dx_c, gs - dx_c)
            dist_c = np.maximum(dy_c, dx_c)
            min_dist_c = np.min(dist_c, axis=1)
        else:
            min_dist_c = np.full(b_pos.shape[0], 9999)
        
        all_alarms.extend(alarm_fired[alive])
        all_near_8.extend((red_dists <= 8)[alive])
        all_near_3.extend((red_dists <= 3)[alive])
        all_near_2.extend((red_dists <= 2)[alive])
        all_low_energy.extend((b_energy <= 0.3)[alive])
        all_crowded.extend((neighbors >= 2)[alive])
        all_res_at_pos.extend(res_at_pos[alive])
        all_res_adj.extend((min_dist_res <= 1)[alive])
        all_puz_adj.extend((min_dist_puz <= 1)[alive])
        all_c_res_at_pos.extend(c_res_at_pos[alive])
        all_c_res_adj.extend((min_dist_c <= 1)[alive])
        
        if (i+1) % 50 == 0:
            print(f"  Step {i+1}/{steps}...")

    all_alarms = np.array(all_alarms, dtype=bool)
    
    def calc_npmi(cond_array, name):
        p_alarm = np.mean(all_alarms)
        p_cond = np.mean(cond_array)
        p_both = np.mean(all_alarms & cond_array)
        
        if p_both == 0 or p_alarm == 0 or p_cond == 0:
            return p_cond, p_both, 0.0, 0.0
            
        pmi = np.log(p_both / (p_alarm * p_cond))
        npmi = pmi / -np.log(p_both)
        return p_cond, p_both, pmi, npmi

    print("\n" + "="*70)
    print(" ALARM NPMI MULTI-HYPOTHESIS SCAN RESULTS")
    print("="*70)
    print(f"Total agent-steps: {len(all_alarms)}")
    print(f"P(Alarm=1):        {np.mean(all_alarms):.4f}")
    print("-" * 70)
    print(f"{'Target Variable':<25} | {'P(Cond)':<8} | {'P(Both)':<8} | {'PMI':<8} | {'NPMI':<8}")
    print("-" * 70)
    
    all_near_8 = np.array(all_near_8, dtype=bool)
    all_near_3 = np.array(all_near_3, dtype=bool)
    all_near_2 = np.array(all_near_2, dtype=bool)
    all_low_energy = np.array(all_low_energy, dtype=bool)
    all_crowded = np.array(all_crowded, dtype=bool)
    all_res_at_pos = np.array(all_res_at_pos, dtype=bool)
    all_res_adj = np.array(all_res_adj, dtype=bool)
    all_puz_adj = np.array(all_puz_adj, dtype=bool)
    all_c_res_at_pos = np.array(all_c_res_at_pos, dtype=bool)
    all_c_res_adj = np.array(all_c_res_adj, dtype=bool)
    
    target_variables = {
        "Red Dist <= 8": all_near_8,
        "Red Dist <= 3": all_near_3,
        "Red Dist <= 2": all_near_2,
        "Energy <= 0.3": all_low_energy,
        "Crowding (Neighbors >= 2)": all_crowded,
        "Resource At Position": all_res_at_pos,
        "Resource Adjacent (<=1)": all_res_adj,
        "Puzzle Adjacent (<=1)": all_puz_adj,
        "Contested Res At Pos": all_c_res_at_pos,
        "Contested Res Adj (<=1)": all_c_res_adj,
    }

    for name, cond_array in target_variables.items():
        p_cond, p_both, pmi, npmi = calc_npmi(cond_array, name)
        print(f"{name:<25} | {p_cond:<8.4f} | {p_both:<8.4f} | {pmi:<8.4f} | {npmi:<8.4f}")
    
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scout-range", type=int, default=8)
    args = parser.parse_args()
    run_npmi_scan(args.checkpoint, args.steps, args.scout_range)
