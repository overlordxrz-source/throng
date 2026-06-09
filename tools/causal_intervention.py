#!/usr/bin/env python3
"""
Causal Intervention script.
Executes the Frozen Counterfactual Causal Test by isolating a Big Green Strike
and swapping the transmitted VQ token.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os
import yaml
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint._src.metadata import sharding as orbax_sharding

# Monkeypatch orbax sharding to ignore missing cuda devices and fallback to cpu
def mock_to_jax_sharding(self):
    return jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('x',)), jax.sharding.PartitionSpec())
orbax_sharding.NamedShardingMetadata.to_jax_sharding = mock_to_jax_sharding
orbax_sharding.SingleDeviceShardingMetadata.to_jax_sharding = mock_to_jax_sharding

import jax_sim.observations_jax as _obs_mod
original_build_obs = _obs_mod.build_observations_jax
def patched_build_obs(*args, **kwargs):
    obs = original_build_obs(*args, **kwargs)
    N = obs.shape[0]
    # Check if the generated obs size matches what we expect from 10 channels (25*10=250 vs 25*9=225)
    # The default builder on this branch creates 10 channels. If our config wants 9 channels, we patch it down.
    import yaml
    with open(ROOT / "config_phase7.yaml") as f:
        cfg = yaml.safe_load(f)
    env_channels = int(cfg.get("env_channels", 9))
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
from jax_sim.network_jax import AgentNetworkJax, init_agent_params, make_model_apply
from jax_sim.population_jax import init_population
from flax.core.frozen_dict import unfreeze, freeze

ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY", 5: "STRK", 6: "PUSH", 7: "GRD", 8: "BUILD"}

def run_causal_intervention(checkpoint_dir: str, strike_token: int, flee_token: int, num_samples: int):
    # 1. Load config and ensure step-by-step control
    with open(ROOT / "config_phase7.yaml") as f:
        config = yaml.safe_load(f)
    config = _normalize_config(config)
    config["ppo_rollout_steps"] = 1  # Crucial for intercepting state

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
        env_channels=int(config.get("env_channels", 9)),
    )
    model_apply = make_model_apply(model)

    _p14t = config.get("phase14_transcendental", {})
    red_hidden_d = int(_p14t.get("hidden_dim", 128))
    _red_vocab = int(_p14t.get("red_vocab_size", 64))
    _red_cross = bool(_p14t.get("cross_attn_enabled", False))
    _p9 = config.get("phase9_canvas", {})
    _cross_heads = int(_p9.get("cross_attn_num_heads", 4))
    from jax_sim.network_jax import PredatorNetworkJax
    model_red = PredatorNetworkJax(
        hidden_dim=red_hidden_d,
        neighbor_k=int(config["neighbor_k"]),
        local_obs_radius=int(config["local_obs_radius"]),
        n_heads=int(config["n_heads"]),
        n_layers=n_layers,
        signal_dim=sig_d,
        symbol_dim=int(config["symbol_dim"]),
        vocab_size=_red_vocab,
        vq_beta=float(config.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(config.get("vq_dead_code_reset", True)),
        cross_attn_enabled=_red_cross,
        cross_attn_num_heads=_cross_heads,
        n_actions=int(config.get("n_actions", 8)),
        env_channels=int(config.get("env_channels", 9)),
    )
    r_model_apply = make_model_apply(model_red)

    # 2. Load checkpoint
    ckpt_mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer())
    _ckpt_latest = ckpt_mngr.latest_step()
    if _ckpt_latest is None:
        sys.exit(f"No valid checkpoint found in {checkpoint_dir}")
    raw_restored = ckpt_mngr.restore(_ckpt_latest)
    b_params = freeze(unfreeze(raw_restored)["b_params"])
    r_params = freeze(unfreeze(raw_restored)["r_params"])
    
    cb = b_params["codebook"]["embedding"]
    flee_embedding = cb[flee_token]
    strike_embedding = cb[strike_token]
    print(f"[Causal Intervention] Checkpoint {_ckpt_latest} loaded.")
    print(f"  Flee token {flee_token} embedding shape: {flee_embedding.shape}")

    # Initialize environment
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 10)
    grid = GridState(gs, symbol_dim=int(config["symbol_dim"]))
    grid = grid.replace(walls=jnp.zeros((gs, gs), dtype=jnp.bool_))

    b_pop = init_population(
        max_pop, hidden_d, sig_d, gs, team_id=0,
        key=keys[0], n_agents=max_pop, memory_slots=20,
    )
    r_pop = init_population(
        max_pop_red, red_hidden_d, sig_d, gs, team_id=1,
        key=keys[1], n_agents=4, memory_slots=20,
    )
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, red_hidden_d))

    sim_step = make_sim_step(config, model, model_apply, r_model_apply=r_model_apply)
    
    samples_collected = 0
    baseline_strike_probs = []
    intervened_strike_probs = []
    
    print(f"[Causal Intervention] Checkpoint {_ckpt_latest} loaded.", flush=True)
    print(f"  Flee token {flee_token} embedding shape: {flee_embedding.shape}", flush=True)
    print(f"\n[Causal Intervention] Seeking {num_samples} isolated BG Strike Coordination events...", flush=True)
    
    # Run forward rollout loop to find events
    step_idx = 0
    while samples_collected < num_samples:
        step_idx += 1
        if step_idx % 10 == 0:
            print(f"  ... searched {step_idx} steps, found {samples_collected} samples...", flush=True)
        step_key = jax.random.fold_in(keys[2], step_idx)
        
        # Save pre-step state to allow rewind
        pre_carry = (grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params)
        
        # Forward pass
        post_carry, rollout = sim_step(pre_carry, (step_key, step_idx))
        grid_out, b_pop_out, r_pop_out, b_carries_out, r_carries_out, _, _ = post_carry
        
        # Unpack arrays for analysis
        actions = np.asarray(rollout["blue"]["actions"])
        action_logits = np.asarray(rollout["blue"]["action_logits"])
        vq_tokens = np.asarray(rollout["blue"]["token_ids"])
        alive = np.asarray(b_pop.alive)
        positions = np.asarray(b_pop.positions)
        bg_mask = np.asarray(b_pop.is_big_green)
        
        # Build grid map for fast lookup
        pos_map = {}
        for i in range(max_pop):
            if alive[i]:
                py, px = positions[i]
                pos_map.setdefault((py, px), []).append(i)

        # We need to find an event where:
        # 1. Receiver is adjacent to Big Green and performs Action 5 (STRK).
        # 2. Receiver has a neighbor (Emitter) that transmitted `strike_token`.
        for receiver_id in range(max_pop):
            if not alive[receiver_id] or actions[receiver_id] != 5:
                continue
                
            # Check adjacency to Big Green
            ry, rx = positions[receiver_id]
            bg_positions = positions[bg_mask & alive]
            adj_bg = False
            for bgy, bgx in bg_positions:
                if abs(bgy - ry) <= 1 and abs(bgx - rx) <= 1:
                    adj_bg = True
                    break
            
            if not adj_bg:
                continue
                
            emitter_id = -1
            prev_signals = np.asarray(b_pop.signals)
            
            # Using pos_map for fast lookup of neighbors within radius 5
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    ny, nx = (ry + dy) % gs, (rx + dx) % gs
                    agents_here = pos_map.get((ny, nx), [])
                    for eid in agents_here:
                        if eid != receiver_id:
                            dist = np.linalg.norm(prev_signals[eid] - np.asarray(strike_embedding))
                            if dist < 1e-4:
                                emitter_id = eid
                                break
                    if emitter_id != -1:
                        break
                if emitter_id != -1:
                    break
                        
            if emitter_id != -1:
                # We found an event! 
                # Receiver is striking Big Green, and Emitter transmitted strike_token.
                # Now we perform the counterfactual rewind.
                
                # Compute baseline probability
                baseline_logits = action_logits[receiver_id]
                baseline_probs = np.exp(baseline_logits) / np.sum(np.exp(baseline_logits))
                p_strike_baseline = baseline_probs[5]
                
                # --- INTERVENTION ---
                # Rewind to pre_carry, but swap emitter's signal
                b_pop_intervened = b_pop.replace(
                    signals=b_pop.signals.at[emitter_id].set(flee_embedding)
                )
                
                intervened_carry = (grid, b_pop_intervened, r_pop, b_carries, r_carries, b_params, r_params)
                
                # Re-run simulation step with SAME key
                _, rollout_int = sim_step(intervened_carry, (step_key, step_idx))
                action_logits_int = np.asarray(rollout_int["blue"]["action_logits"])
                
                # Compute intervened probability
                int_logits = action_logits_int[receiver_id]
                int_probs = np.exp(int_logits) / np.sum(np.exp(int_logits))
                p_strike_int = int_probs[5]
                
                baseline_strike_probs.append(p_strike_baseline)
                intervened_strike_probs.append(p_strike_int)
                samples_collected += 1
                
                print(f"Sample {samples_collected:03d} | Receiver {receiver_id} Emitter {emitter_id} | "
                      f"P(STRK|strike)={p_strike_baseline:.4f} -> P(STRK|flee)={p_strike_int:.4f} (Delta: {p_strike_int - p_strike_baseline:.4f})")
                
                if samples_collected >= num_samples:
                    break

        # Advance state naturally
        grid, b_pop, r_pop, b_carries, r_carries = grid_out, b_pop_out, r_pop_out, b_carries_out, r_carries_out

    # Statistical Evaluation
    baseline_strike_probs = np.array(baseline_strike_probs)
    intervened_strike_probs = np.array(intervened_strike_probs)
    
    mean_delta = np.mean(intervened_strike_probs - baseline_strike_probs)
    t_stat, p_val = stats.ttest_rel(intervened_strike_probs, baseline_strike_probs)
    
    print("\n" + "="*50)
    print(" CAUSAL INTERVENTION RESULTS")
    print("="*50)
    print(f"Total Samples (N): {num_samples}")
    print(f"Mean P(STRK|strike): {np.mean(baseline_strike_probs):.4f}")
    print(f"Mean P(STRK|flee):   {np.mean(intervened_strike_probs):.4f}")
    print(f"Mean Delta (ATE):    {mean_delta:.4f}")
    print(f"Paired t-test:       t={t_stat:.2f}, p={p_val:.2e}")
    if p_val < 0.05 and mean_delta < 0:
        print("\n[CONCLUSION] SIGNIFICANT CAUSAL DIVERGENCE DETECTED.")
        print("The injected 'flee' VQ token conclusively suppressed the Strike behavior.")
    else:
        print("\n[CONCLUSION] NULL HYPOTHESIS.")
        print("The VQ token swap did not produce a statistically significant suppression.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (e.g. runs/.../checkpoints/980000)")
    parser.add_argument("--strike-token", type=int, required=True, help="Token ID representing Strike/Noun context")
    parser.add_argument("--flee-token", type=int, required=True, help="Token ID representing Flee context")
    parser.add_argument("--samples", type=int, default=100, help="Number of independent events to sample")
    
    args = parser.parse_args()
    
    # Ensure scipy is available (used for ttest)
    try:
        import scipy
    except ImportError:
        sys.exit("scipy is required for statistical tests. pip install scipy")
        
    run_causal_intervention(args.checkpoint, args.strike_token, args.flee_token, args.samples)
