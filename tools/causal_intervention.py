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
from jax_sim.network_jax import AgentNetworkJax, init_agent_params, make_model_apply
from jax_sim.population_jax import init_population
from jax_sim.obs_layout import SIGNAL_SLOTS
from flax.core.frozen_dict import unfreeze, freeze

ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY", 5: "STRK", 6: "PUSH", 7: "GRD", 8: "BUILD", 9: "PU", 10: "CRF", 11: "USE"}

# Wire layout is NOT a uniform stride: slots are 12/8/12 wide at offsets 8/20/28
# (see jax_sim/obs_layout.py SIGNAL_SLOTS and network_jax.py codebook_{0,1,2}).
# The previous 8*slot_idx arithmetic here read/wrote the wrong dims.
SLOT_SLICES = [SIGNAL_SLOTS["slot_0"], SIGNAL_SLOTS["slot_1"], SIGNAL_SLOTS["slot_2"]]

def run_causal_intervention(checkpoint_dir: str, token_a: str, token_b: str, context: str, num_samples: int, receiver_dist_min: int = 10, alarm_test: bool = False):
    
    def parse_token(t_str):
        if t_str == "-1": return None
        parts = t_str.split(",")
        lst = [int(x) if x != "" else -1 for x in parts]
        while len(lst) < 3:
            lst.append(-1)
        return lst
        
    t_a_list = parse_token(token_a)
    t_b_list = parse_token(token_b)

    # 1. Load config and ensure step-by-step control
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = _normalize_config(config)
    config["ppo_rollout_steps"] = 1  # Crucial for intercepting state

    gs = int(config["grid_size"])
    neighbor_k = int(config["neighbor_k"])
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
        own_state_dim=int(config.get("own_state_dim", 10)),
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
        env_channels=int(config.get("env_channels", 10)),
        own_state_dim=int(config.get("own_state_dim", 10)),
    )
    r_model_apply = make_model_apply(model_red)

    # 2. Load checkpoint
    ckpt_mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer())
    _ckpt_latest = ckpt_mngr.latest_step()
    if _ckpt_latest is None:
        sys.exit(f"No valid checkpoint found in {checkpoint_dir}")
    raw_restored = ckpt_mngr.restore(_ckpt_latest)
    b_params = unfreeze(raw_restored)["b_params"]
    r_params = unfreeze(raw_restored)["r_params"]
    from jax_sim.network_jax import ensure_aux_head_params
    rng_init = jax.random.PRNGKey(0)
    b_params = ensure_aux_head_params(
        model=model, params=b_params, rng=rng_init, hidden_dim=hidden_d,
        obs_dim=obs_dim, n_layers=n_layers
    )
    b_params = freeze(b_params)
    r_params = freeze(r_params)
    if not alarm_test and t_a_list is not None and t_b_list is not None:
        codebooks = [
            b_params["codebook_0"]["embedding"],
            b_params["codebook_1"]["embedding"],
            b_params["codebook_2"]["embedding"]
        ]
    else:
        codebooks = None

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
        key=keys[1], n_agents=max_pop_red, memory_slots=20,  # Ensure full red population is initialized for distances
    )
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, red_hidden_d))

    sim_step = make_sim_step(config, model, model_apply, r_model_apply=r_model_apply)
    
    samples_collected = 0
    baseline_probs = []
    intervened_probs = []
    
    print(f"[Causal Intervention] Checkpoint {_ckpt_latest} loaded.", flush=True)
    if alarm_test:
        print(f"  ALARM TEST: Injecting Alarm=1 into silent neighbors")
    else:
        print(f"  Token A: {token_a} -> Token B: {token_b}")
    print(f"\n[Causal Intervention] Seeking {num_samples} isolated '{context}' events...", flush=True)
    
    step_idx = 0
    while samples_collected < num_samples:
        step_idx += 1
        if step_idx % 10 == 0:
            print(f"  ... searched {step_idx} steps, found {samples_collected} samples...", flush=True)
        step_key = jax.random.fold_in(keys[2], step_idx)
        
        pre_carry = (grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params)
        post_carry, rollout = sim_step(pre_carry, (step_key, step_idx))
        grid_out, b_pop_out, r_pop_out, b_carries_out, r_carries_out, _, _ = post_carry
        
        actions = np.asarray(rollout["blue"]["actions"])
        action_logits = np.asarray(rollout["blue"]["action_logits"])
        alive = np.asarray(b_pop.alive)
        positions = np.asarray(b_pop.positions)
        
        r_alive = np.asarray(r_pop.alive)
        r_positions = np.asarray(r_pop.positions)
        bg_mask = np.asarray(b_pop.is_big_green)
        
        # Audibility gate. A receiver only ingests the signals of its K nearest
        # alive neighbours (grid_jax.get_neighbour_signals). Intervening on an
        # emitter outside that set cannot reach the receiver at all, so it
        # contributes a structural zero and dilutes the ATE. Mirror the same
        # Chebyshev-on-torus top_k here and search only the audible set.
        _diff = np.abs(positions[:, None, :].astype(np.int64) - positions[None, :, :].astype(np.int64))
        _diff = np.minimum(_diff, gs - _diff)
        _pair_d = np.max(_diff, axis=-1).astype(np.float64)
        _pair_d = _pair_d + np.eye(max_pop) * (gs * 10)
        _pair_d = np.where(alive[None, :], _pair_d, gs * 100)
        nb_idx = np.argsort(_pair_d, axis=1)[:, :neighbor_k]

        for receiver_id in range(max_pop):
            if not alive[receiver_id]:
                continue
                
            ry, rx = positions[receiver_id]
            valid_event = False
            
            if context == "strike":
                if actions[receiver_id] != 5: continue
                bg_positions = positions[bg_mask & alive]
                for bgy, bgx in bg_positions:
                    if abs(bgy - ry) <= 1 and abs(bgx - rx) <= 1:
                        valid_event = True
                        break
            elif context == "flee":
                # Flee actions: 0, 1, 2, 3
                if actions[receiver_id] not in [0, 1, 2, 3]: continue
                # Ensure they are NOT near a red predator, so they only know about danger via signal
                is_near_red = False
                for red_i in range(max_pop_red):
                    if r_alive[red_i]:
                        rdy, rdx = r_positions[red_i]
                        # Compute true distance considering grid boundaries
                        dist_y = min(abs(rdy - ry), gs - abs(rdy - ry))
                        dist_x = min(abs(rdx - rx), gs - abs(rdx - rx))
                        if dist_y < receiver_dist_min and dist_x < receiver_dist_min:
                            is_near_red = True
                            break
                if is_near_red:
                    continue
                valid_event = True
            elif context == "force":
                # Force actions: 5 (STRK), 6 (PUSH), 8 (BUILD)
                if actions[receiver_id] not in [5, 6, 8]: continue
                valid_event = True
            elif context == "energy_high":
                valid_event = True

            if not valid_event:
                continue
                
            emitter_id = -1
            prev_signals = np.asarray(b_pop.signals)

            # Only agents inside the receiver's own neighbour window are audible.
            audible = [int(j) for j in nb_idx[receiver_id] if alive[j] and int(j) != receiver_id]

            if not alarm_test:
                for eid in audible:
                    sig = prev_signals[eid]
                    match = True
                    for slot_idx in range(3):
                        if t_a_list[slot_idx] != -1:
                            expected_emb = np.asarray(codebooks[slot_idx][t_a_list[slot_idx]])
                            actual_emb = sig[SLOT_SLICES[slot_idx]]
                            if np.linalg.norm(actual_emb - expected_emb) > 1e-4:
                                match = False
                                break
                    if match:
                        emitter_id = eid
                        break
            else:
                prev_alarms = np.asarray(b_pop.alarms)
                for eid in audible:
                    # Find an audible agent who was silent (alarm=0)
                    if prev_alarms[eid][1] < 0.5:
                        emitter_id = eid
                        break


            if emitter_id != -1:
                # We found a valid event. Calculate baseline probabilities.
                baseline_logits = action_logits[receiver_id]
                baseline_p = np.exp(baseline_logits) / np.sum(np.exp(baseline_logits))
                
                if context == "strike":
                    p_base = baseline_p[5]
                elif context == "flee":
                    p_base = sum([baseline_p[a] for a in [0, 1, 2, 3]])
                elif context == "force":
                    p_base = sum([baseline_p[a] for a in [5, 6, 8]])
                elif context == "energy_high":
                    ey, ex = positions[emitter_id]
                    dy = ey - ry
                    dx = ex - rx
                    if dy > gs // 2: dy -= gs
                    elif dy < -gs // 2: dy += gs
                    if dx > gs // 2: dx -= gs
                    elif dx < -gs // 2: dx += gs
                    approach_actions = []
                    if dy < 0: approach_actions.append(0) # N
                    elif dy > 0: approach_actions.append(1) # S
                    if dx > 0: approach_actions.append(2) # E
                    elif dx < 0: approach_actions.append(3) # W
                    if not approach_actions: continue
                    p_base = sum([baseline_p[a] for a in approach_actions])
                
                # --- INTERVENTION ---
                if not alarm_test:
                    new_sig = b_pop.signals[emitter_id]
                    for slot_idx in range(3):
                        if t_b_list[slot_idx] != -1:
                            new_sig = new_sig.at[SLOT_SLICES[slot_idx]].set(codebooks[slot_idx][t_b_list[slot_idx]])
                    b_pop_intervened = b_pop.replace(
                        signals=b_pop.signals.at[emitter_id].set(new_sig)
                    )
                else:
                    new_alarm = jnp.array([0.0, 1.0], dtype=jnp.float32)
                    b_pop_intervened = b_pop.replace(
                        alarms=b_pop.alarms.at[emitter_id].set(new_alarm)
                    )
                intervened_carry = (grid, b_pop_intervened, r_pop, b_carries, r_carries, b_params, r_params)
                
                _, rollout_int = sim_step(intervened_carry, (step_key, step_idx))
                int_logits = np.asarray(rollout_int["blue"]["action_logits"])[receiver_id]
                int_p = np.exp(int_logits) / np.sum(np.exp(int_logits))
                
                if context == "strike":
                    p_int = int_p[5]
                elif context == "flee":
                    p_int = sum([int_p[a] for a in [0, 1, 2, 3]])
                elif context == "force":
                    p_int = sum([int_p[a] for a in [5, 6, 8]])
                elif context == "energy_high":
                    p_int = sum([int_p[a] for a in approach_actions])
                
                baseline_probs.append(p_base)
                intervened_probs.append(p_int)
                samples_collected += 1
                
                if not alarm_test:
                    print(f"Sample {samples_collected:03d} | Receiver {receiver_id} Emitter {emitter_id} | "
                          f"P(Action|TokenA)={p_base:.4f} -> P(Action|TokenB)={p_int:.4f} (Delta: {p_int - p_base:.4f})", flush=True)
                else:
                    print(f"Sample {samples_collected:03d} | Receiver {receiver_id} Emitter {emitter_id} | "
                          f"P(Action|Silent)={p_base:.4f} -> P(Action|Alarm)={p_int:.4f} (Delta: {p_int - p_base:.4f})", flush=True)
                
                if samples_collected >= num_samples:
                    break

        grid, b_pop, r_pop, b_carries, r_carries = grid_out, b_pop_out, r_pop_out, b_carries_out, r_carries_out

    baseline_probs = np.array(baseline_probs)
    intervened_probs = np.array(intervened_probs)
    
    # Delta logic
    if alarm_test:
        mean_delta = np.mean(intervened_probs - baseline_probs) 
        t_stat, p_val = stats.ttest_rel(intervened_probs, baseline_probs)
    else:
        mean_delta = np.mean(baseline_probs - intervened_probs) 
        t_stat, p_val = stats.ttest_rel(baseline_probs, intervened_probs)
    
    print("\n" + "="*50)
    print(f" CAUSAL INTERVENTION RESULTS: Context '{context}'")
    print("="*50)
    print(f"Total Samples (N): {len(baseline_probs)} (requested {num_samples})")
    if alarm_test:
        print(f"Mean P(Action|Silent): {np.mean(baseline_probs):.4f}")
        print(f"Mean P(Action|Alarm):  {np.mean(intervened_probs):.4f}")
        print(f"Mean Delta (ATE):      {mean_delta:.4f}")
        print(f"Paired t-test:         t={t_stat:.2f}, p={p_val:.2e}")
        if p_val < 0.05 and mean_delta > 0.05:
            print("\n[CONCLUSION] SIGNIFICANT CAUSAL DIVERGENCE DETECTED.")
            print(f"Injecting the alarm conclusively drives '{context}' behavior.")
        else:
            print("\n[CONCLUSION] NULL HYPOTHESIS.")
            print("The alarm intervention did not produce a statistically significant increase.")
    else:
        print(f"Mean P(Action|TokenA): {np.mean(baseline_probs):.4f}")
        print(f"Mean P(Action|TokenB): {np.mean(intervened_probs):.4f}")
        print(f"Mean Delta (ATE):      {mean_delta:.4f}")
        print(f"Paired t-test:         t={t_stat:.2f}, p={p_val:.2e}")
        if p_val < 0.05 and mean_delta > 0.05:
            print("\n[CONCLUSION] SIGNIFICANT CAUSAL DIVERGENCE DETECTED.")
            print(f"Token {token_a} conclusively drives '{context}' behavior compared to Token {token_b}.")
        else:
            print("\n[CONCLUSION] NULL HYPOTHESIS.")
            print("The VQ token swap did not produce a statistically significant suppression.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--token-a", type=str, required=False, default="-1", help="Comma-separated Slot IDs for Token A (e.g. 57,12,34)")
    parser.add_argument("--token-b", type=str, required=False, default="-1", help="Comma-separated Slot IDs for Token B (e.g. 46,0,0)")
    parser.add_argument("--context", type=str, choices=["strike", "flee", "force", "energy_high"], required=True, help="Behavior context to test")
    parser.add_argument("--samples", type=int, default=100, help="Number of independent events to sample")
    parser.add_argument("--receiver-dist-min", type=int, default=10, help="Minimum distance between receiver and target entity")
    parser.add_argument("--alarm-test", action="store_true", help="Test the 1-bit alarm head instead of VQ tokens")
    
    # Optional flags passed by Cam that don't affect live sim rewind but are kept for CLI compatibility
    parser.add_argument("--corpus", type=str, default="", help="Ignored. Live rewind used.")
    parser.add_argument("--min-step", type=int, default=0, help="Ignored. Live rewind used.")
    parser.add_argument("--n-events", type=int, default=0, help="Alias for --samples")
    
    args = parser.parse_args()
    
    samples = args.n_events if args.n_events > 0 else args.samples
    
    try:
        import scipy
    except ImportError:
        sys.exit("scipy is required for statistical tests. pip install scipy")
        
    run_causal_intervention(args.checkpoint, args.token_a, args.token_b, args.context, samples, args.receiver_dist_min, args.alarm_test)
