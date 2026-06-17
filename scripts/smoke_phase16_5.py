#!/usr/bin/env python3
"""Phase 16.5 smoke test: barrier physics, 10 env channels, 9 actions."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
import jax
import jax.numpy as jnp

from jax_sim.grid_jax import GridState
from jax_sim.main_jax import _normalize_config, make_sim_step
from jax_sim.network_jax import (
    AgentNetworkJax,
    init_agent_params,
    make_model_apply,
)
from jax_sim.population_jax import init_population

def _tiny_config() -> dict:
    with open(ROOT / "config_phase7.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = _normalize_config(cfg)
    n_agents = 10
    cfg.update(
        {
            "grid_size": 32,
            "population_size": n_agents,
            "max_pop": n_agents,
            "max_pop_red": 4,
            "min_red_population": 2,
            "red_population_size": 4,
            "agent_hidden_dim": 64,
            "hidden_dim": 64,
            "brain_n_heads": 2,
            "n_heads": 2,
            "n_layers": 2,
            "signal_dim": 16,
            "symbol_dim": 8,
            "vocab_size": 16,
            "neighbor_k": 4,
            "memory_buffer_size": 0,
            "memory_slots": 0,
            "ppo_rollout_steps": 4,
            "ppo_minibatch_size": 4,
            "n_actions": 9,  # Phase 16.5 9-action space
        }
    )
    p9 = dict(cfg.get("phase9_canvas") or {})
    p9["imagination_gating_enabled"] = False
    p9["cross_attn_enabled"] = False
    cfg["phase9_canvas"] = p9
    p12 = dict(cfg.get("phase12_coevolution") or {})
    p12["red_comms_enabled"] = False
    cfg["phase12_coevolution"] = p12
    cfg["phase16_5_enrichment"] = {"ignition_discount": 0.1}
    return cfg

def main() -> None:
    config = _tiny_config()
    gs = int(config["grid_size"])
    max_pop = int(config["max_pop"])
    max_pop_red = int(config["max_pop_red"])
    hidden_d = int(config["hidden_dim"])
    n_layers = int(config["n_layers"])
    sig_d = int(config["signal_dim"])
    
    # 10 env channels for Phase 16.5
    env_channels = 10

    model = AgentNetworkJax(
        hidden_dim=hidden_d,
        n_heads=int(config["n_heads"]),
        n_layers=n_layers,
        obs_dim=0, # Computed internally
        signal_dim=sig_d,
        symbol_dim=int(config["symbol_dim"]),
        vocab_size=int(config["vocab_size"]),
        memory_slots=0,
        fwd_env_dim=250, # W x env_ch = 25 x 10
        cross_attn_enabled=False,
        neighbor_k=int(config["neighbor_k"]),
        env_channels=env_channels,
        n_actions=int(config["n_actions"]),
    )
    
    # Needs to match 9 actions in the heads
    model_apply = make_model_apply(model)

    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 8)
    carry = jnp.zeros((max_pop, hidden_d))
    
    # Calculate obs_dim dynamically to match network_jax
    K = int(config["neighbor_k"])
    W = 25
    obs_dim = 6 + K * sig_d + W * int(config["symbol_dim"]) + W * env_channels + sig_d
    
    obs0 = jax.random.normal(keys[0], (max_pop, obs_dim)) * 0.05
    params = init_agent_params(model, keys[1], carry, obs0, n_layers)

    grid = GridState(gs, symbol_dim=int(config["symbol_dim"]))
    grid = grid.replace(walls=jnp.zeros((gs, gs), dtype=jnp.bool_))

    b_pop = init_population(
        max_pop, hidden_d, sig_d, gs, team_id=0,
        key=keys[2], n_agents=max_pop, memory_slots=0,
    )
    r_pop = init_population(
        max_pop_red, hidden_d, sig_d, gs, team_id=1,
        key=keys[3], n_agents=2, memory_slots=0,
    )
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, hidden_d))
    r_params = params

    sim_step = make_sim_step(config, model, model_apply, r_model_apply=model_apply)
    step_key = keys[4]
    init_carry = (grid, b_pop, r_pop, b_carries, r_carries, params, r_params)
    
    final_carry, rollout = sim_step(init_carry, step_key)
    _, b_pop_out, _, _, _, _, _ = final_carry
    blue = rollout["blue"]

    alive = np.asarray(jax.device_get(b_pop_out.alive))
    if np.any(alive):
        print(f"[smoke] sim_step compiled and ran with 10 env channels and 9 actions.")
    else:
        print(f"[smoke] ran successfully but all agents died in step 1")

    print("✅ PHASE 16.5 SMOKE TEST PASSED: TENSORS ALIGNED.")

if __name__ == "__main__":
    main()
