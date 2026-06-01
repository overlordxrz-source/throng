#!/usr/bin/env python3
"""Phase 14 smoke: sim_step wire cut + one VQEL monologue update (tensor shapes)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
import jax
import jax.numpy as jnp

from agents.network_torch import compute_obs_dim_torch, compute_fwd_env_dim
from jax_sim.grid_jax import GridState
from jax_sim.main_jax import _normalize_config, make_sim_step
from jax_sim.network_jax import (
    AgentNetworkJax,
    init_agent_params,
    make_model_apply,
    make_vqel_monologue_apply,
)
from jax_sim.obs_layout import make_obs_layout
from jax_sim.population_jax import init_population
from jax_sim.rl_jax import create_optimizer, vqel_monologue_update


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
            "phase14_vqel": {
                "monologue_enabled": True,
                "monologue_lr": 3e-5,
                "recon_coef": 1.0,
                "hash_penalty_coef": 0.5,
                "vq_coef_monologue": 1.0,
                "dialogue_signal_mode": "ste",
            },
        }
    )
    p9 = dict(cfg.get("phase9_canvas") or {})
    p9["imagination_gating_enabled"] = False
    p9["cross_attn_enabled"] = False
    cfg["phase9_canvas"] = p9
    p12 = dict(cfg.get("phase12_coevolution") or {})
    p12["red_comms_enabled"] = False
    cfg["phase12_coevolution"] = p12
    return cfg


def main() -> None:
    config = _tiny_config()
    gs = int(config["grid_size"])
    max_pop = int(config["max_pop"])
    max_pop_red = int(config["max_pop_red"])
    hidden_d = int(config["hidden_dim"])
    n_layers = int(config["n_layers"])
    sig_d = int(config["signal_dim"])
    obs_dim = compute_obs_dim_torch(config)
    fwd_env_dim = compute_fwd_env_dim(config)

    layout = make_obs_layout(
        signal_dim=sig_d,
        symbol_dim=int(config["symbol_dim"]),
        memory_slots=0,
        neighbor_k=int(config["neighbor_k"]),
    )
    spatial_ego_dim = layout.spatial_ego_dim

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
        cross_attn_enabled=False,
        neighbor_k=int(config["neighbor_k"]),
    )
    model_apply = make_model_apply(model)
    monologue_apply = make_vqel_monologue_apply(model)

    rng = jax.random.PRNGKey(14)
    keys = jax.random.split(rng, 8)
    carry = jnp.zeros((max_pop, hidden_d))
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
    sigs = np.asarray(jax.device_get(b_pop_out.signals))
    wire_max = 0.0
    if alive.any():
        alive_sigs = sigs[alive]
        wire_max = float(np.max(np.abs(alive_sigs)))
        if wire_max > 1e-8:
            raise SystemExit(
                f"FAIL: monologue wire cut — alive blue signal max abs={wire_max}"
            )
    print(f"[smoke] sim_step OK — wire cut max|signal|={wire_max:.2e}")

    obs_1 = np.asarray(jax.device_get(blue["obs"]))[None, ...]
    carries_1 = np.asarray(jax.device_get(blue["carries"]))[None, ...]
    alive_1 = np.asarray(jax.device_get(blue["alive"]))[None, ...].astype(np.float32)

    _z_q, _tok, spatial_hat, spatial_ego, _loss_vq, _z_e = monologue_apply(
        params, jnp.array(carries_1[0]), jnp.array(obs_1[0]), n_layers
    )
    if tuple(spatial_hat.shape) != (max_pop, spatial_ego_dim):
        raise SystemExit(
            f"FAIL: spatial_hat {spatial_hat.shape} != (N={max_pop}, ego_dim={spatial_ego_dim})"
        )
    if tuple(spatial_ego.shape) != (max_pop, spatial_ego_dim):
        raise SystemExit(f"FAIL: spatial_ego shape {spatial_ego.shape}")

    p14 = config["phase14_vqel"]
    opt = create_optimizer(float(p14["monologue_lr"]), 2.0)
    opt_state = opt.init(params)
    params, opt_state, metrics = vqel_monologue_update(
        params,
        opt_state,
        opt,
        monologue_apply,
        obs_1,
        carries_1,
        n_layers,
        alive_np=alive_1,
        key=keys[5],
        minibatch_size=int(config["ppo_minibatch_size"]),
        recon_coef=float(p14["recon_coef"]),
        hash_penalty_coef=float(p14["hash_penalty_coef"]),
        vq_coef=float(p14["vq_coef_monologue"]),
    )
    total = float(metrics.get("vqel_total_loss", float("nan")))
    if not np.isfinite(total):
        raise SystemExit(f"FAIL: vqel_total_loss not finite: {total}")
    for k in ("vqel_recon_mse", "vqel_hash_penalty"):
        v = float(metrics.get(k, float("nan")))
        if not np.isfinite(v):
            raise SystemExit(f"FAIL: {k} not finite: {v}")

    print(
        f"[smoke] obs_dim={obs_dim} spatial_ego_dim={spatial_ego_dim} "
        f"vqel_total={total:.6f} recon_mse={metrics['vqel_recon_mse']:.6f}"
    )
    print("✅ PHASE 14 SMOKE TEST PASSED: TENSORS ALIGNED.")


if __name__ == "__main__":
    main()
