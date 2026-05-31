#!/usr/bin/env python3
"""One-step smoke: PredatorNetworkJax forward + red VQ codes (Phase 12)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import jax
import jax.numpy as jnp

from agents.network_torch import compute_obs_dim_torch
from jax_sim.network_jax import PredatorNetworkJax, init_predator_params, make_model_apply


def main() -> None:
    cfg_path = ROOT / "config_phase7.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config.setdefault("n_heads", config.get("brain_n_heads", 4))
    config.setdefault("n_layers", 4)
    config.setdefault("hidden_dim", config.get("agent_hidden_dim", 256))
    config.setdefault("phase12_coevolution", {})
    config["phase12_coevolution"]["red_comms_enabled"] = True

    hidden = int(config.get("red_hidden_dim", 128))
    n_agents = 8
    n_layers = int(config["n_layers"])
    obs_dim = compute_obs_dim_torch(config)
    vocab = int(config["phase12_coevolution"].get("red_vocab_size", 64))

    model = PredatorNetworkJax(
        hidden_dim=hidden,
        neighbor_k=config["neighbor_k"],
        local_obs_radius=config["local_obs_radius"],
        n_heads=config["n_heads"],
        n_layers=n_layers,
        signal_dim=config["signal_dim"],
        symbol_dim=config["symbol_dim"],
        vocab_size=vocab,
        cross_attn_enabled=True,
        cross_attn_num_heads=int(
            (config.get("phase9_canvas") or {}).get("cross_attn_num_heads", 4)
        ),
        memory_slots=config.get("memory_slots", 0),
    )
    rng = jax.random.PRNGKey(0)
    carry = jnp.zeros((n_agents, hidden))
    obs = jax.random.normal(rng, (n_agents, obs_dim)) * 0.1
    params = init_predator_params(model, rng, carry, obs, n_layers)
    apply = make_model_apply(model)
    new_carry, outs = apply(params, carry, obs, n_layers)
    token_ids = outs[5]
    loss_vq = outs[6]
    z_e = outs[7]
    n_unique = int(len(jnp.unique(token_ids)))
    print(f"[smoke] obs_dim={obs_dim} hidden={hidden} vocab={vocab}")
    print(f"[smoke] red_codes_active (batch)={n_unique}/{vocab}")
    print(f"[smoke] loss_vq mean={float(loss_vq.mean()):.4f} z_e std={float(z_e.std()):.4f}")
    print(f"[smoke] carry delta norm={float(jnp.linalg.norm(new_carry - carry)):.4f}")
    if n_unique < 1:
        raise SystemExit("FAIL: no VQ codes active")
    if jnp.isnan(loss_vq).any() or jnp.isnan(z_e).any():
        raise SystemExit("FAIL: NaN in VQ path")
    print("[smoke] OK — PredatorNetworkJax compiles and emits discrete tokens")


if __name__ == "__main__":
    main()
