#!/usr/bin/env python3
"""Cam's diagnostic 3 (2026-09-14): measure z_e variance across agents using
the REAL checkpoint from the just-stopped launch (ppo 2842-2861, codes_active
collapsed to 1|3|3/64, VQ loss=2.30e-09).

If z_e has collapsed to near-constant across agents despite genuinely varied
inputs, the encoder is the cause and the codebook (which faithfully reports
"nothing distinct to quantize") is downstream, not broken.

z_e = concat([z_e_cont(8 zeros), z_e_0(12), z_e_1(8), z_e_2(12)]) -- a
deterministic function of obs (minus the 4 zeroed metabolic dims) through one
Dense+ReLU (gwt_comms_1) then three more Dense heads (head_signal_slot{0,1,2}).
Offline, CPU only -- no GPU needed for this measurement.

Usage: JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python3 scripts/diag_ze_variance.py <checkpoint_dir>
"""
from __future__ import annotations

import sys
import yaml
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import SingleDeviceSharding

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.network_jax import AgentNetworkJax
from jax_sim.obs_layout import make_obs_layout


def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not ckpt_dir:
        print("usage: diag_ze_variance.py <checkpoint_dir_containing_step_subdir_parent>")
        sys.exit(1)

    with open("config.yaml") as f:
        config = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(f)})

    hidden_d = int(config["hidden_dim"])
    n_layers = int(config["n_layers"])
    n_actions = int(config.get("n_actions", 12))
    signal_dim = int(config["signal_dim"])
    symbol_dim = int(config["symbol_dim"])
    neighbor_k = int(config["neighbor_k"])
    env_channels = int(config.get("env_channels", 15))
    own_state_dim = int(config.get("own_state_dim", 22))
    mem_slots = int(config.get("memory_slots", 0))
    local_cells = (2 * int(config["local_obs_radius"]) + 1) ** 2

    layout = make_obs_layout(
        signal_dim=signal_dim, symbol_dim=symbol_dim, memory_slots=mem_slots,
        neighbor_k=neighbor_k, local_cells=local_cells, env_channels=env_channels,
        own_state_dim=own_state_dim,
    )
    obs_dim = layout.total_dim

    _p9 = config.get("phase9_canvas") or {}
    _fwd_env_dim = layout.loc_env_end - layout.loc_env_start
    model = AgentNetworkJax(
        hidden_dim=hidden_d,
        n_heads=config["n_heads"],
        n_layers=n_layers,
        obs_dim=obs_dim,
        signal_dim=config["signal_dim"],
        symbol_dim=config["symbol_dim"],
        vocab_size=config["vocab_size"],
        vq_beta=float(config.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(config.get("vq_dead_code_reset", True)),
        memory_slots=config.get("memory_slots", 0),
        fwd_env_dim=_fwd_env_dim,
        cross_attn_enabled=bool(_p9.get("cross_attn_enabled", False)),
        cross_attn_num_heads=int(_p9.get("cross_attn_num_heads", 4)),
        env_channels=env_channels,
        own_state_dim=own_state_dim,
        n_actions=n_actions,
        local_cells=local_cells,
        neighbor_k=neighbor_k,
    )

    # Restore the real checkpoint's b_params (CPU sharding, matching
    # tests/test_checkpoint_compat.py's pattern for a GPU-saved checkpoint).
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options=options)
    latest = ckpt_mngr.latest_step()
    print(f"[diag] restoring checkpoint step {latest} from {ckpt_dir}")
    meta = ckpt_mngr.item_metadata(latest)
    cpu = jax.devices("cpu")[0]
    target = jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=SingleDeviceSharding(cpu)),
        meta,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    restored = ckpt_mngr.restore(latest, args=ocp.args.StandardRestore(target))
    b_params = restored["b_params"]

    N = 200  # matches production blue population size
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    # Genuinely varied per-agent observations -- same construction pattern as
    # tests/test_severance_sweep.py -- so a collapsed z_e can only be the
    # encoder's own doing, not an artifact of degenerate/identical inputs.
    obs = jax.random.normal(k1, (N, obs_dim)) * 0.1
    obs = obs.at[:, 2].set(0.5)  # avoid the feral mask
    carries = jax.random.normal(k2, (N, hidden_d)) * 0.1

    _, outs = model.apply({"params": b_params}, carries, obs, n_layers)
    z_e = outs.z_e  # (N, 40): [8 zeros | slot0(12) | slot1(8) | slot2(12)]

    slot0 = z_e[:, 8:20]
    slot1 = z_e[:, 20:28]
    slot2 = z_e[:, 28:40]

    for name, slot in (("slot0", slot0), ("slot1", slot1), ("slot2", slot2)):
        per_dim_var = jnp.var(slot, axis=0)          # variance across agents, per dim
        mean_var = float(jnp.mean(per_dim_var))
        max_abs = float(jnp.max(jnp.abs(slot)))
        # Relative spread: how big is the cross-agent variance relative to the
        # signal's own magnitude? Near 0 with genuinely varied inputs means
        # the encoder produces (near-)the same vector regardless of agent.
        rel_spread = mean_var / (max_abs ** 2 + 1e-12)
        print(
            f"[diag] {name}: mean cross-agent variance={mean_var:.6e} | "
            f"max|value|={max_abs:.6e} | relative spread={rel_spread:.6e}"
        )

    overall = jnp.concatenate([slot0, slot1, slot2], axis=-1)
    print(f"[diag] overall z_e (36 non-zeroed dims) mean cross-agent variance = {float(jnp.mean(jnp.var(overall, axis=0))):.6e}")
    print(f"[diag] overall z_e max|value| = {float(jnp.max(jnp.abs(overall))):.6e}")
    print(f"[diag] input obs std actually fed in (sanity check, should be ~0.1) = {float(jnp.std(obs)):.4f}")


if __name__ == "__main__":
    main()
