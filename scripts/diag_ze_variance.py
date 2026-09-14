#!/usr/bin/env python3
"""Cam's diagnostic 3 (2026-09-14), generalized into a calibration ladder
(Blocker 1, 2026-09-14).

Original ask: measure z_e variance across agents using the REAL checkpoint
from the just-stopped launch (ppo 2842-2861, codes_active collapsed to
1|3|3/64, VQ loss=2.30e-09). If z_e has collapsed to near-constant across
agents despite genuinely varied inputs, the encoder is the cause and the
codebook (which faithfully reports "nothing distinct to quantize") is
downstream, not broken.

Cam accepted that finding but flagged it as necessary, not sufficient: the
"20-25x below healthy" comparison was against a *hypothetical* uniform
spread, not a measured reference on this network. Two things this script
now gets that the single-checkpoint version couldn't:

  1. A measured healthy reference -- the spread this exact encoder produced
     when it was demonstrably working (Phase 18.6, confirmed healthy in
     THRONG.md: "blue VQ verified healthy, loss~=0.003-0.007, codes
     45|39|43/64").
  2. The onset step -- if an early Phase 18 checkpoint is already collapsed,
     the causal story is not "stage 0 caused this," it predates the CtD run
     entirely.

Same method at every step: real checkpoint restore (CPU-sharded target,
matching tests/test_checkpoint_compat.py), real forward pass, same fixed
RNG seeds for obs/carries across all steps so only the PARAMS differ between
measurements -- the ladder is only meaningful if the inputs are held
constant.

Usage:
  JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python3 scripts/diag_ze_variance.py <checkpoint_parent_dir> [step ...]

  <checkpoint_parent_dir> is the directory Orbax's CheckpointManager expects
  (containing per-step subdirectories, e.g. .../checkpoints/2763/).
  If no [step ...] is given, measures every step CheckpointManager finds.
  Pass explicit steps to measure only those (e.g. a costly Modal-volume
  fetch where you only downloaded specific steps).
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


def build_model_and_inputs(config: dict):
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

    N = 200  # matches production blue population size
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    # Genuinely varied per-agent observations, held IDENTICAL across every
    # checkpoint in the ladder -- so a difference in z_e spread can only be
    # the encoder's own doing (different params), never an artifact of
    # different inputs.
    obs = jax.random.normal(k1, (N, obs_dim)) * 0.1
    obs = obs.at[:, 2].set(0.5)  # avoid the feral mask
    carries = jax.random.normal(k2, (N, hidden_d)) * 0.1
    return model, n_layers, obs, carries


def restore_b_params(ckpt_mngr: ocp.CheckpointManager, step: int):
    cpu = jax.devices("cpu")[0]
    meta = ckpt_mngr.item_metadata(step)
    target = jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=SingleDeviceSharding(cpu)),
        meta,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    restored = ckpt_mngr.restore(step, args=ocp.args.StandardRestore(target))
    # Some backup checkpoints stored the flat dict directly under 'b_params';
    # accept both that and an already-flat structure defensively.
    return restored["b_params"] if "b_params" in restored else restored


def measure_ze_spread(model, n_layers, obs, carries, b_params) -> dict:
    """Real forward pass; returns per-slot + overall z_e cross-agent spread,
    plus codes_active (distinct codes used by these N agents this pass) --
    directly comparable to THRONG.md's documented healthy reference
    ("codes 45|39|43/64", Phase 18.6) and this session's collapsed reference
    ("codes_active=1|3|3", the stopped launch)."""
    _, outs = model.apply({"params": b_params}, carries, obs, n_layers)
    z_e = outs.z_e  # (N, 40): [8 zeros | slot0(12) | slot1(8) | slot2(12)]
    token_ids = outs.token_ids  # (N, 3)

    slot0 = z_e[:, 8:20]
    slot1 = z_e[:, 20:28]
    slot2 = z_e[:, 28:40]

    result = {}
    for i, (name, slot) in enumerate((("slot0", slot0), ("slot1", slot1), ("slot2", slot2))):
        per_dim_var = jnp.var(slot, axis=0)
        mean_var = float(jnp.mean(per_dim_var))
        max_abs = float(jnp.max(jnp.abs(slot)))
        rel_spread = mean_var / (max_abs ** 2 + 1e-12)
        codes_active = int(jnp.unique(token_ids[:, i]).shape[0])
        result[name] = {
            "mean_var": mean_var, "max_abs": max_abs, "rel_spread": rel_spread,
            "codes_active": codes_active,
        }

    overall = jnp.concatenate([slot0, slot1, slot2], axis=-1)
    result["overall"] = {
        "mean_var": float(jnp.mean(jnp.var(overall, axis=0))),
        "max_abs": float(jnp.max(jnp.abs(overall))),
    }
    result["overall"]["rel_spread"] = result["overall"]["mean_var"] / (result["overall"]["max_abs"] ** 2 + 1e-12)
    return result


def main():
    if len(sys.argv) < 2:
        print("usage: diag_ze_variance.py <checkpoint_parent_dir> [step ...]")
        sys.exit(1)
    ckpt_dir = sys.argv[1]
    requested_steps = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else None

    with open("config.yaml") as f:
        config = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(f)})

    model, n_layers, obs, carries = build_model_and_inputs(config)
    print(f"[diag] input obs std actually fed in (sanity check, should be ~0.1) = {float(jnp.std(obs)):.4f}")

    options = ocp.CheckpointManagerOptions(create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options=options)
    all_steps = sorted(ckpt_mngr.all_steps())
    if not all_steps:
        print(f"[diag] no checkpoints found under {ckpt_dir}")
        sys.exit(1)
    steps = requested_steps if requested_steps else all_steps
    missing = [s for s in steps if s not in all_steps]
    if missing:
        print(f"[diag] WARNING: requested steps not found in {ckpt_dir}: {missing}")
        steps = [s for s in steps if s in all_steps]

    print(f"[diag] measuring {len(steps)} checkpoint(s) from {ckpt_dir}: {steps}")

    import flax.errors

    ladder = []
    skipped = []
    for step in steps:
        print(f"\n[diag] === step {step} ===")
        try:
            b_params = restore_b_params(ckpt_mngr, step)
            spread = measure_ze_spread(model, n_layers, obs, carries, b_params)
        except flax.errors.ScopeParamShapeError as exc:
            # A checkpoint from before the current obs layout (own_state /
            # env_channels grafts documented in THRONG.md) has genuinely
            # different param shapes. Forcing it through the current
            # architecture would require grafting/padding -- which injects
            # fresh-initialized weights into exactly the dims this
            # measurement is trying to characterize, contaminating the
            # spread number. Skip and report rather than silently graft.
            print(f"[diag] SKIPPED step {step}: architecture-incompatible ({exc})")
            skipped.append(step)
            continue
        for name in ("slot0", "slot1", "slot2", "overall"):
            s = spread[name]
            ca = f" | codes_active={s['codes_active']}/64" if "codes_active" in s else ""
            print(
                f"[diag] {name}: mean cross-agent variance={s['mean_var']:.6e} | "
                f"max|value|={s['max_abs']:.6e} | relative spread={s['rel_spread']:.6e}{ca}"
            )
        codes_str = "|".join(str(spread[n]["codes_active"]) for n in ("slot0", "slot1", "slot2"))
        ladder.append((step, spread["overall"]["rel_spread"], codes_str))

    print("\n[diag] ===== CALIBRATION LADDER (overall relative spread, ascending step) =====")
    for step, rel_spread, codes_str in ladder:
        bar = "#" * max(1, int(rel_spread * 2000))
        print(f"[diag]   step {step:>6}: rel_spread={rel_spread:.6e}  codes={codes_str}/64  {bar}")
    if skipped:
        print(f"[diag] skipped (architecture-incompatible, pre-graft): {skipped}")


if __name__ == "__main__":
    main()
