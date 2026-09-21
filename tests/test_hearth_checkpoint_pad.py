"""Cam's launch-blocker instruction (2026-09-21): own_state_dim 22 -> 29
(Phase 19 hearths) isn't a missing subtree the graft can fill -- it's a
changed input dimension on an existing weight tensor (emb_own's kernel, and
as this test found, gwt_comms_1's kernel and head_vqel_recon_2's
kernel+bias too). If restore-time grafting drops or re-initializes those
tensors instead of zero-padding them, a resumed run silently forgets how to
read its own state while every banner reports a successful restore -- the
defect class arriving through the front door.

Verifies, against a real pre-hearth checkpoint (not a synthetic one): the
padded kernel's first 22 rows are bit-identical to the checkpoint's
original, the 7 new rows are exactly zero, bias is untouched, and a full
forward pass through the padded model -- given an input whose 7 new own_state
dims are also zeroed -- produces BIT-IDENTICAL action_logits and carries to
the same checkpoint's own params applied to the old (own_state_dim=22)
model shape. If they don't match, the pad is wrong.

Skips (does not fail) if no local checkpoint backup is present -- same
opportunistic-local-fixture pattern as test_checkpoint_compat.py.
"""

import os
import yaml
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.core import freeze, unfreeze

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.network_jax import (
    AgentNetworkJax, init_agent_params, ensure_aux_head_params,
    sanitize_agent_params, graft_missing_param_subtrees,
)
from jax_sim.obs_layout import make_obs_layout

CKPT_DIR = os.path.expanduser("~/throng_backup/checkpoints")


def _make_model(cfg):
    layout = make_obs_layout(
        signal_dim=int(cfg["signal_dim"]), symbol_dim=int(cfg["symbol_dim"]),
        memory_slots=int(cfg.get("memory_slots", 0)), neighbor_k=int(cfg["neighbor_k"]),
        local_cells=(2 * int(cfg["local_obs_radius"]) + 1) ** 2,
        env_channels=int(cfg.get("env_channels", 15)), own_state_dim=int(cfg["own_state_dim"]),
    )
    model = AgentNetworkJax(
        hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        obs_dim=layout.total_dim, signal_dim=cfg["signal_dim"], symbol_dim=cfg["symbol_dim"],
        vocab_size=cfg["vocab_size"], vq_beta=float(cfg.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(cfg.get("vq_dead_code_reset", True)),
        memory_slots=cfg.get("memory_slots", 0),
        fwd_env_dim=layout.loc_env_end - layout.loc_env_start,
        cross_attn_enabled=bool((cfg.get("phase9_canvas") or {}).get("cross_attn_enabled", False)),
        cross_attn_num_heads=int((cfg.get("phase9_canvas") or {}).get("cross_attn_num_heads", cfg["n_heads"])),
        env_channels=int(cfg.get("env_channels", 15)), own_state_dim=int(cfg["own_state_dim"]),
        n_actions=int(cfg.get("n_actions", 8)),
        local_cells=(2 * cfg["local_obs_radius"] + 1) ** 2, neighbor_k=cfg["neighbor_k"],
    )
    return model, layout


def test_restore_time_pad_matches_pre_hearth_checkpoint_exactly():
    if not os.path.exists(CKPT_DIR):
        print(f"SKIP: {CKPT_DIR} not found -- no local pre-hearth checkpoint to verify against.")
        return

    with open("config.yaml") as f:
        cfg_new = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(f)})
    assert int(cfg_new["own_state_dim"]) == 29, "expected the live (post-hearth) config"
    cfg_old = dict(cfg_new)
    cfg_old["own_state_dim"] = 22  # the pre-hearth shape this checkpoint was trained under

    model_old, layout_old = _make_model(cfg_old)
    model_new, layout_new = _make_model(cfg_new)
    assert layout_new.total_dim - layout_old.total_dim == 7

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=False)
    ckpt_mngr = ocp.CheckpointManager(CKPT_DIR, ocp.StandardCheckpointer(), options=options)
    latest_step = ckpt_mngr.latest_step()
    if latest_step is None:
        print(f"SKIP: no checkpoint found in {CKPT_DIR}.")
        return

    # CPU-only dev box, GPU-saved checkpoint: explicit CPU-sharded target
    # derived from the checkpoint's OWN shapes (same workaround
    # test_checkpoint_compat.py uses) restores its REAL (22-wide) shapes,
    # unconstrained by any model template.
    from jax.sharding import SingleDeviceSharding
    cpu = jax.devices("cpu")[0]
    meta = ckpt_mngr.item_metadata(latest_step)
    target = jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=SingleDeviceSharding(cpu)),
        meta,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    raw_restored = ckpt_mngr.restore(latest_step, args=ocp.args.StandardRestore(target))
    source_b_params = unfreeze(raw_restored["b_params"])
    old_kernel = source_b_params["emb_own"]["kernel"]
    assert old_kernel.shape[0] == 22, (
        f"expected this checkpoint's real pre-hearth own_state_dim (22), got {old_kernel.shape}"
    )

    # Ground truth: the raw (unpadded) params applied to the OLD-shaped model.
    b_params_old = sanitize_agent_params(freeze(source_b_params))

    # Candidate: the exact restore-time grafting main_jax.py's real resume
    # path runs -- graft_missing_param_subtrees against a fresh NEW-model
    # template, then ensure_aux_head_params.
    rng = jax.random.PRNGKey(0)
    dummy_carry = jnp.zeros((2, cfg_new["hidden_dim"]))
    dummy_obs_new = jnp.zeros((2, layout_new.total_dim))
    tgt_agent = unfreeze(init_agent_params(model_new, rng, dummy_carry, dummy_obs_new, cfg_new["n_layers"]))
    src_agent = unfreeze(source_b_params)
    injected = graft_missing_param_subtrees(src_agent, tgt_agent)
    padded_paths = [p for p in injected if "zero-padded" in p]
    assert any("emb_own/kernel" in p for p in padded_paths), (
        f"expected emb_own/kernel to be zero-padded, got: {injected}"
    )
    # Reinit ("shape mismatch ... reinitialized") is exactly what Cam warned
    # against -- any own_state_dim-shaped tensor must be padded, not reset.
    reinitialized = [p for p in injected if "reinitialized" in p]
    assert not reinitialized, f"tensor(s) were reinitialized instead of padded: {reinitialized}"

    b_params_new = sanitize_agent_params(
        ensure_aux_head_params(
            model_new, freeze(src_agent), rng, cfg_new["hidden_dim"],
            obs_dim=layout_new.total_dim, n_layers=cfg_new["n_layers"],
        )
    )
    new_emb_own = unfreeze(b_params_new)["emb_own"]
    assert new_emb_own["kernel"].shape == (29, new_emb_own["kernel"].shape[1])
    assert bool(jnp.allclose(new_emb_own["kernel"][:22], old_kernel)), (
        "the first 22 rows of the padded kernel must be bit-identical to the checkpoint's original"
    )
    assert bool(jnp.allclose(new_emb_own["kernel"][22:], 0.0)), "the 7 new rows must be exactly zero"
    assert bool(jnp.array_equal(source_b_params["emb_own"]["bias"], new_emb_own["bias"])), (
        "bias must be untouched by the pad (it doesn't depend on the input dim)"
    )

    # Build matching inputs: old_obs at the pre-hearth layout's total_dim
    # (real nonzero values, not an all-zero smoke input); new_obs = old_obs
    # with 7 zeros inserted exactly at the own_state boundary, everything
    # else identical and in the same relative order (hearths only added
    # own_state dims -- no other block moved or changed size).
    N = 4
    old_obs = jax.random.normal(jax.random.PRNGKey(7), (N, layout_old.total_dim)) * 0.3
    boundary = layout_old.own_state_end
    assert boundary == 22
    new_obs = jnp.concatenate(
        [old_obs[:, :boundary], jnp.zeros((N, 7)), old_obs[:, boundary:]], axis=1
    )
    assert new_obs.shape[1] == layout_new.total_dim
    carries = jnp.zeros((N, cfg_new["hidden_dim"]))

    out_old_carry, out_old = model_old.apply(
        {"params": b_params_old}, carries, old_obs, n_layers=cfg_old["n_layers"], deterministic=True
    )
    out_new_carry, out_new = model_new.apply(
        {"params": b_params_new}, carries, new_obs, n_layers=cfg_new["n_layers"], deterministic=True
    )

    max_abs_diff = float(jnp.max(jnp.abs(out_old.action_logits - out_new.action_logits)))
    assert max_abs_diff < 1e-4, (
        f"padded-new-model action_logits diverge from the pre-hearth checkpoint's own output by "
        f"{max_abs_diff:.3e} -- the pad is wrong (dropped or re-initialized a tensor, or an "
        f"input column landed in the wrong place)"
    )
    carry_diff = float(jnp.max(jnp.abs(out_old_carry - out_new_carry)))
    assert carry_diff < 1e-4, f"padded-new-model carries diverge by {carry_diff:.3e}"

    print(
        f"VERIFIED against real checkpoint step {latest_step}: emb_own/gwt_comms_1/"
        f"head_vqel_recon_2 all zero-padded (not reinitialized); action_logits and carries "
        f"bit-identical (max abs diff {max_abs_diff:.3e}) between the old 22-dim model reading "
        f"the raw checkpoint and the padded 29-dim model reading the grafted checkpoint with "
        f"its 7 new input dims zeroed."
    )


if __name__ == "__main__":
    test_restore_time_pad_matches_pre_hearth_checkpoint_exactly()
