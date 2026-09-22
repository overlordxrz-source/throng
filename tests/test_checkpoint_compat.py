import os
import json
import yaml
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.core import freeze, unfreeze

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config, make_obs_layout
from jax_sim.network_jax import (
    AgentNetworkJax, PredatorNetworkJax,
    init_agent_params, init_predator_params,
    ensure_aux_head_params, ensure_predator_params,
    graft_missing_param_subtrees, sanitize_agent_params,
)

def test_checkpoint_compatibility():
    # 2026-09-21 (Cam): this test used to call model.apply on RAW, ungrafted
    # checkpoint params -- a stricter bar than production ever hits (main_jax.py's
    # real resume path always grafts: graft_missing_param_subtrees against a
    # fresh template, then ensure_aux_head_params/ensure_predator_params).
    # Phase 19 hearths' own_state_dim 22 -> 29 change made that strict,
    # unrealistic bar fail permanently -- "a permanently-red test in a green
    # suite is not [fine]... the next real regression will hide exactly
    # there." Fixed by routing through the same grafting production actually
    # uses, for BOTH teams (test_hearth_checkpoint_pad.py separately proves
    # blue's pad is bit-identical to the pre-hearth checkpoint's own output;
    # this test is the only one covering red's ensure_predator_params path
    # against a real checkpoint, and is now a real smoke check of what
    # production does, not a check of what it deliberately avoids doing).
    # 1. Find checkpoint dir
    ckpt_dir = "/mnt/throng-runs/checkpoints"
    config_path = "/mnt/throng-runs/config.json"

    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.expanduser("~/throng_backup/checkpoints")
        config_path = os.path.expanduser("~/throng_backup/config.json")

    if not os.path.exists(ckpt_dir):
        print(f"SKIP: Checkpoint directory {ckpt_dir} not found. Skipping compatibility test.")
        return

    print(f"Loading checkpoint from: {ckpt_dir}")

    # 2. Load config. DEFAULT_CONFIG is a sparse fallback (e.g. it has no
    # `n_actions` key at all, silently defaulting downstream `.get("n_actions",
    # 8)` calls to a pre-Phase-18 shape) — base on the repo's actual
    # config.yaml instead, which is what production training actually runs,
    # and only layer a per-checkpoint config.json on top when one happens to
    # sit next to the backup.
    config = dict(DEFAULT_CONFIG)
    repo_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    if os.path.exists(repo_config_path):
        with open(repo_config_path, "r") as f:
            config.update(yaml.safe_load(f))
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config.update(json.load(f))
    config = _normalize_config(config)
    
    # 3. Instantiate Checkpoint Manager
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options=options)
    latest_step = ckpt_mngr.latest_step()
    
    if latest_step is None:
        print(f"SKIP: No checkpoint found in {ckpt_dir}.")
        return
        
    print(f"Latest step: {latest_step}")
    
    # 4. Construct Models exactly as main_jax.py
    _layout = make_obs_layout(
        signal_dim=config["signal_dim"],
        symbol_dim=config.get("symbol_dim", 8),
        memory_slots=config.get("memory_slots", 0),
        neighbor_k=config["neighbor_k"],
        local_cells=(2 * config["local_obs_radius"] + 1)**2,
        env_channels=int(config.get("env_channels", 15)),
        own_state_dim=int(config.get("own_state_dim", 22)),
    )
    _loc_env_start = _layout.loc_env_start
    _loc_env_end = _layout.loc_env_end
    _fwd_env_dim = _loc_env_end - _loc_env_start
    
    _p9 = config.get("phase9_canvas") or {}
    _cross_attn = bool(_p9.get("cross_attn_enabled", False))
    _cross_heads = int(_p9.get("cross_attn_num_heads", config["n_heads"]))
    _p12 = config.get("phase12_red") or {}
    _red_cross = bool(_p12.get("red_cross_attn_enabled", True))
    _red_vocab = int(_p12.get("red_vocab_size", config.get("vocab_size", 64)))
    
    obs_dim = _layout.total_dim
    
    model = AgentNetworkJax(
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        obs_dim=obs_dim,
        signal_dim=config["signal_dim"],
        symbol_dim=config["symbol_dim"],
        vocab_size=config["vocab_size"],
        vq_beta=float(config.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(config.get("vq_dead_code_reset", True)),
        memory_slots=config.get("memory_slots", 0),
        fwd_env_dim=_fwd_env_dim,
        cross_attn_enabled=_cross_attn,
        cross_attn_num_heads=_cross_heads,
        env_channels=int(config.get("env_channels", 15)),
        own_state_dim=int(config.get("own_state_dim", 22)),
        n_actions=int(config.get("n_actions", 8)),
        local_cells=(2 * config["local_obs_radius"] + 1)**2,
        neighbor_k=config["neighbor_k"],
    )
    
    _p14t = config.get("phase14_transcendental") or {}
    red_hidden_d = config.get("red_hidden_dim", config["hidden_dim"])
    model_red = PredatorNetworkJax(
        hidden_dim=red_hidden_d,
        neighbor_k=config["neighbor_k"],
        local_obs_radius=config["local_obs_radius"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        signal_dim=config["signal_dim"],
        symbol_dim=config["symbol_dim"],
        vocab_size=_red_vocab,
        vq_beta=float(config.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(config.get("vq_dead_code_reset", True)),
        simvq_w_clip=float(_p14t.get("simvq_w_clip", 2.0)),
        simvq_out_scale=float(_p14t.get("simvq_out_scale", 2.0)),
        vq_noise_scale=float(_p12.get("red_vq_noise_scale", 0.0)),
        memory_slots=config.get("memory_slots", 0),
        cross_attn_enabled=_red_cross,
        cross_attn_num_heads=_cross_heads,
        env_channels=int(config.get("env_channels", 15)),
        own_state_dim=int(config.get("own_state_dim", 22)),
        n_actions=int(config.get("n_actions", 8)),
    )
    
    # 5. Restore Checkpoint
    # Checkpoints saved on a GPU box record `cuda:0` sharding metadata; restoring
    # on a CPU-only dev machine with no target raises "Topology mismatch". Give
    # Orbax an explicit CPU-sharded target (same shapes, new device) instead of
    # trusting the checkpoint's own device metadata.
    from jax.sharding import SingleDeviceSharding
    _cpu = jax.devices("cpu")[0]
    _meta = ckpt_mngr.item_metadata(latest_step)
    _target = jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=SingleDeviceSharding(_cpu)),
        _meta,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    raw_restored = ckpt_mngr.restore(latest_step, args=ocp.args.StandardRestore(_target))

    source_dict = unfreeze(raw_restored)

    print("Checkpoint loaded. Grafting through the real production restore path...")

    # 5b. Graft exactly as main_jax.py's real resume path does: walk a fresh
    # model template and zero-pad/inject anything missing or shape-changed
    # (own_state_dim 22 -> 29 among them), rather than applying raw params.
    rng = jax.random.PRNGKey(0)
    bsz = 2
    b_obs = jnp.zeros((bsz, obs_dim))
    b_carries = jnp.zeros((bsz, config["hidden_dim"]))
    r_carries = jnp.zeros((bsz, red_hidden_d))

    b_template = unfreeze(init_agent_params(model, rng, b_carries, b_obs, config["n_layers"]))
    src_b = unfreeze(source_dict["b_params"])
    graft_missing_param_subtrees(src_b, b_template)
    b_params = sanitize_agent_params(
        ensure_aux_head_params(
            model, freeze(src_b), rng, config["hidden_dim"],
            obs_dim=obs_dim, n_layers=config["n_layers"],
        )
    )

    r_template = unfreeze(init_predator_params(model_red, rng, r_carries, b_obs, config["n_layers"]))
    src_r = unfreeze(source_dict["r_params"])
    graft_missing_param_subtrees(src_r, r_template)
    r_params = sanitize_agent_params(
        ensure_predator_params(
            model_red, freeze(src_r), rng, red_hidden_d,
            obs_dim=obs_dim, n_layers=config["n_layers"],
        )
    )
    
    try:
        new_b_c, b_outs = model.apply(
            {"params": b_params}, b_carries, b_obs, n_layers=config["n_layers"]
        )
        print("Blue network forward pass: SUCCESS")
    except Exception as e:
        print(f"Blue network forward pass FAILED: {e}")
        raise
        
    try:
        new_r_c, r_outs = model_red.apply(
            {"params": r_params}, r_carries, b_obs, n_layers=config["n_layers"], rngs={'dropout': rng}
        )
        print("Red network forward pass: SUCCESS")
    except Exception as e:
        print(f"Red network forward pass FAILED: {e}")
        raise
        
    print("OK: Checkpoint compatibility test passed. Shape matches and forward passes succeed.")

if __name__ == "__main__":
    test_checkpoint_compatibility()
