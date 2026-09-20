from __future__ import annotations

"""
jax_sim/main_jax.py — Full simulation loop with jax.lax.scan.

Design:
  1. Init: create grid, populations, network params, optimizer state
  2. Rollout: scan over T steps, each step = observe → forward → act → env step
  3. Update: GAE + PPO gradient step
  4. Repeat

Everything inside scan is @jit-compiled to a single XLA kernel.
"""

import os
import uuid
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress INFO/WARN
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")  # disable autotune spam
# Leave headroom for PPO backward after rollout scan (default JAX grabs ~90% of VRAM).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import orbax.checkpoint as ocp
from flax.training import train_state
import optax
from typing import Dict, Tuple, Any
import yaml

from jax_sim.grid_jax import (
    GridState, wrap, apply_moves, consume_resources, apply_catches,
    write_to_grid, decay_grid, decay_barrier_grid, get_local_patches, get_neighbour_signals,
    generate_puzzle_nodes, update_puzzle_grid, decay_puzzle_timeout, check_puzzle_solved,
    generate_resource_patches, generate_shelter_spots, generate_contested_nodes,
    update_scent_trails, material_zone_masks, resolve_crafting,
)
from communication.analysis import SignalCorpusWriter
from jax_sim.population_jax import (
    PopState, init_population, kill_agents, update_memory_buffer, apply_mind_meld,
    apply_auto_reproduce
)
from jax_sim.action_space import MASKED_ACTIONS, mask_disabled_actions, masked_actions_banner
from jax_sim.ctd_ramp import staged_recipe_counts, red_shaping_term, CRAFT_RAMP_STAGE_UNITS
from jax_sim.network_jax import (
    AgentNetworkJax,
    AUX_HEAD_KEYS,
    PredatorNetworkJax,
    ensure_aux_head_params,
    ensure_predator_params,
    graft_missing_param_subtrees,
    dead_code_reset_codebook_params,
    init_agent_params,
    init_predator_params,
    reset_predator_vq_on_resume,
    reset_confidence_head_on_resume,
    make_model_apply,
    make_vqel_monologue_apply,
    params_apply_variables,
    sanitize_agent_params,
)
from jax_sim.rl_jax import (
    compute_gae,
    ppo_loss,
    create_optimizer,
    ppo_update,
    auxiliary_update,
    red_auxiliary_update,
    vqel_monologue_update,
)
from jax_sim.obs_layout import make_obs_layout

def apply_medal_adr_carry_reset(pop: PopState, carries: jnp.ndarray, prob: float):
    valid_ages = jnp.where(pop.alive, pop.ages, -1)
    n_alive = jnp.sum(pop.alive)
    n_reset = jnp.ceil(prob * n_alive).astype(jnp.int32)
    
    sorted_idx = jnp.argsort(-valid_ages)
    ranks = jnp.argsort(sorted_idx)
    reset_mask = pop.alive & (ranks < n_reset)
    
    new_carries = jnp.where(reset_mask[:, None], 0.0, carries)
    new_steps = jnp.where(reset_mask, 0, pop.steps_since_dropout)
    
    return pop.replace(steps_since_dropout=new_steps), new_carries, jnp.sum(reset_mask)

from jax_sim.observations_jax import (
    RED_NEIGHBOR_SIGNAL_API_VERSION,
    RED_SENSE_API_VERSION,
    build_observations_jax,
)


# ── Config defaults ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "grid_size": 128,
    "population_size": 500,
    "red_population_size": 75,
    "neighbor_k": 6,
    "local_obs_radius": 2,
    "signal_dim": 40,
    "symbol_dim": 16,
    "agent_hidden_dim": 256,
    "brain_n_heads": 4,
    "n_layers": 4,
    "brain_token_dim": 128,
    "vocab_size": 64,
    "vq_beta": 0.25,
    "vq_loss_coef": 0.1,
    "vq_dead_code_reset": True,
    "memory_buffer_size": 20,
    "ppo_rollout_steps": 512,
    "ppo_epochs": 1,
    "ppo_clip": 0.2,
    "ppo_lr": 1e-4,
    "ppo_gamma": 0.99,
    "ppo_gae_lam": 0.95,
    "ppo_vf_coef": 0.25,
    "ppo_entropy_coef": 0.05,
    "ppo_max_grad_norm": 2.0,
    "reward_blue_alive": 0.05,
    "reward_blue_caught": -1.0,
    "reward_red_catch": 1.0,
    "reward_red_starve_per_step": -0.01,
    "resource_decay": 0.05,
    "symbol_decay": 0.993,
    "culture_fast_decay": 0.90,
    "culture_slow_decay": 0.995,
    "wall_density": 0.15,
    "n_resource_patches": 8,
    "n_shelter_spots": 5,
    "n_contested_nodes": 3,
    "red_detection_radius": 0,
    "max_age": 500,
    "energy_decay": 0.001,
    "starvation_threshold": 0.05,
    "tom_reward_coef": 0.002,
    "puzzle_enabled": False,
}

def _normalize_config(cfg: Dict) -> Dict:
    """Map PyTorch config names to JAX config names."""
    cfg = dict(cfg)
    if "n_actions" not in cfg:
        # AUDIT_SEP2026.md Finding 8's cousin: every model-construction call
        # site used to read config.get("n_actions", 8) — a silent fallback to
        # a pre-Phase-18 action-space size. test_checkpoint_compat.py caught
        # this in the wild: with no config.json sidecar next to a real backup
        # checkpoint, that default silently built an 8-action model against a
        # checkpoint whose head_action kernel is (256, 12). A wrong n_actions
        # doesn't just misbehave — it's a genuine checkpoint-shape footgun
        # (Rule "action amputation = logit-mask only" exists for exactly this
        # class of risk), so this must fail loudly rather than pick a guess.
        raise KeyError(
            "config is missing 'n_actions' — refusing to silently default to "
            "a pre-Phase-18 action-space size. Set n_actions explicitly in "
            "config.yaml (or whatever config source is being loaded)."
        )
    cfg.setdefault("max_pop", cfg.get("population_size", 500))
    cfg.setdefault("max_pop_red", cfg.get("red_population_size", 75))
    cfg.setdefault("hidden_dim", cfg.get("agent_hidden_dim", 256))
    cfg.setdefault("n_heads", cfg.get("brain_n_heads", 4))
    cfg.setdefault("n_layers", cfg.get("n_layers", 4))
    cfg.setdefault("vocab_size", cfg.get("vocab_size", 64))
    cfg.setdefault("memory_slots", cfg.get("memory_buffer_size", 20))
    cfg.setdefault("ppo_rollout_steps", cfg.get("ppo_rollout_steps", 512))

    # Phase 14.2 — hoist apex-predator decay to top-level for sim_step + telemetry
    _p14t = cfg.get("phase14_transcendental") or {}
    _blue_decay = float(cfg.get("energy_decay", 0.001))
    if "red_energy_decay" in _p14t:
        cfg["red_energy_decay"] = float(_p14t["red_energy_decay"])
    else:
        cfg["red_energy_decay"] = _blue_decay
        print(
            "[JAX] WARN: phase14_transcendental.red_energy_decay missing — "
            f"falling back to energy_decay={_blue_decay}",
            flush=True,
        )
    return cfg


def assert_checkpoint_dir_is_not_protected_backup(resolved_ckpt_dir: str) -> None:
    """
    STRUCTURAL SAFEGUARD (Sep 2026 restart pre-flight, Cam's instruction):
    called at the one and only place a WRITE-capable CheckpointManager gets
    constructed for a run — refuse outright if `checkpoint_dir` resolves
    anywhere under the local recovered-backup directory. That backup
    (~/throng_backup) is irreplaceable physical evidence (it is the only
    copy of the corpus behind the Slot 2 ATE); a training run pointed at it
    by a stale/typo'd config would start writing new checkpoint steps into
    it. "Remember not to do this" is exactly the failure mode Rule 13 exists
    to eliminate — make it impossible instead.

    `resolved_ckpt_dir` must already be an absolute, symlink-resolved path
    (as produced by `Path(...).resolve()`), matching how the caller derives
    `_protected_backup_root` below — a caller that skips resolution could
    defeat this check with a relative path or a symlink.
    """
    protected_backup_root = str(Path(os.path.expanduser("~/throng_backup")).resolve())
    if resolved_ckpt_dir == protected_backup_root or resolved_ckpt_dir.startswith(protected_backup_root + os.sep):
        raise ValueError(
            f"checkpoint_dir resolves to {resolved_ckpt_dir!r}, inside the protected "
            f"recovered-backup directory ({protected_backup_root!r}). This directory is "
            f"read-only source material, never a live training checkpoint_dir. Copy the "
            f"checkpoint you need into a separate working directory and point "
            f"checkpoint_dir there instead."
        )


def find_fossil_checkpoints(ckpt_mngr, resume_target: int) -> list:
    """
    STRUCTURAL SAFEGUARD (2026-09-15, Cam's instruction): the guard that
    would have caught both of that day's checkpoint losses before a dollar
    was spent. Orbax's `max_to_keep` retention prunes by STEP NUMBER, not
    save recency -- a checkpoint directory holding a step numerically ahead
    of `resume_target` (a fossil from an abandoned lineage, or any other
    reason the directory isn't single-lineage) means every new checkpoint
    the resumed run saves is "older" than that fossil by the metric Orbax
    actually uses, and gets silently garbage collected on save, before
    `Volume.commit()` is even in the picture. Nothing downstream can detect
    this: "[CKPT] Saved" / "[CKPT] Committed" print unconditionally
    regardless of whether the save survived retention. See
    docs/THE-ECOLOGY-NEVER-RAN.md instance 6.

    Returns the sorted list of fossil steps (empty if the directory is
    clean relative to `resume_target`). Callers should refuse to launch if
    this is non-empty -- checked here, not left to be discovered 7.5 hours
    later.
    """
    all_steps = sorted(int(s) for s in ckpt_mngr.all_steps())
    return [s for s in all_steps if s > resume_target]


# ── Rollout → CPU (free GPU before PPO backward) ───────────────────────────

def _rollout_to_cpu(rollout_data: Dict) -> Dict:
    """Move scan outputs off GPU so PPO backward has room (obs alone ~2.4GB)."""
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jax.Array, jnp.ndarray)) else x,
        rollout_data,
    )


# ── Single simulation step (for scan) ──────────────────────────────────────

def make_sim_step(
    config: Dict,
    model: AgentNetworkJax,
    model_apply=None,
    r_model_apply=None,
):
    """Factory: returns a jittable step function.
    Params are passed through carry to avoid stale closure capture."""
    if model_apply is None:
        model_apply = make_model_apply(model)
    _r_apply = r_model_apply if r_model_apply is not None else model_apply
    gs = config["grid_size"]
    K = config["neighbor_k"]
    r = config["local_obs_radius"]
    sym_d = config["symbol_dim"]
    sig_d = config["signal_dim"]
    hidden_d = config["hidden_dim"]

    # Pre-extract config values for JIT closure
    _reward_blue_alive = float(config.get("reward_blue_alive", 0.05))
    _reward_blue_caught = float(config.get("reward_blue_caught", -1.0))
    _reward_move = float(config.get("reward_move", 0.01))
    _reward_resource = float(config.get("reward_resource", 0.1))
    _reward_red_catch = float(config.get("reward_red_catch", 1.0))
    _reward_red_move = float(config.get("reward_red_move", 0.0))
    _reward_red_starve = float(config.get("reward_red_starve_per_step", -0.01))
    _red_catch_radius = int(config.get("red_catch_radius", 1))
    _red_catch_prob = float(config.get("red_catch_prob", 1.0))
    _reward_craft_success = float(config.get("reward_craft_success", 3.0))
    _reward_futile_craft = float(config.get("reward_futile_craft", -0.20))

    # CtD competence ramp (jax_sim/ctd_ramp.py) -- Cam's sign-off, 2026-09-14.
    # Crafting stages (solo -> pair -> full) are fixed by
    # jax_sim.ctd_ramp.CRAFT_RAMP_STAGE_UNITS, not configurable per-run.
    _ramp_cfg = config.get("ctd_competence_ramp", {})
    _red_ramp_beta = float(_ramp_cfg.get("red_ramp_beta", 2.5))
    _ppo_gamma_for_shaping = float(config.get("ppo_gamma", 0.999))

    # Phase 16 parameters
    p16 = config.get("phase16_combinatorial_syntax", {})
    _rew_small_blue = float(p16.get("reward_small_blue", 3.0))
    _rew_big_green_coop = float(p16.get("reward_big_green_success", 8.0))
    _rew_big_green_solo_pen = float(p16.get("reward_big_green_solo_penalty", -1.0))
    _rew_big_green_solo_catch = float(p16.get("reward_big_green_solo_catchable", 2.0))
    _rew_coord = float(p16.get("r_coord", 1.5))
    _coop_threshold_step = int(p16.get("coop_threshold_step", 100000))
    _puzzle_reward = float(config.get("puzzle_reward", 5.0))
    _energy_decay = float(config["energy_decay"])
    _red_energy_decay = float(config["red_energy_decay"])
    _starv_thresh = float(config["starvation_threshold"])
    _max_age = int(config["max_age"])
    _min_pop_blue = int(config.get("min_population", 200))
    _repro_energy_thresh = float(config.get("repro_energy_thresh", 0.8))
    _repro_energy_cost = float(config.get("repro_energy_cost", 0.4))
    _mind_meld = config.get("mind_meld_enabled", False)
    _mm_radius = int(config.get("mind_meld_radius", 1))
    _mm_rate = float(config.get("mind_meld_rate", 0.1))
    _mm_dir = config.get("mind_meld_direction", "older_to_younger")
    _scent_intensity = float(config.get("scent_intensity", 0.8))
    _scent_decay_steps = int(config.get("scent_decay_steps", 20))
    _red_starvation_steps = int(config.get("red_starvation_steps", 400))
    _contested_min_harv = int(config.get("contested_min_harvesters", 2))
    _resource_max = float(config.get("resource_max", 1.0))
    _resource_spawn_boost = float(config.get("resource_spawn_boost", 0.2))
    _n_layers = config["n_layers"]
    _p9 = config.get("phase9_canvas") or {}
    _img_gate_enabled = bool(_p9.get("imagination_gating_enabled", False))
    _conf_multiplier = float(_p9.get("confidence_multiplier", 1.0))
    _imagination_k = int(_p9.get("imagination_k", 5))
    _imagination_metabolic_delta = float(
        _p9.get("imagination_metabolic_delta", 0.0005)
    )
    _imagination_gamma = float(
        _p9.get("imagination_gamma", config.get("ppo_gamma", 0.999))
    )
    # Default 5 = legacy behaviour (imagine Stay + N/S/E/W only). Set to n_actions
    # (12) to let the epistemic gate deliberate over Strike/Push/Guard and the
    # Phase-18 tool actions too. Reversible knob; off by default.
    _imagine_n_actions = int(_p9.get("imagination_n_actions", 12))
    _p14 = config.get("phase14_vqel") or {}
    _vqel_monologue = bool(_p14.get("monologue_enabled", False))
    _dialogue_signal_mode = str(_p14.get("dialogue_signal_mode", "ste")).lower()
    
    _p15 = config.get("phase15_cumulative_culture") or {}
    _medal_adr_enabled = bool(_p15.get("medal_adr_enabled", False))
    _medal_adr_prob = float(_p15.get("medal_adr_prob", 0.0))
    
    _imagine_fn = None
    if _img_gate_enabled:
        from jax_sim.imagination_jax import make_imagination_fn
        _imagine_fn = make_imagination_fn(
            model, K=_imagination_k, gamma=_imagination_gamma,
            n_imagine_actions=_imagine_n_actions,
        )
        print(
            f"[JAX] Epistemic gate: K={_imagination_k} gamma={_imagination_gamma} "
            f"imagine_actions={_imagine_n_actions}/{int(getattr(model, 'n_actions', 12))} "
            f"({'full action space' if _imagine_n_actions >= int(getattr(model, 'n_actions', 12)) else 'legacy: Stay+moves only'})",
            flush=True,
        )
    from jax_sim import observations_jax as _obs


    @jax.jit
    def sim_step(carry, scan_input):
        """
        carry = (grid, blue_pop, red_pop, blue_carries, red_carries, b_params, r_params)
        scan_input = (step_key, step_idx)
        Returns: new_carry, rollout_data
        """
        step_key, step_idx = scan_input
        grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params = carry
        b_params_sg = jax.tree.map(jax.lax.stop_gradient, b_params)
        r_params_sg = jax.tree.map(jax.lax.stop_gradient, r_params)
        key_obs, key_act, key_b_obs, key_r_obs, key_misc = jax.random.split(step_key, 5)

        # ── Build presence maps ─────────────────────────────────
        blue_map = jnp.zeros((gs, gs), dtype=jnp.bool_)
        blue_map = blue_map.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].set(b_pop.alive & ~b_pop.is_big_green)
        
        blue_bg_map = jnp.zeros((gs, gs), dtype=jnp.bool_)
        blue_bg_map = blue_bg_map.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].set(b_pop.alive & b_pop.is_big_green)
        
        red_map = jnp.zeros((gs, gs), dtype=jnp.bool_)
        red_map = red_map.at[r_pop.positions[:, 0], r_pop.positions[:, 1]].set(r_pop.alive)

        # ── Observations (via observations_jax — reload with train_entry) ──
        b_obs = _obs.build_observations_jax(
            b_pop, grid, blue_map, red_map, config, 0,
            key=key_b_obs, limit_red_sensing=True, blue_bg_map=blue_bg_map,
        )
        r_obs = _obs.build_observations_jax(
            r_pop, grid, blue_map, red_map, config, 0, key=key_r_obs,
            blue_bg_map=blue_bg_map,
        )

        # ── Forward passes ──────────────────────────────────────
        noise_key, key_misc = jax.random.split(key_misc)
        alarm_key, key_misc = jax.random.split(key_misc)
        b_new_c, b_outs = model_apply(b_params_sg, b_carries, b_obs, _n_layers)
        r_new_c, r_outs = _r_apply(r_params_sg, r_carries, r_obs, _n_layers, rngs={'dropout': noise_key})

        b_action_logits = b_outs.action_logits
        b_signal_out = b_outs.signal_out
        b_sym_w = b_outs.symbol_write
        b_vals = b_outs.values
        b_tom = b_outs.tom_logits
        b_token_ids = b_outs.token_ids
        b_alarm_logits = b_outs.alarm_out
        b_loss_vq = b_outs.loss_vq
        b_z_e = b_outs.z_e
        b_cult_f = b_outs.culture_fast
        b_cult_s = b_outs.culture_slow

        r_action_logits = r_outs.action_logits
        r_signal_out = r_outs.signal_out
        r_sym_w = r_outs.symbol_write
        r_vals = r_outs.values
        r_tom = r_outs.tom_logits
        r_token_ids = r_outs.token_ids
        r_loss_vq = r_outs.loss_vq
        r_z_e = r_outs.z_e
        r_cult_f = r_outs.culture_fast
        r_cult_s = r_outs.culture_slow

        # Sample alarm action from logits
        b_alarm_keys = jax.random.split(alarm_key, b_pop.max_pop)
        b_alarm_action = jax.vmap(jax.random.categorical)(b_alarm_keys, b_alarm_logits)
        b_alarm_out = jax.nn.one_hot(b_alarm_action, 2, dtype=jnp.float32)
        
        # Phase 18: Continuous-to-Discrete (CtD) bootstrap (100k-step decay ramp)
        ctd_decay_steps = 100000.0
        # step_idx goes from 0 upwards; alpha is 1.0 (continuous) -> 0.0 (discrete)
        alpha = jnp.maximum(0.0, 1.0 - (step_idx / ctd_decay_steps))
        b_sig_broadcast = (alpha * b_z_e) + ((1.0 - alpha) * b_signal_out)

        # Phase 18: Tier-3 Causal Gates (Ablation)
        ablate_slot0 = config.get("ablate_slot0", False)
        ablate_slot1 = config.get("ablate_slot1", False)
        
        from jax_sim.obs_layout import SIGNAL_SLOTS
        if ablate_slot0:
            b_sig_broadcast = b_sig_broadcast.at[:, SIGNAL_SLOTS['slot_0']].set(0.0)
        if ablate_slot1:
            b_sig_broadcast = b_sig_broadcast.at[:, SIGNAL_SLOTS['slot_1']].set(0.0)

        # ── Write VQ signals for neighbours ──
        if _vqel_monologue:
            b_sig_broadcast = jnp.zeros_like(b_signal_out)
        else:
            b_sig_broadcast = b_signal_out
            
        b_pop = b_pop.replace(
            signals=jnp.where(b_pop.alive[:, None], b_sig_broadcast, 0.0),
            alarms=jnp.where(b_pop.alive[:, None], b_alarm_out, 0.0)
        )
        r_pop = r_pop.replace(
            signals=jnp.where(r_pop.alive[:, None], r_signal_out, 0.0),
            alarms=jnp.where(r_pop.alive[:, None], jnp.zeros((r_pop.max_pop, 2)), 0.0)
        )

        # ── Sample actions (Phase 11.3 epistemic gate on blues) ──
        # Logit-mask disabled actions (Push/Guard: no dispatch in grid_jax.py
        # for either team; Build: disabled for both teams per the Sep 2026
        # restart ruling) identically for both teams, matching imagination_jax.py
        # and rl_jax.py's PPO backward pass exactly — a mismatch at any one of
        # these three sites corrupts either PPO's log-prob ratio (rollout vs
        # backward) or the epistemic gate's imagined-action choice. See
        # jax_sim/action_space.py.
        b_action_logits = mask_disabled_actions(b_action_logits, axis=-1)
        r_action_logits = mask_disabled_actions(r_action_logits, axis=-1)

        key_act_b, key_act_r = jax.random.split(key_act)
        b_action_keys = jax.random.split(key_act_b, b_pop.max_pop)
        r_action_keys = jax.random.split(key_act_r, r_pop.max_pop)
        b_actions_reactive = jax.vmap(jax.random.categorical)(b_action_keys, b_action_logits)
        r_actions = jax.vmap(jax.random.categorical)(r_action_keys, r_action_logits)

        if _imagine_fn is not None:
            action_oh_table = jnp.eye(model.n_actions, dtype=b_carries.dtype)
            b_action_oh_reactive = action_oh_table[b_actions_reactive]
            b_conf_pred = model.apply(
                params_apply_variables(b_params_sg),
                b_carries,
                b_action_oh_reactive,
                method=model.predict_carry_fwd_confidence,
            )
            b_conf_pred = jax.lax.stop_gradient(b_conf_pred)
            b_a_imagined, b_im_gain, _b_im_agree_greedy = _imagine_fn(
                b_params_sg, b_carries, b_action_logits, b_pop.alive
            )
            b_alive_f = b_pop.alive.astype(jnp.float32)
            mean_conf = jnp.sum(b_conf_pred * b_alive_f) / (jnp.sum(b_alive_f) + 1e-8)
            dynamic_tau = mean_conf * _conf_multiplier
            b_gate_imagine = b_conf_pred < dynamic_tau
            b_actions = jnp.where(b_gate_imagine, b_a_imagined, b_actions_reactive)
            b_imagination_agree = (
                (b_a_imagined == b_actions_reactive).astype(jnp.float32) * b_alive_f
            )
            b_conf_gate_frac = b_gate_imagine.astype(jnp.float32) * b_alive_f
            b_used_imagination = b_gate_imagine
        else:
            b_actions = b_actions_reactive
            b_a_imagined = jnp.zeros((b_pop.max_pop,), dtype=jnp.int32)
            b_im_gain = jnp.zeros((b_pop.max_pop,), dtype=jnp.float32)
            b_imagination_agree = jnp.zeros((b_pop.max_pop,), dtype=jnp.float32)
            b_conf_gate_frac = jnp.zeros((b_pop.max_pop,), dtype=jnp.float32)
            b_used_imagination = jnp.zeros((b_pop.max_pop,), dtype=jnp.bool_)

        # ── Update episodic memory buffer ───────────────────────
        b_nb_flat = b_obs[:, 6 : 6 + K * sig_d]
        b_mean_nb_sig = b_nb_flat.reshape(b_pop.max_pop, K, sig_d).mean(axis=1)
        b_pop = update_memory_buffer(b_pop, b_mean_nb_sig, b_actions, b_pop.alive)

        r_nb_flat = r_obs[:, 6 : 6 + K * sig_d]
        r_mean_nb_sig = r_nb_flat.reshape(r_pop.max_pop, K, sig_d).mean(axis=1)
        r_pop = update_memory_buffer(r_pop, r_mean_nb_sig, r_actions, r_pop.alive)

        # ── Movement ────────────────────────────────────────────
        # Captured for the CtD red shaping term (jax_sim/ctd_ramp.py) -- the
        # "before" state of the potential function, prior to this step's move.
        _b_pos_before_move = b_pop.positions
        _r_pos_before_move = r_pop.positions

        b_new_pos, _ = apply_moves(b_pop.positions, b_actions, b_pop.alive, gs, grid.walls, grid.barrier_hp_map, False)
        r_new_pos, r_intended_pos = apply_moves(r_pop.positions, r_actions, r_pop.alive, gs, grid.walls, grid.barrier_hp_map, True)

        b_moved = (b_new_pos != b_pop.positions).any(axis=-1) & b_pop.alive
        r_moved = (r_new_pos != r_pop.positions).any(axis=-1) & r_pop.alive
        b_pop = b_pop.replace(positions=b_new_pos)
        r_pop = r_pop.replace(positions=r_new_pos)

        # ── Phase 16.5: Constructible Obstacles ─────────────────
        _p16_5 = config.get("phase16_5_enrichment", {})
        _barrier_build_cost = float(_p16_5.get("barrier_build_cost", 0.06))
        b_building = (b_actions == 8) & b_pop.alive
        barrier_addition = b_building.astype(jnp.float32) * 3.0
        
        r_blocked = (grid.barrier_hp_map[r_intended_pos[:, 0], r_intended_pos[:, 1]] > 0.5) & r_pop.alive
        barrier_damage = r_blocked.astype(jnp.float32)
        
        new_barrier_hp = grid.barrier_hp_map.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].add(barrier_addition)
        new_barrier_hp = new_barrier_hp.at[r_intended_pos[:, 0], r_intended_pos[:, 1]].add(-barrier_damage)
        
        _barrier_decay_rate = float(_p16_5.get("barrier_decay_rate", 0.05))
        new_barrier_hp = decay_barrier_grid(new_barrier_hp, _barrier_decay_rate)
        grid = grid.replace(barrier_hp_map=new_barrier_hp)
        
        # Apply Barrier Build Cost
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy - (b_building.astype(jnp.float32) * _barrier_build_cost), 0.0, 1.0))

        # ── Phase 17.5: Timescale Grammar Metabolic Cost ─────────────────
        # b_token_ids now represents the VQ code. We need to derive the alarm from b_alarm_out.
        # b_alarm_out is shape (N, 2), one-hot encoded (0 = Safe, 1 = Alarm).
        alarm_triggered = (b_alarm_out[:, 1] > 0.5) & b_pop.alive
        _alarm_metabolic_cost = float(config.get("alarm_metabolic_cost", 0.006))
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy - (alarm_triggered.astype(jnp.float32) * _alarm_metabolic_cost), 0.0, 1.0))

        # ── Scent trails (reds deposit scent) ───────────────────
        new_scent = update_scent_trails(
            grid.scent_trails, r_pop.positions, r_pop.alive,
            intensity=_scent_intensity, decay_steps=_scent_decay_steps,
        )
        grid = grid.replace(scent_trails=new_scent)

        # ── Resource consumption ────────────────────────────────
        b_energy_gain, new_res = consume_resources(
            b_pop.positions, b_pop.alive, grid.resources,
            decay=float(config.get("resource_decay", 0.05))
        )
        grid = grid.replace(resources=new_res)
        
        # ── Phase 18: Crafting Mechanics ────────────────────────
        # 1. USE_TOOL (Action 11) -> multiplier
        is_use_tool = (b_actions == 11) & b_pop.alive & b_pop.inventory_axe
        b_energy_gain = jnp.where(is_use_tool, b_energy_gain * 2.0, b_energy_gain)

        # 2. PICK_UP (Action 9) -> Capacity=1 enforcement
        is_pickup = (b_actions == 9) & b_pop.alive
        currently_empty = (b_pop.inventory_wood == 0) & (b_pop.inventory_stone == 0) & (b_pop.inventory_flint == 0) & (b_pop.inventory_clay == 0) & (b_pop.inventory_vine == 0)
        
        on_wood = grid.wood_grid[b_pop.positions[:, 0], b_pop.positions[:, 1]] > 0
        on_stone = grid.stone_grid[b_pop.positions[:, 0], b_pop.positions[:, 1]] > 0
        on_flint = grid.flint_grid[b_pop.positions[:, 0], b_pop.positions[:, 1]] > 0
        on_clay = grid.clay_grid[b_pop.positions[:, 0], b_pop.positions[:, 1]] > 0
        on_vine = grid.vine_grid[b_pop.positions[:, 0], b_pop.positions[:, 1]] > 0
        
        pickup_wood = is_pickup & on_wood & currently_empty
        pickup_stone = is_pickup & on_stone & currently_empty & ~pickup_wood
        pickup_flint = is_pickup & on_flint & currently_empty & ~pickup_wood & ~pickup_stone
        pickup_clay = is_pickup & on_clay & currently_empty & ~pickup_wood & ~pickup_stone & ~pickup_flint
        pickup_vine = is_pickup & on_vine & currently_empty & ~pickup_wood & ~pickup_stone & ~pickup_flint & ~pickup_clay
        
        new_inv_wood = b_pop.inventory_wood + pickup_wood.astype(jnp.int32)
        new_inv_stone = b_pop.inventory_stone + pickup_stone.astype(jnp.int32)
        new_inv_flint = b_pop.inventory_flint + pickup_flint.astype(jnp.int32)
        new_inv_clay = b_pop.inventory_clay + pickup_clay.astype(jnp.int32)
        new_inv_vine = b_pop.inventory_vine + pickup_vine.astype(jnp.int32)
        
        new_wood_grid = grid.wood_grid.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].add(-pickup_wood.astype(jnp.float32))
        new_stone_grid = grid.stone_grid.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].add(-pickup_stone.astype(jnp.float32))
        new_flint_grid = grid.flint_grid.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].add(-pickup_flint.astype(jnp.float32))
        new_clay_grid = grid.clay_grid.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].add(-pickup_clay.astype(jnp.float32))
        new_vine_grid = grid.vine_grid.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].add(-pickup_vine.astype(jnp.float32))
        
        grid = grid.replace(
            wood_grid=jnp.maximum(new_wood_grid, 0.0),
            stone_grid=jnp.maximum(new_stone_grid, 0.0),
            flint_grid=jnp.maximum(new_flint_grid, 0.0),
            clay_grid=jnp.maximum(new_clay_grid, 0.0),
            vine_grid=jnp.maximum(new_vine_grid, 0.0)
        )
        
        # 3. CRAFT (Action 10) -> shared reward
        is_craft = (b_actions == 10) & b_pop.alive
        
        _craft_result = resolve_crafting(
            b_pop.positions, is_craft,
            new_inv_wood, new_inv_stone, new_inv_flint, new_inv_clay, new_inv_vine,
            grid.current_recipe, gs,
        )
        craft_success = _craft_result["craft_success"]
        futile_uncoordinated = _craft_result["futile_uncoordinated"]
        futile_wrong_mats = _craft_result["futile_wrong_mats"]
        futile_empty = _craft_result["futile_empty"]

        consume_wood = craft_success & (new_inv_wood > 0)
        consume_stone = craft_success & (new_inv_stone > 0)
        consume_flint = craft_success & (new_inv_flint > 0)
        consume_clay = craft_success & (new_inv_clay > 0)
        consume_vine = craft_success & (new_inv_vine > 0)
        
        new_inv_wood = new_inv_wood - consume_wood.astype(jnp.int32)
        new_inv_stone = new_inv_stone - consume_stone.astype(jnp.int32)
        new_inv_flint = new_inv_flint - consume_flint.astype(jnp.int32)
        new_inv_clay = new_inv_clay - consume_clay.astype(jnp.int32)
        new_inv_vine = new_inv_vine - consume_vine.astype(jnp.int32)
        
        new_inv_axe = b_pop.inventory_axe | craft_success
        
        b_pop = b_pop.replace(
            energy=jnp.clip(b_pop.energy + b_energy_gain, 0.0, 1.0),
            inventory_wood=new_inv_wood,
            inventory_stone=new_inv_stone,
            inventory_flint=new_inv_flint,
            inventory_clay=new_inv_clay,
            inventory_vine=new_inv_vine,
            inventory_axe=new_inv_axe
        )

        # ── Contested resource bonus (requires 2+ agents) ──────
        agent_count = jnp.zeros((gs, gs), dtype=jnp.float32)
        agent_count = agent_count.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].add(
            b_pop.alive.astype(jnp.float32)
        )
        contested_bonus_map = (agent_count >= _contested_min_harv) & (grid.contested_res > 0)
        contested_at_agent = (
            contested_bonus_map[b_pop.positions[:, 0], b_pop.positions[:, 1]]
            & b_pop.alive
        )
        contested_gain = jnp.where(contested_at_agent, 0.1, 0.0)
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy + contested_gain, 0.0, 1.0))

        # ── Resource & Material respawning ──────────────────────
        res_key, wood_key, stone_key, flint_key, clay_key, vine_key = jax.random.split(
            jax.random.split(key_misc)[0], 6
        )
        regen_rate = float(config.get("resource_regen_rate", 0.005))
        spawn_mask = jax.random.bernoulli(res_key, regen_rate, (gs, gs))
        new_res = grid.resources + spawn_mask.astype(jnp.float32) * _resource_spawn_boost

        # Spatial material respawn -- see material_zone_masks() for why banding
        # (not uniform spawn) is the point, not an incidental choice.
        _zones = material_zone_masks(gs)

        wood_spawn = jax.random.bernoulli(wood_key, regen_rate * 0.5, (gs, gs)) & _zones["wood"]
        stone_spawn = jax.random.bernoulli(stone_key, regen_rate * 0.5, (gs, gs)) & _zones["stone"]
        flint_spawn = jax.random.bernoulli(flint_key, regen_rate * 0.5, (gs, gs)) & _zones["flint"]
        clay_spawn = jax.random.bernoulli(clay_key, regen_rate * 0.5, (gs, gs)) & _zones["clay"]
        vine_spawn = jax.random.bernoulli(vine_key, regen_rate * 0.5, (gs, gs)) & _zones["vine"]

        grid = grid.replace(
            resources=jnp.clip(new_res, 0.0, _resource_max),
            wood_grid=jnp.clip(grid.wood_grid + wood_spawn.astype(jnp.float32), 0.0, 1.0),
            stone_grid=jnp.clip(grid.stone_grid + stone_spawn.astype(jnp.float32), 0.0, 1.0),
            flint_grid=jnp.clip(grid.flint_grid + flint_spawn.astype(jnp.float32), 0.0, 1.0),
            clay_grid=jnp.clip(grid.clay_grid + clay_spawn.astype(jnp.float32), 0.0, 1.0),
            vine_grid=jnp.clip(grid.vine_grid + vine_spawn.astype(jnp.float32), 0.0, 1.0)
        )

        # ── Age ─────────────────────────────────────────────────
        b_pop = b_pop.replace(ages=b_pop.ages + 1)
        r_pop = r_pop.replace(ages=r_pop.ages + 1)

        # ── Max age death ───────────────────────────────────────
        b_old = b_pop.alive & (b_pop.ages >= _max_age)
        r_old = r_pop.alive & (r_pop.ages >= _max_age)
        b_pop = kill_agents(b_pop, b_old)
        r_pop = kill_agents(r_pop, r_old)

        # ── Reproduction ────────────────────────────────────────
        repro_key, step_key2 = jax.random.split(step_key)
        b_pop = apply_auto_reproduce(
            b_pop, repro_key, gs,
            min_pop=_min_pop_blue,
            energy_thresh=_repro_energy_thresh,
            energy_cost=_repro_energy_cost,
        )

        # ── Mind-Melding ─────────────────────────────────────────
        if _mind_meld:
            b_pop = apply_mind_meld(
                b_pop, gs, radius=_mm_radius, rate=_mm_rate, direction=_mm_dir,
            )

        # ── Catch detection (optional predator jitter via red_catch_prob) ──
        catch_rng = jax.random.split(key_misc)[1]
        # Captured for the CtD red shaping term -- alive state as seen by
        # apply_catches, before this step's catches are resolved.
        _b_alive_before_catch = b_pop.alive
        b_new_alive, caught_b, r_caught_small, r_caught_big_coop, r_caught_big_solo, r_mauled, b_catch_pen, catch_attempted = apply_catches(
            b_pop.positions, b_pop.alive, b_pop.is_big_green,
            r_pop.positions, r_pop.alive, r_actions,
            gs, step_idx, _coop_threshold_step,
            catch_radius=_red_catch_radius,
            catch_prob=_red_catch_prob,
            rng=catch_rng,
        )
        # Shelter protection: blues on shelter spots can't be caught
        on_shelter = grid.shelter_spots[b_pop.positions[:, 0], b_pop.positions[:, 1]]
        sheltered_catch = caught_b & on_shelter
        b_new_alive = b_new_alive | sheltered_catch
        b_catch_pen = jnp.where(sheltered_catch, 0.0, b_catch_pen)

        b_pop = b_pop.replace(alive=b_new_alive)
        b_pop = b_pop.replace(
            carries=jnp.where(~b_new_alive[:, None], 0.0, b_pop.carries)
        )

        # ── Red starvation tracking ─────────────────────────────
        r_caught_any = (r_caught_small > 0) | (r_caught_big_coop > 0) | (r_caught_big_solo > 0)

        # CtD red shaping ramp (Cam's sign-off, 2026-09-14): potential-based
        # reward shaping on distance to the nearest living blue, active only
        # while grid.red_ramp_active. Zeroed on catch steps (jax_sim/ctd_ramp.py's
        # deliberate, logged exception -- Phi jumps discontinuously when a catch
        # removes the nearest blue, which would otherwise claw back part of the
        # catch reward it's meant to lead into).
        _red_shaping_raw = red_shaping_term(
            _r_pos_before_move, r_new_pos,
            _b_pos_before_move, _b_alive_before_catch,
            b_new_pos, b_new_alive,
            r_caught_any,
            _red_ramp_beta, _ppo_gamma_for_shaping, gs,
        )
        red_shaping = jnp.where(grid.red_ramp_active, _red_shaping_raw, 0.0)
        new_steps_since = jnp.where(r_caught_any, 0, r_pop.steps_since_catch + 1)
        new_steps_since = jnp.where(r_pop.alive, new_steps_since, 0)
        r_pop = r_pop.replace(steps_since_catch=new_steps_since)
        r_starved_hunt = r_pop.alive & (r_pop.steps_since_catch >= _red_starvation_steps)
        r_pop = kill_agents(r_pop, r_starved_hunt)

        # ── Phase 18.7 Recipe Rotation & Visibility ────────────────────────
        k_rec1, k_rec2, k_vis = jax.random.split(jax.random.split(key_misc)[2], 3)
        new_recipe_timer = grid.recipe_timer[0] - 1
        
        def update_recipe(_):
            items = jax.random.randint(k_rec1, (4,), 0, 5)
            mask = jax.random.bernoulli(k_rec2, 0.5)
            items = items.at[3].set(jnp.where(mask, items[3], 5))
            full_counts = jnp.bincount(items, length=6)[:5].astype(jnp.int32)

            # CtD crafting ramp (Cam's sign-off, 2026-09-14, corrected same
            # day): while active, recipes are solo-satisfiable (stage 0, 1
            # unit) then pair-satisfiable (stage 1, 2 units) instead of up to
            # 4 units across 5 materials -- see jax_sim/ctd_ramp.py.
            k_rec_ramp = jax.random.split(k_rec1)[0]
            ramp_counts = staged_recipe_counts(k_rec_ramp, grid.craft_ramp_stage)
            counts = jnp.where(grid.craft_ramp_active, ramp_counts, full_counts)

            progress = jnp.clip(step_idx / 1500000.0, 0.0, 1.0)
            vis_prob = 0.5 - 0.3 * progress
            new_vis = jax.random.bernoulli(k_vis, vis_prob, (b_pop.max_pop,))

            return counts, jnp.array([1000], dtype=jnp.int32), new_vis
            
        def keep_recipe(_):
            return grid.current_recipe, jnp.array([new_recipe_timer], dtype=jnp.int32), b_pop.can_see_recipe
            
        new_recipe, recipe_timer, new_can_see = jax.lax.cond(
            new_recipe_timer <= 0,
            update_recipe,
            keep_recipe,
            operand=None
        )
        
        grid = grid.replace(current_recipe=new_recipe, recipe_timer=recipe_timer)
        b_pop = b_pop.replace(can_see_recipe=new_can_see)

        # ── Puzzle logic ────────────────────────────────────────
        p_act, p_cool = decay_puzzle_timeout(grid.puzzle_active, grid.puzzle_cooldown)
        p_rew, p_solved, p_act, p_cool = check_puzzle_solved(
            b_pop.positions, b_pop.alive, grid.puzzle_nodes, p_act, p_cool, gs
        )
        p_grid = update_puzzle_grid(gs, grid.puzzle_nodes, p_act)
        grid = grid.replace(puzzle_active=p_act, puzzle_cooldown=p_cool, puzzle_grid=p_grid)
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy + p_rew * 0.5, 0.0, 1.0))

        # ── Phase 13.0 metabolic cognition tax (after gains/catches, before decay) ──
        cog_cost = (
            b_used_imagination.astype(jnp.float32)
            * b_pop.alive.astype(jnp.float32)
            * (_imagination_metabolic_delta * _imagination_k)
        )
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy - cog_cost, 0.0, 1.0))

        # ── Energy decay (Phase 14.2: asymmetric — blues fast, reds apex) ──
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy - _energy_decay, 0.0, 1.0))
        r_pop = r_pop.replace(energy=jnp.clip(r_pop.energy - _red_energy_decay, 0.0, 1.0))

        # ── Phase 16.5: Ignition Tracking (Metabolic Routing) ───
        _feral_threshold = float(config.get("phase16_5_enrichment", {}).get("feral_threshold", 0.20))
        pre_step_energy = b_obs[:, 2]
        post_step_energy = b_pop.energy
        ignition = (pre_step_energy < _feral_threshold) & (post_step_energy >= _feral_threshold)
        
        _ignition_discount = float(config.get("phase16_5_enrichment", {}).get("ignition_discount", 0.1))
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy - (ignition.astype(jnp.float32) * _ignition_discount), 0.0, 1.0))

        # ── Starvation (after metabolic tax + decay) ────────────
        b_starved = b_pop.alive & (b_pop.energy < _starv_thresh)
        r_starved = r_pop.alive & (r_pop.energy < _starv_thresh)
        b_pop = kill_agents(b_pop, b_starved)
        r_pop = kill_agents(r_pop, r_starved)

        # ── Update dropout tracker ──────────────
        new_steps = jnp.where(b_pop.alive, b_pop.steps_since_dropout + 1, 0)
        b_pop = b_pop.replace(steps_since_dropout=new_steps)

        # ── Rewards (all from config) ───────────────────────────
        b_rew = jnp.where(b_pop.alive, _reward_blue_alive, 0.0)
        b_rew = b_rew + 0.02 * b_pop.energy
        b_rew = b_rew + b_catch_pen * jnp.abs(_reward_blue_caught)
        b_rew = b_rew + _reward_resource * b_energy_gain
        b_rew = b_rew + _reward_move * b_moved.astype(jnp.float32)
        b_rew = b_rew + _puzzle_reward * p_rew
        b_rew = b_rew + contested_gain * 0.5
        
        # ── Phase 17.5 Direct Alarm Penalty ─────────────────────
        _alarm_penalty_coef = float(config.get("alarm_penalty_coef", 0.02))
        alarm_fired = b_alarm_out.argmax(-1).astype(jnp.float32)  # 1 if alarmed, 0 if silent
        b_rew = b_rew - _alarm_penalty_coef * alarm_fired

        # ── Phase 18 Futile Action Penalty ──────────────────────
        futile_craft = (b_actions == 10) & b_pop.alive & ~craft_success
        futile_use = (b_actions == 11) & b_pop.alive & (b_pop.inventory_axe == 0)
        b_rew = b_rew + _reward_futile_craft * (futile_craft | futile_use).astype(jnp.float32)
        b_rew = b_rew + _reward_craft_success * craft_success.astype(jnp.float32)

        r_rew = _rew_small_blue * r_caught_small
        r_rew = r_rew + (_rew_big_green_coop + _rew_coord) * r_caught_big_coop
        r_rew = r_rew + _rew_big_green_solo_catch * r_caught_big_solo
        r_rew = r_rew + _rew_big_green_solo_pen * r_mauled
        # reward_red_catch (config.yaml, "P4: stronger hunt incentive") was read into
        # _reward_red_catch and never applied anywhere -- confirmed dead by exhaustive
        # grep. Wired in additively alongside the phase16_combinatorial_syntax terms
        # above, not replacing them, per Cam's ruling (Rule 12 doesn't gate making a
        # configured-but-unwired mechanism actually run).
        r_rew = r_rew + _reward_red_catch * r_caught_any.astype(jnp.float32)
        r_rew = r_rew + red_shaping
        r_rew = r_rew + jnp.where(r_pop.alive, _reward_red_starve, 0.0)
        r_rew = r_rew + _reward_red_move * r_moved.astype(jnp.float32)

        # ── Write symbols / culture ─────────────────────────────
        grid = grid.replace(
            symbols=write_to_grid(grid.symbols, b_pop.positions, b_sym_w, b_pop.alive),
            cultural_fast=write_to_grid(
                grid.cultural_fast, b_pop.positions, b_cult_f, b_pop.alive,
                intensity=0.3,
            ),
            cultural_slow=write_to_grid(
                grid.cultural_slow, b_pop.positions, b_cult_s, b_pop.alive,
                intensity=0.3,
            ),
        )

        # ── Decay ───────────────────────────────────────────────
        grid = grid.replace(
            symbols=decay_grid(grid.symbols, config["symbol_decay"]),
            cultural_fast=decay_grid(grid.cultural_fast, config["culture_fast_decay"]),
            cultural_slow=decay_grid(grid.cultural_slow, config["culture_slow_decay"]),
        )

        # ── Log probs for PPO ─────────────────────────────────
        b_log_probs = jax.nn.log_softmax(b_action_logits, axis=-1)
        b_log_probs_taken = jnp.take_along_axis(b_log_probs, b_actions[:, None], axis=-1).squeeze(-1)

        b_alarm_log_probs = jax.nn.log_softmax(b_alarm_logits, axis=-1)
        b_alarm_log_probs_taken = jnp.take_along_axis(b_alarm_log_probs, b_alarm_action[:, None], axis=-1).squeeze(-1)
        b_joint_log_probs_taken = b_log_probs_taken + b_alarm_log_probs_taken

        r_log_probs = jax.nn.log_softmax(r_action_logits, axis=-1)
        r_log_probs_taken = jnp.take_along_axis(r_log_probs, r_actions[:, None], axis=-1).squeeze(-1)

        # ── Rollout data ──────────────────────────────────────
        b_done = (~b_pop.alive).astype(jnp.float32)
        r_done = (~r_pop.alive).astype(jnp.float32)

        b_rollout = {
            "obs": b_obs, "actions": b_actions, "alarm_actions": b_alarm_action, "action_logits": b_action_logits, "log_probs": b_joint_log_probs_taken,
            "values": b_vals, "rewards": b_rew, "dones": b_done,
            "carries": b_carries,
            "loss_vq": b_loss_vq,
            "z_e": b_z_e,
            "token_ids": b_token_ids,
            "alarm_out": b_alarm_out,
            "positions": b_pop.positions,
            "signals": b_pop.signals,
            "energy": b_pop.energy,
            "alive": b_pop.alive,
            "blue_caught": caught_b.astype(jnp.float32),
            "catch_attempted": catch_attempted.astype(jnp.float32),
            "imagined_action": b_a_imagined,
            "imagination_gain": b_im_gain,
            "imagination_agree": b_imagination_agree,
            "conf_gate_imagine_frac": b_conf_gate_frac,
            "imagination_metabolic_cost": cog_cost,
            "ignition": ignition,
            "barrier_sum": jnp.sum(grid.barrier_hp_map),
            "craft_success": craft_success.astype(jnp.float32),
            "futile_craft": futile_craft.astype(jnp.float32),
            "futile_uncoordinated": futile_uncoordinated.astype(jnp.float32),
            "futile_wrong_mats": futile_wrong_mats.astype(jnp.float32),
            "futile_empty": futile_empty.astype(jnp.float32),
            "current_recipe": grid.current_recipe,
            "steps_since_dropout": b_pop.steps_since_dropout,
            "craft_ramp_active": grid.craft_ramp_active,
            "red_ramp_active": grid.red_ramp_active,
        }
        r_rollout = {
            "obs": r_obs, "actions": r_actions, "log_probs": r_log_probs_taken,
            "values": r_vals, "rewards": r_rew, "dones": r_done,
            "carries": r_carries,
            "loss_vq": r_loss_vq,
            "z_e": r_z_e,
            "token_ids": r_token_ids,
            "positions": r_pop.positions,
            "signals": r_pop.signals,
            "energy": r_pop.energy,
            "alive": r_pop.alive,
            "craft_ramp_active": grid.craft_ramp_active,
            "red_ramp_active": grid.red_ramp_active,
            "red_shaping": red_shaping,
        }

        new_carry = (grid, b_pop, r_pop, b_new_c, r_new_c, b_params, r_params)
        return new_carry, {"blue": b_rollout, "red": r_rollout}

    return sim_step


# ── Main entry point ───────────────────────────────────────────────────────

def run_simulation(
    config: Dict,
    seed: int = 42,
    n_steps: int = 100000,
    on_checkpoint_saved: Any = None,
) -> Tuple[Dict, Dict]:
    """Prefer ``from jax_sim.train_entry import run_simulation`` after git pull."""
    from jax_sim.train_entry import run_simulation as _fresh_run

    return _fresh_run(config, seed=seed, n_steps=n_steps, on_checkpoint_saved=on_checkpoint_saved)


def _print_grafted_vq_counter_stats(src_agent: dict, injected_paths: list, agent_type: str) -> None:
    """Cam, 2026-09-14: "prove what's in there" for any usage_ema/dead_streak
    grafted fresh onto a resumed checkpoint -- printed as "randomly
    initialized" by the caller, but never actually measured. Both fields are
    zero-initialized by ensure_vq_usage_state() when built via
    init_agent_params(), and graft_missing_param_subtrees() copies that
    template value verbatim when the key is missing from the checkpoint -- so
    this should always read exactly 0/0/0, but "should" is not "measured."
    """
    for path in injected_paths:
        clean_path = path.split(" (")[0]  # strip "(zero-padded ...)"/"(shape mismatch ...)" suffixes
        if not (clean_path.endswith("/usage_ema") or clean_path.endswith("/dead_streak")):
            continue
        node = src_agent
        try:
            for part in clean_path.split("/"):
                node = node[part]
        except (KeyError, TypeError):
            continue
        arr = np.asarray(node)
        print(
            f"[JAX] Grafted {clean_path} ({agent_type}): "
            f"min={float(arr.min()):.4f} max={float(arr.max()):.4f} "
            f"mean={float(arr.mean()):.4f} shape={arr.shape}",
            flush=True,
        )


def _pad_comms_history(history: list) -> list:
    """Pad the comms-freeze tripwire's per-update codes_active history to a
    fixed (40, 3) shape for checkpointing -- 40 is the same hard-cap as the
    stage-0 escape and the forced-C0-capture deadline. Unfilled rows get a
    (-1, -1, -1) sentinel."""
    return [list(history[i]) if i < len(history) else [-1, -1, -1] for i in range(40)]


def _run_simulation_impl(
    config: Dict,
    seed: int = 42,
    n_steps: int = 100000,
    on_checkpoint_saved: Any = None,
) -> Tuple[Dict, Dict]:
    """
    Run full JAX simulation.
    Returns: (final_params, metrics_history)
    """
    config = _normalize_config(config)
    print(
        f"[JAX] Phase14.2 Metabolic Asymmetry: red_energy_decay={config['red_energy_decay']}",
        flush=True,
    )

    # Persistent JAX cache on a network volume (Modal) deserializes slowly and often
    # looks like a hang → spurious KeyboardInterrupt when the notebook times out.
    _jax_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if _jax_cache.startswith("/mnt"):
        _local_cache = "/tmp/throng_jax_cache"
        os.makedirs(_local_cache, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = _local_cache
        print(
            f"[JAX] Using local compilation cache {_local_cache} "
            f"(skipped volume path {_jax_cache})",
            flush=True,
        )

    run_name = config.get("run_name", "jax_run")
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    
    # ── Init corpus writers ─────────────────────────────────
    _p12_early = config.get("phase12_coevolution") or {}
    _corpus_frac = float(config.get("corpus_sample_frac", 0.08))
    _corpus_every = int(config.get("corpus_every_n_steps", 20))
    # 2026-09-20 (Cam): one id per process, shared by both writers below, so
    # every record this launch ever writes (blue or red) can be attributed
    # to it -- see SignalCorpusWriter's launch_id docstring for why this
    # exists (overlapping step ranges across launches into the same
    # append-only corpus, undetectable without a per-launch marker).
    _launch_id = uuid.uuid4().hex[:12]
    print(f"[JAX] launch_id={_launch_id} (tags every corpus record this process writes)", flush=True)
    corpus_writer = SignalCorpusWriter(
        path=f"runs/{run_name}/signal_corpus.jsonl",
        sample_frac=_corpus_frac,
        every_n_steps=_corpus_every,
        launch_id=_launch_id,
    )
    corpus_writer_red = None
    _red_corpus_enabled = (
        bool(_p12_early.get("red_corpus_enabled", False))
        and bool(_p12_early.get("red_comms_enabled", False))
    )
    _hunt_range = float(
        _p12_early.get(
            "hunt_scout_range",
            config.get("alarm_scout_range", 8),
        )
    )
    if _red_corpus_enabled:
        corpus_writer_red = SignalCorpusWriter(
            path=f"runs/{run_name}/signal_corpus_red.jsonl",
            sample_frac=_corpus_frac,
            every_n_steps=_corpus_every,
            launch_id=_launch_id,
        )
        print(
            f"[JAX] Red corpus: signal_corpus_red.jsonl "
            f"(hunter = blue_dist <= hunt_scout_range={_hunt_range})",
            flush=True,
        )

    # ── Init Checkpointing ──────────────────────────────────
    ckpt_dir = config.get("checkpoint_dir") or os.path.abspath(
        f"runs/{run_name}/checkpoints"
    )
    # 2026-09-14 (Cam): `str(Path(ckpt_dir).resolve())` used to sit here --
    # added to dodge an old Orbax mkdir-on-symlink failure, but on a Modal
    # Volume mount it silently resolves /mnt/throng-runs/checkpoints to the
    # internal backing path (/__modal/volumes/vo-.../checkpoints), which
    # bypasses the FUSE mount's write-tracking entirely: every checkpoint
    # save since this line was added has gone to a path Volume.commit()
    # never sees. Confirmed directly (Cam's one-measurement test, run live
    # against the resumed container): the resolved path and every other
    # view agreed on the same stale 3-checkpoint listing; a probe file
    # written through the UNRESOLVED mount path committed and became
    # visible externally on the first try. mkdir(parents=True,
    # exist_ok=True) on the unresolved path was also verified to raise no
    # symlink error on this Python/OS combination -- the historical failure
    # this call was added to avoid does not reproduce here, so there is
    # nothing left for .resolve() to buy us and a real cost to what it
    # breaks. os.path.abspath() gives the same "always absolute, no
    # trailing slash weirdness" guarantee `checkpoint_dir` already needs,
    # without following symlinks -- for the already-absolute production
    # value ("/mnt/throng-runs/checkpoints") it is a complete no-op.
    ckpt_dir = os.path.abspath(ckpt_dir)
    # The protected-backup guard still wants a resolved path (it compares
    # against a resolved ~/throng_backup) -- resolve a LOCAL copy just for
    # that comparison, so the guard still catches a checkpoint_dir that
    # resolves into the backup via a symlink, without changing what the
    # CheckpointManager itself writes through.
    assert_checkpoint_dir_is_not_protected_backup(str(Path(ckpt_dir).resolve()))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # 2026-09-15 (Cam): max_to_keep=2 was the actual mechanism behind both
    # of tonight's "durable checkpoint never appears" losses -- Orbax prunes
    # by STEP NUMBER, not save recency, so a directory holding a
    # higher-numbered fossil from an abandoned lineage silently garbage
    # collects every new checkpoint a lower-numbered resume saves, the
    # instant it saves it (see docs/THE-ECOLOGY-NEVER-RAN.md instance 6).
    # 10 at ~26MB/checkpoint is ~260MB -- nothing -- and removes the
    # single-fault-tolerance failure mode where one bad save plus one bad
    # predecessor loses everything. The startup guard just below is the
    # actual fix (refuses to run at all into a fossil-contaminated
    # directory); this is defense in depth, not a substitute for it.
    options = ocp.CheckpointManagerOptions(max_to_keep=10, create=True)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options=options)
    # Cam's one-measurement diagnostic, made permanent: the resolved write
    # path is not visible from the process's own print statements alone --
    # the 2026-09-14 durability bug (checkpoints silently writing to a path
    # Volume.commit() couldn't see) would have printed nothing wrong right
    # up until the volume was checked externally. Print what we actually
    # resolved to and prove a write there is visible on this same
    # filesystem view, every launch, not just when something is suspected.
    print(
        f"[CKPT-PATH] ckpt_dir={ckpt_dir!r} | os.path.exists={os.path.exists(ckpt_dir)} | "
        f"parent listing={sorted(os.listdir(ckpt_dir)) if os.path.exists(ckpt_dir) else 'N/A'}",
        flush=True,
    )

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)

    gs = config["grid_size"]
    max_pop = config["max_pop"]
    max_pop_red = config["max_pop_red"]
    hidden_d = config["hidden_dim"]
    n_layers = config["n_layers"]
    T = config["ppo_rollout_steps"]

    # ── Init grid ─────────────────────────────────────────────
    grid = GridState(gs, symbol_dim=config["symbol_dim"])
    _ramp_cfg_init = config.get("ctd_competence_ramp", {})
    grid = grid.replace(
        craft_ramp_active=jnp.array(bool(_ramp_cfg_init.get("craft_ramp_enabled", False)), dtype=jnp.bool_),
        red_ramp_active=jnp.array(bool(_ramp_cfg_init.get("red_ramp_enabled", False)), dtype=jnp.bool_),
    )
    wall_mask = jax.random.bernoulli(keys[0], config.get("wall_density", 0.08), (gs, gs))
    
    # Resource patches (structured hotspots, not uniform drizzle)
    k_res, k_shelter, k_contest, k_puzzle = jax.random.split(keys[9], 4)
    _resource_max_init = float(config.get("resource_max", 1.0))
    resources = jnp.clip(
        generate_resource_patches(
            k_res, gs,
            n_patches=int(config.get("resource_n_patches", 20)),
            patch_radius=5.0,
        ),
        0.0,
        _resource_max_init,
    )
    
    # Shelter spots (safe zones)
    shelter = generate_shelter_spots(
        k_shelter, gs,
        n_spots=int(config.get("shelter_n_spots", 5)),
        radius=2,
    )
    
    # Contested nodes (require cooperation to harvest)
    contested = generate_contested_nodes(
        k_contest, gs,
        n_nodes=int(config.get("contested_n_nodes", 3)),
        yield_mult=float(config.get("contested_yield_multiplier", 3.0)),
        radius=2,
    )
    
    # Puzzle init
    p_nodes, p_act, p_cool = generate_puzzle_nodes(k_puzzle, n_nodes=3, grid_size=gs)
    p_grid = update_puzzle_grid(gs, p_nodes, p_act)
    
    grid = grid.replace(
        walls=wall_mask,
        resources=resources,
        shelter_spots=shelter,
        contested_res=contested,
        puzzle_nodes=p_nodes,
        puzzle_active=p_act,
        puzzle_cooldown=p_cool,
        puzzle_grid=p_grid,
    )

    # ── Init populations ────────────────────────────────────
    b_pop = init_population(
        max_pop, hidden_d, config["signal_dim"], gs, team_id=0,
        key=keys[1], n_agents=max_pop, memory_slots=config.get("memory_slots", 0),
    )
    _red_stages = list(config.get("red_curriculum_stages", [80, 150, 200, 250]))
    _red_start_n = int(config.get("min_red_population", _red_stages[0]))
    _red_start_n = min(_red_start_n, max_pop_red)
    _p12 = config.get("phase12_coevolution") or {}
    _red_comms = bool(_p12.get("red_comms_enabled", False))
    red_hidden_d = int(config.get("red_hidden_dim", hidden_d // 2))
    r_pop_hidden = red_hidden_d if _red_comms else hidden_d

    r_pop = init_population(
        max_pop_red, r_pop_hidden, config["signal_dim"], gs, team_id=1,
        key=keys[2], n_agents=_red_start_n, memory_slots=config.get("memory_slots", 0),
    )

    # ── Init model ──────────────────────────────────────────
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
    _conf_enabled = bool(_p9.get("confidence_enabled", False))
    _conf_coef = float(_p9.get("confidence_coef", 0.05)) if _conf_enabled else 0.0
    _red_cross = bool(_p12.get("red_cross_attn_enabled", True)) if _red_comms else False
    _red_vocab = int(_p12.get("red_vocab_size", config.get("vocab_size", 64)))
    
    obs_dim = _layout.total_dim
    print(f"[JAX] obs_dim = {obs_dim}")
    
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
        cross_attn_enabled=_cross_attn,
        cross_attn_num_heads=_cross_heads,
        env_channels=int(config.get("env_channels", 15)),
        own_state_dim=int(config.get("own_state_dim", 22)),
        n_actions=int(config["n_actions"]),
        local_cells=(2 * config["local_obs_radius"] + 1)**2,
        neighbor_k=config["neighbor_k"],
    )
    model_apply = make_model_apply(model)
    vqel_monologue_apply = make_vqel_monologue_apply(model)
    _p14 = config.get("phase14_vqel") or {}
    _vqel_monologue = bool(_p14.get("monologue_enabled", False))
    _dialogue_signal_mode = str(_p14.get("dialogue_signal_mode", "ste")).lower()
    _vqel_recon_coef = float(_p14.get("recon_coef", 1.0))
    _vqel_hash_coef = float(_p14.get("hash_penalty_coef", 0.5))
    _vqel_vq_coef = float(_p14.get("vq_coef_monologue", 1.0))
    _vqel_lr = float(_p14.get("monologue_lr", 3.0e-5))
    _graduate_recon_mse = float(_p14.get("graduate_recon_mse", 0.02))
    _graduate_consecutive = int(_p14.get("graduate_consecutive_updates", 10))
    _vqel_grad_streak = 0
    model_red = None
    r_model_apply = None
    if _red_comms:
        _p14t = config.get("phase14_transcendental") or {}
        model_red = PredatorNetworkJax(
            hidden_dim=red_hidden_d,
            neighbor_k=config["neighbor_k"],
            local_obs_radius=config["local_obs_radius"],
            n_heads=config["n_heads"],
            n_layers=n_layers,
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
            n_actions=int(config["n_actions"]),
        )
        r_model_apply = make_model_apply(model_red)

    # Initialize model memory buffers
    _ppo_mb = int(config.get("ppo_minibatch_size", 512))
    if _ppo_mb > 768:
        print(
            f"[JAX] WARN: ppo_minibatch_size={_ppo_mb} is large for 500 agents — "
            "use 512 to avoid PPO backward OOM on A100"
        )

    # Verify the interpreter loaded this repo (not a stale clone / wrong cwd).
    import inspect as _inspect
    from jax_sim import rl_jax as _rl_jax_mod
    _main_path = _inspect.getfile(run_simulation)
    _repo_root = os.path.dirname(os.path.dirname(_main_path))
    try:
        import subprocess as _subprocess
        _git_sha = _subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_repo_root,
            stderr=_subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        _git_sha = "unknown"
    if hasattr(_rl_jax_mod, "auxiliary_update"):
        _phase9 = "ON (AuxLoss line on dashboard)"
    elif hasattr(_rl_jax_mod, "fwd_dynamics_update"):
        _phase9 = "PARTIAL (FwdDyn only — pull 8e09f6a+ for self-pred)"
    else:
        _phase9 = "OFF — git pull required"
    print(f"[JAX] code: {_main_path}")
    print(f"[JAX] git={_git_sha} | Phase9 auxiliary: {_phase9}")
    if _cross_attn:
        print(
            f"[JAX] Phase9.4 cross-attn receiver: heads={_cross_heads} "
            f"(Q=self+carry, KV=neighbor signals → 1 Other token)"
        )
    if hasattr(_rl_jax_mod, "auxiliary_update") and "carry_fwd_coef" in str(
        _inspect.getsource(_rl_jax_mod.auxiliary_update)
    ):
        print("[JAX] Phase11 carry_fwd: head_fwd_dyn_1/2 → carry_{t+1} MSE (stop_grad target)")
    if _conf_enabled:
        print(
            "[JAX] Phase9.1 confidence: head_confidence predicts carry_fwd MSE "
            f"(coef={_conf_coef}, target stop_grad)"
        )
    _img_gate = bool(_p9.get("imagination_gating_enabled", False))
    if _img_gate:
        _mult = float(_p9.get("confidence_multiplier", 1.0))
        _ik = int(_p9.get("imagination_k", 5))
        _ig = float(_p9.get("imagination_gamma", config.get("ppo_gamma", 0.999)))
        print(
            f"[JAX] Phase12.1 spatial epistemic gate: conf_pred < mean(conf|alive)*{_mult} "
            f"→ imagined_action (batch-relative, stateless); else reactive | K={_ik} γ={_ig}"
        )
    from jax_sim import observations_jax as _obs_mod

    _bo_path = _inspect.getfile(_obs_mod.build_observations_jax)
    with open(_bo_path, encoding="utf-8") as _f:
        _disk_bo = _f.read()
    if "red_map.any()" in _disk_bo:
        raise RuntimeError(
            f"Git is at {_git_sha} but {_bo_path} still has red_map.any().\n"
            "Run: cd /root/throng && git fetch origin && git reset --hard origin/master"
        )
    print(f"[JAX] red_sense_api=v{RED_SENSE_API_VERSION} (observations_jax)")
    if _red_comms:
        print(
            f"[JAX] Phase12 red comms: PredatorNetworkJax hidden={red_hidden_d} "
            f"red_codebook vocab={_red_vocab} cross_attn={_red_cross} "
            f"heads={_cross_heads} | red_neighbor_api=v{RED_NEIGHBOR_SIGNAL_API_VERSION}"
        )

    # ── Red curriculum state ─────────────────────────────────
    red_curriculum_stages = list(config.get("red_curriculum_stages", [6, 15, 30, 75]))
    red_curriculum_idx = 0
    red_sustain_count = 0
    red_sustain_threshold = float(config.get("curriculum_survival_threshold", 0.80))
    red_sustain_needed = int(config.get("curriculum_sustain_updates", 5))
    print(
        f"[CURRICULUM] Red stages: {red_curriculum_stages}, start={red_curriculum_stages[0]} "
        f"| catch_radius={int(config.get('red_catch_radius', 1))} "
        f"| catch_prob={float(config.get('red_catch_prob', 1.0))}"
    )

    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_carry = jnp.zeros((1, hidden_d))
    dummy_carry_red = jnp.zeros((1, r_pop_hidden))
    _ckpt_latest = ckpt_mngr.latest_step()
    # 2026-09-14 (Cam): resume_from_step pins an EARLIER-than-latest
    # checkpoint deliberately -- the calibration ladder's codes_active rule
    # ("latest checkpoint whose codes_active is within 2x of the per-slot
    # ladder maximum, on every slot independently") picked 2541, not
    # whatever the volume's newest checkpoint happens to be. Set in
    # config.yaml, not a volume deletion -- explicit, auditable, reversible.
    _resume_pin = config.get("resume_from_step")
    if _resume_pin is not None:
        _resume_pin = int(_resume_pin)
        if _ckpt_latest is not None and _resume_pin > _ckpt_latest:
            raise ValueError(
                f"resume_from_step={_resume_pin} is AHEAD of the volume's "
                f"latest checkpoint ({_ckpt_latest}) -- refusing a config "
                f"that can't possibly be satisfied from disk."
            )
        print(
            f"[JAX] resume_from_step={_resume_pin} set in config.yaml -- "
            f"pinning resume to this step instead of the volume's latest "
            f"({_ckpt_latest}). Deliberate rollback, not a bug.",
            flush=True,
        )
        _ckpt_latest = _resume_pin
    # 2026-09-15 (Cam): the guard that would have caught both of tonight's
    # losses before a dollar was spent. A deliberate rollback into a
    # directory that also holds a numerically-higher checkpoint from a
    # different lineage is a self-deleting run: Orbax's max_to_keep
    # retention prunes by step number, not save recency, so every new
    # checkpoint this run saves is "older" than the fossil by that metric
    # and gets silently garbage collected on save, before Volume.commit()
    # is even in the picture. Nothing downstream can detect this --
    # "[CKPT] Saved" / "[CKPT] Committed" print unconditionally regardless.
    # Check the actual on-disk state against what we're about to resume
    # into, before the first rollout, not after 7.5 hours: assert the
    # environment is what we think it is, same shape as the durability
    # gate in scripts/modal_app.py.
    if _ckpt_latest is not None:
        _fossil_steps = find_fossil_checkpoints(ckpt_mngr, _ckpt_latest)
        if _fossil_steps:
            _all_steps = sorted(int(s) for s in ckpt_mngr.all_steps())
            print(
                f"[FOSSIL-GUARD] FATAL: {ckpt_dir!r} contains checkpoint(s) "
                f"{_fossil_steps} numerically AHEAD of the resume target "
                f"({_ckpt_latest}). Resuming here would let Orbax's own "
                f"max_to_keep retention silently prune every checkpoint this "
                f"run saves, the instant it saves it -- the exact mechanism "
                f"that lost two real launches on 2026-09-15 (see "
                f"docs/THE-ECOLOGY-NEVER-RAN.md instance 6). All steps "
                f"present: {_all_steps}. Move the fossil(s) out (see "
                f"fossils/ on the volume) or point checkpoint_dir at a fresh, "
                f"fossil-free directory before launching. Refusing to run.",
                flush=True,
            )
            raise SystemExit(1)
    start_update = 0
    # Training-loop state carried across resumes (CtD ramp progress, red
    # curriculum) -- captured from the raw checkpoint below if present,
    # mirroring how usage_ema/dead_streak already solve this for VQ codebooks.
    # None on a fresh start, or when the checkpoint predates this mechanism.
    _restored_training_state = None
    if _ckpt_latest is not None:
        print(
            f"[JAX] Checkpoint on volume: latest PPO update = {_ckpt_latest}",
            flush=True,
        )
        print(
            "[JAX] Compiling model.init (template for restore) — often 1–5 min with "
            "no new lines. Not frozen; do not Stop the kernel.",
            flush=True,
        )
        b_params = sanitize_agent_params(
            init_agent_params(model, keys[3], dummy_carry, dummy_obs, n_layers)
        )
        if _red_comms:
            r_params = sanitize_agent_params(
                init_predator_params(
                    model_red, keys[5], dummy_carry_red, dummy_obs, n_layers
                )
            )
            print(
                "[JAX] model.init done (blue template + fresh predator) — "
                "restoring checkpoint...",
                flush=True,
            )
        else:
            r_params = jax.tree_util.tree_map(jnp.copy, b_params)
            print("[JAX] model.init done — restoring checkpoint weights...", flush=True)
        abstract_tree = {"b_params": b_params, "r_params": r_params}
        # Always restore the raw dictionary from disk and graft manually.
        # Passing items=abstract_tree causes Orbax to auto-create empty parent dicts
        # for missing heads, which bypasses our graft_missing_param_subtrees logic.
        try:
            raw_restored = ckpt_mngr.restore(_ckpt_latest)
            from flax.core import freeze, unfreeze
            target_dict = unfreeze(abstract_tree)
            source_dict = unfreeze(raw_restored)
            if "training_state" in source_dict:
                _restored_training_state = source_dict["training_state"]
            _restore_agents = ("b_params", "r_params")
            for agent_type in _restore_agents:
                if agent_type not in source_dict or agent_type not in target_dict:
                    continue
                src_agent = unfreeze(source_dict[agent_type])
                tgt_agent = unfreeze(target_dict[agent_type])
                injected_paths = graft_missing_param_subtrees(src_agent, tgt_agent)
                if injected_paths:
                    for path in injected_paths:
                        print(
                            f"[JAX] Injected randomly initialized {path} "
                            f"into {agent_type}",
                            flush=True,
                        )
                    _print_grafted_vq_counter_stats(src_agent, injected_paths, agent_type)
                    source_dict[agent_type] = freeze(src_agent)
            restored = freeze(source_dict)
        except ValueError as exc:
            msg = str(exc)
            if "do not match" in msg or "Topology mismatch" in msg:
                print(
                    f"[JAX] Orbax strict match failed (schema evolution). "
                    f"Merging new heads manually... ({msg})",
                    flush=True,
                )
                from flax.core import freeze, unfreeze

                target_dict = unfreeze(abstract_tree)
                source_dict = {"b_params": {}, "r_params": {}}
                
                try:
                    raw_restored = ckpt_mngr.restore(_ckpt_latest)
                except ValueError:
                    # 2026-09-14: this fallback had never been exercised
                    # successfully -- `ckpt_mngr.restore(step, items=target_dict)`
                    # goes through CheckpointManager's "default" item, which is
                    # bound to StandardCheckpointHandler; StandardRestoreArgs's
                    # `strict=False` does NOT loosen a key-set mismatch (tested:
                    # raises the identical "do not match" error), and orbax's
                    # own error message ("pass partial_restore=True") names a
                    # flag StandardRestore doesn't expose at all -- it only
                    # exists on the lower-level PyTreeRestoreArgs, which this
                    # CheckpointManager's handler registration refuses ("does
                    # not match with any registered handler"). First checkpoint
                    # old enough to need this path (2541, predates head_signal
                    # reactivation / nb_cross_attn / red_codebook, and carries
                    # codebook_N.usage_ema/dead_streak fields the current
                    # model.init() doesn't produce) is what surfaced it -- every
                    # prior resume in this project's history was close enough
                    # in architecture to succeed on the unconstrained restore
                    # above and never reach here. Fix: bypass the Standard-bound
                    # CheckpointManager for this one restore and go straight to
                    # a PyTreeCheckpointer against the on-disk "default" item,
                    # which does honor partial_restore -- verified locally
                    # against a real bidirectional mismatch (extra keys on both
                    # sides) before landing here.
                    # partial_restore alone still hits "Topology mismatch" /
                    # "sharding ... Got None" when the checkpoint was saved on
                    # a different device topology (GPU) than this restore call
                    # runs on (CPU preflight) -- construct_restore_args derives
                    # concrete restore_args from target_dict's own (already
                    # correctly-placed) leaves, same fix diag_ze_variance.py
                    # and test_checkpoint_compat.py already use for the plain
                    # (non-partial) restore path.
                    _pytree_ckptr = ocp.PyTreeCheckpointer()
                    _raw_default_dir = os.path.join(str(ckpt_dir), str(_ckpt_latest), "default")
                    _restore_args = ocp.checkpoint_utils.construct_restore_args(target_dict)
                    raw_restored = _pytree_ckptr.restore(
                        _raw_default_dir,
                        args=ocp.args.PyTreeRestore(
                            item=target_dict, restore_args=_restore_args, partial_restore=True,
                        ),
                    )
                
                source_dict = unfreeze(raw_restored)
                if "training_state" in source_dict:
                    _restored_training_state = source_dict["training_state"]
                _restore_agents = ("b_params", "r_params")
                for agent_type in _restore_agents:
                    if agent_type not in source_dict or agent_type not in target_dict:
                        continue
                    src_agent = unfreeze(source_dict[agent_type])
                    tgt_agent = unfreeze(target_dict[agent_type])
                    injected_paths = graft_missing_param_subtrees(src_agent, tgt_agent)
                    for path in injected_paths:
                        print(
                            f"[JAX] Injected randomly initialized {path} "
                            f"into {agent_type}",
                            flush=True,
                        )
                    if injected_paths:
                        _print_grafted_vq_counter_stats(src_agent, injected_paths, agent_type)
                    source_dict[agent_type] = freeze(src_agent)
                restored = freeze(source_dict)
            elif "not compatible" in msg or "stored shape" in msg:
                print(
                    f"[JAX] Checkpoint step {_ckpt_latest} incompatible with current model "
                    f"(architecture changed) — re-init from scratch.",
                    flush=True,
                )
                print("[JAX] Delete old ckpts: rm -rf /mnt/throng-runs/checkpoints")
                if _red_comms:
                    r_params = sanitize_agent_params(
                        init_predator_params(
                            model_red, keys[5], dummy_carry_red, dummy_obs, n_layers
                        )
                    )
                else:
                    r_params = sanitize_agent_params(
                        init_agent_params(model, keys[5], dummy_carry, dummy_obs, n_layers)
                    )
                start_update = 0
                restored = None
            else:
                raise
        if restored is not None:
            b_params = sanitize_agent_params(
                ensure_aux_head_params(
                    model, restored["b_params"], keys[3], hidden_d,
                    obs_dim=obs_dim, n_layers=n_layers,
                )
            )
            if bool(_p9.get("reset_confidence_head_on_resume", False)):
                b_params = reset_confidence_head_on_resume(
                    model, b_params, keys[3], hidden_d,
                    obs_dim=obs_dim, n_layers=n_layers,
                )
            if _red_comms:
                r_params = sanitize_agent_params(
                    ensure_predator_params(
                        model_red, restored["r_params"], keys[5], red_hidden_d,
                        obs_dim=obs_dim, n_layers=n_layers
                    )
                )
                if bool(_p14t.get("reset_red_vq_on_resume", False)):
                    r_params = reset_predator_vq_on_resume(
                        model_red, r_params, keys[5], red_hidden_d,
                        obs_dim=obs_dim, n_layers=n_layers,
                    )
            else:
                r_params = sanitize_agent_params(
                    ensure_aux_head_params(
                        model, restored["r_params"], keys[5], hidden_d,
                        obs_dim=obs_dim, n_layers=n_layers,
                    )
                )
            start_update = int(_ckpt_latest)
            print(
                f"[JAX] Restored params from step {start_update}. Population starts fresh.",
                flush=True,
            )
    else:
        print("[JAX] No checkpoint on volume — training from update 0.", flush=True)
        print(
            "[JAX] Compiling model.init (blue + red) — two passes, 2–8 min total. "
            "Do not Stop the kernel.",
            flush=True,
        )
        b_params = sanitize_agent_params(
            init_agent_params(model, keys[3], dummy_carry, dummy_obs, n_layers)
        )
        if _red_comms:
            r_params = sanitize_agent_params(
                init_predator_params(
                    model_red, keys[5], dummy_carry_red, dummy_obs, n_layers
                )
            )
            print("[JAX] model.init done (blue + fresh predator).", flush=True)
        else:
            r_params = sanitize_agent_params(
                init_agent_params(model, keys[5], dummy_carry, dummy_obs, n_layers)
            )
            print("[JAX] model.init done (blue + red).", flush=True)
    from flax.core import unfreeze as _unfreeze_params
    _bp = _unfreeze_params(b_params)
    _emb_ok = "kernel" in _bp.get("emb_own", {})
    _aux_ok = all(k in _bp for k in AUX_HEAD_KEYS)
    try:
        model_apply(b_params, dummy_carry, dummy_obs, n_layers)
        _apply_ok = True
    except Exception as _apply_err:
        _apply_ok = False
        print(f"[DEBUG] apply smoke-test FAILED: {_apply_err}")
    _vq_on = any(k in _bp for k in ("codebook", "codebook_0"))
    print(
        f"[JAX] signal_bottleneck={'VQ' if _vq_on else 'LEGACY softmax'} "
        f"| vocab={config.get('vocab_size', 64)} "
        f"| vq_beta={config.get('vq_beta', 0.25)} "
        f"| vq_loss_coef={config.get('vq_loss_coef', 0.1)} "
        f"| dead_code_reset={config.get('vq_dead_code_reset', True)}"
    )
    print(
        f"[DEBUG] Params OK: emb_own={_emb_ok} | aux_heads={_aux_ok} | apply={_apply_ok}"
    )
    if not (_emb_ok and _aux_ok and _apply_ok):
        raise RuntimeError(
            "Parameter init failed — delete runs/jax_run/checkpoints and restart runtime"
        )

    # ── Auxiliary heads apply function (forward dynamics + self-prediction) ──
    import functools as _functools
    def _aux_apply_fn(params, carry_t, action_oh):
        return model.apply(
            params_apply_variables(params), carry_t, action_oh,
            method=model.auxiliary_heads,
        )

    b_aux_apply_fn = _aux_apply_fn
    r_aux_apply_fn = _aux_apply_fn

    def _proprio_apply_fn(params, carry_t):
        return model.apply(
            params_apply_variables(params),
            carry_t,
            method=model.predict_proprio_energy,
        )

    b_proprio_apply_fn = _proprio_apply_fn
    r_proprio_apply_fn = _proprio_apply_fn
    r_red_aux_apply_fn = None
    if model_red is not None:
        def _red_aux_apply_fn(params, carry_t, obs_seq):
            def scan_fn(c, o):
                new_c, _ = model_red.apply(
                    params_apply_variables(params),
                    c, o, n_layers, deterministic=True,
                )
                return new_c, None
            
            # obs_seq is shape (lag, M, obs_dim)
            final_carry, _ = jax.lax.scan(scan_fn, carry_t, obs_seq)
            
            return model_red.apply(
                params_apply_variables(params),
                final_carry,
                method=model_red.red_auxiliary_heads,
            )
        r_red_aux_apply_fn = _red_aux_apply_fn

    _proprio_coef = float(config.get("proprio_coef", 0.05))
    if _proprio_coef > 0.0:
        print(
            f"[JAX] Phase14.1b proprio: head_proprio → energy (coef={_proprio_coef}) "
            "— disentangle metabolic state from VQ wire"
        )

    print(
        f"[JAX] Phase14.2 Metabolic Asymmetry: red_energy_decay={config['red_energy_decay']}",
        flush=True,
    )

    # ── NaN debug after init ────────────────────────────────
    flat_params = jax.tree_util.tree_leaves(b_params)
    has_nan_params = any(bool(jnp.isnan(p).any()) for p in flat_params)
    print(f"[DEBUG] Params NaN after init: {has_nan_params}")

    # ── JAX pmap setup ──────────────────────────────────────
    n_devices = jax.device_count()
    use_pmap = config.get("use_pmap", False) and n_devices > 1
    if use_pmap:
        print(f"[JAX] Found {n_devices} devices. Using pmap for multi-GPU rollout!")
        grid = jax.tree_util.tree_map(lambda x: jnp.stack([x]*n_devices), grid)
        b_pop = jax.tree_util.tree_map(lambda x: jnp.stack([x]*n_devices), b_pop)
        r_pop = jax.tree_util.tree_map(lambda x: jnp.stack([x]*n_devices), r_pop)
        b_carries = jnp.stack([b_carries]*n_devices)
        r_carries = jnp.stack([r_carries]*n_devices)

    # ── Debug: inspect initial action logits / entropy ──────
    test_carry = jnp.zeros((1, hidden_d))
    test_obs = jnp.zeros((1, obs_dim))
    _, test_outs = model_apply(b_params, test_carry, test_obs, n_layers)
    test_logits = test_outs.action_logits
    test_probs = jax.nn.softmax(test_logits, axis=-1)
    test_entropy = -jnp.sum(test_probs * jnp.log(test_probs + 1e-10), axis=-1)
    print(f"[DEBUG] Init action_logits mean={float(test_logits.mean()):.4f} std={float(test_logits.std()):.4f}")
    print(f"[DEBUG] Init entropy mean={float(test_entropy.mean()):.4f} (expected ~1.6 for uniform 5-action)")

    # ── Init optimizer (after checkpoint restore so momentum matches weights) ──
    b_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    r_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    b_opt_state = b_optimizer.init(b_params)
    
    b_aux_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    r_aux_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    b_aux_opt_state = b_aux_optimizer.init(b_params)
    r_aux_opt_state = r_aux_optimizer.init(r_params)
    
    b_vqel_optimizer = None
    b_vqel_opt_state = None
    if _vqel_monologue:
        b_vqel_optimizer = create_optimizer(_vqel_lr, config["ppo_max_grad_norm"])
        b_vqel_opt_state = b_vqel_optimizer.init(b_params)
        print(
            f"[JAX] Phase14 VQEL monologue: ON (lr={_vqel_lr}, "
            f"recon={_vqel_recon_coef}, hash={_vqel_hash_coef}, vq={_vqel_vq_coef}) "
            "— blue PPO/aux skipped; policy heads frozen; blue broadcast=WIRE CUT (silence)"
        )
        print(
            f"[JAX] Phase14 graduation: recon_mse < {_graduate_recon_mse} for "
            f"{_graduate_consecutive} consecutive updates → dialogue hard + blue PPO resume"
        )
    elif _dialogue_signal_mode == "hard":
        print("[JAX] Phase14 dialogue_signal_mode=hard (discrete z_q broadcast on blue wire)")
        
    print(f"[JAX] Phase 18 action_space=12 (added PICK_UP=9, CRAFT=10, USE_TOOL=11)")
    print(f"[JAX] Phase 18.x Logit-mask active: {masked_actions_banner()} set to -1e9 before sampling, both teams (no-op/disabled amputation)")
    print(f"[JAX] Phase 18 Continuous-to-Discrete (CtD) bootstrap: 100k-step decay ramp active.")
    print(f"[JAX] Phase 18 Wire budget: 40D (8D cont + 12/8/12 discrete slots). Codebooks initialized.")
    print(f"[JAX] Phase 16.6 GWT Router mask active: Zero out age(0), mat(1), energy(2), layers(3)")
    _p15_run = config.get("phase15_cumulative_culture") or {}
    _medal_adr_enabled = bool(_p15_run.get("medal_adr_enabled", False))
    _medal_adr_prob = float(_p15_run.get("medal_adr_prob", 0.0))
    
    if _medal_adr_enabled and _medal_adr_prob > 0.0:
        print(f"[JAX] MEDAL-ADR soft carry-reset: prob={_medal_adr_prob}, target=oldest agents by age")
    else:
        print("[JAX] MEDAL-ADR disabled or prob=0 — soft reset inactive")
    
    r_opt_state = r_optimizer.init(r_params)

    # ── Carries ─────────────────────────────────────────────
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, r_pop_hidden))

    # ── Training loop ───────────────────────────────────────
    n_updates = n_steps // T
    update_keys = jax.random.split(keys[4], n_updates)
    print(
        f"[JAX] Training PPO updates {start_update} → {n_updates - 1} "
        f"(~env steps {start_update * T} → {n_steps})",
        flush=True,
    )
    if start_update < n_updates:
        print(
            f"[JAX] About to run lax.scan rollout ({T}×{max_pop} agents). "
            "First compile can take 5–15+ min with no new lines — not frozen.",
            flush=True,
        )

    # ── Brain vote state (capacity-based, not survival-based) ─
    brain_max_layers_val = int(config.get("brain_max_layers", 6))
    brain_vote_interval_updates = max(1, int(config.get("brain_vote_interval", 5000)) // T)
    brain_vote_window = max(5, brain_vote_interval_updates)
    brain_ent_history = []
    brain_vf_history = []
    brain_sig_diversity_history = []

    def _rebuild_sim_step(cur_n_layers):
        cfg_copy = dict(config)
        cfg_copy["n_layers"] = cur_n_layers
        cfg_copy["red_energy_decay"] = float(config["red_energy_decay"])
        p14_live = dict(cfg_copy.get("phase14_vqel") or {})
        p14_live["monologue_enabled"] = _vqel_monologue
        p14_live["dialogue_signal_mode"] = _dialogue_signal_mode
        cfg_copy["phase14_vqel"] = p14_live
        return make_sim_step(
            cfg_copy,
            model,
            model_apply,
            r_model_apply=r_model_apply if _red_comms else None,
        )

    sim_step_fn = _rebuild_sim_step(n_layers)
    if use_pmap:
        # in_axes: step_keys(0), grid(0), b_pop(0), r_pop(0), b_carries(0), r_carries(0), b_params(None), r_params(None)
        # But wait, sim_step_fn signature is: (carry, xs) -> (carry, ys)
        # The carry is (grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params)
        # So we pmap a wrapper that takes them separately.
        # lax.scan already handles the scan over T.
        def pmap_rollout(p_grid, p_b_pop, p_r_pop, p_b_carries, p_r_carries, p_step_keys):
            init_carry = (p_grid, p_b_pop, p_r_pop, p_b_carries, p_r_carries, b_params, r_params)
            final_carry, rollout_data = lax.scan(sim_step_fn, init_carry, p_step_keys)
            return final_carry, rollout_data
            
        sim_step_mapped = jax.pmap(pmap_rollout, in_axes=(0, 0, 0, 0, 0, 0))
    else:
        sim_step_mapped = None

    loop_key = keys[5]
    _t_start = __import__('time').time()
    _t_last = _t_start
    all_metrics = []
    # Lag-1 buffers for corpus / decode (blue scouts, red hunters)
    _lag1_scout_pos = None
    _lag1_scout_sig = None
    _lag1_scout_dist = None
    _lag1_scout_tok = None
    _lag1_hunter_pos = None
    _lag1_hunter_sig = None
    _lag1_hunter_dist = None
    _lag1_hunter_tok = None
    _alarm_range = float(config.get("alarm_scout_range", 8))
    print(
        f"[JAX] corpus scout label: is_scout = (red_dist <= alarm_scout_range={_alarm_range})",
        flush=True,
    )
    _corpus_sig_dim = int(config["signal_dim"])

    # ── CtD competence ramp state (Cam's sign-off, 2026-09-14) ───────
    # Ratchet decision lives here (outer Python loop), mirroring the
    # red_curriculum_idx/red_sustain_count pattern above -- floor + sustained
    # bar + hard ceiling, evaluated once per PPO update from rollout_data.
    #
    # Persisted in the checkpoint under "training_state" (Cam's correction,
    # 2026-09-14): red_curriculum_idx/red_sustain_count were plain Python ints,
    # reset to zero on every process start, never restored -- the same defect
    # class as the rest of the audit (state that exists, is read, shapes
    # behaviour, has no durable write path), just the mirror image of the
    # confidence-head fossil: there, stale state survived a reset it should
    # have had; here, live state resets across a boundary it should survive.
    # Both fields have been resetting on every resume since Phase 15/16
    # (whenever red curriculum staging landed) -- see RESEARCH_PROTOCOL.md /
    # THE-ECOLOGY-NEVER-RAN.md. Fixed the same way usage_ema/dead_streak
    # already solve this for VQ codebooks: captured from the raw checkpoint
    # above if present, defaulted fresh otherwise.
    _ramp_cfg_outer = config.get("ctd_competence_ramp", {})
    craft_ramp_enabled = bool(_ramp_cfg_outer.get("craft_ramp_enabled", False))
    # 2026-09-14 (Cam): 5_120 (10 updates), not 50_000 -- the old floor was
    # sized for the retired success/attempts bar and now dominates the
    # per-capita bar entirely (see config.yaml's ctd_competence_ramp comment
    # for the measured live numbers). Registered as a next-launch change,
    # not applied retroactively to any run already in flight.
    craft_ramp_min_steps = int(_ramp_cfg_outer.get("craft_ramp_min_steps", 5_120))
    # 2026-09-14 (Cam): per-capita bar -- success as a fraction of the living
    # blue population, not success/attempts (attempts is agent-controlled;
    # that metric can't measure competence by construction, same instrument-
    # validity standard as the rel_spread retirement). 10% sustained 3
    # updates.
    craft_ramp_success_bar = float(_ramp_cfg_outer.get("craft_ramp_success_bar", 0.10))
    craft_ramp_success_window = int(_ramp_cfg_outer.get("craft_ramp_success_window", 3))
    craft_ramp_max_steps = int(_ramp_cfg_outer.get("craft_ramp_max_steps", 1_000_000))
    # 2026-09-14 (Cam): a hard escape scoped to stage 0 specifically, tighter
    # than the general craft_ramp_max_steps ceiling. Stage 0 has no
    # communication pressure -- it's the phase that ate the channel last
    # time -- and shortening it was the whole point of the three-stage
    # design. A bar that may never clear (observed live: craft rate 1.3% ->
    # 0.4% while futile_wrong_mats nearly tripled, receding not approaching)
    # would silently convert a deliberately short phase into an unbounded
    # one. 40 updates, in PPO updates not steps, measured from stage 0's own
    # start (== run start on a fresh resume into stage 0).
    craft_ramp_stage0_max_updates = int(_ramp_cfg_outer.get("craft_ramp_stage0_max_updates", 40))
    craft_ramp_active_outer = craft_ramp_enabled
    # Two capped stages (Cam's correction, 2026-09-14): max_units=2 alone was
    # still a cooperative problem at smaller scale, not the solo-catchable
    # analogue -- stage 0 (1 unit, solo-satisfiable) was missing entirely.
    # Stage index into jax_sim.ctd_ramp.CRAFT_RAMP_STAGE_UNITS = (1, 2).
    craft_ramp_stage_outer = 0
    craft_ramp_start_step = start_update * T  # start of the CURRENT stage
    craft_ramp_success_streak = 0
    # 2026-09-15 (Cam): ppo update index at which the comms subtree last
    # unfroze, or None before that's happened / once the 10-update warmup
    # below has fully elapsed. Drives comms_lr_warmup in the ppo_update call
    # below -- see the comment in rl_jax._minibatch_step for why a step-size
    # warmup (not a gradient-magnitude one) is what defuses the freeze-exit
    # spring. Not restored across a resume: the window is 10 updates, short
    # enough that a resume landing inside it is an edge case left unhandled.
    _comms_unfreeze_at_update = None
    _comms_lr_warmup_updates = 10

    red_ramp_enabled = bool(_ramp_cfg_outer.get("red_ramp_enabled", False))
    red_ramp_min_steps = int(_ramp_cfg_outer.get("red_ramp_min_steps", 200_000))
    red_ramp_catch_bar = float(_ramp_cfg_outer.get("red_ramp_catch_bar", 1.0))
    red_ramp_catch_window = int(_ramp_cfg_outer.get("red_ramp_catch_window", 10))
    red_ramp_max_steps = int(_ramp_cfg_outer.get("red_ramp_max_steps", 1_000_000))
    red_ramp_active_outer = red_ramp_enabled
    red_ramp_start_step = start_update * T
    red_ramp_catch_streak = 0

    # Comms-freeze tripwires (Cam, 2026-09-14, Blocker 2 behavioral check;
    # thresholds revised 2026-09-14 -- see RESEARCH_PROTOCOL.md Part 2, Cam's
    # own instance): the freeze proves comms PARAMS don't move; it can't
    # prove codes_active doesn't drift, because z_e comes out of the frozen
    # head applied to a trunk that keeps training under the policy loss.
    #
    # C0 = per-slot MEDIAN of codes_active over updates 3-7 after resume
    # (updates 1-2 discarded outright). Capture is stability-gated, not a
    # fixed window (Cam's second registered correction, 2026-09-14): the
    # first fixed-window draft (median of updates 3-7) landed on a
    # transient -- slot1 sampled 14|13|13|30|52 while it was still
    # recovering toward its settled ~50s, giving a fast trip that would
    # fire on a 70% collapse silently. C0 = median of the first 5
    # consecutive updates (window start >= 3) in which every internal
    # update-over-update transition is <=20% relative change on all three
    # slots. If no such window forms by update 40, force-capture the
    # median of updates 35-40 and log it loudly -- tied to the same
    # 40-update budget as the stage-0 hard escape. Tripwires arm the
    # update after capture. Never hardcoded from the offline calibration
    # ladder, which used a different batch and isn't comparable in
    # absolute scale. All of this persists across resumes so a crash
    # mid-search or mid-stage-0 doesn't restart C0 collection.
    comms_updates_since_resume = 0   # counts qualifying updates since THIS resume
    comms_sample_history = []        # list of (u0,u1,u2) per-slot codes_active, one per update, index 0 = update 1
    comms_c0_stage0 = None                  # (u0, u1, u2) per-slot median, stage-0 (frozen-channel) baseline
    comms_c0_stage0_captured_at_update = 0  # update index capture happened at; tripwires arm the update after
    comms_fast_streak = [0, 0, 0]    # consecutive updates below 0.5*C0, per slot
    comms_drift_streak = [0, 0, 0]   # consecutive updates below 0.75*C0, per slot
    comms_floor_streak = [0, 0, 0]   # consecutive updates below 0.85*C0, per slot (stage 0 only)
    comms_stage1_updates_elapsed = 0
    comms_stage1_uncoordinated_seen = False
    # 2026-09-20 (Cam): stage-1 (live comms channel) baseline, separate from
    # comms_c0_stage0. The single-baseline design compared LIVE stage-1
    # codes_active against a median captured while the comms subtree was
    # FROZEN (stage 0) -- correct for the floor tripwire (explicitly
    # stage-0-only) but wrong for fast/drift, which kept comparing
    # post-unfreeze dynamics against a frozen-channel number two updates
    # stale by the time the channel came back to life (armed at update 8,
    # unfreeze at ~update 10 on the run this was measured against). The
    # tripwire firing on the 2026-09-20 collapse was still correct (82%
    # drop, no baseline makes that healthy) -- this is about calibrating
    # against what's actually live going forward, not about that call.
    # comms_c0_stage1 uses the identical stability-gated capture algorithm
    # as stage0, applied to codes_active samples collected only AFTER the
    # comms_lr_warmup ramp fully completes (not during it -- the channel is
    # still deliberately ramping in, not settled, so a "stable window"
    # found there would be measuring the ramp schedule, not the channel).
    comms_stage1_c0_search_active = False    # True once warmup has elapsed and stage-1 sample collection has begun
    comms_stage1_c0_search_updates = 0       # counts qualifying updates since warmup completion (this resume)
    comms_stage1_sample_history = []
    comms_c0_stage1 = None
    comms_c0_stage1_captured_at_update = 0

    _training_state_source = "fresh start (no saved training_state)"
    if _restored_training_state is not None:
        _training_state_source = "RESTORED from checkpoint"
        if craft_ramp_enabled:
            craft_ramp_active_outer = bool(_restored_training_state.get("craft_ramp_active", craft_ramp_active_outer))
            craft_ramp_stage_outer = int(_restored_training_state.get("craft_ramp_stage", craft_ramp_stage_outer))
            craft_ramp_start_step = int(_restored_training_state.get("craft_ramp_start_step", craft_ramp_start_step))
            craft_ramp_success_streak = int(_restored_training_state.get("craft_ramp_success_streak", craft_ramp_success_streak))
            comms_updates_since_resume = int(_restored_training_state.get(
                "comms_updates_since_resume", comms_updates_since_resume
            ))
            _tw_history_restored = _restored_training_state.get("comms_sample_history", None)
            if _tw_history_restored is not None:
                _tw_history_arr = np.asarray(_tw_history_restored)  # (40, 3), -1 row = not yet collected
                comms_sample_history = [
                    tuple(int(v) for v in row) for row in _tw_history_arr.tolist() if row[0] >= 0
                ]
            # Dict keys unchanged from before the stage0/stage1 split (only
            # the local variable names changed, comms_c0 -> comms_c0_stage0)
            # so this still restores correctly from a checkpoint saved by
            # the pre-split code -- e.g. checkpoint 2544, saved 2026-09-15.
            comms_c0_stage0_captured_at_update = int(_restored_training_state.get(
                "comms_c0_captured_at_update", comms_c0_stage0_captured_at_update
            ))
            _tw_c0_restored = _restored_training_state.get("comms_tripwire_c0", None)
            if _tw_c0_restored is not None:
                _tw_c0_vals = [float(x) for x in np.asarray(_tw_c0_restored).tolist()]
                comms_c0_stage0 = None if any(x < 0 for x in _tw_c0_vals) else tuple(_tw_c0_vals)
            comms_fast_streak = [int(x) for x in np.asarray(
                _restored_training_state.get("comms_fast_streak", comms_fast_streak)
            ).tolist()]
            comms_drift_streak = [int(x) for x in np.asarray(
                _restored_training_state.get("comms_drift_streak", comms_drift_streak)
            ).tolist()]
            comms_floor_streak = [int(x) for x in np.asarray(
                _restored_training_state.get("comms_floor_streak", comms_floor_streak)
            ).tolist()]
            comms_stage1_updates_elapsed = int(_restored_training_state.get(
                "comms_stage1_updates_elapsed", comms_stage1_updates_elapsed
            ))
            comms_stage1_uncoordinated_seen = bool(_restored_training_state.get(
                "comms_stage1_uncoordinated_seen", comms_stage1_uncoordinated_seen
            ))
            # New fields (2026-09-20): absent on any checkpoint saved before
            # this split, defaults above (None/0/False/[]) apply -- a
            # restore from an old checkpoint into stage 1 just starts the
            # stage-1 search fresh, which is correct (no stage-1 baseline
            # was ever captured under the old single-baseline code anyway).
            comms_stage1_c0_search_active = bool(_restored_training_state.get(
                "comms_stage1_c0_search_active", comms_stage1_c0_search_active
            ))
            comms_stage1_c0_search_updates = int(_restored_training_state.get(
                "comms_stage1_c0_search_updates", comms_stage1_c0_search_updates
            ))
            _tw_stage1_history_restored = _restored_training_state.get("comms_stage1_sample_history", None)
            if _tw_stage1_history_restored is not None:
                _tw_stage1_history_arr = np.asarray(_tw_stage1_history_restored)
                comms_stage1_sample_history = [
                    tuple(int(v) for v in row) for row in _tw_stage1_history_arr.tolist() if row[0] >= 0
                ]
            comms_c0_stage1_captured_at_update = int(_restored_training_state.get(
                "comms_c0_stage1_captured_at_update", comms_c0_stage1_captured_at_update
            ))
            _tw_c0_stage1_restored = _restored_training_state.get("comms_tripwire_c0_stage1", None)
            if _tw_c0_stage1_restored is not None:
                _tw_c0_stage1_vals = [float(x) for x in np.asarray(_tw_c0_stage1_restored).tolist()]
                comms_c0_stage1 = None if any(x < 0 for x in _tw_c0_stage1_vals) else tuple(_tw_c0_stage1_vals)
        if red_ramp_enabled:
            red_ramp_active_outer = bool(_restored_training_state.get("red_ramp_active", red_ramp_active_outer))
            red_ramp_start_step = int(_restored_training_state.get("red_ramp_start_step", red_ramp_start_step))
            red_ramp_catch_streak = int(_restored_training_state.get("red_ramp_catch_streak", red_ramp_catch_streak))
        red_curriculum_idx = int(_restored_training_state.get("red_curriculum_idx", red_curriculum_idx))
        red_sustain_count = int(_restored_training_state.get("red_sustain_count", red_sustain_count))

    # Sync grid (constructed before the checkpoint restore above) to whatever
    # the ramp-active flags ended up being -- config-driven default, or
    # restored-and-possibly-already-ratcheted history.
    grid = grid.replace(
        craft_ramp_active=jnp.array(craft_ramp_active_outer, dtype=jnp.bool_),
        craft_ramp_stage=jnp.array(craft_ramp_stage_outer, dtype=jnp.int32),
        red_ramp_active=jnp.array(red_ramp_active_outer, dtype=jnp.bool_),
    )

    if craft_ramp_enabled or red_ramp_enabled:
        print(
            f"[CTD-RAMP] state: {_training_state_source} | "
            f"red_curriculum_idx={red_curriculum_idx} red_sustain_count={red_sustain_count}",
            flush=True,
        )
        _craft_stage_units = CRAFT_RAMP_STAGE_UNITS[craft_ramp_stage_outer] if craft_ramp_active_outer else "full"
        print(
            f"[CTD-RAMP] crafting={craft_ramp_active_outer} "
            f"(stage={craft_ramp_stage_outer}/{len(CRAFT_RAMP_STAGE_UNITS) - 1} units={_craft_stage_units}, "
            f"floor={craft_ramp_min_steps:_}, bar={craft_ramp_success_bar:.0%}/"
            f"{craft_ramp_success_window}upd, ceiling={craft_ramp_max_steps:_}, "
            f"stage0_hard_escape={craft_ramp_stage0_max_updates}upd, "
            f"streak={craft_ramp_success_streak}, stage_start_step={craft_ramp_start_step:_}) | "
            f"red={red_ramp_active_outer} "
            f"(beta={float(_ramp_cfg_outer.get('red_ramp_beta', 2.5))}, "
            f"floor={red_ramp_min_steps:_}, bar={red_ramp_catch_bar}/upd/"
            f"{red_ramp_catch_window}upd, ceiling={red_ramp_max_steps:_}, "
            f"streak={red_ramp_catch_streak}, start_step={red_ramp_start_step:_})",
            flush=True,
        )
        _c0_str = f"{comms_c0_stage0[0]:.1f}|{comms_c0_stage0[1]:.1f}|{comms_c0_stage0[2]:.1f}/64" if comms_c0_stage0 is not None else "not yet captured"
        _c0_stage1_str = f"{comms_c0_stage1[0]:.1f}|{comms_c0_stage1[1]:.1f}|{comms_c0_stage1[2]:.1f}/64" if comms_c0_stage1 is not None else "not yet captured"
        print(
            f"[TRIPWIRE] comms-freeze tripwires: C0_stage0={_c0_str} "
            f"(stability-gated: median of first 5-consecutive-update window with all "
            f"transitions <=20%, window start >=3, forced at 35-40 if none stabilizes; "
            f"armed the update after capture) | "
            f"updates_since_resume={comms_updates_since_resume} | "
            f"captured_at_update={comms_c0_stage0_captured_at_update} | "
            f"C0_stage1={_c0_stage1_str} (same stability-gated algorithm, applied to "
            f"post-warmup stage-1 samples only) | "
            f"stage1_search_active={comms_stage1_c0_search_active} "
            f"stage1_search_updates={comms_stage1_c0_search_updates} "
            f"captured_at_update={comms_c0_stage1_captured_at_update} | "
            f"fast_streak={comms_fast_streak} (halt >=3 below 0.5*active_C0) | "
            f"drift_streak={comms_drift_streak} (halt >=15 below 0.75*active_C0) | "
            f"floor_streak={comms_floor_streak} (halt >=3 below 0.85*C0_stage0, stage 0 only) | "
            f"stage1_updates_elapsed={comms_stage1_updates_elapsed} "
            f"uncoordinated_seen={comms_stage1_uncoordinated_seen} (halt if still 0 after 10 updates in stage 1)",
            flush=True,
        )

    for ui in range(start_update, n_updates):
        update_key = update_keys[ui]
        step_keys = jax.random.split(update_key, T)
        step_idxs = jnp.arange(ui * T, (ui + 1) * T)

        if use_pmap:
            # step_keys needs to be shaped (n_devices, T, 2)
            step_keys_pmap = jax.random.split(update_key, n_devices * T).reshape(n_devices, T, -1)
            step_idxs_pmap = jnp.broadcast_to(step_idxs[None, :], (n_devices, T))
            scan_input_pmap = (step_keys_pmap, step_idxs_pmap)
            final_carry, rollout_data = sim_step_mapped(grid, b_pop, r_pop, b_carries, r_carries, scan_input_pmap)
            grid, b_pop, r_pop, b_carries, r_carries, _, _ = final_carry
            
            # Flatten rollout data across devices
            def flatten_pmap(x):
                return x.reshape(n_devices * T, *x.shape[2:])
            rollout_data = jax.tree_util.tree_map(flatten_pmap, rollout_data)
        else:
            scan_input = (step_keys, step_idxs)
            init_carry = (grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params)
            if ui == start_update:
                print(f"[JAX] lax.scan rollout starting (update {ui + 1})...", flush=True)
            _t_rollout0 = __import__("time").time()
            final_carry, rollout_data = lax.scan(sim_step_fn, init_carry, scan_input)
            if ui == start_update:
                _dt0 = __import__("time").time() - _t_rollout0
                print(
                    f"[JAX] lax.scan rollout done in {_dt0:.1f}s (update {ui + 1}) — PPO next.",
                    flush=True,
                )
            grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params = final_carry

        # final_carry unpack order: (b_pop, b_carries, r_pop, r_carries, ...)
        # b_carries shape: (max_pop, carry_dim)
        # Soft reset targets b_carries only — do not pass full carry tuple
        _md = 0.0
        if _medal_adr_enabled and _medal_adr_prob > 0.0:
            if use_pmap:
                def _pmap_wrapper(p, c):
                    return apply_medal_adr_carry_reset(p, c, _medal_adr_prob)
                b_pop, b_carries, reset_counts = jax.pmap(_pmap_wrapper)(b_pop, b_carries)
                _md = float(np.sum(reset_counts))
            else:
                b_pop, b_carries, reset_count = apply_medal_adr_carry_reset(b_pop, b_carries, _medal_adr_prob)
                _md = float(reset_count)
        # ── Free GPU: full (T×N) rollout must not sit on device during PPO backward
        rollout_data = _rollout_to_cpu(rollout_data)
        jax.clear_caches()

        # ── NaN debug after rollout ─────────────────────────────
        b_batch = rollout_data["blue"]
        has_nan_obs = bool(np.isnan(b_batch["obs"]).any())
        has_nan_vals = bool(np.isnan(b_batch["values"]).any())
        has_nan_logp = bool(np.isnan(b_batch["log_probs"]).any())
        has_nan_rew = bool(np.isnan(b_batch["rewards"]).any())
        if ui == 0:
            print(f"[DEBUG] Rollout data NaN: obs={has_nan_obs} vals={has_nan_vals} logp={has_nan_logp} rew={has_nan_rew}")

        # ── Comms-freeze tripwires (Cam, 2026-09-14; thresholds revised
        # 2026-09-14, Cam's own registered correction -- see
        # RESEARCH_PROTOCOL.md Part 2) ────────────────────────────────────
        # Computed every update, unconditionally -- not nested inside any
        # print-cadence gate, since "N consecutive updates" means consecutive
        # PPO updates. Evaluated against craft_ramp_stage_outer/active_outer
        # AS THEY STAND RIGHT NOW: the ratchet-advance decision runs later in
        # this same loop body, so at this point they still describe the
        # stage that produced the rollout just collected.
        #
        # C0 capture is stability-gated (Cam's second registered correction,
        # 2026-09-14), not a fixed window: the first fixed-window draft
        # (median of updates 3-7) landed on a transient for slot1 (sampled
        # 14|13|13|30|52 while still recovering toward its settled ~50s),
        # which would have let a 70% real collapse pass the fast trip
        # silently. C0 = median of the first 5 consecutive updates (window
        # start >= 3) in which every internal update-over-update transition
        # is <=20% relative change on all three slots -- forced at the
        # median of updates 35-40 if nothing stabilizes by then, tied to the
        # same 40-update budget as the stage-0 hard escape. Tripwires arm
        # the update after capture.
        if craft_ramp_active_outer:
            _tw_alive_mask = np.array(b_pop.alive).astype(bool)
            _tw_codes_now = None
            if "token_ids" in b_batch:
                _tw_tok_last = np.array(b_batch["token_ids"])[-1]
                if _tw_alive_mask.sum() > 0 and _tw_tok_last.ndim == 2 and _tw_tok_last.shape[1] == 3:
                    _tw_alive_tok = _tw_tok_last[_tw_alive_mask]
                    _tw_codes_now = (
                        len(np.unique(_tw_alive_tok[:, 0])),
                        len(np.unique(_tw_alive_tok[:, 1])),
                        len(np.unique(_tw_alive_tok[:, 2])),
                    )
            _tw_futile_uncoordinated_now = (
                int(np.asarray(b_batch["futile_uncoordinated"]).sum())
                if "futile_uncoordinated" in b_batch else 0
            )

            if _tw_codes_now is not None:
                comms_updates_since_resume += 1
                _u = comms_updates_since_resume

                if comms_c0_stage0 is None:
                    comms_sample_history.append(_tw_codes_now)

                    # Earliest possible 5-consecutive-update window with a
                    # start >= 3 ends at update 7 (window [3,4,5,6,7]).
                    if _u >= 7:
                        _win_start = _u - 4
                        # comms_sample_history is 0-indexed by (update - 1);
                        # window covers updates [_win_start .. _u].
                        _window = comms_sample_history[_win_start - 1: _u]
                        _transition_strs = []
                        _window_stable = True
                        for _t in range(4):
                            _old, _new = _window[_t], _window[_t + 1]
                            _changes = [
                                abs(_new[_s] - _old[_s]) / max(_old[_s], 1e-9) for _s in range(3)
                            ]
                            _t_pass = all(_c <= 0.20 for _c in _changes)
                            _window_stable = _window_stable and _t_pass
                            _tag = "OK" if _t_pass else "FAIL"
                            _transition_strs.append(
                                f"{_win_start + _t}->{_win_start + _t + 1}:{_tag}"
                                f"({','.join(f'{c:+.0%}' for c in _changes)})"
                            )
                        print(
                            f"[TRIPWIRE] C0_stage0 search: window upd{_win_start}-upd{_u} "
                            f"[{'; '.join(f'{s0}|{s1}|{s2}' for s0, s1, s2 in _window)}] "
                            f"| {' '.join(_transition_strs)} "
                            f"| {'STABLE' if _window_stable else 'not stable'}",
                            flush=True,
                        )
                        if _window_stable:
                            comms_c0_stage0 = tuple(
                                float(np.median([w[_s] for w in _window])) for _s in range(3)
                            )
                            comms_c0_stage0_captured_at_update = _u
                            print(
                                f"[TRIPWIRE] C0_stage0 captured (stable window upd{_win_start}-upd{_u}, "
                                f"5 updates, all transitions <=20%): {comms_c0_stage0[0]:.1f}|"
                                f"{comms_c0_stage0[1]:.1f}|{comms_c0_stage0[2]:.1f}/64 -- tripwires ARM at "
                                f"update {_u + 1}",
                                flush=True,
                            )

                    if comms_c0_stage0 is None and _u == 40:
                        _fallback_window = comms_sample_history[34:40]  # updates 35-40
                        comms_c0_stage0 = tuple(
                            float(np.median([w[_s] for w in _fallback_window])) for _s in range(3)
                        )
                        comms_c0_stage0_captured_at_update = 40
                        print(
                            "=" * 70,
                            flush=True,
                        )
                        print(
                            f"[TRIPWIRE] *** C0_stage0 FORCED at update 40 -- no stable window "
                            f"(all transitions <=20%) ever formed. Capturing median of "
                            f"updates 35-40: {comms_c0_stage0[0]:.1f}|{comms_c0_stage0[1]:.1f}|"
                            f"{comms_c0_stage0[2]:.1f}/64. This is a finding, not an "
                            f"inconvenience -- the channel never settled within the "
                            f"same 40-update budget as the stage-0 hard escape. "
                            f"Tripwires ARM at update 41. ***",
                            flush=True,
                        )
                        print(
                            "=" * 70,
                            flush=True,
                        )

                # 2026-09-20 (Cam): stage-1 baseline search -- same
                # stability-gated algorithm, applied only to samples
                # collected after the comms_lr_warmup ramp has fully
                # elapsed (the channel is deliberately still ramping in
                # during warmup, not settled; a "stable window" found there
                # would measure the ramp schedule, not the channel).
                if (
                    craft_ramp_stage_outer == 1
                    and _comms_unfreeze_at_update is None
                    and comms_c0_stage1 is None
                ):
                    comms_stage1_c0_search_active = True
                    comms_stage1_c0_search_updates += 1
                    _u1 = comms_stage1_c0_search_updates
                    comms_stage1_sample_history.append(_tw_codes_now)

                    if _u1 >= 7:
                        _win1_start = _u1 - 4
                        _window1 = comms_stage1_sample_history[_win1_start - 1: _u1]
                        _transition1_strs = []
                        _window1_stable = True
                        for _t in range(4):
                            _old, _new = _window1[_t], _window1[_t + 1]
                            _changes = [
                                abs(_new[_s] - _old[_s]) / max(_old[_s], 1e-9) for _s in range(3)
                            ]
                            _t_pass = all(_c <= 0.20 for _c in _changes)
                            _window1_stable = _window1_stable and _t_pass
                            _tag = "OK" if _t_pass else "FAIL"
                            _transition1_strs.append(
                                f"{_win1_start + _t}->{_win1_start + _t + 1}:{_tag}"
                                f"({','.join(f'{c:+.0%}' for c in _changes)})"
                            )
                        print(
                            f"[TRIPWIRE] C0_stage1 search: window upd{_win1_start}-upd{_u1} "
                            f"post-warmup "
                            f"[{'; '.join(f'{s0}|{s1}|{s2}' for s0, s1, s2 in _window1)}] "
                            f"| {' '.join(_transition1_strs)} "
                            f"| {'STABLE' if _window1_stable else 'not stable'}",
                            flush=True,
                        )
                        if _window1_stable:
                            comms_c0_stage1 = tuple(
                                float(np.median([w[_s] for w in _window1])) for _s in range(3)
                            )
                            comms_c0_stage1_captured_at_update = _u1
                            # Discard any partial streak accrued while
                            # stage 0's (now stale) baseline was still
                            # active -- a streak must be consecutive under
                            # ONE baseline, not span the switch.
                            comms_fast_streak = [0, 0, 0]
                            comms_drift_streak = [0, 0, 0]
                            print(
                                f"[TRIPWIRE] C0_stage1 captured (stable window "
                                f"upd{_win1_start}-upd{_u1} post-warmup, 5 updates, all "
                                f"transitions <=20%): {comms_c0_stage1[0]:.1f}|"
                                f"{comms_c0_stage1[1]:.1f}|{comms_c0_stage1[2]:.1f}/64 -- "
                                f"fast/drift tripwires now compare against the live-channel "
                                f"baseline from update {_u1 + 1} (streaks reset to isolate "
                                f"from stage-0 carryover)",
                                flush=True,
                            )

                    if comms_c0_stage1 is None and _u1 == 40:
                        _fallback1_window = comms_stage1_sample_history[34:40]
                        comms_c0_stage1 = tuple(
                            float(np.median([w[_s] for w in _fallback1_window])) for _s in range(3)
                        )
                        comms_c0_stage1_captured_at_update = 40
                        comms_fast_streak = [0, 0, 0]
                        comms_drift_streak = [0, 0, 0]
                        print("=" * 70, flush=True)
                        print(
                            f"[TRIPWIRE] *** C0_stage1 FORCED at update {_u1} post-warmup -- "
                            f"no stable window ever formed. Capturing median of the last 5 "
                            f"post-warmup updates: {comms_c0_stage1[0]:.1f}|"
                            f"{comms_c0_stage1[1]:.1f}|{comms_c0_stage1[2]:.1f}/64. This is a "
                            f"finding, not an inconvenience -- the live channel never settled "
                            f"within the same 40-update budget as the stage-0 search. "
                            f"Tripwires ARM at update {_u1 + 1}. ***",
                            flush=True,
                        )
                        print("=" * 70, flush=True)

                # Which baseline (if any) governs fast/drift this update:
                # stage 0's while in stage 0, stage 1's once stage 1's OWN
                # baseline has been captured -- never the stale stage-0
                # number against live stage-1 dynamics. Neither during the
                # blind window (mid-warmup, or post-warmup before a stage-1
                # baseline forms) -- mirrors how stage 0's own pre-capture
                # window already suppresses comparison entirely.
                _active_c0 = None
                _active_captured_at = None
                _active_u = None
                if craft_ramp_stage_outer == 0 and comms_c0_stage0 is not None:
                    _active_c0 = comms_c0_stage0
                    _active_captured_at = comms_c0_stage0_captured_at_update
                    _active_u = comms_updates_since_resume
                elif craft_ramp_stage_outer == 1 and comms_c0_stage1 is not None:
                    _active_c0 = comms_c0_stage1
                    _active_captured_at = comms_c0_stage1_captured_at_update
                    _active_u = comms_stage1_c0_search_updates

                # 2026-09-20 (Cam, self-caught): _tw_halt_reasons used to be
                # initialized (and the PRESSURE check below used to live)
                # INSIDE the `_active_c0 is not None` gate below -- meaning
                # the whole PRESSURE tripwire (comms_stage1_updates_elapsed,
                # entirely about futile_uncoordinated, nothing to do with
                # codes_active or any C0 baseline) silently stopped
                # incrementing for as long as comms_c0_stage1 hadn't
                # captured -- which on the run this was found on was every
                # update since the unfreeze. It did not change that run's
                # outcome (uncoordinated_seen had already latched True), but
                # the halt condition this counter exists to detect could
                # never have fired while gated this way. Initialized here,
                # unconditionally, so PRESSURE is checked on every real
                # stage-1 update regardless of whether a codes_active
                # baseline is active.
                _tw_halt_reasons = []

                if _active_c0 is not None and _active_u > _active_captured_at:
                    for _i in range(3):
                        _now = _tw_codes_now[_i]
                        _c0 = _active_c0[_i]
                        if _now < 0.5 * _c0:
                            comms_fast_streak[_i] += 1
                        else:
                            comms_fast_streak[_i] = 0
                        if _now < 0.75 * _c0:
                            comms_drift_streak[_i] += 1
                        else:
                            comms_drift_streak[_i] = 0
                        if (
                            craft_ramp_stage_outer == 0
                            and comms_c0_stage0 is not None
                            and _now < 0.85 * comms_c0_stage0[_i]
                        ):
                            comms_floor_streak[_i] += 1
                        else:
                            comms_floor_streak[_i] = 0

                        if comms_fast_streak[_i] >= 3:
                            _tw_halt_reasons.append(
                                f"FAST: slot{_i} codes_active={_now} < 0.5*C0_stage{craft_ramp_stage_outer}"
                                f"={0.5 * _c0:.1f} for {comms_fast_streak[_i]} consecutive updates"
                            )
                        if comms_drift_streak[_i] >= 15:
                            _tw_halt_reasons.append(
                                f"DRIFT: slot{_i} codes_active={_now} < 0.75*C0_stage{craft_ramp_stage_outer}"
                                f"={0.75 * _c0:.1f} for {comms_drift_streak[_i]} consecutive updates"
                            )
                        if comms_floor_streak[_i] >= 3:
                            _tw_halt_reasons.append(
                                f"STAGE-0 FLOOR: slot{_i} codes_active={_now} < "
                                f"0.85*C0_stage0={0.85 * comms_c0_stage0[_i]:.1f} for {comms_floor_streak[_i]} "
                                f"consecutive updates"
                            )

                # PRESSURE: deliberately OUTSIDE the _active_c0 gate above --
                # this check is about futile_uncoordinated, not about
                # codes_active or any C0 baseline, and must not stop
                # incrementing just because a baseline hasn't captured yet.
                if craft_ramp_stage_outer == 1:
                    comms_stage1_updates_elapsed += 1
                    if _tw_futile_uncoordinated_now > 0:
                        comms_stage1_uncoordinated_seen = True
                    if comms_stage1_updates_elapsed > 10 and not comms_stage1_uncoordinated_seen:
                        _tw_halt_reasons.append(
                            f"PRESSURE: futile_uncoordinated still 0 after "
                            f"{comms_stage1_updates_elapsed} updates in stage 1 -- "
                            f"the coordination pressure stage 1 is supposed to apply "
                            f"is not showing up"
                        )

                if _tw_halt_reasons:
                    print("=" * 70, flush=True)
                    print("[TRIPWIRE] HALT -- pre-registered comms-freeze tripwire fired:", flush=True)
                    for _r in _tw_halt_reasons:
                        print(f"[TRIPWIRE]   {_r}", flush=True)
                    _active_c0_str = (
                        f"{_active_c0[0]:.1f}|{_active_c0[1]:.1f}|{_active_c0[2]:.1f}/64"
                        if _active_c0 is not None else "not active (halted on a check independent of any C0 baseline)"
                    )
                    print(
                        f"[TRIPWIRE] C0_stage{craft_ramp_stage_outer} (active baseline)={_active_c0_str} | "
                        f"now={_tw_codes_now[0]}|{_tw_codes_now[1]}|{_tw_codes_now[2]}/64 | "
                        f"stage={craft_ramp_stage_outer} | ppo={ui}",
                        flush=True,
                    )
                    print("=" * 70, flush=True)
                    _tw_training_state = {
                        "craft_ramp_active": jnp.array(craft_ramp_active_outer, dtype=jnp.bool_),
                        "craft_ramp_stage": jnp.array(craft_ramp_stage_outer, dtype=jnp.int32),
                        "craft_ramp_start_step": jnp.array(craft_ramp_start_step, dtype=jnp.int32),
                        "craft_ramp_success_streak": jnp.array(craft_ramp_success_streak, dtype=jnp.int32),
                        "red_ramp_active": jnp.array(red_ramp_active_outer, dtype=jnp.bool_),
                        "red_ramp_start_step": jnp.array(red_ramp_start_step, dtype=jnp.int32),
                        "red_ramp_catch_streak": jnp.array(red_ramp_catch_streak, dtype=jnp.int32),
                        "red_curriculum_idx": jnp.array(red_curriculum_idx, dtype=jnp.int32),
                        "red_sustain_count": jnp.array(red_sustain_count, dtype=jnp.int32),
                        "comms_updates_since_resume": jnp.array(comms_updates_since_resume, dtype=jnp.int32),
                        "comms_sample_history": jnp.array(_pad_comms_history(comms_sample_history), dtype=jnp.int32),
                        # Dict keys unchanged (comms_c0_captured_at_update /
                        # comms_tripwire_c0) so old checkpoints still restore --
                        # these now specifically hold the stage-0 baseline.
                        "comms_c0_captured_at_update": jnp.array(comms_c0_stage0_captured_at_update, dtype=jnp.int32),
                        "comms_tripwire_c0": jnp.array(
                            comms_c0_stage0 if comms_c0_stage0 is not None else (-1.0, -1.0, -1.0), dtype=jnp.float32
                        ),
                        "comms_fast_streak": jnp.array(comms_fast_streak, dtype=jnp.int32),
                        "comms_drift_streak": jnp.array(comms_drift_streak, dtype=jnp.int32),
                        "comms_floor_streak": jnp.array(comms_floor_streak, dtype=jnp.int32),
                        "comms_stage1_updates_elapsed": jnp.array(comms_stage1_updates_elapsed, dtype=jnp.int32),
                        "comms_stage1_uncoordinated_seen": jnp.array(comms_stage1_uncoordinated_seen, dtype=jnp.bool_),
                        "comms_stage1_c0_search_active": jnp.array(comms_stage1_c0_search_active, dtype=jnp.bool_),
                        "comms_stage1_c0_search_updates": jnp.array(comms_stage1_c0_search_updates, dtype=jnp.int32),
                        "comms_stage1_sample_history": jnp.array(_pad_comms_history(comms_stage1_sample_history), dtype=jnp.int32),
                        "comms_c0_stage1_captured_at_update": jnp.array(comms_c0_stage1_captured_at_update, dtype=jnp.int32),
                        "comms_tripwire_c0_stage1": jnp.array(
                            comms_c0_stage1 if comms_c0_stage1 is not None else (-1.0, -1.0, -1.0), dtype=jnp.float32
                        ),
                    }
                    ckpt_mngr.save(ui, items={
                        "b_params": b_params, "r_params": r_params,
                        "training_state": _tw_training_state,
                    })
                    ckpt_mngr.wait_until_finished()
                    print(f"[TRIPWIRE] Emergency checkpoint saved at step {ui}.", flush=True)
                    if on_checkpoint_saved is not None:
                        # A checkpoint save that isn't committed (Modal
                        # Volumes: writes aren't guaranteed durable/visible
                        # to other containers until Volume.commit()) is
                        # not proof the emergency save survives the
                        # SystemExit about to happen. Commit before exiting.
                        on_checkpoint_saved()
                    raise SystemExit(1)

        # ── Red Curriculum Advancement ────────────────────────────
        surv_rate = float(b_pop.alive.sum()) / float(max_pop)

        if red_curriculum_idx < len(red_curriculum_stages) - 1:
            if surv_rate >= red_sustain_threshold:
                red_sustain_count += 1
                if red_sustain_count >= red_sustain_needed:
                    red_curriculum_idx += 1
                    red_sustain_count = 0
                    print(f"[CURRICULUM] Red floor advanced to {red_curriculum_stages[red_curriculum_idx]}")
            else:
                red_sustain_count = 0

        # Apply red reproduction at curriculum floor (outside JIT)
        if r_pop is not None:
            red_floor = red_curriculum_stages[red_curriculum_idx]
            repro_key_r, loop_key = jax.random.split(loop_key)
            r_pop = apply_auto_reproduce(
                r_pop, repro_key_r, gs,
                min_pop=red_floor,
                energy_thresh=float(config.get("repro_energy_thresh", 0.8)),
                energy_cost=float(config.get("repro_energy_cost", 0.4)),
            )

        # PPO update (not JIT — Python loop)
        _fwd_coef = float(config.get("fwd_coef", 0.05))
        _carry_fwd_coef = float(config.get("carry_fwd_coef", 0.05))
        _fwd_mb   = int(config.get("ppo_minibatch_size", 512))

        if ui == start_update:
            print(
                "  [JAX] Compiling Blue PPO backward (_minibatch_step) — "
                "often 3–10+ min on first update; do not Stop.",
                flush=True,
            )
        b_batch = rollout_data["blue"]
        _b_carries_np = np.asarray(b_batch["carries"])
        _b_actions_np = np.asarray(b_batch["actions"])
        _b_obs_np = np.asarray(b_batch["obs"])
        _b_alive_np = np.asarray(b_batch["alive"]) if "alive" in b_batch else None

        if _vqel_monologue:
            print("  [DEBUG] --- Blue VQEL Monologue Update ---")
            vqel_key, update_key = jax.random.split(update_key)
            _t_vqel0 = __import__("time").time()
            b_params, b_vqel_opt_state, b_metrics = vqel_monologue_update(
                b_params,
                b_vqel_opt_state,
                b_vqel_optimizer,
                vqel_monologue_apply,
                _b_obs_np,
                _b_carries_np,
                n_layers,
                alive_np=_b_alive_np,
                key=vqel_key,
                minibatch_size=_fwd_mb,
                recon_coef=_vqel_recon_coef,
                hash_penalty_coef=_vqel_hash_coef,
                vq_coef=_vqel_vq_coef,
            )
            if ui == start_update:
                print(
                    f"  [JAX] Blue VQEL monologue done in "
                    f"{__import__('time').time() - _t_vqel0:.1f}s",
                    flush=True,
                )
            _recon_mse = float(b_metrics.get("vqel_recon_mse", float("inf")))
            if _recon_mse < _graduate_recon_mse:
                _vqel_grad_streak += 1
            else:
                _vqel_grad_streak = 0
            if _vqel_grad_streak >= _graduate_consecutive:
                print("\n" + "=" * 70, flush=True)
                print(
                    "[JAX] VQEL MONOLOGUE GRADUATION ACHIEVED",
                    flush=True,
                )
                print(
                    f"  recon_mse={_recon_mse:.5f} < {_graduate_recon_mse} for "
                    f"{_graduate_consecutive} consecutive updates",
                    flush=True,
                )
                print(
                    "  → monologue_enabled=False | dialogue_signal_mode=hard | blue PPO resumed",
                    flush=True,
                )
                print("=" * 70 + "\n", flush=True)
                _vqel_monologue = False
                _dialogue_signal_mode = "hard"
                _vqel_grad_streak = 0
                if config.get("phase14_vqel") is not None:
                    config["phase14_vqel"]["monologue_enabled"] = False
                    config["phase14_vqel"]["dialogue_signal_mode"] = "hard"
                sim_step_fn = _rebuild_sim_step(n_layers)
        else:
            print("  [DEBUG] --- Blue PPO Update ---")
            _t_ppo0 = __import__("time").time()
            
            _base_vq_coef = float(config.get("vq_loss_coef", 0.1))
            # Phase 18 VQ Reconnection Warmup (0.5x for 20 updates)
            _vq_coef = _base_vq_coef * 0.5 if ui < start_update + 20 else _base_vq_coef
            
            # 2026-09-14 (Cam, Blocker 2): stage 0 is solo-satisfiable, so it
            # gives the sender encoder / codebook / receiver read path zero
            # reward gradient while the VQ commitment loss keeps pulling
            # unopposed -- a one-way ratchet toward encoder collapse. Freeze
            # that subtree (grads zeroed pre-optimizer, see
            # rl_jax.COMMS_SUBTREE_KEYS) for all of stage 0; unfreeze at the
            # stage-1 transition. b_batch was collected under whatever stage
            # was active during the rollout just finished, so gate on the
            # pre-advance craft_ramp_stage_outer (the ratchet decision below
            # only fires after this update).
            _freeze_comms = bool(craft_ramp_active_outer and craft_ramp_stage_outer == 0)
            # 2026-09-15 (Cam): linear warmup on the comms subtree's applied
            # update for the 10 PPO updates immediately after unfreeze --
            # 0.0 on the first unfrozen update, 1.0 (no-op) from update 10
            # onward. See the comment at the comms_lr_warmup scaling site in
            # rl_jax._minibatch_step for the mechanism this defuses.
            if _comms_unfreeze_at_update is None:
                _comms_lr_warmup = 1.0
            else:
                _updates_since_unfreeze = ui - _comms_unfreeze_at_update
                _comms_lr_warmup = min(1.0, max(0.0, _updates_since_unfreeze / _comms_lr_warmup_updates))
                if _updates_since_unfreeze >= _comms_lr_warmup_updates:
                    _comms_unfreeze_at_update = None  # warmup elapsed, stop computing/logging it
                else:
                    print(
                        f"  [COMMS-WARMUP] update {ui + 1}: warmup_factor={_comms_lr_warmup:.2f} "
                        f"effective_comms_lr={float(config.get('ppo_lr', 1e-4)) * _comms_lr_warmup:.2e} "
                        f"({_updates_since_unfreeze}/{_comms_lr_warmup_updates} updates since unfreeze)",
                        flush=True,
                    )
            b_params, b_opt_state, b_metrics = ppo_update(
                b_params, b_opt_state, b_optimizer, model_apply,
                b_batch, n_layers, update_key,
                clip_eps=float(config.get("ppo_clip_eps", config.get("ppo_clip", 0.2))),
                vf_coef=float(config.get("ppo_value_coef", 0.25)),
                ent_coef=float(config.get("ppo_entropy_coef", 0.02)),
                vq_coef=_vq_coef,
                minibatch_size=_fwd_mb,
                gamma=float(config.get("ppo_gamma", 0.99)),
                lam=float(config.get("ppo_gae_lam", 0.95)),
                team="blue",
                ignition_discount=float(config.get("phase16_5_enrichment", {}).get("ignition_discount", 0.1)),
                alarm_ent_coef=float(config.get("alarm_ent_coef", 0.0)),
                freeze_comms=_freeze_comms,
                comms_lr_warmup=_comms_lr_warmup,
            )
            if _freeze_comms and ui == start_update:
                print("  [CTD-RAMP] comms subtree FROZEN for stage 0 (gwt_comms_1, "
                      "head_signal_slot0/1/2, codebook_0/1/2, emb_nb) -- gradients "
                      "zeroed pre-optimizer, unfreezes at the stage-1 transition",
                      flush=True)
            if ui == start_update:
                print(
                    f"  [JAX] Blue PPO done in {__import__('time').time() - _t_ppo0:.1f}s",
                    flush=True,
                )
            _self_pred_coef = float(config.get("self_pred_coef", 0.1))
            fwd_key, update_key = jax.random.split(update_key)
            _b_energy_np = np.asarray(b_batch["energy"]) if "energy" in b_batch else None
            b_params, b_aux_opt_state, b_fwd_loss, b_carry_fwd_loss, b_sp_loss, b_sp_acc, b_conf_loss, b_conf_pred, b_proprio_loss = auxiliary_update(
                b_params, b_aux_opt_state, b_aux_optimizer, b_aux_apply_fn,
                _b_carries_np, _b_actions_np, _b_obs_np,
                _loc_env_start, _loc_env_end,
                _b_alive_np, fwd_key, minibatch_size=_fwd_mb,
                fwd_coef=_fwd_coef, carry_fwd_coef=_carry_fwd_coef,
                self_pred_coef=_self_pred_coef, conf_coef=_conf_coef,
                energy_np=_b_energy_np,
                proprio_coef=_proprio_coef,
                n_actions=int(config["n_actions"]),
            )
            b_metrics["fwd_loss"] = b_fwd_loss
            b_metrics["carry_fwd_loss"] = b_carry_fwd_loss
            b_metrics["sp_loss"] = b_sp_loss
            b_metrics["sp_acc"] = b_sp_acc
            b_metrics["conf_loss"] = b_conf_loss
            b_metrics["conf_pred"] = b_conf_pred
            b_metrics["proprio_loss"] = b_proprio_loss
            
            # Phase 18 early diagnostic: monitor actions 8-11 (Build, PickUp, Craft, UseTool)
            if ui < start_update + 200:
                _flat_actions = _b_actions_np[_b_alive_np] if _b_alive_np is not None else _b_actions_np
                _flat_actions = _flat_actions.reshape(-1)
                _tot = len(_flat_actions)
                _c_bld = np.sum(_flat_actions == 8)
                _c_pu = np.sum(_flat_actions == 9)
                _c_crf = np.sum(_flat_actions == 10)
                _c_use = np.sum(_flat_actions == 11)
                _tot_new = _c_bld + _c_pu + _c_crf + _c_use
                
                print(f"  [DEBUG] New Actions 8-11: {(_tot_new/_tot)*100:.1f}% [Bld:{(_c_bld/_tot)*100:.1f}% PU:{(_c_pu/_tot)*100:.1f}% Crf:{(_c_crf/_tot)*100:.1f}% Use:{(_c_use/_tot)*100:.1f}%]", flush=True)


        if config.get("vq_dead_code_reset", True) and "z_e" in b_batch:
            _dc_key, update_key = jax.random.split(update_key)
            _toks = jnp.asarray(b_batch["token_ids"])
            _ze = jnp.asarray(b_batch["z_e"])
            
            # Reshape (steps, agents, features) to (steps * agents, features)
            if _toks.ndim > 1:
                _toks = _toks.reshape(-1, _toks.shape[-1])
            if _ze.ndim > 1:
                _ze = _ze.reshape(-1, _ze.shape[-1])
                
            _alive = jnp.asarray(b_batch["alive"]).reshape(-1).astype(bool)

            # Task 4: fraction of the rollout window that had a live agent —
            # a thin pool (e.g. right after a predation spike) can't support
            # a vocab_size-code usage estimate; the reset call skips its
            # entire update rather than reading a noisy small-pool bincount.
            _alive_pool_frac = float(jnp.mean(_alive.astype(jnp.float32)))
            _dc_min_pool_frac = float(config.get("vq_dead_code_min_pool_frac", 0.25))
            _dc_window = int(config.get("vq_dead_code_window", 5))
            _dc_ema_decay = float(config.get("vq_dead_code_ema_decay", 0.8))

            _toks_alive = _toks[_alive]
            _ze_alive = _ze[_alive]

            # 2026-09-14 (Cam): codes grafted fresh onto this resume start with
            # usage_ema=dead_streak=0, and updates 1-7 post-resume are exactly
            # when usage is least stable (measured on this run: slot1 sampled
            # 14|13|13|30|52 while still settling). A code transiently unused
            # during that window would hit dead_streak_window at update 5-6
            # and reset for a "death" that's really just settling noise --
            # a mechanism that CAN seed a cascade (reset perturbs codebook,
            # strands more codes, resets more). MEASURED on this resume: 42
            # of 64 codes in slot1 sat past the threshold pre-stability, all
            # fired at once on arming, and codes_active recovered immediately
            # (13-19 -> 47/49/49/52, held). CONJECTURED, not measured: that
            # this same mechanism caused the ORIGINAL collapse -- that run
            # resumed from 2763, not 2541, a different starting vocabulary,
            # so it's a suggestive pair, not a controlled comparison. The
            # ladder also shows codes declining gradually across ~220
            # updates of ordinary training (2541->2763), far slower than a
            # resume cascade -- most likely two distinct mechanisms: a fast
            # resume cascade (now defused here) and a slow decay under
            # absent communication pressure (unaddressed, the actual subject
            # of the receiver-necessity thesis). Suppress the reset action
            # (not the bookkeeping) until the SAME condition that arms the
            # comms-freeze tripwires: C0 captured, one update later.
            _dc_armed = bool(
                comms_c0_stage0 is not None and comms_updates_since_resume > comms_c0_stage0_captured_at_update
            )
            _suppress_dead_code_reset = bool(craft_ramp_active_outer and not _dc_armed)

            if _toks_alive.ndim > 1 and _toks_alive.shape[-1] == 3:
                # Phase 18: 3 slots. z_e layout: 8D cont + 12D slot0 + 8D slot1 + 12D slot2
                _dc_key_0, _dc_key_1, _dc_key_2 = jax.random.split(_dc_key, 3)
                _vocab_size = int(config["vocab_size"])
                _dc_kwargs = dict(
                    alive_pool_frac=_alive_pool_frac, min_pool_frac=_dc_min_pool_frac,
                    dead_streak_window=_dc_window, ema_decay=_dc_ema_decay,
                    suppress_reset=_suppress_dead_code_reset,
                )
                b_params = dead_code_reset_codebook_params(
                    b_params, _toks_alive[:, 0], _ze_alive[:, 8:20], _vocab_size, _dc_key_0, "codebook_0", **_dc_kwargs
                )
                b_params = dead_code_reset_codebook_params(
                    b_params, _toks_alive[:, 1], _ze_alive[:, 20:28], _vocab_size, _dc_key_1, "codebook_1", **_dc_kwargs
                )
                b_params = dead_code_reset_codebook_params(
                    b_params, _toks_alive[:, 2], _ze_alive[:, 28:40], _vocab_size, _dc_key_2, "codebook_2", **_dc_kwargs
                )
            else:
                _toks_alive = _toks_alive.reshape(-1)
                _ze_alive = _ze_alive.reshape(-1, int(config["signal_dim"]))
                b_params = dead_code_reset_codebook_params(
                    b_params, _toks_alive, _ze_alive, int(config["vocab_size"]), _dc_key, "codebook",
                    alive_pool_frac=_alive_pool_frac, min_pool_frac=_dc_min_pool_frac,
                    dead_streak_window=_dc_window, ema_decay=_dc_ema_decay,
                    suppress_reset=_suppress_dead_code_reset,
                )

        if ui == start_update:
            import gc as _gc
            del _b_carries_np, _b_actions_np, _b_obs_np, _b_alive_np
            _gc.collect()
            print(
                "  [JAX] Blue aux + VQ reset done — Red PPO next "
                "(may compile ~3–10 min on first update; do not Stop).",
                flush=True,
            )

        print("  [DEBUG] --- Red PPO Update ---")
        r_batch = rollout_data["red"]
        _r_carries_np = np.asarray(r_batch["carries"])
        _r_actions_np = np.asarray(r_batch["actions"])
        _r_obs_np = np.asarray(r_batch["obs"])
        _r_alive_np   = np.asarray(r_batch["alive"]) if "alive" in r_batch else None
        _t_rppo0 = __import__("time").time()
        _r_ppo_apply = r_model_apply if _red_comms else model_apply
        # Phase 18.6: Red VQ is DECOUPLED from the PPO gradient (red_vq_loss_coef=0.0).
        # Red language is forged by reward_red_catch only (standing directive #8 — no
        # blind VQ shaping on red). Historically red's VQ loss was a static no-op; the
        # Phase 18 blue reconnection accidentally fed red's (pathological, 1.5e11) DCVQ
        # loss into its PPO objective, saturating the trunk clip at 2.0 and starving
        # red's actual policy learning. Red remains a lethal ecological pressure: its
        # policy/value train on catch reward and its proprio/SRL aux still run.
        # Reversible: set red_vq_loss_coef>0 (+ a one-time cold restart to heal the
        # collapsed codebook) if red comms is ever promoted back to a science target.
        r_params, r_opt_state, r_metrics = ppo_update(
            r_params, r_opt_state, r_optimizer, _r_ppo_apply,
            r_batch, n_layers, update_key,
            clip_eps=float(config.get("ppo_clip_eps", config.get("ppo_clip", 0.2))),
            vf_coef=float(config.get("ppo_value_coef", 0.25)),
            ent_coef=float(config.get("ppo_entropy_coef", 0.02)),
            vq_coef=float(config.get("red_vq_loss_coef", 0.0)),
            minibatch_size=_fwd_mb,
            gamma=float(config.get("ppo_gamma", 0.99)),
            lam=float(config.get("ppo_gae_lam", 0.95)),
            team="red",
            alarm_ent_coef=float(config.get("alarm_ent_coef", 0.0)),
        )
        if ui == start_update:
            print(
                f"  [JAX] Red PPO done in {__import__('time').time() - _t_rppo0:.1f}s",
                flush=True,
            )
        _r_energy_np = np.asarray(r_batch["energy"]) if "energy" in r_batch else None
        if not _red_comms:
            fwd_key, update_key = jax.random.split(update_key)
            r_params, r_aux_opt_state, r_fwd_loss, r_carry_fwd_loss, r_sp_loss, r_sp_acc, r_conf_loss, r_conf_pred, r_proprio_loss = auxiliary_update(
                r_params, r_aux_opt_state, r_aux_optimizer, r_aux_apply_fn,
                _r_carries_np, _r_actions_np, _r_obs_np,
                _loc_env_start, _loc_env_end,
                _r_alive_np, fwd_key, minibatch_size=_fwd_mb,
                fwd_coef=_fwd_coef, carry_fwd_coef=_carry_fwd_coef,
                self_pred_coef=_self_pred_coef, conf_coef=_conf_coef,
                energy_np=_r_energy_np,
                proprio_coef=_proprio_coef,
                n_actions=int(config["n_actions"]),
            )
            r_metrics["fwd_loss"] = r_fwd_loss
            r_metrics["carry_fwd_loss"] = r_carry_fwd_loss
            r_metrics["sp_acc"] = r_sp_acc
            r_metrics["conf_loss"] = r_conf_loss
            r_metrics["conf_pred"] = r_conf_pred
            r_metrics["proprio_loss"] = r_proprio_loss
        elif _proprio_coef > 0.0 and _r_energy_np is not None and r_red_aux_apply_fn is not None:
            _rprop_key, update_key = jax.random.split(update_key)
            
            # Phase 15.3 SRL setup: extract nb_sigs_target
            _layout = make_obs_layout(
                signal_dim=int(config["signal_dim"]),
                symbol_dim=int(config.get("symbol_dim", 16)),
                memory_slots=0,
                neighbor_k=int(config.get("red_neighbor_k", config["neighbor_k"])),
                local_cells=int(config.get("local_cells", 25)),
                env_channels=int(config.get("env_channels", 10)),
            )
            _r_nb_sigs_np = _r_obs_np[:, :, _layout.nb_sigs_start:_layout.nb_sigs_end]
            
            # Temporal Slicing on Axis 0 (lag = 5)
            _lag = 5
            _r_carries_t0 = _r_carries_np[:-_lag]
            _r_energy_lag = _r_energy_np[_lag:]
            _r_nb_sigs_target = _r_nb_sigs_np[:-_lag]
            _r_alive_lag = _r_alive_np[_lag:] if _r_alive_np is not None else None
            
            # Build obs_seq of shape (T-lag, N, lag, obs_dim)
            _T, _N, _D = _r_obs_np.shape
            _obs_seq = np.empty((_T - _lag, _N, _lag, _D), dtype=_r_obs_np.dtype)
            for l in range(_lag):
                _obs_seq[:, :, l, :] = _r_obs_np[l + 1 : _T - _lag + l + 1]

            r_params, r_aux_opt_state, r_proprio_loss, r_retention_loss = red_auxiliary_update(
                r_params,
                r_aux_opt_state,
                r_aux_optimizer,
                r_red_aux_apply_fn,
                _r_carries_t0,
                _obs_seq,
                _r_energy_lag,
                _r_nb_sigs_target,
                _r_alive_lag,
                key=_rprop_key,
                minibatch_size=_fwd_mb,
                proprio_coef=_proprio_coef,
                retention_coef=float(config.get("retention_coef", 0.1)),
            )
            r_metrics["proprio_loss"] = r_proprio_loss
            r_metrics["retention_loss"] = r_retention_loss

        if config.get("vq_dead_code_reset", True) and "z_e" in r_batch and not _red_comms:
            _dc_key, update_key = jax.random.split(update_key)
            _tok = jnp.asarray(r_batch["token_ids"]).reshape(-1)
            _ze = jnp.asarray(r_batch["z_e"]).reshape(-1, int(config["signal_dim"]))
            _alive = jnp.asarray(r_batch["alive"]).reshape(-1).astype(bool)
            _r_vocab = _red_vocab if _red_comms else int(config["vocab_size"])
            _cb_key = "red_codebook" if _red_comms else "codebook"
            r_params = dead_code_reset_codebook_params(
                r_params,
                _tok[_alive],
                _ze[_alive],
                _r_vocab,
                _dc_key,
                codebook_key=_cb_key,
                alive_pool_frac=float(jnp.mean(_alive.astype(jnp.float32))),
                min_pool_frac=float(config.get("vq_dead_code_min_pool_frac", 0.25)),
                dead_streak_window=int(config.get("vq_dead_code_window", 5)),
                ema_decay=float(config.get("vq_dead_code_ema_decay", 0.8)),
            )
            if _red_comms:
                r_params = ensure_predator_params(
                    model_red, r_params, _dc_key, red_hidden_d, obs_dim, n_layers
                )

        # ── Brain Vote (capacity-based, runs after PPO) ──────────
        _bv_ent = float(b_metrics.get('ppo_entropy', 0)) if isinstance(b_metrics, dict) else 0
        _bv_vf = float(b_metrics.get('ppo_vf_loss', 0)) if isinstance(b_metrics, dict) else 0
        brain_ent_history.append(_bv_ent)
        brain_vf_history.append(_bv_vf)
        b_signals_snap = np.array(jax.device_get(b_pop.signals))
        b_alive_snap = np.array(jax.device_get(b_pop.alive))
        _bv_sig_uniq = int(len(np.unique(b_signals_snap[b_alive_snap], axis=0))) if b_alive_snap.any() else 0
        brain_sig_diversity_history.append(_bv_sig_uniq)

        if (ui + 1) % brain_vote_interval_updates == 0 and len(brain_ent_history) >= brain_vote_window:
            w = brain_vote_window
            ent_window = brain_ent_history[-w:]
            vf_window = brain_vf_history[-w:]
            sig_window = brain_sig_diversity_history[-w:]

            ent_std = float(np.std(ent_window))
            vf_mean = float(np.mean(vf_window))
            sig_mean = float(np.mean(sig_window))
            sig_std = float(np.std(sig_window))

            ent_plateaued = ent_std < 0.05
            sig_plateaued = sig_std < 2.0
            vf_struggling = vf_mean > 0.1
            under_pressure = surv_rate < float(config.get("brain_vote_survival_threshold", 0.55))

            if ent_plateaued and sig_plateaued and vf_struggling and under_pressure and n_layers < brain_max_layers_val:
                n_layers += 1
                print(f"[BRAIN VOTE] Capacity saturated → {n_layers}L | ent_std={ent_std:.3f} sig_div={sig_mean:.0f}±{sig_std:.1f} vf={vf_mean:.3f} surv={surv_rate:.2f}")
                sim_step_fn = _rebuild_sim_step(n_layers)
            elif (ui + 1) % (brain_vote_interval_updates * 5) == 0:
                print(f"[BRAIN CHECK] {n_layers}L | ent_std={ent_std:.3f} sig_div={sig_mean:.0f} vf={vf_mean:.3f} surv={surv_rate:.2f}")

        # ── NaN debug after PPO update ──────────────────────────
        if ui == 0:
            flat_p = jax.tree_util.tree_leaves(b_params)
            has_nan_params_after = any(bool(jnp.isnan(p).any()) for p in flat_p)
            print(f"[DEBUG] Params NaN after PPO update: {has_nan_params_after}")

        # ── Telemetry ─────────────────────────────────────────────
        step_val = (ui + 1) * T
        _t_now = __import__('time').time()
        
        if step_val % 512 == 0 or T >= 512:
            # Timing
            elapsed = _t_now - _t_last
            steps_sec = T / max(elapsed, 1e-6)
            _t_last = _t_now
            
            # Pull rollout data to CPU
            b_act_all = np.array(rollout_data["blue"]["actions"])
            b_alive_all = np.array(rollout_data["blue"]["alive"]).astype(bool)
            b_rew_all = np.array(rollout_data["blue"]["rewards"])
            b_energy_all = np.array(rollout_data["blue"]["energy"])
            b_vals_all = np.array(rollout_data["blue"]["values"])
            
            # Population snapshot
            b_pop_np = jax.device_get(b_pop)
            b_alive_now = int(b_pop_np.alive.sum())
            r_alive_now = int(r_pop.alive.sum()) if r_pop is not None else 0
            # 2026-09-14 (Cam): catch_attempted is small-blue-only by design
            # (matches caught_small_potential in apply_catches -- big-green
            # catches are a completely separate, Strike-gated path). Reading
            # catch_attempts=0 for many updates says nothing about red's
            # sensing if the small-blue population it's measured against has
            # collapsed to near-zero (agents matured to big-green) -- that
            # would be an instrument reporting a clean number about an empty
            # set, the same class of defect as the rel_spread retirement.
            # Split and normalize so the denominator is visible, not assumed.
            small_blue_alive_now = int((b_pop_np.alive & ~b_pop_np.is_big_green).sum())
            big_green_alive_now = int((b_pop_np.alive & b_pop_np.is_big_green).sum())
            
            # Action distribution (N=stay, S, E, W, stay=0)
            alive_actions = b_act_all[b_alive_all]
            if len(alive_actions) > 0:
                act_counts = np.bincount(alive_actions, minlength=config["n_actions"])
                act_pct = act_counts / act_counts.sum() * 100
                act_str = f"N={act_pct[1]:.0f}% S={act_pct[2]:.0f}% E={act_pct[3]:.0f}% W={act_pct[4]:.0f}% Stay={act_pct[0]:.0f}% Strk={act_pct[5]:.0f}% Push={act_pct[6]:.0f}% Grd={act_pct[7]:.0f}%"
                if len(act_pct) > 8:
                    act_str += f" Bld={act_pct[8]:.0f}%"
                if len(act_pct) > 11:
                    act_str += f" PU={act_pct[9]:.0f}% Crf={act_pct[10]:.0f}% Use={act_pct[11]:.0f}%"
            else:
                act_str = "no alive agents"

            red_act_str = "N/A"
            red_codes_str = "N/A"
            r_alive_mask_final = np.array(r_pop.alive) if r_pop is not None else np.zeros(0, dtype=bool)
            if "actions" in rollout_data["red"]:
                r_act_all = np.array(rollout_data["red"]["actions"])
                r_alive_all = np.array(rollout_data["red"]["alive"]).astype(bool)
                r_alive_actions = r_act_all[r_alive_all]
                if len(r_alive_actions) > 0:
                    r_counts = np.bincount(r_alive_actions, minlength=config["n_actions"])
                    r_pct = r_counts / r_counts.sum() * 100
                    red_act_str = (
                        f"N={r_pct[1]:.0f}% S={r_pct[2]:.0f}% E={r_pct[3]:.0f}% W={r_pct[4]:.0f}% Stay={r_pct[0]:.0f}% "
                        f"Strk={r_pct[5]:.0f}% Push={r_pct[6]:.0f}% Grd={r_pct[7]:.0f}%"
                    )
                    if len(r_pct) > 8:
                        red_act_str += f" Bld={r_pct[8]:.0f}%"
                    if len(r_pct) > 11:
                        red_act_str += f" PU={r_pct[9]:.0f}% Crf={r_pct[10]:.0f}% Use={r_pct[11]:.0f}%"
                else:
                    red_act_str = "no alive reds"
            if _red_comms and "token_ids" in rollout_data["red"]:
                r_tok_last = np.array(rollout_data["red"]["token_ids"])[-1]
                if r_alive_mask_final.sum() > 0:
                    red_codes_str = (
                        f"{len(np.unique(r_tok_last[r_alive_mask_final]))}/{_red_vocab}"
                    )
            
            # Energy stats
            alive_mask_final = b_pop_np.alive
            if alive_mask_final.sum() > 0:
                alive_energy = b_pop_np.energy[alive_mask_final]
                e_mean, e_std = float(alive_energy.mean()), float(alive_energy.std())
                alive_ages = b_pop_np.ages[alive_mask_final].astype(float)
                age_mean, age_max = float(alive_ages.mean()), float(alive_ages.max())
            else:
                e_mean, e_std, age_mean, age_max = 0, 0, 0, 0
            
            # Value accuracy (is value head learning?)
            val_mean = float(b_vals_all[b_alive_all].mean()) if b_alive_all.any() else 0
            ret_mean = float(b_metrics.get("returns_mean", 0)) if isinstance(b_metrics, dict) else 0
            
            # VQ codebook usage (unique token_ids among alive at final rollout step)
            vq_codes_str = "N/A"
            if "token_ids" in rollout_data["blue"]:
                b_tok_last = np.array(rollout_data["blue"]["token_ids"])[-1]
                if alive_mask_final.sum() > 0:
                    alive_toks = b_tok_last[alive_mask_final]
                    if alive_toks.ndim == 2 and alive_toks.shape[1] == 3:
                        u0 = len(np.unique(alive_toks[:, 0]))
                        u1 = len(np.unique(alive_toks[:, 1]))
                        u2 = len(np.unique(alive_toks[:, 2]))
                        vq_codes_str = f"{u0}|{u1}|{u2}/64"
                    else:
                        vq_codes_str = f"{len(np.unique(alive_toks))}/{config.get('vocab_size', 64)}"

            # Signal vocabulary compression (k-means clusters with >2% occupancy)
            b_signals_np = np.array(b_pop_np.signals)
            active_clusters_str = "N/A"
            if alive_mask_final.sum() >= 16:
                alive_sigs = b_signals_np[alive_mask_final]
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(n_clusters=16, n_init="auto", random_state=42)
                labels = km.fit_predict(alive_sigs)
                counts = np.bincount(labels, minlength=16)
                thresh = max(1, int(0.02 * len(alive_sigs)))
                active_clusters = int((counts > thresh).sum())
                active_clusters_str = f"{active_clusters}/16"
            
            # Alarm Rate
            alarm_rate = 0.0
            if "alarm_out" in rollout_data["blue"]:
                _alarm_out_all = np.array(rollout_data["blue"]["alarm_out"]) # (T, N, 2)
                if b_alive_all.sum() > 0:
                    _alarms_triggered = (_alarm_out_all[:, :, 1] > 0.5) & b_alive_all
                    alarm_rate = float(_alarms_triggered.sum() / b_alive_all.sum())
            if isinstance(b_metrics, dict):
                b_metrics["Alarm_Rate"] = alarm_rate
            
            # Reward breakdown
            rew_alive = b_rew_all[b_alive_all]
            rew_mean = float(rew_alive.mean()) if len(rew_alive) > 0 else 0
            
            # Print concise dashboard
            print(f"\n{'='*70}")
            print(f"[step {step_val:>7}] {steps_sec:.0f} steps/sec | blue={b_alive_now} red={r_alive_now} | ppo={ui+1}")
            print(f"  Actions (blue): {act_str}")
            if "imagined_action" in rollout_data["blue"] and _img_gate:
                im_act_all = np.array(rollout_data["blue"]["imagined_action"])
                im_alive_actions = im_act_all[b_alive_all]
                if len(im_alive_actions) > 0:
                    im_counts = np.bincount(im_alive_actions, minlength=config["n_actions"])
                    im_pct = im_counts / im_counts.sum() * 100
                    im_act_str = f"N={im_pct[1]:.0f}% S={im_pct[2]:.0f}% E={im_pct[3]:.0f}% W={im_pct[4]:.0f}% Stay={im_pct[0]:.0f}% Strk={im_pct[5]:.0f}% Push={im_pct[6]:.0f}% Grd={im_pct[7]:.0f}%"
                    if len(im_pct) > 8:
                        im_act_str += f" Bld={im_pct[8]:.0f}%"
                    if len(im_pct) > 11:
                        im_act_str += f" PU={im_pct[9]:.0f}% Crf={im_pct[10]:.0f}% Use={im_pct[11]:.0f}%"
                    print(f"  Actions (imag): {im_act_str}")
            if _red_comms:
                red_vq_val = float(r_metrics.get("ppo_vq_loss", float("nan")))
                red_ent_val = float(r_metrics.get("ppo_entropy", float("nan")))
                red_ret_val = float(r_metrics.get("retention_loss", float("nan")))
                print(f"  Actions (red):  {red_act_str}")
                # Phase 18.6: red VQ is decoupled from the gradient — this line is
                # diagnostic only (a collapsed/huge value is expected and harmless;
                # red hunts via policy, not comms).
                print(
                    f"  RedVQ(decoupled): loss={red_vq_val:.4g} | red_codes_active={red_codes_str} "
                    f"| red_entropy={red_ent_val:.4f} | RedAux: ret_loss={red_ret_val:.4f}"
                )
            print(f"  Energy:  mean={e_mean:.3f} std={e_std:.3f} | Age: mean={age_mean:.0f} max={age_max:.0f}")
            vf_loss = float(b_metrics.get('ppo_vf_loss', 0)) if isinstance(b_metrics, dict) else 0
            ent_val = float(b_metrics.get('ppo_entropy', 0)) if isinstance(b_metrics, dict) else 0
            clip_frac = float(b_metrics.get('ppo_clip_frac', 0)) if isinstance(b_metrics, dict) else 0
            fwd_loss_val = float(b_metrics.get('fwd_loss', float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            carry_fwd_val = float(b_metrics.get('carry_fwd_loss', float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            sp_acc_val   = float(b_metrics.get('sp_acc',   float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            conf_loss_val = float(b_metrics.get('conf_loss', float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            conf_pred_val = float(b_metrics.get('conf_pred', float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            vq_loss_val  = float(b_metrics.get('ppo_vq_loss', float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            _carry_last = np.asarray(rollout_data["blue"]["carries"][-1])
            _alive_rollout = np.asarray(rollout_data["blue"]["alive"][-1]).astype(bool)
            if _alive_rollout.sum() > 0:
                _carry_alive = _carry_last[_alive_rollout]
                carry_rank = int(np.linalg.matrix_rank(_carry_alive, tol=0.1))
                carry_entropy = float(
                    -np.sum(np.abs(_carry_alive) * np.log(np.abs(_carry_alive) + 1e-8))
                )
            else:
                carry_rank = 0
                carry_entropy = float('nan')
            print(f"  Values:  mean={val_mean:.4f} | VF_loss={vf_loss:.4f} | Clip={clip_frac:.3f}")
            print(f"  Reward:  mean={rew_mean:.4f} | Entropy: {ent_val:.4f} | Alarm_Rate: {alarm_rate:.3f}")
            _aux_conf = (
                f" | conf_loss={conf_loss_val:.6f} conf_pred={conf_pred_val:.6f}"
                if _conf_enabled
                else ""
            )
            proprio_loss_val = float(b_metrics.get("proprio_loss", float("nan"))) if isinstance(b_metrics, dict) else float("nan")
            _aux_proprio = (
                f" | proprio_loss={proprio_loss_val:.4f}"
                if np.isfinite(proprio_loss_val)
                else ""
            )
            print(
                f"  AuxLoss: fwd_env={fwd_loss_val:.4f} | carry_fwd={carry_fwd_val:.4f} "
                f"(↓0.05–0.1) | self_pred_acc={sp_acc_val:.3f}{_aux_conf}{_aux_proprio} | "
                f"carry_rank={carry_rank} | carry_H={carry_entropy:.2f}"
            )
            if _vqel_monologue:
                _vr = float(b_metrics.get("vqel_recon_mse", float("nan")))
                _vh = float(b_metrics.get("vqel_hash_penalty", float("nan")))
                _vt = float(b_metrics.get("vqel_total_loss", float("nan")))
                print(
                    f"  VQEL: recon_mse={_vr:.5f} | hash_penalty={_vh:.5f} "
                    f"| total={_vt:.5f} | grad_streak={_vqel_grad_streak}/"
                    f"{_graduate_consecutive} (target < {_graduate_recon_mse})"
                )
            elif _dialogue_signal_mode == "hard":
                print("  VQEL: GRADUATED — hard z_q dialogue broadcast | blue PPO active")
            blue_caught_rollout = 0
            if "blue_caught" in rollout_data["blue"]:
                blue_caught_rollout = int(np.asarray(rollout_data["blue"]["blue_caught"]).sum())
            # 2026-09-14 (Cam): catch attempts vs conversions, from instrumentation
            # not config -- distinguishes "red can't find blue" (attempts=0) from
            # "the catch path is broken" (attempts>0, conversions=0).
            catch_attempted_rollout = 0
            if "catch_attempted" in rollout_data["blue"]:
                catch_attempted_rollout = int(np.asarray(rollout_data["blue"]["catch_attempted"]).sum())
            barrier_sum_val = 0
            if "barrier_sum" in rollout_data["blue"]:
                barrier_sum_val = float(np.asarray(rollout_data["blue"]["barrier_sum"]).mean())
            print(f"  VQ: loss={vq_loss_val:.2e} | codes_active={vq_codes_str} | clusters={active_clusters_str}")
            medal_str = ""
            if _medal_adr_enabled and _medal_adr_prob > 0.0:
                medal_str = f" | expert_dropouts={int(_md)}"
            print(
                f"  Ecology: blue_caught={blue_caught_rollout} this rollout | "
                f"catch_attempts={catch_attempted_rollout} "
                f"(conversion={blue_caught_rollout / max(1, catch_attempted_rollout):.1%}) | "
                f"pop_split=small:{small_blue_alive_now}|big_green:{big_green_alive_now} | "
                f"attempts/small_blue={catch_attempted_rollout / max(1, small_blue_alive_now):.2f} "
                f"(catch_attempted is small-blue-only by design -- 0 attempts against a "
                f"near-empty small-blue count measures nothing about red's sensing) | "
                f"red_floor={red_curriculum_stages[red_curriculum_idx]} "
                f"sustain={red_sustain_count}/{red_sustain_needed} | brain={n_layers}L{medal_str} | barrier_sum={barrier_sum_val:.1f}"
            )
            _n_craft_success = 0
            _n_futile_uncoordinated = 0
            _n_futile_wrong_mats = 0
            _n_futile_empty = 0
            if "craft_success" in rollout_data["blue"]:
                _n_craft_success = int(np.asarray(rollout_data["blue"]["craft_success"]).sum())
            if "futile_uncoordinated" in rollout_data["blue"]:
                _n_futile_uncoordinated = int(np.asarray(rollout_data["blue"]["futile_uncoordinated"]).sum())
            if "futile_wrong_mats" in rollout_data["blue"]:
                _n_futile_wrong_mats = int(np.asarray(rollout_data["blue"]["futile_wrong_mats"]).sum())
            if "futile_empty" in rollout_data["blue"]:
                _n_futile_empty = int(np.asarray(rollout_data["blue"]["futile_empty"]).sum())
            _n_craft_total = (
                _n_craft_success + _n_futile_uncoordinated + _n_futile_wrong_mats + _n_futile_empty
            )
            # 2026-09-14 (Cam): success/attempts has an agent-controlled
            # denominator -- blue can depress "rate" by spamming Craft harder
            # without materials (observed live: success flat 40->35,
            # futile_wrong_mats nearly tripled 2875->7432, rate fell 1.3%->
            # 0.4%). Retired as the stage-advance gate on that mechanism, not
            # because it failed a threshold (same standard as the rel_spread
            # retirement) -- kept here as diagnostic context only, alongside
            # the per-capita figure that replaces it below.
            print(
                f"  Crafting: success={_n_craft_success} | futile_uncoordinated={_n_futile_uncoordinated} | "
                f"futile_wrong_mats={_n_futile_wrong_mats} | futile_empty={_n_futile_empty} | "
                f"rate={_n_craft_success / max(1, _n_craft_total):.1%} (diagnostic only, not the "
                f"stage-advance gate) | pop_frac={_n_craft_success / max(1, b_alive_now):.1%} "
                f"(this IS the gate, vs living blue={b_alive_now})"
            )

            # ── CtD competence ramp: per-update ratchet decision ─────
            # Floor + sustained bar + hard ceiling, mirroring the red-curriculum
            # pattern above. Evaluated every update (this block runs every ui,
            # since step_val % 512 == 0 or T >= 512 is unconditionally true at
            # T=512) -- not just on a print cadence.
            _red_shaping_mean = float(np.asarray(rollout_data["red"]["red_shaping"]).mean())
            _craft_stage_label = (
                f"stage={craft_ramp_stage_outer}/{len(CRAFT_RAMP_STAGE_UNITS) - 1}"
                if craft_ramp_active_outer else "off"
            )
            print(
                f"  CTD-RAMP: craft_active={craft_ramp_active_outer} ({_craft_stage_label}, "
                f"streak={craft_ramp_success_streak}/{craft_ramp_success_window}) | "
                f"red_active={red_ramp_active_outer} "
                f"(streak={red_ramp_catch_streak}/{red_ramp_catch_window}, "
                f"shaping_mean={_red_shaping_mean:.5f})"
            )

            if craft_ramp_active_outer:
                # Two capped stages (Cam's correction, 2026-09-14): the same
                # floor + sustained bar + hard ceiling shape is reapplied at
                # each stage, clock reset to the stage's own start. Advancing
                # from the last stage ratchets the ramp off entirely (full
                # recipe); advancing from an earlier stage moves to the next.
                _steps_since_craft_ramp = (ui + 1) * T - craft_ramp_start_step
                _updates_since_craft_ramp = (ui + 1) - (craft_ramp_start_step // T)
                # Per-capita bar (Cam's third registered correction,
                # 2026-09-14), replacing success/attempts: attempts is an
                # agent-controlled denominator -- an agent that tries more
                # looks less competent, which cannot measure competence by
                # construction. success/living_population isn't controllable
                # the same way. Default 10%, sustained 3 updates -- at
                # blue~195 that's ~20 successes, already well inside the
                # observed 35-40/update.
                _craft_success_frac_of_pop = _n_craft_success / max(1, b_alive_now)
                if _craft_success_frac_of_pop >= craft_ramp_success_bar:
                    craft_ramp_success_streak += 1
                else:
                    craft_ramp_success_streak = 0
                _craft_advance_now = False
                _craft_advance_reason = None
                if (_steps_since_craft_ramp >= craft_ramp_min_steps
                        and craft_ramp_success_streak >= craft_ramp_success_window):
                    _craft_advance_now = True
                    _craft_advance_reason = "bar met"
                elif (craft_ramp_stage_outer == 0
                        and _updates_since_craft_ramp >= craft_ramp_stage0_max_updates):
                    _craft_advance_now = True
                    _craft_advance_reason = "STAGE-0 HARD ESCAPE -- bar never cleared within 40 updates"
                    print(
                        f"[CTD-RAMP] *** Stage-0 hard escape fired: {_updates_since_craft_ramp} "
                        f"updates since stage 0 began, bar ({craft_ramp_success_bar:.0%} of living "
                        f"population sustained {craft_ramp_success_window} updates) never cleared "
                        f"(current streak={craft_ramp_success_streak}, this update's "
                        f"success={_n_craft_success}/{b_alive_now} living blue = "
                        f"{_craft_success_frac_of_pop:.1%}). Advancing to stage 1 anyway -- a "
                        f"curriculum phase with no exit is a trap, and stage 0 is the phase "
                        f"with no communication pressure. This is a finding, not an "
                        f"inconvenience. ***",
                        flush=True,
                    )
                elif _steps_since_craft_ramp >= craft_ramp_max_steps:
                    _craft_advance_now = True
                    _craft_advance_reason = "CEILING FORCED -- bar never met"
                    print(
                        f"[CTD-RAMP] *** Crafting ramp stage {craft_ramp_stage_outer} forced to "
                        f"advance at hard ceiling ({craft_ramp_max_steps:_} steps) WITHOUT "
                        f"meeting its bar ({craft_ramp_success_bar:.0%} of living population "
                        f"sustained {craft_ramp_success_window} updates). This is a finding, not an "
                        f"inconvenience. ***",
                        flush=True,
                    )
                if _craft_advance_now:
                    _leaving_stage0 = (craft_ramp_stage_outer == 0)
                    if craft_ramp_stage_outer < len(CRAFT_RAMP_STAGE_UNITS) - 1:
                        craft_ramp_stage_outer += 1
                        craft_ramp_start_step = (ui + 1) * T
                        craft_ramp_success_streak = 0
                        grid = grid.replace(craft_ramp_stage=jnp.array(craft_ramp_stage_outer, dtype=jnp.int32))
                        print(
                            f"[CTD-RAMP] Crafting ramp advanced to stage {craft_ramp_stage_outer} "
                            f"({CRAFT_RAMP_STAGE_UNITS[craft_ramp_stage_outer]} unit(s), "
                            f"{_craft_advance_reason}) at ppo={ui + 1}, step={(ui + 1) * T:_}",
                            flush=True,
                        )
                        if _leaving_stage0:
                            _comms_unfreeze_at_update = ui + 1
                            print(
                                "  [CTD-RAMP] comms subtree UNFROZEN (stage-1 transition) -- "
                                "gwt_comms_1, head_signal_slot0/1/2, codebook_0/1/2, emb_nb "
                                "resume normal gradient updates next PPO step, ramping in over "
                                f"{_comms_lr_warmup_updates} updates (2026-09-15, Cam: defuses "
                                "the Adam bias-correction spring on the first post-freeze "
                                "gradient -- see rl_jax._minibatch_step)",
                                flush=True,
                            )
                    else:
                        craft_ramp_active_outer = False
                        grid = grid.replace(craft_ramp_active=jnp.array(False, dtype=jnp.bool_))
                        print(
                            f"[CTD-RAMP] Crafting ramp ratcheted OFF ({_craft_advance_reason}) "
                            f"at ppo={ui + 1}, step={(ui + 1) * T:_}",
                            flush=True,
                        )

            if red_ramp_active_outer:
                _steps_since_red_ramp = (ui + 1) * T - red_ramp_start_step
                if blue_caught_rollout >= red_ramp_catch_bar:
                    red_ramp_catch_streak += 1
                else:
                    red_ramp_catch_streak = 0
                _red_ratchet_now = False
                _red_ratchet_reason = None
                if (_steps_since_red_ramp >= red_ramp_min_steps
                        and red_ramp_catch_streak >= red_ramp_catch_window):
                    _red_ratchet_now = True
                    _red_ratchet_reason = "bar met"
                elif _steps_since_red_ramp >= red_ramp_max_steps:
                    _red_ratchet_now = True
                    _red_ratchet_reason = "CEILING FORCED -- bar never met"
                    print(
                        f"[CTD-RAMP] *** Red shaping ramp forced off at hard ceiling "
                        f"({red_ramp_max_steps:_} steps) WITHOUT meeting its bar "
                        f"({red_ramp_catch_bar}/update sustained {red_ramp_catch_window} "
                        f"updates). This is a finding, not an inconvenience. ***",
                        flush=True,
                    )
                if _red_ratchet_now:
                    red_ramp_active_outer = False
                    grid = grid.replace(red_ramp_active=jnp.array(False, dtype=jnp.bool_))
                    print(
                        f"[CTD-RAMP] Red shaping ramp ratcheted OFF ({_red_ratchet_reason}) "
                        f"at ppo={ui + 1}, step={(ui + 1) * T:_}",
                        flush=True,
                    )

            if bool((_p9 or {}).get("imagination_gating_enabled", False)):
                im_agree_val = float("nan")
                conf_gate_val = float("nan")
                if "imagination_agree" in rollout_data["blue"]:
                    _im_a = np.asarray(rollout_data["blue"]["imagination_agree"])
                    _cg = np.asarray(rollout_data["blue"]["conf_gate_imagine_frac"])
                    _alive_roll = np.asarray(rollout_data["blue"]["alive"]).astype(bool)
                    if _alive_roll.any():
                        im_agree_val = float(_im_a[_alive_roll].mean()) * 100.0
                        conf_gate_val = float(_cg[_alive_roll].mean()) * 100.0
                _mult = float((_p9 or {}).get("confidence_multiplier", 1.0))
                print(
                    f"  EpistemicGate: imagination_agree={im_agree_val:.1f}% "
                    f"| conf_gate_imagine_frac={conf_gate_val:.1f}% "
                    f"(mult={_mult} batch-relative; K={int((_p9 or {}).get('imagination_k', 5))})"
                )
                if "imagination_metabolic_cost" in rollout_data["blue"]:
                    _mc = np.asarray(rollout_data["blue"]["imagination_metabolic_cost"])
                    _alive_mc = np.asarray(rollout_data["blue"]["alive"]).astype(bool)
                    if _alive_mc.any():
                        _delta = float((_p9 or {}).get("imagination_metabolic_delta", 0.0005))
                        _k = int((_p9 or {}).get("imagination_k", 5))
                        print(
                            f"  MetabolicTax: mean_cost={float(_mc[_alive_mc].mean()):.5f} "
                            f"(delta={_delta} K={_k} max={_delta * _k:.4f}/think)"
                        )

            # ── VQ health alert gate (H3, Phase 18.5; scoped to blue in 18.6) ──────
            # Cheap insurance against severance-class bugs on the *trained* channel.
            # Blue VQ is gradient-trained, so a negative loss (wired to the wrong
            # tensor) or a near-total codebook collapse is a real bug — make it LOUD so
            # we notice on update 2, not update 2000. Red VQ is decoupled from the
            # gradient (Phase 18.6), so its telemetry is diagnostic only and is NOT
            # alerted on (a collapsed/huge RedVQ is expected and harmless).
            def _min_codes_active(codes_str):
                try:
                    return min(int(x) for x in codes_str.split("/")[0].split("|"))
                except Exception:
                    return None
            _alerts = []
            if np.isfinite(vq_loss_val) and vq_loss_val < 0.0:
                _alerts.append(f"blue VQ loss < 0 ({vq_loss_val:.2f}) — VQ likely wired to wrong network output")
            _bc_min = _min_codes_active(vq_codes_str)
            if _bc_min is not None and _bc_min < 4:
                _alerts.append(f"blue codes_active collapsed ({vq_codes_str})")
                
            if any(np.isnan(x) for x in [vq_loss_val, vf_loss, ent_val, val_mean, rew_mean]):
                _alerts.append("NaN detected in blue PPO metrics (check gradients/logit-masking invariant)")
                
            if b_alive_now < 5:
                _alerts.append(f"blue population collapsed (N={b_alive_now})")

            for _a in _alerts:
                print(f"  [ALERT] {_a}", flush=True)
            print(f"{'='*70}\n")

        # ── Evolutionary Distillation (CPU, Outer Loop) ─────────
        if config.get("distill_enabled", False):
            _distill_interval = int(config.get("distill_interval", 10000))
            _distill_updates = max(1, _distill_interval // T)
            if (ui + 1) % _distill_updates == 0:
                print(f"  [step {(ui+1)*T}] DISTILL — population")
                keep_frac = float(config.get("distill_keep_frac", 0.05))
                noise_std = float(config.get("distill_noise_std", 0.1))
                
                if use_pmap:
                    print("  [WARN] Distillation is currently disabled when use_pmap=True.")
                else:
                    # Blue Distill
                    b_pop_np = jax.tree_util.tree_map(lambda x: np.array(x), b_pop)
                    n_alive = np.sum(b_pop_np.alive)
                    if n_alive >= 10:
                        ages_masked = np.where(b_pop_np.alive, b_pop_np.ages, -999999)
                        sorted_idx = np.argsort(ages_masked)[::-1]
                        n_keep = max(1, int(n_alive * keep_frac))
                        elites = sorted_idx[:n_keep]
                        to_kill = sorted_idx[n_keep:]
                        
                        b_pop_np.alive[to_kill] = False
                        b_pop_np.ages[to_kill] = 0
                        b_pop_np.carries[to_kill] = 0.0
                        b_pop_np.energy[to_kill] = 0.0
                        
                        dead_idx = np.where(~b_pop_np.alive)[0]
                        for slot in dead_idx:
                            parent_idx = np.random.choice(elites)
                            b_pop_np.positions[slot] = np.random.randint(0, config["grid_size"], size=2)
                            b_pop_np.ages[slot] = 0
                            b_pop_np.alive[slot] = True
                            b_pop_np.team[slot] = b_pop_np.team[parent_idx]
                            b_pop_np.carries[slot] = b_pop_np.carries[parent_idx] + np.random.normal(0, noise_std, size=b_pop_np.carries[parent_idx].shape).astype(np.float32)
                            b_pop_np.signals[slot] = 0.0
                            b_pop_np.energy[slot] = 1.0
                            
                        b_pop = b_pop.replace(
                            positions=jnp.array(b_pop_np.positions),
                            ages=jnp.array(b_pop_np.ages),
                            alive=jnp.array(b_pop_np.alive),
                            team=jnp.array(b_pop_np.team),
                            carries=jnp.array(b_pop_np.carries),
                            signals=jnp.array(b_pop_np.signals),
                            energy=jnp.array(b_pop_np.energy),
                        )

                    # Red Distill
                    r_pop_np = jax.tree_util.tree_map(lambda x: np.array(x), r_pop)
                    n_alive_r = np.sum(r_pop_np.alive)
                    if n_alive_r >= 10:
                        ages_masked = np.where(r_pop_np.alive, r_pop_np.ages, -999999)
                        sorted_idx = np.argsort(ages_masked)[::-1]
                        n_keep = max(1, int(n_alive_r * keep_frac))
                        elites = sorted_idx[:n_keep]
                        to_kill = sorted_idx[n_keep:]
                        
                        r_pop_np.alive[to_kill] = False
                        r_pop_np.ages[to_kill] = 0
                        r_pop_np.carries[to_kill] = 0.0
                        r_pop_np.energy[to_kill] = 0.0
                        
                        dead_idx = np.where(~r_pop_np.alive)[0]
                        for slot in dead_idx:
                            parent_idx = np.random.choice(elites)
                            r_pop_np.positions[slot] = np.random.randint(0, config["grid_size"], size=2)
                            r_pop_np.ages[slot] = 0
                            r_pop_np.alive[slot] = True
                            r_pop_np.team[slot] = r_pop_np.team[parent_idx]
                            r_pop_np.carries[slot] = r_pop_np.carries[parent_idx] + np.random.normal(0, noise_std, size=r_pop_np.carries[parent_idx].shape).astype(np.float32)
                            r_pop_np.signals[slot] = 0.0
                            r_pop_np.energy[slot] = 1.0
                            
                        r_pop = r_pop.replace(
                            positions=jnp.array(r_pop_np.positions),
                            ages=jnp.array(r_pop_np.ages),
                            alive=jnp.array(r_pop_np.alive),
                            team=jnp.array(r_pop_np.team),
                            carries=jnp.array(r_pop_np.carries),
                            signals=jnp.array(r_pop_np.signals),
                            energy=jnp.array(r_pop_np.energy),
                        )


        # ── Corpus Writing (CPU) ───────────────────────────────
        b_pos_all = np.array(rollout_data["blue"]["positions"])
        b_sig_all = np.array(rollout_data["blue"]["signals"])
        b_tok_all = np.array(rollout_data["blue"]["token_ids"])
        b_act_all = np.array(rollout_data["blue"]["actions"])
        b_alive_all = np.array(rollout_data["blue"]["alive"])
        b_energy_all = np.array(rollout_data["blue"]["energy"])
        b_obs_all = np.array(rollout_data["blue"]["obs"])
        r_pos_all = np.array(rollout_data["red"]["positions"])
        r_alive_all = np.array(rollout_data["red"]["alive"])
        r_sig_all = np.array(rollout_data["red"]["signals"])
        r_tok_all = np.array(rollout_data["red"]["token_ids"])
        r_act_all = np.array(rollout_data["red"]["actions"])
        r_energy_all = np.array(rollout_data["red"]["energy"])
        r_carry_fwd_all = np.array(rollout_data["red"]["carries"])
        b_steps_since_dropout_all = np.array(rollout_data["blue"]["steps_since_dropout"])

        # loc_env is the 4th block in b_obs (env_channels channels).
        # AUDIT_SEP2026.md Finding 3: this used to hand-roll the offset as
        # `6 + neighbor_k*signal_dim + 25*symbol_dim`, assuming a 6-dim
        # own_state and no neighbor-alarm block — both stale since Phase 17
        # (own_state_dim) and Phase 17.5 (nb_alarms). It was off by 28 columns
        # against the current config, silently reading `contested`/`wood`/
        # `stone` instead of `resource`/`blue_bg`/`barrier`. Use the same
        # canonical `_loc_env_start` (from `make_obs_layout`, computed once at
        # the top of this function from the live config) that every other
        # obs-layout consumer in this file already uses, instead of a second,
        # independently-drifting formula.
        env_channels = int(config.get("env_channels", 15))
        idx_offset = _loc_env_start
        idx_resource = idx_offset + (12 * env_channels) + 3  # 12th cell (center of 5x5), channel 3 = resource

        # Phase 17 NPMI spatial correlates
        adj_cells = [7, 11, 12, 13, 17] # N, W, Center, E, S in 5x5 patch
        idx_adj_bg = [idx_offset + (c * env_channels) + 8 for c in adj_cells]
        idx_adj_barrier = [idx_offset + (c * env_channels) + 9 for c in adj_cells]

        gs_val = int(config["grid_size"])
        start_step = ui * T
        for t in range(T):
            global_step = start_step + t
            if global_step % config.get("corpus_every_n_steps", 20) != 0:
                continue

            b_pos = b_pos_all[t]
            r_pos = r_pos_all[t]
            b_alive = b_alive_all[t]
            r_alive = r_alive_all[t]

            # ── Blue corpus (independent of red population) ─────────────
            if np.any(b_alive):
                alive_idx = np.where(b_alive)[0]
                n_alive = len(alive_idx)
                red_dist = np.full(n_alive, 999.0)
                red_bear = np.zeros(n_alive)
                is_scout = np.zeros(n_alive, dtype=bool)

                active_red_idx = np.where(r_alive)[0]
                if len(active_red_idx) > 0:
                    pos_b_alive = b_pos[alive_idx]
                    pos_r_active = r_pos[active_red_idx]
                    diff = np.abs(pos_b_alive[:, None, :] - pos_r_active[None, :, :])
                    diff = np.minimum(diff, gs_val - diff)
                    dists = np.max(diff, axis=-1)
                    min_idx = np.argmin(dists, axis=1)
                    red_dist = dists[np.arange(n_alive), min_idx]
                    is_scout = red_dist <= _alarm_range
                    nearest_red_pos = pos_r_active[min_idx]
                    dy = nearest_red_pos[:, 0] - pos_b_alive[:, 0]
                    dx = nearest_red_pos[:, 1] - pos_b_alive[:, 1]
                    dy = np.where(
                        dy > gs_val / 2, dy - gs_val,
                        np.where(dy < -gs_val / 2, dy + gs_val, dy),
                    )
                    dx = np.where(
                        dx > gs_val / 2, dx - gs_val,
                        np.where(dx < -gs_val / 2, dx + gs_val, dx),
                    )
                    red_bear = np.degrees(np.arctan2(dy, dx)) % 360.0

                pos_b_alive = b_pos[alive_idx]
                diff_b = np.abs(pos_b_alive[:, None, :] - pos_b_alive[None, :, :])
                diff_b = np.minimum(diff_b, gs_val - diff_b)
                dists_b = np.max(diff_b, axis=-1)
                np.fill_diagonal(dists_b, 9999)
                nb_count = np.sum(dists_b <= 4, axis=1)
                loc_res = b_obs_all[t, alive_idx, idx_resource]

                nb_scout_lag1 = np.full((n_alive, _corpus_sig_dim), np.nan, dtype=np.float32)
                nb_scout_dist_lag1 = np.full(n_alive, np.nan, dtype=np.float32)
                nb_scout_token_lag1 = np.full(n_alive, -1, dtype=object)
                if _lag1_scout_pos is not None and len(_lag1_scout_pos) > 0:
                    pos_b_alive_f = pos_b_alive.astype(np.float32)
                    sp2 = _lag1_scout_pos.astype(np.float32)
                    dd2 = np.abs(pos_b_alive_f[:, None, :] - sp2[None, :, :])
                    dd2 = np.minimum(dd2, gs_val - dd2)
                    sc2 = np.maximum(dd2[:, :, 0], dd2[:, :, 1])
                    within_mask = sc2 <= _alarm_range
                    w = within_mask.astype(np.float32)
                    w_sum = w.sum(axis=1)
                    has_donor = w_sum > 0
                    if has_donor.any():
                        nb_scout_lag1[has_donor] = (
                            w[has_donor] @ _lag1_scout_sig
                        ) / w_sum[has_donor, None]
                        nb_scout_dist_lag1[has_donor] = (
                            w[has_donor] @ _lag1_scout_dist
                        ) / w_sum[has_donor]
                        for i in np.where(has_donor)[0]:
                            toks = _lag1_scout_tok[within_mask[i]].astype(np.int64)
                            toks_s0 = np.asarray(toks)
                            if toks_s0.ndim > 1:
                                majority_toks = []
                                for slot_idx in range(toks_s0.shape[1]):
                                    majority_toks.append(int(np.bincount(toks_s0[:, slot_idx].astype(int)).argmax()))
                                nb_scout_token_lag1[i] = majority_toks
                            else:
                                nb_scout_token_lag1[i] = int(np.bincount(toks_s0.astype(int)).argmax())
                adj_red = red_dist <= 2.0
                adj_bg = np.sum(b_obs_all[t, alive_idx][:, idx_adj_bg], axis=-1) > 0.5
                adj_barrier = np.sum(b_obs_all[t, alive_idx][:, idx_adj_barrier], axis=-1) > 0.5

                can_see_recipe = b_obs_all[t, alive_idx][:, 12] > 0.5
                inv_wood = b_obs_all[t, alive_idx][:, 6] > 0.5
                inv_stone = b_obs_all[t, alive_idx][:, 7] > 0.5
                inv_flint = b_obs_all[t, alive_idx][:, 9] > 0.5
                inv_clay = b_obs_all[t, alive_idx][:, 10] > 0.5
                inv_vine = b_obs_all[t, alive_idx][:, 11] > 0.5
                
                inventory = np.where(inv_wood, 0,
                            np.where(inv_stone, 1,
                            np.where(inv_flint, 2,
                            np.where(inv_clay, 3,
                            np.where(inv_vine, 4, -1)))))
                
                cur_rec = np.array(rollout_data["blue"]["current_recipe"][t])
                recipe_id = int(cur_rec[0] + cur_rec[1]*10 + cur_rec[2]*100 + cur_rec[3]*1000 + cur_rec[4]*10000)

                corpus_writer.maybe_record(
                    step=global_step,
                    alive_idx=alive_idx,
                    signals=b_sig_all[t],
                    token_ids=b_tok_all[t],
                    actions=b_act_all[t],
                    is_scout=is_scout,
                    nearest_red_dist=red_dist,
                    nearest_red_bear=red_bear,
                    local_resource=loc_res,
                    own_energy=b_energy_all[t, alive_idx],
                    neighbor_count=nb_count,
                    norm_x=b_pos[alive_idx, 0] / gs_val,
                    norm_y=b_pos[alive_idx, 1] / gs_val,
                    nb_scout_sig_lag1=nb_scout_lag1,
                    nb_scout_dist_lag1=nb_scout_dist_lag1,
                    nb_scout_token_lag1=nb_scout_token_lag1,
                    adj_bg=adj_bg,
                    adj_barrier=adj_barrier,
                    adj_red=adj_red,
                    can_see_recipe=can_see_recipe,
                    current_recipe_id=recipe_id,
                    inventory=inventory,
                    steps_since_dropout=b_steps_since_dropout_all[t, alive_idx],
                    craft_ramp_active=bool(rollout_data["blue"]["craft_ramp_active"][t]),
                    red_ramp_active=bool(rollout_data["blue"]["red_ramp_active"][t]),
                )
                if is_scout.any():
                    pos_b_alive_f = b_pos[alive_idx].astype(np.float32)
                    _lag1_scout_pos = pos_b_alive_f[is_scout].copy()
                    _lag1_scout_sig = b_sig_all[t, alive_idx[is_scout]].copy()
                    _lag1_scout_dist = red_dist[is_scout].copy()
                    _lag1_scout_tok = b_tok_all[t, alive_idx[is_scout]].copy()
                else:
                    _lag1_scout_pos = None
                    _lag1_scout_sig = None
                    _lag1_scout_dist = None
                    _lag1_scout_tok = None

            # ── Red corpus (runs even if blue locally extinct) ───────────
            if corpus_writer_red is not None and np.any(r_alive):
                alive_idx_r = np.where(r_alive)[0]
                n_alive_r = len(alive_idx_r)
                blue_dist = np.full(n_alive_r, 999.0)
                blue_bear = np.zeros(n_alive_r)
                is_hunter = np.zeros(n_alive_r, dtype=bool)

                active_blue_idx = np.where(b_alive)[0]
                if len(active_blue_idx) > 0:
                    pos_r_alive = r_pos[alive_idx_r]
                    pos_b_active = b_pos[active_blue_idx]
                    diff_rb = np.abs(pos_r_alive[:, None, :] - pos_b_active[None, :, :])
                    diff_rb = np.minimum(diff_rb, gs_val - diff_rb)
                    dists_rb = np.max(diff_rb, axis=-1)
                    min_idx_b = np.argmin(dists_rb, axis=1)
                    blue_dist = dists_rb[np.arange(n_alive_r), min_idx_b]
                    is_hunter = blue_dist <= _hunt_range
                    nearest_blue_pos = pos_b_active[min_idx_b]
                    dy = nearest_blue_pos[:, 0] - pos_r_alive[:, 0]
                    dx = nearest_blue_pos[:, 1] - pos_r_alive[:, 1]
                    dy = np.where(
                        dy > gs_val / 2, dy - gs_val,
                        np.where(dy < -gs_val / 2, dy + gs_val, dy),
                    )
                    dx = np.where(
                        dx > gs_val / 2, dx - gs_val,
                        np.where(dx < -gs_val / 2, dx + gs_val, dx),
                    )
                    blue_bear = np.degrees(np.arctan2(dy, dx)) % 360.0

                pos_r_alive = r_pos[alive_idx_r]
                diff_r = np.abs(pos_r_alive[:, None, :] - pos_r_alive[None, :, :])
                diff_r = np.minimum(diff_r, gs_val - diff_r)
                dists_r = np.max(diff_r, axis=-1)
                np.fill_diagonal(dists_r, 9999)
                nb_count_r = np.sum(dists_r <= 4, axis=1)

                nb_hunter_lag1 = np.full((n_alive_r, _corpus_sig_dim), np.nan, dtype=np.float32)
                nb_hunter_dist_lag1 = np.full(n_alive_r, np.nan, dtype=np.float32)
                nb_hunter_token_lag1 = np.full(n_alive_r, -1, dtype=np.int32)
                if _lag1_hunter_pos is not None and len(_lag1_hunter_pos) > 0:
                    pos_r_alive_f = pos_r_alive.astype(np.float32)
                    hp2 = _lag1_hunter_pos.astype(np.float32)
                    dd_h = np.abs(pos_r_alive_f[:, None, :] - hp2[None, :, :])
                    dd_h = np.minimum(dd_h, gs_val - dd_h)
                    sc_h = np.maximum(dd_h[:, :, 0], dd_h[:, :, 1])
                    within_h = sc_h <= _hunt_range
                    w_h = within_h.astype(np.float32)
                    w_h_sum = w_h.sum(axis=1)
                    has_hunter_donor = w_h_sum > 0
                    if has_hunter_donor.any():
                        nb_hunter_lag1[has_hunter_donor] = (
                            w_h[has_hunter_donor] @ _lag1_hunter_sig
                        ) / w_h_sum[has_hunter_donor, None]
                        nb_hunter_dist_lag1[has_hunter_donor] = (
                            w_h[has_hunter_donor] @ _lag1_hunter_dist
                        ) / w_h_sum[has_hunter_donor]
                        for i in np.where(has_hunter_donor)[0]:
                            toks = _lag1_hunter_tok[within_h[i]].astype(np.int64)
                            toks_s0 = np.asarray(toks)
                            if toks_s0.ndim > 1:
                                toks_s0 = toks_s0[:, 0]
                            nb_hunter_token_lag1[i] = int(np.bincount(toks_s0.astype(int)).argmax())

                corpus_writer_red.maybe_record_red(
                    step=global_step,
                    alive_idx=alive_idx_r,
                    signals=r_sig_all[t],
                    token_ids=r_tok_all[t],
                    actions=r_act_all[t],
                    is_hunter=is_hunter,
                    nearest_blue_dist=blue_dist,
                    nearest_blue_bear=blue_bear,
                    own_energy=r_energy_all[t, alive_idx_r],
                    neighbor_count=nb_count_r,
                    nb_hunter_sig_lag1=nb_hunter_lag1,
                    nb_hunter_dist_lag1=nb_hunter_dist_lag1,
                    nb_hunter_token_lag1=nb_hunter_token_lag1,
                    carry_fwd=r_carry_fwd_all[t, alive_idx_r],
                    craft_ramp_active=bool(rollout_data["red"]["craft_ramp_active"][t]),
                    red_ramp_active=bool(rollout_data["red"]["red_ramp_active"][t]),
                )
                if is_hunter.any():
                    pos_r_alive_f = r_pos[alive_idx_r].astype(np.float32)
                    _lag1_hunter_pos = pos_r_alive_f[is_hunter].copy()
                    _lag1_hunter_sig = r_sig_all[t, alive_idx_r[is_hunter]].copy()
                    _lag1_hunter_dist = blue_dist[is_hunter].copy()
                    _lag1_hunter_tok = r_tok_all[t, alive_idx_r[is_hunter]].copy()
                else:
                    _lag1_hunter_pos = None
                    _lag1_hunter_sig = None
                    _lag1_hunter_dist = None
                    _lag1_hunter_tok = None

        corpus_writer.flush_to_disk()
        if corpus_writer_red is not None:
            corpus_writer_red.flush_to_disk()

        # Convert metrics to Python floats for logging (skip non-scalars)
        metrics_py = {}
        for k, v in b_metrics.items():
            try:
                metrics_py[k] = float(v)
            except (TypeError, ValueError):
                pass  # skip non-scalar arrays
        all_metrics.append(metrics_py)

        # ── Checkpoint (params + opt states only — avoids custom pytree issues) ─
        ckpt_interval_steps = int(config.get("checkpoint_interval", 2000))
        ckpt_interval_updates = max(1, ckpt_interval_steps // T)
        if (ui + 1) % ckpt_interval_updates == 0:
            # training_state (Cam, 2026-09-14): CtD ramp progress and red
            # curriculum state, so a resume continues rather than restarting
            # these clocks -- same durability usage_ema/dead_streak already
            # have for VQ codebooks.
            training_state = {
                "craft_ramp_active": jnp.array(craft_ramp_active_outer, dtype=jnp.bool_),
                "craft_ramp_stage": jnp.array(craft_ramp_stage_outer, dtype=jnp.int32),
                "craft_ramp_start_step": jnp.array(craft_ramp_start_step, dtype=jnp.int32),
                "craft_ramp_success_streak": jnp.array(craft_ramp_success_streak, dtype=jnp.int32),
                "red_ramp_active": jnp.array(red_ramp_active_outer, dtype=jnp.bool_),
                "red_ramp_start_step": jnp.array(red_ramp_start_step, dtype=jnp.int32),
                "red_ramp_catch_streak": jnp.array(red_ramp_catch_streak, dtype=jnp.int32),
                "red_curriculum_idx": jnp.array(red_curriculum_idx, dtype=jnp.int32),
                "red_sustain_count": jnp.array(red_sustain_count, dtype=jnp.int32),
                "comms_updates_since_resume": jnp.array(comms_updates_since_resume, dtype=jnp.int32),
                "comms_sample_history": jnp.array(_pad_comms_history(comms_sample_history), dtype=jnp.int32),
                # Dict keys unchanged (comms_c0_captured_at_update /
                # comms_tripwire_c0) so old checkpoints still restore --
                # these now specifically hold the stage-0 baseline.
                "comms_c0_captured_at_update": jnp.array(comms_c0_stage0_captured_at_update, dtype=jnp.int32),
                "comms_tripwire_c0": jnp.array(
                    comms_c0_stage0 if comms_c0_stage0 is not None else (-1.0, -1.0, -1.0), dtype=jnp.float32
                ),
                "comms_fast_streak": jnp.array(comms_fast_streak, dtype=jnp.int32),
                "comms_drift_streak": jnp.array(comms_drift_streak, dtype=jnp.int32),
                "comms_floor_streak": jnp.array(comms_floor_streak, dtype=jnp.int32),
                "comms_stage1_updates_elapsed": jnp.array(comms_stage1_updates_elapsed, dtype=jnp.int32),
                "comms_stage1_uncoordinated_seen": jnp.array(comms_stage1_uncoordinated_seen, dtype=jnp.bool_),
                "comms_stage1_c0_search_active": jnp.array(comms_stage1_c0_search_active, dtype=jnp.bool_),
                "comms_stage1_c0_search_updates": jnp.array(comms_stage1_c0_search_updates, dtype=jnp.int32),
                "comms_stage1_sample_history": jnp.array(_pad_comms_history(comms_stage1_sample_history), dtype=jnp.int32),
                "comms_c0_stage1_captured_at_update": jnp.array(comms_c0_stage1_captured_at_update, dtype=jnp.int32),
                "comms_tripwire_c0_stage1": jnp.array(
                    comms_c0_stage1 if comms_c0_stage1 is not None else (-1.0, -1.0, -1.0), dtype=jnp.float32
                ),
            }
            ckpt_state = {
                "b_params": b_params,
                "r_params": r_params,
                "training_state": training_state,
            }
            ckpt_mngr.save(ui + 1, items=ckpt_state)
            ckpt_mngr.wait_until_finished()
            print(f"  [CKPT] Saved step {(ui+1)*T}")
            if on_checkpoint_saved is not None:
                # 2026-09-14: ckpt_mngr.save()+wait_until_finished() writes to
                # the container's local view of the mount; on Modal Volumes
                # that is NOT guaranteed durable or visible to any other
                # container (including a future resume) until Volume.commit()
                # runs. Without this, "[CKPT] Saved step N" was a success
                # message with no proof behind it (Rule 13) -- every periodic
                # save since the last natural run-completion could vanish on
                # any interruption (stop, crash, preemption) that never
                # reaches the one commit() call at the very end of
                # scripts/modal_app.py's train(). Commit on the same cadence
                # as the checkpoint itself.
                on_checkpoint_saved()
                print(f"  [CKPT] Committed step {(ui+1)*T}", flush=True)

        if (ui + 1) % 10 == 0 or ui == 0:
            alive_count = int(b_pop.alive.sum())
            nan_dbg = f"has_nan={metrics_py.get('has_nan', 0):.0f}"
            if metrics_py.get('has_nan', 0) > 0.5:
                nan_dbg += (f" aL={metrics_py.get('nan_action_logits',0):.0f}"
                           f" vP={metrics_py.get('nan_values_pred',0):.0f}"
                           f" oL={metrics_py.get('nan_old_log_probs',0):.0f}"
                           f" adv={metrics_py.get('nan_advantages',0):.0f}"
                           f" ratio={metrics_py.get('nan_ratio',0):.0f}")
            if _vqel_monologue:
                print(
                    f"  VQEL#{ui+1} pop={alive_count} "
                    f"recon={metrics_py.get('vqel_recon_mse', float('nan')):.4f} "
                    f"hash={metrics_py.get('vqel_hash_penalty', float('nan')):.4f} "
                    f"total={metrics_py.get('vqel_total_loss', float('nan')):.4f} "
                    f"streak={_vqel_grad_streak}/{_graduate_consecutive} {nan_dbg}"
                )
            else:
                print(
                    f"  PPO#{ui+1} pop={alive_count} "
                    f"pg={metrics_py.get('ppo_pg_loss', 0.0):.4f} "
                    f"vf={metrics_py.get('ppo_vf_loss', 0.0):.4f} "
                    f"ent={metrics_py.get('ppo_entropy', 0.0):.4f} {nan_dbg}"
                )
            print(
                f"       (Red) pg={float(r_metrics.get('ppo_pg_loss', 0.0)):.4f} "
                f"vf={float(r_metrics.get('ppo_vf_loss', 0.0)):.4f} "
                f"ent={float(r_metrics.get('ppo_entropy', 0.0)):.4f}"
            )

    corpus_writer.close()
    if corpus_writer_red is not None:
        corpus_writer_red.close()
    return {"blue": b_params, "red": r_params}, {"metrics": all_metrics}


# ── CLI entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config not found: {config_path}, using defaults")
        cfg = DEFAULT_CONFIG

    print("[JAX] Starting simulation...")
    from jax_sim.train_entry import run_simulation as run_simulation_fresh

    final_params, metrics = run_simulation_fresh(cfg, seed=42, n_steps=1024)
    print("[JAX] Done!")
    print(f"Final metrics: {metrics}")
