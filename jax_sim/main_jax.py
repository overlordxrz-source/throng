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
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress INFO/WARN
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")  # disable autotune spam
# Leave headroom for PPO backward after rollout scan (default JAX grabs ~90% of VRAM).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

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
    write_to_grid, decay_grid, get_local_patches, get_neighbour_signals,
    generate_puzzle_nodes, update_puzzle_grid, decay_puzzle_timeout, check_puzzle_solved,
    generate_resource_patches, generate_shelter_spots, generate_contested_nodes,
    update_scent_trails,
)
from communication.analysis import SignalCorpusWriter
from jax_sim.population_jax import (
    PopState, init_population, kill_agents, update_memory_buffer, apply_mind_meld,
    apply_auto_reproduce
)
from jax_sim.network_jax import (
    AgentNetworkJax,
    AUX_HEAD_KEYS,
    ensure_aux_head_params,
    dead_code_reset_codebook_params,
    init_agent_params,
    make_model_apply,
    params_apply_variables,
    sanitize_agent_params,
)
from jax_sim.rl_jax import compute_gae, ppo_loss, create_optimizer, ppo_update, auxiliary_update
from agents.network_torch import compute_obs_dim_torch, compute_fwd_env_dim, loc_env_flat_bounds
from jax_sim.observations_jax import RED_SENSE_API_VERSION, build_observations_jax


# ── Config defaults ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "grid_size": 128,
    "population_size": 500,
    "red_population_size": 75,
    "neighbor_k": 6,
    "local_obs_radius": 2,
    "signal_dim": 32,
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
    cfg.setdefault("max_pop", cfg.get("population_size", 500))
    cfg.setdefault("max_pop_red", cfg.get("red_population_size", 75))
    cfg.setdefault("hidden_dim", cfg.get("agent_hidden_dim", 256))
    cfg.setdefault("n_heads", cfg.get("brain_n_heads", 4))
    cfg.setdefault("n_layers", cfg.get("n_layers", 4))
    cfg.setdefault("vocab_size", cfg.get("vocab_size", 64))
    cfg.setdefault("memory_slots", cfg.get("memory_buffer_size", 20))
    cfg.setdefault("ppo_rollout_steps", cfg.get("ppo_rollout_steps", 512))
    return cfg


# ── Rollout → CPU (free GPU before PPO backward) ───────────────────────────

def _rollout_to_cpu(rollout_data: Dict) -> Dict:
    """Move scan outputs off GPU so PPO backward has room (obs alone ~2.4GB)."""
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jax.Array, jnp.ndarray)) else x,
        rollout_data,
    )


# ── Single simulation step (for scan) ──────────────────────────────────────

def make_sim_step(config: Dict, model: AgentNetworkJax, model_apply=None):
    """Factory: returns a jittable step function.
    Params are passed through carry to avoid stale closure capture."""
    if model_apply is None:
        model_apply = make_model_apply(model)
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
    _puzzle_reward = float(config.get("puzzle_reward", 5.0))
    _energy_decay = float(config["energy_decay"])
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
    from jax_sim import observations_jax as _obs

    @jax.jit
    def sim_step(carry, step_key):
        """
        carry = (grid, blue_pop, red_pop, blue_carries, red_carries, b_params, r_params)
        Returns: new_carry, rollout_data
        """
        grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params = carry
        b_params_sg = jax.tree.map(jax.lax.stop_gradient, b_params)
        r_params_sg = jax.tree.map(jax.lax.stop_gradient, r_params)
        key_obs, key_act, key_b_obs, key_r_obs, key_misc = jax.random.split(step_key, 5)

        # ── Build presence maps ─────────────────────────────────
        blue_map = jnp.zeros((gs, gs), dtype=jnp.bool_)
        blue_map = blue_map.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].set(b_pop.alive)
        red_map = jnp.zeros((gs, gs), dtype=jnp.bool_)
        red_map = red_map.at[r_pop.positions[:, 0], r_pop.positions[:, 1]].set(r_pop.alive)

        # ── Observations (via observations_jax — reload with train_entry) ──
        b_obs = _obs.build_observations_jax(
            b_pop, grid, blue_map, red_map, config, 0,
            key=key_b_obs, limit_red_sensing=True,
        )
        r_obs = _obs.build_observations_jax(
            r_pop, grid, blue_map, red_map, config, 0, key=key_r_obs,
        )

        # ── Forward passes ──────────────────────────────────────
        b_new_c, b_outs = model_apply(b_params_sg, b_carries, b_obs, _n_layers)
        r_new_c, r_outs = model_apply(r_params_sg, r_carries, r_obs, _n_layers)

        b_action_logits, b_signal_out, b_sym_w, b_vals, b_tom, b_token_ids, b_loss_vq, b_z_e, b_cult_f, b_cult_s = b_outs
        r_action_logits, r_signal_out, r_sym_w, r_vals, r_tom, r_token_ids, r_loss_vq, r_z_e, r_cult_f, r_cult_s = r_outs

        # ── Write VQ signals (STE forward ≡ discrete z_q) for neighbours ──
        b_pop = b_pop.replace(
            signals=jnp.where(b_pop.alive[:, None], b_signal_out, 0.0)
        )
        r_pop = r_pop.replace(
            signals=jnp.where(r_pop.alive[:, None], r_signal_out, 0.0)
        )

        # ── Sample actions ──────────────────────────────────────
        b_action_keys = jax.random.split(key_act, b_pop.max_pop)
        r_action_keys = jax.random.split(key_act, r_pop.max_pop)
        b_actions = jax.vmap(jax.random.categorical)(b_action_keys, b_action_logits)
        r_actions = jax.vmap(jax.random.categorical)(r_action_keys, r_action_logits)

        # ── Update episodic memory buffer ───────────────────────
        b_nb_flat = b_obs[:, 6 : 6 + K * sig_d]
        b_mean_nb_sig = b_nb_flat.reshape(b_pop.max_pop, K, sig_d).mean(axis=1)
        b_pop = update_memory_buffer(b_pop, b_mean_nb_sig, b_actions, b_pop.alive)

        r_nb_flat = r_obs[:, 6 : 6 + K * sig_d]
        r_mean_nb_sig = r_nb_flat.reshape(r_pop.max_pop, K, sig_d).mean(axis=1)
        r_pop = update_memory_buffer(r_pop, r_mean_nb_sig, r_actions, r_pop.alive)

        # ── Movement ────────────────────────────────────────────
        b_new_pos = apply_moves(b_pop.positions, b_actions, b_pop.alive, gs, grid.walls)
        r_new_pos = apply_moves(r_pop.positions, r_actions, r_pop.alive, gs, grid.walls)
        b_moved = (b_new_pos != b_pop.positions).any(axis=-1) & b_pop.alive
        r_moved = (r_new_pos != r_pop.positions).any(axis=-1) & r_pop.alive
        b_pop = b_pop.replace(positions=b_new_pos)
        r_pop = r_pop.replace(positions=r_new_pos)

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
        b_pop = b_pop.replace(energy=b_pop.energy + b_energy_gain)

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

        # ── Resource respawning ─────────────────────────────────
        res_key = jax.random.split(key_misc)[0]
        regen_rate = float(config.get("resource_regen_rate", 0.005))
        spawn_mask = jax.random.bernoulli(res_key, regen_rate, (gs, gs))
        new_res = grid.resources + spawn_mask.astype(jnp.float32) * _resource_spawn_boost
        grid = grid.replace(resources=jnp.clip(new_res, 0.0, _resource_max))

        # ── Energy decay ────────────────────────────────────────
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy - _energy_decay, 0.0, 1.0))
        r_pop = r_pop.replace(energy=jnp.clip(r_pop.energy - _energy_decay, 0.0, 1.0))

        # ── Starvation ──────────────────────────────────────────
        b_starved = b_pop.alive & (b_pop.energy < _starv_thresh)
        r_starved = r_pop.alive & (r_pop.energy < _starv_thresh)
        b_pop = kill_agents(b_pop, b_starved)
        r_pop = kill_agents(r_pop, r_starved)

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
        b_new_alive, r_catch_rew, b_catch_pen, caught_b = apply_catches(
            b_pop.positions, b_pop.alive,
            r_pop.positions, r_pop.alive,
            gs,
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
        r_caught_any = r_catch_rew > 0
        new_steps_since = jnp.where(r_caught_any, 0, r_pop.steps_since_catch + 1)
        new_steps_since = jnp.where(r_pop.alive, new_steps_since, 0)
        r_pop = r_pop.replace(steps_since_catch=new_steps_since)
        r_starved_hunt = r_pop.alive & (r_pop.steps_since_catch >= _red_starvation_steps)
        r_pop = kill_agents(r_pop, r_starved_hunt)

        # ── Puzzle logic ────────────────────────────────────────
        p_act, p_cool = decay_puzzle_timeout(grid.puzzle_active, grid.puzzle_cooldown)
        p_rew, p_solved, p_act, p_cool = check_puzzle_solved(
            b_pop.positions, b_pop.alive, grid.puzzle_nodes, p_act, p_cool, gs
        )
        p_grid = update_puzzle_grid(gs, grid.puzzle_nodes, p_act)
        grid = grid.replace(puzzle_active=p_act, puzzle_cooldown=p_cool, puzzle_grid=p_grid)
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy + p_rew * 0.5, 0.0, 1.0))

        # ── Rewards (all from config) ───────────────────────────
        b_rew = jnp.where(b_pop.alive, _reward_blue_alive, 0.0)
        b_rew = b_rew + 0.02 * b_pop.energy
        b_rew = b_rew + b_catch_pen * jnp.abs(_reward_blue_caught)
        b_rew = b_rew + _reward_resource * b_energy_gain
        b_rew = b_rew + _reward_move * b_moved.astype(jnp.float32)
        b_rew = b_rew + _puzzle_reward * p_rew
        b_rew = b_rew + contested_gain * 0.5

        r_rew = _reward_red_catch * r_catch_rew
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

        r_log_probs = jax.nn.log_softmax(r_action_logits, axis=-1)
        r_log_probs_taken = jnp.take_along_axis(r_log_probs, r_actions[:, None], axis=-1).squeeze(-1)

        # ── Rollout data ──────────────────────────────────────
        b_done = (~b_pop.alive).astype(jnp.float32)
        r_done = (~r_pop.alive).astype(jnp.float32)

        b_rollout = {
            "obs": b_obs, "actions": b_actions, "log_probs": b_log_probs_taken,
            "values": b_vals, "rewards": b_rew, "dones": b_done,
            "carries": b_carries,
            "loss_vq": b_loss_vq,
            "z_e": b_z_e,
            "token_ids": b_token_ids,
            "positions": b_pop.positions,
            "signals": b_pop.signals,
            "energy": b_pop.energy,
            "alive": b_pop.alive,
            "blue_caught": caught_b.astype(jnp.float32),
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
        }

        new_carry = (grid, b_pop, r_pop, b_new_c, r_new_c, b_params, r_params)
        return new_carry, {"blue": b_rollout, "red": r_rollout}

    return sim_step


# ── Main entry point ───────────────────────────────────────────────────────

def run_simulation(
    config: Dict,
    seed: int = 42,
    n_steps: int = 100000,
) -> Tuple[Dict, Dict]:
    """Prefer ``from jax_sim.train_entry import run_simulation`` after git pull."""
    from jax_sim.train_entry import run_simulation as _fresh_run

    return _fresh_run(config, seed=seed, n_steps=n_steps)


def _run_simulation_impl(
    config: Dict,
    seed: int = 42,
    n_steps: int = 100000,
) -> Tuple[Dict, Dict]:
    """
    Run full JAX simulation.
    Returns: (final_params, metrics_history)
    """
    config = _normalize_config(config)

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
    
    # ── Init corpus writer ──────────────────────────────────
    corpus_writer = SignalCorpusWriter(
        path=f"runs/{run_name}/signal_corpus.jsonl",
        sample_frac=config.get("corpus_sample_frac", 0.08),
        every_n_steps=config.get("corpus_every_n_steps", 20),
    )

    # ── Init Checkpointing ──────────────────────────────────
    ckpt_dir = config.get("checkpoint_dir") or os.path.abspath(
        f"runs/{run_name}/checkpoints"
    )
    # Orbax mkdir fails on symlinks (FileExistsError); use real volume path.
    ckpt_dir = str(Path(ckpt_dir).resolve())
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options=options)

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
    r_pop = init_population(
        max_pop_red, hidden_d, config["signal_dim"], gs, team_id=1,
        key=keys[2], n_agents=_red_start_n, memory_slots=config.get("memory_slots", 0),
    )

    # ── Init model ──────────────────────────────────────────
    _loc_env_start, _loc_env_end = loc_env_flat_bounds(config)
    _fwd_env_dim = compute_fwd_env_dim(config)
    model = AgentNetworkJax(
        hidden_dim=hidden_d,
        n_heads=config["n_heads"],
        n_layers=n_layers,
        obs_dim=0,  # computed inside
        signal_dim=config["signal_dim"],
        symbol_dim=config["symbol_dim"],
        vocab_size=config["vocab_size"],
        vq_beta=float(config.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(config.get("vq_dead_code_reset", True)),
        memory_slots=config.get("memory_slots", 0),
        fwd_env_dim=_fwd_env_dim,
    )
    model_apply = make_model_apply(model)

    # Compute exact obs_dim and init model
    obs_dim = compute_obs_dim_torch(config)
    print(f"[JAX] obs_dim = {obs_dim}")
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
    if hasattr(_rl_jax_mod, "auxiliary_update") and "carry_fwd_coef" in str(
        _inspect.getsource(_rl_jax_mod.auxiliary_update)
    ):
        print("[JAX] Phase11 carry_fwd: head_fwd_dyn_1/2 → carry_{t+1} MSE (stop_grad target)")
    if bool(config.get("gpu_resident_rollouts", True)):
        print("[JAX] Phase11.1 PPO: GPU-resident rollouts (no CPU offload / H2D)")
    else:
        print("[JAX] PPO: CPU rollout offload enabled (legacy A100 path)")
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
    _ckpt_latest = ckpt_mngr.latest_step()
    start_update = 0
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
        # Same architecture as blue; restore overwrites — skip 2nd full model.init compile.
        r_params = jax.tree_util.tree_map(jnp.copy, b_params)
        print("[JAX] model.init done — restoring checkpoint weights...", flush=True)
        abstract_tree = {"b_params": b_params, "r_params": r_params}
        try:
            restored = ckpt_mngr.restore(_ckpt_latest, items=abstract_tree)
        except ValueError as exc:
            msg = str(exc)
            if "do not match" in msg:
                print(
                    "[JAX] Orbax strict match failed (schema evolution). "
                    "Merging new heads manually...",
                    flush=True,
                )
                from flax.core import freeze, unfreeze

                raw_restored = ckpt_mngr.restore(_ckpt_latest)
                target_dict = unfreeze(abstract_tree)
                source_dict = unfreeze(raw_restored)
                for agent_type in ("b_params", "r_params"):
                    if agent_type not in source_dict or agent_type not in target_dict:
                        continue
                    src_agent = unfreeze(source_dict[agent_type])
                    tgt_agent = unfreeze(target_dict[agent_type])
                    for new_key in AUX_HEAD_KEYS:
                        if new_key in tgt_agent and new_key not in src_agent:
                            src_agent[new_key] = tgt_agent[new_key]
                            print(
                                f"[JAX] Injected randomly initialized {new_key} "
                                f"into {agent_type}",
                                flush=True,
                            )
                    source_dict[agent_type] = freeze(src_agent)
                restored = freeze(source_dict)
            elif "not compatible" in msg or "stored shape" in msg:
                print(
                    f"[JAX] Checkpoint step {_ckpt_latest} incompatible with current model "
                    f"(architecture changed) — re-init from scratch.",
                    flush=True,
                )
                print("[JAX] Delete old ckpts: rm -rf /mnt/throng-runs/checkpoints")
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
    _vq_on = "codebook" in _bp
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
    test_logits = test_outs[0]  # action_logits
    test_probs = jax.nn.softmax(test_logits, axis=-1)
    test_entropy = -jnp.sum(test_probs * jnp.log(test_probs + 1e-10), axis=-1)
    print(f"[DEBUG] Init action_logits mean={float(test_logits.mean()):.4f} std={float(test_logits.std()):.4f}")
    print(f"[DEBUG] Init entropy mean={float(test_entropy.mean()):.4f} (expected ~1.6 for uniform 5-action)")

    # ── Init optimizer (after checkpoint restore so momentum matches weights) ──
    b_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    r_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    b_opt_state = b_optimizer.init(b_params)
    r_opt_state = r_optimizer.init(r_params)

    # ── Carries ─────────────────────────────────────────────
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, hidden_d))

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
        return make_sim_step(cfg_copy, model, model_apply)

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
    # Lag-1 scout buffer for corpus / decode_signals direction LRT (matches main.py)
    _lag1_scout_pos = None
    _lag1_scout_sig = None
    _lag1_scout_dist = None
    _lag1_scout_tok = None
    _alarm_range = float(config.get("alarm_scout_range", 8))
    print(
        f"[JAX] corpus scout label: is_scout = (red_dist <= alarm_scout_range={_alarm_range})",
        flush=True,
    )
    _corpus_sig_dim = int(config["signal_dim"])
    for ui in range(start_update, n_updates):
        update_key = update_keys[ui]
        step_keys = jax.random.split(update_key, T)

        if use_pmap:
            # step_keys needs to be shaped (n_devices, T, 2)
            # Actually step_keys is just a PRNGKey array. We can split it for each device.
            step_keys_pmap = jax.random.split(update_key, n_devices * T).reshape(n_devices, T, -1)
            final_carry, rollout_data = sim_step_mapped(grid, b_pop, r_pop, b_carries, r_carries, step_keys_pmap)
            grid, b_pop, r_pop, b_carries, r_carries, _, _ = final_carry
            
            # Flatten rollout data across devices
            def flatten_pmap(x):
                return x.reshape(n_devices * T, *x.shape[2:])
            rollout_data = jax.tree_util.tree_map(flatten_pmap, rollout_data)
        else:
            init_carry = (grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params)
            if ui == start_update:
                print(f"[JAX] lax.scan rollout starting (update {ui + 1})...", flush=True)
            _t_rollout0 = __import__("time").time()
            final_carry, rollout_data = lax.scan(sim_step_fn, init_carry, step_keys)
            if ui == start_update:
                _dt0 = __import__("time").time() - _t_rollout0
                print(
                    f"[JAX] lax.scan rollout done in {_dt0:.1f}s (update {ui + 1}) — PPO next.",
                    flush=True,
                )
            grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params = final_carry

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
        _gpu_resident = bool(config.get("gpu_resident_rollouts", True))
        _fwd_coef = float(config.get("fwd_coef", 0.05))
        _carry_fwd_coef = float(config.get("carry_fwd_coef", 0.05))
        _fwd_mb   = int(config.get("ppo_minibatch_size", 512))

        if ui == start_update:
            print(
                "  [JAX] Compiling Blue PPO backward (_minibatch_step) — "
                "often 3–10+ min on first update; do not Stop.",
                flush=True,
            )
        print("  [DEBUG] --- Blue PPO Update ---")
        b_batch = rollout_data["blue"]
        # Save carries on CPU before ppo_update deletes them (needed for fwd dynamics)
        _b_carries_np = np.asarray(b_batch["carries"])
        _b_actions_np = np.asarray(b_batch["actions"])
        _b_obs_np = np.asarray(b_batch["obs"])
        _b_alive_np   = np.asarray(b_batch["alive"]) if "alive" in b_batch else None
        _t_ppo0 = __import__("time").time()
        b_params, b_opt_state, b_metrics = ppo_update(
            b_params, b_opt_state, b_optimizer, model_apply,
            b_batch, n_layers, update_key,
            clip_eps=float(config.get("ppo_clip_eps", config.get("ppo_clip", 0.2))),
            vf_coef=float(config.get("ppo_value_coef", 0.25)),
            ent_coef=float(config.get("ppo_entropy_coef", 0.02)),
            vq_coef=float(config.get("vq_loss_coef", 0.1)),
            minibatch_size=_fwd_mb,
            gamma=float(config.get("ppo_gamma", 0.99)),
            lam=float(config.get("ppo_gae_lam", 0.95)),
            team="blue",
            gpu_resident=_gpu_resident,
        )
        if ui == start_update:
            print(
                f"  [JAX] Blue PPO done in {__import__('time').time() - _t_ppo0:.1f}s",
                flush=True,
            )
        # Auxiliary losses (forward dynamics + self-prediction) for blue
        _self_pred_coef = float(config.get("self_pred_coef", 0.1))
        fwd_key, update_key = jax.random.split(update_key)
        b_params, b_opt_state, b_fwd_loss, b_carry_fwd_loss, b_sp_loss, b_sp_acc = auxiliary_update(
            b_params, b_opt_state, b_optimizer, b_aux_apply_fn,
            _b_carries_np, _b_actions_np, _b_obs_np,
            _loc_env_start, _loc_env_end,
            _b_alive_np, fwd_key, minibatch_size=_fwd_mb,
            fwd_coef=_fwd_coef, carry_fwd_coef=_carry_fwd_coef,
            self_pred_coef=_self_pred_coef,
        )
        b_metrics["fwd_loss"] = b_fwd_loss
        b_metrics["carry_fwd_loss"] = b_carry_fwd_loss
        b_metrics["sp_loss"]     = b_sp_loss
        b_metrics["sp_acc"]      = b_sp_acc

        if config.get("vq_dead_code_reset", True) and "z_e" in b_batch:
            _dc_key, update_key = jax.random.split(update_key)
            _tok = jnp.asarray(b_batch["token_ids"]).reshape(-1)
            _ze = jnp.asarray(b_batch["z_e"]).reshape(-1, int(config["signal_dim"]))
            _alive = jnp.asarray(b_batch["alive"]).reshape(-1).astype(bool)
            b_params = dead_code_reset_codebook_params(
                b_params,
                _tok[_alive],
                _ze[_alive],
                int(config["vocab_size"]),
                _dc_key,
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
        r_params, r_opt_state, r_metrics = ppo_update(
            r_params, r_opt_state, r_optimizer, model_apply,
            r_batch, n_layers, update_key,
            clip_eps=float(config.get("ppo_clip_eps", config.get("ppo_clip", 0.2))),
            vf_coef=float(config.get("ppo_value_coef", 0.25)),
            ent_coef=float(config.get("ppo_entropy_coef", 0.02)),
            vq_coef=float(config.get("vq_loss_coef", 0.1)),
            minibatch_size=_fwd_mb,
            gamma=float(config.get("ppo_gamma", 0.99)),
            lam=float(config.get("ppo_gae_lam", 0.95)),
            team="red",
            gpu_resident=_gpu_resident,
        )
        if ui == start_update:
            print(
                f"  [JAX] Red PPO done in {__import__('time').time() - _t_rppo0:.1f}s",
                flush=True,
            )
        fwd_key, update_key = jax.random.split(update_key)
        r_params, r_opt_state, r_fwd_loss, r_carry_fwd_loss, r_sp_loss, r_sp_acc = auxiliary_update(
            r_params, r_opt_state, r_optimizer, r_aux_apply_fn,
            _r_carries_np, _r_actions_np, _r_obs_np,
            _loc_env_start, _loc_env_end,
            _r_alive_np, fwd_key, minibatch_size=_fwd_mb,
            fwd_coef=_fwd_coef, carry_fwd_coef=_carry_fwd_coef,
            self_pred_coef=_self_pred_coef,
        )
        r_metrics["fwd_loss"] = r_fwd_loss
        r_metrics["carry_fwd_loss"] = r_carry_fwd_loss
        r_metrics["sp_acc"]   = r_sp_acc

        if config.get("vq_dead_code_reset", True) and "z_e" in r_batch:
            _dc_key, update_key = jax.random.split(update_key)
            _tok = jnp.asarray(r_batch["token_ids"]).reshape(-1)
            _ze = jnp.asarray(r_batch["z_e"]).reshape(-1, int(config["signal_dim"]))
            _alive = jnp.asarray(r_batch["alive"]).reshape(-1).astype(bool)
            r_params = dead_code_reset_codebook_params(
                r_params,
                _tok[_alive],
                _ze[_alive],
                int(config["vocab_size"]),
                _dc_key,
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
            
            # Action distribution (N=stay, S, E, W, stay=0)
            alive_actions = b_act_all[b_alive_all]
            if len(alive_actions) > 0:
                act_counts = np.bincount(alive_actions, minlength=5)
                act_pct = act_counts / act_counts.sum() * 100
                act_str = f"N={act_pct[1]:.0f}% S={act_pct[2]:.0f}% E={act_pct[3]:.0f}% W={act_pct[4]:.0f}% Stay={act_pct[0]:.0f}%"
            else:
                act_str = "no alive agents"
            
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
            
            # Reward breakdown
            rew_alive = b_rew_all[b_alive_all]
            rew_mean = float(rew_alive.mean()) if len(rew_alive) > 0 else 0
            
            # NB_GAIN correlation
            sp_r = float('nan')
            b_nb_gain_snap = b_pop_np.nb_gain
            if alive_mask_final.sum() > 10:
                try:
                    from scipy.stats import spearmanr
                    _nb_g = b_nb_gain_snap[alive_mask_final]
                    _ages = b_pop_np.ages[alive_mask_final].astype(float)
                    if _nb_g.std() > 1e-6:
                        sp_r, _ = spearmanr(_nb_g, _ages)
                except Exception:
                    pass
            
            # Print concise dashboard
            print(f"\n{'='*70}")
            print(f"[step {step_val:>7}] {steps_sec:.0f} steps/sec | blue={b_alive_now} red={r_alive_now} | ppo={ui+1}")
            print(f"  Actions: {act_str}")
            print(f"  Energy:  mean={e_mean:.3f} std={e_std:.3f} | Age: mean={age_mean:.0f} max={age_max:.0f}")
            vf_loss = float(b_metrics.get('ppo_vf_loss', 0)) if isinstance(b_metrics, dict) else 0
            ent_val = float(b_metrics.get('ppo_entropy', 0)) if isinstance(b_metrics, dict) else 0
            clip_frac = float(b_metrics.get('ppo_clip_frac', 0)) if isinstance(b_metrics, dict) else 0
            fwd_loss_val = float(b_metrics.get('fwd_loss', float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            carry_fwd_val = float(b_metrics.get('carry_fwd_loss', float('nan'))) if isinstance(b_metrics, dict) else float('nan')
            sp_acc_val   = float(b_metrics.get('sp_acc',   float('nan'))) if isinstance(b_metrics, dict) else float('nan')
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
            print(f"  Reward:  mean={rew_mean:.4f} | Entropy: {ent_val:.4f}")
            print(
                f"  AuxLoss: fwd_env={fwd_loss_val:.4f} | carry_fwd={carry_fwd_val:.4f} "
                f"(↓0.05–0.1) | self_pred_acc={sp_acc_val:.3f} | "
                f"carry_rank={carry_rank} | carry_H={carry_entropy:.2f}"
            )
            blue_caught_rollout = 0
            if "blue_caught" in rollout_data["blue"]:
                blue_caught_rollout = int(np.asarray(rollout_data["blue"]["blue_caught"]).sum())
            print(f"  VQ: loss={vq_loss_val:.4f} | codes_active={vq_codes_str} | clusters={active_clusters_str} | NB_GAIN↔surv: {sp_r:.3f}")
            print(
                f"  Ecology: blue_caught={blue_caught_rollout} this rollout | "
                f"red_floor={red_curriculum_stages[red_curriculum_idx]} "
                f"sustain={red_sustain_count}/{red_sustain_needed} | brain={n_layers}L"
            )
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

        # loc_env is the 4th block in b_obs (8 channels: blue, red, wall, resource, shelter, contested, scent, puzzle)
        idx_offset = 6 + (config["neighbor_k"] * config["signal_dim"]) + (25 * config["symbol_dim"])
        idx_resource = idx_offset + (12 * 8) + 3  # 12th cell (center of 5x5), channel 3 = resource

        start_step = ui * T
        for t in range(T):
            global_step = start_step + t
            if global_step % config.get("corpus_every_n_steps", 20) != 0:
                continue

            b_pos = b_pos_all[t]
            r_pos = r_pos_all[t]
            b_alive = b_alive_all[t]
            r_alive = r_alive_all[t]
            
            if not np.any(b_alive):
                continue
                
            alive_idx = np.where(b_alive)[0]
            
            # Compute nearest red distance and bearing
            n_alive = len(alive_idx)
            red_dist = np.full(n_alive, 999.0)
            red_bear = np.zeros(n_alive)
            is_scout = np.zeros(n_alive, dtype=bool)
            
            active_red_idx = np.where(r_alive)[0]
            if len(active_red_idx) > 0:
                pos_b_alive = b_pos[alive_idx]
                pos_r_active = r_pos[active_red_idx]
                
                # Pairwise torus distance
                gs_val = config["grid_size"]
                diff = np.abs(pos_b_alive[:, None, :] - pos_r_active[None, :, :])
                diff = np.minimum(diff, gs_val - diff)
                dists = np.max(diff, axis=-1)  # Chebyshev
                
                min_idx = np.argmin(dists, axis=1)
                red_dist = dists[np.arange(n_alive), min_idx]
                # Corpus scout = red within alarm range (not red_detection_radius, which is 0 when blind)
                is_scout = red_dist <= _alarm_range
                
                # Bearing (simple dy/dx)
                nearest_red_pos = pos_r_active[min_idx]
                dy = nearest_red_pos[:, 0] - pos_b_alive[:, 0]
                dx = nearest_red_pos[:, 1] - pos_b_alive[:, 1]
                # Fix wraparound for bearing
                dy = np.where(dy > gs_val/2, dy - gs_val, np.where(dy < -gs_val/2, dy + gs_val, dy))
                dx = np.where(dx > gs_val/2, dx - gs_val, np.where(dx < -gs_val/2, dx + gs_val, dx))
                red_bear = np.degrees(np.arctan2(dy, dx)) % 360.0

            # Compute neighbor count
            nb_count = np.zeros(n_alive)
            pos_b_alive = b_pos[alive_idx]
            diff_b = np.abs(pos_b_alive[:, None, :] - pos_b_alive[None, :, :])
            diff_b = np.minimum(diff_b, config["grid_size"] - diff_b)
            dists_b = np.max(diff_b, axis=-1)
            np.fill_diagonal(dists_b, 9999) # Self
            nb_count = np.sum((dists_b <= 4), axis=1)

            # Local resource
            loc_res = b_obs_all[t, alive_idx, idx_resource]

            nb_scout_lag1 = np.full((n_alive, _corpus_sig_dim), np.nan, dtype=np.float32)
            nb_scout_dist_lag1 = np.full(n_alive, np.nan, dtype=np.float32)
            nb_scout_token_lag1 = np.full(n_alive, -1, dtype=np.int32)
            if _lag1_scout_pos is not None and len(_lag1_scout_pos) > 0:
                gs_val = int(config["grid_size"])
                pos_b_alive = b_pos[alive_idx].astype(np.float32)
                sp2 = _lag1_scout_pos.astype(np.float32)
                dd2 = np.abs(pos_b_alive[:, None, :] - sp2[None, :, :])
                dd2 = np.minimum(dd2, gs_val - dd2)
                sc2 = np.maximum(dd2[:, :, 0], dd2[:, :, 1])
                for ai in range(n_alive):
                    within = sc2[ai] <= _alarm_range
                    if within.any():
                        nb_scout_lag1[ai] = _lag1_scout_sig[within].mean(axis=0)
                        nb_scout_dist_lag1[ai] = float(_lag1_scout_dist[within].mean())
                        toks = _lag1_scout_tok[within].astype(np.int64)
                        nb_scout_token_lag1[ai] = int(np.bincount(toks).argmax())

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
                nb_scout_sig_lag1=nb_scout_lag1,
                nb_scout_dist_lag1=nb_scout_dist_lag1,
                nb_scout_token_lag1=nb_scout_token_lag1,
            )
            if is_scout.any():
                pos_b_alive = b_pos[alive_idx].astype(np.float32)
                _lag1_scout_pos = pos_b_alive[is_scout].copy()
                _lag1_scout_sig = b_sig_all[t, alive_idx[is_scout]].copy()
                _lag1_scout_dist = red_dist[is_scout].copy()
                _lag1_scout_tok = b_tok_all[t, alive_idx[is_scout]].copy()
            else:
                _lag1_scout_pos = None
                _lag1_scout_sig = None
                _lag1_scout_dist = None
                _lag1_scout_tok = None

        corpus_writer.flush_to_disk()

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
            ckpt_state = {
                "b_params": b_params,
                "r_params": r_params,
            }
            ckpt_mngr.save(ui + 1, items=ckpt_state)
            ckpt_mngr.wait_until_finished()
            print(f"  [CKPT] Saved step {(ui+1)*T}")

        if (ui + 1) % 10 == 0 or ui == 0:
            alive_count = int(b_pop.alive.sum())
            nan_dbg = f"has_nan={metrics_py.get('has_nan', 0):.0f}"
            if metrics_py.get('has_nan', 0) > 0.5:
                nan_dbg += (f" aL={metrics_py.get('nan_action_logits',0):.0f}"
                           f" vP={metrics_py.get('nan_values_pred',0):.0f}"
                           f" oL={metrics_py.get('nan_old_log_probs',0):.0f}"
                           f" adv={metrics_py.get('nan_advantages',0):.0f}"
                           f" ratio={metrics_py.get('nan_ratio',0):.0f}")
            print(f"  PPO#{ui+1} pop={alive_count} pg={metrics_py['ppo_pg_loss']:.4f} "
                  f"vf={metrics_py['ppo_vf_loss']:.4f} ent={metrics_py['ppo_entropy']:.4f} {nan_dbg}")
            print(f"       (Red) pg={float(r_metrics['ppo_pg_loss']):.4f} vf={float(r_metrics['ppo_vf_loss']):.4f} ent={float(r_metrics['ppo_entropy']):.4f}")

    corpus_writer.close()
    return {"blue": b_params, "red": r_params}, {"metrics": all_metrics}


# ── CLI entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_phase7.yaml"
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
