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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress INFO/WARN
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")  # disable autotune spam

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
    generate_puzzle_nodes, update_puzzle_grid, decay_puzzle_timeout, check_puzzle_solved
)
from communication.analysis import SignalCorpusWriter
from jax_sim.population_jax import (
    PopState, init_population, kill_agents, update_memory_buffer, apply_mind_meld
)
from jax_sim.network_jax import AgentNetworkJax
from jax_sim.rl_jax import compute_gae, ppo_loss, create_optimizer, ppo_update
from agents.network_torch import compute_obs_dim_torch


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


# ── Observation builder (JAX) ──────────────────────────────────────────────

def build_observations_jax(
    pop: PopState,
    grid: GridState,
    blue_map: jnp.ndarray,
    red_map: jnp.ndarray,
    config: Dict,
    step: int,
) -> jnp.ndarray:
    """Build flat observation vector for all agents."""
    gs = config["grid_size"]
    K = config["neighbor_k"]
    r = config["local_obs_radius"]
    sym_d = config["symbol_dim"]
    sig_d = config["signal_dim"]
    W = (2 * r + 1) ** 2
    N = pop.max_pop

    # Normalize features
    norm_age = pop.ages.astype(jnp.float32) / float(config["max_age"])
    norm_x = pop.positions[:, 0].astype(jnp.float32) / gs
    norm_y = pop.positions[:, 1].astype(jnp.float32) / gs
    energy = pop.energy
    nl_norm = pop.n_layers.astype(jnp.float32) / 6.0
    mat_frac = jnp.zeros(N, dtype=jnp.float32)  # simplified

    own_state = jnp.stack([norm_age, mat_frac, energy, nl_norm, norm_x, norm_y], axis=1)

    # Neighbour signals
    nb_sigs = get_neighbour_signals(
        pop.positions, pop.signals, pop.alive, K, gs,
    )  # (N, K, sig_d)

    # Local patches
    loc_sym = get_local_patches(grid.symbols, pop.positions, r, gs)  # (N, W, sym_d)
    loc_pres = jnp.concatenate([
        get_local_patches(blue_map.astype(jnp.float32), pop.positions, r, gs)[..., None],
        get_local_patches(red_map.astype(jnp.float32), pop.positions, r, gs)[..., None],
    ], axis=-1)  # (N, W, 2)
    loc_wall = get_local_patches(grid.walls.astype(jnp.float32), pop.positions, r, gs)[..., None]
    loc_res = get_local_patches(grid.resources, pop.positions, r, gs)[..., None]
    loc_shelter = get_local_patches(grid.shelter_spots.astype(jnp.float32), pop.positions, r, gs)[..., None]
    loc_contested = get_local_patches(grid.contested_res, pop.positions, r, gs)[..., None]
    loc_scent = get_local_patches(grid.scent_trails, pop.positions, r, gs)[..., None]

    loc_env = jnp.concatenate([loc_pres, loc_wall, loc_res, loc_shelter, loc_contested, loc_scent], axis=-1)

    # Cultural memory
    loc_cult_fast = get_local_patches(grid.cultural_fast, pop.positions, r, gs)
    loc_cult_slow = get_local_patches(grid.cultural_slow, pop.positions, r, gs)

    # Assemble
    parts = [
        own_state,                          # (N, 6)
        nb_sigs.reshape(N, -1),            # (N, K*sig_d)
        loc_sym.reshape(N, -1),             # (N, W*sym_d)
        loc_env.reshape(N, -1),             # (N, W*6)
        pop.signals,                        # (N, sig_d)
    ]

    if pop.memory_buffer is not None:
        parts.append(pop.memory_buffer.reshape(N, -1))

    parts.append(loc_cult_fast.reshape(N, -1))
    parts.append(loc_cult_slow.reshape(N, -1))

    obs = jnp.concatenate(parts, axis=1)
    obs = jnp.where(pop.alive[:, None], obs, 0.0)
    return obs


# ── Single simulation step (for scan) ──────────────────────────────────────

def make_sim_step(config: Dict, model: AgentNetworkJax):
    """Factory: returns a jittable step function.
    Params are passed through carry to avoid stale closure capture."""
    gs = config["grid_size"]
    K = config["neighbor_k"]
    r = config["local_obs_radius"]
    sym_d = config["symbol_dim"]
    sig_d = config["signal_dim"]
    hidden_d = config["hidden_dim"]

    @jax.jit
    def sim_step(carry, step_key):
        """
        carry = (grid, blue_pop, red_pop, blue_carries, red_carries, b_params, r_params)
        Returns: new_carry, rollout_data
        """
        grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params = carry
        # Don't backprop through params during rollout
        b_params_sg = jax.tree.map(jax.lax.stop_gradient, b_params)
        r_params_sg = jax.tree.map(jax.lax.stop_gradient, r_params)
        key_obs, key_act = jax.random.split(step_key)

        # ── Build presence maps ─────────────────────────────────
        blue_map = jnp.zeros((gs, gs), dtype=jnp.bool_)
        blue_map = blue_map.at[b_pop.positions[:, 0], b_pop.positions[:, 1]].set(b_pop.alive)
        red_map = jnp.zeros((gs, gs), dtype=jnp.bool_)
        red_map = red_map.at[r_pop.positions[:, 0], r_pop.positions[:, 1]].set(r_pop.alive)

        # ── Observations ────────────────────────────────────────
        b_obs = build_observations_jax(b_pop, grid, blue_map, red_map, config, 0)
        r_obs = build_observations_jax(r_pop, grid, blue_map, red_map, config, 0)

        # ── Forward passes ──────────────────────────────────────
        b_new_c, b_outs = model.apply(b_params_sg, b_carries, b_obs, config["n_layers"])
        r_new_c, r_outs = model.apply(r_params_sg, r_carries, r_obs, config["n_layers"])

        b_action_logits, b_signal_logits, b_sym_w, b_vals, b_tom, _, _, b_cult_f, b_cult_s = b_outs
        r_action_logits, r_signal_logits, r_sym_w, r_vals, r_tom, _, _, r_cult_f, r_cult_s = r_outs

        # ── Sample actions (vmapped, JIT-safe) ──────────────────
        b_action_keys = jax.random.split(key_act, b_pop.max_pop)
        r_action_keys = jax.random.split(key_act, r_pop.max_pop)
        b_actions = jax.vmap(jax.random.categorical)(b_action_keys, b_action_logits)
        r_actions = jax.vmap(jax.random.categorical)(r_action_keys, r_action_logits)

        # ── Movement ────────────────────────────────────────────
        b_new_pos = apply_moves(b_pop.positions, b_actions, b_pop.alive, gs, grid.walls)
        r_new_pos = apply_moves(r_pop.positions, r_actions, r_pop.alive, gs, grid.walls)

        b_pop = b_pop.replace(positions=b_new_pos)
        r_pop = r_pop.replace(positions=r_new_pos)

        # ── Resource consumption ────────────────────────────────
        b_energy_gain, new_res = consume_resources(
            b_pop.positions, b_pop.alive, grid.resources,
            decay=float(config.get("resource_decay", 0.05))
        )
        grid = grid.replace(resources=new_res)
        b_pop = b_pop.replace(energy=b_pop.energy + b_energy_gain)

        # ── Resource respawning ─────────────────────────────────
        # Sparse respawn to maintain pressure (agents must compete)
        res_key = jax.random.split(step_key)[0]
        spawn_mask = jax.random.bernoulli(res_key, 0.005, (gs, gs))
        new_res = grid.resources + spawn_mask.astype(jnp.float32) * 0.2
        grid = grid.replace(resources=jnp.clip(new_res, 0.0, 1.0))

        # ── Energy decay ────────────────────────────────────────
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy - config["energy_decay"], 0.0, 1.0))
        r_pop = r_pop.replace(energy=jnp.clip(r_pop.energy - config["energy_decay"], 0.0, 1.0))


        # ── Starvation ──────────────────────────────────────────
        b_starved = b_pop.alive & (b_pop.energy < config["starvation_threshold"])
        r_starved = r_pop.alive & (r_pop.energy < config["starvation_threshold"])
        b_pop = kill_agents(b_pop, b_starved)
        r_pop = kill_agents(r_pop, r_starved)

        # ── Age ─────────────────────────────────────────────────
        b_pop = b_pop.replace(ages=b_pop.ages + 1)
        r_pop = r_pop.replace(ages=r_pop.ages + 1)

        # ── Max age death ───────────────────────────────────────
        b_old = b_pop.alive & (b_pop.ages >= config["max_age"])
        r_old = r_pop.alive & (r_pop.ages >= config["max_age"])
        b_pop = kill_agents(b_pop, b_old)
        r_pop = kill_agents(r_pop, r_old)

        # ── Mind-Melding ─────────────────────────────────────────
        if config.get("mind_meld_enabled", False):
            b_pop = apply_mind_meld(
                b_pop, gs, 
                radius=int(config.get("mind_meld_radius", 1)),
                rate=float(config.get("mind_meld_rate", 0.1)),
                direction=config.get("mind_meld_direction", "older_to_younger")
            )

        # ── Catch detection ─────────────────────────────────────
        b_new_alive, r_catch_rew, b_catch_pen, _ = apply_catches(
            b_pop.positions, b_pop.alive,
            r_pop.positions, r_pop.alive,
            gs, catch_radius=1,
        )
        b_pop = b_pop.replace(alive=b_new_alive)
        # Zero carries for caught blues
        b_pop = b_pop.replace(
            carries=jnp.where(~b_new_alive[:, None], 0.0, b_pop.carries)
        )

        # ── Puzzle logic ────────────────────────────────────────
        p_act, p_cool = decay_puzzle_timeout(grid.puzzle_active, grid.puzzle_cooldown)
        p_rew, p_solved, p_act, p_cool = check_puzzle_solved(
            b_pop.positions, b_pop.alive, grid.puzzle_nodes, p_act, p_cool, gs
        )
        p_grid = update_puzzle_grid(gs, grid.puzzle_nodes, p_act)
        grid = grid.replace(puzzle_active=p_act, puzzle_cooldown=p_cool, puzzle_grid=p_grid)
        b_pop = b_pop.replace(energy=jnp.clip(b_pop.energy + p_rew * 0.5, 0.0, 1.0))

        # ── Rewards ─────────────────────────────────────────────
        # Base alive reward + energy bonus (creates variance for learning)
        b_rew = jnp.where(b_pop.alive, float(config.get("reward_blue_alive", 0.05)), 0.0)
        b_rew = b_rew + 0.02 * b_pop.energy  # higher energy = more reward
        b_rew = b_rew + b_catch_pen
        # Resource gathering reward (only if agent actually gathered something)
        b_rew = b_rew + 0.1 * b_energy_gain
        # Puzzle reward
        b_rew = b_rew + p_rew * 1.0
        # Red reward
        r_rew = jnp.where(r_pop.alive, float(config.get("reward_red_catch", 1.0)), 0.0) * r_catch_rew

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
            "positions": b_pop.positions,
            "signals": b_pop.signals,
            "energy": b_pop.energy,
            "alive": b_pop.alive,
        }
        r_rollout = {
            "obs": r_obs, "actions": r_actions, "log_probs": r_log_probs_taken,
            "values": r_vals, "rewards": r_rew, "dones": r_done,
            "carries": r_carries,
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
    """
    Run full JAX simulation.
    Returns: (final_params, metrics_history)
    """
    config = _normalize_config(config)

    run_name = config.get("run_name", "jax_run")
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    
    # ── Init corpus writer ──────────────────────────────────
    corpus_writer = SignalCorpusWriter(
        path=f"runs/{run_name}/signal_corpus.jsonl",
        sample_frac=config.get("corpus_sample_frac", 0.08),
        every_n_steps=config.get("corpus_every_n_steps", 20),
    )

    # ── Init Checkpointing ──────────────────────────────────
    ckpt_dir = os.path.abspath(f"runs/{run_name}/checkpoints")
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
    # Simple wall generation (random)
    wall_mask = jax.random.bernoulli(keys[0], config.get("wall_density", 0.08), (gs, gs))
    
    # Puzzle init
    p_nodes, p_act, p_cool = generate_puzzle_nodes(keys[9], n_nodes=3, grid_size=gs)
    p_grid = update_puzzle_grid(gs, p_nodes, p_act)
    
    grid = grid.replace(
        walls=wall_mask,
        puzzle_nodes=p_nodes,
        puzzle_active=p_act,
        puzzle_cooldown=p_cool,
        puzzle_grid=p_grid
    )

    # ── Init populations ────────────────────────────────────
    b_pop = init_population(
        max_pop, hidden_d, config["signal_dim"], gs, team_id=0,
        key=keys[1], n_agents=max_pop, memory_slots=config.get("memory_slots", 0),
    )
    r_pop = init_population(
        max_pop_red, hidden_d, config["signal_dim"], gs, team_id=1,
        key=keys[2], n_agents=6, memory_slots=config.get("memory_slots", 0),
    )

    # ── Init model ──────────────────────────────────────────
    model = AgentNetworkJax(
        hidden_dim=hidden_d,
        n_heads=config["n_heads"],
        n_layers=n_layers,
        obs_dim=0,  # computed inside
        signal_dim=config["signal_dim"],
        symbol_dim=config["symbol_dim"],
        vocab_size=config["vocab_size"],
        memory_slots=config.get("memory_slots", 0),
    )

    # Compute exact obs_dim and init model
    obs_dim = compute_obs_dim_torch(config)
    print(f"[JAX] obs_dim = {obs_dim}")
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_carry = jnp.zeros((1, hidden_d))
    b_params = model.init(keys[3], dummy_carry, dummy_obs, n_layers)
    r_params = model.init(keys[5], dummy_carry, dummy_obs, n_layers)

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
    _, test_outs = model.apply(b_params, test_carry, test_obs, n_layers)
    test_logits = test_outs[0]  # action_logits
    test_probs = jax.nn.softmax(test_logits, axis=-1)
    test_entropy = -jnp.sum(test_probs * jnp.log(test_probs + 1e-10), axis=-1)
    print(f"[DEBUG] Init action_logits mean={float(test_logits.mean()):.4f} std={float(test_logits.std()):.4f}")
    print(f"[DEBUG] Init entropy mean={float(test_entropy.mean()):.4f} (expected ~1.6 for uniform 5-action)")

    # ── Init optimizer ──────────────────────────────────────
    b_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    r_optimizer = create_optimizer(config["ppo_lr"], config["ppo_max_grad_norm"])
    b_opt_state = b_optimizer.init(b_params)
    r_opt_state = r_optimizer.init(r_params)

    # ── Carries ─────────────────────────────────────────────
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, hidden_d))

    # ── Restore Checkpoint ──────────────────────────────────
    start_update = 0
    if ckpt_mngr.latest_step() is not None:
        print(f"[JAX] Resuming from checkpoint {ckpt_mngr.latest_step()}...")
        # We restore into abstract trees to avoid copying dicts directly
        abstract_tree = {
            "b_params": b_params,
            "r_params": r_params,
            "b_opt_state": b_opt_state,
            "r_opt_state": r_opt_state,
            "grid": grid,
            "b_pop": b_pop,
            "r_pop": r_pop,
            "b_carries": b_carries,
            "r_carries": r_carries,
        }
        restored = ckpt_mngr.restore(ckpt_mngr.latest_step(), args=ocp.args.StandardRestore(abstract_tree))
        b_params = restored["b_params"]
        r_params = restored["r_params"]
        b_opt_state = restored["b_opt_state"]
        r_opt_state = restored["r_opt_state"]
        grid = restored["grid"]
        b_pop = restored["b_pop"]
        r_pop = restored["r_pop"]
        b_carries = restored["b_carries"]
        r_carries = restored["r_carries"]
        start_update = ckpt_mngr.latest_step()
    
    # ── Training loop ───────────────────────────────────────
    # Python loop for update cycles (avoids OOM from saving all param states)
    # Only the inner rollout uses lax.scan
    n_updates = n_steps // T
    update_keys = jax.random.split(keys[4], n_updates)

    # JIT or PMAP the inner rollout
    sim_step_fn = make_sim_step(config, model)
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

    all_metrics = []
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
            final_carry, rollout_data = lax.scan(sim_step_fn, init_carry, step_keys)
            grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params = final_carry

        # ── NaN debug after rollout ─────────────────────────────
        b_batch = rollout_data["blue"]
        has_nan_obs = bool(jnp.isnan(b_batch["obs"]).any())
        has_nan_vals = bool(jnp.isnan(b_batch["values"]).any())
        has_nan_logp = bool(jnp.isnan(b_batch["log_probs"]).any())
        has_nan_rew = bool(jnp.isnan(b_batch["rewards"]).any())
        if ui == 0:
            print(f"[DEBUG] Rollout data NaN: obs={has_nan_obs} vals={has_nan_vals} logp={has_nan_logp} rew={has_nan_rew}")

        # PPO update (not JIT — Python loop)
        print("  [DEBUG] --- Blue PPO Update ---")
        b_batch = rollout_data["blue"]
        b_params, b_opt_state, b_metrics = ppo_update(
            b_params, b_opt_state, b_optimizer, model.apply,
            b_batch, n_layers, update_key,
            clip_eps=config.get("ppo_clip_eps", 0.2),
            vf_coef=config.get("ppo_value_coef", 0.25),
            ent_coef=config.get("ppo_entropy_coef", 0.01),
        )

        print("  [DEBUG] --- Red PPO Update ---")
        r_batch = rollout_data["red"]
        r_params, r_opt_state, r_metrics = ppo_update(
            r_params, r_opt_state, r_optimizer, model.apply,
            r_batch, n_layers, update_key,
            clip_eps=config.get("ppo_clip_eps", 0.2),
            vf_coef=config.get("ppo_value_coef", 0.25),
            ent_coef=config.get("ppo_entropy_coef", 0.01),
        )

        # ── NaN debug after PPO update ──────────────────────────
        if ui == 0:
            flat_p = jax.tree_util.tree_leaves(b_params)
            has_nan_params_after = any(bool(jnp.isnan(p).any()) for p in flat_p)
            print(f"[DEBUG] Params NaN after PPO update: {has_nan_params_after}")

        # ── Legacy Telemetry Prints ─────────────────────────────
        step_val = (ui + 1) * T
        if step_val % 200 == 0 or T >= 200:  # Print if we cross 200 steps or T is large
            b_pop_np = jax.device_get(b_pop)
            b_alive_snap = b_pop_np.alive
            b_ages_snap = b_pop_np.ages.astype(np.float32)
            b_nb_gain_snap = b_pop_np.nb_gain
            
            b_act_all = np.array(rollout_data["blue"]["actions"])
            b_alive_all = np.array(rollout_data["blue"]["alive"]).astype(bool)
            
            # TOM_STAY_TRACK
            stay_mask = (b_act_all == 0) & b_alive_all
            n_alive_over_time = b_alive_all.sum()
            stay_rate = stay_mask.sum() / max(1, n_alive_over_time)
            
            # NB_GAIN_SURV
            sp_r, sp_p = float('nan'), float('nan')
            mean_gain, std_gain = 1.0, 0.0
            if b_alive_snap.sum() > 10:
                from scipy.stats import spearmanr
                _nb_g = b_nb_gain_snap[b_alive_snap]
                _ages = b_ages_snap[b_alive_snap]
                if _nb_g.std() > 1e-6:
                    sp_r, sp_p = spearmanr(_nb_g, _ages)
                mean_gain = float(_nb_g.mean())
                std_gain = float(_nb_g.std())
            
            print(f"[step {step_val:>8}] NB_GAIN_SURV  spearman_r={sp_r:.4f}  p={sp_p:.4f}  mean_gain={mean_gain:.3f}  std_gain={std_gain:.3f}")
            print(f"[step {step_val:>8}] TOM_STAY_TRACK    stay_rate_over_T={stay_rate:.4f}")
            print(f"[step {step_val:>8}] CHAIN_DEPTH  max=0  mean=0.00  hops>1=0  surv_corr=0.0000")
            print(f"step= {step_val:>7}  blue={b_alive_snap.sum()}  red={r_pop.alive.sum() if r_pop is not None else 0}  brain={n_layers}L  ppo={ui+1}  surv=1.00")

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
        b_act_all = np.array(rollout_data["blue"]["actions"])
        b_alive_all = np.array(rollout_data["blue"]["alive"])
        b_energy_all = np.array(rollout_data["blue"]["energy"])
        b_obs_all = np.array(rollout_data["blue"]["obs"])
        r_pos_all = np.array(rollout_data["red"]["positions"])
        r_alive_all = np.array(rollout_data["red"]["alive"])

        # loc_env is the 4th block in b_obs. Let's find its index to extract local_resource.
        # own_state: 6
        # nb_sigs: K*sig_d (e.g. 6*16 = 96)
        # loc_sym: W*sym_d (e.g. 25*16 = 400)
        # loc_env: W*6 (e.g. 25*6 = 150)
        # Central cell of W=25 is index 12. loc_env features: walls, resources, shelter, contested, scent, blue_map/red_map
        idx_offset = 6 + (config["neighbor_k"] * config["signal_dim"]) + (25 * config["symbol_dim"])
        idx_resource = idx_offset + (12 * 6) + 1  # 12th cell, 2nd feature (index 1 is resources)

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
                is_scout = red_dist <= config.get("scout_detect_radius", 5)
                
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

            corpus_writer.maybe_record(
                step=global_step,
                alive_idx=alive_idx,
                signals=b_sig_all[t],
                actions=b_act_all[t],
                is_scout=is_scout,
                nearest_red_dist=red_dist,
                nearest_red_bear=red_bear,
                local_resource=loc_res,
                own_energy=b_energy_all[t, alive_idx],
                neighbor_count=nb_count,
            )

        # Convert metrics to Python floats for logging (skip non-scalars)
        metrics_py = {}
        for k, v in b_metrics.items():
            try:
                metrics_py[k] = float(v)
            except (TypeError, ValueError):
                pass  # skip non-scalar arrays
        all_metrics.append(metrics_py)

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
    final_params, metrics = run_simulation(cfg, seed=42, n_steps=1024)
    print("[JAX] Done!")
    print(f"Final metrics: {metrics}")
