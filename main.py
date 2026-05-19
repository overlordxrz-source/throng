"""
main.py — THRONG v2 Phase 2: MAPPO

Parameter-sharing MARL:
  - Blues share one policy trained by PPO.
  - Reds share another policy trained by PPO (after red_spawn_step).
  - Both populations co-evolve brain depth via population vote.

Usage:
    python main.py
    python main.py --headless
    python main.py --resume runs/run_xxx/checkpoint_10000.pkl
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

from agents.network_torch import TorchBrain, compute_obs_dim_torch as compute_obs_dim
from agents.population import (
    PopulationState, create_population,
    kill_agent, spawn_agent, inject_random_agent, inject_offspring, expand_brain,
)
from agents.rl import RolloutBuffer
from environment.grid  import ToroidalGrid
from communication.channel  import (
    get_neighbour_signals_padded,
    get_neighbour_indices_padded,
    compute_signal_similarity_pairs,
)
from communication.analysis import (
    CommunicationAnalyser, topographic_similarity, granger_causality_lags,
)
from utils.logging          import RunLogger
from utils.checkpointing    import save_checkpoint, load_checkpoint, find_latest_checkpoint

# Action indices
A_NORTH = 0; A_SOUTH = 1; A_EAST = 2; A_WEST = 3; A_STAY = 4
MOVE_DELTAS = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Observation builder ───────────────────────────────────────────────────────

def build_observations(
    pop:      PopulationState,
    grid:     ToroidalGrid,
    blue_map: np.ndarray,
    red_map:  np.ndarray,
    config:   Dict,
    step:     int = 0,
) -> np.ndarray:
    """Build the flat observation vector for every agent (dead agents get zeros)."""
    gs        = config["grid_size"]
    K         = config["neighbor_k"]
    obs_radius = config["local_obs_radius"]
    max_age   = float(config["max_age"])
    max_l     = float(config.get("brain_max_layers", 6))

    max_pop = pop.max_pop
    sig_dim = config["signal_dim"]
    sym_dim = config.get("symbol_dim", 8)
    W       = (2 * obs_radius + 1) ** 2

    norm_age  = pop.ages.astype(np.float32) / max_age
    norm_x    = pop.positions[:, 0].astype(np.float32) / gs
    norm_y    = pop.positions[:, 1].astype(np.float32) / gs
    mat_th    = float(config["base_maturation_steps"])
    mat_frac  = (pop.ages.astype(np.float32) % mat_th) / mat_th
    nl_norm   = pop.n_layers.astype(np.float32) / max_l
    energy    = pop.energy.astype(np.float32)
    # Phase 4: own_state = 6 dims (added energy), dropped nothing
    own_state = np.stack([norm_age, mat_frac, energy, nl_norm, norm_x, norm_y], axis=1)  # (N, 6)

    # ── Signal gate: randomly mask sensor dims ────────────────────────────
    gate_mask_frac = float(config.get("signal_gate_mask_frac", 0.0))
    if gate_mask_frac > 0:
        # Per-agent unique seed so each agent gets a different blind-spot pattern
        _gate_base = np.uint64(step) ^ np.uint64(0x5EED5EED)
        _agent_seeds = np.arange(max_pop, dtype=np.uint64) ^ _gate_base
        mask_rng = np.random.default_rng(_agent_seeds.tolist())
        mask_own = np.array([np.random.default_rng(int(s)).random(6) for s in _agent_seeds]) < gate_mask_frac
        own_state[mask_own] = 0.0

    nb_sigs = get_neighbour_signals_padded(
        pop.positions, pop.signals, pop.alive, k=K, grid_size=gs,
    )  # (N, K, sig_dim)

    loc_sym  = grid.get_local_symbols(pop.positions, radius=obs_radius)    # (N, W*sym_dim)

    loc_pres = grid.get_local_presence(pop.positions, blue_map, red_map, radius=obs_radius)  # (N, W*2)
    loc_wall = grid.get_local_walls(pop.positions, radius=obs_radius)      # (N, W)
    loc_res  = grid.get_local_resources(pop.positions, radius=obs_radius)  # (N, W)
    # Phase 5: add Gaussian noise to resource observations so agents can't
    # reliably locate food alone — neighbor signals about resources become useful
    resource_noise = float(config.get("resource_obs_noise", 0.0))
    if resource_noise > 0.0:
        loc_res = loc_res + np.random.normal(0.0, resource_noise, loc_res.shape)
        loc_res = np.clip(loc_res, 0.0, 1.0)
    # Phase 4: concat presence + wall + resource into unified env channel (N, W*4)
    loc_env = np.concatenate([loc_pres, loc_wall, loc_res], axis=1).astype(np.float32)

    # Partial observability: blues only see reds within detection_radius cells.
    # Per-agent: zero the red channel of loc_env for those too far from any red.
    # This forces agents that can't see reds to rely on neighbours' signals.
    det_r = int(config.get("red_detection_radius", 0))
    if det_r > 0 and red_map.max() > 0 and pop.alive.any() and pop.team[0] == 0:
        r_cells = np.argwhere(red_map > 0)           # (n_red_cells, 2)
        b_pos   = pop.positions                       # (N, 2)
        diff    = np.abs(b_pos[:, None, :] - r_cells[None, :, :])   # (N, n_rc, 2)
        diff    = np.minimum(diff, gs - diff)         # toroidal
        min_dist = np.maximum(diff[:, :, 0], diff[:, :, 1]).min(axis=1)   # (N,) Chebyshev
        blind    = min_dist > det_r                   # (N,) bool
        W        = (2 * obs_radius + 1) ** 2
        loc_env[blind, W:2*W] = 0.0                   # zero out red channel in loc_env

    # ── Signal gate: randomly mask loc_env red-channel dims ────────────────
    if gate_mask_frac > 0:
        mask_pres = np.array([np.random.default_rng(int(s) + 7).random(W) for s in _agent_seeds]) < gate_mask_frac
        loc_env[:, W:2*W][mask_pres] = 0.0

    obs = np.concatenate([
        own_state,
        nb_sigs.reshape(max_pop, K * sig_dim),
        loc_sym,
        loc_env,
        pop.signals,
    ], axis=1).astype(np.float32)

    obs[~pop.alive] = 0.0
    return obs


# ── Movement ──────────────────────────────────────────────────────────────────

def apply_moves(pop: PopulationState, actions: np.ndarray, gs: int, grid=None) -> None:
    move_mask = pop.alive & (actions < A_STAY)
    if move_mask.any():
        deltas = MOVE_DELTAS[actions[move_mask]]
        new_pos = (pop.positions[move_mask] + deltas) % gs
        # Phase 4: wall collision — if new cell is a wall, stay in place
        if grid is not None:
            wall_mask = grid.is_wall(new_pos)
            new_pos[wall_mask] = pop.positions[move_mask][wall_mask]
        pop.positions[move_mask] = new_pos


# ── Predator catches ──────────────────────────────────────────────────────────

def apply_catches(
    blue_pop:   PopulationState,
    red_pop:    PopulationState,
    config:     Dict,
    blue_rew:   np.ndarray,
    red_rew:    np.ndarray,
) -> None:
    gs        = config["grid_size"]
    catch_r   = int(config.get("red_catch_radius", 0))
    starvation = int(config["red_starvation_steps"])
    r_catch   = float(config["reward_red_catch"])
    b_caught  = float(config["reward_blue_caught"])

    b_idx = np.where(blue_pop.alive)[0]
    r_idx = np.where(red_pop.alive)[0]
    if len(b_idx) == 0 or len(r_idx) == 0:
        return

    b_pos = blue_pop.positions[b_idx]
    r_pos = red_pop.positions[r_idx]

    diff = np.abs(b_pos[:, None, :] - r_pos[None, :, :])
    diff = np.minimum(diff, gs - diff)
    dist = np.maximum(diff[:, :, 0], diff[:, :, 1])
    in_range = dist <= catch_r

    caught_b   = in_range.any(axis=1)
    catching_r = in_range.any(axis=0)

    for idx in b_idx[caught_b]:
        blue_rew[idx] += b_caught
        kill_agent(blue_pop, idx)

    red_rew[r_idx[catching_r]]                       += r_catch
    red_pop.offspring_count[r_idx[catching_r]]       += 1
    red_pop.steps_since_catch[r_idx[catching_r]]      = 0
    red_pop.steps_since_catch[r_idx[~catching_r]]    += 1

    starved = red_pop.alive & (red_pop.steps_since_catch >= starvation)
    red_pop.alive[starved] = False


# ── Reproduction ──────────────────────────────────────────────────────────────

def apply_auto_reproduce(
    pop:      PopulationState,
    config:   Dict,
    rng:      np.random.Generator,
    sig_dim:  int,
    n_layers: int,
    team_id:  int,
    max_pop_override: Optional[int] = None,
) -> PopulationState:
    base_mat = int(config["base_maturation_steps"])
    if max_pop_override is not None:
        max_pop = max_pop_override
    else:
        max_pop = config["population_size"] if team_id == 0 else config["red_population_size"]
    gs       = config["grid_size"]

    alive_idx = np.where(pop.alive & (pop.team == team_id))[0]
    n_alive   = int(len(alive_idx))

    for idx in alive_idx:
        if n_alive >= max_pop:
            break
        age = int(pop.ages[idx])
        if age > 0 and age % base_mat == 0:
            pop, slot = spawn_agent(pop, int(idx), gs, rng, sig_dim, inherit_lineage=True)
            if slot is not None:
                pop.offspring_count[idx] += 1
                n_alive += 1

    return pop


def enforce_population_floor(
    pop:      PopulationState,
    config:   Dict,
    rng:      np.random.Generator,
    sig_dim:  int,
    n_layers: int,
    team_id:  int,
    min_pop_override: Optional[int] = None,
) -> PopulationState:
    if min_pop_override is not None:
        min_pop = min_pop_override
    else:
        min_pop = config["min_population"] if team_id == 0 else config["min_red_population"]
    gs = config["grid_size"]
    while int((pop.alive & (pop.team == team_id)).sum()) < min_pop:
        pop = inject_offspring(pop, gs, rng, n_layers, team_id=team_id)
    return pop


# ── Culture grid analysis ─────────────────────────────────────────────────────

_CULTURE_LAG_STEPS = [1, 5, 10, 15, 20]  # which historical snapshots to test


def culture_metrics(
    grid_symbols:  np.ndarray,   # (gs, gs, sym_dim)
    red_pos_hist:  list,         # up to 20 snapshots of red positions (newest last)
    blue_pop:      "PopulationState",
) -> Dict:
    """
    1. culture_entropy    — grid becoming structured (low H) or noisy (high H)?
    2. culture_red_lag_N  — Pearson r between current culture norms and red positions
                            at lag N steps ago; measured at lags 1,5,10,15,20.
                            Rising r at lag 5-20 = collective danger map written by scouts.
    3. culture_surv_corr  — do agents in high-culture cells survive longer?
    """
    gs, _, sym_dim = grid_symbols.shape
    norms = np.linalg.norm(grid_symbols, axis=-1)           # (gs, gs)
    flat_norm = norms.ravel()

    # 1. Entropy of grid norm distribution
    counts, _ = np.histogram(norms.ravel(), bins=20, range=(0.0, norms.max() + 1e-8))
    p = counts / (counts.sum() + 1e-8)
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    # 2. Per-lag culture–red correlation
    lag_corrs = {}
    n_hist = len(red_pos_hist)
    for lag in _CULTURE_LAG_STEPS:
        hist_idx = n_hist - lag          # index into red_pos_hist (newest = last)
        if hist_idx < 0:
            lag_corrs[f"lag{lag}"] = float("nan")
            continue
        old_red = red_pos_hist[hist_idx]
        if len(old_red) == 0:
            lag_corrs[f"lag{lag}"] = 0.0
            continue
        red_map_old = np.zeros((gs, gs), dtype=np.float32)
        for rx, ry in old_red:
            red_map_old[int(rx) % gs, int(ry) % gs] = 1.0
        flat_red = red_map_old.ravel()
        if flat_red.std() > 1e-6 and flat_norm.std() > 1e-6:
            lag_corrs[f"lag{lag}"] = float(np.corrcoef(flat_red, flat_norm)[0, 1])
        else:
            lag_corrs[f"lag{lag}"] = 0.0

    # 3. Survival correlation
    surv_corr = 0.0
    alive_idx = np.where(blue_pop.alive)[0]
    if len(alive_idx) > 10:
        local_norms = norms[blue_pop.positions[alive_idx, 0],
                            blue_pop.positions[alive_idx, 1]]
        ages = blue_pop.ages[alive_idx].astype(np.float32)
        if local_norms.std() > 1e-6 and ages.std() > 1e-6:
            surv_corr = float(np.corrcoef(local_norms, ages)[0, 1])

    result = {"culture_entropy": entropy, "culture_surv_corr": surv_corr}
    result.update({f"culture_red_{k}": v for k, v in lag_corrs.items()})
    return result


# ── Alarm call propagation ────────────────────────────────────────────────────

def alarm_call_propagation(
    blue_pop:   "PopulationState",
    red_pop:    "PopulationState",
    b_logits:   np.ndarray,        # (max_pop, 5) raw logits before action
    config:     Dict,
) -> Dict:
    """
    Falsifiable test for alarm call communication.

    For each BLIND blue (not near a red), check if it has a SCOUT neighbour
    (a nearby blue that CAN see a red). Compare flee probability:
      P(flee | has scout neighbour with red nearby)  vs
      P(flee | no scout neighbour)

    A delta > 0.15 → alarm calls are real.
    A_STAY = 4; flee = any other action.
    """
    if red_pop is None or not red_pop.alive.any() or not blue_pop.alive.any():
        return {}

    gs     = config["grid_size"]
    det_r  = int(config.get("red_detection_radius", 0))
    K      = config["neighbor_k"]
    A_STAY = 4

    b_alive = np.where(blue_pop.alive)[0]
    r_alive = np.where(red_pop.alive)[0]
    if len(r_alive) == 0:
        return {}

    r_pos = red_pop.positions[r_alive].astype(np.float32)
    b_pos = blue_pop.positions[b_alive].astype(np.float32)

    # Min toroidal Chebyshev distance from each alive blue to any alive red
    diff      = np.abs(b_pos[:, None, :] - r_pos[None, :, :])
    diff      = np.minimum(diff, gs - diff)
    cheb_dist = np.maximum(diff[:, :, 0], diff[:, :, 1])
    nearest_r = cheb_dist.argmin(axis=1)
    min_dist  = cheb_dist.min(axis=1)   # (n_blue,)

    is_scout  = min_dist <= det_r      # can see a red
    is_blind  = ~is_scout              # cannot see a red

    if is_blind.sum() < 2 or is_scout.sum() < 1:
        return {}

    scout_positions = b_pos[is_scout]  # (n_scouts, 2)

    # For each blind agent: does it have a scout within K-neighbour range?
    blind_pos = b_pos[is_blind]        # (n_blind, 2)
    if len(scout_positions) > 0:
        sdiff     = np.abs(blind_pos[:, None, :] - scout_positions[None, :, :])
        sdiff     = np.minimum(sdiff, gs - sdiff)
        scout_dist= np.maximum(sdiff[:, :, 0], sdiff[:, :, 1]).min(axis=1)
        # sig_range = ~k-NN communication radius; 8 cells comfortably covers
        # the ~6.6-cell expected 6th-nearest-neighbour distance at current density.
        sig_range = float(config.get("alarm_scout_range", 8))
        has_scout = scout_dist <= sig_range
    else:
        has_scout = np.zeros(is_blind.sum(), dtype=bool)

    # Require ≥ 20 agents in each group; avoids nan from tiny/empty control groups
    # without distorting the metric by using too tight a radius.
    _min_group = int(config.get("alarm_min_group", 20))

    # Flee probability from logits (softmax then 1 - P(STAY))
    blind_global_idx = b_alive[is_blind]
    logits_blind = b_logits[blind_global_idx]                   # (n_blind, 5)
    probs_blind  = np.exp(logits_blind - logits_blind.max(axis=1, keepdims=True))
    probs_blind /= probs_blind.sum(axis=1, keepdims=True)
    p_flee_blind  = 1.0 - probs_blind[:, A_STAY]                # (n_blind,)

    p_flee_with_scout    = float(p_flee_blind[has_scout].mean())  if has_scout.sum()  >= _min_group else float("nan")
    p_flee_without_scout = float(p_flee_blind[~has_scout].mean()) if (~has_scout).sum() >= _min_group else float("nan")

    signed = r_pos[None, :, :] - b_pos[:, None, :]
    signed = (signed + gs / 2.0) % gs - gs / 2.0
    nearest_vec = signed[np.arange(len(b_alive)), nearest_r][is_blind]
    away_actions = np.full(is_blind.sum(), A_STAY, dtype=np.int64)
    row_dom = np.abs(nearest_vec[:, 0]) >= np.abs(nearest_vec[:, 1])
    away_actions[row_dom & (nearest_vec[:, 0] > 0)] = A_NORTH
    away_actions[row_dom & (nearest_vec[:, 0] < 0)] = A_SOUTH
    away_actions[~row_dom & (nearest_vec[:, 1] > 0)] = A_WEST
    away_actions[~row_dom & (nearest_vec[:, 1] < 0)] = A_EAST
    p_away_blind = probs_blind[np.arange(len(probs_blind)), away_actions]
    p_away_with_scout    = float(p_away_blind[has_scout].mean())  if has_scout.sum()  >= _min_group else float("nan")
    p_away_without_scout = float(p_away_blind[~has_scout].mean()) if (~has_scout).sum() >= _min_group else float("nan")

    away_delta = (p_away_with_scout - p_away_without_scout
                  if not np.isnan(p_away_with_scout) and not np.isnan(p_away_without_scout)
                  else float("nan"))

    delta = (p_flee_with_scout - p_flee_without_scout
             if not np.isnan(p_flee_with_scout) and not np.isnan(p_flee_without_scout)
             else float("nan"))

    return {
        "alarm_delta":          delta,
        "p_flee_scout_nearby":  p_flee_with_scout,
        "p_flee_no_scout":      p_flee_without_scout,
        "away_delta":           away_delta,
        "p_away_scout_nearby":  p_away_with_scout,
        "p_away_no_scout":      p_away_without_scout,
        "n_scouts":             int(is_scout.sum()),
        "n_blind":              int(is_blind.sum()),
    }


# ── Brain growth vote ─────────────────────────────────────────────────────────

def brain_vote(
    pop:          PopulationState,
    config:       Dict,
    current_n:    int,
    recent_alive: list,
) -> int:
    max_l = int(config.get("brain_max_layers", 6))
    if current_n >= max_l:
        return current_n
    threshold = float(config["brain_vote_survival_threshold"])
    if len(recent_alive) < 10:
        return current_n
    mean_survival = float(np.mean(recent_alive))
    if mean_survival < threshold:
        new_n = current_n + 1
        print(f"  [BRAIN VOTE] survival={mean_survival:.2f} < {threshold:.2f} → "
              f"expanding blue brain {current_n}→{new_n} layers")
        return new_n
    return current_n


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(config: Dict, resume_path: Optional[str] = None, headless: bool = False,
        fresh: bool = False, withdrawal: bool = False, blind: bool = False,
        max_steps: int = 0) -> None:
    rng = np.random.default_rng(int(time.time()) & 0xFFFFFFFF)

    blue_n_layers = int(config.get("n_layers", 2))
    red_n_layers  = int(config.get("n_layers", 2))

    from agents.network_torch import DEVICE
    blue_brain = TorchBrain(config)
    red_brain  = TorchBrain(config)

    obs_dim  = compute_obs_dim(config)
    sig_dim  = config["signal_dim"]
    sym_dim  = config.get("symbol_dim", 8)
    gs       = config["grid_size"]
    K        = config["neighbor_k"]
    max_pop_b = config["population_size"]
    max_pop_r = config["red_population_size"]

    # ── Auto-find latest checkpoint if resume_path not given ─────────────────
    if resume_path is None and not fresh:
        resume_path = find_latest_checkpoint(config["log_dir"])
        if resume_path:
            print(f"  [resume] auto-detected checkpoint: {resume_path}")

    grid = ToroidalGrid(size=gs, symbol_dim=sym_dim)

    # Phase 4: generate procedural walls and resource patches on fresh start
    _wall_density = float(config.get("wall_density", 0.08))
    _res_patches  = int(config.get("resource_n_patches", 20))
    if not resume_path:
        grid.generate_walls(rng, density=_wall_density)
        grid.generate_resources(rng, n_patches=_res_patches)
        print(f"[init] walls={int(grid.walls.sum())} cells  resources={grid.resources.sum():.1f} total")

    if resume_path:
        ckpt  = load_checkpoint(resume_path, blue_brain=blue_brain, red_brain=red_brain)
        blue_pop      = ckpt["blue_pop"]
        red_pop       = ckpt["red_pop"]
        blue_n_layers = ckpt["blue_n_layers"]
        red_n_layers  = ckpt["red_n_layers"]
        grid.symbols  = ckpt["grid_symbols"]
        # Phase 4: restore walls and resources from checkpoint
        if ckpt.get("grid_walls") is not None:
            grid.walls = ckpt["grid_walls"]
        else:
            grid.generate_walls(rng, density=_wall_density)
        if ckpt.get("grid_resources") is not None:
            grid.resources = ckpt["grid_resources"]
        else:
            grid.generate_resources(rng, n_patches=_res_patches)
        print(f"  [resume] walls={int(grid.walls.sum())} cells  resources={grid.resources.sum():.1f} total")
        # Phase 4 backward compat: old checkpoints lack energy
        if not hasattr(blue_pop, "energy"):
            blue_pop.energy = np.ones(blue_pop.max_pop, dtype=np.float32)
        if red_pop is not None and not hasattr(red_pop, "energy"):
            red_pop.energy = np.ones(red_pop.max_pop, dtype=np.float32)
        _start_step   = ckpt["step"]
        _start_ppo    = ckpt["ppo_count"]
        # Force brain depth expansion if configured (pre-allocated layers, no reinit)
        _forced_layers = int(config.get("force_brain_layers", 0))
        if _forced_layers > blue_n_layers:
            print(f"  [resume] forcing brain depth {blue_n_layers}L → {_forced_layers}L")
            blue_n_layers = _forced_layers
            expand_brain(blue_pop, blue_n_layers)
        print(f"  [resume] step={_start_step:,}  ppo={_start_ppo}  brain_b={blue_n_layers}L")
    else:
        blue_pop  = create_population(config, rng_seed=42,       grid_size=gs, team_id=0)
        red_pop   = None
        _start_step = 0
        _start_ppo  = 0
        # Ensure no agents spawn inside walls
        for _ in range(5):
            _on_wall = grid.is_wall(blue_pop.positions) & blue_pop.alive
            if not _on_wall.any():
                break
            _open = np.argwhere(~grid.walls)
            _chosen = rng.integers(0, len(_open), size=int(_on_wall.sum()))
            blue_pop.positions[_on_wall] = _open[_chosen]

    _K       = int(config["neighbor_k"])
    _hd      = int(config["agent_hidden_dim"])
    blue_buf = RolloutBuffer(config["ppo_rollout_steps"], max_pop_b, obs_dim,
                             neighbor_k=_K, hidden_dim=_hd)
    red_buf  = None if red_pop is None else RolloutBuffer(
        config["ppo_rollout_steps"], max_pop_r, obs_dim, hidden_dim=_hd
    )

    analyser = CommunicationAnalyser(
        signal_dim        = sig_dim,
        analysis_interval = config["mi_analysis_interval"],
        window            = config["mi_window"],
        cluster_k         = config["signal_cluster_k"],
    )

    if resume_path:
        # Resume into existing run directory so logs/science.log stay contiguous
        _resume_run_id = Path(resume_path).parent.name
        logger = RunLogger(config["log_dir"], run_id=_resume_run_id, headless=headless)
    else:
        logger = RunLogger(config["log_dir"], headless=headless)
    logger.log_run_start(config, step=_start_step)
    ckpt_dir = logger.run_directory

    from communication.analysis import SignalCorpusWriter
    corpus_writer = SignalCorpusWriter(
        path          = str(ckpt_dir / "signal_corpus.jsonl"),
        sample_frac   = float(config.get("corpus_sample_frac", 0.08)),
        every_n_steps = int(config.get("corpus_every_n_steps", 20)),
        rng           = np.random.default_rng(99),
    )

    renderer  = None
    dashboard = None
    if not headless:
        try:
            from visualization.renderer  import Renderer
            from visualization.dashboard import DashboardProcess
            renderer  = Renderer(config["window_size"], gs)
            dashboard = DashboardProcess(update_interval=config["dashboard_update_interval"])
            dashboard.start()
        except Exception as e:
            print(f"[warn] Visualisation unavailable: {e}")

    step      = _start_step
    ppo_count = _start_ppo
    sci_log   = open(ckpt_dir / "science.log", "a", buffering=1)

    def slog(msg):
        print(msg)
        sci_log.write(msg + "\n")

    recent_blue_survival = []
    red_pos_hist         = []   # last ~10 snapshots of red positions (for culture lag)

    # Curriculum state
    _curr_red_floor      = int(config.get("curriculum_red_count", config["min_red_population"]))
    _curr_full_floor     = int(config["min_red_population"])
    _curr_threshold      = int(config.get("curriculum_blue_threshold", 60))
    _curr_sustain        = int(config.get("curriculum_sustain_steps", 2000))
    _curr_sustain_count  = 0    # consecutive steps blues >= threshold
    _curriculum_active   = red_pop is not None  # start in curriculum if reds exist at resume
    steps_per_render     = int(config.get("steps_per_render", 32))
    target_fps           = int(config.get("target_fps", 60))

    # Lag-1 scout state buffer for cross-agent Granger test in corpus
    _lag1_scout_pos  = None   # (n_scouts, 2) positions from T-1
    _lag1_scout_sig  = None   # (n_scouts, sig_dim) signals from T-1
    _lag1_scout_dist = None   # (n_scouts,) nearest-red distance from T-1
    _alarm_range    = float(config.get("alarm_scout_range", 8))

    # Rolling 10-step signal receipt log for group survival bonus
    # _receipt_log[t % 10] = (N, K) int array of broadcaster global indices
    _receipt_log = [np.full((max_pop_b, _K), -1, dtype=np.int32) for _ in range(10)]

    # Echo memory: one slot per agent, slow-decay echo of last significant signal
    _echo_memory = np.zeros((max_pop_b, sig_dim), dtype=np.float32)

    # Chain depth tracking: per-agent, how many hops since signal originated
    _chain_depth = np.zeros(max_pop_b, dtype=np.int32)

    # ── 145k withdrawal test (Cam) ──────────────────────────────────────────
    _wd_step_start = 145000
    _wd_step_end   = 147000
    _wd_active     = False
    _wd_stay_log   = []   # list of (step, stay_rate) during withdrawal
    # Dedicated 500-step buffer for mid-withdrawal Granger
    _wd_obs_buf    = []   # list of (N, obs_dim) arrays
    _wd_act_buf    = []   # list of (N,) action arrays
    _wd_aliv_buf   = []   # list of (N,) alive arrays
    _wd_buf_max    = 500

    slog(f"THRONG v2 MAPPO — {ckpt_dir}")
    slog(f"  blues={max_pop_b} brain={blue_n_layers}L  obs_dim={obs_dim}  device={DEVICE}")

    _last_buf_snapshot = None   # set when PPO fires; used by Granger + ToM tracker

    try:
        running = True
        while running:
            for _sub in range(steps_per_render):
                step += 1

                # ── Dynamic hyperparameter anneal ───────────────────────────
                # Gradually harden token distribution as vocabulary establishes
                _tau_start = float(config.get("gumbel_tau_start", 1.5))
                _tau_end   = float(config.get("gumbel_tau_end", 0.5))
                _tau_s0    = int(config.get("gumbel_tau_anneal_start", 20000))
                _tau_s1    = int(config.get("gumbel_tau_anneal_end", 50000))
                if step <= _tau_s0:
                    _tau = _tau_start
                elif step >= _tau_s1:
                    _tau = _tau_end
                else:
                    _tau = _tau_start - (_tau_start - _tau_end) * (step - _tau_s0) / (_tau_s1 - _tau_s0)
                blue_brain.model.gumbel_tau = _tau
                if red_brain is not None:
                    red_brain.model.gumbel_tau = _tau

                # Anneal signal entropy coef: strong early → weak late
                _se_start = float(config.get("signal_entropy_coef_start", 0.02))
                _se_end   = float(config.get("signal_entropy_coef_end", 0.005))
                _se_s0    = int(config.get("signal_entropy_anneal_start", 20000))
                _se_s1    = int(config.get("signal_entropy_anneal_end", 50000))
                if step <= _se_s0:
                    _se = _se_start
                elif step >= _se_s1:
                    _se = _se_end
                else:
                    _se = _se_start - (_se_start - _se_end) * (step - _se_s0) / (_se_s1 - _se_s0)
                blue_brain.config["signal_entropy_coef"] = _se
                if red_brain is not None:
                    red_brain.config["signal_entropy_coef"] = _se

                if max_steps > 0 and step >= _start_step + max_steps:
                    running = False
                    break

                # ── Build presence maps ───────────────────────────────────────
                blue_map, _ = grid.build_presence_maps(
                    blue_pop.positions, blue_pop.alive, blue_pop.team
                )
                red_map = np.zeros((gs, gs), dtype=np.float32)
                if red_pop is not None and red_pop.alive.any():
                    _, rm = grid.build_presence_maps(
                        red_pop.positions, red_pop.alive, red_pop.team
                    )
                    red_map = rm.astype(np.float32)

                # ── Blue forward pass (TorchBrain) ────────────────────────────
                b_obs = build_observations(blue_pop, grid, blue_map, red_map, config, step)
                # ── Communication bottleneck test: zero own observation ────────
                if blind:
                    b_obs = np.zeros_like(b_obs)
                # Snapshot the carry BEFORE forward — this is the recurrent
                # state that produces b_logits at this step. PPO will replay
                # the policy on (b_carry_before, b_obs) so the gradient is
                # computed under the same recurrent state the rollout used.
                b_carry_before = blue_pop.carries.copy()
                b_new_c, b_logits_np, b_sigs_np, b_sym_w_np, b_vals_np, b_tom_np, b_token_ids = blue_brain.forward(
                    blue_pop.carries, b_obs, blue_n_layers, blue_pop.nb_gain
                )
                blue_pop.carries = b_new_c

                # ── Echo memory: mix slow-decay echo into signal output ──────
                # Decay existing echo memory
                _echo_memory *= 0.9
                # Determine which agents received a significant new signal
                _nb_start_e = 6
                _nb_end_e = 6 + _K * sig_dim
                _b_nb_raw = b_obs[:, _nb_start_e:_nb_end_e].reshape(max_pop_b, _K, sig_dim)
                _nb_max_abs = np.abs(_b_nb_raw).max(axis=2).max(axis=1)  # (N,) max |dim| across all K neighbours
                _sig_thresh = _nb_max_abs > 0.3
                if _sig_thresh.any():
                    # Overwrite echo with mean of significant neighbour signals
                    _echo_memory[_sig_thresh] = _b_nb_raw[_sig_thresh].mean(axis=1)
                    # Reset chain depth for agents hearing a fresh signal
                    _chain_depth[_sig_thresh] = 0
                # Dead agents: reset echo
                _echo_memory[~blue_pop.alive] = 0.0
                _chain_depth[~blue_pop.alive] = 0
                # Mix: 0.7 current + 0.3 echo
                blue_pop.signals = 0.7 * b_sigs_np + 0.3 * _echo_memory
                # Agents rebroadcasting echo get chain_depth + 1
                _echo_rebroadcasting = (~_sig_thresh) & blue_pop.alive & (np.abs(_echo_memory).max(axis=1) > 0.01)
                _chain_depth[_echo_rebroadcasting] += 1

                # ── Record signal receipt log (for group survival bonus) ─────
                _b_neigh_idx_receipt = get_neighbour_indices_padded(
                    blue_pop.positions, blue_pop.alive, _K, gs
                )  # (N, K) global indices of K nearest neighbours
                _receipt_log[step % 10] = _b_neigh_idx_receipt.copy()

                # ── Withdrawal window at 145k (Cam) ────────────────────────────
                if step == _wd_step_start:
                    _wd_active = True
                    slog(f"[step {step:>8,}] === WITHDRAWAL START (2k steps, signals zeroed) ===")
                elif step == _wd_step_end:
                    _wd_active = False
                    slog(f"[step {step:>8,}] === WITHDRAWAL END ===")

                if withdrawal or _wd_active:
                    blue_pop.signals = np.zeros_like(b_sigs_np)

                # Track red positions for culture-lag correlation
                if red_pop is not None and red_pop.alive.any():
                    red_pos_hist.append(red_pop.positions[red_pop.alive].copy())
                    if len(red_pos_hist) > 20:
                        red_pos_hist.pop(0)

                b_log_probs_all = b_logits_np - np.log(
                    np.sum(np.exp(b_logits_np - b_logits_np.max(axis=-1, keepdims=True)),
                           axis=-1, keepdims=True)
                ) - b_logits_np.max(axis=-1, keepdims=True)

                b_actions = np.array([
                    rng.choice(5, p=np.exp(b_log_probs_all[i]))
                    if blue_pop.alive[i] else A_STAY
                    for i in range(max_pop_b)
                ], dtype=np.int32)

                b_log_probs_taken = b_log_probs_all[np.arange(max_pop_b), b_actions]
                b_actions[~blue_pop.alive] = A_STAY

                # ── Withdrawal: log STAY rate every 100 steps ───────────────
                if _wd_active and step % 100 == 0:
                    _alive_mask_wd = blue_pop.alive
                    _n_alive_wd = int(_alive_mask_wd.sum())
                    _stay_rate = 0.0
                    if _n_alive_wd > 0:
                        _stay_rate = float((b_actions[_alive_mask_wd] == A_STAY).sum()) / _n_alive_wd
                    slog(f"[step {step:>8,}] WITHDRAWAL_STAY  rate={_stay_rate:.4f}  n_alive={_n_alive_wd}")

                # ── Withdrawal: accumulate 500-step buffer for mid-Granger ────
                if _wd_active:
                    _wd_obs_buf.append(b_obs.copy())
                    _wd_act_buf.append(b_actions.copy())
                    _wd_aliv_buf.append(blue_pop.alive.copy())
                    if len(_wd_obs_buf) > _wd_buf_max:
                        _wd_obs_buf.pop(0)
                        _wd_act_buf.pop(0)
                        _wd_aliv_buf.pop(0)

                # ── Signal corpus sampling ────────────────────────────────────
                _corp_alive = np.where(blue_pop.alive)[0]
                if len(_corp_alive) > 0:
                    _corp_pos  = blue_pop.positions[_corp_alive]
                    _corp_dist = np.full(len(_corp_alive), np.inf, dtype=np.float32)
                    _corp_bear = np.full(len(_corp_alive), float("nan"), dtype=np.float32)
                    _corp_scout= np.zeros(len(_corp_alive), dtype=bool)
                    if red_pop is not None and red_pop.alive.any():
                        _rp = red_pop.positions[red_pop.alive].astype(np.float32)
                        _bp = _corp_pos.astype(np.float32)
                        _dd = np.abs(_bp[:, None, :] - _rp[None, :, :])
                        _dd = np.minimum(_dd, gs - _dd)
                        _cheb = np.maximum(_dd[:, :, 0], _dd[:, :, 1])
                        _nr   = _cheb.argmin(axis=1)
                        _corp_dist  = _cheb.min(axis=1).astype(np.float32)
                        _corp_scout = _corp_dist <= int(config.get("red_detection_radius", 0))
                        _vec = (_rp[_nr] - _bp + gs / 2.0) % gs - gs / 2.0
                        _corp_bear = (np.degrees(
                            np.arctan2(_vec[:, 1], _vec[:, 0])
                        ) % 360).astype(np.float32)
                    _corp_res = np.clip(
                        np.linalg.norm(
                            grid.symbols[_corp_pos[:, 0], _corp_pos[:, 1]], axis=1
                        ) / (sym_dim ** 0.5 + 1e-8), 0, 1
                    ).astype(np.float32)
                    _corp_nrg = blue_pop.energy[_corp_alive]
                    _corp_nb  = np.clip(
                        blue_map[_corp_pos[:, 0], _corp_pos[:, 1]] / 5.0, 0, 1
                    ).astype(np.float32)
                    # Compute lag-1 scout signal and scout red-dist for each agent
                    _nb_scout_lag1      = np.full((len(_corp_alive), sig_dim), np.nan, dtype=np.float32)
                    _nb_scout_dist_lag1 = np.full(len(_corp_alive), np.nan, dtype=np.float32)
                    if _lag1_scout_pos is not None and len(_lag1_scout_pos) > 0:
                        _bp2 = _corp_pos.astype(np.float32)
                        _sp2 = _lag1_scout_pos.astype(np.float32)
                        _dd2 = np.abs(_bp2[:, None, :] - _sp2[None, :, :])
                        _dd2 = np.minimum(_dd2, gs - _dd2)
                        _sc2 = np.maximum(_dd2[:, :, 0], _dd2[:, :, 1])  # (n_alive, n_scouts)
                        for _ai in range(len(_corp_alive)):
                            _within = _sc2[_ai] <= _alarm_range
                            if _within.any():
                                _nb_scout_lag1[_ai]      = _lag1_scout_sig[_within].mean(axis=0)
                                _nb_scout_dist_lag1[_ai] = _lag1_scout_dist[_within].mean()
                    corpus_writer.maybe_record(
                        step               = step,
                        alive_idx          = _corp_alive,
                        signals            = blue_pop.signals,
                        actions            = b_actions,
                        is_scout           = _corp_scout,
                        nearest_red_dist   = _corp_dist,
                        nearest_red_bear   = _corp_bear,
                        local_resource     = _corp_res,
                        own_energy         = _corp_nrg,
                        neighbor_count     = _corp_nb,
                        nb_scout_sig_lag1  = _nb_scout_lag1,
                        nb_scout_dist_lag1 = _nb_scout_dist_lag1,
                    )
                    # Update lag-1 buffer with this step's scout state
                    if _corp_scout.any():
                        _lag1_scout_pos  = _corp_pos[_corp_scout].copy()
                        _lag1_scout_sig  = blue_pop.signals[_corp_alive[_corp_scout]].copy()
                        _lag1_scout_dist = _corp_dist[_corp_scout].copy()
                    else:
                        _lag1_scout_pos  = None
                        _lag1_scout_sig  = None
                        _lag1_scout_dist = None

                # ── Theory-of-mind reward bonus ───────────────────────────────
                _b_neigh_idx  = get_neighbour_indices_padded(
                    blue_pop.positions, blue_pop.alive, _K, gs
                )  # (N, K) global indices
                # Safe neighbor action lookup: pad slots (_b_neigh_idx==-1) get
                # sentinel -1 so cross-entropy can ignore them via ignore_index.
                _safe_idx      = np.clip(_b_neigh_idx, 0, max_pop_b - 1)
                _b_tom_targets = np.where(
                    _b_neigh_idx >= 0,
                    b_actions[_safe_idx],
                    -1,
                ).astype(np.int64)  # (N, K) — -1 for missing neighbours
                _b_tom_lp = (
                    b_tom_np
                    - np.log(np.sum(np.exp(b_tom_np - b_tom_np.max(axis=-1, keepdims=True)),
                                    axis=-1, keepdims=True))
                    - b_tom_np.max(axis=-1, keepdims=True)
                )  # log_softmax, shape (N, K, 5)
                _valid_tgt   = np.clip(_b_tom_targets, 0, 4)   # safe index (ignore -1 slots)
                _b_tom_lp_nb = _b_tom_lp[np.arange(max_pop_b)[:, None],
                                          np.arange(_K)[None, :],
                                          _valid_tgt]           # (N, K)
                # ToM reward = mutual-information-style score: log p(actual)
                # minus the uniform-prior baseline log(1/5). This pays only
                # for ABOVE-CHANCE prediction so a monoculture (where everyone
                # is trivially predictable AND signals are useless) gets zero
                # bonus instead of being maximally rewarded. Coefficient kept
                # well below the survival reward (+0.05/step) so it informs
                # selection without dominating it.
                _baseline_lp = float(np.log(1.0 / 5.0))   # = -1.6094
                _b_tom_bonus = (
                    np.where(_b_tom_targets >= 0,
                             _b_tom_lp_nb - _baseline_lp, 0.0)
                    .sum(axis=1) * float(config.get("tom_reward_coef", 0.002))
                )  # (N,) — informativeness bonus; coef from config (0 = scaffold withdrawn)

                apply_moves(blue_pop, b_actions, gs, grid=grid)

                grid.write_symbols(
                    blue_pop.positions, b_sym_w_np, blue_pop.alive
                )
                blue_pop.ages[blue_pop.alive] += 1

                # Phase 4: energy mechanics
                _energy_decay = float(config.get("energy_decay", 0.001))
                blue_pop.energy[blue_pop.alive] -= _energy_decay
                np.clip(blue_pop.energy, 0.0, 1.0, out=blue_pop.energy)
                # Consume resources
                _energy_gained = grid.consume_resources(blue_pop.positions, blue_pop.alive)
                blue_pop.energy[blue_pop.alive] += _energy_gained[blue_pop.alive]
                np.clip(blue_pop.energy, 0.0, 1.0, out=blue_pop.energy)
                # Starvation
                _starve_thresh = float(config.get("starvation_threshold", 0.05))
                _starved = blue_pop.alive & (blue_pop.energy < _starve_thresh)
                if _starved.any():
                    blue_pop.alive[_starved] = False
                    b_rew[_starved] += float(config.get("reward_starvation", -0.5))

                old_age_deaths = blue_pop.alive & (blue_pop.ages >= config["max_age"])
                blue_pop.alive[old_age_deaths] = False

                # ── Blue step rewards ─────────────────────────────────────────
                b_rew  = np.where(blue_pop.alive, config["reward_blue_alive"], 0.0).astype(np.float32)
                b_rew += (_b_tom_bonus * blue_pop.alive.astype(np.float32)).astype(np.float32)

                # ── Movement reward: tiny bonus for non-STAY to break freeze attractor (Cam)
                _move_reward = float(config.get("reward_move", 0.0))
                if _move_reward > 0:
                    b_rew += _move_reward * (b_actions < A_STAY).astype(np.float32) * blue_pop.alive.astype(np.float32)

                # ── Group survival bonus: 5% of survival reward to recent broadcasters
                _surv_bonus_frac = 0.05
                _alive_now = np.where(blue_pop.alive)[0]
                if len(_alive_now) > 0:
                    # Vectorized: gather all broadcaster IDs from receipt logs for alive agents
                    _all_bcasts = np.concatenate([_rl[_alive_now].ravel() for _rl in _receipt_log])
                    _all_bcasts = _all_bcasts[_all_bcasts >= 0]  # remove -1 padding
                    if len(_all_bcasts) > 0:
                        _unique_bcasts = np.unique(_all_bcasts)
                        _valid_bc = _unique_bcasts[(_unique_bcasts < max_pop_b) & blue_pop.alive[_unique_bcasts]]
                        _base_surv_rew = float(config["reward_blue_alive"])
                        b_rew[_valid_bc] += _surv_bonus_frac * _base_surv_rew

                b_done = np.zeros(max_pop_b, dtype=np.float32)
                b_done[old_age_deaths] = 1.0

                # ── Red forward pass ──────────────────────────────────────────
                r_rew = np.zeros(max_pop_r, dtype=np.float32) if red_pop is not None else None
                r_carry_before = (red_pop.carries.copy()
                                  if red_pop is not None else None)
                if red_pop is not None and red_pop.alive.any():
                    bm2, _ = grid.build_presence_maps(
                        blue_pop.positions, blue_pop.alive, blue_pop.team
                    )
                    _, rm2 = grid.build_presence_maps(
                        red_pop.positions, red_pop.alive, red_pop.team
                    )
                    r_obs = build_observations(red_pop, grid, bm2, rm2.astype(np.float32), config, step)
                    r_new_c, r_logits_np, r_sigs_np, r_sym_w_np, r_vals_np, _, _ = red_brain.forward(
                        red_pop.carries, r_obs, red_n_layers, red_pop.nb_gain
                    )
                    red_pop.carries = r_new_c
                    red_pop.signals = r_sigs_np

                    r_log_probs_all = r_logits_np - np.log(
                        np.sum(np.exp(r_logits_np - r_logits_np.max(axis=-1, keepdims=True)),
                               axis=-1, keepdims=True)
                    ) - r_logits_np.max(axis=-1, keepdims=True)

                    r_actions = np.array([
                        rng.choice(5, p=np.exp(r_log_probs_all[i]))
                        if red_pop.alive[i] else A_STAY
                        for i in range(max_pop_r)
                    ], dtype=np.int32)
                    r_actions[~red_pop.alive] = A_STAY

                    apply_moves(red_pop, r_actions, gs, grid=grid)
                    grid.write_symbols(
                        red_pop.positions, r_sym_w_np, red_pop.alive
                    )
                    red_pop.ages[red_pop.alive] += 1
                    # Phase 4: reds also have energy (simpler: just decay, no resource consumption for reds)
                    _r_energy_decay = float(config.get("energy_decay", 0.001))
                    if hasattr(red_pop, "energy"):
                        red_pop.energy[red_pop.alive] -= _r_energy_decay
                        np.clip(red_pop.energy, 0.0, 1.0, out=red_pop.energy)
                        _r_starve_thresh = float(config.get("starvation_threshold", 0.05))
                        _r_starved = red_pop.alive & (red_pop.energy < _r_starve_thresh)
                        red_pop.alive[_r_starved] = False
                    red_pop.alive[red_pop.alive & (red_pop.ages >= config["max_age"])] = False
                    r_rew += np.where(red_pop.alive, config["reward_red_starve_per_step"], 0.0)

                # ── Catch detection ───────────────────────────────────────────
                if red_pop is not None:
                    apply_catches(blue_pop, red_pop, config, b_rew, r_rew)

                # ── Rollout collection ────────────────────────────────────────
                b_alive_before = np.array(blue_pop.alive, dtype=np.float32)
                b_done_final   = np.zeros(max_pop_b, dtype=np.float32)
                b_done_final[~blue_pop.alive] = 1.0

                _wu_steps   = int(config.get("ppo_warmup_steps", 10))
                b_warmup_ok = (blue_pop.ages >= _wu_steps).astype(np.float32)

                blue_buf.push(
                    obs         = b_obs,
                    actions     = b_actions,
                    log_probs   = b_log_probs_taken.astype(np.float32),
                    values      = b_vals_np.astype(np.float32),
                    rewards     = b_rew,
                    dones       = b_done_final,
                    alive       = np.where(b_alive_before, 1.0, 0.0).astype(np.float32),
                    warmup_ok   = b_warmup_ok,
                    tom_targets = _b_tom_targets,
                    carries     = b_carry_before,
                )

                if red_buf is not None and red_pop is not None:
                    r_done_f = np.zeros(max_pop_r, dtype=np.float32)
                    r_done_f[~red_pop.alive] = 1.0
                    r_alive_before = (red_pop.alive).astype(np.float32)
                    r_warmup_ok = (red_pop.ages >= _wu_steps).astype(np.float32)
                    red_buf.push(
                        obs        = r_obs if red_pop.alive.any() else np.zeros((max_pop_r, obs_dim), dtype=np.float32),
                        actions    = r_actions if red_pop.alive.any() else np.zeros(max_pop_r, dtype=np.int32),
                        log_probs  = (r_log_probs_all[np.arange(max_pop_r), r_actions]
                                      if red_pop.alive.any()
                                      else np.zeros(max_pop_r, dtype=np.float32)),
                        values     = r_vals_np if red_pop.alive.any() else np.zeros(max_pop_r, dtype=np.float32),
                        rewards    = r_rew,
                        dones      = r_done_f,
                        alive      = r_alive_before,
                        warmup_ok  = r_warmup_ok,
                        carries    = r_carry_before,
                    )

                # ── PPO update ────────────────────────────────────────────────
                if blue_buf.full:
                    b_last_obs = build_observations(blue_pop, grid, blue_map, red_map, config, step)
                    _, _, _, _, b_last_val_np, _, _ = blue_brain.forward(
                        blue_pop.carries, b_last_obs, blue_n_layers
                    )
                    b_last_val_np *= blue_pop.alive.astype(np.float32)

                    ppo_stats = blue_brain.ppo_update(
                        blue_buf.get(), b_last_val_np, blue_n_layers, rng
                    )
                    ppo_count += 1
                    _last_buf_snapshot = blue_buf.get()   # snapshot before reset
                    blue_buf.reset()

                    if ppo_count % 5 == 0:
                        _tom_str = ""
                        if "tom_loss" in ppo_stats:
                            _tom_str = f"  tom={ppo_stats['tom_loss']:.4f}"
                        if "tom_acc_per_action" in ppo_stats:
                            _acc = ppo_stats["tom_acc_per_action"]
                            _labels = ["stay", "N", "S", "E", "W"]
                            _acc_str = "  ".join(
                                f"{_labels[i]}={_acc[i]:.2f}" for i in range(5)
                            )
                            _tom_str += f"\n[step {step:>8,}] TOM_ACC  {_acc_str}"
                        slog(f"[step {step:>8,}] PPO#{ppo_count}  "
                             f"pg={ppo_stats['ppo_pg_loss']:.4f}  "
                             f"vf={ppo_stats['ppo_vf_loss']:.4f}  "
                             f"ent={ppo_stats['ppo_entropy']:.4f}  "
                             f"clip={ppo_stats['ppo_clip_frac']:.2f}  "
                             f"sig_ent={ppo_stats.get('ppo_sig_ent', 0):.4f}  "
                             f"tau={blue_brain.model.gumbel_tau:.3f}"
                             f"{_tom_str}")

                    # Granger causality every 10 PPO cycles (cross-agent)
                    if ppo_count % 10 == 0 and red_pop is not None:
                        _bd = _last_buf_snapshot
                        _g_obs  = _bd["obs"]          # (T, N, obs_dim)
                        _g_acts = _bd["actions"]       # (T, N)
                        _g_aliv = _bd["alive"].astype(bool)
                        # Neighbour signals: obs dims 6..6+K*sig_dim (from build_observations)
                        _nb_start = 6
                        _nb_end = 6 + config["neighbor_k"] * sig_dim
                        _g_nb_sigs = _g_obs[:, :, _nb_start:_nb_end]  # (T, N, K*sig_dim)
                        _g_res  = granger_causality_lags(
                            _g_nb_sigs, _g_acts, _g_aliv,
                            lags=[1, 2, 3, 5, 10],
                        )
                        _g_str = "  ".join(
                            f"k{k}={v:.2f}" if not (v != v) else f"k{k}=nan"
                            for k, v in sorted(_g_res.items())
                        )
                        slog(f"[step {step:>8,}] GRANGER  {_g_str}")

                if red_buf is not None and red_buf.full and red_pop is not None:
                    r_last_obs = build_observations(red_pop, grid, blue_map, red_map, config, step)
                    _, _, _, _, r_last_val_np, _, _ = red_brain.forward(
                        red_pop.carries, r_last_obs, red_n_layers, red_pop.nb_gain
                    )
                    r_last_val_np *= red_pop.alive.astype(np.float32)
                    red_brain.ppo_update(red_buf.get(), r_last_val_np, red_n_layers, rng)
                    red_buf.reset()

                # ── Within-lifetime Hebbian learning on signal reception gain ─
                _hebb_lr = float(config.get("hebb_lr", 0.0))
                if _hebb_lr > 0:
                    _nb_start = 6
                    _nb_end = 6 + K * sig_dim
                    _b_nb_sigs = b_obs[:, _nb_start:_nb_end].reshape(max_pop_b, K, sig_dim)
                    _sig_exp = np.abs(_b_nb_sigs).mean(axis=(1, 2))  # (N,) mean |signal|
                    _survived = blue_pop.alive
                    blue_pop.nb_gain[_survived] += _hebb_lr * _sig_exp[_survived]
                    blue_pop.nb_gain[~_survived] = 1.0  # reset for dead agents
                    blue_pop.nb_gain = np.clip(blue_pop.nb_gain, 0.5, 3.0)

                # ── nb_gain × survival correlation (Spearman, every 200 steps) ─
                if _hebb_lr > 0 and step % 200 == 0:
                    _alive_mask = blue_pop.alive
                    if _alive_mask.sum() > 10:
                        from scipy.stats import spearmanr
                        _nb_g = blue_pop.nb_gain[_alive_mask]
                        _ages = blue_pop.ages[_alive_mask].astype(np.float32)
                        _sp_r, _sp_p = spearmanr(_nb_g, _ages)
                        slog(f"[step {step:>8,}] NB_GAIN_SURV  "
                             f"spearman_r={_sp_r:.4f}  p={_sp_p:.4f}  "
                             f"mean_gain={_nb_g.mean():.3f}  "
                             f"std_gain={_nb_g.std():.3f}")

                # ── ToM stay accuracy tracker (every 200 steps, Cam) ──────────
                if step % 200 == 0:
                    _tom_log = ""
                    if blue_buf.full and _last_buf_snapshot is not None:
                        _bd_tom = _last_buf_snapshot
                        _g_acts_t = _bd_tom["actions"]       # (T, N)
                        _g_aliv_t = _bd_tom["alive"].astype(bool)
                        _stay_rate_t = 0.0
                        _n_t = 0
                        for _tt in range(_g_acts_t.shape[0]):
                            _mask_t = _g_aliv_t[_tt]
                            if _mask_t.sum() > 0:
                                _stay_rate_t += float((_g_acts_t[_tt][_mask_t] == A_STAY).sum()) / _mask_t.sum()
                                _n_t += 1
                        if _n_t > 0:
                            _stay_rate_t /= _n_t
                            _tom_log = f"  stay_rate_over_T={_stay_rate_t:.4f}"
                    slog(f"[step {step:>8,}] TOM_STAY_TRACK  {_tom_log}")

                # ── Mid-withdrawal 500-step Granger (step 145500, Cam) ────────
                if step == 145500 and len(_wd_obs_buf) == _wd_buf_max:
                    _wd_obs_arr = np.array(_wd_obs_buf)      # (500, N, obs_dim)
                    _wd_act_arr = np.array(_wd_act_buf)      # (500, N)
                    _wd_aliv_arr = np.array(_wd_aliv_buf).astype(bool)  # (500, N)
                    _nb_start_wd = 6
                    _nb_end_wd   = 6 + config["neighbor_k"] * sig_dim
                    _wd_nb_sigs = _wd_obs_arr[:, :, _nb_start_wd:_nb_end_wd]
                    _wd_gr = granger_causality_lags(
                        _wd_nb_sigs, _wd_act_arr, _wd_aliv_arr,
                        lags=[1, 2, 3, 5, 10],
                    )
                    _wd_g_str = "  ".join(
                        f"k{k}={v:.2f}" if not (v != v) else f"k{k}=nan"
                        for k, v in sorted(_wd_gr.items())
                    )
                    slog(f"[step {step:>8,}] WITHDRAWAL_GRANGER  {_wd_g_str}")

                # ── Chain depth metric (every 200 steps) ─────────────────────
                if step % 200 == 0:
                    _alive_cd = blue_pop.alive
                    if _alive_cd.sum() > 0:
                        _cd_alive = _chain_depth[_alive_cd]
                        _cd_max = int(_cd_alive.max())
                        _cd_mean = float(_cd_alive.mean())
                        _cd_gt1 = int((_cd_alive > 1).sum())
                        # Correlation between chain depth and age (survival)
                        _cd_surv_r = 0.0
                        if _cd_alive.std() > 1e-6 and _alive_cd.sum() > 10:
                            from scipy.stats import spearmanr as _sp
                            _cd_surv_r, _ = _sp(_cd_alive, blue_pop.ages[_alive_cd].astype(np.float32))
                        slog(f"[step {step:>8,}] CHAIN_DEPTH  "
                             f"max={_cd_max}  mean={_cd_mean:.2f}  "
                             f"hops>1={_cd_gt1}  surv_corr={_cd_surv_r:.4f}")

                # ── Symbol decay + resource regeneration ────────────────────
                grid.decay_symbols(config["symbol_decay"])
                grid.regenerate_resources(rng, step,
                    regen_rate=float(config.get("resource_regen_rate", 0.002)),
                    n_patches=int(config.get("resource_n_patches", 20)),
                )

                # ── Reproduction ──────────────────────────────────────────────
                blue_pop = apply_auto_reproduce(blue_pop, config, rng, sig_dim, blue_n_layers, 0)
                if red_pop is not None:
                    _eff_red_cap = _curr_red_floor if _curriculum_active else _curr_full_floor
                    red_pop = apply_auto_reproduce(
                        red_pop, config, rng, sig_dim, red_n_layers, 1,
                        max_pop_override=_eff_red_cap,
                    )

                # ── Population floors ─────────────────────────────────────────
                blue_pop = enforce_population_floor(
                    blue_pop, config, rng, sig_dim, blue_n_layers, team_id=0
                )

                # Curriculum: track blue recovery, graduate when stable
                if red_pop is not None:
                    _nb_cur = int(blue_pop.alive.sum())
                    if _curriculum_active and _nb_cur >= _curr_threshold:
                        _curr_sustain_count += 1
                        if _curr_sustain_count >= _curr_sustain:
                            _curriculum_active  = False
                            _curr_sustain_count = 0
                            slog(f"[step {step:>8,}] CURRICULUM GRADUATED — "
                                 f"blues sustained >{_curr_threshold} for {_curr_sustain} steps. "
                                 f"Red floor: {_curr_red_floor}→{_curr_full_floor}")
                            # Write graduation marker so the user knows to launch config_large.yaml
                            _grad_flag = Path(ckpt_dir) / "GRADUATED.flag"
                            _grad_flag.write_text(
                                f"step={step}  ppo={ppo_count}  "
                                f"blue_alive={int(blue_pop.alive.sum())}\n"
                                f"Next: python main.py --config config_large.yaml\n"
                            )
                            if config.get("exit_on_graduation", False):
                                slog("[ckpt] exit_on_graduation=true — saving checkpoint and exiting.")
                                save_checkpoint(
                                    ckpt_dir=ckpt_dir, step=step, ppo_count=ppo_count,
                                    blue_n_layers=blue_n_layers, red_n_layers=red_n_layers,
                                    blue_pop=blue_pop, red_pop=red_pop,
                                    blue_brain=blue_brain,
                                    red_brain=red_brain if red_pop is not None else None,
                                    grid_symbols=grid.symbols, config=config,
                                    grid_walls=grid.walls, grid_resources=grid.resources,
                                )
                                return
                            # Fire TOPO_SIM + full culture snapshot at graduation
                            # (better than brain vote time because reds are now at 15)
                            _grad_obs   = build_observations(blue_pop, grid, blue_map, red_map, config, step)
                            _grad_r     = int(config["local_obs_radius"])
                            _grad_W     = (2 * _grad_r + 1) ** 2
                            _grad_sigs, _grad_ctx = [], []
                            _grad_agent = int(np.where(blue_pop.alive)[0][0]) if blue_pop.alive.any() else -1
                            if _grad_agent >= 0:
                                for _gb in [0, 72, 144, 216, 288]:
                                    for _gd in [2, 4, 6, 8, 10]:
                                        _gsyn = _grad_obs.copy()
                                        _gps  = obs_dim - sig_dim - _grad_W * 4
                                        _gsyn[:, _gps:_gps + _grad_W * 4] = 0.0
                                        _gcr  = max(0, min(int(round(_gd * np.sin(np.radians(_gb)))) + _grad_r, 2 * _grad_r))
                                        _gcc  = max(0, min(int(round(_gd * np.cos(np.radians(_gb)))) + _grad_r, 2 * _grad_r))
                                        _gci  = _gcr * (2 * _grad_r + 1) + _gcc
                                        if _gci < _grad_W:
                                            _gsyn[:, _gps + _grad_W + _gci] = 1.0
                                        _, _, _gsp, _, _, _, _ = blue_brain.forward(
                                            blue_pop.carries, _gsyn, blue_n_layers, blue_pop.nb_gain
                                        )
                                        _grad_sigs.append(_gsp[_grad_agent].copy())
                                        _grad_ctx.append(np.array([float(_gb), float(_gd)]))
                                if len(_grad_sigs) >= 10:
                                    _grs = topographic_similarity(
                                        np.array(_grad_sigs), np.array(_grad_ctx)
                                    )
                                    slog(f"[step {step:>8,}] CTX_SENSITIVITY@GRADUATION  "
                                         f"r_s={_grs:.4f}  reds={_curr_full_floor}  "
                                         f"(single-agent perturbation; offline TOPO_SIM differs)")
                                    cm = culture_metrics(
                                        np.array(grid.symbols), red_pos_hist, blue_pop
                                    )
                                    _lag_str = "  ".join(
                                        f"r@{lag}={cm.get(f'culture_red_lag{lag}', float('nan')):.3f}"
                                        for lag in _CULTURE_LAG_STEPS
                                    )
                                    slog(f"[step {step:>8,}] CULTURE@GRADUATION  "
                                         f"H={cm['culture_entropy']:.3f}  "
                                         f"surv_corr={cm['culture_surv_corr']:.3f}  "
                                         f"{_lag_str}")
                    elif _curriculum_active:
                        _curr_sustain_count = 0   # reset streak if blues dip

                    _eff_red_floor = _curr_red_floor if _curriculum_active else _curr_full_floor
                    red_pop = enforce_population_floor(
                        red_pop, config, rng, sig_dim, red_n_layers, team_id=1,
                        min_pop_override=_eff_red_floor,
                    )

                # ── Spawn red predators ───────────────────────────────────────
                if red_pop is None and step >= config["red_spawn_step"]:
                    red_pop = create_population(config, rng_seed=int(step), grid_size=gs, team_id=1)
                    # Phase 4: ensure reds don't spawn inside walls
                    for _ in range(5):
                        _r_on_wall = grid.is_wall(red_pop.positions) & red_pop.alive
                        if not _r_on_wall.any():
                            break
                        _open = np.argwhere(~grid.walls)
                        _chosen = rng.integers(0, len(_open), size=int(_r_on_wall.sum()))
                        red_pop.positions[_r_on_wall] = _open[_chosen]
                    if _curr_red_floor < max_pop_r:
                        red_pop.alive[_curr_red_floor:] = False
                        red_pop.ages[_curr_red_floor:] = 0
                        red_pop.carries[_curr_red_floor:] = 0.0
                        red_pop.signals[_curr_red_floor:] = 0.0
                    _curriculum_active = True
                    red_buf = RolloutBuffer(config["ppo_rollout_steps"], max_pop_r, obs_dim,
                                            hidden_dim=_hd)
                    slog(f"[step {step:,}] *** RED PREDATORS SPAWNED — "
                         f"curriculum count={int(red_pop.alive.sum())} ***")

                # ── MI sampling ───────────────────────────────────────────────
                if blue_pop.alive.sum() > 10:
                    b_alive = np.where(blue_pop.alive)[0]
                    b_pos   = blue_pop.positions[b_alive]
                    sym_norm = np.linalg.norm(
                        grid.symbols[b_pos[:, 0], b_pos[:, 1]], axis=1
                    ) / (sym_dim ** 0.5 + 1e-8)
                    analyser.record_samples(
                        signals         = blue_pop.signals[b_alive],
                        local_resource  = np.clip(sym_norm, 0, 1),
                        neighbor_count  = np.clip(blue_map[b_pos[:, 0], b_pos[:, 1]] / 5.0, 0, 1),
                        own_energy      = blue_pop.ages[b_alive].astype(np.float32) / float(config["max_age"]),
                        dist_to_red     = np.clip(red_map[b_pos[:, 0], b_pos[:, 1]] / 5.0, 0, 1),
                    )
                    analyser.maybe_analyse(step)
                    try:
                        snap = analyser.result_queue.get_nowait()
                        logger.log_mi_snapshot(step, snap)
                        slog(f"[step {step:>8,}] MI_max={float(snap.mi_matrix.max()):.4f}  "
                             f"brain={blue_n_layers}L")

                        # ── Per-dimension MI breakdown ────────────────────────
                        from communication.analysis import ENV_FEATURE_NAMES
                        _feat = ENV_FEATURE_NAMES
                        _mat  = snap.mi_matrix              # (sig_dim, n_feat)
                        _top5 = np.dstack(np.unravel_index(
                            np.argsort(_mat.ravel())[::-1][:5], _mat.shape
                        ))[0]
                        top5_str = "  ".join(
                            f"dim{d}/'{_feat[f]}'={_mat[d,f]:.3f}"
                            for d, f in _top5
                        )
                        slog(f"[step {step:>8,}] MI_TOP5  {top5_str}")

                        # ── Compositionality test (5-way flee direction) ──────
                        # Predict which direction agents flee (N/S/E/W/STAY) from
                        # top-2 MI signal dims. If dimA+dimB >> max(dimA, dimB),
                        # the dims encode orthogonal aspects (distance + bearing).
                        if snap.n_samples >= 50:
                            from sklearn.linear_model import LogisticRegression
                            _top2_dims = [int(_top5[0][0]), int(_top5[1][0])]
                            _sigs = snap.signal_vectors              # (n, sig_dim)
                            _buf_data  = _last_buf_snapshot if "_last_buf_snapshot" in dir() else blue_buf.get()
                            _flat_acts = _buf_data["actions"].ravel()
                            _flat_aliv = _buf_data["alive"].ravel().astype(bool)
                            _n = min(snap.n_samples, int(_flat_aliv.sum()))
                            if _n >= 30:
                                _dirs = _flat_acts[_flat_aliv][:_n]  # 0-4 target
                                # Skip if target has fewer than 2 unique classes
                                if len(np.unique(_dirs)) >= 2:
                                    _s   = _sigs[:_n]
                                    _acc = {}
                                    for _combo, _name in [
                                        ([_top2_dims[0]], "A"),
                                        ([_top2_dims[1]], "B"),
                                        (_top2_dims,      "A+B"),
                                    ]:
                                        try:
                                            _lr = LogisticRegression(
                                                max_iter=200, C=1.0,
                                                multi_class="multinomial",
                                            )
                                            _lr.fit(_s[:, _combo], _dirs)
                                            _acc[_name] = float(
                                                (_lr.predict(_s[:, _combo]) == _dirs).mean()
                                            )
                                        except Exception:
                                            _acc[_name] = float("nan")
                                    _gain = (_acc.get("A+B", 0)
                                             - max(_acc.get("A", 0), _acc.get("B", 0)))
                                    slog(f"[step {step:>8,}] COMPOSE  "
                                         f"dim{_top2_dims[0]}={_acc.get('A', 0):.3f}  "
                                         f"dim{_top2_dims[1]}={_acc.get('B', 0):.3f}  "
                                         f"A+B={_acc.get('A+B', 0):.3f}  "
                                         f"gain={_gain:+.3f}")

                        # ── Culture metrics ───────────────────────────────────
                        cm = culture_metrics(
                            np.array(grid.symbols), red_pos_hist, blue_pop
                        )
                        _lag_str = "  ".join(
                            f"r@{lag}={cm.get(f'culture_red_lag{lag}', float('nan')):.3f}"
                            for lag in _CULTURE_LAG_STEPS
                        )
                        slog(f"[step {step:>8,}] CULTURE  "
                             f"H={cm['culture_entropy']:.3f}  "
                             f"surv_corr={cm['culture_surv_corr']:.3f}  "
                             f"{_lag_str}")

                        # ── Alarm call propagation ────────────────────────────
                        ac = alarm_call_propagation(
                            blue_pop, red_pop, b_logits_np, config
                        )
                        if ac:
                            delta_str = (f"{ac['alarm_delta']:+.3f}"
                                         if not (isinstance(ac['alarm_delta'], float)
                                                 and ac['alarm_delta'] != ac['alarm_delta'])
                                         else "nan")
                            slog(f"[step {step:>8,}] ALARM  "
                                 f"delta={delta_str}  "
                                 f"flee|scout={ac['p_flee_scout_nearby']:.3f}  "
                                 f"flee|blind={ac['p_flee_no_scout']:.3f}  "
                                 f"awayΔ={ac['away_delta']:+.3f}  "
                                 f"away|scout={ac['p_away_scout_nearby']:.3f}  "
                                 f"away|blind={ac['p_away_no_scout']:.3f}  "
                                 f"scouts={ac['n_scouts']}  blind={ac['n_blind']}")
                    except Exception:
                        pass

                # ── Survival tracking for brain vote ─────────────────────────
                n_b = int(blue_pop.alive.sum())
                recent_blue_survival.append(n_b / max_pop_b)
                if len(recent_blue_survival) > 200:
                    recent_blue_survival.pop(0)

                # ── Brain vote ────────────────────────────────────────────────
                if step % config["brain_vote_interval"] == 0:
                    # Snapshot per-layer stats BEFORE potential growth
                    _sample_obs = build_observations(blue_pop, grid, blue_map, red_map, config, step)
                    _lstats = blue_brain.layer_entropy(
                        blue_pop.carries, _sample_obs, blue_n_layers, blue_pop.nb_gain
                    )
                    _ls_str = "  ".join(
                        f"L{li}: H={v['action_entropy']:.3f} sig={v['signal_norm']:.3f} Δ={v['logit_delta']:.3f}"
                        for li, v in [(k.split("_")[1], v) for k, v in _lstats.items()]
                    )
                    slog(f"[step {step:>8,}] LAYERS  {_ls_str}")

                    # Topographic similarity probe: vary red context, measure r_s
                    # Probe 25 contexts: 5 bearings × 5 distances using the presence channel
                    _gs = config["grid_size"]
                    _r  = int(config["local_obs_radius"])
                    _W  = (2 * _r + 1) ** 2
                    _od = obs_dim
                    _probe_sigs = []
                    _probe_ctx  = []
                    _probe_agent = int(np.where(blue_pop.alive)[0][0]) if blue_pop.alive.any() else -1
                    if _probe_agent >= 0:
                        _base_obs = build_observations(blue_pop, grid, blue_map, red_map, config, step)
                        for _bearing in [0, 72, 144, 216, 288]:  # degrees: 5 directions
                            for _dist in [2, 4, 6, 8, 10]:       # 5 distances
                                _syn = _base_obs.copy()
                                _pres_start = _od - sig_dim - _W * 4
                                _syn[:, _pres_start : _pres_start + _W * 4] = 0.0
                                _cell_r = max(0, min(int(round(_dist * np.sin(np.radians(_bearing)))) + _r, 2 * _r))
                                _cell_c = max(0, min(int(round(_dist * np.cos(np.radians(_bearing)))) + _r, 2 * _r))
                                _cell_idx = _cell_r * (2 * _r + 1) + _cell_c
                                if _cell_idx < _W:
                                    _syn[:, _pres_start + _W + _cell_idx] = 1.0
                                _, _, _psig, _, _, _, _ = blue_brain.forward(
                                    blue_pop.carries, _syn, blue_n_layers, blue_pop.nb_gain
                                )
                                _probe_sigs.append(_psig[_probe_agent].copy())
                                _probe_ctx.append(np.array([float(_bearing), float(_dist)]))
                        if len(_probe_sigs) >= 10:
                            _rs = topographic_similarity(
                                np.array(_probe_sigs), np.array(_probe_ctx)
                            )
                            slog(f"[step {step:>8,}] CTX_SENSITIVITY  r_s={_rs:.4f}  "
                                 f"(single-agent perturbation; offline TOPO_SIM differs)")

                    new_n = brain_vote(blue_pop, config, blue_n_layers, recent_blue_survival)
                    if new_n != blue_n_layers:
                        blue_n_layers = new_n
                        expand_brain(blue_pop, blue_n_layers)
                        recent_blue_survival.clear()

                # ── Token telemetry log (every 100 steps, Phase 4+) ───────────
                if step % 100 == 0 and blue_pop.alive.sum() > 0:
                    _tok_alive = np.where(blue_pop.alive)[0]
                    if len(_tok_alive) > 0 and b_token_ids is not None and (b_token_ids >= 0).any():
                        # Compute received tokens for each agent (neighbors' token IDs)
                        _tok_neigh_idx = get_neighbour_indices_padded(
                            blue_pop.positions, blue_pop.alive, K, gs
                        )  # (N, K)
                        _tok_received = b_token_ids[_tok_neigh_idx]  # (N, K)
                        _tok_sample = _tok_alive[:min(20, len(_tok_alive))]
                        for _ti in _tok_sample:
                            # Compute nearest red distance/direction for context
                            _nr_dist = float("nan")
                            _nr_bear = float("nan")
                            if red_pop is not None and red_pop.alive.any():
                                _rp_t = red_pop.positions[red_pop.alive].astype(np.float32)
                                _bp_t = blue_pop.positions[_ti].astype(np.float32)
                                _dd_t = np.abs(_bp_t[None, :] - _rp_t)
                                _dd_t = np.minimum(_dd_t, gs - _dd_t)
                                _cheb_t = np.maximum(_dd_t[:, 0], _dd_t[:, 1])
                                _nr_dist = float(_cheb_t.min())
                                _nr_vec = (_rp_t[_cheb_t.argmin()] - _bp_t + gs / 2.0) % gs - gs / 2.0
                                _nr_bear = float((np.degrees(np.arctan2(_nr_vec[1], _nr_vec[0])) % 360))
                            logger.log_evo_event({
                                "type": "token_telemetry",
                                "step": step,
                                "agent": int(_ti),
                                "token_id": int(b_token_ids[_ti]),
                                "received_tokens": [int(t) for t in _tok_received[_ti]],
                                "position": [int(blue_pop.positions[_ti, 0]), int(blue_pop.positions[_ti, 1])],
                                "energy": float(blue_pop.energy[_ti]),
                                "action": int(b_actions[_ti]),
                                "nearest_red_dist": round(_nr_dist, 2),
                                "nearest_red_bear": round(_nr_bear, 2),
                                "on_wall": bool(grid.is_wall(blue_pop.positions[_ti:_ti+1])[0]),
                                "local_resource": float(grid.resources[blue_pop.positions[_ti, 0], blue_pop.positions[_ti, 1]]),
                            })

                # ── Periodic log ──────────────────────────────────────────────
                if step % config["log_interval"] == 0:
                    n_r   = int(red_pop.alive.sum()) if red_pop is not None else 0
                    srate = float(np.mean(recent_blue_survival)) if recent_blue_survival else 1.0
                    print(f"step={step:>8,}  blue={n_b}  red={n_r}  "
                          f"brain={blue_n_layers}L  ppo={ppo_count}  "
                          f"surv={srate:.2f}")
                    _alive_ages = blue_pop.ages[blue_pop.alive]
                    logger.log_step_metrics(
                        step            = step,
                        population      = n_b,
                        mean_fitness    = float(_alive_ages.mean()) if len(_alive_ages) else 0.0,
                        max_fitness     = float(_alive_ages.max())  if len(_alive_ages) else 0.0,
                        mean_energy     = srate,
                        evo_steps       = ppo_count,
                        top_lineage_age = int(_alive_ages.max()) if len(_alive_ages) else 0,
                    )

                # ── Checkpoint ────────────────────────────────────────────────
                if step % config["checkpoint_interval"] == 0:
                    ckpt_path = save_checkpoint(
                        ckpt_dir      = ckpt_dir,
                        step          = step,
                        ppo_count     = ppo_count,
                        blue_n_layers = blue_n_layers,
                        red_n_layers  = red_n_layers,
                        blue_pop      = blue_pop,
                        red_pop       = red_pop,
                        blue_brain    = blue_brain,
                        red_brain     = red_brain if red_pop is not None else None,
                        grid_symbols  = grid.symbols,
                        config        = config,
                        grid_walls    = grid.walls,
                        grid_resources = grid.resources,
                    )
                    slog(f"[step {step:,}] checkpoint → {ckpt_path}")

                # ── 150k decoder checklist: social dim dot product test ───────
                if step == 150000 and blue_pop.alive.sum() > 30 and red_pop is not None:
                    import torch
                    slog(f"[step {step:>8,}] === 150k DECODER CHECKPOINT ===")
                    _dec_obs = build_observations(blue_pop, grid, blue_map, red_map, config, step)
                    _dec_alive = np.where(blue_pop.alive)[0]
                    _dec_n = len(_dec_alive)
                    if _dec_n > 30:
                        # Get obs for alive agents
                        _d_obs_t = torch.tensor(_dec_obs[_dec_alive], dtype=torch.float32, requires_grad=True)
                        _d_car_t = torch.tensor(blue_pop.carries[_dec_alive], dtype=torch.float32)
                        # Forward to get action logits
                        _d_net = blue_brain.model
                        _d_net.eval()
                        _d_new_c, (_d_alog, _, _, _, _, _) = _d_net(_d_car_t, _d_obs_t, blue_n_layers)
                        # Compute policy gradient: grad of log-prob of chosen action w.r.t. obs
                        _d_probs = torch.softmax(_d_alog, dim=-1)
                        # Movement-only: sum prob of movement actions (0-3), exclude STAY
                        _d_move_prob = _d_probs[:, :4].sum(dim=-1)  # (N_alive,)
                        _d_move_prob.sum().backward()
                        _d_grad = _d_obs_t.grad.detach().cpu().numpy()  # (N_alive, obs_dim)
                        # Phase 4: dynamic signal block start
                        # obs layout: own_state(6) + nb_sigs(K*sig_dim) + loc_sym(W*sym_dim) + loc_env(W*4) + sig(sig_dim)
                        _sig_start = 6 + _K * sig_dim + _W * sym_dim + _W * 4
                        _social_dims = [_sig_start, _sig_start + 1, _sig_start + 2]
                        _d_social_grad = _d_grad[:, _social_dims]  # (N_alive, 3)
                        # Movement gradient: first 6 dims (own_state)
                        _d_move_grad = _d_grad[:, :6]  # (N_alive, 6)
                        # Determine threat-present vs threat-absent per agent
                        _det_r_150 = int(config.get("red_detection_radius", 0))
                        _r_pos_150 = red_pop.positions[red_pop.alive].astype(np.float32) if red_pop.alive.any() else np.empty((0, 2))
                        _b_pos_150 = blue_pop.positions[_dec_alive].astype(np.float32)
                        _threat_present = np.zeros(_dec_n, dtype=bool)
                        if len(_r_pos_150) > 0:
                            _dd150 = np.abs(_b_pos_150[:, None, :] - _r_pos_150[None, :, :])
                            _dd150 = np.minimum(_dd150, gs - _dd150)
                            _min_d150 = np.maximum(_dd150[:, :, 0], _dd150[:, :, 1]).min(axis=1)
                            _threat_present = _min_d150 <= _det_r_150 * 2  # within 2× detection radius
                        # Dot product: social_grad · move_grad (project social into movement space)
                        # Use norm of social gradient as alignment measure
                        _soc_norm = np.linalg.norm(_d_social_grad, axis=1)
                        _soc_norm_threat = float(_soc_norm[_threat_present].mean()) if _threat_present.sum() > 5 else float("nan")
                        _soc_norm_safe = float(_soc_norm[~_threat_present].mean()) if (~_threat_present).sum() > 5 else float("nan")
                        # Dot product between social grad and own_state grad
                        # Pad to same dim: use first 3 of own_state grad
                        _dot_threat = float(np.mean(np.sum(_d_social_grad[_threat_present] * _d_move_grad[_threat_present, :3], axis=1))) if _threat_present.sum() > 5 else float("nan")
                        _dot_safe = float(np.mean(np.sum(_d_social_grad[~_threat_present] * _d_move_grad[~_threat_present, :3], axis=1))) if (~_threat_present).sum() > 5 else float("nan")
                        slog(f"[step {step:>8,}] SOCIAL_DIM_DOT  "
                             f"threat_dot={_dot_threat:.4f}  safe_dot={_dot_safe:.4f}  "
                             f"threat_norm={_soc_norm_threat:.4f}  safe_norm={_soc_norm_safe:.4f}  "
                             f"n_threat={int(_threat_present.sum())}  n_safe={int((~_threat_present).sum())}")
                        _d_net.train()

            # ── Render ───────────────────────────────────────────────────────
            if renderer is not None:
                sig_pairs = compute_signal_similarity_pairs(
                    blue_pop.signals, blue_pop.alive, threshold=0.75, max_pairs=150
                )
                all_pos  = blue_pop.positions.copy()
                all_live = blue_pop.alive.copy()
                all_team = blue_pop.team.copy()
                all_nl   = blue_pop.n_layers.copy()
                if red_pop is not None:
                    all_pos  = np.concatenate([all_pos,  red_pop.positions])
                    all_live = np.concatenate([all_live, red_pop.alive])
                    all_team = np.concatenate([all_team, red_pop.team])
                    all_nl   = np.concatenate([all_nl,   red_pop.n_layers])

                n_r = int(red_pop.alive.sum()) if red_pop is not None else 0
                running = renderer.render(
                    symbols      = grid.symbols,
                    symbol_dim   = sym_dim,
                    positions    = all_pos,
                    alive        = all_live,
                    team         = all_team,
                    n_layers     = all_nl,
                    signal_pairs = sig_pairs,
                    hud_stats    = {
                        "step":    step,
                        "blues":   n_b,
                        "reds":    n_r,
                        "brain_b": f"{blue_n_layers}L",
                        "brain_r": f"{red_n_layers}L",
                        "evo":     ppo_count,
                    },
                    target_fps = target_fps,
                )
                if not running:
                    break

    finally:
        sci_log.close()
        if renderer:
            renderer.shutdown()
        if dashboard:
            dashboard.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--fresh",    action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--withdrawal", action="store_true",
                        help="Zero out blue signals (withdrawal experiment)")
    parser.add_argument("--blind", action="store_true",
                        help="Zero out own observations (communication bottleneck test)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N steps from resume point (0=unlimited)")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config, resume_path=args.resume, headless=args.headless, fresh=args.fresh,
        withdrawal=args.withdrawal, blind=args.blind, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
