"""
agents/population.py — Population state for MAPPO Phase 2.

Key design:
  - params: single shared PyTree per team (all agents on same team share weights)
  - opt_state: optimizer state for the shared params
  - signals: (max_pop, signal_dim) float32 continuous broadcast vectors
  - carries: (max_pop, hidden_dim) per-agent recurrent memory
  - n_layers: per-agent int8 (used for display; shared policy uses team n_layers)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class PopulationState:
    positions:          np.ndarray   # (max_pop, 2)           int32
    ages:               np.ndarray   # (max_pop,)             int32
    alive:              np.ndarray   # (max_pop,)             bool
    team:               np.ndarray   # (max_pop,)             int8
    lineage_ids:        np.ndarray   # (max_pop,)             int32
    offspring_count:    np.ndarray   # (max_pop,)             int32
    steps_since_catch:  np.ndarray   # (max_pop,)             int32 (reds)
    n_layers:           np.ndarray   # (max_pop,)             int8  (display only)

    carries:   np.ndarray            # (max_pop, hidden_dim)  per-agent memory (numpy)
    signals:   np.ndarray            # (max_pop, signal_dim)  float32 broadcasts
    nb_gain:   np.ndarray            # (max_pop,)             float32 per-agent signal reception gain
    energy:    np.ndarray            # (max_pop,)             float32 0..1 per-agent energy

    max_pop:         int
    next_lineage_id: int

    # Phase 7: episodic memory buffer (max_pop, memory_slots, signal_dim + 2)
    # Each slot: [signal_received (signal_dim), action_taken (1), survived (1)]
    memory_buffer: Optional[np.ndarray] = None


def create_population(
    config:    Dict,
    rng_seed:  int,
    grid_size: int,
    team_id:   int = 0,
) -> PopulationState:
    max_pop  = config["population_size"] if team_id == 0 else config["red_population_size"]
    hidden   = config["agent_hidden_dim"] if team_id == 0 else config.get("red_hidden_dim", config["agent_hidden_dim"])
    sig_dim  = config["signal_dim"]
    n_layers = int(config.get("n_layers", 2))

    rng = np.random.default_rng(rng_seed)
    pos = rng.integers(0, grid_size, size=(max_pop, 2), dtype=np.int32)

    mem_size = int(config.get("memory_buffer_size", 0))
    mem_buf = None
    if mem_size > 0 and config.get("memory_buffer_enabled", False):
        mem_buf = np.zeros((max_pop, mem_size, sig_dim + 2), dtype=np.float32)

    return PopulationState(
        positions         = pos,
        ages              = np.zeros(max_pop, dtype=np.int32),
        alive             = np.ones(max_pop,  dtype=bool),
        team              = np.full(max_pop, team_id, dtype=np.int8),
        lineage_ids       = np.arange(max_pop, dtype=np.int32),
        offspring_count   = np.zeros(max_pop, dtype=np.int32),
        steps_since_catch = np.zeros(max_pop, dtype=np.int32),
        n_layers          = np.full(max_pop, n_layers, dtype=np.int8),
        carries           = np.zeros((max_pop, hidden), dtype=np.float32),
        signals           = np.zeros((max_pop, sig_dim), dtype=np.float32),
        nb_gain           = np.ones(max_pop, dtype=np.float32),
        energy            = np.ones(max_pop, dtype=np.float32),  # start full
        memory_buffer     = mem_buf,
        max_pop           = max_pop,
        next_lineage_id   = max_pop,
    )


# ── Lifecycle helpers ─────────────────────────────────────────────────────────

def kill_agent(pop: PopulationState, idx: int) -> None:
    pop.alive[idx]    = False
    pop.ages[idx]     = 0
    pop.carries[idx]  = 0.0
    pop.energy[idx]   = 0.0
    if pop.memory_buffer is not None:
        pop.memory_buffer[idx] = 0.0


def _find_free_slot(pop: PopulationState) -> Optional[int]:
    dead = np.where(~pop.alive)[0]
    return int(dead[0]) if len(dead) > 0 else None


def spawn_agent(
    pop:          PopulationState,
    parent_idx:   int,
    grid_size:    int,
    rng:          np.random.Generator,
    sig_dim:      int,
    inherit_lineage: bool = True,
) -> Tuple[PopulationState, Optional[int]]:
    slot = _find_free_slot(pop)
    if slot is None:
        return pop, None

    offsets   = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    child_pos = (pop.positions[parent_idx] + offsets[rng.integers(0, 4)]) % grid_size

    pop.positions[slot]         = child_pos
    pop.ages[slot]              = 0
    pop.alive[slot]             = True
    pop.team[slot]              = pop.team[parent_idx]
    pop.n_layers[slot]          = pop.n_layers[parent_idx]
    pop.offspring_count[slot]   = 0
    pop.steps_since_catch[slot] = 0
    pop.signals[slot]  = 0.0
    pop.carries[slot]  = 0.0
    pop.nb_gain[slot]  = 1.0
    pop.energy[slot]   = 1.0  # offspring start with full energy
    if pop.memory_buffer is not None:
        pop.memory_buffer[slot] = 0.0

    if inherit_lineage:
        pop.lineage_ids[slot] = pop.lineage_ids[parent_idx]
    else:
        pop.lineage_ids[slot]  = pop.next_lineage_id
        pop.next_lineage_id   += 1

    return pop, slot


def inject_random_agent(
    pop:       PopulationState,
    grid_size: int,
    rng:       np.random.Generator,
    sig_dim:   int,
    n_layers:  int,
    team_id:   int = 0,
) -> PopulationState:
    """Legacy cold-start injection (blank carry). Prefer inject_offspring."""
    slot = _find_free_slot(pop)
    if slot is None:
        return pop

    pop.positions[slot]         = rng.integers(0, grid_size, size=2)
    pop.ages[slot]              = 0
    pop.alive[slot]             = True
    pop.team[slot]              = np.int8(team_id)
    pop.n_layers[slot]          = np.int8(n_layers)
    pop.offspring_count[slot]   = 0
    pop.steps_since_catch[slot] = 0
    pop.lineage_ids[slot]       = pop.next_lineage_id
    pop.next_lineage_id        += 1
    pop.signals[slot]           = 0.0
    pop.carries[slot]           = 0.0
    pop.nb_gain[slot]           = 1.0
    pop.energy[slot]            = 1.0
    if pop.memory_buffer is not None:
        pop.memory_buffer[slot] = 0.0
    return pop


def inject_offspring(
    pop:       PopulationState,
    grid_size: int,
    rng:       np.random.Generator,
    n_layers:  int,
    team_id:   int = 0,
) -> PopulationState:
    """
    Priority 2 replacement for inject_random_agent.

    Instead of blank-carry cold-start, clone the carry state of the
    highest-age alive agent and place the new agent adjacent to it.
    The offspring inherits a warm recurrent state, so it behaves
    sensibly from step 1 — no warmup mask needed.
    """
    slot = _find_free_slot(pop)
    if slot is None:
        return pop

    alive_idx = np.where(pop.alive & (pop.team == team_id))[0]
    if len(alive_idx) == 0:
        return inject_random_agent(pop, grid_size, rng, pop.signals.shape[1], n_layers, team_id)

    # Find highest-age parent (for warm carry inheritance)
    parent_idx = alive_idx[np.argmax(pop.ages[alive_idx])]

    # Scatter to a random grid position — do NOT cluster next to parent.
    # With high predation (radius=1, 15 reds) all clones placed adjacent to one
    # parent are caught in the next step. Random scatter spreads offspring away
    # from red clusters so at least some survive long enough to learn.
    child_pos = rng.integers(0, grid_size, size=2)

    pop.positions[slot]         = child_pos
    pop.ages[slot]              = 0
    pop.alive[slot]             = True
    pop.team[slot]              = np.int8(team_id)
    pop.n_layers[slot]          = np.int8(n_layers)
    pop.offspring_count[slot]   = 0
    pop.steps_since_catch[slot] = 0
    pop.lineage_ids[slot]       = pop.lineage_ids[parent_idx]   # same lineage
    pop.signals[slot]           = 0.0
    pop.carries[slot]           = 0.0   # zero carry — avoid cloning behavioral bias into all offspring
    pop.nb_gain[slot]           = 1.0
    pop.energy[slot]            = 1.0
    if pop.memory_buffer is not None:
        pop.memory_buffer[slot] = 0.0
    return pop


def expand_brain(
    pop:       PopulationState,
    new_n:     int,
) -> PopulationState:
    """Update the display n_layers field when the team's shared brain grows."""
    pop.n_layers[:] = np.int8(new_n)
    return pop
