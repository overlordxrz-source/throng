"""
utils/checkpointing.py — Full simulation state save / load.

Saves everything needed to resume from an exact step:
  - step, ppo_count, brain depths
  - blue and red PopulationState (positions, ages, alive, signals, carries, n_layers, …)
  - blue and red params + opt_state (serialised as numpy trees)
  - grid.symbols (culture layer)
  - config snapshot

Uses pickle (protocol 4) via a helper that converts JAX arrays → numpy before serialising,
and numpy → jnp on load. This avoids JAX serialisation headaches.

Checkpoint file: <ckpt_dir>/checkpoint_<step>.pkl
Latest symlink:  <ckpt_dir>/checkpoint_latest.pkl  → most recent file
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ── Population state ↔ dict ───────────────────────────────────────────────────

def _pop_to_dict(pop) -> Dict:
    return {
        "positions":         np.array(pop.positions),
        "ages":              np.array(pop.ages),
        "alive":             np.array(pop.alive),
        "team":              np.array(pop.team),
        "lineage_ids":       np.array(pop.lineage_ids),
        "offspring_count":   np.array(pop.offspring_count),
        "steps_since_catch": np.array(pop.steps_since_catch),
        "n_layers":          np.array(pop.n_layers),
        "carries":           np.array(pop.carries),
        "signals":           np.array(pop.signals),
        "nb_gain":           np.array(pop.nb_gain),
        "energy":            np.array(pop.energy),
        "max_pop":           pop.max_pop,
        "next_lineage_id":   pop.next_lineage_id,
    }


def _dict_to_pop(d: Dict):
    """Restore a PopulationState from its serialised dict (all numpy)."""
    from agents.population import PopulationState
    max_pop = d["max_pop"]
    return PopulationState(
        positions         = d["positions"],
        ages              = d["ages"],
        alive             = d["alive"],
        team              = d["team"],
        lineage_ids       = d["lineage_ids"],
        offspring_count   = d["offspring_count"],
        steps_since_catch = d["steps_since_catch"],
        n_layers          = d["n_layers"],
        carries           = d["carries"].astype(np.float32),
        signals           = d["signals"].astype(np.float32),
        nb_gain           = d.get("nb_gain", np.ones(max_pop, dtype=np.float32)),
        energy            = d.get("energy", np.ones(max_pop, dtype=np.float32)),
        max_pop           = max_pop,
        next_lineage_id   = d["next_lineage_id"],
    )


# ── Save ──────────────────────────────────────────────────────────────────────

def save_checkpoint(
    ckpt_dir:      Path,
    step:          int,
    ppo_count:     int,
    blue_n_layers: int,
    red_n_layers:  int,
    blue_pop,
    red_pop,           # may be None before red_spawn_step
    blue_brain,        # TorchBrain
    red_brain,         # TorchBrain | None
    grid_symbols:  np.ndarray,
    config:        Dict,
    grid_walls:    np.ndarray = None,
    grid_resources: np.ndarray = None,
) -> Path:
    payload = {
        "step":          step,
        "ppo_count":     ppo_count,
        "blue_n_layers": blue_n_layers,
        "red_n_layers":  red_n_layers,
        "blue_pop":      _pop_to_dict(blue_pop),
        "red_pop":       _pop_to_dict(red_pop) if red_pop is not None else None,
        "blue_brain":    blue_brain.state_dict(),
        "red_brain":     red_brain.state_dict() if red_brain is not None else None,
        "grid_symbols":  np.array(grid_symbols),
        "config":        config,
    }
    if grid_walls is not None:
        payload["grid_walls"] = np.array(grid_walls)
    if grid_resources is not None:
        payload["grid_resources"] = np.array(grid_resources)

    ckpt_path = ckpt_dir / f"checkpoint_{step}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    latest = ckpt_dir / "checkpoint_latest.pkl"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_path.name)
    except OSError:
        pass

    return ckpt_path


# ── Load ──────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str, blue_brain=None, red_brain=None) -> Dict:
    """
    Load a checkpoint.  Returns a dict with keys:
      step, ppo_count, blue_n_layers, red_n_layers,
      blue_pop (PopulationState), red_pop (PopulationState | None),
      grid_symbols (np.ndarray), config (dict)

    If blue_brain / red_brain are provided, their weights are loaded in-place.
    Handles legacy checkpoints (no blue_brain key) gracefully.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    blue_pop = _dict_to_pop(payload["blue_pop"])
    red_pop  = (_dict_to_pop(payload["red_pop"])
                if payload["red_pop"] is not None else None)

    is_legacy = "blue_brain" not in payload or payload.get("blue_brain") is None
    if not is_legacy:
        if blue_brain is not None:
            blue_brain.load_state_dict(payload["blue_brain"])
        if red_brain is not None and "red_brain" in payload and payload["red_brain"] is not None:
            red_brain.load_state_dict(payload["red_brain"])
    else:
        blue_pop.carries[:] = 0.0
        if red_pop is not None:
            red_pop.carries[:] = 0.0
        print("  [ckpt] legacy JAX checkpoint — carries zeroed for clean PyTorch re-warm")

    return {
        "step":          payload["step"],
        "ppo_count":     payload["ppo_count"],
        "blue_n_layers": payload["blue_n_layers"],
        "red_n_layers":  payload["red_n_layers"],
        "blue_pop":      blue_pop,
        "red_pop":       red_pop,
        "grid_symbols":  payload["grid_symbols"],
        "grid_walls":    payload.get("grid_walls"),
        "grid_resources": payload.get("grid_resources"),
        "config":        payload["config"],
    }


def find_latest_checkpoint(log_dir: str) -> Optional[str]:
    """
    Search log_dir and one level of subdirectories for the most recently
    modified checkpoint_latest.pkl.  Returns absolute path or None.
    """
    base = Path(log_dir)
    candidates = list(base.glob("checkpoint_latest.pkl")) + \
                 list(base.glob("*/checkpoint_latest.pkl"))
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(best.resolve())
