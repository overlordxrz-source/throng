"""
environment/grid.py — Toroidal grid with continuous float symbol culture layer,
procedural walls, and localized depleting resource patches.

Layers:
  symbols    (size, size, symbol_dim)  float32 — persistent cultural memory
  walls      (size, size)  bool — impassable obstacles
  resources  (size, size)  float32 — 0..1, depleting food patches
  presence   computed on-the-fly from agent positions

Agents write tanh vectors to cells; the symbol layer decays multiplicatively.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


class ToroidalGrid:

    def __init__(self, size: int, symbol_dim: int = 8) -> None:
        self.size       = size
        self.symbol_dim = symbol_dim
        self.symbols    = np.zeros((size, size, symbol_dim), dtype=np.float32)
        self.walls      = np.zeros((size, size), dtype=bool)
        self.resources  = np.zeros((size, size), dtype=np.float32)

    # ── Wrapping ─────────────────────────────────────────────────────────────

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        return positions % self.size

    # ── Symbols ────────────────────────────────────────────────────────────────

    def decay_symbols(self, decay: float = 0.993) -> None:
        self.symbols *= decay

    def write_symbols(
        self,
        positions: np.ndarray,   # (max_pop, 2) int
        writes:    np.ndarray,   # (max_pop, symbol_dim) float32
        alive:     np.ndarray,   # (max_pop,) bool
    ) -> None:
        idx = np.where(alive)[0]
        if len(idx) == 0:
            return
        pos = positions[idx]
        w   = writes[idx]
        np.add.at(self.symbols, (pos[:, 0], pos[:, 1]), w)
        self.symbols = np.clip(self.symbols, -1.0, 1.0)

    def get_local_symbols(
        self,
        positions: np.ndarray,   # (max_pop, 2)
        radius:    int = 1,
    ) -> np.ndarray:
        """Return (max_pop, W * symbol_dim) flattened local symbol window."""
        dy, dx  = np.mgrid[-radius:radius+1, -radius:radius+1]
        offsets = np.stack([dy.ravel(), dx.ravel()], axis=1)   # (W, 2)
        cells   = (positions[:, None, :] + offsets[None, :, :]) % self.size
        rows    = cells[:, :, 0]
        cols    = cells[:, :, 1]
        syms    = self.symbols[rows, cols]    # (max_pop, W, symbol_dim)
        return syms.reshape(positions.shape[0], -1).astype(np.float32)

    # ── Presence maps ──────────────────────────────────────────────────────────

    def build_presence_maps(
        self,
        positions: np.ndarray,
        alive:     np.ndarray,
        team:      np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        blue_map = np.zeros((self.size, self.size), dtype=np.float32)
        red_map  = np.zeros((self.size, self.size), dtype=np.float32)
        blue_mask = alive & (team == 0)
        red_mask  = alive & (team == 1)
        if blue_mask.any():
            np.add.at(blue_map, (positions[blue_mask, 0], positions[blue_mask, 1]), 1.0)
        if red_mask.any():
            np.add.at(red_map,  (positions[red_mask,  0], positions[red_mask,  1]), 1.0)
        return blue_map, red_map

    def get_local_presence(
        self,
        positions: np.ndarray,
        blue_map:  np.ndarray,
        red_map:   np.ndarray,
        radius:    int = 1,
    ) -> np.ndarray:
        """Return (max_pop, W*2) normalised presence — [blue_density, red_density]."""
        dy, dx  = np.mgrid[-radius:radius+1, -radius:radius+1]
        offsets = np.stack([dy.ravel(), dx.ravel()], axis=1)
        cells   = (positions[:, None, :] + offsets[None, :, :]) % self.size
        rows    = cells[:, :, 0]
        cols    = cells[:, :, 1]
        bv = np.clip(blue_map[rows, cols], 0, 4) / 4.0
        rv = np.clip(red_map[rows, cols],  0, 4) / 4.0
        return np.concatenate([bv, rv], axis=1).astype(np.float32)

    # ── Wall helpers ───────────────────────────────────────────────────────────

    def is_wall(self, positions: np.ndarray) -> np.ndarray:
        """Return (N,) bool — True where position is a wall."""
        return self.walls[positions[:, 0], positions[:, 1]]

    def generate_walls(
        self,
        rng:      np.random.Generator,
        density:  float = 0.08,
        min_room: int = 3,
    ) -> None:
        """Procedural wall generation: cellular automata cave generation.
        Produces maze-like walls with guaranteed open connectivity."""
        size = self.size
        # 1. Random seed at higher density than target
        _init_prob = min(0.45, density * 5)
        walls = rng.random((size, size)) < _init_prob

        # 2. Cellular automata smoothing (cave-like walls)
        for _ in range(5):
            # Count 8 neighbors (orthogonal + diagonal)
            _n8 = (
                np.roll(walls, 1, axis=0) + np.roll(walls, -1, axis=0) +
                np.roll(walls, 1, axis=1) + np.roll(walls, -1, axis=1) +
                np.roll(np.roll(walls, 1, axis=0), 1, axis=1) +
                np.roll(np.roll(walls, 1, axis=0), -1, axis=1) +
                np.roll(np.roll(walls, -1, axis=0), 1, axis=1) +
                np.roll(np.roll(walls, -1, axis=0), -1, axis=1)
            )
            walls = _n8 >= 5

        # 3. Ensure target density by adjusting threshold
        _curr_d = walls.mean()
        if _curr_d > density * 1.5:
            # Too many walls — randomly open some cells
            _excess = (_curr_d - density) / _curr_d
            walls &= rng.random((size, size)) > _excess
        elif _curr_d < density * 0.5:
            # Too few walls — add random noise back
            walls |= rng.random((size, size)) < (density - _curr_d)

        # 4. Ensure open connectivity (flood-fill from center)
        _open = ~walls
        _visited = np.zeros_like(_open, dtype=bool)
        _stack = [(size // 2, size // 2)]
        while _stack:
            cy, cx = _stack.pop()
            if 0 <= cy < size and 0 <= cx < size and not _visited[cy, cx] and _open[cy, cx]:
                _visited[cy, cx] = True
                _stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
        # Remove unreachable open pockets by turning them into walls
        _pocket = _open & ~_visited
        walls |= _pocket

        # 5. Clean up isolated wall cells (no orthogonal neighbors)
        _n4 = (
            np.roll(walls, 1, axis=0) + np.roll(walls, -1, axis=0) +
            np.roll(walls, 1, axis=1) + np.roll(walls, -1, axis=1)
        )
        walls &= (_n4 > 0) | walls  # keep walls with at least 1 orthogonal neighbor

        # 6. Keep edges open for spawn safety
        walls[0, :] = False
        walls[-1, :] = False
        walls[:, 0] = False
        walls[:, -1] = False

        self.walls = walls

    def get_local_walls(
        self,
        positions: np.ndarray,
        radius:    int = 1,
    ) -> np.ndarray:
        """Return (max_pop, W) bool — wall presence in local window."""
        dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        offsets = np.stack([dy.ravel(), dx.ravel()], axis=1)
        cells = (positions[:, None, :] + offsets[None, :, :]) % self.size
        rows = cells[:, :, 0]
        cols = cells[:, :, 1]
        return self.walls[rows, cols].astype(np.float32)

    # ── Resource helpers ───────────────────────────────────────────────────────

    def generate_resources(
        self,
        rng:           np.random.Generator,
        n_patches:     int = 20,
        patch_radius:  int = 3,
        patch_value:   float = 0.8,
    ) -> None:
        """Place Gaussian resource blobs at random cluster centers."""
        size = self.size
        self.resources.fill(0.0)
        centers = rng.integers(0, size, size=(n_patches, 2))
        yy, xx = np.mgrid[0:size, 0:size]
        for cy, cx in centers:
            if self.walls[cy, cx]:
                continue  # don't spawn inside walls
            dy = np.minimum(np.abs(yy - cy), size - np.abs(yy - cy))
            dx = np.minimum(np.abs(xx - cx), size - np.abs(xx - cx))
            dist = np.sqrt(dy ** 2 + dx ** 2)
            blob = patch_value * np.exp(-(dist ** 2) / (2 * (patch_radius / 2.0) ** 2))
            self.resources = np.maximum(self.resources, blob)
        self.resources[self.walls] = 0.0  # no resources on walls
        np.clip(self.resources, 0.0, 1.0, out=self.resources)

    def consume_resources(
        self,
        positions: np.ndarray,   # (max_pop, 2)
        alive:     np.ndarray,   # (max_pop,)
    ) -> np.ndarray:
        """Return (max_pop,) float — energy gained per agent. Depletes grid."""
        energy_gained = np.zeros(positions.shape[0], dtype=np.float32)
        idx = np.where(alive)[0]
        if len(idx) == 0:
            return energy_gained
        pos = positions[idx]
        # Each agent consumes up to 0.3 from its cell
        available = self.resources[pos[:, 0], pos[:, 1]]
        consumed = np.minimum(available, 0.3)
        self.resources[pos[:, 0], pos[:, 1]] -= consumed
        np.clip(self.resources, 0.0, 1.0, out=self.resources)
        energy_gained[idx] = consumed
        return energy_gained

    def regenerate_resources(
        self,
        rng:          np.random.Generator,
        step:         int,
        regen_rate:   float = 0.002,
        n_patches:    int = 20,
        patch_radius: int = 3,
    ) -> None:
        """Slow regrowth + occasional fresh patches."""
        # Global slow regrowth on all cells
        self.resources += regen_rate
        np.clip(self.resources, 0.0, 1.0, out=self.resources)
        self.resources[self.walls] = 0.0
        # Every 500 steps, respawn a few fresh patches in depleted areas
        if step % 500 == 0:
            size = self.size
            depleted = np.argwhere(self.resources < 0.1)
            if len(depleted) > 0:
                n_new = min(3, len(depleted))
                chosen = depleted[rng.integers(0, len(depleted), size=n_new)]
                yy, xx = np.mgrid[0:size, 0:size]
                for cy, cx in chosen:
                    if self.walls[cy, cx]:
                        continue
                    dy = np.minimum(np.abs(yy - cy), size - np.abs(yy - cy))
                    dx = np.minimum(np.abs(xx - cx), size - np.abs(xx - cx))
                    dist = np.sqrt(dy ** 2 + dx ** 2)
                    blob = 0.6 * np.exp(-(dist ** 2) / (2 * (patch_radius / 2.0) ** 2))
                    self.resources = np.maximum(self.resources, blob)
                np.clip(self.resources, 0.0, 1.0, out=self.resources)
                self.resources[self.walls] = 0.0

    def get_local_resources(
        self,
        positions: np.ndarray,
        radius:    int = 1,
    ) -> np.ndarray:
        """Return (max_pop, W) float — resource density in local window."""
        dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        offsets = np.stack([dy.ravel(), dx.ravel()], axis=1)
        cells = (positions[:, None, :] + offsets[None, :, :]) % self.size
        rows = cells[:, :, 0]
        cols = cells[:, :, 1]
        return self.resources[rows, cols].astype(np.float32)
