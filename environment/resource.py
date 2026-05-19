"""
resource.py — Resource patch generation and regeneration logic.

Resources are distributed in Gaussian clusters rather than uniformly to
create spatial heterogeneity.  This forces agents to either compete for
known patches or explore — pressure that drives spatial strategy.

Regeneration is local (strongest at cluster centres) and diffuses outward,
modelling slow ecological renewal.  Environmental drift (cluster centre
random walk) prevents agents from converging on a static optimum and keeps
the open-ended dynamic alive.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ResourceManager:
    """Manages resource cluster positions and per-step regeneration."""

    grid_size: int
    n_clusters: int
    regen_rate: float     # peak regen per step at cluster centres
    diffuse_sigma: float  # spatial spread of regen (Gaussian std, in cells)

    # Precomputed regen kernel (refreshed when clusters drift)
    _regen_map: np.ndarray = None  # (grid_size, grid_size)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, rng: np.random.Generator) -> np.ndarray:
        """
        Place cluster centres randomly, generate the initial resource grid,
        and precompute the regen map.  Returns cluster_centres array.
        """
        self.cluster_centres = rng.integers(
            0, self.grid_size, size=(self.n_clusters, 2)
        ).astype(np.float32)
        self._rebuild_regen_map()
        return self.cluster_centres

    # ------------------------------------------------------------------
    # Per-step regeneration
    # ------------------------------------------------------------------

    def regenerate(self, resources: np.ndarray) -> None:
        """
        Apply one step of resource regeneration in-place.
        Adds regen_map to the grid and clamps to [0, 1].
        Using in-place ops to avoid allocation pressure in the hot loop.
        """
        np.add(resources, self._regen_map, out=resources)
        np.clip(resources, 0.0, 1.0, out=resources)

    # ------------------------------------------------------------------
    # Environmental drift (open-endedness mechanism)
    # ------------------------------------------------------------------

    def drift(self, rng: np.random.Generator, sigma: float = 8.0) -> None:
        """
        Shift each cluster centre by a random walk step.
        This prevents agents from permanently optimising for a static
        resource layout, maintaining evolutionary pressure.
        """
        delta = rng.normal(0.0, sigma, size=self.cluster_centres.shape)
        self.cluster_centres = (self.cluster_centres + delta) % self.grid_size
        self._rebuild_regen_map()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_regen_map(self) -> None:
        """
        Precompute the (grid_size, grid_size) float32 regen contribution map.

        We use a sum-of-Gaussians approach: for each cluster centre, add a
        Gaussian blob to the regen map.  Precomputing avoids repeating this
        every step (clusters only move at drift events).

        The grid coordinates are broadcast against each cluster centre;
        toroidal distance uses the minimum of direct and wrapped distances.
        """
        size = self.grid_size
        regen = np.zeros((size, size), dtype=np.float32)

        # Grid coordinate arrays
        ys, xs = np.mgrid[0:size, 0:size]  # (size, size) each

        for cx, cy in self.cluster_centres:
            # Toroidal distance: take min of direct and wrap-around distances
            dx = np.minimum(np.abs(xs - cx), size - np.abs(xs - cx))
            dy = np.minimum(np.abs(ys - cy), size - np.abs(ys - cy))
            dist_sq = dx ** 2 + dy ** 2
            gaussian = np.exp(-dist_sq / (2.0 * self.diffuse_sigma ** 2))
            regen += gaussian

        # Normalise so peak regen equals regen_rate, not n_clusters * regen_rate
        peak = regen.max()
        if peak > 0:
            regen = (regen / peak) * self.regen_rate

        self._regen_map = regen.astype(np.float32)

    # ------------------------------------------------------------------
    # Snapshot for checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "cluster_centres": self.cluster_centres.copy(),
            "regen_map": self._regen_map.copy(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.cluster_centres = d["cluster_centres"].copy()
        self._regen_map = d["regen_map"].copy()
