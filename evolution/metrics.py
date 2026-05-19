"""
evolution/metrics.py — Fitness tracking and lineage tree management.

Lineage tracking is the key genealogical tool: it lets us answer "which
ancestral line survived the longest?" and "did a single lineage come to
dominate the population?" (population takeover is a sign of strong selection).

We store lineage data in a simple dict rather than a formal tree structure
because the tree is sparse — most lineages die out quickly and we only need
the top 10 for visualisation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LineageRecord:
    """Metadata for one lineage line."""
    lineage_id:    int
    birth_step:    int
    last_seen_step: int
    peak_count:    int        # max simultaneous members
    total_agents:  int        # total agents ever in this lineage
    is_alive:      bool = True


@dataclass
class MetricsTracker:
    """
    Tracks fitness history and lineage data across the full run.

    History arrays grow unboundedly but are only read for visualisation
    so memory is not a concern for typical run lengths.
    """

    # --- Time-series (appended each evolution step) ---
    step_history:       List[int]   = field(default_factory=list)
    population_history: List[int]   = field(default_factory=list)
    mean_fitness_hist:  List[float] = field(default_factory=list)
    max_fitness_hist:   List[float] = field(default_factory=list)
    mean_energy_hist:   List[float] = field(default_factory=list)
    mi_history:         List[Dict]  = field(default_factory=list)  # MI snapshots

    # --- Lineage tracking ---
    lineage_records:    Dict[int, LineageRecord] = field(default_factory=dict)
    _last_lineage_counts: Dict[int, int]         = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Step-level recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step:      int,
        alive:     np.ndarray,   # (max_pop,) bool
        energies:  np.ndarray,   # (max_pop,) float
        ages:      np.ndarray,   # (max_pop,) int
        consumed:  np.ndarray,   # (max_pop,) float
        lineage_ids: np.ndarray, # (max_pop,) int
    ) -> None:
        """Append one data point to history series."""
        alive_mask = alive.astype(bool)
        n_alive = int(alive_mask.sum())

        fitness = ages.astype(np.float32) * consumed
        fitness_alive = fitness[alive_mask]
        energies_alive = energies[alive_mask]

        self.step_history.append(step)
        self.population_history.append(n_alive)
        self.mean_fitness_hist.append(float(fitness_alive.mean()) if n_alive else 0.0)
        self.max_fitness_hist.append(float(fitness_alive.max())  if n_alive else 0.0)
        self.mean_energy_hist.append(float(energies_alive.mean()) if n_alive else 0.0)

        # Update lineage records
        self._update_lineages(step, lineage_ids, alive_mask)

    def _update_lineages(
        self,
        step:        int,
        lineage_ids: np.ndarray,
        alive_mask:  np.ndarray,
    ) -> None:
        """
        Update lineage presence counts.  A lineage is "alive" as long as
        at least one agent bearing its id is alive in the population.
        """
        alive_lineages = lineage_ids[alive_mask]
        current_counts: Dict[int, int] = defaultdict(int)
        for lid in alive_lineages:
            current_counts[int(lid)] += 1

        # Update existing + create new records
        for lid, count in current_counts.items():
            if lid not in self.lineage_records:
                self.lineage_records[lid] = LineageRecord(
                    lineage_id=lid, birth_step=step, last_seen_step=step,
                    peak_count=count, total_agents=count, is_alive=True,
                )
            else:
                rec = self.lineage_records[lid]
                rec.last_seen_step = step
                rec.is_alive       = True
                rec.peak_count     = max(rec.peak_count, count)
                # Count newly added agents (count > previous count)
                prev = self._last_lineage_counts.get(lid, 0)
                if count > prev:
                    rec.total_agents += (count - prev)

        # Mark lineages that have gone extinct
        for lid in list(self._last_lineage_counts.keys()):
            if lid not in current_counts and lid in self.lineage_records:
                self.lineage_records[lid].is_alive = False

        self._last_lineage_counts = dict(current_counts)

    # ------------------------------------------------------------------
    # Queries for visualisation
    # ------------------------------------------------------------------

    def top_lineages(self, n: int = 10) -> List[LineageRecord]:
        """
        Return top-n lineages by longevity (last_seen - birth_step).
        Includes both alive and extinct lineages.
        """
        records = list(self.lineage_records.values())
        records.sort(
            key=lambda r: r.last_seen_step - r.birth_step,
            reverse=True,
        )
        return records[:n]

    def dominant_lineage(self) -> Optional[int]:
        """
        Return the lineage_id currently with the most living members.
        Returns None if population is empty.
        """
        if not self._last_lineage_counts:
            return None
        return max(self._last_lineage_counts, key=self._last_lineage_counts.get)

    def record_mi(self, step: int, mi_data: Dict) -> None:
        """Append a mutual-information snapshot."""
        self.mi_history.append({"step": step, **mi_data})

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "step_history":       self.step_history,
            "population_history": self.population_history,
            "mean_fitness_hist":  self.mean_fitness_hist,
            "max_fitness_hist":   self.max_fitness_hist,
            "mean_energy_hist":   self.mean_energy_hist,
            "mi_history":         self.mi_history,
            "lineage_records":    {
                k: vars(v) for k, v in self.lineage_records.items()
            },
        }

    @classmethod
    def from_state_dict(cls, d: Dict) -> "MetricsTracker":
        tracker = cls()
        tracker.step_history       = d.get("step_history", [])
        tracker.population_history = d.get("population_history", [])
        tracker.mean_fitness_hist  = d.get("mean_fitness_hist", [])
        tracker.max_fitness_hist   = d.get("max_fitness_hist", [])
        tracker.mean_energy_hist   = d.get("mean_energy_hist", [])
        tracker.mi_history         = d.get("mi_history", [])
        for k, v in d.get("lineage_records", {}).items():
            tracker.lineage_records[int(k)] = LineageRecord(**v)
        return tracker
