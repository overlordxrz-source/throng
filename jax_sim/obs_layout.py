"""Centralized observation layout helpers for JAX THRONG agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsLayout:
    own_state_start: int
    own_state_end: int
    nb_sigs_start: int
    nb_sigs_end: int
    loc_sym_start: int
    loc_sym_end: int
    loc_env_start: int
    loc_env_end: int
    own_sig_start: int
    own_sig_end: int
    mem_start: int
    mem_end: int
    loc_cult_fast_start: int
    loc_cult_fast_end: int
    loc_cult_slow_start: int
    loc_cult_slow_end: int
    total_dim: int
    neighbor_k: int
    local_cells: int
    env_channels: int
    own_state_dim: int
    symbol_dim: int
    signal_dim: int
    memory_slots: int

    @property
    def spatial_ego_dim(self) -> int:
        """Own state + local env channels: monologue reconstruction target."""
        return self.own_state_dim + self.local_cells * self.env_channels


def make_obs_layout(
    signal_dim: int,
    symbol_dim: int,
    memory_slots: int,
    neighbor_k: int = 6,
    local_cells: int = 25,
    env_channels: int = 10,
    own_state_dim: int = 10,
) -> ObsLayout:
    """Return canonical flat-observation slice boundaries."""
    idx = 0
    own_state_start = idx
    idx += own_state_dim
    own_state_end = idx

    nb_sigs_start = idx
    idx += neighbor_k * signal_dim
    nb_sigs_end = idx

    loc_sym_start = idx
    idx += local_cells * symbol_dim
    loc_sym_end = idx

    loc_env_start = idx
    idx += local_cells * env_channels
    loc_env_end = idx

    own_sig_start = idx
    idx += signal_dim
    own_sig_end = idx

    mem_start = idx
    idx += memory_slots * (signal_dim + 2)
    mem_end = idx

    loc_cult_fast_start = idx
    idx += local_cells * symbol_dim
    loc_cult_fast_end = idx

    loc_cult_slow_start = idx
    idx += local_cells * symbol_dim
    loc_cult_slow_end = idx

    return ObsLayout(
        own_state_start=own_state_start,
        own_state_end=own_state_end,
        nb_sigs_start=nb_sigs_start,
        nb_sigs_end=nb_sigs_end,
        loc_sym_start=loc_sym_start,
        loc_sym_end=loc_sym_end,
        loc_env_start=loc_env_start,
        loc_env_end=loc_env_end,
        own_sig_start=own_sig_start,
        own_sig_end=own_sig_end,
        mem_start=mem_start,
        mem_end=mem_end,
        loc_cult_fast_start=loc_cult_fast_start,
        loc_cult_fast_end=loc_cult_fast_end,
        loc_cult_slow_start=loc_cult_slow_start,
        loc_cult_slow_end=loc_cult_slow_end,
        total_dim=idx,
        neighbor_k=neighbor_k,
        local_cells=local_cells,
        env_channels=env_channels,
        own_state_dim=own_state_dim,
        symbol_dim=symbol_dim,
        signal_dim=signal_dim,
        memory_slots=memory_slots,
    )

