"""
communication/channel.py — Neighbour signal aggregation.

Each agent broadcasts a 16-dim signal vector.  Neighbours receive the
mean-pooled signal of their k nearest live neighbours.  This models a
simple acoustic / chemical broadcast channel: you can hear the k closest
agents and receive an averaged "mixture" of what they're broadcasting.

Mean pooling (vs. concatenation or attention) was chosen to keep the
input dimensionality fixed regardless of actual neighbour count, and to
make the channel differentiable in principle.

KDTree gives O(n log n) nearest-neighbour queries, acceptable for ≤600
agents per step.  For truly massive populations (10k+), consider grid-based
spatial hashing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def aggregate_neighbour_signals(
    positions:  np.ndarray,   # (max_pop, 2) integer coords
    signals:    np.ndarray,   # (max_pop, signal_dim)
    alive:      np.ndarray,   # (max_pop,) bool
    k:          int = 6,
    grid_size:  Optional[int] = None,  # if given, wrap distance toroidally
) -> np.ndarray:
    """
    For each alive agent, compute the mean signal of its k nearest alive
    neighbours (excluding self).  Dead agents receive zero vectors.

    Returns array of shape (max_pop, signal_dim).

    Why exclude self?
    Including self would make the received signal partially redundant with
    the agent's own last broadcast, diluting the information from others.
    """
    signal_dim = signals.shape[1]
    output     = np.zeros_like(signals)

    alive_idx = np.where(alive)[0]
    n_alive   = len(alive_idx)

    if n_alive < 2:
        return output  # nothing to aggregate

    alive_positions = positions[alive_idx].astype(np.float32)  # (n_alive, 2)
    alive_signals   = signals[alive_idx]                         # (n_alive, signal_dim)

    # If grid_size given, use toroidal (periodic) distance.
    # cKDTree supports boxsize for periodic boundary conditions.
    boxsize = float(grid_size) if grid_size is not None else None
    tree    = cKDTree(alive_positions, boxsize=boxsize)

    # k+1 because the query includes the point itself at distance 0
    actual_k = min(k + 1, n_alive)
    distances, indices = tree.query(alive_positions, k=actual_k)

    # indices shape: (n_alive, actual_k)
    # Skip self (index 0 is always self at distance 0)
    neighbour_indices = indices[:, 1:]  # (n_alive, k)

    # Mean-pool neighbour signals
    # Shape: (n_alive, k, signal_dim)
    neighbour_signals = alive_signals[neighbour_indices]
    aggregated = neighbour_signals.mean(axis=1)  # (n_alive, signal_dim)

    # Write back to full-population output array
    output[alive_idx] = aggregated
    return output.astype(np.float32)


def get_neighbour_signals_padded(
    positions:  np.ndarray,   # (max_pop, 2) integer coords
    signals:    np.ndarray,   # (max_pop, signal_dim) float32
    alive:      np.ndarray,   # (max_pop,) bool
    k:          int = 6,
    grid_size:  Optional[int] = None,
) -> np.ndarray:
    """
    For each agent, return the signals of its k nearest alive neighbours as
    separate tokens — shape (max_pop, k, signal_dim).

    Slots with no neighbour are padded with zeros.
    """
    signal_dim = signals.shape[1] if signals.ndim > 1 else 1
    output     = np.zeros((len(positions), k, signal_dim), dtype=np.float32)

    alive_idx = np.where(alive)[0]
    n_alive   = len(alive_idx)
    if n_alive < 2:
        return output

    alive_positions = positions[alive_idx].astype(np.float32)
    alive_signals   = signals[alive_idx]   # (n_alive, signal_dim)

    boxsize  = float(grid_size) if grid_size is not None else None
    tree     = cKDTree(alive_positions, boxsize=boxsize)

    actual_k   = min(k + 1, n_alive)
    _, indices = tree.query(alive_positions, k=actual_k)
    neigh_idx  = indices[:, 1:]          # (n_alive, actual_k-1) — skip self

    n_got = neigh_idx.shape[1]
    if n_got < k:
        pad       = np.zeros((n_alive, k - n_got), dtype=np.intp)
        neigh_idx = np.concatenate([neigh_idx, pad], axis=1)

    neigh_sigs        = alive_signals[neigh_idx]   # (n_alive, k, signal_dim)
    output[alive_idx] = neigh_sigs
    return output


def get_neighbour_indices_padded(
    positions: np.ndarray,   # (max_pop, 2) integer coords
    alive:     np.ndarray,   # (max_pop,) bool
    k:         int = 6,
    grid_size: Optional[int] = None,
) -> np.ndarray:
    """
    For each agent, return the global population indices of its k nearest alive
    neighbours.  Shape (max_pop, k), dtype int32.

    Slots with no neighbour are padded with the agent's own index (so
    tom_targets[i, pad_slot] == b_actions[i], which gives zero CE contribution
    when predictions default to uniform).
    """
    N      = len(positions)
    output = np.arange(N, dtype=np.int32)[:, None].repeat(k, axis=1)  # self-pad

    alive_idx = np.where(alive)[0]
    n_alive   = len(alive_idx)
    if n_alive < 2:
        return output

    alive_positions = positions[alive_idx].astype(np.float32)
    boxsize         = float(grid_size) if grid_size is not None else None
    tree            = cKDTree(alive_positions, boxsize=boxsize)

    actual_k        = min(k + 1, n_alive)
    _, indices      = tree.query(alive_positions, k=actual_k)
    neigh_local     = indices[:, 1:]   # (n_alive, actual_k-1) — skip self

    n_got = neigh_local.shape[1]
    if n_got < k:
        pad         = np.zeros((n_alive, k - n_got), dtype=np.intp)
        neigh_local = np.concatenate([neigh_local, pad], axis=1)

    neigh_global      = alive_idx[neigh_local].astype(np.int32)   # (n_alive, k)
    output[alive_idx] = neigh_global
    return output


def compute_signal_similarity_pairs(
    signals:   np.ndarray,  # (max_pop, signal_dim) float32
    alive:     np.ndarray,  # (max_pop,) bool
    threshold: float = 0.75,
    max_pairs: int   = 150,
) -> np.ndarray:
    """
    Return pairs of alive agent indices whose signal vectors have cosine
    similarity >= threshold.  Used by the renderer for communication lines.
    """
    alive_idx = np.where(alive)[0]
    if len(alive_idx) < 2:
        return np.empty((0, 2), dtype=np.int32)

    vecs  = signals[alive_idx].astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs  = vecs / norms

    pairs = []
    for i in range(len(alive_idx)):
        for j in range(i + 1, len(alive_idx)):
            if float(np.dot(vecs[i], vecs[j])) >= threshold:
                pairs.append((alive_idx[i], alive_idx[j]))
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)
