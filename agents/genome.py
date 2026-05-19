"""
genome.py — Genome encoding: weight serialisation, mutation, crossover.

The "genome" is simply the full set of Flax model parameters for one agent,
flattened to a 1D float32 array.  This makes mutation trivially vectorisable
(add Gaussian noise to the flat array) and crossover straightforward
(splice two flat arrays).

Why flat arrays instead of operating on pytrees directly?
- Easier to apply element-wise mutation with a single JAX op
- Easier to store/serialise for checkpoints
- Crossover point is well-defined on a 1D sequence
Reconstruction back to a pytree is done via the stored structure metadata.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# Type aliases
Pytree = Any
FlatGenome = jnp.ndarray   # 1D float32
TreeDef = Any              # jax.tree_util.PyTreeDef
ShapeList = List[Tuple[int, ...]]


# ------------------------------------------------------------------
# Flatten / unflatten
# ------------------------------------------------------------------

def flatten_params(params: Pytree) -> Tuple[FlatGenome, TreeDef, ShapeList]:
    """
    Serialise a Flax parameter pytree into a 1D JAX array plus metadata
    needed to reconstruct it.

    Returns:
        flat      — 1D float32 array (genome)
        treedef   — PyTreeDef for tree reconstruction
        shapes    — list of leaf shapes (in tree-traversal order)
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    flat   = jnp.concatenate([jnp.asarray(leaf, dtype=jnp.float32).ravel()
                               for leaf in leaves])
    return flat, treedef, shapes


def unflatten_params(
    flat: FlatGenome,
    treedef: TreeDef,
    shapes: ShapeList,
) -> Pytree:
    """Reconstruct a Flax parameter pytree from a flat genome array."""
    leaves = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(shape))
        leaves.append(flat[idx : idx + size].reshape(shape))
        idx += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


# ------------------------------------------------------------------
# Batch helpers (operate on full population stacked into one pytree)
# ------------------------------------------------------------------

def get_agent_params(batched_params: Pytree, idx: int) -> Pytree:
    """
    Extract a single agent's params from a population-stacked pytree.
    Each leaf of batched_params has shape (pop_size, ...).
    """
    return jax.tree_util.tree_map(lambda leaf: leaf[idx], batched_params)


def set_agent_params(batched_params: Pytree, idx: int, new_params: Pytree) -> Pytree:
    """
    Return a new batched_params pytree with agent `idx` replaced.
    Uses JAX's functional index-update (.at[].set()) to stay jit-compatible.
    """
    return jax.tree_util.tree_map(
        lambda batched_leaf, new_leaf: batched_leaf.at[idx].set(new_leaf),
        batched_params,
        new_params,
    )


def copy_agent_params(
    batched_params: Pytree,
    src_idx: int,
    dst_idx: int,
) -> Pytree:
    """Copy agent src_idx's params over dst_idx's slot."""
    src = get_agent_params(batched_params, src_idx)
    return set_agent_params(batched_params, dst_idx, src)


# ------------------------------------------------------------------
# Mutation
# ------------------------------------------------------------------

def mutate_params(
    params: Pytree,
    key: jax.random.KeyArray,
    sigma_small: float = 0.01,
    sigma_large: float = 0.1,
    prob_large: float  = 0.05,
) -> Pytree:
    """
    Apply Gaussian mutation to a single agent's params.

    Two-scale mutation strategy:
    - Small perturbations (sigma_small) on every weight — fine-tuning
    - Occasional large perturbations (sigma_large, prob prob_large) —
      escape local optima / exploration jumps

    This mimics biological point mutations of different magnitudes.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    mutated_leaves = []

    for leaf in leaves:
        key, k1, k2 = jax.random.split(key, 3)
        noise_small  = jax.random.normal(k1, shape=leaf.shape) * sigma_small
        noise_large  = jax.random.normal(k2, shape=leaf.shape) * sigma_large
        # Element-wise: use large noise with probability prob_large
        mask = jax.random.bernoulli(key, prob_large, shape=leaf.shape)
        noise = jnp.where(mask, noise_large, noise_small)
        mutated_leaves.append(leaf + noise)

    return jax.tree_util.tree_unflatten(treedef, mutated_leaves)


def crossover_params(
    parent_a: Pytree,
    parent_b: Pytree,
    key: jax.random.KeyArray,
) -> Pytree:
    """
    Single-point crossover on the flattened genome.

    We cross over at the flat-array level rather than at the pytree-leaf
    level to keep the crossover point semantically neutral (no bias toward
    splitting at particular layer boundaries).
    """
    flat_a, treedef, shapes = flatten_params(parent_a)
    flat_b, _,       _      = flatten_params(parent_b)

    n = flat_a.shape[0]
    # Choose a random crossover point
    point = jax.random.randint(key, shape=(), minval=1, maxval=n)
    child_flat = jnp.concatenate([flat_a[:point], flat_b[point:]])
    return unflatten_params(child_flat, treedef, shapes)
