"""
Phase 17: The Rosetta Stone
Unsupervised Semantic Translation Layer
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
import ott
from ott.geometry import geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein_lr

class RosettaStone(nn.Module):
    vocab_size: int = 64
    llm_dim: int = 256
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mode: str = "fwd") -> jnp.ndarray:
        """
        x: if mode=="fwd" (Discrete -> Continuous), x is one-hot or soft distribution [batch, vocab_size]
           if mode=="bwd" (Continuous -> Discrete), x is [batch, llm_dim]
        """
        if mode == "fwd":
            # F_theta: Discrete (vocab) to Continuous (llm_dim)
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(self.llm_dim)(x)
            return x
        elif mode == "bwd":
            # G_phi: Continuous (llm_dim) to Discrete (vocab)
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            logits = nn.Dense(self.vocab_size)(x)
            return logits
        else:
            raise ValueError(f"Unknown mode {mode}")

def compute_distance_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """Computes pairwise Euclidean distance matrix."""
    x_sq = jnp.sum(x ** 2, axis=-1, keepdims=True)
    dist = x_sq + x_sq.T - 2 * jnp.dot(x, x.T)
    # Clip for numerical stability
    return jnp.sqrt(jnp.maximum(dist, 1e-8))

def gumbel_softmax(logits: jnp.ndarray, rng: jax.random.PRNGKey, temperature: float = 1.0, hard: bool = True) -> jnp.ndarray:
    """Gumbel-Softmax estimator for discrete distributions."""
    gumbels = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape) + 1e-20) + 1e-20)
    y_soft = jax.nn.softmax((logits + gumbels) / temperature)
    
    if hard:
        # Straight-through estimator
        index = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(index, logits.shape[-1])
        y = jax.lax.stop_gradient(y_hard - y_soft) + y_soft
        return y
    return y_soft

@jax.jit
def rosetta_stone_loss(
    params, 
    model: nn.Module, 
    batch_marl: jnp.ndarray,   # [batch, vocab_size]
    batch_llm: jnp.ndarray,    # [batch, llm_dim]
    rng: jax.random.PRNGKey,
    temperature: float = 1.0,
    gw_rank: int = 10,
    epsilon: float = 1e-2
):
    """
    Computes the Unsupervised Semantic Translation Loss:
    1. Cycle Consistency (Forward & Backward)
    2. Gromov-Wasserstein Alignment (Low-Rank)
    3. Minimum Description Length (Isometry)
    """
    rng1, rng2 = jax.random.split(rng)
    
    # Forward & Backward Mappings
    # MARL -> LLM
    mapped_llm = model.apply(params, batch_marl, mode="fwd")
    
    # LLM -> MARL
    logits_marl = model.apply(params, batch_llm, mode="bwd")
    mapped_marl = gumbel_softmax(logits_marl, rng1, temperature=temperature, hard=True)
    
    # 1. Cycle Consistency Losses
    # Forward cycle: MARL -> LLM -> MARL
    logits_marl_cycle = model.apply(params, mapped_llm, mode="bwd")
    mapped_marl_cycle = gumbel_softmax(logits_marl_cycle, rng2, temperature=temperature, hard=True)
    loss_cycle_fwd = jnp.mean(jnp.sum(-batch_marl * jax.nn.log_softmax(logits_marl_cycle), axis=-1))
    
    # Backward cycle: LLM -> MARL -> LLM
    mapped_llm_cycle = model.apply(params, mapped_marl, mode="fwd")
    loss_cycle_bwd = jnp.mean(jnp.sum((batch_llm - mapped_llm_cycle)**2, axis=-1))
    
    loss_cycle = loss_cycle_fwd + loss_cycle_bwd
    
    # 2. Gromov-Wasserstein Alignment Loss
    # Distance matrices for the batch
    C_marl = compute_distance_matrix(batch_marl)
    C_llm = compute_distance_matrix(batch_llm)
    
    geom_marl = geometry.Geometry(cost_matrix=C_marl)
    geom_llm = geometry.Geometry(cost_matrix=C_llm)
    
    # We apply GW alignment to see how the mapped representations' geometry aligns 
    # with the target continuous geometry.
    geom_mapped_llm = geometry.Geometry(cost_matrix=compute_distance_matrix(mapped_llm))
    prob_mapped = quadratic_problem.QuadraticProblem(geom_marl, geom_mapped_llm)
    
    # Low-Rank Gromov-Wasserstein solver
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=gw_rank,
        epsilon=epsilon
    )
    
    out_mapped = solver(prob_mapped)
    gw_loss_mapped = out_mapped.reg_gw_cost
    
    # 3. Minimum Description Length / Isometry Loss
    # Pairwise distances in MARL space should match pairwise distances in mapped LLM space
    C_marl_norm = C_marl / (jnp.max(C_marl) + 1e-8)
    C_llm_mapped = compute_distance_matrix(mapped_llm)
    C_llm_mapped_norm = C_llm_mapped / (jnp.max(C_llm_mapped) + 1e-8)
    
    loss_isometry = jnp.mean((C_marl_norm - C_llm_mapped_norm)**2)
    
    total_loss = loss_cycle + gw_loss_mapped + loss_isometry
    
    metrics = {
        "loss_total": total_loss,
        "loss_cycle_fwd": loss_cycle_fwd,
        "loss_cycle_bwd": loss_cycle_bwd,
        "loss_gw": gw_loss_mapped,
        "loss_isometry": loss_isometry
    }
    
    return total_loss, metrics
