# THRONG Contingency Code: Advanced VQ Bottlenecks (Phase 14.4 / 15)

This document contains the exact JAX/Flax blueprints synthesized for the Phase 14.4 / 15 contingencies. If the Phase 14.3 GWT Router fails the spatial decode gate (or if representation collapse occurs in future multi-task scaling), these modules can be directly adapted into `jax_sim/network_jax.py`.

## 1. SimVQ (Linear Reparameterization)
Prevents disjoint optimization (the flaw of the Straight-Through Estimator) by updating the entire vocabulary space via a learnable linear transformation layer. Prevents "dead codes".

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class SimVQ(nn.Module):
    n_e: int          # Vocabulary/codebook size (K)
    e_dim: int        # Latent dimension (d)
    beta: float = 0.25 # Commitment loss multiplier

    def setup(self):
        # Base codebook coefficient matrix (C)
        self.C = self.param('C', 
                            nn.initializers.normal(stddev=1.0 / self.e_dim), 
                            (self.n_e, self.e_dim))
        # Shared basis transformation matrix (W)
        self.W = self.param('W', 
                            nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"), 
                            (self.e_dim, self.e_dim))

    def __call__(self, z):
        # 1. Compute effective lookup codebook E = C * W
        E = jnp.matmul(self.C, self.W)  # Shape: (K, d)

        # 2. Distance calculation
        z_sq = jnp.sum(z**2, axis=-1, keepdims=True)
        e_sq = jnp.sum(E**2, axis=-1)
        distances = z_sq + e_sq - 2.0 * jnp.matmul(z, E.T)

        # 3. Quantization
        indices = jnp.argmin(distances, axis=-1)
        z_q = E[indices]

        # 4. STE Gradient Routing
        z_q_ste = z + jax.lax.stop_gradient(z_q - z)

        # 5. Loss
        loss_codebook = jnp.mean((jax.lax.stop_gradient(z) - z_q) ** 2)
        loss_commit = self.beta * jnp.mean((z - jax.lax.stop_gradient(z_q)) ** 2)
        vq_loss = loss_codebook + loss_commit

        return z_q_ste, indices, vq_loss
```

## 2. DCVQ (Divide-and-Conquer VQ)
Splits the latent space into independent parallel subspaces (e.g., creating Subject/Direction syntactic slots). Multiplies representation capacity.

```python
class DCVQ(nn.Module):
    n_e: int            # Subspace vocabulary size (K)
    e_dim: int          # Total concatenated latent dimension (d)
    num_subspaces: int   # Number of parallel subspaces (G)
    beta: float = 0.25

    def setup(self):
        assert self.e_dim % self.num_subspaces == 0
        self.sub_dim = self.e_dim // self.num_subspaces

        self.embeddings = self.param('subspace_embeddings',
                                     nn.initializers.normal(stddev=1.0 / self.sub_dim),
                                     (self.num_subspaces, self.n_e, self.sub_dim))

    def __call__(self, z):
        batch_size = z.shape[0]

        # 1. Partition latent space
        z_split = z.reshape((batch_size, self.num_subspaces, self.sub_dim))

        # 2. Parallel distance calculation across G subspaces
        z_sq = jnp.sum(z_split**2, axis=-1, keepdims=True)
        e_sq = jnp.sum(self.embeddings**2, axis=-1)
        e_sq_broad = jnp.expand_dims(e_sq, axis=0)

        dot_prod = jnp.einsum('bgd,gkd->bgk', z_split, self.embeddings)
        distances = z_sq + e_sq_broad - 2.0 * dot_prod

        # 3. Parallel Quantization
        indices = jnp.argmin(distances, axis=-1)

        batch_idx = jnp.arange(batch_size)[:, None]
        group_idx = jnp.arange(self.num_subspaces)[None, :]
        z_q_split = self.embeddings[group_idx, indices]

        # 4. Reconstruction & STE
        z_q_concat = z_q_split.reshape((batch_size, self.e_dim))
        z_q_ste = z + jax.lax.stop_gradient(z_q_concat - z)

        # 5. Total Subspace Loss
        loss_codebook = jnp.mean((jax.lax.stop_gradient(z_split) - z_q_split) ** 2)
        loss_commit = self.beta * jnp.mean((z_split - jax.lax.stop_gradient(z_q_split)) ** 2)
        vq_loss = loss_codebook + loss_commit

        return z_q_ste, indices, vq_loss
```

## 3. VQ-VIB (Variational Information Bottleneck)
Adds a KL-divergence penalty on the message complexity, forcing the network to drop predictable variables (hunger) and only transmit maximum-entropy spatial surprises.

```python
class VQVIB(nn.Module):
    n_e: int
    e_dim: int
    beta: float = 0.25

    def setup(self):
        self.embedding = nn.Embedding(self.n_e, self.e_dim)

    def __call__(self, mu, logvar, rng):
        # 1. Reparameterization Trick
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, shape=mu.shape)
        z = mu + eps * std  

        # 2. Nearest Neighbor Quantizer
        E = self.embedding.embedding 
        z_sq = jnp.sum(z**2, axis=-1, keepdims=True)
        e_sq = jnp.sum(E**2, axis=-1)
        distances = z_sq + e_sq - 2.0 * jnp.matmul(z, E.T)

        indices = jnp.argmin(distances, axis=-1)
        z_q = E[indices]

        # 3. STE
        z_q_ste = z + jax.lax.stop_gradient(z_q - z)

        # 4. VQ Losses
        loss_codebook = jnp.mean((jax.lax.stop_gradient(z) - z_q) ** 2)
        loss_commit = self.beta * jnp.mean((z - jax.lax.stop_gradient(z_q)) ** 2)
        vq_loss = loss_codebook + loss_commit

        # 5. KL Divergence Loss
        kl_loss = -0.5 * jnp.sum(1.0 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
        kl_loss = jnp.mean(kl_loss)

        return z_q_ste, indices, vq_loss, kl_loss
```
