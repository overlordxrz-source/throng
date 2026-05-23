"""
jax_sim — JAX rewrite of THRONG simulation.

Pure-JAX multi-agent RL with:
  - @jit-compiled environment step
  - Flax transformer agent brain
  - Optax PPO
  - jax.lax.scan for rollouts
"""

__version__ = "0.1.0"
