import jax
import jax.numpy as jnp
rng = jax.random.PRNGKey(42)
keys = jax.random.split(rng, 10)
step_key = jax.random.fold_in(keys[2], 1)
print(step_key.shape)
print(step_key.ndim)
k1, k2 = jax.random.split(step_key, 2)
