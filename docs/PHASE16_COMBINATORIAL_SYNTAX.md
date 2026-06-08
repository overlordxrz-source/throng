# Phase 16: Open-Ended Combinatorial Complexity

## The Hypothesis
To force the emergence of combinatorial syntax (nouns, verbs, adjectives), the environment's complexity must exceed the capacity of a flat spatial hash table. By introducing "Big Greens" (requiring cooperative strikes) alongside "Small Blues" (solo catches), reds must encode not just spatial coordinates, but also target types (nouns) and intended actions (verbs).

## JAX Cooperative Threshold Mechanics
We must evaluate cooperative strikes purely spatially within the current 512-step slice to avoid bloating the recurrent state and triggering an XLA OOM.

**Mechanism:**
When a red agent executes a `Strike` action near a Big Green, we calculate the number of adjacent reds who are *also* executing a `Strike` action on that exact timestep.
```python
# Pseudo JAX for stateless evaluation in env_step
red_striking = (red_actions == ACTION_STRIKE)
# Convolve to count striking neighbors in Chebyshev distance <= 1
moore_kernel = jnp.ones((3, 3))
striking_neighbors = jax.scipy.signal.convolve2d(red_striking_grid, moore_kernel, mode='same')
successful_strike = red_striking & (striking_neighbors >= 2) & near_big_green
```
This is fully stateless and evaluated entirely within the transition function `env_step` without carrying any "lock" across `lax.scan`.

## Action Space Expansion
We need to expand from 5 actions (N/S/E/W/Stay) to 8 actions (+ Strike, Push, Guard).

**Constraint:** We cannot break the existing MAPPO actor heads.

**Proposed Solution: Parameter Grafting with Masked Initialization**
1. Update `num_actions = 8` in the config.
2. In `init_predator_params`, when loading the old checkpoint (which has `head_action.kernel` of shape `[H, 5]`), we will dynamically allocate a new kernel of shape `[H, 8]`.
3. We will **copy** the learned weights for the first 5 actions directly from the checkpoint to preserve movement semantics.
4. We will initialize the biases for the 3 new actions to heavily negative values (e.g., `-5.0`). This ensures the new actions start with near-zero probability, preserving the current policy's performance while allowing PPO entropy to gradually explore the new interactions over time.
