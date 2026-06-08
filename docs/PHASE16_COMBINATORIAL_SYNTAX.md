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

## Evaluation Metrics (TRE & PosDis)
To detect non-trivial compositionality and grammatical structure, we will deploy two new metrics in `decode_signals.py`:
1. **Positional Disentanglement (PosDis)**: Measures whether specific DCVQ codebook dimensions spontaneously specialize to encode independent variables (e.g., target type vs bearing). A PosDis score approaching 1.0 indicates rigid grammatical slots.
2. **Tree Reconstruction Error (TRE)**: Evaluates whether messages compose according to the structural derivations of the inputs.

## Execution Scaffold: Composition through Decomposition (CtD)
To prevent the emergence of degenerate lookup codes, we will employ a CtD phase gate:
- **0–100k Steps**: Big Greens are deployed but are catchable *solo* at a reduced reward (`reward=2.0`). This forces the VQ codebook to learn target discrimination (nouns) without the pressure of coordination (verbs).
- **100k+ Steps**: Cooperative threshold activates. Big Greens now require simultaneous multi-agent strikes, unlocking the maximum reward (`8.0`) but risking a mauling penalty (`-1.0`) if attempted alone.

## Stag Hunt Payoff Mathematics
Standard PPO with large negative penalties (e.g., `-5.0`) for solo strikes leads to "risk-dominant" convergence (foraging small blues instead of hunting big greens). A moderate `-1.0` penalty is required to keep the cooperative basin ("payoff-dominant" cooperative equilibrium) accessible to policy gradients. The `+1.5` `r_coord` bonus ensures the total cooperative reward (`8.0 + 1.5 = 9.5`) cleanly dominates the baseline foraging behavior.
