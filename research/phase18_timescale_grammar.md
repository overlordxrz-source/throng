# Phase 18: Timescale Grammar Architecture

## Theoretical Foundation

In Phase 18, we resolved a critical failure mode observed in Phase 17: the agents were smuggling continuous spatial geometry through the discrete VQ bottleneck by rapidly oscillating between VQ tokens. This "chattering" behaviour allowed them to encode precise floating-point trajectories (via time-averaging) while bypassing the intended discrete categorical bottleneck.

To correct this, we implemented **Timescale Separation** (The "Timescale Grammar" Architecture). The core insight is that continuous geometry (pointing) and categorical urgency (alerting) naturally require different bandwidths and thermodynamic costs.

## Architectural Changes

1. **Continuous Spatial Channel (`30D`)**:
   We removed the Vector Quantization (VQ) bottleneck from the main signalling head. The agent now broadcasts a purely continuous 30-dimensional vector. This channel acts as a free "pointing" mechanism, allowing agents to share precise spatial geometry without the constraints of a discrete codebook.

2. **Discrete Categorical Channel (`2D` -> 1 bit)**:
   We introduced a new, parallel 1-bit categorical channel (`head_alarm`). This channel uses a Gumbel-Softmax straight-through estimator to output a hard binary signal (e.g., `0` for Safe, `1` for Alarm). 

3. **Thermodynamic Grounding (Metabolic Cost)**:
   Crucially, this discrete channel is NOT free. Emitting an "Alarm" signal (state `1`) directly deducts a fixed physical energy cost (`0.05`) from the agent's internal energy pool. 
   
## The "Grammar" Hypothesis

By forcing the discrete channel to carry a heavy metabolic cost, we explicitly prevent the "babble" and "chattering" behaviours seen in earlier phases. The agent cannot rapidly oscillate the discrete bit without starving to death. 

Instead, the agent must learn to use the discrete bit as a **rare, categorical modifier** to the continuous spatial channel. For example, the continuous channel may constantly broadcast "I see something at coordinates X, Y", while the discrete channel is reserved to say "...AND it's a predator, flee!"

This separation of a high-bandwidth, low-cost continuous channel (verbs/directions) and a low-bandwidth, high-cost discrete channel (nouns/categories) mimics the foundational conditions necessary for the emergence of structured grammar.

## Implementation Details

- **`jax_sim/network_jax.py`**: Modified `__call__` to concatenate the 30D continuous `z_e` with the 2D one-hot `alarm_out`, producing a 32D `signal_out` vector that seamlessly integrates with the existing observation builders.
- **`jax_sim/main_jax.py`**: Added explicit logic to deduct `0.05` energy from any agent where `b_token_ids == 1`.
