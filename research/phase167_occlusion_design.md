# Phase 16.6: Barrier Occlusion Design

## Information-Theoretic Argument
The fundamental barrier to emergent language in MARL is information redundancy. If an agent can directly perceive a threat, any communication about that threat is evolutionarily useless. The cross-attention weights for the communication channel will collapse because direct perception is higher bandwidth, lower latency, and zero-noise compared to parsing a neighbor's discrete VQ token.

To force causal grounding, communication must be the *only* pathway to retrieve survival-critical information. Phase 16.6 introduces **Barrier Occlusion** to break this redundancy.

## Occlusion Mechanism
By introducing physical barriers (walls/terrain) into the environment, we can create localized "fog of war." 
If Agent A is behind a wall and a predator is approaching from the other side, Agent A should be informationally blind to the predator, even if the predator is within the raw `local_obs_radius`. Agent B, who has line-of-sight to the predator, must broadcast a warning. Agent A is forced to use the communication channel to survive.

## Implementation Spec (`observations_jax.py`)
Currently, `red_dist` is not passed as a flat scalar. Agents receive a `(2r+1) x (2r+1)` local patch array (`loc_env`), where index 1 represents the red predator map via `get_local_patches(red_map...)`.

Additionally, `_mask_loc_env_red_channel` zero-masks the red channel if the minimum Chebyshev distance to a predator is greater than `det_r`.

**The Occlusion Mask:**
We need to insert a raycasting or line-of-sight (LoS) check inside the observation builder.
1. When generating `loc_pres` (the red channel patch), we must cross-reference the `loc_wall` or `loc_barrier` patch.
2. If the line connecting the blue agent's center to the red agent intersects a barrier, the red agent's presence in `loc_pres` must be zeroed out.
3. This creates a genuine shadow. The agent is safe behind the wall but completely blind to what is on the other side.

## Predicted Impact
Post-occlusion, we expect initial survival rates to drop as agents lose their "x-ray vision." However, evolutionary pressure will heavily penalize ignoring the communication channel. We predict the cross-attention weights will recover, and a subsequent causal intervention test will yield an ATE > 0.05.
