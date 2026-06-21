# THRONG Project Rules

## 1. Cam Context Window Resets (Horcrux Protocol)
If the user indicates that Cam has lost context or has hallucinated an outdated project state (e.g., asking to plan a phase that is already live):
- **DO NOT** just summarize the chat.
- **DO** write an authoritative "Context Sync" prompt for the user to copy-paste to Cam.
- The prompt MUST use Cam's analytical lenses (Software / Physics / RL) and explicitly state the exact current Phase, the active Git branch, the current codebook/action dimensionality, and the precise empirical telemetry from the cluster.

## 2. JAX Parameter Grafting (Amnesia Prevention)
When modifying the architecture to expand the output dimensions of existing neural network heads (e.g., increasing actions from 8 to 12):
- **NEVER** allow the layer to reinitialize due to shape mismatch. This causes catastrophic amnesia of the agents' survival policies.
- **ALWAYS** ensure `jax_sim/network_jax.py:graft_missing_param_subtrees` gracefully zero-pads the expansion. Kernels must be padded on `axis=1` (outputs), and biases on `axis=0`.

## 3. Multi-Slot Array Handling
With the transition to the Phase 18 multi-slot communication architecture, agent tokens are 2D arrays `(N, slots)`.
- When writing offline analysis or telemetry scripts that use scalar operations like `np.bincount()`, **always check `ndim`** and explicitly slice the active slot (e.g., `toks[:, 0]`) to prevent `ValueError: object too deep for desired array`.
- Do not assume `vq_token` or `token_ids` are scalar integers.
