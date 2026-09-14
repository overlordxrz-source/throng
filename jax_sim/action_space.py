"""Canonical disabled/no-op action indices, shared by rollout, imagination, and PPO.

Push (6) and Guard (7) have no dispatch in grid_jax.py for either team (true
no-ops). Build (8) is barrier construction, disabled for both teams per the
Sep 2026 restart ruling (neither team should build).

All three must be masked identically everywhere an agent's action logits are
sampled, scored, or trained on, for both blue and red. A mismatch between any
two of those sites corrupts either PPO's log-prob ratio (rollout vs backward
pass computing different distributions over the same action) or the
epistemic gate's imagined-action choice (imagination scoring an action that
rollout would never sample). This was exactly the failure mode found in the
Sep 2026 restart pre-flight: Build was masked at red's rollout but not at
blue's anywhere, nor at the shared PPO backward pass for either team.

`mask_disabled_actions` is the single operation all three call sites use --
not just the constant -- so a test can assert against the real function
production code runs, not a reimplementation of it.
"""

import jax.numpy as jnp

MASKED_ACTIONS: tuple[int, ...] = (6, 7, 8)

ACTION_NAMES: dict[int, str] = {
    6: "Push",
    7: "Guard",
    8: "Build",
}


def mask_disabled_actions(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Set MASKED_ACTIONS indices along `axis` to -1e9, if that axis is long
    enough to contain them (otherwise a no-op, e.g. a legacy 5-action config).

    `axis=-1` covers rollout/PPO logits shaped (..., n_actions); `axis=0`
    covers imagination's scores shaped (n_actions, N).
    """
    if x.shape[axis] <= max(MASKED_ACTIONS):
        return x
    idx = jnp.array(MASKED_ACTIONS)
    if axis in (-1, x.ndim - 1):
        return x.at[..., idx].set(-1e9)
    if axis == 0:
        return x.at[idx, ...].set(-1e9)
    raise NotImplementedError(f"mask_disabled_actions: unsupported axis {axis}")


def masked_actions_banner() -> str:
    """Human-readable ``Name(index)`` list, derived from MASKED_ACTIONS so the
    printed banner can never drift from the mask actually applied."""
    return ", ".join(
        f"{ACTION_NAMES.get(i, str(i))}({i})" for i in MASKED_ACTIONS
    )
