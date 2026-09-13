"""
Cam's ruling (docs/WILL_RESTART_SEP2026.md, "Confidence head — enable
training AND re-initialise the fossil"): AUDIT_SEP2026.md Finding 2 proved
head_confidence_1/2 is bit-identical between checkpoints 2244 and 2763 —
trained once in the Phase 9.1 era, then carried frozen through every graft
since. `reset_confidence_head_on_resume` is a deliberate EXCEPTION to the
grafting rule that never resets an existing parameter (every other
`ensure_*_params` helper only fills in what's MISSING). This test pins that
it actually replaces the fossil's weights rather than leaving them alone,
while every other param stays untouched.
"""

import jax
import jax.numpy as jnp

from jax_sim.network_jax import (
    AgentNetworkJax,
    init_agent_params,
    reset_confidence_head_on_resume,
    CONFIDENCE_HEAD_KEYS,
)

HIDDEN = 256
N_ACTIONS = 12


def _fresh_model_and_params():
    model = AgentNetworkJax(hidden_dim=HIDDEN, n_actions=N_ACTIONS)
    rng = jax.random.PRNGKey(0)
    obs = jnp.zeros((2, model.own_state_dim + model.obs_dim))
    carries = jnp.zeros((2, HIDDEN))
    params = init_agent_params(model, rng, carries, obs, n_layers=2)
    obs_dim = model.own_state_dim + model.obs_dim
    return model, params, obs_dim


def _make_fossil(params):
    """Simulate a checkpoint whose confidence head was actually trained at
    some point — a non-zero bias, since Flax's nn.Dense default is a ZERO
    bias, so any non-zero value could only have come from real gradient
    descent (this is exactly how AUDIT_SEP2026.md Finding 2 was proven)."""
    fossil = dict(params)
    fossil["head_confidence_1"] = {
        "kernel": params["head_confidence_1"]["kernel"],
        "bias": jnp.ones_like(params["head_confidence_1"]["bias"]) * 0.5,
    }
    return fossil


def test_reset_replaces_the_fossil_weights():
    model, params, obs_dim = _fresh_model_and_params()
    fossil = _make_fossil(params)
    before_bias = fossil["head_confidence_1"]["bias"]

    new_params = reset_confidence_head_on_resume(
        model, fossil, jax.random.PRNGKey(99), HIDDEN, obs_dim, n_layers=2,
    )

    after_bias = new_params["head_confidence_1"]["bias"]
    assert not bool(jnp.array_equal(before_bias, after_bias)), (
        "reset_confidence_head_on_resume must actually replace the existing "
        "(fossil) weights, not leave them in place — that's the whole point "
        "of it being an EXCEPTION to the normal graft-only-if-missing rule"
    )
    # Flax's default Dense bias_init is zeros — a fresh init's bias should be
    # exactly zero, confirming this really is a fresh init, not some other
    # arbitrary perturbation.
    assert bool(jnp.all(after_bias == 0.0))


def test_reset_does_not_touch_unrelated_params():
    model, params, obs_dim = _fresh_model_and_params()
    fossil = _make_fossil(params)
    before_action_kernel = fossil["head_action"]["kernel"]
    before_codebook = fossil["codebook_0"]["embedding"]

    new_params = reset_confidence_head_on_resume(
        model, fossil, jax.random.PRNGKey(99), HIDDEN, obs_dim, n_layers=2,
    )

    assert bool(jnp.array_equal(before_action_kernel, new_params["head_action"]["kernel"]))
    assert bool(jnp.array_equal(before_codebook, new_params["codebook_0"]["embedding"]))


def test_reset_is_a_noop_when_confidence_head_is_entirely_absent():
    """If the checkpoint predates the confidence head existing at all,
    there's nothing to reset — ensure_aux_head_params's normal graft-if-
    missing path handles that case, not this one."""
    model, params, obs_dim = _fresh_model_and_params()
    stripped = {k: v for k, v in params.items() if k not in CONFIDENCE_HEAD_KEYS}
    result = reset_confidence_head_on_resume(
        model, stripped, jax.random.PRNGKey(99), HIDDEN, obs_dim, n_layers=2,
    )
    assert result is stripped or set(result.keys()) == set(stripped.keys())
    for k in CONFIDENCE_HEAD_KEYS:
        assert k not in result


if __name__ == "__main__":
    test_reset_replaces_the_fossil_weights()
    test_reset_does_not_touch_unrelated_params()
    test_reset_is_a_noop_when_confidence_head_is_entirely_absent()
    print("OK: reset_confidence_head_on_resume replaces the fossil, leaves everything else alone.")
