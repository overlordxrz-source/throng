"""
Rule 13, instance 9 (docs/WILL_RESTART_SEP2026.md): `ppo_update` computed
`n_minibatches = M // minibatch_size` with no floor. At M=160 (a small
smoke-test population) against the default `ppo_minibatch_size=512`, this
silently produced n_minibatches=0 — the minibatch loop body never executed,
no gradient step was ever taken, and the caller still got a
"Blue PPO done in 0.1s" success line and an empty metrics dict back. Several
smoke tests earlier in this project's history "validated" PPO at small
population sizes without ever exercising a real minibatch.

`ppo_update` now raises a clear ValueError instead of silently returning a
no-op. This test pins that: it must raise when M < minibatch_size, and it
must NOT raise (and must actually run) when M >= minibatch_size.
"""

import jax
import jax.numpy as jnp

from jax_sim.network_jax import NetworkOutputs
from jax_sim.rl_jax import ppo_update, create_optimizer


def _tiny_batch(T, N):
    return {
        "obs": jnp.zeros((T, N, 4)),
        "actions": jnp.zeros((T, N), dtype=jnp.int32),
        "log_probs": jnp.zeros((T, N)),
        "rewards": jnp.zeros((T, N)),
        "dones": jnp.zeros((T, N)),
        "values": jnp.zeros((T, N)),
        "carries": jnp.zeros((T, N, 8)),
        "alive": jnp.ones((T, N)),
    }


def _tiny_params_and_apply():
    """A minimal fake network: action_logits linear in obs via `kernel`, so
    the params tree is actually differentiable (needed for a real PPO
    backward pass, not just a shape check)."""
    params = {"head_action": {"kernel": jnp.zeros((4, 4))}}

    def apply_fn(p, c, o, nl, **kw):
        m = o.shape[0]
        action_logits = o @ p["head_action"]["kernel"]
        return c, NetworkOutputs(
            action_logits=action_logits,
            signal_out=None, symbol_write=None,
            values=jnp.zeros((m,)),
            tom_logits=None, token_ids=None, alarm_out=None,
            loss_vq=jnp.zeros((m,)), z_e=None,
            culture_fast=None, culture_slow=None,
        )
    return params, apply_fn


def test_ppo_update_raises_when_m_below_minibatch_size():
    T, N = 8, 20  # M = 160
    params, apply_fn = _tiny_params_and_apply()
    opt = create_optimizer()
    opt_state = opt.init(params)
    batch = _tiny_batch(T, N)

    try:
        ppo_update(
            params, opt_state, opt, apply_fn, batch, n_layers=2,
            key=jax.random.PRNGKey(0), minibatch_size=512, team="blue",
        )
    except ValueError as e:
        assert "M=160" in str(e) and "minibatch_size=512" in str(e)
    else:
        raise AssertionError(
            "ppo_update must raise when M < minibatch_size instead of silently "
            "running zero minibatches (Rule 13, instance 9)"
        )


def test_ppo_update_runs_when_m_at_least_minibatch_size():
    T, N = 8, 20  # M = 160
    params, apply_fn = _tiny_params_and_apply()
    opt = create_optimizer()
    opt_state = opt.init(params)
    batch = _tiny_batch(T, N)

    # Should NOT raise: 160 >= 32, so n_minibatches = 5.
    new_params, new_opt_state, metrics = ppo_update(
        params, opt_state, opt, apply_fn, batch, n_layers=2,
        key=jax.random.PRNGKey(0), minibatch_size=32, team="blue",
    )
    assert metrics, "metrics must be non-empty when minibatches actually ran"


if __name__ == "__main__":
    test_ppo_update_raises_when_m_below_minibatch_size()
    test_ppo_update_runs_when_m_at_least_minibatch_size()
    print("OK: ppo_update fails loudly on M < minibatch_size and runs cleanly otherwise.")
