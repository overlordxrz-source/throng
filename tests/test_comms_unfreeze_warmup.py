"""Cam's fix (2026-09-15) for the freeze-exit collapse measured on the
2026-09-14 run: the stage-0 comms freeze (tests/test_comms_freeze.py) holds
Adam's m/v at exactly zero for COMMS_SUBTREE_KEYS, since b_opt_state starts
zero-initialized and every frozen step feeds it a zero gradient. That makes
the freeze bitwise-exact, which was the point. It also means the first
unfrozen gradient hits Adam's t=1 case: m_hat = g, v_hat = g**2, so
m_hat/sqrt(v_hat) = sign(g) regardless of g's magnitude -- a maximal
lr*sign(g) step on whatever trunk drift accumulated during the freeze. That
is what collapsed codes_active=47|52|46 -> 9|10|13 in one update at
ppo=2582->2583 on the live run (grad_norm was unremarkable that update --
the gradient was fine, the Adam-normalized step was not).

Fix (jax_sim/rl_jax.py, _minibatch_step): comms_lr_warmup scales the
already-Adam-normalized update for COMMS_SUBTREE_KEYS, after
optimizer.update() and before optax.apply_updates() -- not the gradient fed
into Adam, which would not work (Adam is scale-invariant to its first input
by construction, the same property that made the freeze exact). Verification
here is exact, not statistical: same starting (params, opt_state) -- reached
via several real frozen steps, so opt_state genuinely has the zeroed m/v the
live bug depended on -- same batch, same RNG, comms_lr_warmup=0.0 must leave
the comms subtree bitwise unchanged (defuses the spring completely), and
comms_lr_warmup=0.5 must move it by exactly half of what comms_lr_warmup=1.0
does -- proving the scaling is linear on the applied step, plus a negative
control that head_action (never scaled) is identical regardless of
comms_lr_warmup, proving the warmup is scoped to the comms subtree only.
"""

import jax
import jax.numpy as jnp

from jax_sim.rl_jax import _minibatch_step, create_optimizer, COMMS_SUBTREE_KEYS
from tests.test_comms_freeze import (
    N, N_LAYERS, N_ACTIONS, OBS_DIM, HIDDEN, VQ_COEF, VF_COEF, ENT_COEF,
    _apply_fn, _build_blue, _run_steps,
)


def _one_batch(seed: int):
    k_obs, k_c, k_act, k_alarm, k_adv, k_ret, k_rng = jax.random.split(
        jax.random.PRNGKey(seed), 7
    )
    obs = jax.random.normal(k_obs, (N, OBS_DIM)) * 0.1
    obs = obs.at[:, 2].set(0.5)
    carries = jax.random.normal(k_c, (N, HIDDEN)) * 0.1
    actions = jax.random.randint(k_act, (N,), 0, N_ACTIONS)
    alarm_actions = jax.random.randint(k_alarm, (N,), 0, 2)
    old_log_probs = jnp.zeros((N,))
    advantages = jax.random.normal(k_adv, (N,))
    returns = jax.random.normal(k_ret, (N,))
    old_values = jnp.zeros((N,))
    alive = jnp.ones((N,))
    return (obs, actions, alarm_actions, old_log_probs, advantages, returns,
            carries, old_values, alive, k_rng)


def _post_freeze_state():
    """params/opt_state after several real frozen steps -- opt_state's m/v
    for the comms subtree are genuinely zero here, the same state the live
    freeze-exit spring depended on (not a fixture shortcut)."""
    model, params0 = _build_blue()
    optimizer = create_optimizer(lr=3e-4, max_grad_norm=2.0)
    opt_state0 = optimizer.init(params0)
    apply_fn = _apply_fn(model)

    params, opt_state = params0, opt_state0
    for step in range(5):
        (obs, actions, alarm_actions, old_log_probs, advantages, returns,
         carries, old_values, alive, k_rng) = _one_batch(1000 * step)
        params, opt_state, _m, _g = _minibatch_step(
            params, opt_state, apply_fn, optimizer,
            obs, actions, alarm_actions, old_log_probs, advantages, returns, carries,
            N_LAYERS, old_values, 0.2, VF_COEF, ENT_COEF, VQ_COEF, None, alive, k_rng,
            None, 0.1, 0.0, True,  # freeze_comms=True
        )
    for key in COMMS_SUBTREE_KEYS:
        for before, after in zip(jax.tree_util.tree_leaves(params0[key]), jax.tree_util.tree_leaves(params[key])):
            assert bool(jnp.array_equal(before, after)), "fixture bug: freeze wasn't exact, test premise broken"
    return apply_fn, optimizer, params, opt_state


def _first_unfrozen_step(apply_fn, optimizer, params, opt_state, comms_lr_warmup, seed=99):
    (obs, actions, alarm_actions, old_log_probs, advantages, returns,
     carries, old_values, alive, k_rng) = _one_batch(seed)
    new_params, _opt_state, _m, _g = _minibatch_step(
        params, opt_state, apply_fn, optimizer,
        obs, actions, alarm_actions, old_log_probs, advantages, returns, carries,
        N_LAYERS, old_values, 0.2, VF_COEF, ENT_COEF, VQ_COEF, None, alive, k_rng,
        None, 0.1, 0.0, False,  # freeze_comms=False -- this IS the unfrozen step
        comms_lr_warmup,
    )
    return new_params


def test_warmup_zero_leaves_comms_subtree_bitwise_unchanged():
    apply_fn, optimizer, params, opt_state = _post_freeze_state()
    new_params = _first_unfrozen_step(apply_fn, optimizer, params, opt_state, comms_lr_warmup=0.0)

    for key in COMMS_SUBTREE_KEYS:
        for before, after in zip(jax.tree_util.tree_leaves(params[key]), jax.tree_util.tree_leaves(new_params[key])):
            assert bool(jnp.array_equal(before, after)), (
                f"{key} moved on the first unfrozen step even with comms_lr_warmup=0.0 -- "
                f"the warmup did not suppress the freeze-exit step."
            )


def test_warmup_scales_the_applied_step_linearly():
    apply_fn, optimizer, params, opt_state = _post_freeze_state()

    params_full = _first_unfrozen_step(apply_fn, optimizer, params, opt_state, comms_lr_warmup=1.0)
    params_half = _first_unfrozen_step(apply_fn, optimizer, params, opt_state, comms_lr_warmup=0.5)

    for key in COMMS_SUBTREE_KEYS:
        for base, full, half in zip(
            jax.tree_util.tree_leaves(params[key]),
            jax.tree_util.tree_leaves(params_full[key]),
            jax.tree_util.tree_leaves(params_half[key]),
        ):
            delta_full = full - base
            delta_half = half - base
            # atol here is float32 JIT-compilation noise (~1e-8, confirmed by
            # direct inspection), several orders below delta_full's own
            # magnitude (~1e-4) -- not slack in what's being proven.
            assert jnp.allclose(delta_half, 0.5 * delta_full, atol=1e-6), (
                f"{key}: comms_lr_warmup=0.5 did not produce exactly half the step of "
                f"comms_lr_warmup=1.0 -- the warmup is not scaling the applied update linearly."
            )
        # and warmup=1.0 must be a genuine no-op vs the unscaled path -- some leaf must
        # actually have moved, or this comparison is vacuous.
    moved_any = any(
        not bool(jnp.array_equal(b, a))
        for key in COMMS_SUBTREE_KEYS
        for b, a in zip(jax.tree_util.tree_leaves(params[key]), jax.tree_util.tree_leaves(params_full[key]))
    )
    assert moved_any, "comms subtree didn't move at comms_lr_warmup=1.0 -- fixture generates no real gradient"


def test_warmup_does_not_touch_head_action_negative_control():
    """head_action is never in COMMS_SUBTREE_KEYS -- comms_lr_warmup must not
    change its update at all, proving the scaling is scoped correctly."""
    apply_fn, optimizer, params, opt_state = _post_freeze_state()

    params_full = _first_unfrozen_step(apply_fn, optimizer, params, opt_state, comms_lr_warmup=1.0)
    params_zero = _first_unfrozen_step(apply_fn, optimizer, params, opt_state, comms_lr_warmup=0.0)

    for full, zero in zip(
        jax.tree_util.tree_leaves(params_full["head_action"]),
        jax.tree_util.tree_leaves(params_zero["head_action"]),
    ):
        assert bool(jnp.array_equal(full, zero)), (
            "head_action differs between comms_lr_warmup=0.0 and 1.0 -- the warmup is "
            "leaking outside the comms subtree."
        )
    moved = any(
        not bool(jnp.array_equal(b, a))
        for b, a in zip(jax.tree_util.tree_leaves(params["head_action"]), jax.tree_util.tree_leaves(params_full["head_action"]))
    )
    assert moved, "head_action didn't move at all in this fixture -- negative control is vacuous"


if __name__ == "__main__":
    test_warmup_zero_leaves_comms_subtree_bitwise_unchanged()
    test_warmup_scales_the_applied_step_linearly()
    test_warmup_does_not_touch_head_action_negative_control()
    print(
        "OK: comms_lr_warmup=0.0 leaves the comms subtree bitwise unchanged on the first "
        "unfrozen step (defuses the freeze-exit spring completely); comms_lr_warmup=0.5 "
        "produces exactly half the step of 1.0 (linear scaling of the applied update, not "
        "the gradient); head_action is untouched by the warmup at any value (scoped "
        "correctly to the comms subtree)."
    )
