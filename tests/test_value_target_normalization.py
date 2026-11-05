"""Value-loss fix (2026-11-04, Cam -- learner testbed Step 0, critic-fix branch): opt-in via value_clip=False --
no raw-units value clip; value error measured in running-return-std units. Default (value_clip=True) must stay
exactly the production formula.

The old loss clipped predictions to old_values +/- clip_eps (0.2) in RAW return units and took the max of the
clipped and unclipped Huber terms. Once V moved more than 0.2 toward a far target, the clipped term dominated and
its gradient was zero -- V could advance at most 0.2 per update. Testbed EV sat at ~0 in every arm.

Each behavioural assertion is paired with the old formula, which must fail it (mutation check).
"""
import jax
import jax.numpy as jnp
import numpy as np

from jax_sim.network_jax import NetworkOutputs
from jax_sim.rl_jax import RunningReturnStd, ppo_loss

M, NA = 16, 12


def _fake_apply(params, carries, obs, n_layers, detach_value=False, **kw):
    z = jnp.zeros((M,))
    outs = NetworkOutputs(
        action_logits=jnp.zeros((M, NA)), signal_out=jnp.zeros((M, 1)), symbol_write=jnp.zeros((M, 1)),
        values=params["v"], tom_logits=jnp.zeros((M, 1, NA)), token_ids=jnp.zeros((M,), jnp.int32),
        alarm_out=None, loss_vq=z, z_e=jnp.zeros((M, 1)), culture_fast=jnp.zeros((M, 1)),
        culture_slow=jnp.zeros((M, 1)),
    )
    return carries, outs


def _vf_loss(v, returns, old_values, value_scale=1.0, value_clip=False):
    loss, mets = ppo_loss(
        {"v": v}, _fake_apply, jnp.zeros((M, 4)), jnp.zeros((M,), jnp.int32), None,
        jnp.full((M,), -np.log(NA)), jnp.zeros((M,)), returns, jnp.zeros((M, 4)), 1, old_values,
        clip_eps=0.2, vf_coef=1.0, ent_coef=0.0, vq_coef=0.0, value_scale=value_scale, value_clip=value_clip,
    )
    return mets["ppo_vf_loss"]


def _old_clipped_vf_loss(v, returns, old_values, clip_eps=0.2, delta=0.5):
    """The pre-fix formula, verbatim in substance -- the mutant."""
    hub = lambda e: jnp.where(e < delta, 0.5 * e ** 2, delta * (e - 0.5 * delta))
    v_cl = old_values + jnp.clip(v - old_values, -clip_eps, clip_eps)
    return jnp.mean(jnp.maximum(hub(jnp.abs(v - returns)), hub(jnp.abs(v_cl - returns))))


def test_value_gradient_survives_past_the_old_clip_range():
    old = jnp.zeros((M,))
    v = old + 0.5          # already moved 0.5 toward the target -- past the old 0.2 clip
    ret = old + 5.0        # target still far away
    g_new = jax.grad(lambda x: _vf_loss(x, ret, old))(v)
    g_old = jax.grad(lambda x: _old_clipped_vf_loss(x, ret, old))(v)
    assert float(jnp.abs(g_new).sum()) > 0.0, "fixed loss must still pull V toward its target"
    assert float(jnp.abs(g_old).sum()) == 0.0, "mutant check: the old clipped loss freezes V here"
    assert bool(jnp.all(g_new < 0)), "gradient must point toward the (higher) target"


def test_value_loss_is_invariant_to_return_scale_given_running_std():
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.normal(size=M), jnp.float32)
    ret = jnp.asarray(rng.normal(size=M) * 3, jnp.float32)
    old = jnp.zeros((M,))
    base = float(_vf_loss(v, ret, old, value_scale=1.0))
    for k in (0.1, 7.0, 50.0):
        scaled = float(_vf_loss(v * k, ret * k, old * k, value_scale=k))
        assert abs(scaled - base) < 1e-4 * max(1.0, base), (k, scaled, base)
    # mutant: without the scale, the same problem at 50x reads as a different loss regime
    assert abs(float(_vf_loss(v * 50, ret * 50, old * 50, value_scale=1.0)) - base) > 1.0


def test_running_return_std_matches_pooled_moments():
    rng = np.random.default_rng(1)
    batches = [rng.normal(loc=i, scale=1 + i, size=(512, 8)) for i in range(5)]
    rms = RunningReturnStd()
    for b in batches:
        rms.update(b)
    pooled = np.concatenate([b.ravel() for b in batches])
    assert abs(rms.mean - pooled.mean()) < 1e-3
    assert abs(rms.std - pooled.std()) < 1e-3


def test_default_path_is_the_unchanged_production_formula():
    rng = np.random.default_rng(2)
    old = jnp.asarray(rng.normal(size=M), jnp.float32)
    v = old + jnp.asarray(rng.normal(size=M), jnp.float32)
    ret = jnp.asarray(rng.normal(size=M) * 4, jnp.float32)
    default = float(_vf_loss(v, ret, old, value_scale=7.0, value_clip=True))  # value_scale ignored on this path
    assert abs(default - float(_old_clipped_vf_loss(v, ret, old))) < 1e-5
    g = jax.grad(lambda x: _vf_loss(x, old + 5.0, old, value_clip=True))(old + 0.5)
    assert float(jnp.abs(g).sum()) == 0.0, "production path keeps the clip (V frozen past +/-0.2)"
