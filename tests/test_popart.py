"""PopArt (critic-fix branch, 2026-11-05): output preservation under a statistics update, EMA tracking of a
climbing target mean, and the normalised-target value loss. Each property is paired with a mutant that must
fail it."""
import jax.numpy as jnp
import numpy as np

from jax_sim.rl_jax import PopArt, RunningReturnStd


def _params(rng, d=8):
    return {"params": {"head_value": {"kernel": jnp.asarray(rng.normal(size=(d, 1)), jnp.float32),
                                      "bias": jnp.asarray(rng.normal(size=(1,)), jnp.float32)},
                       "trunk": {"kernel": jnp.ones((2, 2))}}}


def _denorm_out(pa, params, x):
    h = params["params"]["head_value"]
    return pa.denormalise((x @ h["kernel"] + h["bias"])[:, 0])


def test_update_preserves_denormalised_outputs():
    rng = np.random.default_rng(0)
    pa = PopArt(beta=0.1)
    params = pa.update_and_preserve(rng.normal(3.0, 2.0, size=4096), _params(rng))
    x = jnp.asarray(rng.normal(size=(64, 8)), jnp.float32)
    before = _denorm_out(pa, params, x)
    params2 = pa.update_and_preserve(rng.normal(80.0, 5.0, size=4096), params)   # big shift in stats
    after = _denorm_out(pa, params2, x)
    assert float(jnp.max(jnp.abs(after - before))) < 1e-3 * float(jnp.max(jnp.abs(before)) + 1)
    assert params2["params"]["trunk"]["kernel"] is params["params"]["trunk"]["kernel"], "only the head is rescaled"
    # mutant: the same stats move WITHOUT rescaling the head shifts every prediction
    mutant = _denorm_out(pa, params, x)
    assert float(jnp.max(jnp.abs(mutant - before))) > 1.0


def test_ema_tracks_the_current_scale_where_pooled_std_lags():
    rng = np.random.default_rng(1)
    pa, pooled = PopArt(beta=0.1), RunningReturnStd()
    params = _params(rng)
    for i in range(80):                       # target mean climbs 0 -> ~100, per-update spread stays ~2
        x = rng.normal(min(i, 40) * 2.5, 2.0, size=2048)
        params = pa.update_and_preserve(x, params)
        pooled.update(x)
    assert pa.sigma < 6.0, pa.sigma            # tracks the recent spread (plus a little residual climb)
    assert pooled.std > 20.0, pooled.std       # mutant: pooled Welford is dominated by the climb


def test_popart_value_loss_uses_normalised_target():
    import jax
    from jax_sim.network_jax import NetworkOutputs
    from jax_sim.rl_jax import ppo_loss

    M, NA = 8, 12
    def fake_apply(params, carries, obs, n_layers, detach_value=False, **kw):
        z = jnp.zeros((M,))
        return carries, NetworkOutputs(
            action_logits=jnp.zeros((M, NA)), signal_out=jnp.zeros((M, 1)), symbol_write=jnp.zeros((M, 1)),
            values=params["v"], tom_logits=jnp.zeros((M, 1, NA)), token_ids=jnp.zeros((M,), jnp.int32),
            alarm_out=None, loss_vq=z, z_e=jnp.zeros((M, 1)), culture_fast=jnp.zeros((M, 1)),
            culture_slow=jnp.zeros((M, 1)))
    mu, sigma = 100.0, 4.0
    ret = jnp.full((M,), 104.0)
    def vf(v_hat):
        _, m = ppo_loss({"v": v_hat}, fake_apply, jnp.zeros((M, 2)), jnp.zeros((M,), jnp.int32), None,
                        jnp.full((M,), -np.log(NA)), jnp.zeros((M,)), ret, jnp.zeros((M, 2)), 1, jnp.zeros((M,)),
                        vf_coef=1.0, ent_coef=0.0, vq_coef=0.0, value_scale=sigma, value_shift=mu,
                        value_clip=False, popart_mode=True)
        return m["ppo_vf_loss"]
    assert float(vf(jnp.full((M,), 1.0))) < 1e-8            # v_hat = (104 - 100) / 4 = 1 -> zero loss
    assert float(vf(jnp.full((M,), 0.0))) > 0.1
    g = jax.grad(vf)(jnp.full((M,), 0.0))
    assert bool(jnp.all(g < 0)), "pulls v_hat up toward the normalised target"
