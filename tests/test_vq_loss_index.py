"""Regression test for the Phase 18.5 team-aware VQ-loss index fix.

Blue (`AgentNetworkJax`) and red (`PredatorNetworkJax`) used to return DIFFERENT
positional output tuples:

    blue:  (action_logits[0], signal_out[1], symbol_write[2], values[3],
            tom_logits[4], token_ids[5], alarm_out[6], loss_vq[7], z_e[8], ...)
    red:   (action_logits[0], signal_out[1], symbol_write[2], values[3],
            tom_logits[4], token_ids[5], loss_vq[6], z_e[7], ...)   # no alarm head

`ppo_loss` previously read the VQ loss at a hard-coded ``outs[7]``. That is the
``loss_vq`` for blue but the raw 40-D continuous wire (``z_e``) for red — so red
PPO minimised ``Σz_e`` instead of a commitment loss, collapsing the predator
codebook (2/64) and reporting ``RedVQ ≈ -24225``. The interim fix threaded a
``vq_loss_idx`` kwarg (7 for blue, 6 for red) through `ppo_loss`/`ppo_update`.

The July `NetworkOutputs` migration (H1) replaced the positional tuple with a
`flax.struct.dataclass` whose `__getitem__` raises `TypeError` by design —
there is no integer index left to get wrong, for either team. `ppo_loss` now
reads `outs.loss_vq` unconditionally; the field exists on both blue's and
red's `NetworkOutputs` instance regardless of whether `alarm_out` is present.
This test no longer pins an index (there isn't one); it pins the invariant
that made the index moot: `ppo_loss` reports the true positive commitment
loss for BOTH teams, built from `NetworkOutputs` — blue with `alarm_out` set,
red with `alarm_out=None` — rather than a raw tuple where team-shape drift can
silently disagree with hard-coded call-site knowledge. They use a fake
`apply_fn` so they run in milliseconds with no network, obs layout, or GPU
dependency.
"""

import jax.numpy as jnp

from jax_sim.network_jax import NetworkOutputs
from jax_sim.rl_jax import ppo_loss


M = 16
N_ACT = 12
WIRE = 40

# Distinguishable sentinels: the TRUE per-agent VQ loss is a small positive
# number; z_e is a large-negative wire whose row-sum would be "garbage" if a
# caller ever mistakenly read z_e where loss_vq belongs.
TRUE_VQ = 0.5
Z_E_FILL = -25.0  # row-sum = -25 * 40 = -1000


def _make_outs(team: str) -> NetworkOutputs:
    action_logits = jnp.zeros((M, N_ACT))
    values = jnp.zeros((M,))
    loss_vq = jnp.full((M,), TRUE_VQ)
    z_e = jnp.full((M, WIRE), Z_E_FILL)
    return NetworkOutputs(
        action_logits=action_logits,
        signal_out=None,
        symbol_write=None,
        values=values,
        tom_logits=None,
        token_ids=None,
        alarm_out=jnp.zeros((M, 2)) if team == "blue" else None,
        loss_vq=loss_vq,
        z_e=z_e,
        culture_fast=None,
        culture_slow=None,
    )


def _fake_apply(team: str):
    def _apply(params, carries, obs, n_layers, detach_value=False, **kwargs):
        return carries, _make_outs(team)
    return _apply


def _common():
    return dict(
        params={},
        obs=jnp.zeros((M, 4)),
        actions=jnp.zeros((M,), dtype=jnp.int32),
        old_log_probs=jnp.zeros((M,)),
        advantages=jnp.zeros((M,)),
        returns=jnp.zeros((M,)),
        carries=jnp.zeros((M, 8)),
        n_layers=2,
        old_values=jnp.zeros((M,)),
        alive=jnp.ones((M,)),
    )


def test_blue_reports_true_vq_loss():
    _, metrics = ppo_loss(
        apply_fn=_fake_apply("blue"),
        alarm_actions=jnp.zeros((M,), dtype=jnp.int32),
        **_common(),
    )
    assert abs(float(metrics["ppo_vq_loss"]) - TRUE_VQ) < 1e-4


def test_red_reports_true_vq_loss_with_no_alarm_head():
    _, metrics = ppo_loss(
        apply_fn=_fake_apply("red"),
        alarm_actions=None,
        **_common(),
    )
    vq = float(metrics["ppo_vq_loss"])
    assert vq > 0.0, "red VQ loss must be the positive commitment loss, not z_e"
    assert abs(vq - TRUE_VQ) < 1e-4


def test_networkoutputs_rejects_positional_indexing():
    """The dataclass makes the original bug class structurally impossible:
    there is no integer index left for a team-shape mismatch to silently
    read the wrong field from."""
    outs = _make_outs("red")
    try:
        outs[7]
    except TypeError:
        pass
    else:
        raise AssertionError("NetworkOutputs.__getitem__ should raise TypeError")


if __name__ == "__main__":
    test_blue_reports_true_vq_loss()
    test_red_reports_true_vq_loss_with_no_alarm_head()
    test_networkoutputs_rejects_positional_indexing()
    print("OK: ppo_loss reads the true VQ commitment loss for both teams via NetworkOutputs.")
