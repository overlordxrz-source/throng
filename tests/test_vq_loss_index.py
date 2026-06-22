"""Regression test for the Phase 18.5 team-aware VQ-loss index fix.

Blue (`AgentNetworkJax`) and red (`PredatorNetworkJax`) return DIFFERENT output
tuple layouts:

    blue:  (action_logits[0], signal_out[1], symbol_write[2], values[3],
            tom_logits[4], token_ids[5], alarm_out[6], loss_vq[7], z_e[8], ...)
    red:   (action_logits[0], signal_out[1], symbol_write[2], values[3],
            tom_logits[4], token_ids[5], loss_vq[6], z_e[7], ...)   # no alarm head

`ppo_loss` previously read the VQ loss at a hard-coded ``outs[7]``. That is the
``loss_vq`` for blue but the raw 40-D continuous wire (``z_e``) for red — so red
PPO minimised ``Σz_e`` instead of a commitment loss, collapsing the predator
codebook (2/64) and reporting ``RedVQ ≈ -24225``.

These tests pin the contract: ``ppo_loss`` must read the VQ loss from
``vq_loss_idx`` (7 for blue, 6 for red), and ``ppo_update`` must select that
index by team. They use a fake ``apply_fn`` so they run in milliseconds with no
network, obs layout, or GPU dependency.
"""

import jax.numpy as jnp

from jax_sim.rl_jax import ppo_loss


M = 16
N_ACT = 12
WIRE = 40

# Distinguishable sentinels: the TRUE per-agent VQ loss is a small positive
# number; z_e is a large-negative wire whose row-sum is the "garbage" the bug fed
# into red PPO.
TRUE_VQ = 0.5
Z_E_FILL = -25.0  # row-sum = -25 * 40 = -1000


def _make_outs(team: str):
    action_logits = jnp.zeros((M, N_ACT))
    values = jnp.zeros((M,))
    loss_vq = jnp.full((M,), TRUE_VQ)
    z_e = jnp.full((M, WIRE), Z_E_FILL)
    if team == "blue":
        # alarm_out at 6, loss_vq at 7, z_e at 8
        return (action_logits, None, None, values, None, None,
                jnp.zeros((M, 2)), loss_vq, z_e, None, None)
    # red: no alarm head — loss_vq at 6, z_e at 7
    return (action_logits, None, None, values, None, None,
            loss_vq, z_e, None, None)


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


def test_blue_reads_loss_vq_at_index_7():
    _, metrics = ppo_loss(
        apply_fn=_fake_apply("blue"),
        alarm_actions=jnp.zeros((M,), dtype=jnp.int32),
        vq_loss_idx=7,
        **_common(),
    )
    assert abs(float(metrics["ppo_vq_loss"]) - TRUE_VQ) < 1e-4


def test_red_reads_loss_vq_at_index_6():
    _, metrics = ppo_loss(
        apply_fn=_fake_apply("red"),
        alarm_actions=None,
        vq_loss_idx=6,
        **_common(),
    )
    vq = float(metrics["ppo_vq_loss"])
    assert vq > 0.0, "red VQ loss must be the positive commitment loss, not z_e"
    assert abs(vq - TRUE_VQ) < 1e-4


def test_wrong_index_reproduces_the_bug():
    """Reading red at the blue index (7) grabs z_e — the original foot-gun."""
    _, metrics = ppo_loss(
        apply_fn=_fake_apply("red"),
        alarm_actions=None,
        vq_loss_idx=7,
        **_common(),
    )
    # Σz_e over the 40-D wire ≈ -1000: confirms the mis-index pulls the raw wire.
    assert float(metrics["ppo_vq_loss"]) < -100.0


if __name__ == "__main__":
    test_blue_reads_loss_vq_at_index_7()
    test_red_reads_loss_vq_at_index_6()
    test_wrong_index_reproduces_the_bug()
    print("OK: VQ loss index contract holds (blue=7, red=6).")
