"""Cam's Blocker 2 (2026-09-14): a subnetwork whose objective is identically
uninformative during a curriculum phase must not be trained during that
phase. CtD ramp stage 0 (solo-satisfiable craft) removes all coordination
pressure, so the sender encoder (gwt_comms_1, head_signal_slot0/1/2), the
codebook (codebook_0/1/2), and the receiver read path (emb_nb) get zero
reward gradient while the VQ commitment loss keeps pulling unopposed -- a
one-way ratchet toward encoder collapse (measured directly in
scripts/diag_ze_variance.py). The dead-code reset mechanism cannot rescue a
collapsed encoder: it only redistributes codes across whatever manifold the
encoder currently produces, so if that manifold is a point, every reseeded
code is the same point.

Fix (jax_sim/rl_jax.py): freeze_comms=True zeroes gradients for the comms
subtree (jax_sim.rl_jax.COMMS_SUBTREE_KEYS) before they reach the optimizer,
for all of stage 0. Verification is exact, not statistical: run several real
frozen minibatch steps against the real _minibatch_step and confirm the
comms subtree is bitwise identical (jnp.array_equal, not allclose) between
the first and last step -- plus two negative controls proving the test can
actually detect movement: (1) head_action, which is NOT frozen, must change
under the same frozen run (proves gradients are genuinely flowing elsewhere,
not that freeze_comms silently no-ops the whole step); (2) the same comms
subtree run WITHOUT freeze_comms must change (proves the harness would have
caught a broken freeze).
"""

import jax
import jax.numpy as jnp
import yaml

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.network_jax import AgentNetworkJax
from jax_sim.obs_layout import make_obs_layout
from jax_sim.rl_jax import _minibatch_step, create_optimizer, COMMS_SUBTREE_KEYS

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})

HIDDEN = int(_cfg["hidden_dim"])
N_HEADS = int(_cfg["n_heads"])
N_LAYERS = int(_cfg["n_layers"])
SIGNAL_DIM = int(_cfg["signal_dim"])
SYMBOL_DIM = int(_cfg["symbol_dim"])
VOCAB = int(_cfg["vocab_size"])
VQ_BETA = float(_cfg.get("vq_beta", 0.25))
MEM_SLOTS = int(_cfg.get("memory_slots", 0))
NEIGHBOR_K = int(_cfg["neighbor_k"])
LOCAL_CELLS = (2 * int(_cfg["local_obs_radius"]) + 1) ** 2
ENV_CH = int(_cfg.get("env_channels", 15))
OWN_STATE = int(_cfg.get("own_state_dim", 22))
N_ACTIONS = int(_cfg.get("n_actions", 12))
VQ_COEF = float(_cfg.get("vq_loss_coef", 0.1))
VF_COEF = float(_cfg.get("ppo_value_coef", 0.25))
ENT_COEF = float(_cfg.get("ppo_entropy_coef", 0.02))

_p9 = _cfg.get("phase9_canvas") or {}
_layout = make_obs_layout(
    signal_dim=SIGNAL_DIM, symbol_dim=SYMBOL_DIM, memory_slots=MEM_SLOTS,
    neighbor_k=NEIGHBOR_K, local_cells=LOCAL_CELLS, env_channels=ENV_CH,
    own_state_dim=OWN_STATE,
)
OBS_DIM = _layout.total_dim
FWD_ENV_DIM = _layout.loc_env_end - _layout.loc_env_start

N = 32  # minibatch agents
N_STEPS = 5  # several real steps, not one -- catches slow drift, not just a single no-op


def _build_blue():
    model = AgentNetworkJax(
        hidden_dim=HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS, obs_dim=OBS_DIM,
        signal_dim=SIGNAL_DIM, symbol_dim=SYMBOL_DIM, vocab_size=VOCAB, vq_beta=VQ_BETA,
        vq_dead_code_reset=True, memory_slots=MEM_SLOTS, fwd_env_dim=FWD_ENV_DIM,
        cross_attn_enabled=bool(_p9.get("cross_attn_enabled", False)),
        cross_attn_num_heads=int(_p9.get("cross_attn_num_heads", N_HEADS)),
        env_channels=ENV_CH, own_state_dim=OWN_STATE, n_actions=N_ACTIONS,
        local_cells=LOCAL_CELLS, neighbor_k=NEIGHBOR_K,
    )
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    obs = jax.random.normal(k1, (N, OBS_DIM)) * 0.1
    obs = obs.at[:, 2].set(0.5)  # keep agents fed -- avoid the feral silence mask
    carries = jax.random.normal(k2, (N, HIDDEN)) * 0.1
    params = model.init(k3, carries, obs, n_layers=N_LAYERS)["params"]
    return model, params


def _apply_fn(model):
    return lambda p, c, o, nl, **kw: model.apply({"params": p}, c, o, nl, **kw)


def _run_steps(model, params, freeze_comms: bool, n_steps: int = N_STEPS):
    """Real _minibatch_step calls, real optimizer, genuinely varying data
    each step -- not a single frozen snapshot."""
    optimizer = create_optimizer(lr=3e-4, max_grad_norm=2.0)
    opt_state = optimizer.init(params)
    apply_fn = _apply_fn(model)

    for step in range(n_steps):
        base = 1000 * step
        k_obs, k_c, k_act, k_alarm, k_adv, k_ret, k_rng = jax.random.split(
            jax.random.PRNGKey(base), 7
        )
        obs = jax.random.normal(k_obs, (N, OBS_DIM)) * 0.1
        obs = obs.at[:, 2].set(0.5)
        carries = jax.random.normal(k_c, (N, HIDDEN)) * 0.1
        actions = jax.random.randint(k_act, (N,), 0, N_ACTIONS)
        alarm_actions = jax.random.randint(k_alarm, (N,), 0, 2)
        old_log_probs = jnp.zeros((N,))
        # Non-degenerate advantages: all-zero advantage makes the clipped
        # surrogate identically zero regardless of what moved, which would
        # make head_action look frozen too -- a fixture bug, not evidence.
        advantages = jax.random.normal(k_adv, (N,))
        returns = jax.random.normal(k_ret, (N,))
        old_values = jnp.zeros((N,))
        alive = jnp.ones((N,))

        params, opt_state, _metrics, _grads = _minibatch_step(
            params, opt_state, apply_fn, optimizer,
            obs, actions, alarm_actions, old_log_probs, advantages, returns, carries,
            N_LAYERS, old_values, 0.2, VF_COEF, ENT_COEF, VQ_COEF, None, alive, k_rng,
            None, 0.1, 0.0, freeze_comms,
        )
    return params


def test_comms_subtree_bitwise_identical_across_frozen_stage0_steps():
    model, params0 = _build_blue()
    params_after = _run_steps(model, params0, freeze_comms=True)

    for key in COMMS_SUBTREE_KEYS:
        assert key in params0, f"expected comms key {key!r} in blue params -- did the architecture change?"
        before_leaves = jax.tree_util.tree_leaves(params0[key])
        after_leaves = jax.tree_util.tree_leaves(params_after[key])
        for before, after in zip(before_leaves, after_leaves):
            assert bool(jnp.array_equal(before, after)), (
                f"{key} moved under freeze_comms=True after {N_STEPS} real PPO "
                f"minibatch steps -- the stage-0 freeze is not exact. Cam's "
                f"bitwise-identical requirement is violated."
            )


def test_head_action_still_moves_under_freeze_comms_negative_control():
    """Proves freeze_comms=True doesn't silently zero every gradient -- only
    the named comms subtree. If this fails, the comms-subtree test above is
    vacuous (nothing was moving anyway)."""
    model, params0 = _build_blue()
    params_after = _run_steps(model, params0, freeze_comms=True)

    before_leaves = jax.tree_util.tree_leaves(params0["head_action"])
    after_leaves = jax.tree_util.tree_leaves(params_after["head_action"])
    moved = any(
        not bool(jnp.array_equal(b, a)) for b, a in zip(before_leaves, after_leaves)
    )
    assert moved, (
        "head_action did not move under freeze_comms=True -- gradients are not "
        "flowing at all in this fixture, so the comms-freeze test above proves "
        "nothing (everything is frozen, not just comms)."
    )


def test_comms_subtree_does_move_without_freeze_negative_control():
    """Proves the test harness can detect comms-subtree movement at all: the
    same setup, same seeds, WITHOUT freeze_comms, must move at least one
    comms leaf. If it doesn't, the bitwise-identical test above could be
    passing for the wrong reason (e.g. a fixture that never trains anything)."""
    model, params0 = _build_blue()
    params_after = _run_steps(model, params0, freeze_comms=False)

    moved_any = False
    for key in COMMS_SUBTREE_KEYS:
        before_leaves = jax.tree_util.tree_leaves(params0[key])
        after_leaves = jax.tree_util.tree_leaves(params_after[key])
        if any(not bool(jnp.array_equal(b, a)) for b, a in zip(before_leaves, after_leaves)):
            moved_any = True
            break
    assert moved_any, (
        "no comms-subtree leaf moved even WITHOUT freeze_comms -- this fixture "
        "cannot detect comms-subtree movement at all, so the frozen test above "
        "is not a real test of the freeze."
    )


if __name__ == "__main__":
    test_comms_subtree_bitwise_identical_across_frozen_stage0_steps()
    test_head_action_still_moves_under_freeze_comms_negative_control()
    test_comms_subtree_does_move_without_freeze_negative_control()
    print(
        "OK (Blocker 2 verification): comms subtree (gwt_comms_1, "
        "head_signal_slot0/1/2, codebook_0/1/2, emb_nb) is bitwise identical "
        "across 5 real frozen PPO minibatch steps, head_action moves in the "
        "same run (freeze is scoped, not global), and the same subtree DOES "
        "move without freeze_comms (the harness can detect a broken freeze)."
    )
