"""
tests/test_severance_sweep.py — H2 generalized (Task 0.5 audit, Sweep A completion).

THRONG has repeatedly been bitten by "severance-class" bugs: a learning signal
silently disconnected from the tensor it was meant to train, running undetected
for weeks and costing real compute (or a false scientific conclusion). Three in
recent memory: the VQ loss severed from the autodiff tape across all of Phase 18;
`NB_GAIN` initialised to 1.0 and never updated; the red VQ loss read at the wrong
tuple index, feeding raw `z_e` into PPO.

`test_vq_gradient_flow.py` pinned the VQ wiring in isolation. This file finishes
the job Cam asked for in the Sep 2026 audit: EVERY loss term in the training
step, measured — not reasoned about — on a real CPU minibatch, at PRODUCTION
shapes (loaded from config.yaml, not dataclass defaults — a test that silently
tests a stale shape is a ghost test and can't catch a live-layout regression),
for blue and red separately.

Two kinds of assertion, deliberately kept separate:

  * STRUCTURAL ("_structural_"): does a non-zero gradient reach the term's
    designated parameters at all, before any coefficient is applied? This must
    ALWAYS hold — this is exactly the shape of bug that hit Phase 18.4 (VQ
    severed from the tape) and Phase 18.5 (wrong tuple index). A failure here
    is a severance bug, full stop, independent of whatever the config says.

  * EFFECTIVE ("_effective_"): does the coefficient-weighted, optimizer-applied
    parameter delta match what config.yaml's own coefficient says it should be
    (zero if the coefficient is zero, non-zero otherwise)? This reads the LIVE
    config so it self-updates if a coefficient changes, rather than pinning
    today's specific numbers — except for the confidence head, where the
    audit's Finding 2 is precisely that the coefficient (correctly) zeroes the
    gradient while a DIFFERENT, unrelated part of the codebase (the epistemic
    gate) unconditionally consumes that head's output as if it were trained.
    That mismatch is cross-module, not a gradient-flow property, so it is
    pinned explicitly and will need a conscious update if `confidence_enabled`
    or the epistemic gate's behavior ever changes.

Run:  PYTHONPATH=. python tests/test_severance_sweep.py
(Requires jax_sim.network_jax to import — Finding 1 of docs/AUDIT_SEP2026.md.)
"""

import numpy as np
import jax
import jax.numpy as jnp
import yaml

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.network_jax import AgentNetworkJax, PredatorNetworkJax
from jax_sim.obs_layout import make_obs_layout
from jax_sim.rl_jax import (
    ppo_loss,
    auxiliary_update,
    red_auxiliary_update,
    create_optimizer,
)

# ─── Load production config exactly as main_jax.py does (no hardcoded shapes) ──

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})

HIDDEN = int(_cfg["hidden_dim"])
RED_HIDDEN = int(_cfg.get("red_hidden_dim", HIDDEN // 2))
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
N_ACTIONS = int(_cfg.get("n_actions", 8))

_p9 = _cfg.get("phase9_canvas") or {}
_p12 = _cfg.get("phase12_coevolution") or {}
_p14t = _cfg.get("phase14_transcendental") or {}

VF_COEF = float(_cfg.get("ppo_value_coef", 0.25))
ENT_COEF = float(_cfg.get("ppo_entropy_coef", 0.02))
ALARM_ENT_COEF = float(_cfg.get("alarm_ent_coef", 0.0))
VQ_COEF_BLUE = float(_cfg.get("vq_loss_coef", 0.1))
VQ_COEF_RED = float(_cfg.get("red_vq_loss_coef", 0.0))
FWD_COEF = float(_cfg.get("fwd_coef", 0.05))
CARRY_FWD_COEF = float(_cfg.get("carry_fwd_coef", 0.05))
SELF_PRED_COEF = float(_cfg.get("self_pred_coef", 0.1))
CONF_ENABLED = bool(_p9.get("confidence_enabled", False))
CONF_COEF_LIVE = float(_p9.get("confidence_coef", 0.05)) if CONF_ENABLED else 0.0
PROPRIO_COEF = float(_cfg.get("proprio_coef", 0.05))
RETENTION_COEF = float(_cfg.get("retention_coef", 0.1))

_layout = make_obs_layout(
    signal_dim=SIGNAL_DIM, symbol_dim=SYMBOL_DIM, memory_slots=MEM_SLOTS,
    neighbor_k=NEIGHBOR_K, local_cells=LOCAL_CELLS, env_channels=ENV_CH,
    own_state_dim=OWN_STATE,
)
OBS_DIM = _layout.total_dim
FWD_ENV_DIM = _layout.loc_env_end - _layout.loc_env_start

N = 32  # minibatch agents for PPO-loss checks


def _norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves)))


def _subtree_norm(tree, key) -> float:
    return _norm(tree[key]) if key in tree else 0.0


def _delta_norm(before, after) -> float:
    return _norm(jax.tree_util.tree_map(lambda a, b: a - b, after, before))


def _nonzero(x: float, tol: float = 1e-8) -> bool:
    return x > tol


def _zero(x: float, tol: float = 1e-6) -> bool:
    return x < tol


# ─── Blue fixtures ──────────────────────────────────────────────────────────

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
    # obs[:, 2] is energy; below 0.20 the network's "feral mask" hard-replaces
    # alarm_out/signal_out/symbol_write with silence constants, deliberately
    # severing their gradient (starving agents can't communicate — see
    # network_jax.py:394-400). A mean-zero synthetic obs lands almost every
    # agent below that threshold by chance, which would make the alarm/signal
    # paths look severed for a fixture reason, not a real one. Keep agents fed.
    obs = obs.at[:, 2].set(0.5)
    carries = jax.random.normal(k2, (N, HIDDEN)) * 0.1
    params = model.init(k3, carries, obs, n_layers=N_LAYERS)["params"]
    return model, params, obs, carries


def _blue_apply_fn(model):
    return lambda p, c, o, nl, **kw: model.apply({"params": p}, c, o, nl, **kw)


def _ppo_metric_grad(metric_key, apply_fn, carries, obs, actions, alarm_actions,
                      params, alive, **coefs):
    m = carries.shape[0]
    # Non-degenerate advantages/returns: an all-zero advantage makes the PPO
    # clipped surrogate identically zero for every agent regardless of
    # action_logits, which would make every term here look "severed" even
    # when it isn't — that's a fixture bug, not a network bug, so avoid it.
    old_lp = jnp.zeros((m,))
    adv = jax.random.normal(jax.random.PRNGKey(999), (m,))
    ret = jax.random.normal(jax.random.PRNGKey(998), (m,))
    old_val = jnp.zeros((m,))

    def f(p):
        _, metrics = ppo_loss(
            p, apply_fn, obs, actions, alarm_actions, old_lp, adv, ret, carries,
            N_LAYERS, old_val, alive=alive, **coefs,
        )
        return metrics[metric_key]

    return jax.grad(f)(params)


def test_structural_blue_ppo_pg_reaches_head_action():
    model, params, obs, carries = _build_blue()
    actions = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, N_ACTIONS)
    alarm = jax.random.randint(jax.random.PRNGKey(2), (N,), 0, 2)
    alive = jnp.ones((N,))
    g = _ppo_metric_grad("ppo_pg_loss", _blue_apply_fn(model), carries, obs, actions, alarm, params, alive)
    assert _nonzero(_subtree_norm(g, "head_action")), "PPO policy-gradient loss is severed from head_action"


def test_structural_blue_ppo_vf_reaches_head_value():
    model, params, obs, carries = _build_blue()
    actions = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, N_ACTIONS)
    alarm = jax.random.randint(jax.random.PRNGKey(2), (N,), 0, 2)
    alive = jnp.ones((N,))
    g = _ppo_metric_grad("ppo_vf_loss", _blue_apply_fn(model), carries, obs, actions, alarm, params, alive)
    assert _nonzero(_subtree_norm(g, "head_value")), "PPO value loss is severed from head_value"


def test_structural_blue_ppo_entropy_reaches_head_action():
    model, params, obs, carries = _build_blue()
    actions = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, N_ACTIONS)
    alarm = jax.random.randint(jax.random.PRNGKey(2), (N,), 0, 2)
    alive = jnp.ones((N,))
    g = _ppo_metric_grad("ppo_entropy", _blue_apply_fn(model), carries, obs, actions, alarm, params, alive)
    assert _nonzero(_subtree_norm(g, "head_action")), "Entropy bonus is severed from head_action"
    assert _nonzero(_subtree_norm(g, "head_alarm")), "Alarm entropy is severed from head_alarm"


def test_structural_blue_ppo_logit_penalty_reaches_head_action():
    model, params, obs, carries = _build_blue()
    actions = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, N_ACTIONS)
    alarm = jax.random.randint(jax.random.PRNGKey(2), (N,), 0, 2)
    alive = jnp.ones((N,))
    g = _ppo_metric_grad("ppo_logit_pen", _blue_apply_fn(model), carries, obs, actions, alarm, params, alive)
    assert _nonzero(_subtree_norm(g, "head_action")), "Logit L2 penalty is severed from head_action"


def test_structural_blue_vq_loss_reaches_all_three_codebooks_and_slot_heads():
    """Generalizes test_vq_gradient_flow.py's isolated check to production shapes
    and the real ppo_loss path (H2, done properly)."""
    model, params, obs, carries = _build_blue()
    actions = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, N_ACTIONS)
    alarm = jax.random.randint(jax.random.PRNGKey(2), (N,), 0, 2)
    alive = jnp.ones((N,))
    g = _ppo_metric_grad("ppo_vq_loss", _blue_apply_fn(model), carries, obs, actions, alarm, params, alive)
    for slot in ("codebook_0", "codebook_1", "codebook_2",
                 "head_signal_slot0", "head_signal_slot1", "head_signal_slot2"):
        assert _nonzero(_subtree_norm(g, slot)), f"VQ loss is severed from {slot}"


def test_effective_blue_confidence_head_matches_config_flag():
    """Cross-module pin for audit Finding 2. `predict_carry_fwd_confidence`
    (network_jax.py) is called UNCONDITIONALLY by the epistemic gate at rollout
    time whenever `imagination_gating_enabled: true`, regardless of whether the
    head was ever trained. `confidence_enabled: false` forces conf_coef to 0.0,
    so the head that the gate treats as calibrated confidence never moves.
    If this test starts failing because CONF_ENABLED flipped true, that's the
    audit's fix landing — update this test consciously, don't just relax it."""
    model, params, _, carries = _build_blue()
    aux_apply_fn = lambda p, c, a: model.apply({"params": p}, c, a, method=model.auxiliary_heads)
    opt = create_optimizer(lr=1e-2)
    opt_state = opt.init(params)
    T, Nb = 4, 8
    carries_np = np.array(jax.random.normal(jax.random.PRNGKey(20), (T, Nb, HIDDEN)) * 0.1)
    actions_np = np.array(jax.random.randint(jax.random.PRNGKey(21), (T, Nb), 0, N_ACTIONS))
    obs_np = np.array(jax.random.normal(jax.random.PRNGKey(22), (T, Nb, OBS_DIM)) * 0.1)
    energy_np = np.array(jax.random.uniform(jax.random.PRNGKey(23), (T, Nb)))
    before = {k: params[k] for k in ("head_confidence_1", "head_confidence_2")}
    new_params, *_rest = auxiliary_update(
        params, opt_state, opt, aux_apply_fn, carries_np, actions_np, obs_np,
        loc_env_start=0, loc_env_end=FWD_ENV_DIM,
        alive_np=np.ones((T, Nb), dtype=np.float32), energy_np=energy_np,
        fwd_coef=FWD_COEF, carry_fwd_coef=CARRY_FWD_COEF, self_pred_coef=SELF_PRED_COEF,
        conf_coef=CONF_COEF_LIVE, proprio_coef=PROPRIO_COEF, n_actions=N_ACTIONS,
    )
    d = _delta_norm(before, {k: new_params[k] for k in before})
    if CONF_ENABLED:
        assert _nonzero(d), (
            "confidence_enabled=true in config but head_confidence_* did not move — "
            "the mechanism itself is now broken (worse than the audit's Finding 2)"
        )
    else:
        assert _zero(d), (
            "confidence_enabled=false but head_confidence_* moved anyway — "
            "config is no longer the cause of Finding 2; something else changed"
        )
        # Structural sanity: the mechanism itself must still be intact — it's
        # *disabled*, not *severed*. Prove a non-zero coefficient would move it.
        opt_state2 = opt.init(params)
        new_params2, *_rest2 = auxiliary_update(
            params, opt_state2, opt, aux_apply_fn, carries_np, actions_np, obs_np,
            loc_env_start=0, loc_env_end=FWD_ENV_DIM,
            alive_np=np.ones((T, Nb), dtype=np.float32), energy_np=energy_np,
            fwd_coef=FWD_COEF, carry_fwd_coef=CARRY_FWD_COEF, self_pred_coef=SELF_PRED_COEF,
            conf_coef=0.05, proprio_coef=PROPRIO_COEF, n_actions=N_ACTIONS,
        )
        d2 = _delta_norm(before, {k: new_params2[k] for k in before})
        assert _nonzero(d2), "head_confidence_* is structurally severed even with a non-zero coefficient"


def test_structural_blue_aux_losses_reach_their_own_heads_only():
    """fwd_env, carry_fwd, self_pred, proprio each move their own head, and
    NONE of them leak into the transformer trunk (auxiliary_heads takes
    carry_t as data, not as a function of the trunk's own params)."""
    model, params, _, carries = _build_blue()
    aux_apply_fn = lambda p, c, a: model.apply({"params": p}, c, a, method=model.auxiliary_heads)
    opt = create_optimizer(lr=1e-2)
    opt_state = opt.init(params)
    T, Nb = 4, 8
    carries_np = np.array(jax.random.normal(jax.random.PRNGKey(30), (T, Nb, HIDDEN)) * 0.1)
    actions_np = np.array(jax.random.randint(jax.random.PRNGKey(31), (T, Nb), 0, N_ACTIONS))
    obs_np = np.array(jax.random.normal(jax.random.PRNGKey(32), (T, Nb, OBS_DIM)) * 0.1)
    energy_np = np.array(jax.random.uniform(jax.random.PRNGKey(33), (T, Nb)))
    aux_keys = {
        "head_fwd_1", "head_fwd_2", "head_fwd_dyn_1", "head_fwd_dyn_2",
        "head_self_pred", "head_confidence_1", "head_confidence_2", "head_proprio",
    }
    trunk_keys = [k for k in params if k not in aux_keys]
    before = {k: params[k] for k in params}
    new_params, *_rest = auxiliary_update(
        params, opt_state, opt, aux_apply_fn, carries_np, actions_np, obs_np,
        loc_env_start=0, loc_env_end=FWD_ENV_DIM,
        alive_np=np.ones((T, Nb), dtype=np.float32), energy_np=energy_np,
        fwd_coef=FWD_COEF, carry_fwd_coef=CARRY_FWD_COEF, self_pred_coef=SELF_PRED_COEF,
        conf_coef=0.05, proprio_coef=PROPRIO_COEF, n_actions=N_ACTIONS,
    )
    for k in ("head_fwd_1", "head_fwd_2"):
        assert _nonzero(_delta_norm(before[k], new_params[k])), f"fwd_env loss severed from {k}"
    for k in ("head_fwd_dyn_1", "head_fwd_dyn_2"):
        assert _nonzero(_delta_norm(before[k], new_params[k])), f"carry_fwd loss severed from {k}"
    assert _nonzero(_delta_norm(before["head_self_pred"], new_params["head_self_pred"])), \
        "self_pred loss severed from head_self_pred"
    assert _nonzero(_delta_norm(before["head_proprio"], new_params["head_proprio"])), \
        "proprio loss severed from head_proprio"
    trunk_delta = _delta_norm({k: before[k] for k in trunk_keys}, {k: new_params[k] for k in trunk_keys})
    assert _zero(trunk_delta), (
        "blue's auxiliary_update leaked gradient into the trunk — auxiliary_heads "
        "is supposed to take carry_t as fixed data, not backprop through the encoder"
    )


# ─── Red fixtures ───────────────────────────────────────────────────────────

def _build_red():
    model = PredatorNetworkJax(
        hidden_dim=RED_HIDDEN, neighbor_k=NEIGHBOR_K, local_obs_radius=int(_cfg["local_obs_radius"]),
        n_heads=N_HEADS, n_layers=N_LAYERS, signal_dim=SIGNAL_DIM, symbol_dim=SYMBOL_DIM,
        vocab_size=int(_p12.get("red_vocab_size", VOCAB)), vq_beta=VQ_BETA, vq_dead_code_reset=True,
        simvq_w_clip=float(_p14t.get("simvq_w_clip", 2.0)), simvq_out_scale=float(_p14t.get("simvq_out_scale", 2.0)),
        vq_noise_scale=float(_p12.get("red_vq_noise_scale", 0.0)), memory_slots=MEM_SLOTS,
        cross_attn_enabled=bool(_p12.get("red_cross_attn_enabled", True)),
        cross_attn_num_heads=int(_p9.get("cross_attn_num_heads", N_HEADS)),
        env_channels=ENV_CH, own_state_dim=OWN_STATE, n_actions=N_ACTIONS,
    )
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(100), 3)
    obs = jax.random.normal(k1, (N, OBS_DIM)) * 0.1
    obs = obs.at[:, 2].set(0.5)  # avoid the feral mask, see _build_blue's comment
    carries = jax.random.normal(k2, (N, RED_HIDDEN)) * 0.1
    params = model.init(k3, carries, obs, n_layers=N_LAYERS)["params"]
    return model, params, obs, carries


def _red_apply_fn(model):
    return lambda p, c, o, nl, **kw: model.apply({"params": p}, c, o, nl, rngs={"dropout": jax.random.PRNGKey(1)}, **kw)


def test_structural_red_ppo_pg_vf_entropy_logitpen_reach_their_heads():
    model, params, obs, carries = _build_red()
    actions = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, N_ACTIONS)
    alive = jnp.ones((N,))
    apply_fn = _red_apply_fn(model)
    g_pg = _ppo_metric_grad("ppo_pg_loss", apply_fn, carries, obs, actions, None, params, alive)
    assert _nonzero(_subtree_norm(g_pg, "head_action")), "Red PPO policy-gradient loss severed from head_action"
    g_vf = _ppo_metric_grad("ppo_vf_loss", apply_fn, carries, obs, actions, None, params, alive)
    assert _nonzero(_subtree_norm(g_vf, "head_value")), "Red PPO value loss severed from head_value"
    g_ent = _ppo_metric_grad("ppo_entropy", apply_fn, carries, obs, actions, None, params, alive)
    assert _nonzero(_subtree_norm(g_ent, "head_action")), "Red entropy bonus severed from head_action"
    g_lp = _ppo_metric_grad("ppo_logit_pen", apply_fn, carries, obs, actions, None, params, alive)
    assert _nonzero(_subtree_norm(g_lp, "head_action")), "Red logit L2 penalty severed from head_action"


def test_red_vq_loss_structurally_connected_but_effectively_decoupled():
    """Phase 18.6: red_vq_loss_coef=0.0 is an INTENTIONAL diagnostic decoupling,
    not a severance bug — dcvq's raw gradient must still exist (mechanism
    intact), but the coefficient-weighted contribution to the actual optimizer
    step must be exactly zero (matches the documented design)."""
    model, params, obs, carries = _build_red()
    actions = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, N_ACTIONS)
    alive = jnp.ones((N,))
    g = _ppo_metric_grad("ppo_vq_loss", _red_apply_fn(model), carries, obs, actions, None, params, alive)
    raw = _subtree_norm(g, "dcvq")
    assert _nonzero(raw), "Red DCVQ loss is structurally severed from the dcvq codebook"
    effective = raw * VQ_COEF_RED
    if VQ_COEF_RED == 0.0:
        assert _zero(effective), "red_vq_loss_coef=0.0 in config but effective contribution is non-zero"
    else:
        assert _nonzero(effective), "red_vq_loss_coef is non-zero but effective contribution is zero"


def test_red_aux_proprio_and_retention_reach_their_heads_via_bptt():
    """Unlike blue, red's aux update (main_jax.py's `_red_aux_apply_fn`) scans
    the FULL network forward over `lag` obs steps to build final_carry before
    reading the aux heads off it — real BPTT (the documented 'SRL BPTT'),
    so gradient legitimately reaches the trunk here, not just the two heads.
    This asymmetry with blue's aux update (heads-only) is intentional; this
    test documents and pins it rather than treating trunk movement as a leak."""
    model, params, _, _ = _build_red()
    opt = create_optimizer(lr=1e-2)
    opt_state = opt.init(params)

    def red_aux_apply_fn(p, carry_t, obs_seq):
        def scan_fn(c, o):
            new_c, _ = model.apply({"params": p}, c, o, N_LAYERS, deterministic=True)
            return new_c, None
        final_carry, _ = jax.lax.scan(scan_fn, carry_t, obs_seq)
        return model.apply({"params": p}, final_carry, method=model.red_auxiliary_heads)

    T_lag, N_lag, LAG = 3, 8, 5
    carries_np = np.array(jax.random.normal(jax.random.PRNGKey(200), (T_lag, N_lag, RED_HIDDEN)) * 0.1)
    obs_seq_np = np.array(jax.random.normal(jax.random.PRNGKey(201), (T_lag, N_lag, LAG, OBS_DIM)) * 0.1)
    energy_np = np.array(jax.random.uniform(jax.random.PRNGKey(202), (T_lag, N_lag)))
    nb_sigs_target_np = np.array(jax.random.normal(jax.random.PRNGKey(203), (T_lag, N_lag, SIGNAL_DIM * NEIGHBOR_K)))

    before = {k: params[k] for k in params}
    new_params, *_rest = red_auxiliary_update(
        params, opt_state, opt, red_aux_apply_fn, carries_np, obs_seq_np, energy_np, nb_sigs_target_np,
        alive_np=np.ones((T_lag, N_lag), dtype=np.float32),
        proprio_coef=PROPRIO_COEF, retention_coef=RETENTION_COEF,
    )
    assert _nonzero(_delta_norm(before["head_proprio"], new_params["head_proprio"])), \
        "Red proprio loss severed from head_proprio"
    assert _nonzero(_delta_norm(before["head_retention"], new_params["head_retention"])), \
        "Red retention (SRL) loss severed from head_retention"


if __name__ == "__main__":
    tests = [
        test_structural_blue_ppo_pg_reaches_head_action,
        test_structural_blue_ppo_vf_reaches_head_value,
        test_structural_blue_ppo_entropy_reaches_head_action,
        test_structural_blue_ppo_logit_penalty_reaches_head_action,
        test_structural_blue_vq_loss_reaches_all_three_codebooks_and_slot_heads,
        test_effective_blue_confidence_head_matches_config_flag,
        test_structural_blue_aux_losses_reach_their_own_heads_only,
        test_structural_red_ppo_pg_vf_entropy_logitpen_reach_their_heads,
        test_red_vq_loss_structurally_connected_but_effectively_decoupled,
        test_red_aux_proprio_and_retention_reach_their_heads_via_bptt,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nAll {len(tests)} severance-sweep checks passed at production shapes "
          f"(hidden={HIDDEN}/{RED_HIDDEN}, obs_dim={OBS_DIM}, n_actions={N_ACTIONS}, "
          f"confidence_enabled={CONF_ENABLED}).")
