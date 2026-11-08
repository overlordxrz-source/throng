"""Positive signalling on the real network (comm-jit branch; launch gate for the PS confirmation).

1. Off means off: ppo_loss with ps_cols=None is bitwise the loss without the PS arguments.
2. Role mask: the PS value depends only on INFORMED rows -- randomising the uninformed rows' obs leaves it unchanged.
   Mutant (mask removed) must change it.
3. Ascent: one normalised gradient step on PS raises it.
4. Reach: the PS gradient reaches the message heads (head_signal_slot*), and the PS term moves the total loss's
   gradient there when ps_coef > 0.
"""
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from jax_sim import rl_jax
from jax_sim.main_jax import _normalize_config
from jax_sim.network_jax import AgentNetworkJax, init_agent_params, make_model_apply
from jax_sim.obs_layout import make_obs_layout
from jax_sim.rl_jax import positive_signalling_mi, ppo_loss, token_logits

TAU = 0.08
M = 64
SIGNAL_HEADS = ("head_signal_slot0", "head_signal_slot1", "head_signal_slot2")

_cfg = _normalize_config(yaml.safe_load(open("config.yaml")))
_L = make_obs_layout(signal_dim=_cfg["signal_dim"], symbol_dim=_cfg["symbol_dim"],
                     memory_slots=_cfg.get("memory_slots", 0), neighbor_k=_cfg["neighbor_k"],
                     local_cells=(2 * _cfg["local_obs_radius"] + 1) ** 2, env_channels=int(_cfg.get("env_channels", 15)),
                     own_state_dim=int(_cfg.get("own_state_dim", 29)))
_model = AgentNetworkJax(
    hidden_dim=_cfg["hidden_dim"], n_heads=_cfg["n_heads"], n_layers=_cfg["n_layers"], obs_dim=_L.total_dim,
    signal_dim=_cfg["signal_dim"], symbol_dim=_cfg["symbol_dim"], vocab_size=_cfg["vocab_size"],
    memory_slots=_cfg.get("memory_slots", 0), fwd_env_dim=_L.loc_env_end - _L.loc_env_start,
    env_channels=int(_cfg.get("env_channels", 15)), own_state_dim=_L.own_state_dim, n_actions=int(_cfg["n_actions"]),
    local_cells=_L.local_cells, neighbor_k=_cfg["neighbor_k"])
_apply = make_model_apply(_model)
_NL = int(_cfg["n_layers"])
_params = init_agent_params(_model, jax.random.PRNGKey(0), jnp.zeros((1, _cfg["hidden_dim"])),
                            jnp.zeros((1, _L.total_dim)), _NL)
S0 = _L.own_state_start
COLS = (S0 + 12, S0 + 21)
_rng = np.random.default_rng(0)
_obs = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (M, _L.total_dim))) * 0.5
_obs[:, 2] = 1.0
_inf = np.tile([True, False], M // 2)
_lab = _rng.integers(0, 4, M)
_obs[:, S0 + 12] = _inf
_obs[:, S0 + 21:S0 + 25] = np.eye(4)[_lab] * _inf[:, None]
_obs = jnp.asarray(_obs, jnp.float32)
_carries = jnp.zeros((M, _cfg["hidden_dim"]))


def _ps(params, obs, cols=COLS):
    _, o = _apply(params, _carries, obs, _NL)
    return positive_signalling_mi(params, o.z_e, obs, TAU, cols)


def _loss_args():
    _, o = _apply(_params, _carries, _obs, _NL)
    acts = jnp.argmax(o.action_logits, -1)
    toks = jnp.stack([jnp.argmax(lg, -1) for lg in token_logits(_params, o.z_e, TAU)], -1)
    return dict(obs=_obs, actions=acts, alarm_actions=None, old_log_probs=jnp.zeros(M), advantages=jnp.ones(M),
                returns=jnp.zeros(M), carries=_carries, n_layers=_NL, old_values=jnp.zeros(M), token_actions=toks,
                token_tau=TAU)


def test_off_is_bitwise_unchanged():
    kw = _loss_args()
    l0, _ = ppo_loss(_params, _apply, **kw)
    l1, m1 = ppo_loss(_params, _apply, **kw, ps_coef=5.0, ps_cols=None)
    assert np.array_equal(np.asarray(l0), np.asarray(l1))
    assert float(m1["ppo_ps_mi"]) == 0.0


def test_role_mask_and_mutant():
    base = float(_ps(_params, _obs))
    noisy = np.asarray(_obs).copy()
    noisy[~_inf] = np.asarray(jax.random.normal(jax.random.PRNGKey(9), noisy[~_inf].shape))
    noisy[~_inf, S0 + 12] = 0.0
    noisy[~_inf, S0 + 21:S0 + 25] = 0.0
    assert abs(float(_ps(_params, jnp.asarray(noisy))) - base) < 1e-6        # uninformed rows don't enter
    # mutant: mask removed (every row treated as informed) -> must react to the uninformed rows
    mut = lambda o: _ps(_params, o.at[:, S0 + 12].set(1.0))  # noqa: E731
    assert abs(float(mut(jnp.asarray(noisy))) - float(mut(_obs))) > 1e-6


def test_ascent_raises_and_reaches_message_heads():
    g = jax.grad(lambda p: _ps(p, _obs))(_params)
    gp = g["params"] if "params" in g else g
    for h in SIGNAL_HEADS:
        assert float(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(gp[h]))) > 0, h
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    p2 = jax.tree_util.tree_map(lambda x, d: x + 1e-2 * d / (gn + 1e-12), _params, g)
    assert float(_ps(p2, _obs)) > float(_ps(_params, _obs))


def test_ps_term_moves_total_gradient_at_heads():
    kw = _loss_args()
    g0 = jax.grad(lambda p: ppo_loss(p, _apply, **kw)[0])(_params)
    g1 = jax.grad(lambda p: ppo_loss(p, _apply, **kw, ps_coef=1.0, ps_cols=COLS)[0])(_params)
    pk = lambda g: (g["params"] if "params" in g else g)["head_signal_slot0"]["kernel"]  # noqa: E731
    a, b = pk(g0), pk(g1)
    assert float(jnp.abs(a - b).max()) > 0
