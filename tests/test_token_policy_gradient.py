"""Channel-learnability regression (Cam's launch gate for the comm experiment, comm-testbed branch).

With the token as a policy action, a reward that depends on the LISTENER (the shared reward credited to the
sender) must reach the sender's token heads (`head_signal_slot*`) through the real ppo_loss, and push the taken
token's log-prob up. Two mutants must give exactly zero gradient there:
  mutant 1 -- today's design: no token log-prob in the ratio (token_actions=None);
  mutant 2 -- shared reward zeroed (the sender's advantage is then 0).
Every other loss that could touch the token heads is off (vq_coef=0, entropy coefs 0), so the reward path is the
only possible source. The token heads hang off gwt_comms_1, not the policy trunk, so the action loss cannot reach
them either.
"""
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from jax_sim.main_jax import _normalize_config
from jax_sim.network_jax import AgentNetworkJax, init_agent_params, make_model_apply
from jax_sim.obs_layout import make_obs_layout
from jax_sim.rl_jax import ppo_loss, token_logits

TAU = 1.0
M = 8
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
_obs = jax.random.normal(jax.random.PRNGKey(1), (M, _L.total_dim)) * 0.5
_obs = _obs.at[:, 2].set(1.0)  # energy high: keeps the feral mask (which zeroes the signal) off
_carries = jnp.zeros((M, _cfg["hidden_dim"]))
_, _outs0 = _apply(_params, _carries, _obs, _NL)
_actions = jnp.argmax(_outs0.action_logits, -1)
_tokens = jnp.stack([jnp.argmax(lg, -1) for lg in token_logits(_params, _outs0.z_e, TAU)], -1)


def _joint_logp(params, with_tokens):
    _, o = _apply(params, _carries, _obs, _NL)
    from jax_sim.action_space import mask_disabled_actions
    lp = jnp.take_along_axis(jax.nn.log_softmax(mask_disabled_actions(o.action_logits, axis=-1), -1),
                             _actions[:, None], -1)[:, 0]
    if with_tokens:
        for i, lg in enumerate(token_logits(params, o.z_e, TAU)):
            lp = lp + jnp.take_along_axis(jax.nn.log_softmax(lg, -1), _tokens[:, i:i + 1], -1)[:, 0]
    return lp


def _signal_grad(shared_reward, token_policy):
    # Sender's only reward is the shared reward for the listener's arrival; V = 0, so advantage = reward.
    adv = jnp.full((M,), shared_reward)
    old_lp = jax.lax.stop_gradient(_joint_logp(_params, token_policy))  # on-policy: ratio = 1 at the start
    def loss(p):
        l, _ = ppo_loss(p, _apply, _obs, _actions, None, old_lp, adv, jnp.zeros((M,)), _carries, _NL,
                        jnp.zeros((M,)), vf_coef=0.0, ent_coef=0.0, vq_coef=0.0,
                        token_actions=_tokens if token_policy else None, token_tau=TAU, token_ent_coef=0.0)
        return l
    g = jax.grad(loss)(_params)
    return {k: g[k] for k in SIGNAL_HEADS}


def _norm(tree):
    return float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(tree))))


def test_listener_dependent_reward_reaches_the_token_heads_and_raises_the_taken_token():
    g = _signal_grad(shared_reward=2.7, token_policy=True)
    assert _norm(g) > 1e-6, _norm(g)
    # one small descent step on the token heads only must raise the taken tokens' log-prob
    stepped = dict(_params)
    for k in SIGNAL_HEADS:
        stepped[k] = jax.tree_util.tree_map(lambda p, d: p - 1e-2 * d, _params[k], g[k])
    def tok_lp(params):
        _, o = _apply(params, _carries, _obs, _NL)
        return sum(jnp.take_along_axis(jax.nn.log_softmax(lg, -1), _tokens[:, i:i + 1], -1)[:, 0].sum()
                   for i, lg in enumerate(token_logits(params, o.z_e, TAU)))
    assert float(tok_lp(stepped)) > float(tok_lp(_params))


def test_mutant_no_token_logprob_gives_zero_gradient():
    assert _norm(_signal_grad(shared_reward=2.7, token_policy=False)) == 0.0


def test_mutant_shared_reward_zeroed_gives_zero_gradient():
    assert _norm(_signal_grad(shared_reward=0.0, token_policy=True)) == 0.0
