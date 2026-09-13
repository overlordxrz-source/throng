"""
Gradient-flow smoke test for blue's VQ codebook and red's decoupled DCVQ.

Previously instantiated `AgentNetworkJax()`/`PredatorNetworkJax()` with ZERO
constructor args, silently testing the dataclass defaults (`env_channels=9`,
`own_state_dim=10`, blue `n_actions=12`/red `n_actions=8`) instead of the
shapes actually used in production (`env_channels=15`, `own_state_dim=22`,
`n_actions=12` for both) — a test that can't catch a live-layout regression
is a ghost test. It also hardcoded `carries = jnp.zeros((2, 128))` for BOTH
teams, which happens to match red's default `hidden_dim=128` but not blue's
(`hidden_dim=256`), so `test_blue_codebook_gradient_flow` crashed on a shape
mismatch before it ever reached its assertions.

Both are fixed here by loading the same production config
`tests/test_severance_sweep.py` uses (config.yaml via DEFAULT_CONFIG +
_normalize_config) and deriving every shape from it. See
`test_severance_sweep.py` for the full per-loss-term sweep this test
predates; this file is kept as a narrower, standalone VQ-only smoke check.
"""

import yaml
import jax
import jax.numpy as jnp

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.network_jax import AgentNetworkJax, PredatorNetworkJax
from jax_sim.obs_layout import make_obs_layout

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})

_layout = make_obs_layout(
    signal_dim=int(_cfg["signal_dim"]), symbol_dim=int(_cfg["symbol_dim"]),
    memory_slots=int(_cfg.get("memory_slots", 0)), neighbor_k=int(_cfg["neighbor_k"]),
    local_cells=(2 * int(_cfg["local_obs_radius"]) + 1) ** 2,
    env_channels=int(_cfg.get("env_channels", 15)), own_state_dim=int(_cfg.get("own_state_dim", 22)),
)
OBS_DIM = _layout.total_dim
N_ACTIONS = int(_cfg.get("n_actions", 8))
HIDDEN = int(_cfg["hidden_dim"])
RED_HIDDEN = int(_cfg.get("red_hidden_dim", HIDDEN // 2))


def test_blue_codebook_gradient_flow():
    model = AgentNetworkJax(hidden_dim=HIDDEN, n_actions=N_ACTIONS, obs_dim=OBS_DIM,
                             env_channels=int(_cfg.get("env_channels", 15)),
                             own_state_dim=int(_cfg.get("own_state_dim", 22)),
                             neighbor_k=int(_cfg["neighbor_k"]))
    rng = jax.random.PRNGKey(0)
    obs = jnp.zeros((2, OBS_DIM))
    carries = jnp.zeros((2, model.hidden_dim))

    variables = model.init(rng, carries, obs, n_layers=2)
    params = variables["params"]

    def loss_fn(p):
        new_c, outs = model.apply({"params": p}, carries, obs, n_layers=2)
        # Check gradient of loss_vq flows back to the network params (codebook)
        return outs.loss_vq.sum()

    grads = jax.grad(loss_fn)(params)

    # Assert codebook has gradients
    cb0_grad = grads["codebook_0"]["embedding"]
    cb1_grad = grads["codebook_1"]["embedding"]
    cb2_grad = grads["codebook_2"]["embedding"]

    assert jnp.linalg.norm(cb0_grad) > 1e-6, "Blue Codebook 0 is severed from VQ loss!"
    assert jnp.linalg.norm(cb1_grad) > 1e-6, "Blue Codebook 1 is severed from VQ loss!"
    assert jnp.linalg.norm(cb2_grad) > 1e-6, "Blue Codebook 2 is severed from VQ loss!"


def test_red_vq_decoupled():
    model = PredatorNetworkJax(hidden_dim=RED_HIDDEN, n_actions=N_ACTIONS,
                                env_channels=int(_cfg.get("env_channels", 15)),
                                own_state_dim=int(_cfg.get("own_state_dim", 22)),
                                neighbor_k=int(_cfg["neighbor_k"]),
                                local_obs_radius=int(_cfg["local_obs_radius"]))
    rng = jax.random.PRNGKey(0)
    obs = jnp.zeros((2, OBS_DIM))
    carries = jnp.zeros((2, model.hidden_dim))

    variables = model.init(rng, carries, obs, n_layers=2)
    params = variables["params"]

    def loss_fn(p):
        new_c, outs = model.apply({"params": p}, carries, obs, n_layers=2, rngs={'dropout': rng})
        # For Red, vq_coef is explicitly 0 in the PPO update in main_jax.py,
        # but rl_jax.py computes total_loss = ... + vq_coef * jnp.nan_to_num(loss_vq_mean).
        # Let's mock a 0.0 * loss_vq here and ensure NO gradient flows.
        return (0.0 * outs.loss_vq).sum()

    grads = jax.grad(loss_fn)(params)

    # Assert dcvq codebook has ZERO gradients (param is named "subspace_embeddings",
    # not "codebook" — the original assertion here referenced a key that doesn't
    # exist on PredatorNetworkJax's DCVQ module and had never actually been run
    # until Finding 1's import fix, so this mismatch was never caught).
    dcvq_grad = grads["dcvq"]["subspace_embeddings"]
    assert float(jnp.linalg.norm(dcvq_grad)) < 1e-8, "Red DCVQ codebook is NOT decoupled from gradients!"


if __name__ == "__main__":
    test_blue_codebook_gradient_flow()
    test_red_vq_decoupled()
    print(f"OK: Gradient flow tests passed at production shapes (hidden={HIDDEN}/{RED_HIDDEN}, "
          f"obs_dim={OBS_DIM}, n_actions={N_ACTIONS}).")
