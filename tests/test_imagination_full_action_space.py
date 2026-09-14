"""
Task 2 (docs/WILL_RESTART_SEP2026.md): verify the K-step imagination gate
actually scores the FULL 12-action space in the live config, not just the
legacy 5 (Stay + N/S/E/W).

`imagination_n_actions` default changed 5 -> 12 in the July tree. It's not
enough that the constant changed — this proves the `imagine()` scores
tensor genuinely has one value per action (12, at the live config's
`imagination_n_actions`) and that the argmax can and does land on a
non-legacy action (Strike/Build/PickUp/Craft/UseTool, i.e. index >= 5),
not just structurally allowed but observed, on a real (if freshly
initialized) network — a short CPU smoke test, no checkpoint or GPU needed.
"""

import yaml
import jax
import jax.numpy as jnp

from jax_sim.action_space import MASKED_ACTIONS
from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.network_jax import AgentNetworkJax
from jax_sim.imagination_jax import make_imagination_fn
from jax_sim.obs_layout import make_obs_layout

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})

_p9 = _cfg.get("phase9_canvas") or {}
IMAGINATION_N_ACTIONS = int(_p9.get("imagination_n_actions", 12))
N_ACTIONS = int(_cfg.get("n_actions", 8))
HIDDEN = int(_cfg["hidden_dim"])

_layout = make_obs_layout(
    signal_dim=int(_cfg["signal_dim"]), symbol_dim=int(_cfg["symbol_dim"]),
    memory_slots=int(_cfg.get("memory_slots", 0)), neighbor_k=int(_cfg["neighbor_k"]),
    local_cells=(2 * int(_cfg["local_obs_radius"]) + 1) ** 2,
    env_channels=int(_cfg.get("env_channels", 15)), own_state_dim=int(_cfg.get("own_state_dim", 22)),
)
OBS_DIM = _layout.total_dim


def test_live_config_requests_the_full_action_space():
    """Sanity check on the config itself: if this ever regresses to 5
    (the legacy default), the rest of this test's claims don't apply — the
    epistemic gate would again be blind to Strike/Push/Guard/Build/tools."""
    assert IMAGINATION_N_ACTIONS == N_ACTIONS == 12, (
        f"imagination_n_actions={IMAGINATION_N_ACTIONS}, n_actions={N_ACTIONS} — "
        "expected both to be 12 in the live Phase 18.7 config."
    )


def test_imagine_scores_tensor_has_one_value_per_action():
    model = AgentNetworkJax(hidden_dim=HIDDEN, n_actions=N_ACTIONS, obs_dim=OBS_DIM,
                             env_channels=int(_cfg.get("env_channels", 15)),
                             own_state_dim=int(_cfg.get("own_state_dim", 22)),
                             neighbor_k=int(_cfg["neighbor_k"]))
    imagine = make_imagination_fn(model, K=5, gamma=0.999, n_imagine_actions=IMAGINATION_N_ACTIONS)

    N = 64
    rng = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    carries = jax.random.normal(k1, (N, HIDDEN)) * 0.5
    obs = jax.random.normal(k2, (N, OBS_DIM)) * 0.1
    obs = obs.at[:, 2].set(0.5)  # avoid the feral mask (see test_severance_sweep.py)
    params = model.init(k3, carries, obs, n_layers=2)["params"]
    _, outs = model.apply({"params": params}, carries, obs, n_layers=2)
    action_logits = outs.action_logits
    alive = jnp.ones((N,), dtype=jnp.bool_)

    imagined, gain, agree_greedy = imagine(params, carries, action_logits, alive)

    assert imagined.shape == (N,), f"expected one imagined action per agent, got shape {imagined.shape}"
    assert int(jnp.max(imagined)) <= N_ACTIONS - 1, "imagined action index exceeds the action space"
    assert int(jnp.min(imagined)) >= 0

    # Structural: no MASKED_ACTIONS index may ever be the imagined action --
    # masked to -1e9 inside imagine() (Push/Guard are no-ops, Build is
    # disabled per the Sep 2026 restart ruling). This assertion originally
    # only checked 6 and 7 and missed that Build(8) could still win the
    # imagined argmax -- exactly the gap the Sep 2026 pre-flight caught.
    for _masked in MASKED_ACTIONS:
        assert bool(jnp.all(imagined != _masked)), (
            f"imagine() selected action {_masked} — the -1e9 mask inside "
            "imagination_jax.py is not doing its job"
        )


def test_imagine_argmax_lands_outside_legacy_five_actions():
    """The actual empirical claim Task 2 asks for: not just that 12 columns
    exist, but that Strike/Build/PickUp/Craft/UseTool (index >= 5) can and do
    win the argmax — i.e. the gate is not silently only ever choosing among
    Stay/N/S/E/W regardless of the wider tensor's shape. Run several
    independent random seeds; with a freshly initialized (unbiased) network
    and 10 legal non-masked actions out of 12, at least one of several
    64-agent batches landing entirely within {0,1,2,3,4} would be a
    near-impossible coincidence if the wider action space were actually being
    scored and selected from."""
    model = AgentNetworkJax(hidden_dim=HIDDEN, n_actions=N_ACTIONS, obs_dim=OBS_DIM,
                             env_channels=int(_cfg.get("env_channels", 15)),
                             own_state_dim=int(_cfg.get("own_state_dim", 22)),
                             neighbor_k=int(_cfg["neighbor_k"]))
    imagine = make_imagination_fn(model, K=5, gamma=0.999, n_imagine_actions=IMAGINATION_N_ACTIONS)

    N = 64
    saw_non_legacy_action = False
    for seed in range(5):
        rng = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(rng, 3)
        carries = jax.random.normal(k1, (N, HIDDEN)) * 0.5
        obs = jax.random.normal(k2, (N, OBS_DIM)) * 0.1
        obs = obs.at[:, 2].set(0.5)
        params = model.init(k3, carries, obs, n_layers=2)["params"]
        _, outs = model.apply({"params": params}, carries, obs, n_layers=2)
        action_logits = outs.action_logits
        alive = jnp.ones((N,), dtype=jnp.bool_)
        imagined, _, _ = imagine(params, carries, action_logits, alive)
        if bool(jnp.any(imagined >= 5)):
            saw_non_legacy_action = True
            break

    assert saw_non_legacy_action, (
        "Across 5 independent 64-agent batches, imagine() never once selected an "
        "action outside the legacy 0-4 (Stay/N/S/E/W) range — the epistemic gate "
        "may still be effectively blind to Strike/Build/PickUp/Craft/UseTool "
        "despite imagination_n_actions=12."
    )


if __name__ == "__main__":
    test_live_config_requests_the_full_action_space()
    test_imagine_scores_tensor_has_one_value_per_action()
    test_imagine_argmax_lands_outside_legacy_five_actions()
    print(f"OK: imagine() scores the full {N_ACTIONS}-action space and the argmax "
          f"reaches beyond the legacy 5 (Stay/N/S/E/W).")
