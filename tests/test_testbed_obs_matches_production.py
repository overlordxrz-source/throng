"""The learner testbed builds its own observation vector; its hearth-direction block must be exactly what
production's build_observations_jax gives an agent (Cam, 2026-11-05: the transfer test assumes the testbed
policy reads the same compass production agents get). Checked at every ladder size."""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from jax_sim.grid_jax import GridState
from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.observations_jax import build_observations_jax
from jax_sim.population_jax import PopState

_spec = importlib.util.spec_from_file_location(
    "learner_testbed", Path(__file__).resolve().parents[1] / "scripts" / "learner_testbed.py")
_tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tb)

with open(Path(__file__).resolve().parents[1] / "config.yaml") as f:
    _base = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(f)})


def _production_hearth_rel(G, positions):
    cfg = dict(_base, grid_size=G)
    n = positions.shape[0]
    pop = PopState(n, hidden_dim=int(cfg["hidden_dim"]), signal_dim=int(cfg["signal_dim"]), memory_slots=0)
    pop = pop.replace(alive=jnp.ones((n,), bool), positions=jnp.asarray(positions, jnp.int32),
                      can_see_recipe=jnp.ones((n,), bool), energy=jnp.full((n,), 1.0))
    grid = GridState(G, symbol_dim=int(cfg["symbol_dim"]), max_pop=n)
    grid = grid.replace(hearth_positions=jnp.full((4, 2), G // 2, jnp.int32))
    m = jnp.zeros((G, G), bool)
    obs = build_observations_jax(pop, grid, m.at[pop.positions[:, 0], pop.positions[:, 1]].set(True), m, cfg, step=0)
    return np.asarray(obs[:, 13:21])


def test_testbed_hearth_rel_equals_production_at_every_ladder_size():
    rng = np.random.default_rng(0)
    for G in (15, 32, 64, 128):
        pos = rng.integers(0, G, (16, 2))
        prod = _production_hearth_rel(G, pos)
        tb = _tb.testbed_hearth_rel(pos, np.array([G // 2, G // 2]), G)
        np.testing.assert_allclose(tb, prod, atol=1e-6, err_msg=f"G={G}")
        assert np.abs(prod).max() > 0.1, "non-trivial positions"


def test_mutant_without_toroidal_wrap_is_caught():
    G = 32
    pos = np.array([[0, 0], [31, 31], [1, 30]] + [[5, 5]] * 13)
    prod = _production_hearth_rel(G, pos)
    naive = np.tile(((np.array([16, 16]) - pos) / 16.0).astype(np.float32), (1, 4))
    assert not np.allclose(naive, prod, atol=1e-6)
