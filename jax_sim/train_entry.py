"""
Import run_simulation from here after `git pull` (no kernel restart needed).

Clears cached jax_sim.* modules so Python loads .py files from disk.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Dict, Tuple

import jax


def _evict_jax_sim_modules() -> None:
    for name in sorted(list(sys.modules)):
        if name == "jax_sim" or name.startswith("jax_sim."):
            del sys.modules[name]
    jax.clear_caches()


def _assert_observations_module() -> None:
    from jax_sim import observations_jax as obs

    src = inspect.getsource(obs.build_observations_jax)
    if "red_map.any()" in src:
        raise RuntimeError(
            "observations_jax.py on disk is outdated (red_map.any). "
            "Run: cd /root/throng && git fetch origin && git reset --hard origin/master"
        )
    if "limit_red_sensing" not in inspect.signature(obs.build_observations_jax).parameters:
        raise RuntimeError("observations_jax.py on disk is outdated (missing limit_red_sensing).")


def run_simulation(
    config: Dict,
    seed: int = 42,
    n_steps: int = 100_000,
) -> Tuple[Any, Dict]:
    """Run JAX training with a fresh import of jax_sim from disk."""
    _evict_jax_sim_modules()
    from jax_sim.main_jax import _run_simulation_impl

    _assert_observations_module()
    return _run_simulation_impl(config, seed=seed, n_steps=n_steps)
