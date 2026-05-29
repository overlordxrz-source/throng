"""Force-reload jax_sim after git pull (use when the kernel cached old modules)."""
from __future__ import annotations

import importlib
import sys


def reload_throng_jax() -> None:
    """Drop cached jax_sim.* modules and re-import main_jax from disk."""
    for name in sorted(list(sys.modules)):
        if name == "jax_sim" or name.startswith("jax_sim."):
            del sys.modules[name]
    import jax_sim.main_jax as main_jax  # noqa: F401

    importlib.import_module("jax_sim.main_jax")
    import jax

    jax.clear_caches()
