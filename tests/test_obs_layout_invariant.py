"""
Pin the observation layout against the LIVE config (AUDIT_SEP2026.md Finding 3,
restart order item 1.3a).

`own_state_dim` has grown three times over this project's history (6 -> 10 ->
22) and `env_channels` twice (9 -> 10/12 -> 15), and each time a hand-rolled
offset formula elsewhere in the codebase silently kept using a stale value —
most recently `jax_sim/main_jax.py`'s Phase-17 NPMI corpus-diagnostic block,
which computed `loc_env_start` as `6 + neighbor_k*signal_dim + 25*symbol_dim`
(assuming a 6-dim own_state with no neighbor-alarm block) and was off by 28
columns against the config this test loads, silently reading `contested`/
`wood`/`stone` into `signal_corpus.jsonl` fields named `local_resource`/
`adj_bg`/`adj_barrier`.

That call site has been fixed to read `make_obs_layout(...).loc_env_start`
directly instead of re-deriving it — this test's job is to make sure the
NEXT own_state_dim/env_channels change can't repeat the mistake silently: it
pins the current numeric value so a layout change fails loudly here (forcing
whoever changed it to go check every other consumer) rather than three phases
later in a corrupted corpus field nobody is looking at yet.
"""

import yaml

from jax_sim.main_jax import DEFAULT_CONFIG, _normalize_config
from jax_sim.obs_layout import make_obs_layout

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})


def _live_layout():
    return make_obs_layout(
        signal_dim=int(_cfg["signal_dim"]),
        symbol_dim=int(_cfg["symbol_dim"]),
        memory_slots=int(_cfg.get("memory_slots", 0)),
        neighbor_k=int(_cfg["neighbor_k"]),
        local_cells=(2 * int(_cfg["local_obs_radius"]) + 1) ** 2,
        env_channels=int(_cfg.get("env_channels", 15)),
        own_state_dim=int(_cfg.get("own_state_dim", 22)),
    )


def test_loc_env_start_is_pinned_for_the_live_config():
    """If this fails, own_state_dim/env_channels/neighbor_k/symbol_dim/
    memory_slots changed in config.yaml. That's fine — update the pinned
    value below — but ALSO go check every hand-rolled consumer of obs-layout
    boundaries (grep for `idx_offset`, `loc_env_start`, magic slice indices
    into `obs`/`b_obs_all`) instead of just silencing this test."""
    layout = _live_layout()
    assert layout.own_state_dim == 22, (
        f"own_state_dim changed to {layout.own_state_dim} (was 22) — this is exactly "
        "the kind of change that broke the Finding 3 formula three times before."
    )
    assert layout.env_channels == 15, f"env_channels changed to {layout.env_channels} (was 15)"
    assert layout.loc_env_start == 674, (
        f"loc_env_start is {layout.loc_env_start}, expected 674 for the pinned config "
        "(own_state=22, neighbor_k=6, signal_dim=40, symbol_dim=16). Update this pin "
        "AND check every hand-rolled consumer of obs-layout boundaries."
    )
    assert layout.total_dim == 2731, f"total_dim is {layout.total_dim}, expected 2731"


def test_corpus_writer_uses_make_obs_layout_not_a_hand_rolled_formula():
    """Structural guard against reintroducing Finding 3: main_jax.py's corpus
    diagnostic block must derive its loc_env offset from `_loc_env_start`
    (itself sourced from `make_obs_layout` at the top of `_run_simulation_impl`)
    rather than a second, independently-computed formula that can drift."""
    import inspect
    from jax_sim import main_jax

    src = inspect.getsource(main_jax)
    # The Phase-17 NPMI block's own comment names this exact expression as
    # the historical bug; make sure the literal hand-rolled formula is gone.
    assert "6 + (config[\"neighbor_k\"]" not in src, (
        "The hand-rolled `idx_offset = 6 + ...` formula (Finding 3) is back in "
        "main_jax.py — it must derive from make_obs_layout's loc_env_start instead."
    )
    assert "idx_offset = _loc_env_start" in src, (
        "Expected the corpus-diagnostic block to read the canonical _loc_env_start "
        "(from make_obs_layout) rather than recomputing its own offset."
    )


if __name__ == "__main__":
    test_loc_env_start_is_pinned_for_the_live_config()
    test_corpus_writer_uses_make_obs_layout_not_a_hand_rolled_formula()
    print("OK: obs-layout is pinned for the live config and the corpus writer uses make_obs_layout.")
