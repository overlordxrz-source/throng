"""Cam's fix-2 question (2026-09-20): what transitions an agent to big_green,
is it reversible, and can small blue exist in steady state at all?

Answer, read directly from jax_sim/population_jax.py: there is no
transition. is_big_green is a fixed trait set once, at one of exactly two
points -- the initial population's one-time 20% random draw
(init_population, line ~176: `is_bg = jax.random.uniform(...) < 0.2`), or
inherited unchanged from an assigned parent at reproduction
(apply_auto_reproduce, line ~305-306: `parent_is_big_green =
pop.is_big_green[assigned_parents]`). No individual agent's own
is_big_green value is ever read and reassigned during its life -- confirmed
by exhaustive grep, every occurrence in jax_sim/*.py is either a read
(catch/reward logic) or one of those two write sites.

The mechanism that actually matters is who gets chosen as `assigned_parents`
for new spawns: `parent_weights = jnp.where(pop.alive, 1.0, 0.0)` --
UNIFORM across every currently alive agent, zero weight by is_big_green.
A new agent's type is exactly as likely to be small-blue as the CURRENT
alive population's small-blue fraction is. Combined with red's catch
mechanic being small-blue-only (jax_sim/grid_jax.py's in_range_small vs
in_range_big split) -- an elevated death rate for small-blue with no
compensating protection in who reproduces -- this is a one-way demographic
ratchet, not a steady state: once small-blue's alive count reaches 0, the
parent-sampling weight for producing a new small-blue agent is exactly 0,
permanently, from that population state onward. Verified here directly
against apply_auto_reproduce, not inferred from reading it.
"""

import jax
import jax.numpy as jnp

from jax_sim.population_jax import PopState, apply_auto_reproduce

MAX_POP = 200
HIDDEN = 8
SIGNAL_DIM = 4
GRID_SIZE = 128


def _pop_with_composition(n_alive_small, n_alive_big, energy=0.5):
    """max_pop slots; first n_alive_small are small-blue alive, next
    n_alive_big are big-green alive, the rest dead (empty slots for
    reproduction to fill)."""
    pop = PopState(MAX_POP, HIDDEN, SIGNAL_DIM, memory_slots=0)
    n_alive = n_alive_small + n_alive_big
    alive = jnp.arange(MAX_POP) < n_alive
    is_bg = (jnp.arange(MAX_POP) >= n_alive_small) & (jnp.arange(MAX_POP) < n_alive)
    positions = jax.random.randint(jax.random.PRNGKey(0), (MAX_POP, 2), 0, GRID_SIZE)
    pop = pop.replace(
        alive=alive,
        is_big_green=is_bg,
        energy=jnp.where(alive, energy, 0.0),
        positions=positions,
    )
    return pop


def test_zero_small_blue_alive_locks_out_small_blue_from_every_new_spawn():
    """The decisive case: population is currently 0 small-blue, 150
    big-green alive (matching the live run's observed pop_split=small:0).
    Force reproduction (min_pop > alive_count, so new agents MUST spawn)
    and confirm every single new spawn is big-green -- none can be
    small-blue, because no small-blue parent exists to assign."""
    pop = _pop_with_composition(n_alive_small=0, n_alive_big=150)
    key = jax.random.PRNGKey(1)
    new_pop = apply_auto_reproduce(pop, key, GRID_SIZE, min_pop=200, energy_thresh=2.0, energy_cost=0.4)

    newly_alive = new_pop.alive & ~pop.alive
    n_new = int(newly_alive.sum())
    assert n_new > 0, "fixture bug: reproduction produced no new spawns to check"
    n_new_small_blue = int((newly_alive & ~new_pop.is_big_green).sum())
    assert n_new_small_blue == 0, (
        f"{n_new_small_blue} of {n_new} new spawns were small-blue despite ZERO small-blue "
        f"parents being alive -- either a bug, or is_big_green is no longer parent-inherited"
    )
    total_small_blue_after = int((new_pop.alive & ~new_pop.is_big_green).sum())
    assert total_small_blue_after == 0, (
        "small-blue count changed from 0 despite no small-blue parents existing -- "
        "confirms the lock-in is total, not partial"
    )


def test_reproduction_is_proportional_to_current_alive_composition_not_protected():
    """Negative control / mechanism check: with a MIXED alive population,
    new spawns should land close to the CURRENT alive fraction (no hidden
    protection or rebalancing toward the minority type) -- confirming the
    lock-in above is a direct, expected consequence of proportional
    sampling, not a separate bug only triggered at the n=0 edge case."""
    # 20% small-blue alive (matching the ORIGINAL init_population split),
    # to show reproduction preserves roughly that ratio when it isn't zero.
    pop = _pop_with_composition(n_alive_small=30, n_alive_big=120)
    key = jax.random.PRNGKey(2)
    new_pop = apply_auto_reproduce(pop, key, GRID_SIZE, min_pop=200, energy_thresh=2.0, energy_cost=0.4)

    newly_alive = new_pop.alive & ~pop.alive
    n_new = int(newly_alive.sum())
    assert n_new > 0, "fixture bug: reproduction produced no new spawns to check"
    n_new_small_blue = int((newly_alive & ~new_pop.is_big_green).sum())
    observed_frac = n_new_small_blue / n_new
    expected_frac = 30 / 150  # = 0.20, the current alive small-blue fraction
    assert abs(observed_frac - expected_frac) < 0.15, (
        f"new-spawn small-blue fraction ({observed_frac:.2f}) is far from the current alive "
        f"fraction ({expected_frac:.2f}) -- reproduction is not simply proportional, mechanism "
        f"understanding above may be wrong"
    )


if __name__ == "__main__":
    test_zero_small_blue_alive_locks_out_small_blue_from_every_new_spawn()
    test_reproduction_is_proportional_to_current_alive_composition_not_protected()
    print(
        "OK: is_big_green is parent-inherited with zero protection for the minority type "
        "(reproduction samples parents uniformly by current alive count, not by type) -- at "
        "0 small-blue alive (the live run's actual state), every new spawn is confirmed "
        "big-green, and the lock-in is total, not partial. This is a one-way demographic "
        "ratchet, not a steady state: once small-blue reaches 0, predation is structurally "
        "impossible regardless of red's policy, and the population cannot self-heal."
    )
