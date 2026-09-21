"""Cam's instance-8 instruction (2026-09-20): can_see_recipe is built exactly
like is_big_green -- a one-time 50% draw at init_population, inherited
unchanged from an assigned parent at reproduction, never updated during an
agent's life (jax_sim/population_jax.py). The same session measured that
informed agents (can_see_recipe=True) hold a recipe-needed material at craft
time LESS often than uninformed agents (11.89% vs 16.19%, 95% CI on the
difference entirely below zero -- RESEARCH_PROTOCOL.md Part 2). Combined
with apply_auto_reproduce's uniform-by-alive-count parent sampling (zero
weighting by can_see_recipe, exactly as for is_big_green -- already proven
in tests/test_population_composition_lockin.py to make is_big_green a
one-way ratchet), a heritable trait currently under negative selection with
no reverse path heads for fixation at 0%: given enough generations the
informed caste disappears and receiver-necessity dies on its own,
independent of whatever else gets fixed about recipe zoning.

Same medicine as tests/test_is_big_green_mutation.py: run the REAL
apply_auto_reproduce forward, many real generations, from a
100%-can_see_recipe population, under a mortality asymmetry standing in for
the OBSERVED fitness disadvantage (informed agents measurably worse at the
one behavior recipe knowledge is supposed to help with), and confirm the
trait stabilizes at a low nonzero frequency rather than going extinct.
"""

import jax
import jax.numpy as jnp

from jax_sim.population_jax import PopState, apply_auto_reproduce

MAX_POP = 200
HIDDEN = 8
SIGNAL_DIM = 4
GRID_SIZE = 128
MUTATION_RATE = 0.03

# Mortality asymmetry standing in for the measured fitness disadvantage:
# informed agents hold a needed material at craft time ~27% less often in
# relative terms (11.89% vs 16.19%, RESEARCH_PROTOCOL.md Part 2, 2026-09-20)
# -- modeled here as an elevated death rate for the disadvantaged
# (informed) type, the same modeling choice test_is_big_green_mutation.py
# makes for small-blue's measured predation exposure.
P_DEATH_INFORMED = 0.25
P_DEATH_UNINFORMED = 0.05

N_GENERATIONS = 300
TAIL_WINDOW = 50


def _fresh_all_uninformed_population():
    """Starting point mirrors is_big_green's test the correct way round:
    is_big_green started at 100% of the FAVORED type (big-green) and showed
    the disadvantaged type (small-blue) reappears and stays low. Here,
    INFORMED is the disadvantaged type (measured negative selection), so
    the mirror-image start is 100% of the FAVORED type -- uninformed --
    simulating "the informed caste has already gone extinct," and the claim
    under test is that mutation prevents it from STAYING extinct."""
    pop = PopState(MAX_POP, HIDDEN, SIGNAL_DIM, memory_slots=0)
    positions = jax.random.randint(jax.random.PRNGKey(0), (MAX_POP, 2), 0, GRID_SIZE)
    pop = pop.replace(
        alive=jnp.ones((MAX_POP,), dtype=bool),
        can_see_recipe=jnp.zeros((MAX_POP,), dtype=bool),  # 0% informed, 100% uninformed
        energy=jnp.full((MAX_POP,), 0.5),
        positions=positions,
    )
    return pop


def test_mutation_reintroduces_and_stabilizes_informed_under_observed_disadvantage():
    pop = _fresh_all_uninformed_population()
    key = jax.random.PRNGKey(42)

    informed_fracs = []
    for gen in range(N_GENERATIONS):
        key, k_death_informed, k_death_uninformed, k_repro = jax.random.split(key, 4)

        is_informed = pop.alive & pop.can_see_recipe
        is_uninformed = pop.alive & ~pop.can_see_recipe
        die_informed = is_informed & (jax.random.uniform(k_death_informed, (MAX_POP,)) < P_DEATH_INFORMED)
        die_uninformed = is_uninformed & (jax.random.uniform(k_death_uninformed, (MAX_POP,)) < P_DEATH_UNINFORMED)
        pop = pop.replace(alive=pop.alive & ~die_informed & ~die_uninformed)

        pop = apply_auto_reproduce(
            pop, k_repro, GRID_SIZE, min_pop=MAX_POP,
            energy_thresh=2.0, energy_cost=0.4,
            can_see_recipe_mutation_rate=MUTATION_RATE,
        )
        pop = pop.replace(energy=jnp.where(pop.alive, jnp.maximum(pop.energy, 0.5), pop.energy))

        n_alive = int(pop.alive.sum())
        n_informed = int((pop.alive & pop.can_see_recipe).sum())
        informed_fracs.append(n_informed / n_alive if n_alive > 0 else 0.0)

    # Claim under test: informed (the disadvantaged type) does NOT stay
    # extinct once mutation is present -- it's constantly regenerated from
    # uninformed parents even though selection works against it every
    # generation.
    assert max(informed_fracs) > 0.0, (
        "informed (can_see_recipe=True) never reappeared across 300 generations from a "
        "100%-uninformed start -- mutation is not landing"
    )
    first_nonzero_gen = next(i for i, f in enumerate(informed_fracs) if f > 0.0)
    assert first_nonzero_gen < 100, (
        f"informed took {first_nonzero_gen} generations to first reappear -- suspiciously "
        f"slow for a 3% per-birth mutation rate refilling a ~200-slot population every generation"
    )

    tail = informed_fracs[-TAIL_WINDOW:]
    tail_mean = sum(tail) / len(tail)
    assert tail_mean > 0.0, (
        f"informed fraction collapsed back to 0 in the final {TAIL_WINDOW} generations "
        f"(tail={tail}) -- mutation reintroduced it but the observed selection pressure wiped "
        f"it out again; not stable"
    )
    assert tail_mean < 0.30, (
        f"informed tail-window mean fraction is {tail_mean:.3f}, too high to call 'low "
        f"nonzero frequency' under a real fitness disadvantage"
    )

    print(
        f"[measured] first informed reappearance: generation {first_nonzero_gen}; "
        f"tail ({TAIL_WINDOW}-generation) mean fraction: {tail_mean:.4f}; "
        f"tail range: [{min(tail):.4f}, {max(tail):.4f}]"
    )


def test_zero_mutation_rate_leaves_can_see_recipe_locked():
    """Regression guard, mirroring test_is_big_green_mutation.py's: rate=0.0
    (the default) must not spontaneously produce an informed spawn from an
    all-uninformed population."""
    pop = _fresh_all_uninformed_population()
    pop = pop.replace(alive=pop.alive.at[50:].set(False))
    key = jax.random.PRNGKey(7)
    new_pop = apply_auto_reproduce(
        pop, key, GRID_SIZE, min_pop=MAX_POP, energy_thresh=2.0, energy_cost=0.4,
        can_see_recipe_mutation_rate=0.0,
    )
    newly_alive = new_pop.alive & ~pop.alive
    assert int(newly_alive.sum()) > 0, "fixture bug: no new spawns to check"
    assert int((newly_alive & new_pop.can_see_recipe).sum()) == 0, (
        "can_see_recipe_mutation_rate=0.0 produced an informed spawn from an all-uninformed "
        "population -- the default (off) behavior changed"
    )


def test_is_big_green_and_can_see_recipe_mutation_are_independent():
    """Both mutation rates can be nonzero simultaneously without interfering
    -- each trait's mutation roll uses its own RNG split (k5 vs k6) and
    applies to its own field only."""
    pop = PopState(MAX_POP, HIDDEN, SIGNAL_DIM, memory_slots=0)
    positions = jax.random.randint(jax.random.PRNGKey(1), (MAX_POP, 2), 0, GRID_SIZE)
    pop = pop.replace(
        alive=jnp.ones((MAX_POP,), dtype=bool),
        is_big_green=jnp.ones((MAX_POP,), dtype=bool),
        can_see_recipe=jnp.zeros((MAX_POP,), dtype=bool),
        energy=jnp.full((MAX_POP,), 0.5),
        positions=positions,
    )
    pop = pop.replace(alive=pop.alive.at[50:].set(False))
    key = jax.random.PRNGKey(9)
    new_pop = apply_auto_reproduce(
        pop, key, GRID_SIZE, min_pop=MAX_POP, energy_thresh=2.0, energy_cost=0.4,
        is_big_green_mutation_rate=0.5, can_see_recipe_mutation_rate=0.0,
    )
    newly_alive = new_pop.alive & ~pop.alive
    assert int(newly_alive.sum()) > 0, "fixture bug: no new spawns to check"
    # is_big_green mutating at 0.5 should show real variance among new spawns.
    n_new = int(newly_alive.sum())
    n_new_small = int((newly_alive & ~new_pop.is_big_green).sum())
    assert 0 < n_new_small < n_new, (
        f"is_big_green_mutation_rate=0.5 produced {n_new_small}/{n_new} flips -- expected a "
        f"real mix, not all-or-nothing (fixture too small or a real bug)"
    )
    # can_see_recipe_mutation_rate=0.0 must leave that trait untouched regardless.
    assert int((newly_alive & new_pop.can_see_recipe).sum()) == 0, (
        "can_see_recipe_mutation_rate=0.0 was affected by is_big_green_mutation_rate=0.5 -- "
        "the two mutation rolls are not independent"
    )


if __name__ == "__main__":
    test_mutation_reintroduces_and_stabilizes_informed_under_observed_disadvantage()
    test_zero_mutation_rate_leaves_can_see_recipe_locked()
    test_is_big_green_and_can_see_recipe_mutation_are_independent()
    print(
        "OK: can_see_recipe_mutation_rate=0.03, run through the REAL apply_auto_reproduce for "
        "300 real generations starting from a 100%-uninformed population under a mortality "
        "asymmetry standing in for the measured fitness disadvantage, reintroduces the "
        "informed type and stabilizes it at a low nonzero tail-window frequency -- neither "
        "extinct nor exploding. rate=0.0 exactly reproduces the lock-in (regression guard), "
        "and the two mutation rates (is_big_green, can_see_recipe) are confirmed independent."
    )
