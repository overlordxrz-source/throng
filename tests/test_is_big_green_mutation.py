"""Cam's fix-4 instruction (2026-09-20): land the mutation rate, and prove it
breaks the ratchet the way it's meant to -- not a floor (an external hand
preventing an outcome selection is producing), a rule of the world (a
constant per-birth chance the offspring's type flips relative to its
parent, present regardless of current population composition).

tests/test_population_composition_lockin.py already proved the ratchet:
apply_auto_reproduce's parent-sampling is uniform by current alive count,
zero weighting by is_big_green, so at 0 small-blue alive every new spawn is
confirmed big-green, permanently. is_big_green_mutation_rate fixes that at
the point where a fixed trait becomes an inherited-with-mutation one.

This test runs the REAL apply_auto_reproduce forward, many real generations,
starting from a 100%-big-green population (the live run's actual state) --
not a toy shortcut. Each generation applies asymmetric mortality (small-blue
dies at an elevated rate, matching the real ecology: red's catch mechanic is
small-blue-only, so small-blue is measurably more exposed than big-green,
which requires a 2+-red coordinated strike per jax_sim/grid_jax.py's
apply_catches -- see docs/RESEARCH_PROTOCOL.md 2026-09-20 findings) before
reproduction refills the shortfall. This is the actual claim being tested:
does mutation, under real selection pressure working against it, produce a
low STABLE nonzero frequency (mutation-selection balance) rather than
staying extinct (mutation too weak / never landing) or exploding to parity
(mutation overwhelming selection)?
"""

import jax
import jax.numpy as jnp

from jax_sim.population_jax import PopState, apply_auto_reproduce

MAX_POP = 200
HIDDEN = 8
SIGNAL_DIM = 4
GRID_SIZE = 128
MUTATION_RATE = 0.03

# Asymmetric mortality standing in for real predation pressure: small-blue is
# the only type red's proximity-catch mechanic can kill without 2+-red
# coordination (docs/RESEARCH_PROTOCOL.md 2026-09-20, item 3) -- elevated,
# not equal, death rate is the actual selection force mutation has to work
# against for "low nonzero" (not 50/50) to be the right claim.
P_DEATH_SMALL_BLUE = 0.25
P_DEATH_BIG_GREEN = 0.05

N_GENERATIONS = 300
TAIL_WINDOW = 50  # generations to average over once the system should have settled


def _fresh_all_big_green_population():
    pop = PopState(MAX_POP, HIDDEN, SIGNAL_DIM, memory_slots=0)
    positions = jax.random.randint(jax.random.PRNGKey(0), (MAX_POP, 2), 0, GRID_SIZE)
    pop = pop.replace(
        alive=jnp.ones((MAX_POP,), dtype=bool),
        is_big_green=jnp.ones((MAX_POP,), dtype=bool),  # 100% big-green, 0% small-blue
        energy=jnp.full((MAX_POP,), 0.5),
        positions=positions,
    )
    return pop


def test_mutation_reintroduces_and_stabilizes_small_blue_under_real_selection():
    pop = _fresh_all_big_green_population()
    key = jax.random.PRNGKey(42)

    small_blue_fracs = []
    for gen in range(N_GENERATIONS):
        key, k_death_small, k_death_big, k_repro = jax.random.split(key, 4)

        is_small = pop.alive & ~pop.is_big_green
        is_big = pop.alive & pop.is_big_green
        die_small = is_small & (jax.random.uniform(k_death_small, (MAX_POP,)) < P_DEATH_SMALL_BLUE)
        die_big = is_big & (jax.random.uniform(k_death_big, (MAX_POP,)) < P_DEATH_BIG_GREEN)
        pop = pop.replace(alive=pop.alive & ~die_small & ~die_big)

        # energy_thresh unreachable (2.0 > any real energy) -- every spawn
        # this test produces comes through the min_pop shortfall path, the
        # exact path find_fossil-adjacent reasoning and
        # test_population_composition_lockin.py already characterized.
        pop = apply_auto_reproduce(
            pop, k_repro, GRID_SIZE, min_pop=MAX_POP,
            energy_thresh=2.0, energy_cost=0.4,
            is_big_green_mutation_rate=MUTATION_RATE,
        )
        pop = pop.replace(energy=jnp.where(pop.alive, jnp.maximum(pop.energy, 0.5), pop.energy))

        n_alive = int(pop.alive.sum())
        n_small = int((pop.alive & ~pop.is_big_green).sum())
        small_blue_fracs.append(n_small / n_alive if n_alive > 0 else 0.0)

    # 1. Small-blue must actually reappear -- not stuck at 0 the whole run.
    assert max(small_blue_fracs) > 0.0, (
        "small-blue never reappeared across 300 generations from a 100%-big-green start -- "
        "mutation is not landing"
    )
    first_nonzero_gen = next(i for i, f in enumerate(small_blue_fracs) if f > 0.0)
    assert first_nonzero_gen < 100, (
        f"small-blue took {first_nonzero_gen} generations to first reappear -- "
        f"suspiciously slow for a 3% per-birth mutation rate refilling a ~200-slot population every generation"
    )

    # 2. Must STABILIZE, not go extinct again in the tail.
    tail = small_blue_fracs[-TAIL_WINDOW:]
    tail_mean = sum(tail) / len(tail)
    assert tail_mean > 0.0, (
        f"small-blue fraction collapsed back to 0 in the final {TAIL_WINDOW} generations "
        f"(tail={tail}) -- mutation reintroduced it but selection wiped it out again; not stable"
    )

    # 3. Must be LOW, not exploding toward parity -- this is the actual
    # mutation-selection-balance claim, not just "nonzero somewhere."
    assert tail_mean < 0.30, (
        f"small-blue tail-window mean fraction is {tail_mean:.3f}, too high to call 'low nonzero "
        f"frequency' -- either mutation is overwhelming the asymmetric selection pressure in this "
        f"fixture, or the rate needs reconsidering"
    )

    print(
        f"[measured] first small-blue reappearance: generation {first_nonzero_gen}; "
        f"tail ({TAIL_WINDOW}-generation) mean fraction: {tail_mean:.4f}; "
        f"tail range: [{min(tail):.4f}, {max(tail):.4f}]"
    )


def test_zero_mutation_rate_reproduces_the_original_lockin_exactly():
    """Regression guard: is_big_green_mutation_rate=0.0 (the default) must
    reproduce the exact lock-in test_population_composition_lockin.py
    already proved -- this fix must be opt-in via the rate, not a change to
    default reproduction behavior."""
    pop = _fresh_all_big_green_population()
    pop = pop.replace(alive=pop.alive.at[50:].set(False))  # 50 alive, all big-green, 150 dead slots
    key = jax.random.PRNGKey(7)
    new_pop = apply_auto_reproduce(
        pop, key, GRID_SIZE, min_pop=MAX_POP, energy_thresh=2.0, energy_cost=0.4,
        is_big_green_mutation_rate=0.0,
    )
    newly_alive = new_pop.alive & ~pop.alive
    assert int(newly_alive.sum()) > 0, "fixture bug: no new spawns to check"
    assert int((newly_alive & ~new_pop.is_big_green).sum()) == 0, (
        "is_big_green_mutation_rate=0.0 produced a small-blue spawn from an all-big-green "
        "population -- the default (off) behavior changed"
    )


if __name__ == "__main__":
    test_mutation_reintroduces_and_stabilizes_small_blue_under_real_selection()
    test_zero_mutation_rate_reproduces_the_original_lockin_exactly()
    print(
        "OK: is_big_green_mutation_rate=0.03, run through the REAL apply_auto_reproduce for "
        "300 real generations starting from a 100%-big-green population under asymmetric "
        "(predation-like) mortality, reintroduces small-blue within 100 generations and "
        "stabilizes it at a low (<30%) nonzero tail-window frequency -- neither extinct nor "
        "exploding. mutation_rate=0.0 exactly reproduces the original lock-in (regression guard)."
    )
