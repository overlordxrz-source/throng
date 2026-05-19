"""
evolution/selection.py — Pure survival fitness, steady-state tournament selection.

Fitness = age * (offspring_count + 1)

No hand-designed bonus terms. If communication, cooperation, or brain depth
help an agent survive and reproduce, evolution finds it. If they don't, they
don't. The environment (predators) provides all the pressure needed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from agents.population import PopulationState
from agents.genome import get_agent_params, set_agent_params, mutate_params


def compute_fitness(pop: PopulationState) -> np.ndarray:
    """
    fitness = age * (offspring_count + 1)

    The +1 ensures agents that haven't reproduced yet still accumulate
    fitness over time, so early-life survivors can be selected as parents.
    Dead agents get 0.
    """
    offspring = getattr(pop, "offspring_count", np.zeros(pop.max_pop, dtype=np.int32))
    fitness   = pop.ages.astype(np.float32) * (offspring.astype(np.float32) + 1.0)
    fitness[~pop.alive] = 0.0
    return fitness


def run_evolution_step(
    pop:       PopulationState,
    config:    Dict,
    prng_key:  jax.random.KeyArray,
    model,
    grid_size: int,
    evo_count: int,
) -> Tuple[PopulationState, jax.random.KeyArray, Dict]:
    """
    Steady-state micro-selection pass.
    1. Compute fitness for all alive agents.
    2. Bottom replace_fraction are overwritten by mutated copies of top agents.
    3. Brain depth (n_layers) mutates with prob brain_layer_mutation_prob.
    """
    rng = np.random.default_rng(int(jax.random.randint(prng_key, (), 0, 2**30)))
    prng_key, _ = jax.random.split(prng_key)

    fitness       = compute_fitness(pop)
    alive_indices = np.where(pop.alive)[0]

    if len(alive_indices) < 4:
        return pop, prng_key, {"skipped": True}

    n_alive   = len(alive_indices)
    n_top     = max(1, int(n_alive * config["evolution_top_fraction"]))
    n_replace = max(1, int(n_alive * config["evolution_replace_fraction"]))

    alive_fitness     = fitness[alive_indices]
    sorted_by_fitness = alive_indices[np.argsort(alive_fitness)]
    targets           = sorted_by_fitness[:n_replace]

    parent_mask = np.zeros(pop.max_pop, dtype=bool)
    parent_mask[sorted_by_fitness[-n_top:]] = True

    min_l      = config.get("brain_min_layers", 1)
    max_l      = config.get("brain_max_layers", 4)
    layer_prob = config.get("brain_layer_mutation_prob", 0.12)

    replaced = 0
    for tgt in targets:
        eligible = np.where(parent_mask)[0]
        if len(eligible) == 0:
            break
        k         = min(config["tournament_k"], len(eligible))
        candidates = rng.choice(eligible, size=k, replace=False)
        parent_idx = int(candidates[np.argmax(fitness[candidates])])
        if parent_idx == int(tgt):
            continue

        prng_key, mut_key = jax.random.split(prng_key)
        parent_params = get_agent_params(pop.params, parent_idx)
        child_params  = mutate_params(
            parent_params, mut_key,
            sigma_small = config["mutation_sigma_small"],
            sigma_large = config["mutation_sigma_large"],
            prob_large  = config["mutation_large_prob"],
        )
        pop.params = set_agent_params(pop.params, int(tgt), child_params)

        parent_layers = int(pop.n_layers[parent_idx])
        if rng.random() < layer_prob:
            delta        = int(rng.choice([-1, 1]))
            child_layers = int(np.clip(parent_layers + delta, min_l, max_l))
        else:
            child_layers = parent_layers

        pop.n_layers[int(tgt)]          = np.int8(child_layers)
        pop.ages[tgt]                   = 0
        pop.offspring_count[tgt]        = 0
        pop.steps_since_catch[tgt]      = 0
        pop.lineage_ids[tgt]            = pop.next_lineage_id
        pop.next_lineage_id            += 1

        carries_np = np.array(pop.carries)
        carries_np[int(tgt)] = 0.0
        pop.carries = jnp.array(carries_np)

        replaced += 1

    mean_nl = float(pop.n_layers[alive_indices].mean())
    stats = {
        "evo_step":        evo_count,
        "n_alive":         n_alive,
        "mean_fitness":    float(np.mean(alive_fitness)),
        "max_fitness":     float(np.max(alive_fitness)),
        "n_replaced":      replaced,
        "mean_n_layers":   mean_nl,
        "top_lineage_age": int(pop.ages[sorted_by_fitness[-1]]),
    }
    return pop, prng_key, stats
