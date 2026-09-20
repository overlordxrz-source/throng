"""Cam's fix-1 instruction (2026-09-20): forty updates of stage 1, 200 agents,
tens of thousands of futile_uncoordinated attempts, zero successes. Before
anyone redesigns the ecology -- prove the pair-craft success path can fire at
all, at production shapes, under ideal conditions: two adjacent agents, both
holding the demanded material, recipe matching, capacity-1 inventory. If it
can't fire under ideal conditions, that's the fifth instance of the same
defect (per catch_attempts=0, per the durability gate, per the retention
bug, per the Adam spring) and nothing else matters.

Production shapes: grid_size=128, max_pop=200 (config.yaml), stage 1's real
2-unit recipe (jax_sim/ctd_ramp.staged_recipe_counts, stage=1) -- tested with
both a same-material draw (e.g. 2 wood) and a different-material draw (e.g.
1 wood + 1 stone), since staged_recipe_counts can produce either and
capacity-1 inventory makes them meaningfully different cases. Two ideal
agents are embedded in a realistic 200-agent population where the other 198
are scattered, mostly not crafting, and where a THIRD, distant crafting
agent with no partner deliberately produces its own
futile_uncoordinated/futile_empty/futile_wrong_mats noise -- proving success
isn't an artifact of an otherwise-empty population.

Tests jax_sim.grid_jax.resolve_crafting() -- the exact function
jax_sim/main_jax.py calls (line ~607) -- and the exact reward-payment line
(jax_sim/main_jax.py line ~848: `b_rew + reward_craft_success *
craft_success.astype(jnp.float32)`), reproduced here as the same one-line
computation, not a reimplementation of grid logic.
"""

import jax
import jax.numpy as jnp

from jax_sim.grid_jax import resolve_crafting

GRID_SIZE = 128
N = 200
REWARD_CRAFT_SUCCESS = 3.0  # config.yaml default (reward_craft_success)


def _build_population(recipe_list, agent_a_mat, agent_b_mat, seed=0):
    """Agent 0 and agent 1: adjacent, crafting, holding agent_a_mat/agent_b_mat
    respectively (each capacity-1, i.e. exactly one unit of exactly one
    material). Agent 2: crafting, alone (far away, no partner) -- deliberately
    produces futile_uncoordinated noise. Agents 3..199: scattered, alive,
    NOT crafting -- realistic population background, contribute nothing to
    any craft outcome category."""
    key = jax.random.PRNGKey(seed)
    k_pos = key
    positions = jax.random.randint(k_pos, (N, 2), 0, GRID_SIZE)
    # Force the two ideal agents adjacent (Chebyshev distance 1) -- capacity-1
    # inventory, one unit each, exactly matching the pair-satisfiable design.
    positions = positions.at[0].set(jnp.array([64, 64]))
    positions = positions.at[1].set(jnp.array([64, 65]))
    # Agent 2: far from everyone, crafting alone.
    positions = positions.at[2].set(jnp.array([10, 10]))

    is_craft = jnp.zeros((N,), dtype=bool).at[0].set(True).at[1].set(True).at[2].set(True)

    mats = {"wood": 0, "stone": 1, "flint": 2, "clay": 3, "vine": 4}
    inv = {m: jnp.zeros((N,), dtype=jnp.int32) for m in mats}
    inv[agent_a_mat] = inv[agent_a_mat].at[0].set(1)
    inv[agent_b_mat] = inv[agent_b_mat].at[1].set(1)
    # Agent 2 crafts empty-handed (futile_empty), agents 3.. carry nothing.
    inv["wood"] = inv["wood"].at[5].set(1)  # a non-crafting agent holding material -- must not affect anything

    recipe = jnp.array(recipe_list, dtype=jnp.int32)
    out = resolve_crafting(
        positions, is_craft,
        inv["wood"], inv["stone"], inv["flint"], inv["clay"], inv["vine"],
        recipe, GRID_SIZE,
    )
    return out


def _assert_pair_succeeds(out, label):
    assert bool(out["craft_success"][0]), f"[{label}] agent 0 (ideal pair member) did not succeed"
    assert bool(out["craft_success"][1]), f"[{label}] agent 1 (ideal pair member) did not succeed"
    assert int(out["craft_success"].astype(jnp.int32).sum()) == 2, (
        f"[{label}] expected exactly 2 successes (agents 0,1), got "
        f"{int(out['craft_success'].astype(jnp.int32).sum())} -- someone else succeeded who shouldn't have"
    )
    # Agent 2: alone, crafting, no partner -- must be futile_uncoordinated (group_size=1 < total_req=2).
    assert bool(out["futile_uncoordinated"][2]), f"[{label}] agent 2 (alone, crafting) should be futile_uncoordinated"
    # Reward line, reproduced exactly as jax_sim/main_jax.py line ~848 computes it.
    b_rew = REWARD_CRAFT_SUCCESS * out["craft_success"].astype(jnp.float32)
    assert float(b_rew[0]) == REWARD_CRAFT_SUCCESS, f"[{label}] agent 0's craft reward was not paid"
    assert float(b_rew[1]) == REWARD_CRAFT_SUCCESS, f"[{label}] agent 1's craft reward was not paid"
    assert float(b_rew[2]) == 0.0, f"[{label}] agent 2 (failed) was paid a craft reward"
    assert float(b_rew.sum()) == 2 * REWARD_CRAFT_SUCCESS, f"[{label}] total craft reward paid out is wrong"


def test_pair_success_same_material_recipe_production_shapes():
    """Stage-1 recipe drawing the same material twice (e.g. staged_recipe_counts
    could draw [wood, wood] -> recipe [2,0,0,0,0]) -- two capacity-1 agents
    each holding one wood exactly satisfy it."""
    out = _build_population([2, 0, 0, 0, 0], agent_a_mat="wood", agent_b_mat="wood", seed=1)
    _assert_pair_succeeds(out, "same-material (2 wood)")


def test_pair_success_different_material_recipe_production_shapes():
    """Stage-1 recipe drawing two different materials (e.g. [wood, stone] ->
    recipe [1,1,0,0,0]) -- one agent holding wood, one holding stone."""
    out = _build_population([1, 1, 0, 0, 0], agent_a_mat="wood", agent_b_mat="stone", seed=2)
    _assert_pair_succeeds(out, "different-material (1 wood + 1 stone)")


def test_staged_recipe_counts_stage1_is_always_satisfiable_by_two_capacity1_agents():
    """staged_recipe_counts(stage=1) draws 2 material units total (possibly
    same or different materials) -- confirm every recipe it can actually
    produce is satisfiable by two capacity-1 agents, i.e. total_req is
    always exactly 2 and never concentrates more than 2 units on one
    material (which two capacity-1 agents could still cover, but this
    pins the actual invariant staged_recipe_counts relies on)."""
    from jax_sim.ctd_ramp import staged_recipe_counts
    key = jax.random.PRNGKey(0)
    for i in range(200):
        key, sub = jax.random.split(key)
        recipe = staged_recipe_counts(sub, jnp.array(1))
        total = int(recipe.sum())
        assert total == 2, f"stage-1 recipe #{i} has total_req={total}, expected exactly 2"
        assert int(recipe.max()) <= 2, f"stage-1 recipe #{i} concentrates >2 units on one material: {recipe}"


if __name__ == "__main__":
    test_pair_success_same_material_recipe_production_shapes()
    test_pair_success_different_material_recipe_production_shapes()
    test_staged_recipe_counts_stage1_is_always_satisfiable_by_two_capacity1_agents()
    print(
        "OK: the pair-craft success path fires under ideal conditions at production "
        "shapes (grid_size=128, N=200, embedded in a realistic scattered population "
        "with a genuine futile_uncoordinated distractor) for BOTH recipe shapes "
        "stage 1 can actually draw (same-material and different-material), and the "
        "craft-success reward is paid exactly to the two agents who earned it, "
        "reproducing main_jax.py's real reward-payment line. The mechanism is not "
        "the fifth instance of the empty-denominator defect -- it can fire."
    )
