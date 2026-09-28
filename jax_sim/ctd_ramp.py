"""CtD competence ramp for blue's crafting-avoidance and red's hunting-avoidance
fossils (RESEARCH_PROTOCOL.md Part 3), modelled on the Phase 16 Big-Green
solo-catchable gate: one easy regime, one ratchet to full difficulty, config-
driven, no interpolation. Both fossils are policies trained ~1.4M steps under
a broken ecology; the ramp gives each a period where the mechanism it avoided
is actually reachable, so the policy has something to learn from before facing
the real difficulty. Proposed and reviewed 2026-09-13/14 (Cam).

Ratchet decision (floor + sustained bar + hard ceiling) lives in the outer
Python training loop (jax_sim/main_jax.py's per-update loop), mirroring the
existing red_curriculum_idx/red_sustain_count pattern -- it is not persisted
across process resume, matching that precedent's existing (undocumented until
now) limitation. The two functions below are the per-env-step mechanics that
run inside the JIT-compiled rollout and must be traced-value-correct.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


HEARTH_RAMP_STAGE_N = (1, 1, 2, 3)  # required deposits to complete a hearth, per stage
HEARTH_RAMP_ACCEPT_ANY = (True, False, False, False)  # does ANY material complete it, or only the hearth's own need
HEARTH_RAMP_STAGE_LABELS = ("A0", "A1", "B", "B+")  # human labels, purely for prints/dashboards
# Index of the first stage requiring coordination (N>1) -- module-level, not
# a per-function local, so every place that needs "which stage is the
# frozen/solo regime vs the live/coordination one" (main_jax.py's
# _freeze_comms, the C0_stage1 search trigger, the fast/drift baseline
# selector, and the PRESSURE tripwire) derives it the same way instead of
# each recomputing or -- worse -- hardcoding the index locally.
HEARTH_COORD_STAGE_IDX = next(
    (_i for _i, _n in enumerate(HEARTH_RAMP_STAGE_N) if _n > 1), len(HEARTH_RAMP_STAGE_N)
)
# 2026-09-21 (Cam), hearths replace pair-adjacency crafting entirely: the old
# CRAFT_RAMP_STAGE_UNITS = (1, 2) recipe-unit cap and staged_recipe_counts()
# are deleted, not kept alongside the new mechanism -- "every parallel
# mechanism is another place for a silent bug, and we've found eight." The
# same floor/bar/window/ceiling ratchet in jax_sim/main_jax.py's per-update
# loop (generic over len(...)) now drives a hearth's required deposit count N
# instead of a recipe's unit count.
#
# A0/A1 split (2026-09-28, Cam, after three stage-0 escapes at three
# different hearth geometries -- single-tile, then 7x7, with the bottleneck
# never moving): the old stage 0 (N=1) was a CONJUNCTION of "on a hearth" and
# "holding the hearth's specific need," and only the first conjunct had ever
# been measured or fixed. Widening the hearth radius made the first nearly
# free (of all on-hearth-tile agent-steps, 66% were already holding SOME
# material) but the second still failed ~99% of the time (0.65% held the
# NEEDED material) -- so arrivals were almost never rewarded, so navigation
# was never learned, so distance-to-hearth stayed flat at the random-walk
# expectation for three consecutive stage-0 windows. Splitting isolates the
# two conjuncts instead of asking the population to solve both at once:
#   A0 (N=1, accept_any=True):  any material completes it. Pure navigation +
#     gathering, no recipe knowledge required or rewarded. If a population
#     that already finds hearths constantly can't clear this, the problem is
#     in the learner (reward scale, credit assignment, hearth-position
#     observability) -- not the ecology -- and that's where to look next.
#   A1 (N=1, accept_any=False): only the needed material completes it --
#     identical to the old stage 0. Learn to carry the RIGHT thing. This is
#     the first rung in this project's history where knowing the recipe
#     (can_see_recipe) is worth something and not everyone has it -- Gate 2
#     becomes a meaningful measurement here, not before.
#   B  (N=2, accept_any=False): unchanged from the old stage 1 -- multiple
#     deposits required before the shared decay clock (see
#     grid_jax.resolve_hearth_deposits) erases the earlier ones, at which
#     point coordination becomes the strictly faster path without being the
#     only possible one.
#   B+ (N=3, accept_any=False): unchanged from the old stage 2, carried over
#     as-is; not part of the A0/A1 redesign.
#
# Comms freeze/unfreeze and the hard escape are both derived generically from
# HEARTH_RAMP_STAGE_N/keyed off "is this stage solo-satisfiable" (N==1), not
# from a hardcoded stage index -- see main_jax.py's _freeze_comms and the
# per-update ratchet block. That means A0 and A1 both keep comms frozen (no
# coordination pressure in either), and the hard escape (40 updates, same
# budget as before) now applies to every rung, not just the first.


def _nearest_alive_blue_dist(
    red_pos: jnp.ndarray,       # (R, 2)
    blue_pos: jnp.ndarray,      # (B, 2)
    blue_alive: jnp.ndarray,    # (B,)
    grid_size: int,
) -> jnp.ndarray:
    """Toroidal Chebyshev distance from each red agent to its nearest living
    blue -- the same metric apply_catches/resolve_crafting already use. 0 when
    no blue is alive (sentinel: with Phi = -d/D_max, d=0 gives Phi=0, i.e. no
    shaping rather than an artificial reward for an empty map)."""
    diff = jnp.abs(red_pos[:, None, :] - blue_pos[None, :, :])
    diff = jnp.minimum(diff, grid_size - diff)
    dist = jnp.max(diff, axis=-1)  # (R, B)
    dist = jnp.where(blue_alive[None, :], dist, jnp.inf)
    min_dist = jnp.min(dist, axis=1)
    any_alive = jnp.any(blue_alive)
    return jnp.where(any_alive, min_dist, 0.0)


def hearth_shaping_term(
    agent_pos_before: jnp.ndarray,   # (N, 2)
    agent_pos_after: jnp.ndarray,    # (N, 2)
    held_material: jnp.ndarray,      # (N,) int, -1 = empty; same snapshot used for both evaluations below
    hearth_positions: jnp.ndarray,   # (H, 2), fixed
    hearth_need: jnp.ndarray,        # (H,) int, the need this step actually checks (pre-resolution)
    made_deposit: jnp.ndarray,       # (N,) bool -- this agent deposited this step
    beta: float,
    gamma: float,
    grid_size: int,
) -> jnp.ndarray:
    """Potential-based reward shaping (Ng, Harada & Russell 1999) for approaching
    the nearest hearth whose CURRENT need matches the material this agent holds.

    2026-09-28 (Cam, after the emb_own/gwt_comms_1 row-norm check came back
    alive, not dead -- RESEARCH_PROTOCOL.md Part 2): distance-to-hearth has
    been flat at the random-walk expectation in every configuration measured
    so far (single-tile, 7x7 stage 0, A0, A1). Vision isn't the problem;
    credit is -- reward for a deposit only arrives after navigation AND
    material-matching both already succeeded, which is far too sparse a
    signal to shape approach behavior on its own. This adds dense gradient on
    exactly the behavior that has never once been learned, the same way
    red_shaping_term already does for red approaching blue.

    Mirrors red_shaping_term's exact form: F_t = beta*(gamma*Phi(s')-Phi(s)),
    Phi(s)=-d(s)/D_max. gamma MUST be the same discount PPO's own GAE uses,
    for the same policy-invariance reason red's shaping requires it.

    Zeroed on the step a deposit happens, for the identical reason red's is
    zeroed on a catch: hearth_need re-rolls on completion, so "nearest
    matching hearth" can teleport to a different, farther hearth the instant
    a deposit lands -- an artifact of the target changing, not of the agent
    moving away from it, which would otherwise claw back part of the very
    reward this term exists to lead into.

    An agent holding nothing, or holding a material no hearth currently
    wants, has no well-defined target hearth: Phi is 0 for it (no artificial
    reward or penalty for an undefined state), matching
    _nearest_alive_blue_dist's any_alive sentinel. held_material is a single
    snapshot used for both the before- and after-move evaluation (only
    position differs between them, matching red's pattern) -- deliberately
    the post-pickup/pre-deposit material, i.e. exactly what
    resolve_hearth_deposits itself judges this same step, so shaping and the
    actual deposit decision are never targeting different things.
    """
    d_max = grid_size // 2

    def _dist_to_matching_hearth(positions):
        diff = jnp.abs(positions[:, None, :] - hearth_positions[None, :, :])  # (N, H, 2)
        diff = jnp.minimum(diff, grid_size - diff)
        dist = jnp.max(diff, axis=-1)  # (N, H), toroidal Chebyshev
        return dist

    matches = held_material[:, None] == hearth_need[None, :]  # (N, H)
    is_valid = jnp.any(matches, axis=1) & (held_material >= 0)  # (N,)

    dist_before = _dist_to_matching_hearth(agent_pos_before)
    dist_after = _dist_to_matching_hearth(agent_pos_after)
    min_dist_before = jnp.min(jnp.where(matches, dist_before, 1e9), axis=1)
    min_dist_after = jnp.min(jnp.where(matches, dist_after, 1e9), axis=1)

    phi_before = jnp.where(is_valid, -min_dist_before / d_max, 0.0)
    phi_after = jnp.where(is_valid, -min_dist_after / d_max, 0.0)
    f_t = beta * (gamma * phi_after - phi_before)
    return jnp.where(made_deposit, 0.0, f_t)


def red_shaping_term(
    red_pos_before: jnp.ndarray,   # (R, 2)
    red_pos_after: jnp.ndarray,    # (R, 2)
    blue_pos_before: jnp.ndarray,  # (B, 2)
    blue_alive_before: jnp.ndarray,  # (B,)
    blue_pos_after: jnp.ndarray,   # (B, 2)
    blue_alive_after: jnp.ndarray,   # (B,)
    made_catch: jnp.ndarray,       # (R,) bool -- this red agent caught a blue this step
    beta: float,
    gamma: float,
    grid_size: int,
) -> jnp.ndarray:
    """Potential-based reward shaping (Ng, Harada & Russell 1999):
    F_t = beta * [gamma * Phi(s') - Phi(s)], Phi(s) = -d(s) / (grid_size // 2).

    gamma MUST be the same discount PPO's own GAE uses (ppo_gamma) -- using a
    different one breaks the policy-invariance guarantee; it is not a free
    parameter here.

    Deliberate, bounded, logged exception to strict potential-based form
    (Cam, 2026-09-14): F_t is zeroed on any step where that red agent made a
    catch. Phi is defined over distance to the *nearest* living blue, so a
    catch removes that blue and d can jump discontinuously to the next-
    nearest, firing a large *negative* shaping value at the exact event the
    ramp exists to encourage -- e.g. adjacent (d=1) with the next-nearest
    blue at d=10 gives F = beta*(1 - gamma*10)/D_max = -0.351 at beta=2.5;
    in a sparse patch with the next-nearest at d=40 it's -1.52, half a catch
    clawed back by the term meant to lead into it. Catch steps are rare, so
    zeroing F_t there leaves the shaping signal on every other step
    essentially unaffected.
    """
    d_max = grid_size // 2
    d_before = _nearest_alive_blue_dist(red_pos_before, blue_pos_before, blue_alive_before, grid_size)
    d_after = _nearest_alive_blue_dist(red_pos_after, blue_pos_after, blue_alive_after, grid_size)
    phi_before = -d_before / d_max
    phi_after = -d_after / d_max
    f_t = beta * (gamma * phi_after - phi_before)
    return jnp.where(made_catch, 0.0, f_t)
