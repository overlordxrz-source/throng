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


CRAFT_RAMP_STAGE_UNITS = (1, 2)  # stage 0: solo-satisfiable; stage 1: pair-satisfiable


def staged_recipe_counts(key: jax.Array, stage: jnp.ndarray) -> jnp.ndarray:
    """Recipe for the crafting ramp's two capped stages (Cam's correction,
    2026-09-14: craft_ramp_max_units=2 alone was still a cooperative problem
    at smaller scale, not the solo-catchable analogue -- max_units=1 was
    missing entirely).

    Stage 0 draws exactly 1 material unit -- solo-satisfiable, because
    resolve_crafting's adjacency check counts an agent as adjacent to
    itself, so one agent holding the one correct material succeeds alone.
    Stage 1 draws exactly 2 -- pair-satisfiable, the old single-stage
    behaviour. Neither wastes a slot (unlike the full recipe draw in
    jax_sim/main_jax.py's update_recipe), so sum(counts) is exactly 1 or 2.

    Both slots are drawn unconditionally regardless of `stage`, and only the
    *values* are masked by stage -- the traced shape never depends on
    `stage`, so advancing stages doesn't retrace/recompile update_recipe.
    """
    items = jax.random.randint(key, (2,), 0, 5)
    counts_stage0 = jnp.bincount(items[:1], length=5).astype(jnp.int32)
    counts_stage1 = jnp.bincount(items, length=5).astype(jnp.int32)
    return jnp.where(stage == 0, counts_stage0, counts_stage1)


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
