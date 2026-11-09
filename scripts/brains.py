"""Separate-brains toy: population mechanics (Cam, Nov 9 ~02:40 item 6). Code + unit tests only; not wired into
training until Cam signs off on the registered design (docs/RESEARCH_PROTOCOL.md, "separate-brains toy").

B brains, each with its own parameters and its own Adam state, stacked along a leading brain axis. Every world pairs an
ordered (sender brain, listener brain); the toy's agent layout is kept (agent 2w = world w's sender, 2w+1 = its
listener). Traits (M approved, Nov 9): speech that settles with age (message-head plasticity falls after a sensitive
period; hearing stays plastic) and varied learning speeds (a fixed per-lineage learning-rate multiplier).

Everything here is a pure function of arrays, so each mechanism has a unit test (tests/test_brains.py):
  pair_schedule        balanced, reshuffled ordered pairs per update
  agent_brain_map      world pairs -> per-agent brain ids in the toy layout
  apply_rows           brain-aware forward: each row runs its own brain's parameters (gradients stay inside a brain)
  clip_per_brain       the toy's per-group gradient clip, applied per brain (a global clip would couple brains)
  plasticity           message-head plasticity by age (1 in the sensitive period, then decays to a floor)
  scale_updates        per-brain learning-speed multiplier x (message head only) plasticity, applied to the Adam
                       UPDATES (Adam is invariant to gradient scale, so scaling gradients would do nothing)
  newborn_reset        replace one brain's parameters and Adam state, untouched elsewhere
  elder_rows           the imitation buffer's rows: only sender rows of brains older than the newborn
  msg_grad_mask        sender-rows-only message-head rule per brain (the stop-grad result decides the default)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

MSG_KEYS = ("gwt_comms_1", "head_signal_slot0", "head_signal_slot1", "head_signal_slot2")   # trainable message head
CODEBOOKS = ("codebook_0", "codebook_1", "codebook_2")


def stack(trees):
    """list of B param trees -> one tree with a leading brain axis"""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *trees)


def take(stacked, b):
    """brain b's own tree out of a stacked tree"""
    return jax.tree_util.tree_map(lambda x: x[b], stacked)


def pair_schedule(W: int, B: int, rng: np.random.Generator):
    """(sender_brain[W], listener_brain[W]): every ordered pair (i != j) appears exactly W / (B (B - 1)) times; the
    assignment of pairs to worlds is reshuffled on every call (rotating partners)."""
    pairs = [(i, j) for i in range(B) for j in range(B) if i != j]
    if W % len(pairs):
        raise ValueError(f"W = {W} must be a multiple of B (B - 1) = {len(pairs)}")
    rep = np.array(pairs * (W // len(pairs)), np.int32)
    rep = rep[rng.permutation(W)]
    return rep[:, 0], rep[:, 1]


def agent_brain_map(sender_b, listener_b):
    """toy layout: agent 2w is world w's sender, 2w + 1 its listener -> (N = 2W,) brain ids"""
    return np.stack([sender_b, listener_b], 1).reshape(-1).astype(np.int32)


def apply_rows(apply_fn, stacked, brain_ids, carries, obs):
    """row r runs brain brain_ids[r]'s parameters. Every brain runs the whole batch and each row keeps its own brain's
    output (B-way compute-and-select: 2.7x a plain forward + gradient at 512 rows, vs 8.7x for a per-row vmap). Only
    the selected outputs reach the loss, so the gradient w.r.t. `stacked` of anything computed from rows of brain A is
    exactly zero on every other brain."""
    from types import SimpleNamespace
    ids = jnp.asarray(brain_ids); rows = jnp.arange(ids.shape[0])
    per = [apply_fn(take(stacked, b), carries, obs) for b in range(jax.tree_util.tree_leaves(stacked)[0].shape[0])]
    new_c = jnp.stack([c for c, _ in per])[ids, rows]
    dicts = [o if isinstance(o, dict) else dict(vars(o)) for _, o in per]   # the toy's Outs is a plain attribute bag
    out = {k: (None if dicts[0][k] is None else
               jax.tree_util.tree_map(lambda *xs: jnp.stack(xs)[ids, rows], *[d[k] for d in dicts])) for k in dicts[0]}
    return new_c, SimpleNamespace(**out)


def _clip_group(g, max_norm):
    norm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    s = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda x: x * s, g), norm


def clip_per_brain(grads_stacked, groups, max_norm=2.0):
    """the toy's per-group clip (head_action / head_value / rest, max norm 2.0), computed separately for each brain.
    groups: list of lists of top-level param keys. Returns (clipped grads, per-brain per-group norms (B, G))."""
    g = grads_stacked["params"]

    def per_brain(gb):
        out, norms = dict(gb), []
        for grp in groups:
            c, n = _clip_group({k: gb[k] for k in grp}, max_norm)
            out.update(c); norms.append(n)
        return out, jnp.stack(norms)
    out, norms = jax.vmap(per_brain)(g)
    return {"params": out}, norms


def plasticity(age, sensitive: int, decay: int, floor: float):
    """message-head plasticity multiplier: 1.0 for age < sensitive, then linear to `floor` over `decay` updates, then
    floor. Hearing (every non-message parameter) is not scaled."""
    age = np.asarray(age, np.float64)
    frac = np.clip((age - sensitive) / max(decay, 1), 0.0, 1.0)
    return 1.0 - (1.0 - floor) * frac


def scale_updates(updates_stacked, lr_mult, msg_mult, msg_keys=MSG_KEYS):
    """per-brain multipliers on the Adam UPDATES: every parameter x lr_mult[b] (learning speed); the message head
    additionally x msg_mult[b] (age plasticity)."""
    lr = jnp.asarray(lr_mult, jnp.float32); mm = jnp.asarray(msg_mult, jnp.float32)
    u = dict(updates_stacked["params"])

    def sc(x, m):
        return x * m.reshape((-1,) + (1,) * (x.ndim - 1))
    for k in u:
        m = lr * mm if k in msg_keys else lr
        u[k] = jax.tree_util.tree_map(lambda x: sc(x, m), u[k])
    return {"params": u}


def newborn_reset(stacked, opt_stacked, b: int, new_params, fresh_opt_state):
    """brain b gets new parameters and a fresh optimizer state; every other brain is untouched (bit-identical)."""
    put = lambda S, x: S.at[b].set(x)  # noqa: E731
    return (jax.tree_util.tree_map(put, stacked, new_params),
            jax.tree_util.tree_map(put, opt_stacked, fresh_opt_state))


def elder_rows(brain_ids, is_sender, ages, newborn: int):
    """rows the newborn may imitate: SENDER rows of brains strictly older than the newborn (never its own, never a
    younger brain's, never a listener row)."""
    brain_ids = np.asarray(brain_ids); ages = np.asarray(ages)
    older = ages[brain_ids] > ages[newborn]
    return np.asarray(is_sender, bool) & older & (brain_ids != newborn)


def msg_grad_mask(is_sender, rule: str):
    """row weights for the message-head gradient: 'all' = every row (the shared-brain default), 'sender' = sender rows
    only (stop-grad). With separate brains a brain's listener rows still train its OWN message head unless 'sender'."""
    s = np.asarray(is_sender, np.float32)
    if rule == "all":
        return np.ones_like(s)
    if rule == "sender":
        return s
    raise ValueError(rule)


def opt_init_per_brain(opt, stacked):
    """one independent Adam state per brain (vmapped init: each brain gets its own step count and moments)"""
    return jax.vmap(opt.init)(stacked)


def opt_update_per_brain(opt, grads_stacked, opt_stacked, params_stacked):
    """vmapped Adam update: identical to running each brain's own optimizer separately (a plain update on the stacked
    tree would share one step count's bias correction across the brain axis by broadcasting)"""
    return jax.vmap(opt.update)(grads_stacked, opt_stacked, params_stacked)
