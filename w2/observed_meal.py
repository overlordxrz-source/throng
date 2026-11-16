"""Cam Nov 16 ~07:30Z, M-approved (stage 1 only): the OBSERVED-MEAL label, flag --head-observed-meal (default off).

A hearing record also closes on a bout in which the young SAW the parent at the called item, if the young did not eat
the item itself. Its displacement is the young's own-frame displacement to that cell at the moment it saw it. If the
young eats the item later in the same bout, the own meal wins: one record, one label, per bout.

The world has no parent "eat": the scripted parent walks to the item and stays on its cell, and the item stays for the
young. So the event used here is the one that is fully in the young's own observation (STRICT):
  the parent stands on the called item's cell AND that cell is adjacent to the young (Chebyshev <= 1),
so the item (food channel, visible only when adjacent) and the parent (agent window) are both in this step's obs.
The displacement is computed from own-frame quantities only: the young's displacement from its hearing place (its
input) plus the parent's offset in its view.

The choice is deferred to the bout close (so 'own meal wins' never rewrites a record). An observed record carries
the ate bit = 1, meaning 'a meal was located at this displacement'; this choice is flagged for Cam's ruling."""
import numpy as np
import jax.numpy as jnp

from w2.named_item import torus


def event(dem_on, item_act, dem_pos, item_pos, pos, L):
    """STRICT seen-parent-meal event this step: parent shown, on the called item's cell, the cell adjacent to the young."""
    on = jnp.all(dem_pos == item_pos, -1)
    adj = jnp.abs(torus(dem_pos - pos, L)).max(-1) <= 1
    return dem_on & item_act & on & adj


def own_frame_disp(pos, start, dem_pos, L):
    """Displacement from the hearing place to the parent's cell, from the young's own frame: (own displacement from
    the hearing place) + (the parent's offset in the young's view)."""
    return torus(pos - start, L) + torus(dem_pos - pos, L)


def keep_first(flag, disp, ev, disp_now):
    """Keep the FIRST observed event of a bout."""
    new = ev & ~flag
    return flag | ev, jnp.where(new[:, None], disp_now, disp)


def record(on_item, obs_flag, own_disp, obs_disp):
    """At a bout close (not by death): the record's displacement and ate bit. Own meal wins."""
    use_obs = obs_flag & ~on_item
    disp = jnp.where(use_obs[:, None], obs_disp, own_disp)
    ate = on_item | use_obs
    return disp, ate, use_obs


def sector_of(disp):
    """Sector 0..7 of a displacement (the placement's convention: angle = sec * pi/4, x = cos, y = sin)."""
    a = np.arctan2(np.asarray(disp)[..., 1], np.asarray(disp)[..., 0])
    return (np.round(a / (np.pi / 4)).astype(int)) % 8


def label_mask(hear, closed, ate, dead, obs_close, obs_tgt=None):
    """(B, T) numpy. Generalises learner_w2_kv.own_meal_mask: a hearing step is labelled if its bout's next end within
    the sequence is the agent eating the item (closed & ate) OR, with the flag, an observed-meal close (closed &
    obs_close & ~ate). An end by T_max without an observed meal, by death, or no end in the sequence: unlabelled.
    Also returns, per hearing step, whether its label is the observed one, and (if obs_tgt, the sector of the observed
    displacement at each close step, is given) that observed target carried back to the hearing step (-1 elsewhere)."""
    B_, T_ = hear.shape; cur = np.zeros(B_, bool); cob = np.zeros(B_, bool); mk = np.zeros((B_, T_), bool); mo = np.zeros((B_, T_), bool)
    ot = np.full(B_, -1); tg = np.full((B_, T_), -1); obs_tgt = np.full((B_, T_), -1) if obs_tgt is None else obs_tgt
    for t in range(T_ - 1, -1, -1):
        end = closed[:, t] | dead[:, t]
        own = ate[:, t] & closed[:, t]; obs = obs_close[:, t] & closed[:, t] & ~ate[:, t] & ~dead[:, t]
        cur = np.where(end, own | obs, cur); cob = np.where(end, obs, cob); ot = np.where(end, np.where(obs, obs_tgt[:, t], -1), ot)
        mk[:, t] = hear[:, t] & cur; mo[:, t] = hear[:, t] & cur & cob; tg[:, t] = np.where(mo[:, t], ot, -1)
    return mk, mo, tg
