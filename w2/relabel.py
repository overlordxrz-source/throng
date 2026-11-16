"""Cam Nov 16 ~10:15Z, M-approved Nov 16 10:46Z (stage 1 only): CROSS-HEARING RELABEL, flag --head-relabel-token (default off).

The head target at EVERY hearing of token X in a life = the sector of the young's own displacement in a SOLVED record of X
from that life (a record whose ate bit is set: the called item was eaten). Earlier or later hearing:
  - the most recent solved record of X written before the hearing (in memory at that step), else
  - the earliest solved record of X written at or after the hearing, before any death, within the training sequence.
Masked if X has no solved record that life (within what the sequence can see). Labels never cross a death.

Inputs are ONLY the young's own: the heard token (its hearing input, x[100:108]), its record buffer (the memory at the
sequence start + the records it writes during the sequence) and its deaths. This module never sees the world's code table
or the item's sector. The 'is this step a hearing' mask is the same one the own-meal label already uses."""
import numpy as np

S_REC = 11


def sector_of(disp):
    """Sector 0..7 of a displacement (placement convention: angle = sec * pi/4, x = cos, y = sin)."""
    d = np.asarray(disp, dtype=np.float64)
    return (np.round(np.arctan2(d[..., 1], d[..., 0]) / (np.pi / 4)).astype(int)) % 8


def _rec(r):
    """(token, sector, solved) of one raw record [token one-hot 8, displacement / 24 (2), ate]."""
    return int(np.argmax(r[:8])), int(sector_of(np.round(r[8:10] * 24.0))), bool(r[10] > 0.5)


def initial_records(mem_flat, ptr, S):
    """Records in memory at the sequence start, most recent first (ring buffer written at ptr - 1, ptr - 2, ...)."""
    mem = np.asarray(mem_flat).reshape(S, S_REC); out = []
    p = int(ptr) % S
    for j in range(S):
        r = mem[(p - 1 - j) % S]
        if r[:8].sum() > 0:
            out.append(_rec(r))
    return out


def relabel(hear, tok_onehot, mem0, ptr0, wflag, wrec, dead, S):
    """hear (B, T) bool: hearing steps. tok_onehot (B, T, 8): the heard token (the young's own input). mem0 (B, S*11) and
    ptr0 (B,): the record buffer at the sequence start. wflag (B, T) bool / wrec (B, T, 11): the record written at step t
    (available from t + 1). dead (B, T) bool: a death at step t wipes the buffer after it.
    Returns target (B, T) int (-1 where masked), mask (B, T) bool, source (B, T) int (0 none, 1 earlier, 2 later),
    and stats dict(n_hear, n_labelled, n_earlier, n_later, n_disagree)."""
    hear = np.asarray(hear, bool); B, T = hear.shape; wflag = np.asarray(wflag, bool); dead = np.asarray(dead, bool)
    tgt = np.full((B, T), -1); src = np.zeros((B, T), int); st = dict(n_hear=int(hear.sum()), n_labelled=0, n_earlier=0, n_later=0, n_disagree=0)
    for b in range(B):
        recs = initial_records(mem0[b], ptr0[b], S)[::-1]          # oldest first; append new ones at the end
        wr = [(_rec(wrec[b, t]) if wflag[b, t] else None) for t in range(T)]
        for t in range(T):
            if hear[b, t]:
                X = int(np.argmax(tok_onehot[b, t])); before = [r[1] for r in recs if r[0] == X and r[2]]
                after = None
                for t2 in range(t, T):                               # earliest solved record of X at/after t, before a death
                    if wr[t2] is not None and wr[t2][0] == X and wr[t2][2]:
                        after = wr[t2][1]; break
                    if dead[b, t2]:
                        break
                if before:
                    tgt[b, t] = before[-1]; src[b, t] = 1; st["n_earlier"] += 1
                    if len(set(before)) > 1 or (after is not None and after != before[-1]):
                        st["n_disagree"] += 1
                elif after is not None:
                    tgt[b, t] = after; src[b, t] = 2; st["n_later"] += 1
            if wr[t] is not None:
                recs.append(wr[t])
            if dead[b, t]:
                recs = []
    mask = tgt >= 0; st["n_labelled"] = int(mask.sum())
    return tgt, mask, src, st
