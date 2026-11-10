"""Kaplan-Meier survival median with right-censoring (Cam Nov 10 ~05:15: lifespan medians are KM from now on)."""
from __future__ import annotations

import numpy as np


def km_median(t, event):
    """t: lifespans; event: True if the life ended (death), False if right-censored (still alive at the horizon).
    Returns (median, reached): the smallest t with S(t) <= 0.5; if S never falls to 0.5, (max t, False): the median
    is then only known to exceed the largest observed time."""
    t = np.asarray(t, float); e = np.asarray(event, bool)
    if t.size == 0:
        return float("nan"), False
    times = np.unique(t[e])
    S = 1.0
    for u in times:
        n_at_risk = np.sum(t >= u); d = np.sum((t == u) & e)
        S *= 1.0 - d / n_at_risk
        if S <= 0.5:
            return float(u), True
    return float(t.max()), False
