"""Cam Nov 17 ~19:10Z G2: the LR anneal rule (shared by port_w2 / learner_w2_rl2).
LR = lr0 until START = max(after, t24), where t24 = the step at which the frontier first reached `frontier` (24); then linear
lr0 -> lr_end over `length` steps; then lr_end. While the frontier has not reached 24 (t24 None) the LR stays lr0.
The schedule is applied as a scale on the Adam update (update * lr / lr0), which equals running Adam at that LR."""


def anneal_start(t24, after):
    return None if t24 is None else max(float(after), float(t24))


def lr_at(step, t24, lr0=3e-4, lr_end=3e-5, after=5e7, length=5e7):
    st = anneal_start(t24, after)
    if st is None or step < st:
        return lr0
    return lr0 + (lr_end - lr0) * min(1.0, (step - st) / length)


def t24_from_curve(curve, frontier=24):
    """For checkpoints written without t24: the first logged curve step with frontier >= `frontier` (an upper bound on the
    true t24 by at most one curve interval), else None."""
    for c in curve:
        if c["frontier"] >= frontier:
            return float(c["steps"])
    return None
