"""GAE for the stage-1 trainer, factored out (Nov 13, Cam's GAE/death audit) so it can be unit-tested.
r, v, dn: (..., T, ...) with time on axis 1. dn[t] = 1 if the agent died during step t (terminal: no bootstrap past t).
v_last: the value used to bootstrap after the final step; the stage-1 trainer passes v[:, T-1] (the last state seen;
an approximation of v(s_T)). Mutant switches exist only for the tests: ignore_done, end_terminal."""
import numpy as np


def gae(r, v, dn, gamma, lam, v_last=None, ignore_done=False, end_terminal=False):
    T = r.shape[1]; adv = np.zeros_like(r); last = 0.0
    for t in reversed(range(T)):
        if t + 1 < T:
            nv = v[:, t + 1]
        else:
            nv = (v[:, t] if v_last is None else v_last) * (0.0 if end_terminal else 1.0)
        nt = 1.0 if ignore_done else 1.0 - dn[:, t]
        last = r[:, t] + gamma * nv * nt - v[:, t] + gamma * lam * nt * last; adv[:, t] = last
    return adv
