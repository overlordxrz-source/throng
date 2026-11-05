"""EV ceiling for a critic that cannot see time-in-rollout, against the testbed's truncated-MC targets.
Deposits as a Bernoulli process at the observed per-step rate (state carries no information about far-future
deposits -- the generous case is a critic that knows the NEXT k steps exactly). gamma .999, T 512, reward 2.7."""
import numpy as np
rng = np.random.default_rng(0)
T, N, g, R = 512, 20000, 0.999, 2.7
for per_upd in (1.0, 3.6, 10.6, 60.0):      # deposits per env per 512 steps (E-level, A1, A0, C-level)
    p = per_upd / T
    r = (rng.random((T, N)) < p) * R
    G = np.zeros((T, N)); c = np.zeros(N)
    for t in range(T - 1, -1, -1):
        c = r[t] + g * c; G[t] = c
    var = G.var()
    t_mean = G.mean(1, keepdims=True)                     # best predictor that knows only t
    ev_t = 1 - ((G - t_mean) ** 2).mean() / var
    # t-blind critic that knows the next 20 steps' rewards exactly (generous state information), constant otherwise
    k = 20
    near = np.zeros_like(G); c = np.zeros(N)
    for t in range(T - 1, -1, -1):
        c = r[t] + g * c - (g ** k) * (r[t + k] if t + k < T else 0); near[t] = c
    far_mean = (G - near).mean()                          # t-blind: one constant for the rest
    ev_blind = 1 - ((G - near - far_mean) ** 2).mean() / var
    print(f"{per_upd:5.1f} deposits/rollout: EV(knows t only) {ev_t:.3f} | EV(t-blind, knows next {k} steps) {ev_blind:.3f}")
