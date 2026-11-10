"""Free check (Cam Nov 10 ~00:35): does MATERIAL matter to the step-6 listener? Scripted listeners in the exact step-6
world rules (comm_toy.py --comp): G 12 torus, 4 fires at generate_hearth_positions(12), pair q = fire * 4 + material,
a delivery needs a Craft within Chebyshev 1 of the lit fire WITH the material's craft action (CRAFT_ACTS[material]);
every failed Craft costs 0.45; R_DEP 2.7 per delivery; the pair re-rolls to a different training pair on a delivery
or after --reroll steps (default 96); a deliverer respawns uniformly over cells not within 1 of a fire.
  FULL       knows fire and material: walks to the fire, Crafts once with the right material.
  FIRE-ONLY  knows the fire only: walks there, then tries the 4 materials in a random order until one delivers
             (each miss costs 0.45 and a step).
Reports deliveries and net reward per world per 128 steps; GAP = 1 - fire-only / full deliveries (Cam: < 25% -> the
world gives no reason to name material).
"""
import numpy as np

G, T, W, R_DEP, COST = 12, 128, 4096, 2.7, 0.45
HEARTHS = np.array([[3, 3], [3, 9], [9, 3], [9, 9]])
HELD = (1, 11)                      # the step-6 held-out pairs (comm_toy --held-out default; training pool = the other 14)
POOL = [q for q in range(16) if q not in HELD]


def steps_adj(p, h):
    d = np.abs(p - h); d = np.minimum(d, G - d)
    return max(d[0] - 1, 0) + max(d[1] - 1, 0)


def step_toward(p, h):
    p = p.copy(); d = (h - p + G // 2) % G - G // 2
    ax = 0 if abs(d[0]) - 1 >= abs(d[1]) - 1 else 1
    if abs(d[ax]) > 1:
        p[ax] = (p[ax] + np.sign(d[ax])) % G
    elif abs(d[1 - ax]) > 1:
        p[1 - ax] = (p[1 - ax] + np.sign(d[1 - ax])) % G
    return p


def run(mode, reroll=96, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    dd = np.abs(cells[:, None] - HEARTHS[None]); dd = np.minimum(dd, G - dd).max(-1)
    valid = cells[(dd > 1).all(1)]
    dep = np.zeros(W); rew = np.zeros(W)
    for w in range(W):
        pos = valid[rng.integers(len(valid))].copy(); q = POOL[rng.integers(len(POOL))]; since = 0
        tried = []; order = list(rng.permutation(4))
        for t in range(T):
            fire, mat = q // 4, q % 4; h = HEARTHS[fire]
            delivered = False
            if steps_adj(pos, h) == 0:
                m = mat if mode == "full" else order[len(tried)]
                if m == mat:
                    dep[w] += 1; rew[w] += R_DEP; delivered = True
                    pos = valid[rng.integers(len(valid))].copy()
                else:
                    rew[w] -= COST; tried.append(m)
            else:
                pos = step_toward(pos, h)
            since += 1
            if delivered or since >= reroll:
                q = [x for x in POOL if x != q][rng.integers(len(POOL) - 1)]; since = 0
                tried = []; order = list(rng.permutation(4))
    return dep.mean(), rew.mean()


if __name__ == "__main__":
    full = run("full"); fo = run("fire_only")
    print(f"FULL      (fire + material): deliveries/world/128 steps {full[0]:.2f}, net reward {full[1]:+.2f}")
    print(f"FIRE-ONLY (fire only)      : deliveries/world/128 steps {fo[0]:.2f}, net reward {fo[1]:+.2f}")
    gap = 1 - fo[0] / full[0]; rgap = 1 - fo[1] / full[1]
    print(f"GAP in deliveries = {gap:.1%} (Cam's line: 25%) -> {'REAL: rerun with material on fresh input rows' if gap >= 0.25 else 'SMALL: fix the world first'}"
          f" | gap in net reward {rgap:.1%}")
