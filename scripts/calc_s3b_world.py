"""Free calculation for the retuned step-3b world (Cam, Nov 9): blind-search break-even craft cost, NOSIG / informed
delivery levels under scripted policies, and the geometry rule. Pure numpy; no learning, no checkpoint.

World rules copied from comm_toy.py: G x G torus, 4-neighbour moves (1 cell / step), a Craft delivers iff the agent is
within Chebyshev 1 of the lit fire; R_DEP = 2.7 to the deliverer (plus 2.7 shared to the partner); a failed Craft
costs c; the lit fire re-rolls to a DIFFERENT fire on any delivery in the world or after `reroll` steps without one;
the deliverer respawns uniformly over the spawn cells (cells not within 1 of a fire). Episode = 128 steps.

Scripted policies (per agent, independent):
  blind   -- knows nothing about which fire is lit. Tours the fires in a fixed nearest-neighbour cycle starting from
             its nearest fire: walk until adjacent, Craft once, move on to the next fire in the cycle.
  informed-- walks straight to the lit fire and Crafts on arrival (re-targets when the fire re-rolls).
Break-even c* for blind search: an agent's own Craft is worth q * R - (1 - q) * c, q = P(success) per blind Craft,
measured from the blind simulation; c* = R q / (1 - q).

usage: calc_s3b_world.py [--spawn-box x0,y0,x1,y1] [--cost c] [--worlds 4096]
"""
import argparse

import numpy as np

R_DEP = 2.7


def steps_to_adjacent(p, h, G):
    d = np.abs(np.asarray(p) - np.asarray(h)); d = np.minimum(d, G - d)
    return int(max(d[0] - 1, 0) + max(d[1] - 1, 0))


def step_toward(p, h, G):
    """one 4-neighbour move reducing the larger axis gap to the target's Chebyshev-1 neighbourhood"""
    p = np.array(p)
    d = (np.asarray(h) - p + G // 2) % G - G // 2
    ax = 0 if abs(d[0]) - 1 >= abs(d[1]) - 1 else 1
    if abs(d[ax]) > 1:
        p[ax] = (p[ax] + np.sign(d[ax])) % G
    else:
        o = 1 - ax
        if abs(d[o]) > 1:
            p[o] = (p[o] + np.sign(d[o])) % G
    return p


def simulate(G, fires, spawn, reroll, cost, policy, worlds, T, rng):
    F = np.asarray(fires); nF = len(F)
    near = lambda p: np.where(np.minimum(np.abs(F - p), G - np.abs(F - p)).max(1) <= 1)[0]  # noqa: E731
    # nearest-neighbour cycle over the fires (fixed order), each agent starts at its nearest fire in that cycle
    order = [0]
    while len(order) < nF:
        last = order[-1]; rest = [i for i in range(nF) if i not in order]
        order.append(min(rest, key=lambda i: steps_to_adjacent(F[last], F[i], G)))
    dep_w, crafts, succ = np.zeros(worlds), 0, 0
    for w in range(worlds):
        pos = [spawn[rng.integers(len(spawn))].copy() for _ in range(2)]
        lit = int(rng.integers(nF)); since = 0
        tgt = [order.index(min(range(nF), key=lambda i: steps_to_adjacent(pos[k], F[i], G))) for k in range(2)]
        for t in range(T):
            delivered = False
            for k in range(2):
                goal = F[lit] if policy == "informed" else F[order[tgt[k]]]
                if steps_to_adjacent(pos[k], goal, G) == 0 and len(near(pos[k])):
                    crafts += 1
                    if lit in near(pos[k]):
                        succ += 1; dep_w[w] += 1; delivered = True
                        pos[k] = spawn[rng.integers(len(spawn))].copy()
                        tgt[k] = order.index(min(range(nF), key=lambda i: steps_to_adjacent(pos[k], F[i], G)))
                    elif policy == "blind":
                        tgt[k] = (tgt[k] + 1) % nF
                else:
                    pos[k] = step_toward(pos[k], goal, G)
            since += 1
            if delivered or since >= reroll:
                lit = (lit + int(rng.integers(1, nF))) % nF; since = 0
    q = succ / max(crafts, 1)
    return dict(deliveries_per_world=float(dep_w.mean()), crafts_per_world=crafts / worlds, q=q,
                net_reward_per_world=float(dep_w.mean() * 2 * R_DEP - (crafts - succ) / worlds * cost))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--hearths", default="3,3;3,9;9,3;9,9")
    ap.add_argument("--lookout", default="18,18,1")
    ap.add_argument("--reroll", type=int, default=10)
    ap.add_argument("--spawn-box", default="1,1,11,11", help="inclusive x0,y0,x1,y1 (the fire cluster's region)")
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--cost", type=float, default=None, help="craft cost for the reward line (default: c*/2)")
    a = ap.parse_args()
    G = a.grid; F = np.array([[int(v) for v in q.split(",")] for q in a.hearths.split(";")])
    x0, y0, x1, y1 = (int(v) for v in a.spawn_box.split(","))
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    dd = np.abs(cells[:, None, :] - F[None]); dd = np.minimum(dd, G - dd).max(-1)
    valid = cells[(dd > 1).all(1)]
    box = valid[(valid[:, 0] >= x0) & (valid[:, 0] <= x1) & (valid[:, 1] >= y0) & (valid[:, 1] <= y1)]
    lx, ly, lr = (int(v) for v in a.lookout.split(","))
    print(f"grid {G}, fires {F.tolist()}, reroll {a.reroll}, spawn box {a.spawn_box}: {len(box)} spawn cells (of {len(valid)} valid)")
    # geometry rule: from the lookout (any of its cells), walking to the lit fire vs telling a partner in the spawn region
    look = [(lx + i, ly + j) for i in range(-lr, lr + 1) for j in range(-lr, lr + 1)]
    walk = min(steps_to_adjacent(c, f, G) for c in look for f in F[:1])
    walk_all = [min(steps_to_adjacent(c, f, G) for c in look) for f in F]
    told = [[steps_to_adjacent(s, f, G) for f in F] for s in box]
    told = np.array(told)
    print(f"geometry: lookout -> each fire (best lookout cell) {walk_all} steps (+1 Craft); reroll {a.reroll} -> walking "
          f"{'LOSES' if min(walk_all) + 1 > a.reroll else 'can win'}")
    print(f"          a partner anywhere in the spawn region -> lit fire: mean {told.mean():.2f}, max {told.max()} steps; told "
          f"(+1 step wire delay, +1 Craft) mean {told.mean() + 2:.2f}, max {told.max() + 2} -> telling "
          f"{'always makes it' if told.max() + 2 <= a.reroll else 'can miss'} within {a.reroll}")
    chance_occ = 9 / len(valid)
    print(f"          lookout occupancy by chance if agents wandered uniformly over the whole grid: {chance_occ:.4f}; with spawns "
          f"in the box the lookout is >= {min(steps_to_adjacent(s, (lx, ly), G) for s in box)} steps from every spawn cell")
    rng = np.random.default_rng(0)
    bl = simulate(G, F, box, a.reroll, 0.0, "blind", a.worlds, a.steps, rng)
    cstar = R_DEP * bl["q"] / (1 - bl["q"])
    cost = a.cost if a.cost is not None else cstar / 2
    bl = simulate(G, F, box, a.reroll, cost, "blind", a.worlds, a.steps, np.random.default_rng(0))
    inf = simulate(G, F, box, a.reroll, cost, "informed", a.worlds, a.steps, np.random.default_rng(1))
    print(f"\nblind scripted (both agents): q = P(success | blind Craft) = {bl['q']:.4f}; deliveries/world/episode "
          f"{bl['deliveries_per_world']:.2f}; Crafts/world {bl['crafts_per_world']:.1f}")
    print(f"BREAK-EVEN craft cost c* = R q / (1 - q) = {cstar:.3f}  ->  set craft cost = c*/2 = {cstar / 2:.3f}")
    print(f"at cost {cost:.3f}: blind net reward/world/episode {bl['net_reward_per_world']:+.2f}; informed scripted (both "
          f"agents always told) deliveries/world {inf['deliveries_per_world']:.2f}, q {inf['q']:.3f} -> informed / blind = "
          f"{inf['deliveries_per_world'] / max(bl['deliveries_per_world'], 1e-9):.2f}x")
    # the old VoI world, for comparison: spawns anywhere on the 24-grid
    old = simulate(G, F, valid, a.reroll, 0.45, "blind", a.worlds // 4, a.steps, np.random.default_rng(2))
    print(f"(old VoI world, spawns anywhere, cost 0.45: blind q {old['q']:.4f}, deliveries/world {old['deliveries_per_world']:.2f}, "
          f"break-even {R_DEP * old['q'] / (1 - old['q']):.3f})")


if __name__ == "__main__":
    main()
