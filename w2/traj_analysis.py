"""Cam Nov 16 ~17:05Z, policy-side diagnosis D1 / D3 / D4 / D5 from learner_w2_rl_traj.py's per-step records.

Bouts: a bout starts at a hearing step s0 (hear) and ends at its closing step s1 (cl_item | cl_tmax | cl_death). At step s
the record holds the state BEFORE the action a[s]; pos[s + 1] is the position after it.
"Later" bouts = not demonstrated and not a first hearing (K = 1). "Head right" = the head's argmax at s0 == the item's
sector (from the bout start).
Sectors: angle = sec * 45 deg, x = cos, y = sin (placement convention); sector_of(v) = round(atan2(y, x) / 45 deg) mod 8.
A move (a in 1..4, MOVES) "points into" sector k if the angle between the move and sector k's direction is <= 45 deg
(one move for a cardinal sector, two for a diagonal one).

D1  per bout step k = 0..12, at each d: head accuracy against the TRUE sector from the young's CURRENT position, against
    the start sector, and the share of moves pointing into the head's argmax sector (and into the true current sector).
D3  right-head later bouts that time out, flags (overlapping) and an exclusive class in Cam's order:
    (i) never toward: the projection of (pos - start) on the sector direction never reaches 1 cell;
    (ii) arc: within Chebyshev 1 of a cell of the item's arc (cells at distance d +- 0.5 from the start, in the item's
         sector) but the item not eaten; also reported: the item itself came within Chebyshev 1 (visible);
    (iii) generic food eaten during the bout;
    (iv) dithers: reversals >= 30% of consecutive move pairs, or < 25% of the steps are moves;
    plus the minimum Chebyshev distance to the item. For an oracle file every bout counts (no head, no demonstrator).
D4  right-head later bouts: reach by item sector, and by whether the young's previous heading (its last move before the
    hearing, same life) is within 45 deg of the item sector; P(head argmax within 45 deg of the previous heading).
D5  head accuracy at later hearings by d, and by whether the item was visible (Chebyshev <= 1) at the hearing.
"""
import argparse, json
import numpy as np

L = 96
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])
U = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], -1)
COS45 = np.cos(np.pi / 4) - 1e-6


def torus(o):
    return (o + L // 2) % L - L // 2


def sector_of(v):
    return int(np.round(np.arctan2(v[1], v[0]) / (np.pi / 4))) % 8


def into(move, k):
    m = MOVES[move].astype(float)
    return bool(m.any() and (m @ U[k]) / np.linalg.norm(m) >= COS45)


_ARC = {}


def arc(sec, d):
    if (sec, d) not in _ARC:
        cells = [(x, y) for x in range(-d - 2, d + 3) for y in range(-d - 2, d + 3)
                 if abs(np.hypot(x, y) - d) <= 0.5 and sector_of((x, y)) == sec]
        _ARC[(sec, d)] = np.array(cells)
    return _ARC[(sec, d)]


def bouts(z):
    """Yield one dict per closed bout."""
    Wn, T, N = z["hear"].shape
    for w in range(Wn):
        for n in range(N):
            hear = z["hear"][w, :, n]; close = z["cl_item"][w, :, n] | z["cl_tmax"][w, :, n] | z["cl_death"][w, :, n]
            starts = np.flatnonzero(hear)
            for s0 in starts:
                ends = np.flatnonzero(close[s0:])
                if len(ends) == 0:
                    continue
                s1 = s0 + ends[0]
                yield dict(w=w, n=n, s0=s0, s1=s1)


def analyse(f, oracle=False):
    z = dict(np.load(f)); out = {}
    D1 = {}; D3 = {}; D4 = {"by_sec": {}, "head_match": [0, 0], "match": [0, 0], "nomatch": [0, 0], "unknown": [0, 0]}; D5 = {}
    for b in bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
        g = lambda k, s: z[k][w, s, n]  # noqa: E731
        d = int(g("d", s0)); sec = int(g("sec", s0)); start = g("start", s0).astype(int); ipos = g("ipos", s0).astype(int)
        later = oracle or (not g("dem", s0) and not g("first", s0)); hr = oracle or int(g("hl", s0)) == sec
        ate = bool(g("cl_item", s1)); tmax = bool(g("cl_tmax", s1)); death = bool(g("cl_death", s1))
        vis0 = np.abs(torus(ipos - g("pos", s0).astype(int))).max() <= 1
        if not oracle and later:                                            # D5
            e = D5.setdefault(f"d{d}_{'vis' if vis0 else 'unseen'}", [0, 0]); e[0] += int(int(g("hl", s0)) == sec); e[1] += 1
        if not later:
            continue
        if hr and not oracle:                                               # D1
            for k in range(0, min(13, s1 - s0 + 1)):
                s = s0 + k; pos = g("pos", s).astype(int); rel = torus(ipos - pos)
                e = D1.setdefault(f"d{d}", {}).setdefault(k, dict(n=0, cur=0, start=0, nmove=0, into_head=0, into_cur=0))
                hk = int(g("hl", s)); e["n"] += 1
                if rel.any():
                    e["cur"] += int(hk == sector_of(rel))
                e["start"] += int(hk == sec)
                a = int(g("a", s))
                if 1 <= a <= 4:
                    e["nmove"] += 1; e["into_head"] += int(into(a, hk)); e["into_cur"] += int(rel.any() and into(a, sector_of(rel)))
        if not oracle:                                                      # D4 (true and wiped files alike)
            hd = None
            for s in range(s0 - 1, max(s0 - 200, -1), -1):                  # the previous heading: last move before the hearing, same life
                if g("dead", s):
                    break
                a = int(g("a", s))
                if 1 <= a <= 4:
                    hd = a; break
            if hd is not None:
                D4["head_match"][0] += int(into(hd, int(g("hl", s0)))); D4["head_match"][1] += 1
            if hr:
                e = D4["by_sec"].setdefault(sec, [0, 0]); e[0] += int(ate); e[1] += 1
                key = "unknown" if hd is None else ("match" if into(hd, sec) else "nomatch")
                D4[key][0] += int(ate); D4[key][1] += 1
        if hr:
            if tmax:                                                        # D3
                P = [g("pos", s).astype(int) for s in range(s0, s1 + 1)]
                if s1 + 1 < z["pos"].shape[1]:
                    P.append(z["pos"][w, s1 + 1, n].astype(int))
                offs = np.array([torus(p - start) for p in P]); A = np.array([int(g("a", s)) for s in range(s0, s1 + 1)])
                mind = int(min(np.abs(torus(ipos - p)).max() for p in P))
                never = (offs @ U[sec]).max() < 1.0
                ac = arc(sec, d); onarc = bool((np.abs(offs[:, None, :] - ac[None]).max(-1) <= 1).any())
                vis = mind <= 1; gen = bool(g("gen", s1))
                mv = A[(A >= 1) & (A <= 4)]; rev = sum(1 for x, y in zip(mv[:-1], mv[1:]) if (MOVES[x] + MOVES[y] == 0).all())
                dith = (len(mv) < 0.25 * len(A)) or (len(mv) > 1 and rev >= 0.3 * (len(mv) - 1))
                cls = "i" if never else "ii" if onarc else "iii" if gen else "iv" if dith else "other"
                e = D3.setdefault(f"d{d}", dict(n=0, i=0, ii=0, iii=0, iv=0, vis=0, excl={}, mind=[]))
                e["n"] += 1; e["i"] += int(never); e["ii"] += int(onarc); e["iii"] += int(gen); e["iv"] += int(dith); e["vis"] += int(vis)
                e["excl"][cls] = e["excl"].get(cls, 0) + 1; e["mind"].append(mind)
    for v in D3.values():
        m = np.array(v.pop("mind")); v["mind_hist"] = {int(k): int((m == k).sum()) for k in np.unique(m)}; v["mind_mean"] = float(m.mean())
    out = dict(file=f, D1=D1, D3=D3, D4=D4 if not oracle else None, D5=D5 if not oracle else None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--oracle", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump([analyse(f, a.oracle) for f in a.files], open(a.out, "w"), indent=1, default=int)
    print("WROTE", a.out)
