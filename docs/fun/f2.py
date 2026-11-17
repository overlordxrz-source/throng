"""FUN F2 analysis: second hearing of each token in a life (registered restriction). Heading = sector of (pos - start)
at the first bout step with Chebyshev >= 3. Implanted sector = true + R / 45 (mod 8). Also: the record edited was
solved? = the first (parent-led) bout of that token in this life ended by eating."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2"); import traj_analysis as TA
def run(f, R):
    z = dict(np.load(f)); Wn, T, N = z["hear"].shape; rows = []
    for w in range(Wn):
        for n in range(N):
            hear = z["hear"][w, :, n]; close = z["cl_item"][w, :, n] | z["cl_tmax"][w, :, n] | z["cl_death"][w, :, n]
            dead = z["dead"][w, :, n]; cnt = {}; solved1 = {}; last_dead = -1
            for s0 in np.flatnonzero(hear):
                if dead[last_dead + 1: s0].any():
                    last_dead = int(np.flatnonzero(dead[:s0])[-1]); cnt = {}; solved1 = {}
                ends = np.flatnonzero(close[s0:])
                if len(ends) == 0:
                    continue
                s1 = s0 + ends[0]; sec = int(z["sec"][w, s0, n]); k = cnt.get(sec, 0) + 1; cnt[sec] = k
                if k == 1:
                    solved1[sec] = bool(z["cl_item"][w, s1, n]); continue
                if k != 2 or not z["impl"][w, s0, n] or z["cl_death"][w, s1, n]:
                    continue
                st = z["start"][w, s0, n].astype(int); hd = None
                for s in range(s0, s1 + 2 if s1 + 1 < T else s1 + 1):
                    o = TA.torus(z["pos"][w, s, n].astype(int) - st)
                    if np.abs(o).max() >= 3:
                        hd = TA.sector_of(o); break
                rows.append(dict(sec=sec, imp=(sec + R // 45) % 8, head=int(z["hl"][w, s0, n]), hd=hd, ate=bool(z["cl_item"][w, s1, n]), solved1=solved1.get(sec), nrec=int(z["nrec"][w, s0, n])))
    n = len(rows); f_ = lambda c: round(sum(c(r) for r in rows) / max(n, 1), 3)  # noqa: E731
    return dict(R=R, n=n, heading_implanted=f_(lambda r: r["hd"] == r["imp"]), heading_true=f_(lambda r: r["hd"] == r["sec"]), heading_none=f_(lambda r: r["hd"] is None),
                head_vs_true=f_(lambda r: r["head"] == r["sec"]), head_vs_implanted=f_(lambda r: r["head"] == r["imp"]), reach=f_(lambda r: r["ate"]),
                edited_record_solved=f_(lambda r: bool(r["solved1"])), mean_records_edited=round(np.mean([r["nrec"] for r in rows]), 2) if rows else None)
print(json.dumps([run(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)], indent=1))
