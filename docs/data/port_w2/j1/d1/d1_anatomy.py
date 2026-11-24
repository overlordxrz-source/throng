"""Cam Nov 23 ~17:10Z item 2, D1 + D3 (eval only): seed 10 at 1e8 (F3 learner), from learner_w2_rl_traj.py npz files made in
the diag worktree (offset-proj abed942 + --head-probs + --mem-audit, commit 9bd3dc5) on the laptop.

  d1_anatomy.py d1 FILE_m0.npz [FILE_m0.npz ...]   -> D1 a-d per file (later bouts: not demonstrated, not a first hearing)
  d1_anatomy.py d3 FILE_m3.npz FILE_m4.npz          -> D3 split of the control modes' reaches
  d1_anatomy.py ctl OFF_m0.npz AUDIT_m0.npz JSON    -> controls: flag-off == audit run on every shared field (bitwise);
                                                       reach / 2nd-half / hungry-2nd-half rebuilt from the traj == the eval JSON

Memory audit (the head's input memory at the hearing step, h[HG : HG + S*11], read-only): for the heard token, solved
(ate = 1) and unsolved (ate = 0) records counted by stored label (sector of the stored displacement; 8 = zero
displacement), and the most recent matching record (solved, label, age in records, |disp|_inf). In mode 0 the word for a
sector is fixed for a life, so "same word" = same (world, agent, life, sector)."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2"); import traj_analysis as TA  # noqa: E402

HALF = 1500; R = 1   # the item is visible only within Chebyshev 1 (named_item.observe: |OFF|_inf <= 1)


def ages(z):
    """age after each step (= s2.age, the eval's age_close), rebuilt from the death flags: a life starts at step 0 or the
    step after a death; age_close(t) = t - start + 1."""
    dead = z["dead"]; W, T, N = dead.shape; age = np.zeros((W, T, N), np.int32)
    for w in range(W):
        for n in range(N):
            st = 0
            for t in range(T):
                age[w, t, n] = t - st + 1
                if dead[w, t, n]:
                    st = t + 1
    return age


def lives(z):
    dead = z["dead"].astype(np.int32); lid = np.cumsum(dead, 1) - dead          # life index of each step (a death ends that life)
    return lid


def rel(a, b):
    k = (int(a) - int(b)) % 8
    return "same" if k == 0 else ("neighbour" if k in (1, 7) else ("opposite" if k == 4 else "other"))


def d1(f):
    z = dict(np.load(f)); lid = lives(z); out = dict(file=f.split("/")[-1])
    rows = []
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]
        if z["dem"][w, s0, n] or z["first"][w, s0, n]:
            continue
        sec = int(z["sec"][w, s0, n]); hl = int(z["hl"][w, s0, n]); ns = z["ma_ns"][w, s0, n].astype(int); nu = z["ma_nu"][w, s0, n].astype(int)
        ls, ll, la, ld = (int(v) for v in z["ma_last"][w, s0, n])
        rows.append(dict(key=(w, n, int(lid[w, s0, n]), sec), s0=s0, sec=sec, hl=hl, ok=hl == sec, ate=bool(z["cl_item"][w, s1, n]),
                         ns=ns, nu=nu, nsol=int(ns.sum()), nuns=int(nu.sum()), last=(ls, ll, la, ld),
                         p_true=float(z["hpf"][w, s0, n][sec]) if "hpf" in z else None, tok=int(z["ma_tok"][w, s0, n])))
    nL = len(rows); miss = [r for r in rows if not r["ok"]]; hit = [r for r in rows if r["ok"]]
    out["n_later"] = nL; out["p_right"] = round(len(hit) / nL, 4); out["n_miss"] = len(miss)
    out["reach_right"] = round(np.mean([r["ate"] for r in hit]), 4); out["reach_wrong"] = round(np.mean([r["ate"] for r in miss]), 4) if miss else None
    # control: a solved record's stored label is the item's true sector by construction (disp = item - start when eaten)
    off = sum(int(r["ns"].sum() - r["ns"][r["sec"]]) for r in rows); tot = sum(r["nsol"] for r in rows)
    out["CTL_solved_label_off_true"] = f"{off} of {tot} solved matching records"
    # a. wrong-sector type
    out["a_miss_type"] = {k: sum(rel(r["hl"], r["sec"]) == k for r in miss) for k in ("neighbour", "opposite", "other")}
    # b. clustering by word (same world, agent, life, sector)
    G = {}
    for r in sorted(rows, key=lambda r: r["s0"]):
        G.setdefault(r["key"], []).append(r["ok"])
    multi = [g for g in G.values() if len(g) >= 2]; pm = len(miss) / nL
    rep = [(g[i], g[i + 1]) for g in G.values() for i in range(len(g) - 1)]
    after_miss = [nx for pr, nx in rep if not pr]; after_hit = [nx for pr, nx in rep if pr]
    gm = [g for g in multi if not all(g)]
    out["b_cluster"] = dict(words=len(G), words_2plus=len(multi), words_with_miss=sum(not all(g) for g in G.values()),
                            miss_share_in_words_all_missed=round(sum(len(g) for g in G.values() if not any(g)) / max(len(miss), 1), 3),
                            p_miss_next_given_miss=round(1 - np.mean(after_miss), 3) if after_miss else None, n_after_miss=len(after_miss),
                            p_miss_next_given_hit=round(1 - np.mean(after_hit), 3) if after_hit else None, base_miss=round(pm, 4),
                            miss_per_missing_word_2plus=round(np.mean([sum(not x for x in g) / len(g) for g in gm]), 3) if gm else None)
    # c. memory audit at each miss (and at hits, for comparison)
    def cat(r):
        if r["nsol"] > 0:
            return "solved_record_present"
        return "only_unsolved_records" if r["nuns"] > 0 else "no_record"
    C = {}
    for nm, rs in (("miss", miss), ("hit", hit)):
        e = {}
        for r in rs:
            c = cat(r); q = e.setdefault(c, dict(n=0, ate=0, hl_eq_last_label=0, last_dinf=[], last_age=[], nuns=[]))
            q["n"] += 1; q["ate"] += r["ate"]; q["hl_eq_last_label"] += int(r["last"][1] == r["hl"]); q["last_dinf"].append(r["last"][3]); q["last_age"].append(r["last"][2]); q["nuns"].append(r["nuns"])
        C[nm] = {c: dict(n=q["n"], share=round(q["n"] / max(len(rs), 1), 3), reach=round(q["ate"] / q["n"], 3), head_eq_latest_record_label=round(q["hl_eq_last_label"] / q["n"], 3),
                         latest_dinf_median=float(np.median(q["last_dinf"])), latest_age_median=float(np.median(q["last_age"])), n_unsolved_median=float(np.median(q["nuns"])))
                 for c, q in sorted(e.items())}
    out["c_audit"] = C
    # for misses with only unsolved records: does the head follow an unsolved record's label (the latest, or any)?
    ou = [r for r in miss if cat(r) == "only_unsolved_records"]
    out["c_only_unsolved_miss"] = dict(n=len(ou), head_eq_some_unsolved_label=sum(int(r["nu"][r["hl"]] > 0) for r in ou),
                                       type={k: sum(rel(r["hl"], r["sec"]) == k for r in ou) for k in ("neighbour", "opposite", "other")})
    sp = [r for r in miss if cat(r) == "solved_record_present"]
    out["c_misread_with_solved"] = dict(n=len(sp), type={k: sum(rel(r["hl"], r["sec"]) == k for r in sp) for k in ("neighbour", "opposite", "other")},
                                        nsol_median=float(np.median([r["nsol"] for r in sp])) if sp else None,
                                        also_unsolved_with_head_label=sum(int(r["nu"][r["hl"]] > 0) for r in sp))
    # d. P(right) by memory state (for the cross-d comparison: is the d-dependence carried by the mix or within a state?)
    out["d_by_state"] = {c: dict(n=sum(cat(r) == c for r in rows), share=round(sum(cat(r) == c for r in rows) / nL, 4),
                                 p_right=round(np.mean([r["ok"] for r in rows if cat(r) == c]), 4) if any(cat(r) == c for r in rows) else None)
                         for c in ("solved_record_present", "only_unsolved_records", "no_record")}
    return out


def d3(f):
    """Each closed bout of a control-mode run (later in life, hungry, = the bar's population; and all bouts), by how an
    eaten item was found: first hearing / demonstrated / head right / head wrong and found en route toward another
    sector (before reaching the head's named point's view) / head wrong and found after reaching it (a sweep) / other."""
    z = dict(np.load(f)); age = ages(z); out = dict(file=f.split("/")[-1]); E = {}
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; ate = bool(z["cl_item"][w, s1, n])
        pop = ["all"] + (["hungry_2nd"] if (z["hung"][w, s1, n] and age[w, s1, n] >= HALF) else [])
        sec = int(z["sec"][w, s0, n]); hl = int(z["hl"][w, s0, n]); d = int(z["d"][w, s0, n])
        if z["first"][w, s0, n]:
            k = "first_hearing"
        elif z["dem"][w, s0, n]:
            k = "demonstrated"
        elif hl == sec:
            k = "head_right"
        else:
            st = z["start"][w, s0, n].astype(int); ip = z["ipos"][w, s0, n].astype(int)
            ang = hl * np.pi / 4; hp = st + np.array([np.round(d * np.cos(ang)), np.round(d * np.sin(ang))]).astype(int)
            P = z["pos"][w, s0:s1 + 1, n].astype(int)
            see = np.abs(TA.torus(ip[None] - P)).max(-1) <= R; reach_h = np.abs(TA.torus(hp[None] - P)).max(-1) <= R
            ts = int(np.argmax(see)) if see.any() else None; th = int(np.argmax(reach_h)) if reach_h.any() else None
            if not ate or ts is None:
                k = "head_wrong"
            elif th is not None and th <= ts:
                k = "head_wrong_sweep_after_miss"
            else:
                dsp = TA.torus(P[ts] - st); hd = np.array([np.cos(ang), np.sin(ang)])
                k = "head_wrong_en_route_other" if dsp @ hd > 0 else "head_wrong_other"
        for p_ in pop:
            e = E.setdefault(p_, {}).setdefault(k, [0, 0]); e[0] += ate; e[1] += 1
    for p_, e in E.items():
        nb = sum(v[1] for v in e.values()); na = sum(v[0] for v in e.values())
        out[p_] = dict(bouts=nb, reach=round(na / nb, 4), eaten=na,
                       split={k: dict(bouts=v[1], eaten=v[0], share_of_eaten=round(v[0] / max(na, 1), 3), reach=round(v[0] / max(v[1], 1), 3)) for k, v in sorted(e.items())})
    return out


def origin(f):
    """D1c, how the bad state is written: for each word (world, agent, life, sector) that has a later hearing whose memory
    holds ONLY unsolved records, its FIRST bout in that life (the record that started it): demonstrated? first hearing?
    outcome; generic (world) food eaten during the bout; the young's Chebyshev distance from the item and from the hearing
    point at the bout's last step (pre-move position, so within 1 cell); and the run of bouts with that word."""
    z = dict(np.load(f)); lid = lives(z); W = {}
    for b in TA.bouts(z):
        w, n, s0, s1 = b["w"], b["n"], b["s0"], b["s1"]; sec = int(z["sec"][w, s0, n])
        st = z["start"][w, s0, n].astype(int); ip = z["ipos"][w, s0, n].astype(int); p1 = z["pos"][w, s1, n].astype(int)
        W.setdefault((w, n, int(lid[w, s0, n]), sec), []).append(dict(
            s0=s0, first=bool(z["first"][w, s0, n]), dem=bool(z["dem"][w, s0, n]), ate=bool(z["cl_item"][w, s1, n]), tmax=bool(z["cl_tmax"][w, s1, n]),
            death=bool(z["cl_death"][w, s1, n]), gen=bool(z["gen"][w, s1, n]), to_item=int(np.abs(TA.torus(ip - p1)).max()), from_start=int(np.abs(TA.torus(p1 - st)).max()),
            only_uns=bool(not z["first"][w, s0, n] and not z["dem"][w, s0, n] and z["ma_ns"][w, s0, n].sum() == 0 and z["ma_nu"][w, s0, n].sum() > 0),
            hl_ok=int(z["hl"][w, s0, n]) == sec))
    bad = [sorted(v, key=lambda r: r["s0"]) for v in W.values() if any(r["only_uns"] for r in v)]
    O = [v[0] for v in bad]; n = len(O)
    cnt = lambda key: sum(o[key] for o in O)  # noqa: E731
    later = [r for v in bad for r in v[1:]]
    first_solve = [next((i for i, r in enumerate(v) if r["ate"]), None) for v in bad]
    return dict(file=f.split("/")[-1], words_with_only_unsolved_state=n, words_total=len(W),
                origin=dict(first_hearing=cnt("first"), demonstrated=cnt("dem"), ate=cnt("ate"), timeout=cnt("tmax"), death=cnt("death"), generic_food_eaten=cnt("gen"),
                            to_item_median=float(np.median([o["to_item"] for o in O])) if O else None, from_start_median=float(np.median([o["from_start"] for o in O])) if O else None,
                            to_item_hist={k: sum(o["to_item"] == k for o in O) for k in sorted({o["to_item"] for o in O})}),
                all_first_bouts=dict(n=sum(1 for v in W.values() if sorted(v, key=lambda r: r["s0"])[0]["first"]),
                                     failed=sum(1 for v in W.values() if (lambda r: r["first"] and not r["ate"])(sorted(v, key=lambda r: r["s0"])[0]))),
                later_bouts_of_these_words=dict(n=len(later), ate=sum(r["ate"] for r in later), head_right=sum(r["hl_ok"] for r in later), gen=sum(r["gen"] for r in later)),
                bout_index_of_first_solve={str(k): first_solve.count(k) for k in sorted(set(first_solve), key=lambda x: (x is None, x or 0))})


def ctl(off, au, js):
    a = dict(np.load(off)); b = dict(np.load(au)); sh = sorted(set(a) & set(b))
    diff = [k for k in sh if not np.array_equal(a[k], b[k])]
    J = json.load(open(js))["modes"]["0"]; age = ages(b); cl = b["cl_item"] | b["cl_tmax"]; late = age >= HALF   # the eval counts closed & ~dead
    rb = dict(item_reach=b["cl_item"].sum() / cl.sum(), reach_2nd_half=(b["ate"] & late).sum() / (cl & late).sum())
    return dict(shared_fields=len(sh), extra_fields=sorted(set(b) - set(a)), differing=diff,
                rebuilt={k: round(float(v), 6) for k, v in rb.items()}, json={k: round(J[k], 6) for k in rb})


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "d1":
        for f in sys.argv[2:]:
            print(json.dumps(d1(f)))
    elif m == "d3":
        for f in sys.argv[2:]:
            print(json.dumps(d3(f)))
    elif m == "origin":
        for f in sys.argv[2:]:
            print(json.dumps(origin(f)))
    else:
        print(json.dumps(ctl(*sys.argv[2:5])))
