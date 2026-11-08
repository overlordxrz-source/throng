"""Erosion diagnostic analysis (registration in docs/RESEARCH_PROTOCOL.md, 2026-11-08). Committed after the runs
finished, before their data were read (disclosed). usage: analyze_diag.py <diag_on.jsonl> <diag_off.jsonl>

Means over updates 1,501-1,800. Predictions (wire-on run):
  Dilution: cos(-g, back) > 0 AND optimizer displacement toward theta_1500 (Delta_opt . back_hat) < 0.1 x ||Delta_wd||
            AND shared share < 0.2.
  Live-wire collapse: cos(-g, back) < 0.
  Listener-first: reliance (deposit drop) < 50% of its 1,500 value at an earlier probe than the first update with
            MI < 50% of its 1,500 value; within 10 updates = simultaneous.
"""
import json
import sys

import numpy as np


def load(p):
    return [json.loads(l) for l in open(p) if '"diag"' in l]


def summarise(name, D):
    w = [d for d in D if d["update"] > 1500]
    m = lambda k: float(np.mean([d[k] for d in w if d.get(k) is not None]))  # noqa: E731
    # optimizer displacement component toward theta_1500: ||opt|| * cos(opt, back)
    toward = float(np.mean([d["opt_disp_norm"] * d["cos_optdisp_back"] for d in w if d.get("cos_optdisp_back") is not None]))
    print(f"== {name}")
    print(f"   (a) ||g|| {m('grad_head_norm'):.4g} vs wd*||theta|| {m('decay_pull_norm'):.4g} | realised: ||D_opt|| "
          f"{m('opt_disp_norm'):.4g}, ||D_wd|| {m('decay_disp_norm'):.4g}, D_opt toward theta_1500 {toward:+.4g} "
          f"({toward / m('decay_disp_norm'):+.3f} x ||D_wd||)")
    print(f"   (b) cos(-g, back) {m('cos_negrad_back'):+.4f} | cos(D_opt, back) {m('cos_optdisp_back'):+.4f}")
    print(f"   (c) shared share {m('shared_share'):.3f} (shared var {m('shared_var'):.3g}, own var {m('own_var'):.3g})")
    probes = [d for d in D if "rel_drop" in d]
    mi = {d["update"]: d["mi_plugin"] for d in D}
    r0 = probes[0]["rel_drop"]; mi0 = mi[1500]
    r_half = next((d["update"] for d in probes if d["update"] > 1500 and d["rel_drop"] < 0.5 * r0), None)
    mi_half = next((u for u in sorted(mi) if u > 1500 and mi[u] < 0.5 * mi0), None)
    print(f"   (d) reliance @1500 {r0:+.2f}, first < 50%: {r_half} | MI @1500 {mi0:.3f}, first < 50%: {mi_half}")
    print("       probes: " + " ".join(f"{d['update']}:{d['rel_drop']:+.2f}/MI{mi[d['update']]:.2f}" for d in probes))
    return dict(cos=m("cos_negrad_back"), toward_ratio=toward / m("decay_disp_norm"), share=m("shared_share"),
                r_half=r_half, mi_half=mi_half)


def main(on, off):
    s_on = summarise("wire ON (seed 90)", load(on))
    summarise("wire OFF (seed 91, reference)", load(off))
    dil = s_on["cos"] > 0 and s_on["toward_ratio"] < 0.1 and s_on["share"] < 0.2
    lwc = s_on["cos"] < 0
    if s_on["r_half"] is None or s_on["mi_half"] is None:
        lf = "undetermined (a threshold was never crossed)"
    elif abs(s_on["r_half"] - s_on["mi_half"]) <= 10:
        lf = "simultaneous"
    else:
        lf = "listener-first" if s_on["r_half"] < s_on["mi_half"] else "MI-first"
    print(f"\n== PREDICTIONS (wire on): dilution {'SUPPORTED' if dil else 'not supported'} | live-wire collapse "
          f"{'SUPPORTED' if lwc else 'not supported'} | order: {lf}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
