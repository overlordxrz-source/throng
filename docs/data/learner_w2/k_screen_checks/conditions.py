"""Cam Nov 16 ~08:05Z item 5: the K-screen launches only if (i) seed 4 stopped at 3e7 or 5e7, (ii) 4a >= 10k / 100k at K = 3,
lookahead 0.25 (d >= 3; mean of the seed 1 and seed 2 final checkpoints), (iii) 4c passes (both held-out decoders <= chance
+ 0.05 at K = 2 and 3, and the positive-control decoder > chance + 0.2), plus the K gate controls and the registration."""
import json, os
SP = "/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad"; P = SP + "/post/yield"; r = {}
qa = open(SP + "/learner_la/queue_la.out").read()
r["seed4_stopped"] = ("EARLY_STOP_A" in qa or "EARLY_STOP_B" in qa) and "LADONE" in qa
def cnt(S, Kh, la):
    d = json.load(open(f"{P}/s{S}_K{Kh}_la{la}.json")); b = d["by_d"]; H = sum(b["hear"])
    return 1e5 * sum(b["own_later_unseen"][3:]) / H, d["leak"]
a1, l1 = cnt(1, 3, "0.25"); a2, l2 = cnt(2, 3, "0.25"); r["4a_s1"], r["4a_s2"] = a1, a2; r["4a_mean"] = (a1 + a2) / 2; r["4a_pass"] = r["4a_mean"] >= 10000 and max(a1, a2) >= 10000          # Cam 08:25Z: the average AND at least one seed
ok = True; r["4c"] = []
for S in (1, 2):
    for Kh in (2, 3):
        for la in ("0", "0.25"):
            L = json.load(open(f"{P}/s{S}_K{Kh}_la{la}.json"))["leak"]; ch = L["chance_majority_test"]; pc = L["positive_control_parent_offset_at_t2"]
            good = L["parent_offset_heldout"] <= ch + 0.05 and L["agent_window_heldout"] <= ch + 0.05 and pc["heldout"] > pc["chance_majority_test"] + 0.2
            r["4c"].append(dict(S=S, K=Kh, la=la, chance=ch, offset=L["parent_offset_heldout"], window=L["agent_window_heldout"], pos_ctl=pc["heldout"], pos_chance=pc["chance_majority_test"], n=L["n"], ok=good)); ok &= good
r["4c_pass"] = ok
C = json.load(open(SP + "/learner_k/checks/summary.json"))                # Cam 08:25Z item 5: fallback for the positive control
r["k_identity"] = C["identity_k0_vs_la"]; r["neg_ok"] = C["neg_frontier"] == 1 and C["neg_windows"] >= 10
r["pos_frontier"] = C["pos_frontier"]; r["pos_mastered"] = C["pos_frontier"] if C["pos_frontier"] >= 24 else C["pos_frontier"] - 1   # mastered = last frontier the gate passed
r["pos_status"] = "normal pass (24)" if C["pos_frontier"] >= 24 else ("PASS as a recorded deviation (mastered >= 13)" if r["pos_mastered"] >= 13 else "GATE DEFECT (mastered < 13)")
r["controls_pass"] = bool(r["k_identity"] and r["neg_ok"] and r["pos_mastered"] >= 13); r["registered"] = os.path.exists(SP + "/learner_k/k_registered")
r["LAUNCH"] = bool(r["seed4_stopped"] and r["4a_pass"] and r["4c_pass"] and r["controls_pass"] and r["registered"])
json.dump(r, open(SP + "/learner_k/conditions.json", "w"), indent=1); print(json.dumps({k: v for k, v in r.items() if k != "4c"}))
