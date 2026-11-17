"""Compare the G2 / merged-learner check runs (scratchpad/g2chk) with the pre-merge references."""
import json, sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8wt/w2")
import lr_anneal as LRA  # noqa: E402
I = "/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/g2chk2"  # rerun 2 (first run: g2chk)
R = "/Users/overlord/CascadeProjects/throng/docs/data/port_w2"
def lines(f, tag): return [json.loads(l[len(tag) + 1:]) for l in open(f) if l.startswith(tag + " ")]
def curve(f): return [{k: v for k, v in c.items() if k != "sec"} for c in json.load(open(f))["curve"]]
def strip(L, drop=("lr", "t24")): return [{k: v for k, v in x.items() if k not in drop} for x in L]
res = []
def rep(name, ok, info=""): res.append(ok); print(("PASS " if ok else "FAIL ") + name, info)
rep("unit test", "ALL PASS" in open(f"{I}/unit.txt").read())
rep("port flag-off == d1 port_m8_log16 (curve, PPO)", curve(f"{I}/p_off.json") == curve(f"{R}/d1/identity_check/port_m8_log16.json") and lines(f"{I}/p_off.log", "PPO") == lines(f"{R}/d1/identity_check/port_m8_log16.log", "PPO"))
rep("port target-KL (sc = 1.0) == E1 check on (curve, PPO, TKL)", curve(f"{I}/p_tkl.json") == curve(f"{R}/e1/identity_check/on.json") and lines(f"{I}/p_tkl.log", "PPO") == lines(f"{R}/e1/identity_check/on.log", "PPO") and lines(f"{I}/p_tkl.log", "TKL") == lines(f"{R}/e1/identity_check/on.log", "TKL"))
A = lines(f"{I}/p_anA.log", "TKL"); steps = [8192 * x["it"] for x in A]; t24 = next((x["t24"] for x in A if x["t24"] is not None), None)
exp = [LRA.lr_at(s, (x["t24"] if x["t24"] is not None else None), 3e-4, 3e-5, 81920, 81920) for s, x in zip(steps, A)]
rep("port anneal: logged LR == schedule at every iteration", all(abs(x["lr"] - e) < 1e-12 for x, e in zip(A, exp)), f"t24 {t24}, lr first/last {A[0]['lr']:.2e}/{A[-1]['lr']:.2e}, n {len(A)}")
rep("port anneal: LR actually moved and reached lr_end", A[0]["lr"] == 3e-4 and abs(A[-1]["lr"] - 3e-5) < 1e-12 and t24 is not None)
Bl = lines(f"{I}/p_anB.log", "TKL"); nres = open(f"{I}/p_anB.log").read().count("LR_ANNEAL resume t24")
rep("port anneal resume (chunked) == continuous (TKL incl. LR, PPO, curve)", Bl == A and lines(f"{I}/p_anB.log", "PPO") == lines(f"{I}/p_anA.log", "PPO") and curve(f"{I}/p_anB.json") == curve(f"{I}/p_anA.json"), f"resumes {nres}")
o = open(f"{I}/p_old.log").read(); ol = lines(f"{I}/p_old.log", "TKL")
rep("port resume from an old (no-t24) E1 5e7 checkpoint: t24 from curve, LR 3e-4 at 5e7", "(curve)" in o and all(abs(x["lr"] - 3e-4) < 1e-12 for x in ol) and all(x["t24"] is not None and x["t24"] <= 2.5e7 for x in ol), f"{[l for l in o.splitlines() if 'LR_ANNEAL resume' in l]} n {len(ol)}")
rep("rl2 flag-off (seed 7) == pre-merge m8 learner_identity rl2_off (curve)", curve(f"{I}/l_off.json") == curve(f"{R}/m8/learner_identity/rl2_off.json"))
rep("rl2 moves8 (seed 8) == pre-merge rl2_on (curve)", curve(f"{I}/l_on.json") == curve(f"{R}/m8/learner_identity/rl2_on.json"))
rep("rl2 target-KL 1e9 == rl2_on (curve; info)", curve(f"{I}/l_big.json") == curve(f"{R}/m8/learner_identity/rl2_on.json"))
S = lines(f"{I}/l_syn.log", "TKL"); rep("rl2 synthetic high-KL batch: 0 steps, dparam 0", len(S) > 0 and all(x["mb_done"] == 0 and x["dparam"] == 0.0 for x in S), f"kl_stop {S[0]['kl_stop'] if S else None}")
T = lines(f"{I}/l_tkl.log", "TKL"); cut = 1 - sum(x["mb_done"] for x in T) / max(sum(x["mb_total"] for x in T), 1); rep("rl2 target-KL .015 runs (cut fraction reported)", len(T) > 0, f"cut {cut:.3f} n {len(T)}")
L = lines(f"{I}/l_an.log", "TKL"); lo = open(f"{I}/l_an.log").read(); t = next((x["t24"] for x in L if x["t24"] is not None), None)
st = None if t is None else LRA.anneal_start(t, 81920) + 163840
rep("rl2 anneal + --lr-anneal-stop: LR follows the schedule and the run ends at start + len", "LR_ANNEAL_STOP" in lo and L[-1]["lr"] <= 3e-4 and t is not None, f"t24 {t}, stop expected at {st}, {[l for l in lo.splitlines() if 'LR_ANNEAL_STOP' in l]}")
a, b = json.load(open(f"{I}/traj_on_d3.json")), json.load(open(f"{R}/m8/learner_identity/traj_on_d3.json"))
ka, kb = a["modes"], b["modes"]
rep("traj ep moves8 eval == pre-merge traj_on_d3 (all keys but time/paths)", ka == kb, f"keys {len(ka)}")
print("ALL PASS" if all(res) else "SOME FAIL", f"{sum(res)}/{len(res)}")
