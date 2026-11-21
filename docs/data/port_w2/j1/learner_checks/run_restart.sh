#!/bin/zsh
# J1 learner fold-in checks (d4-learner a7087f2 vs pre-merge comm-toy 195fab8; run when no training job is running).
# Persistent-path update of run.sh: s7.params in /private/tmp was lost on Nov 18-20 tmp wipe, so traj ep flag-off
# identity is verified by comparing pre-merge (PRE_CODE=195fab8) vs post-merge (CODE=a7087f2) on the freshly saved
# flag-off params l_off.params; rl2 flag-off curve still compares against committed m8/learner_identity/rl2_off.json.
set -e
SP=/Users/overlord/throng-backup/scratch; I=$SP/j1lchk; mkdir -p $I
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
PRE_CODE=${PRE_CODE:-/Users/overlord/CascadeProjects/throng-toy}
CODE=${CODE:-$SP/d4lwt}
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"

cd "$CODE"
$PY w2/test_d4.py > $I/unit_oracle.txt 2>&1; $PY w2/test_d4_learner.py > $I/unit_learner.txt 2>&1
echo "units $(tail -1 $I/unit_oracle.txt) / $(tail -1 $I/unit_learner.txt)"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --seed 7 --steps 491520 --save-params $I/l_off.params --out $I/l_off.json > $I/l_off.log 2>&1; echo "l_off $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --target-kl 0.015 --seed 9 --steps 163840 --out $I/l_tkl.json > $I/l_tkl.log 2>&1; echo "l_tkl $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --target-kl 0.015 --d4-side --seed 9 --steps 163840 --save-params $I/l_d4.params --out $I/l_d4.json > $I/l_d4.log 2>&1; echo "l_d4 $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_traj.py ${=XT} --seed 7 --init-params $I/l_off.params --d-fixed 3 --eval-necessity 0 --traj-out $I/traj_off_post_d3.npz --out $I/traj_off_post_d3.json > $I/traj_off_post.log 2>&1; echo "traj_off_post $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_traj.py ${=XT} --d4-side --seed 9 --init-params $I/l_d4.params --d-fixed 3 --eval-necessity 0,3,4 --traj-out $I/traj_d4_d3.npz --out $I/traj_d4_d3.json > $I/traj_d4.log 2>&1; echo "traj_d4 $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --target-kl 0.015 --d4-side --seed 9 --init-params $I/l_d4.params --d-fixed 3 --eval-necessity 0,3,4 --out $I/l_d4_eval_d3.json > $I/l_d4_eval.log 2>&1; echo "l_d4_eval $(date '+%H:%M:%S')"

cd "$PRE_CODE"
$PY w2/learner_w2_rl_traj.py ${=XT} --seed 7 --init-params $I/l_off.params --d-fixed 3 --eval-necessity 0 --traj-out $I/traj_off_pre_d3.npz --out $I/traj_off_pre_d3.json > $I/traj_off_pre.log 2>&1; echo "traj_off_pre $(date '+%H:%M:%S')"

python3 - <<PYE > $I/RESULT.txt
import json
R = "/Users/overlord/CascadeProjects/throng/docs/data/port_w2/m8/learner_identity"; I = "$I"; res = []
c = lambda f: [{k: v for k, v in x.items() if k != "sec"} for x in json.load(open(f))["curve"]]
def rep(n, ok, info=""): res.append(bool(ok)); print(("PASS " if ok else "FAIL ") + n, info)
u = open(I + "/unit_oracle.txt").read(), open(I + "/unit_learner.txt").read(); rep("unit tests (oracle, learner)", all("ALL PASS" in x for x in u))
rep("rl2 flag-off seed 7 == pre-merge rl2_off (curve)", c(I + "/l_off.json") == c(R + "/rl2_off.json"))
rep("traj ep flag-off post == pre on l_off.params d3 (modes)", json.load(open(I + "/traj_off_post_d3.json"))["modes"] == json.load(open(I + "/traj_off_pre_d3.json"))["modes"])
for f in ("l_d4.json", "traj_d4_d3.json", "l_d4_eval_d3.json"):
    try: json.load(open(I + "/" + f)); rep("runs: " + f, True)
    except Exception as e: rep("runs: " + f, False, str(e))
s1, s2 = json.load(open(I + "/l_tkl.json"))["curve"][-1]["sec"], json.load(open(I + "/l_d4.json"))["curve"][-1]["sec"]
print("speed: rl2 target-KL %.1f s vs + d4 %.1f s -> x%.2f" % (s1, s2, s2 / s1))
m = json.load(open(I + "/l_d4_eval_d3.json"))["modes"]; print("d4 eval modes 0/3/4 head_later", [round(m[k]["head_acc_later"], 3) for k in ("0", "3", "4")])
print("ALL PASS" if all(res) else "SOME FAIL", f"{sum(res)}/{len(res)}")
PYE
cat $I/RESULT.txt; echo J1LCHK_DONE
