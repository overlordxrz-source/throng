#!/bin/zsh
# --offset-proj RUN-LEVEL flag-off identity (oracle + learner), on the offset-proj worktree. Run ONLY when no training job
# is running (Cam 21:25Z: code only until then). References: d1/identity_check/port_off.json (4-way oracle flag off) and
# m8/learner_identity/rl2_off.json (learner seed 7 flag off); plus a --offset-proj smoke run of each.
I=/Users/overlord/throng-backup/scratch/opchk; mkdir -p $I; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; cd /Users/overlord/throng-backup/scratch/opwt
R=/Users/overlord/CascadeProjects/throng/docs/data/port_w2
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
$PY w2/port_w2.py --oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --steps 491520 --out $I/port_off.json > $I/port_off.log 2>&1
$PY w2/port_w2.py --oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --offset-proj --steps 163840 --out $I/port_proj.json > $I/port_proj.log 2>&1
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --seed 7 --steps 491520 --out $I/l_off.json > $I/l_off.log 2>&1
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --target-kl 0.015 --d4-side --offset-proj --seed 9 --steps 163840 --out $I/l_proj.json > $I/l_proj.log 2>&1
python3 - <<PYE | tee $I/RESULT.txt
import json
c = lambda f: [{k: v for k, v in x.items() if k != "sec"} for x in json.load(open(f))["curve"]]
r = [c("$I/port_off.json") == c("$R/d1/identity_check/port_off.json"), c("$I/l_off.json") == c("$R/m8/learner_identity/rl2_off.json")]
for f in ("port_proj.json", "l_proj.json"):
    try: json.load(open("$I/" + f)); r.append(True)
    except Exception: r.append(False)
print("port flag-off identity", r[0], "| rl2 flag-off identity", r[1], "| proj smoke runs", r[2:], "|", "ALL PASS" if all(r) else "SOME FAIL")
PYE
