#!/bin/zsh
# Cam / M Nov 16 ~19:00Z: the LEARNER SCREEN after a qualifying chain arm: relabel (seed 4 config + --head-relabel-token) + that
# arm's lever, seed 7 (new). Early stops as registered: (a) head on later hearings at d = 3 < 0.25 at 3e7; (b) frontier <= 6 at 5e7.
# Per-sector table at every pause. Bar: bar3.py against the arm's re-derived oracle reference. $1 = lever flags, $2 = oracle ref file.
LEV=$1; REF=$2; S=7; L=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl2; SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
cd /Users/overlord/CascadeProjects/throng-toy
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
T="--steps 1e8 --early-stop-frontier 6 --seed $S --ckpt $L/s$S.ckpt --max-seconds 480 --out $L/s${S}_train.json"
sectors() {   # per-sector table on params $1, label $2
  for d in 3 12; do $PY w2/learner_w2_rl_traj.py ${=X} --seed $S --init-params $1 --d-fixed $d --eval-necessity 0 --traj-out $L/s${S}_$2_traj_d$d.npz --out $L/s${S}_$2_traj_d$d.json >> $L/s$S.log 2>&1; done
  python3 $SP/traj/bysec.py $L/s${S}_$2_traj_d3_m0.npz $L/s${S}_$2_traj_d12_m0.npz > $L/s${S}_$2_bysec.txt 2>&1
}
echo "RL2 start $(date '+%H:%M:%S') lever: $LEV ref: $REF"
for i in $(seq 1 200); do $PY w2/learner_w2_rl2.py ${=X} ${=LEV} ${=T} --stop-at 3e7 --save-params $L/s${S}_3e7.params >> $L/s$S.log 2>&1
  grep -q "PAUSE_AT\|Traceback\|EARLY_STOP" $L/s$S.log && break; [ -f $L/s${S}_train.json ] && break; done
grep -q Traceback $L/s$S.log && { echo "seed $S CRASH"; exit 0; }
cp $L/s$S.ckpt $L/s${S}_3e7.ckpt
$PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $L/s${S}_3e7.params --d-fixed 3 --eval-necessity 0,3,4 --out $L/s${S}_3e7_d3.json >> $L/s$S.log 2>&1
$PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $L/s${S}_3e7.params --d-fixed 12 --eval-necessity 0 --out $L/s${S}_3e7_d12.json >> $L/s$S.log 2>&1
$PY w2/learner_w2_labels.py --learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --count-iters 100 --lookahead 0.25 --count-relabel --ckpt $L/s${S}_3e7.ckpt --out $L/s${S}_3e7_labels.json >> $L/s$S.log 2>&1
sectors $L/s${S}_3e7.params 3e7
HA=$(python3 -c "import json;print(json.load(open('$L/s${S}_3e7_d3.json'))['modes']['0']['head_acc_later'])")
echo "PAUSE_3E7 head later d=3 $HA $(date '+%H:%M:%S')"
python3 -c "import sys; sys.exit(0 if float('$HA') >= 0.25 else 1)" || { echo "seed $S EARLY_STOP_A head $HA"; exit 0; }
for i in $(seq 1 400); do $PY w2/learner_w2_rl2.py ${=X} ${=LEV} ${=T} --save-params $L/s$S.params >> $L/s$S.log 2>&1
  [ -f $L/s${S}_train.json ] && break; grep -q Traceback $L/s$S.log && break; done
grep -q Traceback $L/s$S.log && { echo "seed $S CRASH"; exit 0; }
sectors $L/s$S.params final
grep -q "EARLY_STOP frontier" $L/s$S.log && { echo "seed $S EARLY_STOP_B $(date '+%H:%M:%S')"; exit 0; }
$PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $L/s$S.params --d-fixed 12 --eval-necessity 0,3,4 --out $L/s${S}_eval_d12.json >> $L/s$S.log 2>&1
cp $L/s${S}_eval_d12.json $L/s${S}_eval_d12_long.json
for ES in 6000 12000 24000; do
  NB=$(python3 -c "import json;m=json.load(open('$L/s${S}_eval_d12_long.json'))['modes'];print(int(min(m['0']['hungry_bouts_2nd_half'], m['3']['hungry_bouts_2nd_half'])))"); [ $NB -ge 2000 ] && break
  $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $L/s$S.params --d-fixed 12 --eval-necessity 0,3,4 --eval-steps $ES --out $L/s${S}_eval_d12_long.json >> $L/s$S.log 2>&1
done
python3 $SP/bar/bar3.py $L/s${S}_eval_d12.json $L/s${S}_eval_d12_long.json $REF $SP/twin3/eval_d12.json > $L/s${S}_bar.json
grep -q '"PASS": true' $L/s${S}_bar.json && echo "seed $S PASS" || echo "seed $S FAIL"
