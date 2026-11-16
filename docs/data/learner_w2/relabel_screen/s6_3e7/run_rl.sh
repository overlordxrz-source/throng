#!/bin/zsh
# Cam Nov 16 ~01:00: RELABEL screen (Cam Nov 16 10:15Z / M 10:46Z), 1 seed (seed 6, fresh): seed 4 config + --head-relabel-token.
# Early stops: (a) head on 2nd+ hearings at d = 3 < 0.25 at 3e7 (pause, evaluate, decide); (b) frontier <= 6 at 5e7 (in the trainer).
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; S=6; L=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
T="--steps 1e8 --early-stop-frontier 6 --seed $S --ckpt $L/s$S.ckpt --max-seconds 480 --out $L/s${S}_train.json"
if [ ! -f $L/s${S}_3e7.params ]; then
  for i in $(seq 1 200); do $PY w2/learner_w2_rl.py ${=X} ${=T} --stop-at 3e7 --save-params $L/s${S}_3e7.params >> $L/s$S.log 2>&1
    grep -q "PAUSE_AT\|Traceback\|EARLY_STOP" $L/s$S.log && break; [ -f $L/s${S}_train.json ] && break; done
fi
grep -q Traceback $L/s$S.log && { echo "seed $S CRASH"; exit 0; }
cp $L/s$S.ckpt $L/s${S}_3e7.ckpt
$PY w2/learner_w2_rl.py ${=X} --seed $S --init-params $L/s${S}_3e7.params --d-fixed 3 --eval-necessity 0 --out $L/s${S}_3e7_d3.json >> $L/s$S.log 2>&1
$PY w2/learner_w2_rl.py ${=X} --seed $S --init-params $L/s${S}_3e7.params --d-fixed 12 --eval-necessity 0 --out $L/s${S}_3e7_d12.json >> $L/s$S.log 2>&1
$PY w2/learner_w2_labels.py --learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --count-iters 100 --lookahead 0.25 --count-relabel --ckpt $L/s${S}_3e7.ckpt --out $L/s${S}_3e7_labels.json >> $L/s$S.log 2>&1
HA=$(python3 -c "import json;print(json.load(open('$L/s${S}_3e7_d3.json'))['modes']['0']['head_acc_later'])")
echo "head 2nd+ at d=3 at 3e7: $HA" >> $L/s$S.log
python3 -c "import sys; sys.exit(0 if float('$HA') >= 0.25 else 1)" || { echo "seed $S EARLY_STOP_A head $HA"; exit 0; }
for i in $(seq 1 400); do $PY w2/learner_w2_rl.py ${=X} ${=T} --save-params $L/s$S.params >> $L/s$S.log 2>&1
  [ -f $L/s${S}_train.json ] && break; grep -q Traceback $L/s$S.log && break; done
grep -q "EARLY_STOP frontier" $L/s$S.log && { echo "seed $S EARLY_STOP_B"; exit 0; }
grep -q Traceback $L/s$S.log && { echo "seed $S CRASH"; exit 0; }
$PY w2/learner_w2_rl.py ${=X} --seed $S --init-params $L/s$S.params --d-fixed 12 --eval-necessity 0,3,4 --out $L/s${S}_eval_d12.json >> $L/s$S.log 2>&1
cp $L/s${S}_eval_d12.json $L/s${S}_eval_d12_long.json   # life from the default (3000-step) eval; reach from the lengthened one (bar3.py)
for ES in 6000 12000 24000; do   # Cam 08:25Z eval power: >= 2000 2nd-half hungry bouts for the learner AND the shuffled control
  NB=$(python3 -c "import json;m=json.load(open('$L/s${S}_eval_d12_long.json'))['modes'];print(int(min(m['0']['hungry_bouts_2nd_half'], m['3']['hungry_bouts_2nd_half'])))")
  echo "eval d12 (reach): min 2nd-half hungry bouts (true, shuffled) = $NB" >> $L/s$S.log; [ $NB -ge 2000 ] && break
  $PY w2/learner_w2_rl.py ${=X} --seed $S --init-params $L/s$S.params --d-fixed 12 --eval-necessity 0,3,4 --eval-steps $ES --out $L/s${S}_eval_d12_long.json >> $L/s$S.log 2>&1
done
$PY w2/learner_w2_rl.py ${=X} --seed $S --init-params $L/s$S.params --eval-d-max 22 --eval-necessity 0 --out $L/s${S}_eval_d22.json >> $L/s$S.log 2>&1
$PY /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/bar/bar3.py $L/s${S}_eval_d12.json $L/s${S}_eval_d12_long.json /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oracle3/eval_d12_long.json /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/twin3/eval_d12.json > $L/s${S}_bar.json
grep -q '"PASS": true' $L/s${S}_bar.json && echo "seed $S PASS" || echo "seed $S FAIL"
