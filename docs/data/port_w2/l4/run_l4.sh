#!/bin/zsh
# Cam Nov 17 ~07:55Z, L4 SEQUENCE. (i) oracle (b') + L4 fresh to 1e8 (legs of 1e7, params + checkpoint copy at every 1e7;
# per-sector tables at 3e7, 5e7, 1e8 at d = 3, 12, 21); late-lock classes vs its own 3e7 table; reference from 1e8 if it
# qualifies. STOP and report if LOCK or not qualifying. (ii) lever twin (--no-nav + L4, input zeroed, (i)'s schedule, 3e7).
# (iii) learner seed 8 (relabel + --no-nav + L4) to 1e8: pause every 1e7 (params + checkpoint copy + per-sector table at
# d = 3, 12, 21); early stops as registered; bar3 vs (i)'s reference, life vs (ii).
L4=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/l4; SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; CH=$SP/chain; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
cd /Users/overlord/CascadeProjects/throng-toy
O="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --offset-wide"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --offset-wide"
otable() {
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=E} --init-params $L4/o4_$1.params --d-fixed $d --eval-necessity 0 --traj-out $L4/o4_$1_d$d.npz --out $L4/o4_$1_d$d.json > $L4/o4_$1_d$d.log 2>&1; done
  python3 $CH/decide.py O4_$1 $L4/o4_$1_d3_m0.npz $L4/o4_$1_d12_m0.npz $L4/o4_$1_train.json > $L4/o4_$1_verdict.json
  python3 $SP/oscreen/bysec_o.py $L4/o4_$1_d21_m0.npz > $L4/o4_$1_d21_bysec.txt 2>&1
  echo "O4 TABLE $1 $(cat $L4/o4_$1_verdict.json) | d21: $(grep reach $L4/o4_$1_d21_bysec.txt) $(date '+%H:%M:%S')"
}
echo "L4 start $(date '+%H:%M:%S')"
for K in 1 2 3 4 5 6 7 8 9 10; do
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=O} --steps ${K}e7 --ckpt $L4/o4.ckpt --max-seconds 480 --save-params $L4/o4_${K}e7.params --out $L4/o4_${K}e7_train.json >> $L4/o4.log 2>&1
    [ -f $L4/o4_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $L4/o4.log && { echo "O4 CRASH"; exit 1; }; done
  cp $L4/o4.ckpt $L4/o4_ckpt_${K}e7.ckpt
  echo "O4 ${K}e7 frontier $(python3 -c "import json;print(json.load(open('$L4/o4_${K}e7_train.json'))['frontier'])") $(date '+%H:%M:%S')"
  case $K in 3|5|10) otable ${K}e7;; esac
done
python3 $L4/lockcheck.py $L4/o4_3e7_verdict.json $L4/o4_10e7_verdict.json > $L4/o4_lock.json; echo "O4 LATE-LOCK CLASS $(cat $L4/o4_lock.json)"
grep -q '"LOCK"' $L4/o4_lock.json && { echo "STOP: late diagonal lock despite L4 (report; L3 next per Cam) $(date '+%H:%M:%S')"; exit 0; }
grep -q '"QUALIFIES": true' $L4/o4_10e7_verdict.json || { echo "STOP: (i) does not qualify at 1e8 (report; Cam rules) $(date '+%H:%M:%S')"; exit 0; }
cp $L4/o4_10e7_d12.json $L4/o4_ref_d12_long.json
for ES in 6000 12000 24000; do
  NB=$(python3 -c "import json;print(int(json.load(open('$L4/o4_ref_d12_long.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
  $PY w2/learner_w2_rl_traj.py ${=E} --init-params $L4/o4_10e7.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $L4/o4_ref_d12_long.json > $L4/o4_ref.log 2>&1
done
python3 -c "import json;m=json.load(open('$L4/o4_ref_d12_long.json'))['modes']['0'];print(json.dumps(dict(ref_file='$L4/o4_ref_d12_long.json', hungry_2nd_half=m['hungry_reach_2nd_half'], n=m['hungry_bouts_2nd_half'], bar=round(0.9*m['hungry_reach_2nd_half'],4))))" > $L4/QUALIFIED
echo "O4 QUALIFIED $(cat $L4/QUALIFIED) $(date '+%H:%M:%S')"
# (ii) the lever twin
T="--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --offset-wide --frontier-schedule $L4/o4_10e7_train.json"
for K in 1 2 3; do
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=T} --steps ${K}e7 --ckpt $L4/tw.ckpt --max-seconds 480 --save-params $L4/tw_${K}e7.params --out $L4/tw_${K}e7_train.json >> $L4/tw.log 2>&1
    [ -f $L4/tw_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $L4/tw.log && { echo "TWIN CRASH"; exit 1; }; done
  cp $L4/tw.ckpt $L4/tw_ckpt_${K}e7.ckpt; echo "TWIN ${K}e7 done $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --offset-wide --init-params $L4/tw_3e7.params --d-fixed 12 --eval-necessity 0 --out $L4/tw_eval_d12.json >> $L4/tw.log 2>&1
echo "TWIN eval $(python3 -c "import json;m=json.load(open('$L4/tw_eval_d12.json'))['modes']['0'];print('hungry_2nd', m.get('hungry_reach_2nd_half'))") $(date '+%H:%M:%S')"
# (iii) learner seed 8
S=8
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --offset-wide"
X="$XT --no-nav"
TR="--steps 1e8 --early-stop-frontier 6 --seed $S --ckpt $L4/s$S.ckpt --max-seconds 480 --out $L4/s${S}_train.json"
for K in 1 2 3 4 5 6 7 8 9 10; do
  ST=""; [ $K -lt 10 ] && ST="--stop-at ${K}e7"
  for i in $(seq 1 200); do $PY w2/learner_w2_rl2.py ${=X} ${=TR} ${=ST} --save-params $L4/s${S}_${K}e7.params >> $L4/s$S.log 2>&1
    [ -f $L4/s${S}_${K}e7.params ] && break; [ -f $L4/s${S}_train.json ] && break; grep -q Traceback $L4/s$S.log && { echo "S8 CRASH"; exit 1; }; done
  cp $L4/s$S.ckpt $L4/s${S}_ckpt_${K}e7.ckpt
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=XT} --seed $S --init-params $L4/s${S}_${K}e7.params --d-fixed $d --eval-necessity 0 --traj-out $L4/s${S}_${K}e7_traj_d$d.npz --out $L4/s${S}_${K}e7_traj_d$d.json >> $L4/s${S}_evals.log 2>&1; done
  python3 $SP/traj/bysec.py $L4/s${S}_${K}e7_traj_d3_m0.npz $L4/s${S}_${K}e7_traj_d12_m0.npz $L4/s${S}_${K}e7_traj_d21_m0.npz > $L4/s${S}_${K}e7_bysec.txt 2>&1
  FR=$(grep -o "frontier [0-9.]*" $L4/s$S.log | tail -1); echo "S8 ${K}e7 $FR $(date '+%H:%M:%S')"; grep "later_right" $L4/s${S}_${K}e7_bysec.txt
  if [ $K -eq 3 ]; then
    $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $L4/s${S}_3e7.params --d-fixed 3 --eval-necessity 0,3,4 --out $L4/s${S}_3e7_d3.json >> $L4/s${S}_evals.log 2>&1
    HA=$(python3 -c "import json;print(json.load(open('$L4/s${S}_3e7_d3.json'))['modes']['0']['head_acc_later'])"); echo "S8 3e7 head later d=3 $HA"
    python3 -c "import sys; sys.exit(0 if float('$HA') >= 0.25 else 1)" || { echo "S8 EARLY_STOP_A head $HA (FAIL)"; exit 0; }
  fi
  grep -q "EARLY_STOP frontier" $L4/s$S.log && { echo "S8 EARLY_STOP_B (FAIL) $(date '+%H:%M:%S')"; exit 0; }
  [ -f $L4/s${S}_train.json ] && break
done
P=$L4/s${S}_10e7.params; [ -f $P ] || P=$(ls -t $L4/s${S}_*e7.params | head -1)
$PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --out $L4/s${S}_eval_d12.json >> $L4/s${S}_evals.log 2>&1
cp $L4/s${S}_eval_d12.json $L4/s${S}_eval_d12_long.json
for ES in 6000 12000 24000; do
  NB=$(python3 -c "import json;m=json.load(open('$L4/s${S}_eval_d12_long.json'))['modes'];print(int(min(m['0']['hungry_bouts_2nd_half'], m['3']['hungry_bouts_2nd_half'])))"); [ $NB -ge 2000 ] && break
  $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --eval-steps $ES --out $L4/s${S}_eval_d12_long.json >> $L4/s${S}_evals.log 2>&1
done
python3 $SP/bar/bar3.py $L4/s${S}_eval_d12.json $L4/s${S}_eval_d12_long.json $L4/o4_ref_d12_long.json $L4/tw_eval_d12.json > $L4/s${S}_bar.json
grep -q '"PASS": true' $L4/s${S}_bar.json && echo "S8 PASS $(date '+%H:%M:%S')" || echo "S8 FAIL $(date '+%H:%M:%S')"
