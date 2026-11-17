#!/bin/zsh
# Cam Nov 17 ~13:00Z STEP B (A1 clean): L2 8-WAY MOVEMENT. Oracle (b') + --moves8 on seeds 0 and 1, 3e7 each (legs of 1e7:
# params + checkpoint copy at every 1e7; d = 3 by-sector table at 1e7 / 2e7, logged); at 3e7 the chain's evals at d = 3, 12, 21
# + decide.py + d = 21 by sector. Both QUALIFY -> reference from seed 0's 3e7 params (lengthened to >= 2000 2nd-half hungry
# bouts) + the lever twin (8-way, --no-nav, seed 0's frontier curve, 3e7) + its d = 12 eval, then M8_ORACLE_TWIN_DONE.
# Otherwise STOP and report. The learner (seed 8) is queued separately once its 8-way build passes its checks.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; M=$SP/m8; CH=$SP/chain; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
cd /Users/overlord/CascadeProjects/throng-toy
O="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --moves8"
echo "M8 start $(date '+%H:%M:%S')"
for SD in 0 1; do
  for K in 1 2 3; do
    for i in $(seq 1 100); do $PY w2/port_w2.py ${=O} --seed $SD --steps ${K}e7 --ckpt $M/o8s$SD.ckpt --max-seconds 480 --save-params $M/o8s${SD}_${K}e7.params --out $M/o8s${SD}_${K}e7_train.json >> $M/o8s$SD.log 2>&1
      [ -f $M/o8s${SD}_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/o8s$SD.log && { echo "M8 CRASH seed $SD"; exit 1; }; done
    cp $M/o8s$SD.ckpt $M/o8s${SD}_ckpt_${K}e7.ckpt
    echo "M8 s$SD ${K}e7 frontier $(python3 -c "import json;print(json.load(open('$M/o8s${SD}_${K}e7_train.json'))['frontier'])") $(date '+%H:%M:%S')"
    if [ $K -lt 3 ]; then
      $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/o8s${SD}_${K}e7.params --d-fixed 3 --eval-necessity 0 --traj-out $M/o8s${SD}_${K}e7_d3.npz --out $M/o8s${SD}_${K}e7_d3.json > $M/o8s${SD}_${K}e7_d3.log 2>&1
      echo "M8 s$SD ${K}e7 d3 $(python3 $SP/oscreen/bysec_o.py $M/o8s${SD}_${K}e7_d3_m0.npz | grep reach)"
    fi
  done
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/o8s${SD}_3e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/o8s${SD}_3e7_d$d.npz --out $M/o8s${SD}_3e7_d$d.json > $M/o8s${SD}_3e7_d$d.log 2>&1; done
  python3 $CH/decide.py O8s${SD}_3e7 $M/o8s${SD}_3e7_d3_m0.npz $M/o8s${SD}_3e7_d12_m0.npz $M/o8s${SD}_3e7_train.json > $M/o8s${SD}_3e7_verdict.json
  python3 $SP/oscreen/bysec_o.py $M/o8s${SD}_3e7_d21_m0.npz > $M/o8s${SD}_3e7_d21_bysec.txt 2>&1
  echo "M8 TABLE s$SD 3e7 $(cat $M/o8s${SD}_3e7_verdict.json) | d21: $(grep reach $M/o8s${SD}_3e7_d21_bysec.txt) $(date '+%H:%M:%S')"
done
for SD in 0 1; do grep -q '"QUALIFIES": true' $M/o8s${SD}_3e7_verdict.json || { echo "M8 STOP: seed $SD does not qualify (report) $(date '+%H:%M:%S')"; exit 0; }; done
cp $M/o8s0_3e7_d12.json $M/o8_ref_d12_long.json
for ES in 6000 12000 24000; do
  NB=$(python3 -c "import json;print(int(json.load(open('$M/o8_ref_d12_long.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
  $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/o8s0_3e7.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $M/o8_ref_d12_long.json > $M/o8_ref.log 2>&1
done
python3 -c "import json;m=json.load(open('$M/o8_ref_d12_long.json'))['modes']['0'];print(json.dumps(dict(ref_file='$M/o8_ref_d12_long.json', hungry_2nd_half=m['hungry_reach_2nd_half'], n=m['hungry_bouts_2nd_half'], bar=round(0.9*m['hungry_reach_2nd_half'],4))))" > $M/QUALIFIED
echo "M8 QUALIFIED $(cat $M/QUALIFIED) $(date '+%H:%M:%S')"
T="--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --frontier-schedule $M/o8s0_3e7_train.json"
for K in 1 2 3; do
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=T} --steps ${K}e7 --ckpt $M/tw8.ckpt --max-seconds 480 --save-params $M/tw8_${K}e7.params --out $M/tw8_${K}e7_train.json >> $M/tw8.log 2>&1
    [ -f $M/tw8_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/tw8.log && { echo "M8 TWIN CRASH"; exit 1; }; done
  cp $M/tw8.ckpt $M/tw8_ckpt_${K}e7.ckpt; echo "M8 TWIN ${K}e7 done $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --moves8 --init-params $M/tw8_3e7.params --d-fixed 12 --eval-necessity 0 --out $M/tw8_eval_d12.json >> $M/tw8.log 2>&1
echo "M8 TWIN eval $(python3 -c "import json;m=json.load(open('$M/tw8_eval_d12.json'))['modes']['0'];print('hungry_2nd', m.get('hungry_reach_2nd_half'), 'life', m.get('mean_life_steps'))") $(date '+%H:%M:%S')"
echo "M8_ORACLE_TWIN_DONE $(date '+%H:%M:%S')"
