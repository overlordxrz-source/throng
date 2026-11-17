#!/bin/zsh
# Cam Nov 17 ~16:40Z E1 (a) / ~16:55Z: learner seed 8 = relabel + --no-nav + 8-way + TARGET-KL .015 (rl2), to 1e8, launched after
# the E1 twin and the learner fold-in checks (e1/learner_identity). Same structure as m8/run_m8l.sh: pause every 1e7 (params +
# checkpoint copy + per-sector table at d = 3, 12, 21); early stop A (head on later hearings at d = 3 < .25 at 3e7), early stop B
# (frontier <= 6 at 5e7); final d = 12 evals lengthened; bar3 vs the E1 oracle reference (s0, mean of the 8e7/9e7/1e8 long evals,
# e1_ref_mean.json) and the E1 twin (tw_eval_d12.json). Report at 3e7 and 5e7.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; M=$SP/e1l; R=$SP/e1; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
cd /Users/overlord/CascadeProjects/throng-toy
S=8
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --moves8"
X="$XT --no-nav --target-kl 0.015"
TR="--steps 1e8 --early-stop-frontier 6 --seed $S --ckpt $M/s$S.ckpt --max-seconds 480 --out $M/s${S}_train.json"
echo "E1L start $(date '+%H:%M:%S')"
for K in 1 2 3 4 5 6 7 8 9 10; do
  ST=""; [ $K -lt 10 ] && ST="--stop-at ${K}e7"
  N0=$(grep -c '^TKL' $M/s$S.log 2>/dev/null); N0=${N0:-0}
  for i in $(seq 1 200); do $PY w2/learner_w2_rl2.py ${=X} ${=TR} ${=ST} --save-params $M/s${S}_${K}e7.params >> $M/s$S.log 2>&1
    [ -f $M/s${S}_${K}e7.params ] && break; [ -f $M/s${S}_train.json ] && break; grep -q Traceback $M/s$S.log && { echo "E1L S8 CRASH"; exit 1; }; done
  cp $M/s$S.ckpt $M/s${S}_ckpt_${K}e7.ckpt
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=XT} --seed $S --init-params $M/s${S}_${K}e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/s${S}_${K}e7_traj_d$d.npz --out $M/s${S}_${K}e7_traj_d$d.json >> $M/s${S}_evals.log 2>&1; done
  python3 $SP/traj/bysec.py $M/s${S}_${K}e7_traj_d3_m0.npz $M/s${S}_${K}e7_traj_d12_m0.npz $M/s${S}_${K}e7_traj_d21_m0.npz > $M/s${S}_${K}e7_bysec.txt 2>&1
  FR=$(grep -o "frontier [0-9.]*" $M/s$S.log | tail -1); echo "E1L S8 ${K}e7 $FR $(date '+%H:%M:%S')"; grep "later_right" $M/s${S}_${K}e7_bysec.txt
  echo "E1L S8 ${K}e7 WATCH head_later $(python3 /Users/overlord/CascadeProjects/throng/docs/data/port_w2/e1/head_later.py $M/s${S}_${K}e7_traj_d3_m0.npz $M/s${S}_${K}e7_traj_d12_m0.npz $M/s${S}_${K}e7_traj_d21_m0.npz 2>&1) cut $(grep '^TKL' $M/s$S.log | tail -n +$((N0+1)) | python3 -c "
import sys,json;r=[json.loads(l[4:]) for l in sys.stdin];print('this-leg %.3f (n_it %d; chunk restarts replay <= 20 its)' % (1-sum(x['mb_done'] for x in r)/max(sum(x['mb_total'] for x in r),1), len(r)))")"
  if [ $K -eq 3 ]; then
    $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $M/s${S}_3e7.params --d-fixed 3 --eval-necessity 0,3,4 --out $M/s${S}_3e7_d3.json >> $M/s${S}_evals.log 2>&1
    HA=$(python3 -c "import json;print(json.load(open('$M/s${S}_3e7_d3.json'))['modes']['0']['head_acc_later'])"); echo "E1L S8 3e7 head later d=3 $HA"
    python3 -c "import sys; sys.exit(0 if float('$HA') >= 0.50 else 1)" || echo "E1L S8 3e7 WATCH_REPORT head later d=3 $HA < .50 (Cam 18:35Z: report at once; recipe unchanged)"
    python3 -c "import sys; sys.exit(0 if float('$HA') >= 0.25 else 1)" || { echo "E1L S8 EARLY_STOP_A head $HA (FAIL)"; exit 0; }
  fi
  grep -q "EARLY_STOP frontier" $M/s$S.log && { echo "E1L S8 EARLY_STOP_B (FAIL) $(date '+%H:%M:%S')"; exit 0; }
  [ -f $M/s${S}_train.json ] && break
done
P=$M/s${S}_10e7.params; [ -f $P ] || P=$(ls -t $M/s${S}_*e7.params | head -1)
$PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --out $M/s${S}_eval_d12.json >> $M/s${S}_evals.log 2>&1
cp $M/s${S}_eval_d12.json $M/s${S}_eval_d12_long.json
for ES in 6000 12000 24000; do
  NB=$(python3 -c "import json;m=json.load(open('$M/s${S}_eval_d12_long.json'))['modes'];print(int(min(m['0']['hungry_bouts_2nd_half'], m['3']['hungry_bouts_2nd_half'])))"); [ $NB -ge 2000 ] && break
  $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --eval-steps $ES --out $M/s${S}_eval_d12_long.json >> $M/s${S}_evals.log 2>&1
done
python3 $SP/bar/bar3.py $M/s${S}_eval_d12.json $M/s${S}_eval_d12_long.json $R/e1_ref_mean.json $R/tw_eval_d12.json > $M/s${S}_bar.json
grep -q '"PASS": true' $M/s${S}_bar.json && echo "E1L S8 PASS $(date '+%H:%M:%S')" || echo "E1L S8 FAIL $(date '+%H:%M:%S')"
