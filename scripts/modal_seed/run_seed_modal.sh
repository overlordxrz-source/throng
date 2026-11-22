#!/bin/zsh
# MODAL container version of docs/data/port_w2/j1/run_seed.sh (the seed-9 recipe; only the seed / paths / chunk length
# changed: --max-seconds 3000 instead of 480; chunked resume is deterministic, so the dynamics are unchanged), plus the
# 3-table verdict (verdict3) at the end. Legs are idempotent, so a preempted / retried container resumes from the volume.
# Cam Nov 18 ~16:06Z / ~16:15Z J1 learner (M's card: yes, stage 1): seed 9 = relabel + --no-nav + 4-WAY + target-KL .015 +
# --d4-side (the ep learner's D4-tied side path, w2/d4.py EpAgentDevD4), to 1e8; launched after the J1 reference + twin and
# the merged-learner checks (j1/learner_checks). Same structure as e1/run_e1l.sh: per 1e7 params + checkpoint + per-sector
# tables at d = 3 / 12 / 21 + Cam's head watch (head_later.py + cut); early stop A (head later d = 3 < .25 at 3e7);
# WATCH_REPORT if < .50 at 3e7; early stop B (frontier <= 6 at 5e7); final d = 12 evals lengthened; bar3 vs the J1
# reference (j1_ref_mean.json) and the J1 twin (tw_eval_d12.json). Reports at 3e7 / 5e7 / 1e8.
SP=/mnt/runs; R=/root/aux; TRJ=/root/aux; BAR=/root/aux; PY=python3
cd /root/tt
S=${1:?seed}; M=$SP/seeds/s$S; mkdir -p $M
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --d4-side"
X="$XT --no-nav --target-kl 0.015"
TR="--steps 1e8 --early-stop-frontier 6 --seed $S --ckpt $M/s$S.ckpt --max-seconds 3000 --out $M/s${S}_train.json"
echo "J1L S$S start $(date '+%H:%M:%S')"
for K in 1 2 3 4 5 6 7 8 9 10; do
  [ -f $M/s${S}_${K}e7.params ] && [ -f $M/s${S}_${K}e7_bysec.txt ] && { echo "J1L S$S ${K}e7 leg already done (idempotent; container retry)"; continue; }
  ST=""; [ $K -lt 10 ] && ST="--stop-at ${K}e7"
  N0=$(grep -c '^TKL' $M/s$S.log 2>/dev/null); N0=${N0:-0}
  for i in $(seq 1 200); do $PY w2/learner_w2_rl2.py ${=X} ${=TR} ${=ST} --save-params $M/s${S}_${K}e7.params >> $M/s$S.log 2>&1
    [ -f $M/s${S}_${K}e7.params ] && break; [ -f $M/s${S}_train.json ] && break; grep -q Traceback $M/s$S.log && { echo "J1L S$S CRASH"; exit 1; }; done
  cp $M/s$S.ckpt $M/s${S}_ckpt_${K}e7.ckpt
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=XT} --seed $S --init-params $M/s${S}_${K}e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/s${S}_${K}e7_traj_d$d.npz --out $M/s${S}_${K}e7_traj_d$d.json >> $M/s${S}_evals.log 2>&1; done
  python3 $TRJ/bysec.py $M/s${S}_${K}e7_traj_d3_m0.npz $M/s${S}_${K}e7_traj_d12_m0.npz $M/s${S}_${K}e7_traj_d21_m0.npz > $M/s${S}_${K}e7_bysec.txt 2>&1
  FR=$(grep -o "frontier [0-9.]*" $M/s$S.log | tail -1); echo "J1L S$S ${K}e7 $FR $(date '+%H:%M:%S')"; grep "later_right" $M/s${S}_${K}e7_bysec.txt
  echo "J1L S$S ${K}e7 WATCH head_later $(python3 /root/aux/head_later.py $M/s${S}_${K}e7_traj_d3_m0.npz $M/s${S}_${K}e7_traj_d12_m0.npz $M/s${S}_${K}e7_traj_d21_m0.npz 2>&1) cut $(grep '^TKL' $M/s$S.log | tail -n +$((N0+1)) | python3 -c "
import sys,json;r=[json.loads(l[4:]) for l in sys.stdin];print('this-leg %.3f (n_it %d; chunk restarts replay <= 20 its)' % (1-sum(x['mb_done'] for x in r)/max(sum(x['mb_total'] for x in r),1), len(r)))")"
  if [ $K -eq 3 ]; then
    $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $M/s${S}_3e7.params --d-fixed 3 --eval-necessity 0,3,4 --out $M/s${S}_3e7_d3.json >> $M/s${S}_evals.log 2>&1
    HA=$(python3 -c "import json;print(json.load(open('$M/s${S}_3e7_d3.json'))['modes']['0']['head_acc_later'])"); echo "J1L S$S 3e7 head later d=3 $HA"
    python3 -c "import sys; sys.exit(0 if float('$HA') >= 0.50 else 1)" || echo "J1L S$S 3e7 WATCH_REPORT head later d=3 $HA < .50 (Cam 18:35Z: report at once; recipe unchanged)"
    python3 -c "import sys; sys.exit(0 if float('$HA') >= 0.25 else 1)" || { echo "J1L S$S EARLY_STOP_A head $HA (FAIL)"; exit 0; }
  fi
  grep -q "EARLY_STOP frontier" $M/s$S.log && { echo "J1L S$S EARLY_STOP_B (FAIL) $(date '+%H:%M:%S')"; exit 0; }
  [ -f $M/s${S}_train.json ] && break
done
P=$M/s${S}_10e7.params; [ -f $P ] || P=$(ls -t $M/s${S}_*e7.params | head -1)
$PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --out $M/s${S}_eval_d12.json >> $M/s${S}_evals.log 2>&1
cp $M/s${S}_eval_d12.json $M/s${S}_eval_d12_long.json
for ES in 6000 12000 24000; do
  NB=$(python3 -c "import json;m=json.load(open('$M/s${S}_eval_d12_long.json'))['modes'];print(int(min(m['0']['hungry_bouts_2nd_half'], m['3']['hungry_bouts_2nd_half'])))"); [ $NB -ge 2000 ] && break
  $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --eval-steps $ES --out $M/s${S}_eval_d12_long.json >> $M/s${S}_evals.log 2>&1
done
python3 $BAR/bar3.py $M/s${S}_eval_d12.json $M/s${S}_eval_d12_long.json $R/j1_ref_mean.json $R/tw_eval_d12.json > $M/s${S}_bar.json
grep -q '"PASS": true' $M/s${S}_bar.json && echo "J1L S$S PASS $(date '+%H:%M:%S')" || echo "J1L S$S FAIL $(date '+%H:%M:%S')"

# ---- 3-table verdict (= docs/data/port_w2/j1/verdict3.sh, container paths) ----
for K in 8 9 10; do
  P=$M/s${S}_${K}e7.params; E=$M/v3_s${S}_${K}e7_d12.json; EL=$M/v3_s${S}_${K}e7_d12_long.json
  if [ $K -eq 10 ] && [ -f $M/s${S}_eval_d12_long.json ]; then cp $M/s${S}_eval_d12.json $E; cp $M/s${S}_eval_d12_long.json $EL
  else
    $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --out $E >> $M/v3_s${S}.log 2>&1; cp $E $EL
    for ES in 6000 12000 24000; do
      NB=$(python3 -c "import json;m=json.load(open('$EL'))['modes'];print(int(min(m['0']['hungry_bouts_2nd_half'], m['3']['hungry_bouts_2nd_half'])))"); [ $NB -ge 2000 ] && break
      $PY w2/learner_w2_rl2.py ${=X} --seed $S --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --eval-steps $ES --out $EL >> $M/v3_s${S}.log 2>&1
    done
  fi
  python3 $BAR/bar3.py $E $EL $R/j1_ref_mean.json $R/tw_eval_d12.json > $M/v3_s${S}_${K}e7_bar.json
  python3 /root/aux/decomp_d12.py $M/s${S}_${K}e7_traj_d12_m0.npz > $M/v3_s${S}_${K}e7_decomp.json
done
python3 /root/aux/verdict3.py $M/v3_s${S}_{8,9,10}e7_bar.json $M/s${S}_10e7_bysec.txt > $M/v3_s${S}_verdict.json
echo "V3 S$S VERDICT $(python3 -c "import json;v=json.load(open('$M/v3_s${S}_verdict.json'));print({k: v[k] for k in ('mean_true','mean_shuffled','mean_wiped','need','c_reach','c_shuf30','c_2x','mean_life_ratio','life_lbs','c_life','final_dead','PASS')})") $(date '+%H:%M:%S')"
