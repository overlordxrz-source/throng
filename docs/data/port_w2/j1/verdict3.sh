#!/bin/zsh
# Cam Nov 21 ~21:20Z: the learner's 3-table verdict for seed $1 in dir $2 (its run_j1l-style dir). For K = 8, 9, 10: the
# default d = 12 eval (life) + the lengthened eval (>= 2000 2nd-half hungry bouts in modes 0 and 3; 6000 / 12000 / 24000
# steps), bar3 per checkpoint, then verdict3.py; plus the d = 12 decomposition (decomp_d12.py) on each checkpoint's table npz.
S=$1; M=$2; SP=/Users/overlord/throng-backup/scratch; R=$SP/j1; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
J=/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1; BAR=/Users/overlord/CascadeProjects/throng/docs/data/learner_w2
cd /Users/overlord/CascadeProjects/throng-toy
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --d4-side"
X="$XT --no-nav --target-kl 0.015"
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
  python3 $J/decomp_d12.py $M/s${S}_${K}e7_traj_d12_m0.npz > $M/v3_s${S}_${K}e7_decomp.json
  echo "V3 S$S ${K}e7 bar3 $(python3 -c "import json;b=json.load(open('$M/v3_s${S}_${K}e7_bar.json'));print({k: (round(b[k],4) if isinstance(b[k], float) else b[k]) for k in ('hungry_2nd_true','hungry_2nd_shuffled','hungry_2nd_wiped','life_ratio','life_lb','head_later','PASS')})") $(date '+%H:%M:%S')"
done
python3 $J/verdict3.py $M/v3_s${S}_{8,9,10}e7_bar.json $M/s${S}_10e7_bysec.txt > $M/v3_s${S}_verdict.json
echo "V3 S$S VERDICT $(python3 -c "import json;v=json.load(open('$M/v3_s${S}_verdict.json'));print({k: v[k] for k in ('mean_true','mean_shuffled','mean_wiped','need','c_reach','c_shuf30','c_2x','mean_life_ratio','life_lbs','c_life','final_dead','PASS')})") $(date '+%H:%M:%S')"
