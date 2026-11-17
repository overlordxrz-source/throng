#!/bin/zsh
# Cam Nov 17 ~14:10Z STEP D1 (H-COST): oracle (b') + 8-way + d_move 0, seeds 0 and 1, 3e7 each (legs of 1e7; params + checkpoint
# copy every 1e7; tables at every 1e7 at d = 3 / 12 / 21 + decide.py; PPO diagnostics logged). Outcome by d1_outcome.py.
# (a) -> s0 continues to 1e8 (tables every 1e7); the 3-table rule (decide3.py) on 8e7 / 9e7 / 1e8; if it qualifies, the
# reference = mean of the long d = 12 evals at 8e7 / 9e7 / 1e8; then the twin (8-way + d_move 0, s0's curve, 3e7) + its eval.
# The learner is NOT started here (life gate + fold-in by hand, per Cam). (b) / (c) / unregistered -> stop and report.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; M=$SP/d1; CH=$SP/chain; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
cd /Users/overlord/CascadeProjects/throng-toy
O="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --d-move 0 --ppo-log"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --moves8 --d-move 0"
table() {  # $1 seed, $2 K
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/d1s$1_$2e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/d1s$1_$2e7_d$d.npz --out $M/d1s$1_$2e7_d$d.json > $M/d1s$1_$2e7_d$d.log 2>&1; done
  python3 $CH/decide.py D1s$1_$2e7 $M/d1s$1_$2e7_d3_m0.npz $M/d1s$1_$2e7_d12_m0.npz $M/d1s$1_$2e7_train.json > $M/d1s$1_$2e7_verdict.json
  python3 $SP/oscreen/bysec_o.py $M/d1s$1_$2e7_d21_m0.npz > $M/d1s$1_$2e7_d21_bysec.txt 2>&1
  echo "D1 TABLE s$1 $2e7 $(cat $M/d1s$1_$2e7_verdict.json) | d21: $(grep reach $M/d1s$1_$2e7_d21_bysec.txt) $(date '+%H:%M:%S')"
}
leg() {  # $1 seed, $2 K
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=O} --seed $1 --steps $2e7 --ckpt $M/d1s$1.ckpt --max-seconds 480 --save-params $M/d1s$1_$2e7.params --out $M/d1s$1_$2e7_train.json >> $M/d1s$1.log 2>&1
    [ -f $M/d1s$1_$2e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/d1s$1.log && { echo "D1 CRASH seed $1"; exit 1; }; done
  cp $M/d1s$1.ckpt $M/d1s$1_ckpt_$2e7.ckpt
  echo "D1 s$1 $2e7 frontier $(python3 -c "import json;print(json.load(open('$M/d1s$1_$2e7_train.json'))['frontier'])") $(date '+%H:%M:%S')"
}
echo "D1 start $(date '+%H:%M:%S')"
for SD in 0 1; do for K in 1 2 3; do leg $SD $K; table $SD $K; done; done
python3 $M/d1_outcome.py $M/d1s0_{1,2,3}e7_verdict.json $M/d1s1_{1,2,3}e7_verdict.json > $M/d1_outcome.json
echo "D1 OUTCOME $(python3 -c "import json;o=json.load(open('$M/d1_outcome.json'));print(o['outcome'], 'events_d3', o['n_events_d3'], 'dead', o['any_dead'], 'both_qualify', o['both_qualify'])") $(date '+%H:%M:%S')"
grep -q '"outcome": "a"' $M/d1_outcome.json || { echo "D1 STOP (report) $(date '+%H:%M:%S')"; exit 0; }
for K in 4 5 6 7 8 9 10; do leg 0 $K; table 0 $K; done
python3 $M/decide3.py $M/d1s0_{8,9,10}e7_verdict.json > $M/d1s0_rule3.json; echo "D1 s0 RULE3 $(cat $M/d1s0_rule3.json) $(date '+%H:%M:%S')"
grep -q '"QUALIFIES": true' $M/d1s0_rule3.json || { echo "D1 STOP: s0 does not qualify at 1e8 under the 3-table rule (report) $(date '+%H:%M:%S')"; exit 0; }
for K in 8 9 10; do
  cp $M/d1s0_${K}e7_d12.json $M/d1_ref_${K}e7.json
  for ES in 6000 12000 24000; do
    NB=$(python3 -c "import json;print(int(json.load(open('$M/d1_ref_${K}e7.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
    $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/d1s0_${K}e7.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $M/d1_ref_${K}e7.json > $M/d1_ref.log 2>&1
  done
done
python3 -c "
import json; F=['$M/d1_ref_%de7.json' % k for k in (8, 9, 10)]; m=[json.load(open(f))['modes']['0'] for f in F]
r=sum(x['hungry_reach_2nd_half'] for x in m)/3; print(json.dumps(dict(ref_files=F, per_ckpt=[x['hungry_reach_2nd_half'] for x in m], n=[x['hungry_bouts_2nd_half'] for x in m], ref_mean=r, bar=round(0.9*r,4))))" > $M/QUALIFIED
json_ref=$M/d1_ref_mean.json; python3 -c "
import json; q=json.load(open('$M/QUALIFIED')); j=json.load(open('$M/d1_ref_10e7.json')); j['modes']['0']['hungry_reach_2nd_half']=q['ref_mean']; j['note']='hungry_reach_2nd_half = mean of the 8e7/9e7/1e8 long evals (Cam 14:10Z rule); other fields from 1e8'; json.dump(j, open('$M/d1_ref_mean.json','w'))"
echo "D1 QUALIFIED $(cat $M/QUALIFIED) $(date '+%H:%M:%S')"
T="--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --d-move 0 --frontier-schedule $M/d1s0_10e7_train.json"
for K in 1 2 3; do
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=T} --steps ${K}e7 --ckpt $M/tw.ckpt --max-seconds 480 --save-params $M/tw_${K}e7.params --out $M/tw_${K}e7_train.json >> $M/tw.log 2>&1
    [ -f $M/tw_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/tw.log && { echo "D1 TWIN CRASH"; exit 1; }; done
  cp $M/tw.ckpt $M/tw_ckpt_${K}e7.ckpt; echo "D1 TWIN ${K}e7 done $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --moves8 --d-move 0 --init-params $M/tw_3e7.params --d-fixed 12 --eval-necessity 0 --out $M/tw_eval_d12.json >> $M/tw.log 2>&1
echo "D1 TWIN eval $(python3 -c "import json;m=json.load(open('$M/tw_eval_d12.json'))['modes']['0'];print('hungry_2nd', m.get('hungry_reach_2nd_half'), 'life', m.get('mean_life_steps'))") $(date '+%H:%M:%S')"
echo "D1_TWIN_DONE (life gate + learner by hand) $(date '+%H:%M:%S')"
