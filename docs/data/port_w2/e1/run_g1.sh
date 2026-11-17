#!/bin/zsh
# Cam Nov 17 ~19:10Z G1: E1 s1 from its 5e7 checkpoint to 1e8 (target-KL .015 only; tables every 1e7 at d = 3 / 12 / 21).
# Verdict: decide3 on 8e7 / 9e7 / 1e8 (3-table mean; final-table < .30 at d3 or d12 = FAIL) + events_v2 on all ten tables (logged).
# (i) qualifies -> reference = mean of the long d = 12 evals at 8e7 / 9e7 / 1e8 -> twin (8-way + target-KL, s1's curve, 3e7) +
#     eval -> STOP (merged-learner checks + seed 8 by hand). (ii) fails -> STOP (G3 by hand after the G2 --lr-anneal checks).
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; M=$SP/e1; CH=$SP/chain; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
R=/Users/overlord/CascadeProjects/throng/docs/data/port_w2/e1
cd /Users/overlord/CascadeProjects/throng-toy
O="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --target-kl 0.015 --ppo-log"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --moves8"
table() {  # $1 seed, $2 K
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/e1s$1_$2e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/e1s$1_$2e7_d$d.npz --out $M/e1s$1_$2e7_d$d.json > $M/e1s$1_$2e7_d$d.log 2>&1; done
  python3 $CH/decide.py E1s$1_$2e7 $M/e1s$1_$2e7_d3_m0.npz $M/e1s$1_$2e7_d12_m0.npz $M/e1s$1_$2e7_train.json > $M/e1s$1_$2e7_verdict.json
  python3 $SP/oscreen/bysec_o.py $M/e1s$1_$2e7_d21_m0.npz > $M/e1s$1_$2e7_d21_bysec.txt 2>&1
  echo "G1 TABLE s$1 $2e7 $(cat $M/e1s$1_$2e7_verdict.json) | d21: $(grep reach $M/e1s$1_$2e7_d21_bysec.txt) $(date '+%H:%M:%S')"
}
leg() {  # $1 seed, $2 K
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=O} --seed $1 --steps $2e7 --ckpt $M/e1s$1.ckpt --max-seconds 480 --save-params $M/e1s$1_$2e7.params --out $M/e1s$1_$2e7_train.json >> $M/e1s$1.log 2>&1
    [ -f $M/e1s$1_$2e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/e1s$1.log && { echo "G1 CRASH seed $1"; exit 1; }; done
  cp $M/e1s$1.ckpt $M/e1s$1_ckpt_$2e7.ckpt
  echo "G1 s$1 $2e7 frontier $(python3 -c "import json;print(json.load(open('$M/e1s$1_$2e7_train.json'))['frontier'])") cut $(grep '^TKL' $M/e1s$1.log | python3 -c "
import sys,json;r=[json.loads(l[4:]) for l in sys.stdin];r=r[-1220:];print('last-1e7 mb_done/total %.3f' % (sum(x['mb_done'] for x in r)/max(sum(x['mb_total'] for x in r),1)))") $(date '+%H:%M:%S')"
}
cmp -s $M/e1s1.ckpt $M/e1s1_ckpt_5e7.ckpt || { echo "G1 ABORT: e1s1.ckpt is not the 5e7 checkpoint"; exit 1; }
echo "G1 start $(date '+%H:%M:%S')"
for K in 6 7 8 9 10; do leg 1 $K; table 1 $K; done
python3 $R/../d1/decide3.py $M/e1s1_{8,9,10}e7_verdict.json > $M/e1s1_rule3.json
python3 -c "
import json; V=[json.load(open('$M/e1s1_%de7_verdict.json'%k)) for k in range(1,11)]
json.dump({'s1_1e8':[dict(t='%de7'%(i+1),frontier=v['frontier'],d3=v['d3']) for i,v in enumerate(V)]}, open('$M/g1_ev10.json','w'))"
python3 $R/events_v2.py $M/g1_ev10.json > $M/e1s1_1e8_events_v2.json
echo "G1 s1 RULE3 $(cat $M/e1s1_rule3.json) EVENTS $(cat $M/e1s1_1e8_events_v2.json) $(date '+%H:%M:%S')"
grep -q '"QUALIFIES": true' $M/e1s1_rule3.json || { echo "G1 STOP (ii): s1 does not qualify at 1e8 -> G3 by hand $(date '+%H:%M:%S')"; exit 0; }
for K in 8 9 10; do
  cp $M/e1s1_${K}e7_d12.json $M/g1_ref_${K}e7.json
  for ES in 6000 12000 24000; do
    NB=$(python3 -c "import json;print(int(json.load(open('$M/g1_ref_${K}e7.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
    $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/e1s1_${K}e7.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $M/g1_ref_${K}e7.json > $M/g1_ref.log 2>&1
  done
done
python3 -c "
import json; F=['$M/g1_ref_%de7.json' % k for k in (8, 9, 10)]; m=[json.load(open(f))['modes']['0'] for f in F]
r=sum(x['hungry_reach_2nd_half'] for x in m)/3; print(json.dumps(dict(ref_files=F, per_ckpt=[x['hungry_reach_2nd_half'] for x in m], n=[x['hungry_bouts_2nd_half'] for x in m], ref_mean=r, bar=round(0.9*r,4))))" > $M/G1_QUALIFIED
python3 -c "
import json; q=json.load(open('$M/G1_QUALIFIED')); j=json.load(open('$M/g1_ref_10e7.json')); j['modes']['0']['hungry_reach_2nd_half']=q['ref_mean']; j['note']='hungry_reach_2nd_half = mean of the s1 8e7/9e7/1e8 long evals (Cam 14:10Z rule, G1 19:10Z); other fields from 1e8'; json.dump(j, open('$M/e1_ref_mean.json','w'))"
echo "G1 QUALIFIED (i) $(cat $M/G1_QUALIFIED) $(date '+%H:%M:%S')"
T="--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --target-kl 0.015 --frontier-schedule $M/e1s1_10e7_train.json"
for K in 1 2 3; do
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=T} --steps ${K}e7 --ckpt $M/tw.ckpt --max-seconds 480 --save-params $M/tw_${K}e7.params --out $M/tw_${K}e7_train.json >> $M/tw.log 2>&1
    [ -f $M/tw_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/tw.log && { echo "G1 TWIN CRASH"; exit 1; }; done
  cp $M/tw.ckpt $M/tw_ckpt_${K}e7.ckpt; echo "G1 TWIN ${K}e7 done $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --moves8 --init-params $M/tw_3e7.params --d-fixed 12 --eval-necessity 0 --out $M/tw_eval_d12.json >> $M/tw.log 2>&1
echo "G1 TWIN eval $(python3 -c "import json;m=json.load(open('$M/tw_eval_d12.json'))['modes']['0'];print('hungry_2nd', m.get('hungry_reach_2nd_half'), 'life', m.get('mean_life_steps'))") $(date '+%H:%M:%S')"
echo "G1_TWIN_DONE (merged-learner checks + seed 8 by hand) $(date '+%H:%M:%S')"
