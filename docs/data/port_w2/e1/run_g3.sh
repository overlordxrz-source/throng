#!/bin/zsh
# Cam Nov 17 ~19:10Z G3 (branch ii, only if G1 fails): target-KL .015 + --lr-anneal (3e-4 -> 3e-5 over 5e7 from max(5e7, t24))
# from BOTH seeds' E1 5e7 checkpoints to 1e8 (t24 from the checkpoint curve). Tables every 1e7 at d = 3 / 12 / 21; decide3 on
# 8e7 / 9e7 / 1e8 per seed; events_v2 on all ten tables per seed (1e7..5e7 = the E1 tables).
# Both qualify -> reference = mean over BOTH seeds of the long d = 12 evals at 8e7 / 9e7 / 1e8 (six checkpoints) -> twin
# (target-KL + anneal rule, s0's curve, 3e7; the anneal never starts inside 3e7) + eval -> STOP (seed 8 by hand). Else STOP.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; M=$SP/g3; E1=$SP/e1; CH=$SP/chain; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
R=/Users/overlord/CascadeProjects/throng/docs/data/port_w2/e1
cd /Users/overlord/CascadeProjects/throng-toy
O="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --target-kl 0.015 --lr-anneal --ppo-log"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --moves8"
table() {  # $1 seed, $2 K
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/g3s$1_$2e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/g3s$1_$2e7_d$d.npz --out $M/g3s$1_$2e7_d$d.json > $M/g3s$1_$2e7_d$d.log 2>&1; done
  python3 $CH/decide.py G3s$1_$2e7 $M/g3s$1_$2e7_d3_m0.npz $M/g3s$1_$2e7_d12_m0.npz $M/g3s$1_$2e7_train.json > $M/g3s$1_$2e7_verdict.json
  python3 $SP/oscreen/bysec_o.py $M/g3s$1_$2e7_d21_m0.npz > $M/g3s$1_$2e7_d21_bysec.txt 2>&1
  echo "G3 TABLE s$1 $2e7 $(cat $M/g3s$1_$2e7_verdict.json) | d21: $(grep reach $M/g3s$1_$2e7_d21_bysec.txt) $(date '+%H:%M:%S')"
}
leg() {  # $1 seed, $2 K
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=O} --seed $1 --steps $2e7 --ckpt $M/g3s$1.ckpt --max-seconds 480 --save-params $M/g3s$1_$2e7.params --out $M/g3s$1_$2e7_train.json >> $M/g3s$1.log 2>&1
    [ -f $M/g3s$1_$2e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/g3s$1.log && { echo "G3 CRASH seed $1"; exit 1; }; done
  cp $M/g3s$1.ckpt $M/g3s$1_ckpt_$2e7.ckpt
  echo "G3 s$1 $2e7 frontier $(python3 -c "import json;print(json.load(open('$M/g3s$1_$2e7_train.json'))['frontier'])") $(grep '^TKL' $M/g3s$1.log | tail -1220 | python3 -c "
import sys,json;r=[json.loads(l[4:]) for l in sys.stdin];print('cut %.3f lr %.2e..%.2e t24 %s' % (1-sum(x['mb_done'] for x in r)/max(sum(x['mb_total'] for x in r),1), r[0]['lr'], r[-1]['lr'], r[-1]['t24']))") $(date '+%H:%M:%S')"
}
mkdir -p $M; [ -f $M/g3s0.ckpt ] || cp $E1/e1s0_ckpt_5e7.ckpt $M/g3s0.ckpt; [ -f $M/g3s1.ckpt ] || cp $E1/e1s1_ckpt_5e7.ckpt $M/g3s1.ckpt
echo "G3 start $(date '+%H:%M:%S')"
for SD in 0 1; do for K in 6 7 8 9 10; do leg $SD $K; table $SD $K; done; done
for SD in 0 1; do
  python3 $R/../d1/decide3.py $M/g3s${SD}_{8,9,10}e7_verdict.json > $M/g3s${SD}_rule3.json
  python3 -c "
import json; V=[json.load(open('$E1/e1s${SD}_%de7_verdict.json'%k)) for k in range(1,6)]+[json.load(open('$M/g3s${SD}_%de7_verdict.json'%k)) for k in range(6,11)]
json.dump({'g3s$SD':[dict(t='%de7'%(i+1),frontier=v['frontier'],d3=v['d3']) for i,v in enumerate(V)]}, open('$M/g3s${SD}_ev10.json','w'))"
  python3 $R/events_v2.py $M/g3s${SD}_ev10.json > $M/g3s${SD}_events_v2.json
  echo "G3 s$SD RULE3 $(cat $M/g3s${SD}_rule3.json) EVENTS $(cat $M/g3s${SD}_events_v2.json) $(date '+%H:%M:%S')"
done
grep -q '"QUALIFIES": true' $M/g3s0_rule3.json && grep -q '"QUALIFIES": true' $M/g3s1_rule3.json || { echo "G3 STOP: not both qualify (report) $(date '+%H:%M:%S')"; exit 0; }
for SD in 0 1; do for K in 8 9 10; do
  cp $M/g3s${SD}_${K}e7_d12.json $M/g3_ref_s${SD}_${K}e7.json
  for ES in 6000 12000 24000; do
    NB=$(python3 -c "import json;print(int(json.load(open('$M/g3_ref_s${SD}_${K}e7.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
    $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/g3s${SD}_${K}e7.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $M/g3_ref_s${SD}_${K}e7.json > $M/g3_ref.log 2>&1
  done
done; done
python3 -c "
import json; F=['$M/g3_ref_s%d_%de7.json' % (s, k) for s in (0, 1) for k in (8, 9, 10)]; m=[json.load(open(f))['modes']['0'] for f in F]
r=sum(x['hungry_reach_2nd_half'] for x in m)/6; print(json.dumps(dict(ref_files=F, per_ckpt=[x['hungry_reach_2nd_half'] for x in m], n=[x['hungry_bouts_2nd_half'] for x in m], ref_mean=r, bar=round(0.9*r,4))))" > $M/G3_QUALIFIED
python3 -c "
import json; q=json.load(open('$M/G3_QUALIFIED')); j=json.load(open('$M/g3_ref_s0_10e7.json')); j['modes']['0']['hungry_reach_2nd_half']=q['ref_mean']; j['note']='hungry_reach_2nd_half = mean of both seeds 8e7/9e7/1e8 long evals (Cam 19:10Z G3); other fields from s0 1e8'; json.dump(j, open('$M/g3_ref_mean.json','w'))"
echo "G3 QUALIFIED $(cat $M/G3_QUALIFIED) $(date '+%H:%M:%S')"
T="--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --target-kl 0.015 --lr-anneal --frontier-schedule $M/g3s0_10e7_train.json"
for K in 1 2 3; do
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=T} --steps ${K}e7 --ckpt $M/tw.ckpt --max-seconds 480 --save-params $M/tw_${K}e7.params --out $M/tw_${K}e7_train.json >> $M/tw.log 2>&1
    [ -f $M/tw_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/tw.log && { echo "G3 TWIN CRASH"; exit 1; }; done
  cp $M/tw.ckpt $M/tw_ckpt_${K}e7.ckpt; echo "G3 TWIN ${K}e7 done $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --moves8 --init-params $M/tw_3e7.params --d-fixed 12 --eval-necessity 0 --out $M/tw_eval_d12.json >> $M/tw.log 2>&1
echo "G3 TWIN eval $(python3 -c "import json;m=json.load(open('$M/tw_eval_d12.json'))['modes']['0'];print('hungry_2nd', m.get('hungry_reach_2nd_half'), 'life', m.get('mean_life_steps'))") $(date '+%H:%M:%S')"
echo "G3_TWIN_DONE (seed 8 by hand) $(date '+%H:%M:%S')"
