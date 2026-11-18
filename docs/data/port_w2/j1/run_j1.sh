#!/bin/zsh
# Cam Nov 18 ~16:15Z J1 (+ ~16:06Z chain change, M's card yes): 4-WAY oracle (b') + target-KL .015 + --d4-side (the D4-tied
# side path, w2/d4.py). Seed 1 FIRST (the H1 lock seed), then seed 0; 5e7 each; tables every 1e7 at d = 3 / 12 / 21; E1 gate
# (e1_outcome.py) and rules unchanged; a logit split on any sector < .5 at d = 3 (light eval).
# Pre-check: 4-way flag off, 60 iterations == d1/identity_check/port_off.
# (a) -> reference: s0 to 1e8 + 3-table rule; if s0 fails, s1 to 1e8; the first that qualifies gives the reference -> twin
#     (4-way + target-KL + tie, 3e7) + eval -> STOP here; the merged-learner checks (with --d4-side) and learner seed 9 follow
#     by hand (the ep D4 module), launched without waiting if the checks pass. (b) / (c) / unregistered -> STOP and report.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; M=$SP/j1; CH=$SP/chain; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
R=/Users/overlord/CascadeProjects/throng/docs/data/port_w2/e1; H=/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1
cd /Users/overlord/CascadeProjects/throng-toy
O="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --ppo-log"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --d4-side"
table() {  # $1 seed, $2 K
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/j1s$1_$2e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/j1s$1_$2e7_d$d.npz --out $M/j1s$1_$2e7_d$d.json > $M/j1s$1_$2e7_d$d.log 2>&1; done
  python3 $CH/decide.py J1s$1_$2e7 $M/j1s$1_$2e7_d3_m0.npz $M/j1s$1_$2e7_d12_m0.npz $M/j1s$1_$2e7_train.json > $M/j1s$1_$2e7_verdict.json
  python3 $SP/oscreen/bysec_o.py $M/j1s$1_$2e7_d21_m0.npz > $M/j1s$1_$2e7_d21_bysec.txt 2>&1
  echo "J1 TABLE s$1 $2e7 $(cat $M/j1s$1_$2e7_verdict.json) | d21: $(grep reach $M/j1s$1_$2e7_d21_bysec.txt) $(date '+%H:%M:%S')"
  if python3 -c "import json,sys; sys.exit(0 if min(json.load(open('$M/j1s$1_$2e7_verdict.json'))['d3']) < 0.5 else 1)"; then
    taskpolicy -c background $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/j1s$1_$2e7.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $M/j1s$1_$2e7_lg.npz --out $M/j1s$1_$2e7_lg.json > $M/j1s$1_$2e7_lg.log 2>&1
    python3 w2/logit_split.py --oracle-files $M/j1s$1_$2e7_lg_m0.npz --out $M/j1s$1_$2e7_split.json > /dev/null 2>&1
    echo "J1 SPLIT s$1 $2e7 $(python3 -c "
import json; r=json.load(open('$M/j1s$1_$2e7_split.json'))[0]
print({s: {k: r['by_sector'][s][k] for k in ('p_stay','p_eat','p_toward','side_stay','side_eat')} for s in r['by_sector'] if r['reach_by_sector'][s] < 0.5})")"
  fi
}
leg() {  # $1 seed, $2 K
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=O} --seed $1 --steps $2e7 --ckpt $M/j1s$1.ckpt --max-seconds 480 --save-params $M/j1s$1_$2e7.params --out $M/j1s$1_$2e7_train.json >> $M/j1s$1.log 2>&1
    [ -f $M/j1s$1_$2e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/j1s$1.log && { echo "J1 CRASH seed $1"; exit 1; }; done
  cp $M/j1s$1.ckpt $M/j1s$1_ckpt_$2e7.ckpt
  echo "J1 s$1 $2e7 frontier $(python3 -c "import json;print(json.load(open('$M/j1s$1_$2e7_train.json'))['frontier'])") cut $(grep '^TKL' $M/j1s$1.log | python3 -c "
import sys,json;r=[json.loads(l[4:]) for l in sys.stdin];r=r[-1220:];print('last-1e7 mb_done/total %.3f' % (sum(x['mb_done'] for x in r)/max(sum(x['mb_total'] for x in r),1)), 'iters_stopped %.3f' % (sum(x['kl_stop'] is not None for x in r)/max(len(r),1)))") $(date '+%H:%M:%S')"
}
mkdir -p $M/idchk
$PY w2/port_w2.py --oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --steps 491520 --out $M/idchk/port_off.json > $M/idchk/port_off.log 2>&1
python3 -c "
import json; c=lambda f: [{k: v for k, v in x.items() if k != 'sec'} for x in json.load(open(f))['curve']]
ok = c('$M/idchk/port_off.json') == c('/Users/overlord/CascadeProjects/throng/docs/data/port_w2/d1/identity_check/port_off.json'); print('J1 IDCHECK 4-way flag-off == pre-merge port_off:', ok); raise SystemExit(0 if ok else 1)" || { echo "J1 ABORT: identity check failed"; exit 1; }
echo "J1 start $(date '+%H:%M:%S')"
for SD in 1 0; do for K in 1 2 3 4 5; do leg $SD $K; table $SD $K; done; done
python3 $R/e1_outcome.py $M/j1s0_{1,2,3,4,5}e7_verdict.json $M/j1s1_{1,2,3,4,5}e7_verdict.json > $M/j1_outcome.json
echo "J1 OUTCOME $(python3 -c "import json;o=json.load(open('$M/j1_outcome.json'));print(o['outcome'], 'events', o['n_events'], 'dead', o['any_dead'], 'pass', {k: v['PASS'] for k, v in o['per_seed'].items()})") $(date '+%H:%M:%S')"
grep -q '"outcome": "a"' $M/j1_outcome.json || { echo "J1 STOP (report) $(date '+%H:%M:%S')"; exit 0; }
REF=""
for SD in 0 1; do
  for K in 6 7 8 9 10; do leg $SD $K; table $SD $K; done
  python3 $R/../d1/decide3.py $M/j1s${SD}_{8,9,10}e7_verdict.json > $M/j1s${SD}_rule3.json; echo "J1 s$SD RULE3 $(cat $M/j1s${SD}_rule3.json) $(date '+%H:%M:%S')"
  grep -q '"QUALIFIES": true' $M/j1s${SD}_rule3.json && { REF=$SD; break; }
done
[ -n "$REF" ] || { echo "J1 STOP: neither seed qualifies at 1e8 under the 3-table rule (report) $(date '+%H:%M:%S')"; exit 0; }
for K in 8 9 10; do
  cp $M/j1s${REF}_${K}e7_d12.json $M/j1_ref_${K}e7.json
  for ES in 6000 12000 24000; do
    NB=$(python3 -c "import json;print(int(json.load(open('$M/j1_ref_${K}e7.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
    $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/j1s${REF}_${K}e7.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $M/j1_ref_${K}e7.json > $M/j1_ref.log 2>&1
  done
done
python3 -c "
import json; F=['$M/j1_ref_%de7.json' % k for k in (8, 9, 10)]; m=[json.load(open(f))['modes']['0'] for f in F]
r=sum(x['hungry_reach_2nd_half'] for x in m)/3; print(json.dumps(dict(ref_seed=$REF, ref_files=F, per_ckpt=[x['hungry_reach_2nd_half'] for x in m], n=[x['hungry_bouts_2nd_half'] for x in m], ref_mean=r, bar=round(0.9*r,4))))" > $M/J1_QUALIFIED
python3 -c "
import json; q=json.load(open('$M/J1_QUALIFIED')); j=json.load(open('$M/j1_ref_10e7.json')); j['modes']['0']['hungry_reach_2nd_half']=q['ref_mean']; j['note']='hungry_reach_2nd_half = mean of the 8e7/9e7/1e8 long evals of seed %d (J1)' % q['ref_seed']; json.dump(j, open('$M/j1_ref_mean.json','w'))"
echo "J1 QUALIFIED $(cat $M/J1_QUALIFIED) $(date '+%H:%M:%S')"
T="--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --frontier-schedule $M/j1s${REF}_10e7_train.json"
for K in 1 2 3; do
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=T} --steps ${K}e7 --ckpt $M/tw.ckpt --max-seconds 480 --save-params $M/tw_${K}e7.params --out $M/tw_${K}e7_train.json >> $M/tw.log 2>&1
    [ -f $M/tw_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/tw.log && { echo "J1 TWIN CRASH"; exit 1; }; done
  cp $M/tw.ckpt $M/tw_ckpt_${K}e7.ckpt; echo "J1 TWIN ${K}e7 done $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --d4-side --init-params $M/tw_3e7.params --d-fixed 12 --eval-necessity 0 --out $M/tw_eval_d12.json >> $M/tw.log 2>&1
echo "J1 TWIN eval $(python3 -c "import json;m=json.load(open('$M/tw_eval_d12.json'))['modes']['0'];print('hungry_2nd', m.get('hungry_reach_2nd_half'), 'life', m.get('mean_life_steps'))") $(date '+%H:%M:%S')"
echo "J1_TWIN_DONE (learner seed 9 by hand) $(date '+%H:%M:%S')"
