#!/bin/zsh
# MODAL container: fallback (3) ORACLE (Cam Nov 22 ~03:35Z item 6): 4-way oracle (b') + target-KL .015 + --d4-side +
# --offset-proj, ONE seed ($1), legs of 1e7 to NLEG x 1e7 (5 for the E1 gate), tables at d = 3 / 12 / 21 every leg
# (decide.py + d21 by sector + a logit split on any d3 sector < .5) -- J1's leg()/table() with container paths, 3000 s
# chunks, idempotent legs (skip a leg whose params + verdict exist). The E1 gate (both seeds) is computed locally after copy-back.
S=${1:?seed}; SP=/mnt/runs; M=$SP/oracle/${OUTTAG:-o$S}; mkdir -p $M; PY=python3; A=/root/aux
cd /root/tt
LEGSTEP=${LEGSTEP:-1e7}; NLEG=${NLEG:-5}; CHUNK=${CHUNK:-3000}; CHAIN=${CHAIN:-}
O="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --offset-proj --ppo-log"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --d4-side --offset-proj"
tgt() { python3 -c "print('${1}e7' if '$LEGSTEP' == '1e7' else int(float('$LEGSTEP') * $1))"; }
echo "F3 s$S start $(date '+%H:%M:%S') legs $NLEG x $LEGSTEP chunk $CHUNK"
for K in $(seq 1 $NLEG); do
  [ -f $M/f3s${S}_${K}e7.params ] && [ -f $M/f3s${S}_${K}e7_verdict.json ] && { echo "F3 s$S ${K}e7 leg already done (idempotent)"; continue; }
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=O} --seed $S --steps $(tgt $K) --ckpt $M/f3s$S.ckpt --max-seconds $CHUNK --save-params $M/f3s${S}_${K}e7.params --out $M/f3s${S}_${K}e7_train.json >> $M/f3s$S.log 2>&1
    [ -f $M/f3s${S}_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/f3s$S.log && { echo "F3 CRASH seed $S"; exit 1; }; done
  cp $M/f3s$S.ckpt $M/f3s${S}_ckpt_${K}e7.ckpt
  echo "F3 s$S ${K}e7 frontier $(python3 -c "import json;print(json.load(open('$M/f3s${S}_${K}e7_train.json'))['frontier'])") $(date '+%H:%M:%S')"
  for d in 3 12 21; do $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/f3s${S}_${K}e7.params --d-fixed $d --eval-necessity 0 --traj-out $M/f3s${S}_${K}e7_d$d.npz --out $M/f3s${S}_${K}e7_d$d.json > $M/f3s${S}_${K}e7_d$d.log 2>&1; done
  python3 $A/decide.py F3s${S}_${K}e7 $M/f3s${S}_${K}e7_d3_m0.npz $M/f3s${S}_${K}e7_d12_m0.npz $M/f3s${S}_${K}e7_train.json > $M/f3s${S}_${K}e7_verdict.json
  python3 $A/bysec_o.py $M/f3s${S}_${K}e7_d21_m0.npz > $M/f3s${S}_${K}e7_d21_bysec.txt 2>&1
  echo "F3 TABLE s$S ${K}e7 $(cat $M/f3s${S}_${K}e7_verdict.json) | d21: $(grep reach $M/f3s${S}_${K}e7_d21_bysec.txt) $(date '+%H:%M:%S')"
  if python3 -c "import json,sys; sys.exit(0 if min(json.load(open('$M/f3s${S}_${K}e7_verdict.json'))['d3']) < 0.5 else 1)"; then
    $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/f3s${S}_${K}e7.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $M/f3s${S}_${K}e7_lg.npz --out $M/f3s${S}_${K}e7_lg.json > $M/f3s${S}_${K}e7_lg.log 2>&1
    python3 w2/logit_split.py --oracle-files $M/f3s${S}_${K}e7_lg_m0.npz --out $M/f3s${S}_${K}e7_split.json > /dev/null 2>&1
    echo "F3 SPLIT s$S ${K}e7 $(python3 -c "
import json; r=json.load(open('$M/f3s${S}_${K}e7_split.json'))[0]
print({s: {k: r['by_sector'][s][k] for k in ('p_stay','p_eat','p_toward','side_stay','side_eat')} for s in r['by_sector'] if r['reach_by_sector'][s] < 0.5})")"
  fi
done
echo "F3 s$S DONE $(date '+%H:%M:%S')"
[ "$CHAIN" = reftwin ] || exit 0
# ---- REFERENCE (3-table rule on 8e7 / 9e7 / 1e8) + TWIN (3e7) for this seed (= run_j1.sh's chain, container paths) ----
python3 $A/decide3.py $M/f3s${S}_{8,9,10}e7_verdict.json > $M/f3s${S}_rule3.json; echo "F3 s$S RULE3 $(cat $M/f3s${S}_rule3.json) $(date '+%H:%M:%S')"
grep -q '"QUALIFIES": true' $M/f3s${S}_rule3.json || { echo "F3 REF STOP: s$S does not qualify at 1e8 under the 3-table rule (report) $(date '+%H:%M:%S')"; exit 0; }
for K in 8 9 10; do
  [ -f $M/f3_ref_${K}e7.json ] || cp $M/f3s${S}_${K}e7_d12.json $M/f3_ref_${K}e7.json
  for ES in 6000 12000 24000; do
    NB=$(python3 -c "import json;print(int(json.load(open('$M/f3_ref_${K}e7.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
    $PY w2/learner_w2_rl_traj.py ${=E} --seed 0 --init-params $M/f3s${S}_${K}e7.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $M/f3_ref_${K}e7.json > $M/f3_ref.log 2>&1
  done
done
python3 -c "
import json; F=['$M/f3_ref_%de7.json' % k for k in (8, 9, 10)]; m=[json.load(open(f))['modes']['0'] for f in F]
r=sum(x['hungry_reach_2nd_half'] for x in m)/3; print(json.dumps(dict(ref_seed=$S, ref_files=F, per_ckpt=[x['hungry_reach_2nd_half'] for x in m], n=[x['hungry_bouts_2nd_half'] for x in m], ref_mean=r, bar=round(0.9*r,4))))" > $M/F3_QUALIFIED
python3 -c "
import json; q=json.load(open('$M/F3_QUALIFIED')); j=json.load(open('$M/f3_ref_10e7.json')); j['modes']['0']['hungry_reach_2nd_half']=q['ref_mean']; j['note']='hungry_reach_2nd_half = mean of the 8e7/9e7/1e8 long evals of seed %d (F3)' % q['ref_seed']; json.dump(j, open('$M/f3_ref_mean.json','w'))"
echo "F3 QUALIFIED $(cat $M/F3_QUALIFIED) $(date '+%H:%M:%S')"
T="--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --offset-proj --frontier-schedule $M/f3s${S}_10e7_train.json"
for K in 1 2 3; do
  [ -f $M/tw_${K}e7.params ] && [ -f $M/tw_${K}e7_train.json ] && { echo "F3 TWIN ${K}e7 already done (idempotent)"; continue; }
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=T} --steps ${K}e7 --ckpt $M/tw.ckpt --max-seconds $CHUNK --save-params $M/tw_${K}e7.params --out $M/tw_${K}e7_train.json >> $M/tw.log 2>&1
    [ -f $M/tw_${K}e7_train.json ] && break; grep -q "Traceback\|NAN_STOP" $M/tw.log && { echo "F3 TWIN CRASH"; exit 1; }; done
  cp $M/tw.ckpt $M/tw_ckpt_${K}e7.ckpt; echo "F3 TWIN ${K}e7 done $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --d4-side --offset-proj --init-params $M/tw_3e7.params --d-fixed 12 --eval-necessity 0 --out $M/tw_eval_d12.json >> $M/tw.log 2>&1
echo "F3 TWIN eval $(python3 -c "import json;m=json.load(open('$M/tw_eval_d12.json'))['modes']['0'];print('hungry_2nd', m.get('hungry_reach_2nd_half'), 'life', m.get('mean_life_steps'))") $(date '+%H:%M:%S')"
echo "F3_REFTWIN_DONE $(date '+%H:%M:%S')"
