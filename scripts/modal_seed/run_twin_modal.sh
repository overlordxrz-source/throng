#!/bin/zsh
# MODAL container: TWIN-K2 (Cam Nov 24 ~23:55Z chain b). The deaf twin of fallback (3)'s reference, now in the SAME world
# as the K = 2 learner: rl2 --learner rec + --d4-side (port_w2's RecAgentD4; bit-exact to port_w2 with the demonstrator
# off) + the reference's frontier schedule + --demonstrator --parent-waits --demo-hearings 2. Legs of 1e7 to 3e7 with
# --stop-at (idempotent: a leg whose params exist is skipped; a retried container resumes from the checkpoint), then the
# twin's LIFE eval (3000 steps, same world incl. K = 2) -> tw_k2_eval_d12.json. Outputs: /mnt/runs/twin/${OUTTAG:-k2}.
SP=/mnt/runs; M=$SP/twin/${OUTTAG:-k2}; mkdir -p $M; PY=python3
cd /root/tt
LEGSTEP=${LEGSTEP:-1e7}; NLEG=${NLEG:-3}; CHUNK=${CHUNK:-3000}; SCHED=${SCHED:-$SP/oracle/o0/f3s0_10e7_train.json}
TOTAL=$(python3 -c "print('%de7' % $NLEG if '$LEGSTEP' == '1e7' else int(float('$LEGSTEP') * $NLEG))")
WORLD="--hunger-gate --item-size 0.1 --W 32 --d4-side --offset-proj --demonstrator --parent-waits --demo-hearings 2"
T="--learner rec $WORLD --gate honest --no-nav --target-kl 0.015 --frontier-schedule $SCHED --steps $TOTAL --ckpt $M/tw.ckpt --max-seconds $CHUNK --out $M/tw_train.json"
[ -f $SCHED ] || { echo "TWIN STOP: no frontier schedule $SCHED"; exit 1; }
echo "TWIN-K2 start $(date '+%H:%M:%S') legs $NLEG x $LEGSTEP chunk $CHUNK sched $SCHED"
for K in $(seq 1 $NLEG); do
  [ -f $M/tw_${K}e7.params ] && { echo "TWIN ${K}e7 already done (idempotent)"; continue; }
  ST=""; [ $K -lt $NLEG ] && ST="--stop-at $(python3 -c "print('${K}e7' if '$LEGSTEP' == '1e7' else int(float('$LEGSTEP') * $K))")"
  for i in $(seq 1 200); do $PY w2/learner_w2_rl2.py ${=T} ${=ST} --save-params $M/tw_${K}e7.params >> $M/tw.log 2>&1
    [ -f $M/tw_${K}e7.params ] && break; grep -q "Traceback\|NAN_STOP" $M/tw.log && { echo "TWIN CRASH"; exit 1; }; done
  cp $M/tw.ckpt $M/tw_ckpt_${K}e7.ckpt; echo "TWIN ${K}e7 done $(grep -o 'frontier [0-9.]*' $M/tw.log | tail -1) $(date '+%H:%M:%S')"
done
$PY w2/learner_w2_rl_traj.py --learner rec ${=WORLD} --init-params $M/tw_${NLEG}e7.params --d-fixed 12 --eval-necessity 0 --out $M/tw_k2_eval_d12.json >> $M/tw.log 2>&1
echo "TWIN eval $(python3 -c "import json;m=json.load(open('$M/tw_k2_eval_d12.json'))['modes']['0'];print('life', m.get('mean_life_steps'), 'hungry_2nd', m.get('hungry_reach_2nd_half'), 'nd', m.get('hungry_reach_2nd_half_nd'), 'demo_bouts', m.get('bouts_demonstrated'))") $(date '+%H:%M:%S')"
[ -f $M/tw_k2_eval_d12.json ] || { echo "TWIN EVAL FAILED (no tw_k2_eval_d12.json)"; exit 1; }
echo "TWIN_K2_DONE $(date '+%H:%M:%S')"
