#!/bin/zsh
# Cam / M Nov 16 ~19:00Z: THE CHAIN. Oracle arms in order (a') -> O-X' -> (b'), max 3; stop at the first that qualifies
# (PASS or secondary), then re-derive the oracle reference from it (>= 2000 2nd-half hungry bouts at d = 12) and write QUALIFIED.
CH=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/chain; SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
cd /Users/overlord/CascadeProjects/throng-toy
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32"
for A in A XP B; do
  case $A in A) F="--reinit-out";; XP) F="--ent-bout 0.05 --ent-anneal 2e7";; B) F="--no-nav";; esac
  echo "$A train start $(date '+%H:%M:%S') flags: $F"
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=B} ${=F} --steps 3e7 --ckpt $CH/$A.ckpt --max-seconds 480 --save-params $CH/$A.params --out $CH/${A}_train.json >> $CH/$A.log 2>&1
    [ -f $CH/${A}_train.json ] && break; grep -q "Traceback\|NAN_STOP" $CH/$A.log && { echo "$A CRASH"; exit 1; }; done
  echo "$A train done $(date '+%H:%M:%S')"
  for d in 3 12; do $PY w2/learner_w2_rl_traj.py ${=E} --init-params $CH/$A.params --d-fixed $d --eval-necessity 0 --traj-out $CH/${A}_d$d.npz --out $CH/${A}_d$d.json > $CH/${A}_d$d.log 2>&1; done
  python3 $CH/decide.py $A $CH/${A}_d3_m0.npz $CH/${A}_d12_m0.npz $CH/${A}_train.json > $CH/${A}_verdict.json
  cat $CH/${A}_verdict.json
  if grep -q '"dead_le010": \[[0-9]' $CH/${A}_verdict.json || grep -q '"dead_lt030": \[[0-9]' $CH/${A}_verdict.json; then   # logit split on any dead sector
    $PY w2/learner_w2_rl_traj.py ${=E} --init-params $CH/$A.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $CH/${A}_log.npz --out $CH/${A}_log.json > $CH/${A}_log.log 2>&1
    python3 w2/logit_split.py --oracle-files $CH/${A}_log_m0.npz --out $CH/${A}_logit_split.json >> $CH/${A}_log.log 2>&1
  fi
  echo "$A ARMDONE $(date '+%H:%M:%S')"
  if grep -q '"QUALIFIES": true' $CH/${A}_verdict.json; then
    cp $CH/${A}_d12.json $CH/${A}_ref_d12_long.json
    for ES in 6000 12000 24000; do
      NB=$(python3 -c "import json;print(int(json.load(open('$CH/${A}_ref_d12_long.json'))['modes']['0']['hungry_bouts_2nd_half']))"); [ $NB -ge 2000 ] && break
      $PY w2/learner_w2_rl_traj.py ${=E} --init-params $CH/$A.params --d-fixed 12 --eval-necessity 0 --eval-steps $ES --out $CH/${A}_ref_d12_long.json > $CH/${A}_ref.log 2>&1
    done
    python3 -c "import json;m=json.load(open('$CH/${A}_ref_d12_long.json'))['modes']['0'];print(json.dumps(dict(arm='$A', ref_file='$CH/${A}_ref_d12_long.json', hungry_2nd_half=m['hungry_reach_2nd_half'], n=m['hungry_bouts_2nd_half'], bar=round(0.9*m['hungry_reach_2nd_half'],4))))" > $CH/QUALIFIED
    echo "QUALIFIED $(cat $CH/QUALIFIED) $(date '+%H:%M:%S')"; exit 0
  fi
done
echo "CHAIN_NONE_QUALIFIED $(date '+%H:%M:%S')"
