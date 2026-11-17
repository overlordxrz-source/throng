#!/bin/zsh
# Cam Nov 17 item 4: oracle (b') continued from its 3e7 checkpoint (a COPY of chain/B.ckpt) to 1e8, same recipe (the chain's
# arm B flags). Per-sector tables (the chain's evals + decide.py) at 5e7 and 1e8.
O=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oc; CH=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/chain; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
cd /Users/overlord/CascadeProjects/throng-toy
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav"
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32"
for LEG in 5e7 1e8; do
  echo "leg $LEG start $(date '+%H:%M:%S')"
  for i in $(seq 1 100); do $PY w2/port_w2.py ${=B} --steps $LEG --ckpt $O/oc.ckpt --max-seconds 480 --save-params $O/oc_$LEG.params --out $O/oc_${LEG}_train.json >> $O/oc.log 2>&1
    [ -f $O/oc_${LEG}_train.json ] && break; grep -q "Traceback\|NAN_STOP" $O/oc.log && { echo "CRASH"; exit 1; }; done
  echo "leg $LEG trained $(date '+%H:%M:%S')"
  for d in 3 12; do $PY w2/learner_w2_rl_traj.py ${=E} --init-params $O/oc_$LEG.params --d-fixed $d --eval-necessity 0 --traj-out $O/oc_${LEG}_d$d.npz --out $O/oc_${LEG}_d$d.json > $O/oc_${LEG}_d$d.log 2>&1; done
  python3 $CH/decide.py B_$LEG $O/oc_${LEG}_d3_m0.npz $O/oc_${LEG}_d12_m0.npz $O/oc_${LEG}_train.json > $O/oc_${LEG}_verdict.json
  echo "leg $LEG table $(cat $O/oc_${LEG}_verdict.json) $(date '+%H:%M:%S')"
done
echo "OC_DONE $(date '+%H:%M:%S')"
