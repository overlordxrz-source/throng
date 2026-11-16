#!/bin/zsh
# Cam Nov 16 ~17:25Z ORACLE SCREEN, one arm: $1 = X | T. o3's recipe + one change; seed 0; 3e7 steps; then evals d = 3 / 12 + per-sector table.
A=$1; O=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oscreen; SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
case $A in X) F="--ent-bout 0.05"; E="";; T) F="--topo"; E="--topo";; *) echo "bad arm"; exit 2;; esac
cd /Users/overlord/CascadeProjects/throng-toy
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest"
echo "$A train start $(date '+%H:%M:%S')"
for i in $(seq 1 100); do $PY w2/port_w2.py ${=B} ${=F} --steps 3e7 --ckpt $O/$A.ckpt --max-seconds 480 --save-params $O/$A.params --out $O/${A}_train.json >> $O/$A.log 2>&1
  [ -f $O/${A}_train.json ] && break; grep -q "Traceback\|NAN_STOP" $O/$A.log && { echo "$A CRASH"; exit 1; }; done
echo "$A train done $(date '+%H:%M:%S')"
for d in 3 12; do $PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 ${=E} --init-params $O/$A.params --d-fixed $d --eval-necessity 0 --traj-out $O/${A}_d$d.npz --out $O/${A}_d$d.json > $O/${A}_d$d.log 2>&1; done
python3 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oscreen/bysec_o.py $O/${A}_d3_m0.npz $O/${A}_d12_m0.npz > $O/${A}_bysec.txt 2>&1
echo "$A ARMDONE $(date '+%H:%M:%S')"
