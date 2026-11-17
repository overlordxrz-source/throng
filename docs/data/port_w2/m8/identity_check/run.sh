#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; M=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8id
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav"
$PY w2/port_w2.py ${=B} --steps 491520 --out $M/port_off.json > $M/port_off.log 2>&1
$PY w2/port_w2.py ${=B} --moves8 --steps 163840 --save-params $M/port_on.params --out $M/port_on.json > $M/port_on.log 2>&1
$PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/chain/B.params --d-fixed 3 --eval-necessity 0 --traj-out $M/traj_off_d3.npz --out $M/traj_off_d3.json > $M/traj_off.log 2>&1
$PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --moves8 --init-params $M/port_on.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $M/traj_on_d3.npz --out $M/traj_on_d3.json > $M/traj_on.log 2>&1
echo M8ID_DONE >> $M/done.txt
