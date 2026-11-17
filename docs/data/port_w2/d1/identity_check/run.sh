#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; I=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/d1id
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav"
$PY w2/port_w2.py ${=B} --steps 491520 --out $I/port_off.json > $I/port_off.log 2>&1
$PY w2/port_w2.py ${=B} --moves8 --ppo-log --steps 491520 --out $I/port_m8_log.json > $I/port_m8_log.log 2>&1
$PY w2/port_w2.py ${=B} --moves8 --d-move 0 --steps 163840 --out $I/port_m8_d0.json > $I/port_m8_d0.log 2>&1
$PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/chain/B.params --d-fixed 3 --eval-necessity 0 --out $I/traj_off_d3.json > $I/traj_off.log 2>&1
echo DONE > $I/done.txt
