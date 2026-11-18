#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; I=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/d4chk; H=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/h1
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav"
$PY w2/test_d4.py > $I/unit.txt 2>&1; echo "unit $(tail -1 $I/unit.txt)"
$PY w2/port_w2.py ${=B} --steps 491520 --out $I/off.json > $I/off.log 2>&1; echo "off $(date '+%H:%M:%S')"
$PY w2/port_w2.py ${=B} --target-kl 0.015 --steps 491520 --out $I/tkl.json > $I/tkl.log 2>&1; echo "tkl $(date '+%H:%M:%S')"
$PY w2/port_w2.py ${=B} --target-kl 0.015 --d4-side --steps 491520 --save-params $I/d4.params --out $I/d4.json > $I/d4.log 2>&1; echo "d4 $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --seed 0 --init-params $H/h1s0_5e7.params --d-fixed 3 --eval-necessity 0 --out $I/traj_off_d3.json > $I/traj_off.log 2>&1; echo "traj_off $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --seed 0 --d4-side --init-params $I/d4.params --d-fixed 3 --eval-necessity 0 --out $I/traj_d4_d3.json > $I/traj_d4.log 2>&1; echo "traj_d4 $(date '+%H:%M:%S')"
echo DONE
