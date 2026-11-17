#!/bin/zsh
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; I=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8lid
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
cd /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8wt
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --seed 7 --steps 491520 --out $I/rl2_off.json > $I/rl2_off.log 2>&1
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --seed 8 --steps 163840 --save-params $I/rl2_on.params --out $I/rl2_on.json > $I/rl2_on.log 2>&1
$PY w2/learner_w2_rl_traj.py ${=XT} --seed 7 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl2/s7.params --d-fixed 3 --eval-necessity 0 --traj-out $I/traj_new_d3.npz --out $I/traj_new_d3.json > $I/traj_new.log 2>&1
$PY w2/learner_w2_rl_traj.py ${=XT} --moves8 --seed 8 --init-params $I/rl2_on.params --d-fixed 3 --eval-necessity 0 --traj-out $I/traj_on_d3.npz --out $I/traj_on_d3.json > $I/traj_on.log 2>&1
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --seed 8 --init-params $I/rl2_on.params --d-fixed 3 --eval-necessity 0,3,4 --out $I/rl2_on_eval_d3.json > $I/rl2_on_eval.log 2>&1
cd /Users/overlord/CascadeProjects/throng-toy
$PY w2/learner_w2_rl_traj.py ${=XT} --seed 7 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl2/s7.params --d-fixed 3 --eval-necessity 0 --traj-out $I/traj_old_d3.npz --out $I/traj_old_d3.json > $I/traj_old.log 2>&1
echo DONE > $I/done.txt
