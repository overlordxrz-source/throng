#!/bin/zsh
# D1/D3/D4/D5 (Cam ~17:05Z), eval only, 3e7 weights; registered in the protocol before running.
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; T=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/traj; P=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl/s6_3e7.params
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
O="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oracle3/o3.params"
echo "start $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_traj.py ${=X} --seed 6 --init-params $P --d-fixed 3 --eval-necessity 0,3,4 --traj-out $T/l_d3.npz --out $T/l_d3.json > $T/l_d3.log 2>&1
for d in 8 12; do $PY w2/learner_w2_rl_traj.py ${=X} --seed 6 --init-params $P --d-fixed $d --eval-necessity 0 --traj-out $T/l_d$d.npz --out $T/l_d$d.json > $T/l_d$d.log 2>&1; done
$PY w2/learner_w2_rl_traj.py ${=X} --seed 6 --init-params $P --eval-mix-la 3 --eval-necessity 0 --traj-out $T/l_mix.npz --out $T/l_mix.json > $T/l_mix.log 2>&1
echo "learner done $(date '+%H:%M:%S')"
for d in 3 8 12; do $PY w2/learner_w2_rl_traj.py ${=O} --d-fixed $d --eval-necessity 0 --traj-out $T/o_d$d.npz --out $T/o_d$d.json > $T/o_d$d.log 2>&1; done
echo "TRAJDONE $(date '+%H:%M:%S')"
