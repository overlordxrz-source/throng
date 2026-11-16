#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; T=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/traj/logit
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
$PY w2/learner_w2_rl_traj.py ${=X} --seed 6 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl/s6_3e7.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $T/l3.npz --out $T/l3.json > $T/l3.log 2>&1
$PY w2/learner_w2_rl_traj.py ${=X} --seed 6 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl/s6.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $T/l5.npz --out $T/l5.json > $T/l5.log 2>&1
$PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oracle3/o3.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $T/o3.npz --out $T/o3.json > $T/o3.log 2>&1
echo LOGITDONE
