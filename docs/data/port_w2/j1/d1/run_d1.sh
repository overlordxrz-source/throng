#!/bin/zsh
# Cam Nov 23 ~17:10Z item 2 (D1-D3, eval only): seed 10 at 1e8 on the laptop, diag worktree d1wt (offset-proj abed942 + head-probs + mem-audit)
cd /Users/overlord/throng-backup/scratch/d1wt
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/d1
P=/Users/overlord/throng-backup/modal_b1/f3l/files/s10/s10_10e7.params
XT=(--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --d4-side --offset-proj)
echo "start $(date +%T) $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"
t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --seed 10 --init-params $P --d-fixed 12 --eval-necessity 0 --traj-out $O/off_d12.npz --out $O/off_d12.json > $O/off_d12.log 2>&1; echo "flagoff d12 m0 exit $? $(( $(date +%s)-t ))s"
t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --seed 10 --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --head-probs --mem-audit --traj-out $O/au_d12.npz --out $O/au_d12.json > $O/au_d12.log 2>&1; echo "audit d12 m0,3,4 exit $? $(( $(date +%s)-t ))s"
for d in 3 21; do t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --seed 10 --init-params $P --d-fixed $d --eval-necessity 0 --head-probs --mem-audit --traj-out $O/au_d$d.npz --out $O/au_d$d.json > $O/au_d$d.log 2>&1; echo "audit d$d m0 exit $? $(( $(date +%s)-t ))s"; done
echo "D1 EVALS DONE $(date +%T)"
