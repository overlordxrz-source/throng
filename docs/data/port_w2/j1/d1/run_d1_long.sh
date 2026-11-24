#!/bin/zsh
# D1-D3 at the bar's own horizon (24,000 eval steps; bar3 reach_eval_steps 24000): same worktree / params / flags as run_d1.sh
cd /Users/overlord/throng-backup/scratch/d1wt
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/d1
P=/Users/overlord/throng-backup/modal_b1/f3l/files/s10/s10_10e7.params
XT=(--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --d4-side --offset-proj)
echo "start $(date +%T) $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"
t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --seed 10 --init-params $P --d-fixed 12 --eval-necessity 0,3,4 --eval-steps 24000 --head-probs --mem-audit --traj-out $O/L_d12.npz --out $O/L_d12.json > $O/L_d12.log 2>&1; echo "long audit d12 m0,3,4 exit $? $(( $(date +%s)-t ))s"
for d in 3 21; do t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --seed 10 --init-params $P --d-fixed $d --eval-necessity 0 --eval-steps 24000 --head-probs --mem-audit --traj-out $O/L_d$d.npz --out $O/L_d$d.json > $O/L_d$d.log 2>&1; echo "long audit d$d m0 exit $? $(( $(date +%s)-t ))s"; done
echo "D1 LONG EVALS DONE $(date +%T)"
