#!/bin/zsh
# D1 counterfactual (registered 20:27:39 stamp): --demo-hearings 2, eval only; waits for the long evals (one job at a time)
until grep -q "LONG EVALS DONE" /Users/overlord/throng-backup/scratch/d1/run_d1_long.out; do sleep 10; done
cd /Users/overlord/throng-backup/scratch/d1wt
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/d1
P=/Users/overlord/throng-backup/modal_b1/f3l/files/s10/s10_10e7.params
XT=(--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --d4-side --offset-proj)
echo "start $(date +%T) $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"
t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --demo-hearings 2 --seed 10 --init-params $P --d-fixed 12 --eval-necessity 0 --eval-steps 24000 --head-probs --mem-audit --traj-out $O/CF2_d12.npz --out $O/CF2_d12.json > $O/CF2_d12.log 2>&1; echo "cf demo2 d12 m0 exit $? $(( $(date +%s)-t ))s"
echo "CF DONE $(date +%T)"
