#!/bin/zsh
# D4 (Cam Nov 24 ~23:55Z item 6; registered 19:17:02 stamp) + the mode-3 K=2 dem<=>index check. Eval only, laptop, diag worktree.
until grep -q "REF DONE" /Users/overlord/throng-backup/scratch/v3chk/run_ref.out; do sleep 10; done   # one job at a time
cd /Users/overlord/throng-backup/scratch/d1wt; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/d4
P=/Users/overlord/throng-backup/modal_b1/f3l/files/s10/s10_10e7.params
XT=(--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --d4-side --offset-proj)
echo "start $(date +%T) $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"
t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --seed 10 --init-params $P --d-fixed 12 --eval-necessity 0 --eval-steps 24000 --mem-audit --birth-audit --traj-out $O/d4_d12.npz --out $O/d4_d12.json > $O/d4_d12.log 2>&1; echo "d4 K1 d12 24k exit $? $(( $(date +%s)-t ))s"
t=$(date +%s); $PY w2/learner_w2_rl_traj.py $XT --demo-hearings 2 --seed 10 --init-params $P --d-fixed 12 --eval-necessity 0,3 --mem-audit --traj-out $O/k2idx.npz --out $O/k2idx.json > $O/k2idx.log 2>&1; echo "K2 index check evals exit $? $(( $(date +%s)-t ))s"
echo "D4 DONE $(date +%T)"
