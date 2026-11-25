#!/bin/zsh
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/twchk
X=(--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token --d4-side --offset-proj --demo-hearings 2 --no-nav --target-kl 0.015 --seed 11 --steps 24576)
for W in opwt v3wt; do cd /Users/overlord/throng-backup/scratch/$W; echo "$W $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"; $PY w2/learner_w2_rl2.py $X --save-params $O/ep_$W.params --out $O/ep_$W.json > $O/ep_$W.log 2>&1; echo "$W exit $?"; done
cmp -s $O/ep_opwt.params $O/ep_v3wt.params && echo "EP LEARNER TRAINING BITWISE EQUAL" || echo "EP LEARNER DIFFERS"
