#!/bin/zsh
# 3e7 controls + early pilot table on seed 6 (registered in the protocol before running). Eval only, light.
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; D=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/diag_rl; P=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl/s6_3e7.params
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
echo "start $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_diag.py ${=X} --seed 6 --init-params $P --d-fixed 3 --eval-necessity 0,3,4 --out $D/d3.json > $D/d3.log 2>&1
for d in 4 6 8 12; do $PY w2/learner_w2_rl_diag.py ${=X} --seed 6 --init-params $P --d-fixed $d --eval-necessity 0 --out $D/d$d.json > $D/d$d.log 2>&1; echo "d$d $(date '+%H:%M:%S')"; done
echo "DIAGDONE $(date '+%H:%M:%S')"
