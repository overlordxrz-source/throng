#!/bin/zsh
# Cam ~12:30Z: read forms but frontier <= 6 at 5e7 -> the pilot table on the 5e7 weights. Waits for seed 6's queue to finish.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; L=$SP/learner_rl; D=$SP/diag_rl5
until grep -qE "EARLY_STOP_B|PASS|FAIL|CRASH|EARLY_STOP_A" $L/queue_rl.out; do sleep 60; done
echo "queue ended: $(tail -1 $L/queue_rl.out) at $(date '+%H:%M:%S')"
grep -q EARLY_STOP_B $L/queue_rl.out || { echo "not an early stop B; no pilot table run"; exit 0; }
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; P=$L/s6.params
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
$PY w2/learner_w2_rl_diag.py ${=X} --seed 6 --init-params $P --d-fixed 3 --eval-necessity 0,3,4 --out $D/d3.json > $D/d3.log 2>&1
for d in 4 6 8 12; do $PY w2/learner_w2_rl_diag.py ${=X} --seed 6 --init-params $P --d-fixed $d --eval-necessity 0 --out $D/d$d.json > $D/d$d.log 2>&1; done
echo "DIAG5DONE $(date '+%H:%M:%S')"
