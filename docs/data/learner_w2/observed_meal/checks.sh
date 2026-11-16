#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; C=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/obsmeal
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --steps 3.3e5 --curve-every 5 --seed 9"
nice -n 5 $PY w2/learner_w2_obs.py ${=X} --out $C/id_obs0_la0.json > $C/id_obs0_la0.log 2>&1
nice -n 5 $PY w2/learner_w2_la.py ${=X} --lookahead 0.25 --out $C/id_la25.json > $C/id_la25.log 2>&1
nice -n 5 $PY w2/learner_w2_obs.py ${=X} --lookahead 0.25 --out $C/id_obs0_la25.json > $C/id_obs0_la25.log 2>&1
nice -n 5 $PY w2/learner_w2_obs.py ${=X} --lookahead 0.25 --head-observed-meal --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner3/s1.params --a5-log $C/a5.npz --out $C/on_la25.json > $C/on_la25.log 2>&1
echo OBSCHKDONE
