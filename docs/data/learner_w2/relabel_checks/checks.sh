#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; R=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/relabel
F="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32"
nice -n 5 $PY w2/learner_w2_labels.py ${=F} --ckpt /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner3/snap/s1_mark4e7_at40304640.ckpt --count-iters 3 --out $R/smoke_default.json > /dev/null 2>&1
X="$F --steps 3.3e5 --curve-every 5 --seed 9 --lookahead 0.25"
nice -n 5 $PY w2/learner_w2_rl.py ${=X} --out $R/id_rl0_la25.json > $R/id_rl0_la25.log 2>&1
nice -n 5 $PY w2/learner_w2_rl.py ${=X} --head-relabel-token --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner3/s1.params --out $R/on_rl_la25.json > $R/on_rl_la25.log 2>&1
for C in /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner3/s1.ckpt /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner3/s2.ckpt /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/s4_3e7.ckpt; do
  nice -n 5 $PY w2/learner_w2_labels.py ${=F} --count-iters 100 --lookahead 0.25 --count-relabel --ckpt $C --out $R/yield_$(basename $C .ckpt).json >> $R/yield.log 2>&1; done
echo RLCHKDONE
