#!/bin/zsh
until grep -q CHKDONE /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/checks/checks.out; do sleep 20; done
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
F="--learner ep --slots 96 --shared-tok --hindsight 0.5 --demonstrator --hunger-gate --item-size 0.1 --W 32 --count-iters 100"
for C in s0_0925 s0_1001 s0_1013 s0_1037 s0_1049 s0_1101 s0_1125; do
  nice -n 10 $PY w2/learner_w2_labels.py ${=F} --ckpt /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner/snap/$C.ckpt --out /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/labels/pilot_$C.json >> /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/labels/count_pilot.log 2>&1; done
nice -n 10 $PY w2/learner_w2_labels.py ${=F} --ckpt /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner/s0.ckpt --out /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/labels/pilot_s0.json >> /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/labels/count_pilot.log 2>&1
echo PDONE >> /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/labels/count_pilot.log
