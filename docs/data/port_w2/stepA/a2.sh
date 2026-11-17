#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; A=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/stepA; L4=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/l4
E="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --offset-wide"
for K in 6 7 8 9; do $PY w2/learner_w2_rl_traj.py ${=E} --init-params $L4/o4_${K}e7.params --d-fixed 3 --eval-necessity 0 --traj-out $A/o4_${K}e7_d3.npz --out $A/o4_${K}e7_d3.json > $A/o4_${K}e7_d3.log 2>&1; python3 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oscreen/bysec_o.py $A/o4_${K}e7_d3_m0.npz >> $A/bysec_6to9.txt 2>&1; done
$PY w2/learner_w2_rl_traj.py ${=E} --init-params $L4/o4_10e7.params --d-fixed 3 --eval-necessity 0 --logits --traj-out $A/o4_1e8_log.npz --out $A/o4_1e8_log.json > $A/o4_1e8_log.log 2>&1
$PY w2/logit_split.py --oracle-files $A/o4_1e8_log_m0.npz --out $A/o4_1e8_logit_split.json >> $A/o4_1e8_log.log 2>&1
echo A2_DONE >> $A/bysec_6to9.txt
