#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; G=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/greedy; O=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oscreen
B="--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --eval-necessity 0 --greedy-bout"
for d in 3 12; do
  $PY w2/learner_w2_rl_traj.py ${=B} --init-params $O/X.params --d-fixed $d --traj-out $G/X_d$d.npz --out $G/X_d$d.json > $G/X_d$d.log 2>&1
  $PY w2/learner_w2_rl_traj.py ${=B} --topo --init-params $O/T.params --d-fixed $d --traj-out $G/T_d$d.npz --out $G/T_d$d.json > $G/T_d$d.log 2>&1
  $PY w2/learner_w2_rl_traj.py ${=B} --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oracle3/o3.params --d-fixed $d --traj-out $G/o3_d$d.npz --out $G/o3_d$d.json > $G/o3_d$d.log 2>&1
done
python3 $O/bysec_o.py $G/X_d3_m0.npz $G/X_d12_m0.npz $G/T_d3_m0.npz $G/T_d12_m0.npz $G/o3_d3_m0.npz $G/o3_d12_m0.npz > $G/bysec.txt 2>&1
echo GREEDYDONE
