#!/bin/zsh
# control 2: 25 iterations with the REAL F3 schedule and checkpoints (it 20), port_w2 vs rl2 rec d4; then the K=2 twin path (demonstrator on) + its life eval
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/twchk; S=/Users/overlord/throng-backup/modal_b1/f3/files/o0/f3s0_10e7_train.json
T=(--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --offset-proj --frontier-schedule $S --steps 204800)
cd /Users/overlord/throng-backup/scratch/opwt; $PY w2/port_w2.py $T --ckpt $O/pw25.ckpt --save-params $O/pw25.params --out $O/pw25_train.json > $O/pw25.log 2>&1; echo "port_w2 25it exit $?"
cd /Users/overlord/throng-backup/scratch/v3wt; $PY w2/learner_w2_rl2.py --learner rec $T --ckpt $O/r225.ckpt --save-params $O/r225.params --out $O/r225_train.json > $O/r225.log 2>&1; echo "rl2 25it exit $?"
$PY w2/learner_w2_rl2.py --learner rec $T[1,-3] --steps 40960 --demonstrator --parent-waits --demo-hearings 2 --save-params $O/k2.params --out $O/k2_train.json > $O/k2.log 2>&1; echo "rl2 twin K2 5it exit $?"
$PY w2/learner_w2_rl_traj.py --learner rec --hunger-gate --item-size 0.1 --W 32 --d4-side --offset-proj --demonstrator --parent-waits --demo-hearings 2 --init-params $O/k2.params --d-fixed 12 --eval-necessity 0 --out $O/k2_eval.json > $O/k2_eval.log 2>&1; echo "twin K2 life eval exit $?"
echo "TWCHK2 DONE $(date +%T)"
