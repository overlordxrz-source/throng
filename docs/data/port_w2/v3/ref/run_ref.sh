#!/bin/zsh
# Cam Nov 24 ~23:55Z chain (a): the F3 REFERENCE on the v3 class, free, laptop. F3 oracle s0 at 8e7 / 9e7 / 1e8, d12, 24k steps,
# eval world (i) as before (no demonstrator) and (ii) + --demonstrator --parent-waits --demo-hearings 2. Code: frozen worktree v3frz at the v3-metric commit.
until grep -q "CHK DONE" /Users/overlord/throng-backup/scratch/v3chk/run_chk.out; do sleep 10; done   # one job at a time
cd /Users/overlord/throng-backup/scratch/v3frz; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/v3chk/ref; mkdir -p $O
Q=/Users/overlord/throng-backup/modal_b1/f3/files/o0
E=(--learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --d4-side --offset-proj)
echo "start $(date +%T) $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"
for K in 8 9 10; do
  t=$(date +%s); $PY w2/learner_w2_rl_traj.py $E --seed 0 --init-params $Q/f3s0_${K}e7.params --d-fixed 12 --eval-necessity 0 --eval-steps 24000 --out $O/ref_i_${K}e7.json > $O/ref_i_${K}e7.log 2>&1; echo "ref (i) ${K}e7 exit $? $(( $(date +%s)-t ))s"
  t=$(date +%s); $PY w2/learner_w2_rl_traj.py $E --demonstrator --parent-waits --demo-hearings 2 --seed 0 --init-params $Q/f3s0_${K}e7.params --d-fixed 12 --eval-necessity 0 --eval-steps 24000 --out $O/ref_ii_${K}e7.json > $O/ref_ii_${K}e7.log 2>&1; echo "ref (ii) ${K}e7 exit $? $(( $(date +%s)-t ))s"
done
echo "REF DONE $(date +%T)"
