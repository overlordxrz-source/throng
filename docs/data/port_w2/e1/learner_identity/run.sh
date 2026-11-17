#!/bin/zsh
# E1 (a) branch, learner fold-in checks on the merged worktree (moves8-learner + comm-toy + rl2 --target-kl). References are the
# pre-merge branch runs in docs/data/port_w2/m8/learner_identity (e94156f). Run AFTER the E1 twin (one training job at a time).
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; I=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/e1lid
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
cd /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8wt
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --seed 7 --steps 491520 --out $I/rl2_off.json > $I/rl2_off.log 2>&1; echo "off $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --seed 8 --steps 163840 --save-params $I/rl2_on.params --out $I/rl2_on.json > $I/rl2_on.log 2>&1; echo "on $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --target-kl 1e9 --seed 8 --steps 163840 --out $I/rl2_big.json > $I/rl2_big.log 2>&1; echo "big $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --target-kl 0.015 --tkl-synthetic-shift 1.0 --seed 8 --steps 40960 --out $I/rl2_syn.json > $I/rl2_syn.log 2>&1; echo "syn $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --target-kl 0.015 --seed 8 --steps 491520 --out $I/rl2_tkl.json > $I/rl2_tkl.log 2>&1; echo "tkl $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_traj.py ${=XT} --moves8 --seed 8 --init-params /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8lid/rl2_on.params --d-fixed 3 --eval-necessity 0 --traj-out $I/traj_on_d3.npz --out $I/traj_on_d3.json > $I/traj_on.log 2>&1; echo "traj $(date '+%H:%M:%S')"
echo DONE
