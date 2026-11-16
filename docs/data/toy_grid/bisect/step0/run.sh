#!/bin/zsh
# BISECTION STEP 0 (Cam Nov 16 ~10:15Z): toy screen 2 seed 8 EXACTLY (committed cfg) + --count-labels (logging only); then evals at d 3 and 12
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; B=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/bisect/step0
X="--d 12 --hidden --lam 0.95 --gamma 0.99 --tier T3 --learner L2e --kl 0.3 --B 256 --T 32 --seed 8 --episodes-per-life 64 --epochs 4 --minibatches 8 --lr 0.0003 --evo-pop 32 --evo-elite 8 --evo-eps 64 --evo-sigma 0.3 --evo-sigma-body 0.02 --curriculum --curriculum-mix --d-max 24 --hindsight 0.5 --hind-at-start --shared-tok --demo-hearings 1 --demonstrator --move-cost 0.003 --curve-every 10"
for i in $(seq 1 200); do $PY w2/toy_grid.py ${=X} --steps 2e7 --test-d 12 --save-params $B/s8.params --skip-final-eval --ckpt $B/s8.ckpt --ckpt-every 20 --max-seconds 480 --eval-episodes 512 --count-labels --out $B/s8_train.json >> $B/s8.log 2>&1
  [ -f $B/s8_train.json ] && break; grep -q Traceback $B/s8.log && { echo CRASH; exit 0; }; done
for TD in 3 12; do $PY w2/toy_grid.py ${=X} --steps 1e6 --test-d $TD --eval-params $B/s8.params --eval-modes 0,3,4 --eval-episodes 9000 --out $B/s8_eval_d$TD.json >> $B/s8.log 2>&1; done
echo STEP0DONE
