#!/bin/zsh
# separating test: step 0 to 2.5e5 steps (exact reproduction), then d = 3 evals true / wiped / shuffled
until grep -q STEP0DONE /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/bisect/step0/run.out; do sleep 20; done
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; B=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/bisect/step0e
X="--d 12 --hidden --lam 0.95 --gamma 0.99 --tier T3 --learner L2e --kl 0.3 --B 256 --T 32 --seed 8 --episodes-per-life 64 --epochs 4 --minibatches 8 --lr 0.0003 --evo-pop 32 --evo-elite 8 --evo-eps 64 --evo-sigma 0.3 --evo-sigma-body 0.02 --curriculum --curriculum-mix --d-max 24 --hindsight 0.5 --hind-at-start --shared-tok --demo-hearings 1 --demonstrator --move-cost 0.003 --curve-every 10"
$PY w2/toy_grid.py ${=X} --steps 2.5e5 --test-d 12 --save-params $B/s8_2p5e5.params --skip-final-eval --eval-episodes 512 --count-labels --out $B/s8_2p5e5_train.json > $B/run.log 2>&1
$PY w2/toy_grid.py ${=X} --steps 1e6 --test-d 3 --eval-params $B/s8_2p5e5.params --eval-modes 0,4,3 --eval-episodes 9000 --out $B/s8_2p5e5_eval_d3.json >> $B/run.log 2>&1
echo SEPDONE
