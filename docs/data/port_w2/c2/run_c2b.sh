#!/bin/zsh
# C2 lengthening (>= 300 hungry bouts per sector): the two 8-way events, eval-steps 6000, XLA single-threaded beside D1.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; C=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/c2
ev() {
  $PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --moves8 --init-params $2 --eval-necessity 0 --eval-d-max $3 --eval-steps 6000 --traj-out $C/$1L_mix.npz --out $C/$1L_mix.json > $C/$1L_mix.log 2>&1
  $PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 --moves8 --init-params $2 --eval-necessity 0 --d-fixed 21 --eval-steps 6000 --traj-out $C/$1L_d21.npz --out $C/$1L_d21.json > $C/$1L_d21.log 2>&1
  python3 $C/c2.py $1L $C/$1L_mix_m0.npz $C/$1L_d21_m0.npz 1 $4 > $C/$1L_c2.json
}
ev o8s0_2e7 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8/o8s0_2e7.params 24 2
ev o8s1_1e7 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8/o8s1_1e7.params 16 4,6
echo C2B_DONE > $C/doneb.txt
