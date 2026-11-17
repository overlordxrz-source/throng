#!/bin/zsh
# C2 evals (eval only, light, beside D1): each event's checkpoint BEFORE the event; d ~ uniform 1..F and d = 21.
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; C=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/c2
ev() {  # label params flags F moves8 dying
  $PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 ${=3} --init-params $2 --eval-necessity 0 --eval-d-max $4 --traj-out $C/$1_mix.npz --out $C/$1_mix.json > $C/$1_mix.log 2>&1
  $PY w2/learner_w2_rl_traj.py --learner rec --oracle --hunger-gate --item-size 0.1 --W 32 ${=3} --init-params $2 --eval-necessity 0 --d-fixed 21 --traj-out $C/$1_d21.npz --out $C/$1_d21.json > $C/$1_d21.log 2>&1
  python3 $C/c2.py $1 $C/$1_mix_m0.npz $C/$1_d21_m0.npz $5 $6 > $C/$1_c2.json
}
ev o8s0_2e7 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8/o8s0_2e7.params "--moves8" 24 1 2
ev o8s1_1e7 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8/o8s1_1e7.params "--moves8" 16 1 4,6
ev o4_5e7 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/l4/o4_5e7.params "--offset-wide" 24 0 5
ev o4_9e7 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/l4/o4_9e7.params "--offset-wide" 24 0 1,2
ev bclip_3e7 /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/chain/B.params "" 24 0 1
echo C2_DONE > $C/done.txt
