#!/bin/zsh
# waits for the clipped oracle continuation to end, folds the L4 copies into the originals, classifies the continuation
# (registered classes vs chain B's 3e7 table), then runs the L4 sequence.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; L4=$SP/l4
until grep -q "OC_DONE\|CRASH" $SP/oc/run_oc.out; do sleep 60; done
grep -q CRASH $SP/oc/run_oc.out && { echo "OC crashed; L4 not started"; exit 1; }
python3 $L4/lockcheck.py $SP/chain/B_verdict.json $SP/oc/oc_1e8_verdict.json > $SP/oc/oc_1e8_class.json; echo "OC 1e8 CLASS $(cat $SP/oc/oc_1e8_class.json)"
python3 $L4/lockcheck.py $SP/chain/B_verdict.json $SP/oc/oc_5e7_verdict.json > $SP/oc/oc_5e7_class.json
cd /Users/overlord/CascadeProjects/throng-toy && mv w2/port_w2_ow.py w2/port_w2.py && mv w2/learner_w2_rl_traj_ow.py w2/learner_w2_rl_traj.py && echo "L4 copies folded in $(date '+%H:%M:%S')"
$L4/run_l4.sh
