#!/bin/zsh
# Waits for run_m8.sh. Both oracles qualified + twin done -> fold in the learner 8-way build (fast-forward merge of branch
# moves8-learner into comm-toy, nothing running) -> run_m8l.sh. A STOP or CRASH -> exit without running the learner.
M=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/m8
until grep -q "M8_ORACLE_TWIN_DONE\|M8 STOP\|CRASH" $M/run_m8.out 2>/dev/null; do sleep 60; done
grep -q "M8_ORACLE_TWIN_DONE" $M/run_m8.out || { echo "no learner: $(grep 'STOP\|CRASH' $M/run_m8.out)"; exit 0; }
[ -f $M/LEARNER_BUILD_OK ] || { echo "no learner: the 8-way learner build is not marked OK"; exit 0; }
sleep 5; pgrep -f "run_m8.sh" > /dev/null && { echo "run_m8.sh still alive; not folding in"; exit 1; }
git -C /Users/overlord/CascadeProjects/throng-toy merge --ff-only moves8-learner > $M/foldin.log 2>&1 || { echo "fold-in FAILED (see foldin.log)"; exit 1; }
echo "learner 8-way build folded in $(date '+%H:%M:%S')"
$M/run_m8l.sh > $M/run_m8l.out 2>&1
echo "M8L ended $(date '+%H:%M:%S')"
