#!/bin/zsh
# Waits for the chain; launches the learner screen only if an arm QUALIFIED (lever from the arm). The tie never enters the chain.
CH=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/chain; L=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_rl2
until grep -qE "QUALIFIED|CHAIN_NONE_QUALIFIED|CRASH" $CH/chain.out 2>/dev/null; do sleep 60; done
[ -f $CH/QUALIFIED ] || { echo "no arm qualified; no learner screen $(date '+%H:%M:%S')"; exit 0; }
ARM=$(python3 -c "import json;print(json.load(open('$CH/QUALIFIED'))['arm'])"); REF=$(python3 -c "import json;print(json.load(open('$CH/QUALIFIED'))['ref_file'])")
case $ARM in A) LEV="--reinit-out";; XP) LEV="--ent-bout 0.05 --ent-anneal 2e7";; B) LEV="--no-nav";; *) echo "unknown arm $ARM"; exit 1;; esac
echo "arm $ARM qualified -> learner screen with $LEV"
$L/run_rl2.sh "$LEV" "$REF"
echo RL2DONE
