#!/bin/zsh
# Cam Nov 21 ~21:20Z items 3-4: wait for seed 9's registered script (run_j1l.sh) to finish, then the 3-table verdict for
# seed 9. PASS -> STAGE-1 CONFIRM: seeds 10 then 11 back to back (run_seed.sh), each followed by its 3-table verdict;
# CONFIRMED if >= 2 of 3 (9 / 10 / 11) pass. FAIL -> stop (report; launch nothing).
J=/Users/overlord/CascadeProjects/throng/docs/data/port_w2/j1; SP=/Users/overlord/throng-backup/scratch
while pgrep -f "run_j1l.sh" > /dev/null; do sleep 60; done
echo "Q9 seed-9 script done $(date '+%H:%M:%S'): $(tail -1 $SP/j1l/run_j1l.out)"
$J/verdict3.sh 9 $SP/j1l
grep -q '"PASS": true' $SP/j1l/v3_s9_verdict.json || { echo "Q FAIL seed 9 (3-table): STOP, report, launch nothing $(date '+%H:%M:%S')"; exit 0; }
echo "Q seed 9 PASSES the 3-table verdict -> confirm: seeds 10, 11 $(date '+%H:%M:%S')"
NP=1
for S in 10 11; do
  $J/run_seed.sh $S > $SP/j1l_s$S.out 2>&1; tail -3 $SP/j1l_s$S.out
  $J/verdict3.sh $S $SP/j1l_s$S
  grep -q '"PASS": true' $SP/j1l_s$S/v3_s${S}_verdict.json && NP=$((NP + 1))
  echo "Q seed $S done; passes so far $NP $(date '+%H:%M:%S')"
done
[ $NP -ge 2 ] && echo "Q STAGE-1 CONFIRMED ($NP of 3) $(date '+%H:%M:%S')" || echo "Q STAGE-1 NOT CONFIRMED ($NP of 3) $(date '+%H:%M:%S')"
