#!/bin/zsh
# Re-armed Nov 21 (after the 16:10 death of seed 9's first script and the VOID 16:16 verdict): wait for seed 9's resume
# script; only if run_j1l.out ends with its own "J1L S9 PASS|FAIL" line, run the 3-table verdict. Cam 21:40Z: seeds 10 / 11
# go to Modal (in parallel) after a PASS, so the laptop confirm chain is DISARMED; this queue stops after seed 9's verdict.
J=/Users/overlord/throng-backup/scratch/confirm_code; SP=/Users/overlord/throng-backup/scratch
while pgrep -f "run_j1l(_resume)?[.]sh" > /dev/null; do sleep 60; done
if ! grep -qE "^J1L S9 (PASS|FAIL) " $SP/j1l/run_j1l.out; then
  echo "Q ALERT $(date '+%H:%M:%S'): seed 9's script ended WITHOUT its verdict line; no verdict run. Last line: $(tail -1 $SP/j1l/run_j1l.out)"
  osascript -e 'display notification "seed 9 script ended without a verdict line" with title "THRONG"' 2>/dev/null; exit 1
fi
echo "Q9 seed-9 script done $(date '+%H:%M:%S'): $(grep -E '^J1L S9 (PASS|FAIL) ' $SP/j1l/run_j1l.out | tail -1)"
$J/verdict3.sh 9 $SP/j1l
echo "Q seed 9 3-table verdict: $(python3 -c "import json;v=json.load(open('$SP/j1l/v3_s9_verdict.json'));print('PASS' if v['PASS'] else 'FAIL')") $(date '+%H:%M:%S') (laptop chain disarmed; Modal per Cam 21:40Z)"
