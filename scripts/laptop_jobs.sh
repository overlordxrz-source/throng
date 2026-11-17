#!/bin/zsh
# Cam Nov 16 ~08:25Z: list / pause / resume / stop every DETACHED laptop job (started via <scratchpad>/detach.py = fork + setsid,
# so quitting the app does not kill them). M's pause word must stop all of them, the spend guard LAST.
#   scripts/laptop_jobs.sh list     # every job: pid, process group, command
#   scripts/laptop_jobs.sh pause    # kill -STOP the training / analysis python (exact resume with 'resume'); queues and guard keep running
#   scripts/laptop_jobs.sh resume   # kill -CONT them
#   scripts/laptop_jobs.sh stop     # stop files, then queues + watchers, then the python jobs, then (last) the guard
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad
QUEUES="run_e1.sh|run_d1.sh|run_c2.sh|run_fun.sh|queue_m8l.sh|run_m8l.sh|run_m8.sh|queue_l4.sh|run_l4.sh|snap_oc.py|snap_oc2.py|run_oc.sh|chain.sh|queue_rl2.sh|run_rl2.sh|run_greedy.sh|queue_os.sh|run_os.sh|queue_rl.sh|run_rl.sh|run_diag.sh|after_5e7.sh|run_traj.sh|run_logit.sh|queue_la.sh|queue_k.sh|queue2.sh|run_la.sh|run_k.sh|post.sh|post2.sh|checks.sh|readout_s2.sh|pos3e7.sh|count_pilot.sh|snap_la.py|snap_k.py|snap2.py"
PYJOBS="w2/learner_w2|w2/port_w2"
case "$1" in
  list)   ps -axo pid,pgid,stat,etime,command | grep -E "cap_guard_r3|$QUEUES|$PYJOBS" | grep -v grep | cut -c1-160 ;;
  pause)  pkill -STOP -f "$PYJOBS"; echo "paused (STOP): $(pgrep -f "$PYJOBS" | tr '\n' ' ')" ;;
  resume) pkill -CONT -f "$PYJOBS"; echo "resumed (CONT)" ;;
  stop)   touch $SP/learner_la/stop $SP/learner_k/stop $SP/learner3/stop 2>/dev/null
          pkill -f "$QUEUES"; sleep 1; pkill -CONT -f "$PYJOBS"; pkill -f "$PYJOBS"; sleep 2
          echo "left (should be only the guard):"; ps -axo pid,command | grep -E "cap_guard_r3|$QUEUES|$PYJOBS" | grep -v grep
          pkill -f cap_guard_r3.sh; sleep 1; echo "guard stopped LAST; remaining: $(pgrep -f cap_guard_r3 | wc -l | tr -d ' ')"
          echo "Resume later: checkpoints are exact (.ckpt); restart each queue with detach.py (see the handoff); the guard with CAP=41.0." ;;
  *) echo "usage: $0 list|pause|resume|stop"; exit 2 ;;
esac
