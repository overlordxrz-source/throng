#!/bin/bash
# THRONG keep-alive (Cam Nov 22 ~03:32Z; M's card: "Yes, Will installs it"). Run by ONE user-crontab line every minute:
#   * * * * * /Users/overlord/CascadeProjects/throng/scripts/ensure_running.sh
# Remove with:  crontab -l | grep -v ensure_running.sh | crontab -
# cron is not attributed to any app, so an app quit / update cannot take down what it starts.
# It (re)starts ONLY our processes: the guard supervisor (which keeps both spend guards up) and the ONE active laptop run
# named in ~/throng-backup/logs/active_run (idempotent resume). It never starts a finished run (its done-pattern is found)
# or anything on Modal. A lock dir prevents overlapping runs. Every start is logged to ~/throng-backup/logs/ensure.log.
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin; H=/Users/overlord; LOG=$H/throng-backup/logs/ensure.log; LOCK=$H/throng-backup/logs/ensure.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  p=$(cat "$LOCK/pid" 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then exit 0; fi
  rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0; echo "$(date '+%b %d %H:%M:%S') stale lock removed" >> $LOG
fi
echo $$ > "$LOCK/pid"; trap 'rm -rf "$LOCK"' EXIT
log() { echo "$(date '+%b %d %H:%M:%S') $*" >> $LOG; }
# 1) guard supervisor (it starts / restarts cap_guard_r3 and cap_guard_b1 itself within 30 s)
if ! pgrep -f "guard_supervise[.]sh" > /dev/null; then
  log "START guard supervisor"; (cd $H/throng-smoke && nohup ./guard_supervise.sh > /dev/null 2>&1 &)
fi
# 2) the active laptop run, if any. active_run lines: NAME= DIR= RESUME= PROC= DONE_FILE= DONE_RE=
A=$H/throng-backup/logs/active_run
if [ -f "$A" ]; then
  NAME=$(sed -n 's/^NAME=//p' $A); DIR=$(sed -n 's/^DIR=//p' $A); RESUME=$(sed -n 's/^RESUME=//p' $A)
  PROC=$(sed -n 's/^PROC=//p' $A); DONE_FILE=$(sed -n 's/^DONE_FILE=//p' $A); DONE_RE=$(sed -n 's/^DONE_RE=//p' $A)
  if grep -qE "$DONE_RE" "$DONE_FILE" 2>/dev/null; then :          # finished: never restart
  elif ! pgrep -f "$PROC" > /dev/null; then
    log "START $NAME resume ($RESUME in $DIR)"; (cd "$DIR" && nohup /bin/zsh -c "$RESUME" > /dev/null 2>&1 &)
  fi
fi
