# THRONG Safe Restart Workflow

1. pkill -9 -f run_bg.py
2. sleep 3
3. pgrep -a -f run_bg.py  ← must return empty before continuing
4. Verify config change took effect with grep
5. Launch single Popen instance with `start_new_session=True`
6. sleep 5
7. pgrep -a -f run_bg.py  ← must return exactly ONE pid
8. tail first 10 lines of train.log to confirm clean startup
