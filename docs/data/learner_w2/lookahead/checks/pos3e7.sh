#!/bin/zsh
# registered deviation (Nov 16 01:20): positive control rerun with 3e7 steps, the only change; on frontier 24 -> controls_pass
until grep -q PDONE /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/labels/count_pilot.log 2>/dev/null; do sleep 20; done
cd /Users/overlord/CascadeProjects/throng-toy
/Users/overlord/CascadeProjects/throng/.venv/bin/python w2/port_w2.py --oracle --hunger-gate --item-size 0.1 --W 32 --steps 3e7 --policy follower --v2 /Users/overlord/throng-backup/evo2/evo2.ckpt --gate honest --curve-every 100 --lookahead 0.25 --out /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/checks/ctl_pos_la_3e7.json > /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/checks/ctl_pos_la_3e7.log 2>&1
F=$(python3 -c "import json;d=json.load(open('/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/checks/ctl_pos_la_3e7.json'));print(d['frontier'], len(d['gate_scores']), min(d['gate_scores']))")
echo "pos 3e7: frontier windows min_window = $F"
python3 -c "import sys; sys.exit(0 if float('$F'.split()[0]) >= 24 else 1)" && { echo ok > /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/controls_pass; echo CONTROLS_PASS; } || echo CONTROLS_FAIL
echo POSDONE
