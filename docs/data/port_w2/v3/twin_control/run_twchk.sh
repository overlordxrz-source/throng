#!/bin/zsh
# TWIN-K2 positive control: port_w2 (abed942 = opwt) vs rl2 rec + D4 (v3wt), twin flags, demonstrator OFF, 3 iterations, bitwise params.
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; O=/Users/overlord/throng-backup/scratch/twchk; S=$O/sched12.json
T=(--hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --offset-proj --frontier-schedule $S --steps 24576)
cd /Users/overlord/throng-backup/scratch/opwt; echo "opwt $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"
$PY w2/port_w2.py $T --save-params $O/pw.params --out $O/pw_train.json > $O/pw.log 2>&1; echo "port_w2 exit $?"
cd /Users/overlord/throng-backup/scratch/v3wt; echo "v3wt $(git rev-parse --short HEAD) dirty: $(git status --short | tr '\n' ' ')"
$PY w2/learner_w2_rl2.py --learner rec $T --save-params $O/r2.params --out $O/r2_train.json > $O/r2.log 2>&1; echo "rl2 rec d4 exit $?"
$PY w2/learner_w2_rl2.py --learner rec --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --offset-proj --frontier-schedule $S --steps 24576 --save-params $O/r2_nod4.params --out $O/r2_nod4_train.json > $O/r2_nod4.log 2>&1; echo "MUTANT rl2 rec no-d4 exit $?"
$PY w2/learner_w2_rl2.py --learner rec --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --target-kl 0.015 --d4-side --offset-proj --frontier0 1 --steps 24576 --save-params $O/r2_nosched.params --out $O/r2_nosched_train.json > $O/r2_nosched.log 2>&1; echo "MUTANT rl2 no-schedule exit $?"
echo "TWCHK DONE $(date +%T)"
