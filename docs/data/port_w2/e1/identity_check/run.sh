#!/bin/zsh
# E1 checks (Cam Nov 17 ~16:40Z): (1) flag-off identity vs d1/identity_check/port_m8_log16 (HEAD 97cb49a, same config);
# (2) --target-kl 1e9 (the upd_kl path, never stops) vs the same; (3) synthetic high-KL batch: shift 1.0, target .015 -> 0 steps
# applied, dparam 0; (4) real --target-kl .015 on 60 iterations: fraction of minibatch steps cut, and the time cost.
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; I=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/e1id
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --ppo-log"
$PY w2/port_w2.py ${=B} --steps 491520 --out $I/off.json > $I/off.log 2>&1; echo "off $(date '+%H:%M:%S')"
$PY w2/port_w2.py ${=B} --target-kl 1e9 --steps 491520 --out $I/big.json > $I/big.log 2>&1; echo "big $(date '+%H:%M:%S')"
$PY w2/port_w2.py ${=B} --target-kl 0.015 --tkl-synthetic-shift 1.0 --steps 40960 --out $I/syn.json > $I/syn.log 2>&1; echo "syn $(date '+%H:%M:%S')"
$PY w2/port_w2.py ${=B} --target-kl 0.015 --steps 491520 --out $I/on.json > $I/on.log 2>&1; echo "on $(date '+%H:%M:%S')"
echo DONE
