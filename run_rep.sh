#!/bin/zsh
# PS replication (registered 2026-11-07): fresh seeds, 2,000 updates, eval also at 1,000.
cd /Users/overlord/CascadeProjects/throng-toy
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
C=(--updates 2000 --eval-at 1000 --warm-start toy_out/ORACLE0.msgpack --wire-init-scale 0.06 --craft-cost 0.45)
for s in 10 11 12 13; do $PY scripts/comm_toy.py --arm COMM $C --seed $s --ps 1.99 --tag PS --save-params toy_out/rep_PS$s.msgpack --out toy_out/rep_PS$s.jsonl > toy_out/log_rep_PS$s.txt 2>&1 & done; wait
$PY scripts/comm_toy.py --arm COMM $C --seed 14 --ps 1.99 --tag PS --save-params toy_out/rep_PS14.msgpack --out toy_out/rep_PS14.jsonl > toy_out/log_rep_PS14.txt 2>&1 &
for s in 20 21 22; do $PY scripts/comm_toy.py --arm COMM $C --seed $s --ps 1.99 --ps-anneal 1000 1500 --tag PSA --save-params toy_out/rep_PSA$s.msgpack --out toy_out/rep_PSA$s.jsonl > toy_out/log_rep_PSA$s.txt 2>&1 & done; wait
for s in 30 31 32; do $PY scripts/comm_toy.py --arm NOSIG $C --seed $s --out toy_out/rep_NOSIG$s.jsonl > toy_out/log_rep_NOSIG$s.txt 2>&1 & done; wait
echo done
