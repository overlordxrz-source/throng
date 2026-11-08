#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
C=(--arm COMM --updates 1000 --warm-start toy_out/ORACLE0.msgpack --wire-init-scale 0.06 --craft-cost 0.45)
for s in 0 1 2; do $PY scripts/comm_toy.py $C --seed $s --ps 1.99 --out toy_out/aux_PS$s.jsonl > toy_out/log_PS$s.txt 2>&1 & done; wait
for s in 0 1 2; do $PY scripts/comm_toy.py $C --seed $s --pl 0.114 --out toy_out/aux_PL$s.jsonl > toy_out/log_PL$s.txt 2>&1 & done; wait
for s in 0 1 2; do $PY scripts/comm_toy.py $C --seed $s --ps 1.99 --pl 0.114 --out toy_out/aux_BOTH$s.jsonl > toy_out/log_BOTH$s.txt 2>&1 & done; wait
echo done
