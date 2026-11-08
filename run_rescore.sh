#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
C=(--arm COMM --updates 1000 --warm-start toy_out/ORACLE0.msgpack --wire-init-scale 0.06 --craft-cost 0.45 --ps 1.99)
for s in 0 1 2; do $PY scripts/comm_toy.py $C --seed $s --save-params toy_out/PS${s}_final.msgpack --out toy_out/rescore_PS$s.jsonl > toy_out/log_rescore_PS$s.txt 2>&1 & done; wait
echo done
