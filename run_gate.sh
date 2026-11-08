#!/bin/zsh
cd /Users/overlord/CascadeProjects/throng-toy
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
C=(--updates 1000 --warm-start toy_out/ORACLE0.msgpack --wire-init-scale 0.06 --craft-cost 0.45)
for s in 0 1 2; do
  $PY scripts/comm_toy.py --arm NOSIG --seed $s $C --out toy_out/gate_NOSIG$s.jsonl > toy_out/log_NOSIG$s.txt 2>&1 &
done
wait
for s in 0 1 2; do
  $PY scripts/comm_toy.py --arm COMM --seed $s $C --out toy_out/gate_BASE$s.jsonl > toy_out/log_BASE$s.txt 2>&1 &
done
wait
echo done
