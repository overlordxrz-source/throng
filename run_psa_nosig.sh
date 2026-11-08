#!/bin/zsh
# Exploratory control (2026-11-07, not registered): PS-anneal with the wire off -- does the code persist without a listener?
cd /Users/overlord/CascadeProjects/throng-toy
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
C=(--arm NOSIG --updates 2000 --eval-at 1000 --warm-start toy_out/ORACLE0.msgpack --wire-init-scale 0.06 --craft-cost 0.45 --ps 1.99 --ps-anneal 1000 1500 --tag PSA_NOSIG)
for s in 40 41 42; do $PY scripts/comm_toy.py $C --seed $s --out toy_out/ctl_PSA_NOSIG$s.jsonl > toy_out/log_ctl_PSA_NOSIG$s.txt 2>&1 & done; wait
echo done
