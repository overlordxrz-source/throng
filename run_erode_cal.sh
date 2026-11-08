#!/bin/zsh
# Erosion calibration (Cam 2026-11-08): wire-off code, separate seeds; weakest token-entropy bonus that drives MI < 0.5 bit within 500 updates.
cd /Users/overlord/CascadeProjects/throng-toy
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
C=(--arm NOSIG --updates 2000 --warm-start toy_out/ORACLE0.msgpack --wire-init-scale 0.06 --craft-cost 0.45 --ps 1.99 --ps-anneal 1000 1500 --erode-from 1500 --tag CAL)
$PY scripts/comm_toy.py $C --seed 50 --erode-tok-ent 0.03 --out toy_out/cal_e0.03.jsonl > toy_out/log_cal_e0.03.txt 2>&1 &
$PY scripts/comm_toy.py $C --seed 51 --erode-tok-ent 0.1 --out toy_out/cal_e0.1.jsonl > toy_out/log_cal_e0.1.txt 2>&1 &
$PY scripts/comm_toy.py $C --seed 52 --erode-tok-ent 0.3 --out toy_out/cal_e0.3.jsonl > toy_out/log_cal_e0.3.txt 2>&1 &
$PY scripts/comm_toy.py $C --seed 53 --erode-tok-ent 1.0 --out toy_out/cal_e1.0.jsonl > toy_out/log_cal_e1.0.txt 2>&1 &
wait; echo done
