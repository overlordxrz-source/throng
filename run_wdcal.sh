#!/bin/zsh
# Exploratory (2026-11-08): weight-decay erosion calibration, wire off, 4 strengths x 2 calibration seeds (laptop).
cd /Users/overlord/CascadeProjects/throng-toy
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python
C=(--arm NOSIG --updates 2000 --warm-start toy_out/ORACLE0.msgpack --wire-init-scale 0.06 --craft-cost 0.45 --ps 1.99 --ps-anneal 1000 1500 --erode-from 1500 --tag WDCAL --swap-pairs 1)
for wd in 1e-4 3e-4 1e-3 3e-3; do $PY scripts/comm_toy.py $C --seed 60 --erode-wd $wd --out toy_out/wdcal_wd${wd}_s60.jsonl > toy_out/log_wdcal_${wd}_60.txt 2>&1 & done; wait
for wd in 1e-4 3e-4 1e-3 3e-3; do $PY scripts/comm_toy.py $C --seed 61 --erode-wd $wd --out toy_out/wdcal_wd${wd}_s61.jsonl > toy_out/log_wdcal_${wd}_61.txt 2>&1 & done; wait
echo done
