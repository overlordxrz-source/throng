"""Spawn jobs on a DEPLOYED Modal app so they survive the client (the Mac) going offline (Nov 9: a network drop on the
Mac stopped two `modal run --detach` apps mid-run).

    MODAL_PROFILE=roadtoagi111 modal deploy scripts/modal_pop.py           # once per code change
    MODAL_PROFILE=roadtoagi111 python scripts/spawn_jobs.py pop             # spawns the 8 population runs
    MODAL_PROFILE=roadtoagi111 modal deploy scripts/modal_comm_toy.py
    MODAL_PROFILE=roadtoagi111 python scripts/spawn_jobs.py toy step6r --only step6r_comp_s311,...

Each job writes its JSONL to the volume (testbed/pop/ or testbed/toy/); fetch with `modal volume get`. The call ids are
appended to ~/throng-smoke/logs/spawned.jsonl.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["pop", "toy", "w2"]); ap.add_argument("plan", nargs="?", default=None)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.kind == "pop":
        jobs = load("modal_pop").POP_JOBS; fn = modal.Function.from_name("throng-pop", "run_pop")
    elif a.kind == "w2":
        m = load("modal_w2jobs"); jobs = m.PLANS[a.plan]; fn = modal.Function.from_name("throng-w2jobs", "run_w2")
    else:
        m = load("modal_comm_toy"); jobs = m.PLANS[a.plan]; fn = modal.Function.from_name("throng-comm-toy", "run_toy")
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only.split(",")]
    log = open(os.path.expanduser("~/throng-smoke/logs/spawned.jsonl"), "a")
    for tag, args in jobs:
        call = fn.spawn(tag, args)
        rec = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), kind=a.kind, tag=tag, call_id=call.object_id)
        log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
