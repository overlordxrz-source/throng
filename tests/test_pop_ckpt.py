"""Checkpoint + resume is bit-identical (Cam Nov 10). A tiny pop_toy run straight through vs the same run killed after
update 5 (checkpoints at 2 and 4) and resumed: same JSONL rows (timing removed) and same final brains.
python -m pytest tests/test_pop_ckpt.py --noconftest -q   (needs the toy navigators; set THRONG_NAVS to their dir)"""
import glob, json, os, subprocess, sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NAVD = os.environ.get("THRONG_NAVS", os.path.expanduser("~/throng-backup/roadtoagi111/testbed/toy/nav"))
NAVS = [f"{NAVD}/nav_s{s}.msgpack" for s in range(260, 268)]


def run(tmp, name, extra):
    out = os.path.join(tmp, f"{name}.jsonl")
    cmd = [sys.executable, os.path.join(ROOT, "scripts/pop_toy.py"), "--arm", "COMM", "--seed", "7", "--updates", "6",
           "--worlds", "12", "--steps", "16", "--first-repl", "3", "--stagger", "1", "--life", "4", "--sensitive", "2",
           "--imit-batch", "16", "--fifth-fire", "--lit-from", "5", "--msg-grad", "all", "--save-at", "6",
           "--save-prefix", os.path.join(tmp, name), "--navs", *NAVS, "--out", out] + extra
    return subprocess.run(cmd, capture_output=True, text=True), out


def rows(p):
    R = [json.loads(l) for l in open(p) if l.strip()]
    for r in R:
        r.pop("t_update", None)
    return [r for r in R if r.get("phase") != "resume"]


@pytest.mark.skipif(not os.path.exists(NAVS[0]), reason="toy navigators not found")
def test_resume_bit_identical(tmp_path):
    t = str(tmp_path)
    a, oa = run(t, "A", [])
    assert a.returncode == 0, a.stderr[-2000:]
    ck = os.path.join(t, "B.ckpt")
    b1, ob = run(t, "B", ["--ckpt", ck, "--ckpt-every", "2", "--stop-after", "5"])
    assert b1.returncode == 3 and os.path.exists(ck)                       # simulated preemption after update 5
    assert max(r.get("update", 0) for r in rows(ob) if r.get("phase") == "train") == 5
    b2, ob = run(t, "B", ["--ckpt", ck, "--ckpt-every", "2"])               # restart on the same input
    assert b2.returncode == 0, b2.stderr[-2000:]
    allb = [json.loads(l) for l in open(ob) if l.strip()]
    assert [r["update"] for r in allb if r.get("phase") == "resume"] == [4]   # it really resumed from the u4 checkpoint
    ra, rb = rows(oa), rows(ob)
    assert any(r.get("phase") == "birth" and r["update"] >= 5 for r in ra)    # births after the resume point are exercised
    tra = [r for r in ra if r.get("phase") == "train"]; trb = [r for r in rb if r.get("phase") == "train"]
    assert [r["update"] for r in trb] == list(range(1, 7))                 # no duplicated or missing updates
    assert ra == rb                                                        # every row identical (births, train, eval)
    assert open(os.path.join(t, "A_u6_brains.msgpack"), "rb").read() == open(os.path.join(t, "B_u6_brains.msgpack"), "rb").read()
