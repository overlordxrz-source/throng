"""learnability.py checkpoint + resume is bit-identical (Cam Nov 10: --ckpt on for Modal runs). A tiny all-levers run
(juvenile, predictive, replay, copy + copy-in-replay, young plasticity; wandering beacons; heard25 + last-heard test)
straight through vs killed after update 6 (checkpoints at 3, 6... the one at 6 is written before the kill) and resumed.
python -m pytest tests/test_learnability_ckpt.py --noconftest -q"""
import glob, json, os, subprocess, sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVS = os.path.join(ROOT, "w2_out/nav_w13/*.msgpack")


def run(tmp, name, extra):
    out = os.path.join(tmp, f"{name}.json")
    cmd = [sys.executable, os.path.join(ROOT, "w2/learnability.py"), "--navs", NAVS, "--world", "0.0013,8,0.001",
           "--updates", "10", "--worlds", "2", "--T", "32", "--eval-eps", "4", "--test-window", "60", "--beacon-walk",
           "--young-seek", "--beacon-spread", "6", "--test", "heard25", "--score", "last-heard", "--min-heard", "3",
           "--min-sectors", "2", "--pred", "1.0", "--replay", "2", "--copy", "1.0", "--copy-replay", "--young-lr", "4",
           "--juvenile", "100", "--out", out] + extra
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT), out


@pytest.mark.skipif(not glob.glob(NAVS), reason="W2 navigators not found")
def test_resume_bit_identical(tmp_path):
    t = str(tmp_path)
    a, oa = run(t, "A", [])
    assert a.returncode == 0, a.stderr[-2000:]
    ck = os.path.join(t, "B.ckpt")
    b1, _ = run(t, "B", ["--ckpt", ck, "--ckpt-every", "3", "--stop-after", "7"])
    assert b1.returncode == 3 and os.path.exists(ck), b1.stderr[-2000:]
    b2, ob = run(t, "B", ["--ckpt", ck, "--ckpt-every", "3"])
    assert b2.returncode == 0 and "resumed from update 6" in b2.stdout, (b2.stdout[-500:], b2.stderr[-2000:])
    A, B = json.load(open(oa)), json.load(open(ob))
    assert A["summary"]["lives"] > 0
    assert A == B
