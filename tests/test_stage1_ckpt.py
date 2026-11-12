"""Stage-1 (--agent recurrent) checkpoint + resume is bit-identical (Cam Nov 12: --ckpt on for the pilot). 40 updates
straight vs killed after 25 (checkpoints every 10) and resumed; deaths fall after the resume point, so the restored RNG
matters (a mutant that skips restoring it diverged when this test was written).
python -m pytest tests/test_stage1_ckpt.py --noconftest -q"""
import glob, json, os, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVS = os.path.join(ROOT, "w2_out/nav_w13/*.msgpack")
ARGS = ["--navs", NAVS, "--world", "0.0013,8,0.001", "--beacon-walk", "--young-seek", "--beacon-spread", "6", "--test", "heard25",
        "--score", "last-heard", "--min-sectors", "5", "--seek-age", "500", "--parent", "--parent-leave", "1000", "--agent", "recurrent",
        "--updates", "40", "--worlds", "4", "--eval-updates", "0"]


def run(out, extra):
    return subprocess.run([sys.executable, os.path.join(ROOT, "w2/learnability.py"), *ARGS, "--out", out, *extra], capture_output=True, text=True, cwd=ROOT)


@pytest.mark.skipif(not glob.glob(NAVS), reason="W2 navigators not found")
def test_stage1_resume_bit_identical(tmp_path):
    t = str(tmp_path); ck = os.path.join(t, "s.ckpt")
    assert run(os.path.join(t, "A.json"), []).returncode == 0
    r1 = run(os.path.join(t, "B.json"), ["--ckpt", ck, "--ckpt-every", "10", "--stop-after", "25"])
    assert r1.returncode == 3 and os.path.exists(ck)
    r2 = run(os.path.join(t, "B.json"), ["--ckpt", ck, "--ckpt-every", "10"])
    assert r2.returncode == 0 and "resumed stage 1 at update 20" in r2.stdout
    k = lambda p: [{x: y for x, y in r.items() if x != "sec"} for r in json.load(open(p))["log"]]  # noqa: E731
    assert k(os.path.join(t, "A.json")) == k(os.path.join(t, "B.json"))
