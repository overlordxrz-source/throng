"""Modal launcher for the W2 learnability / fast-learner jobs (Cam Nov 10: jobs over ~10 min go to Modal, small CPU
containers, --ckpt on). throng-toy w2/ on the comm-toy branch, pinned.

    MODAL_PROFILE=roadtoagi111 modal deploy scripts/modal_w2jobs.py
    MODAL_PROFILE=roadtoagi111 python scripts/spawn_jobs.py w2 <plan> [--only tag,...]
Each job writes testbed/w2jobs/<tag>_<sha7>.json (fast learner: a directory) and checkpoints to testbed/w2jobs/<tag>.ckpt;
the volume is committed every 2 minutes so a preempted job resumes from its last checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")
PINNED_SHA = "b61afb21b7fbcbdb4996e4109a4bb599757a4120"  # comm-toy: apparatus + heard25 rulings + ckpt + nav_w13
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_MOUNT = "/mnt/throng-runs"
OUTD = f"{VOLUME_MOUNT}/testbed/w2jobs"

app = modal.App("throng-w2jobs")
volume = modal.Volume.from_name("throng-runs", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .run_commands(f"git clone {REPO_URL} /root/throng", f"cd /root/throng && git checkout {PINNED_SHA}",
                       f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"")
         .pip_install_from_requirements(_REQUIREMENTS))


@app.function(image=image, cpu=4.0, memory=6144, timeout=2 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_w2(tag: str, args: list) -> str:
    import os
    import subprocess
    import time
    os.chdir("/root/throng")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == PINNED_SHA
    os.makedirs(OUTD, exist_ok=True)
    script, rest = args[0], list(args[1:])
    out = f"{OUTD}/{tag}_{PINNED_SHA[:7]}" + (".json" if script.endswith("learnability.py") else "")
    cmd = ["python", script, *rest, "--out", out, "--ckpt", f"{OUTD}/{tag}.ckpt"]
    log = open(f"{OUTD}/{tag}_{PINNED_SHA[:7]}.log", "a")
    log.write(f"=== start {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n"); log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    while p.poll() is None:
        time.sleep(120); log.flush(); volume.commit()
    log.write(f"=== exit {p.returncode} {time.strftime('%H:%M:%S')}\n"); log.close(); volume.commit()
    return f"{tag}: exit {p.returncode}"


_NAVS = "w2_out/nav_w13/*.msgpack"
_WORLD = ["--world", "0.0013,8,0.001"]
_APP = ["--beacon-walk", "--young-seek", "--beacon-spread", "6", "--test", "heard25", "--score", "last-heard", "--min-sectors", "5",
        "--eval-eps", "48", "--test-window", "600"]
PLANS = {
    # apparatus gate (ii) (Cam Nov 10): scripted control >= 90% on eligible lives, non-listener <= 10%, VoI >= 1.5x
    "app_gate": [("app_gate_scripted", ["w2/learnability.py", "--navs", _NAVS, *_WORLD, *_APP, "--updates", "150",
                                        "--worlds", "16", "--no-train", "--sanity"])],
}
