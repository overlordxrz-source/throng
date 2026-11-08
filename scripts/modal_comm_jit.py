"""Modal launcher for the jitted comm testbed (step 0 timing run, then the PS confirmation).

Same image recipe as scripts/modal_testbed.py, its own pin (comm-jit branch) and app, so the older plans' pin is
untouched. Results go to the volume under testbed/comm_jit/.

    MODAL_PROFILE=rapidcrisp modal run scripts/modal_comm_jit.py --plan timing
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")

PINNED_SHA = "1d3b951af97962079d48b2f1f3d8e0388fa3bed6"  # comm-jit branch
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_NAME = "throng-runs"
VOLUME_MOUNT = "/mnt/throng-runs"
GPU = ["A10G", "L4"]  # the Python-path baseline (~37 s/update) was measured on A10/A10G

app = modal.App("throng-comm-jit")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO_URL} /root/throng",
        f"cd /root/throng && git checkout {PINNED_SHA}",
        f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"",
    )
    .pip_install_from_requirements(_REQUIREMENTS)
    .pip_install("jax[cuda12]")
)


@app.function(image=image, gpu=GPU, timeout=6 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_jit(tag: str, args: list) -> list[dict]:
    import json
    import os
    import subprocess

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir("/root/throng")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_SHA, f"box did not clone the pinned SHA (got {head})"
    os.makedirs(f"{VOLUME_MOUNT}/testbed/comm_jit", exist_ok=True)
    args = [x.replace("{VOL}", VOLUME_MOUNT) for x in args]
    cmd = ["python", "scripts/comm_testbed_jit.py", *args, "--out", f"{VOLUME_MOUNT}/testbed/comm_jit/{tag}_{head[:7]}.jsonl"]
    print(f"[comm-jit] {tag} sha={head} cmd={' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    recs = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("{"):
            recs.append(json.loads(line))
    rc = proc.wait()
    volume.commit()
    if rc != 0:
        raise RuntimeError(f"{tag} exited {rc}")
    return recs


WARM = "{VOL}/testbed/params/comm_ORACLE0_run_b88fe7f_a2.msgpack"
PLANS = {
    # step 0 timing (<= $0.50): 6 COMM updates at the real settings; compile time is in update 1
    "timing": [("timing_COMM0", ["--arm", "COMM", "--seed", "0", "--updates", "6", "--tau", "0.08",
                                 "--craft-cost", "0.1", "--warm-start", WARM, "--no-eval"])],
}


@app.local_entrypoint()
def main(plan: str = "timing"):
    jobs = PLANS[plan]
    for (tag, args), recs in zip(jobs, run_jit.starmap(jobs)):
        tr = [r for r in recs if r.get("phase") == "train"]
        for r in tr:
            print(f"[{tag}] update {r['update']}: t_update {r['t_update']}s (rollout {r['t_rollout']}s, ppo {r['t_ppo']}s)"
                  f" gpu={r.get('gpu')}")
