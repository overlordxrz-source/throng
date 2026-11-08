"""Modal launcher for the jitted comm testbed (step 0 timing run, then the PS confirmation).

Same image recipe as scripts/modal_testbed.py, its own pin (comm-jit branch) and app, so the older plans' pin is
untouched. Results go to the volume under testbed/comm_jit/.

    MODAL_PROFILE=rapidcrisp modal run scripts/modal_comm_jit.py --plan timing
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")

PINNED_SHA = "326c8cb3f366c49b24be97dffbb2c419f2a525d0"  # comm-jit branch
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


def _run_stream(tag: str, args: list) -> list[dict]:
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


@app.function(image=image, gpu=GPU, timeout=6 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_jit(tag: str, args: list) -> list[dict]:
    return _run_stream(tag, args)


def _seq(tag: str, arg_lists: list) -> list[dict]:
    import json
    import os
    import subprocess

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir("/root/throng")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_SHA, f"box did not clone the pinned SHA (got {head})"
    os.makedirs(f"{VOLUME_MOUNT}/testbed/comm_jit", exist_ok=True)
    recs = []
    for j, args in enumerate(arg_lists):
        args = [x.replace("{VOL}", VOLUME_MOUNT) for x in args]
        cmd = ["python", "scripts/comm_testbed_jit.py", *args, "--out",
               f"{VOLUME_MOUNT}/testbed/comm_jit/{tag}_{j}_{head[:7]}.jsonl"]
        print(f"[comm-jit] {tag}[{j}] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                recs.append(json.loads(line))
            elif "[PPO-SCAN CHECK]" in line or "APPARATUS" in line:
                print(line, flush=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-3000:], flush=True)
            recs.append({"phase": "error", "job": j, "rc": proc.returncode})
    volume.commit()
    return recs


@app.function(image=image, gpu="A10G", timeout=60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def seq_a10(tag: str, arg_lists: list) -> list[dict]:
    return _seq(tag, arg_lists)


@app.function(image=image, gpu="L4", timeout=60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def seq_l4(tag: str, arg_lists: list) -> list[dict]:
    return _seq(tag, arg_lists)


@app.function(image=image, gpu="A10G", timeout=8 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_a10(tag: str, args: list) -> list[dict]:
    return _run_stream(tag, args)


WARM = "{VOL}/testbed/params/comm_ORACLE0_run_b88fe7f_a2.msgpack"
CONFIRM = [(f"confirm_PS{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "450", "--tau", "0.08", "--craft-cost", "0.1",
                               "--warm-start", "{VOL}/testbed/params/comm_ORACLE0_run_b88fe7f_a2.msgpack", "--ps", "2.015",
                               "--apparatus-update", "2", "--mi-halt-k", "5", "--mi-floor", "0.92", "--mi-probe-halt",
                               "--eval-at", *[str(u) for u in range(50, 450, 50)], "--swap-pairs", "10",
                               "--swap-modes", "empirical", "--save-params", f"{{VOL}}/testbed/params/confirm_PS{s}.msgpack"])
           for s in (50, 51)]


@app.local_entrypoint()
def confirm():
    for (tag, _), recs in zip(CONFIRM, run_a10.starmap(CONFIRM)):
        print(f"[{tag}] {len(recs)} records", flush=True)


_RUN = ["--arm", "COMM", "--tau", "0.08", "--craft-cost", "0.1", "--warm-start", "{VOL}/testbed/params/comm_ORACLE0_run_b88fe7f_a2.msgpack",
        "--ps", "2.015", "--apparatus-update", "0", "--no-eval"]
SCAN_CHECK = {
    "a10": [*[["--seed", str(s), "--updates", "1", "--check-ppo-scan", *_RUN] for s in (40, 41, 42)],
            ["--seed", "43", "--updates", "4", "--ppo-scan", *_RUN]],
    "l4": [["--seed", "43", "--updates", "4", *_RUN], ["--seed", "43", "--updates", "4", "--ppo-scan", *_RUN]],
}


@app.local_entrypoint()
def scancheck():
    ha = seq_a10.spawn("scancheck_a10", SCAN_CHECK["a10"]); hl = seq_l4.spawn("scantime_l4", SCAN_CHECK["l4"])
    for name, h in (("A10G", ha), ("L4", hl)):
        recs = h.get()
        for r in recs:
            if r.get("phase") == "ppo_scan_check":
                print(f"[{name}] seed {r['seed']}: candidate_pass={r['candidate_pass']} mutants_fail={r['mutants_fail']}"
                      f" ok={r['ok']} | {r['results']['candidate']}", flush=True)
            elif r.get("phase") == "train":
                print(f"[{name}] t_update {r['t_update']} (rollout {r['t_rollout']}, ppo {r['t_ppo']})", flush=True)
            elif r.get("phase") == "error":
                print(f"[{name}] ERROR job {r['job']} rc {r['rc']}", flush=True)
PLANS = {
    # step 0 timing (<= $0.50): 6 COMM updates at the real settings; compile time is in update 1
    "timing": [("timing_COMM0", ["--arm", "COMM", "--seed", "0", "--updates", "6", "--tau", "0.08",
                                 "--craft-cost", "0.1", "--warm-start", WARM, "--no-eval"])],
    # MI-floor / run-length smoke (Cam 2026-11-08 ruling 4): PS arm, 2 fresh seeds the confirmation won't use,
    # <= 120 updates, loop PPO path (current speed), c_ps 2.015 (registered calibration rule on the real network)
    "smoke": [(f"smoke_PS{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "120", "--tau", "0.08",
                                "--craft-cost", "0.1", "--warm-start", WARM, "--ps", "2.015", "--apparatus-update", "2",
                                "--no-eval"]) for s in (20, 21)],
}


@app.local_entrypoint()
def main(plan: str = "timing"):
    jobs = PLANS[plan]
    for (tag, args), recs in zip(jobs, run_jit.starmap(jobs)):
        tr = [r for r in recs if r.get("phase") == "train"]
        for r in tr:
            print(f"[{tag}] update {r['update']}: t_update {r['t_update']}s (rollout {r['t_rollout']}s, ppo {r['t_ppo']}s)"
                  f" gpu={r.get('gpu')}")
