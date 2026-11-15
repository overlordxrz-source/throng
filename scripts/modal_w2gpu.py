"""Cam Nov 15 ~08:30: GPU PRICE PROBE for the W2 learner (learner_w2.py), T4 and L4, ~2 minutes each, the registered
batch config unchanged. Its own app (throng-w2gpu) so it has its own watch (total cap $0.10, stop $0.125).

    MODAL_PROFILE=roadtoagi111 modal deploy scripts/modal_w2gpu.py
    MODAL_PROFILE=roadtoagi111 python scripts/modal_w2gpu.py spawn      # spawns the T4 and the L4 probe
Logs: testbed/w2jobs/gpuprobe_<gpu>_<sha7>.log (first line: jax.devices()).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")
PINNED_SHA = "120ccc8711ec8f31b113c9905becbd0716d19e72"  # comm-toy: W2 learner (learner_w2.py, ep_agent.py)
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_MOUNT = "/mnt/throng-runs"
OUTD = f"{VOLUME_MOUNT}/testbed/w2jobs"

app = modal.App(os.environ.get("W2_APP", "throng-w2gpu"))
volume = modal.Volume.from_name("throng-runs", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .run_commands(f"git clone {REPO_URL} /root/throng", f"cd /root/throng && git checkout {PINNED_SHA}",
                       f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"")
         .pip_install_from_requirements(_REQUIREMENTS)
         .pip_install("jax[cuda12]==0.10.0", "flax==0.12.7", "optax==0.2.8"))

LEARNER = ["w2/learner_w2.py", "--learner", "ep", "--slots", "96", "--shared-tok", "--hindsight", "0.5", "--demonstrator", "--hunger-gate",
           "--item-size", "0.1", "--W", "32"]


def _run(tag: str, args: list) -> str:
    import subprocess
    import time
    os.chdir("/root/throng")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == PINNED_SHA
    os.makedirs(OUTD, exist_ok=True)
    log = open(f"{OUTD}/{tag}_{PINNED_SHA[:7]}.log", "a")
    dev = subprocess.run(["python", "-c", "import jax; print(jax.devices(), jax.default_backend())"], capture_output=True, text=True)
    log.write(f"=== devices {dev.stdout.strip()} {dev.stderr.strip()[-300:]}\n")
    cmd = ["python", *args, "--out", f"{OUTD}/{tag}_{PINNED_SHA[:7]}", "--ckpt", f"{OUTD}/{tag}.ckpt"]
    log.write(f"=== start {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n"); log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    while p.poll() is None:
        time.sleep(60); log.flush(); volume.commit()
    log.write(f"=== exit {p.returncode} {time.strftime('%H:%M:%S')}\n"); log.close(); volume.commit()
    return f"{tag}: exit {p.returncode}"


@app.function(image=image, gpu="T4", cpu=2.0, memory=8192, timeout=30 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_t4(tag: str, args: list) -> str:
    return _run(tag, args)


@app.function(image=image, gpu="L4", cpu=2.0, memory=8192, timeout=30 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_l4(tag: str, args: list) -> str:
    return _run(tag, args)


@app.function(image=image, gpu="T4", cpu=2.0, memory=8192, timeout=24 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_t4_long(tag: str, args: list) -> str:
    return _run(tag, args)


@app.function(image=image, gpu="L4", cpu=2.0, memory=8192, timeout=24 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_l4_long(tag: str, args: list) -> str:
    return _run(tag, args)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "spawn":
    probe = [*LEARNER, "--steps", "2e6", "--curve-every", "5", "--max-seconds", "120"]
    for name in ("run_t4", "run_l4"):
        fn = modal.Function.from_name(os.environ.get("W2_APP", "throng-w2gpu"), name)
        call = fn.spawn(f"gpuprobe_{name[4:]}", probe); print(name, call.object_id, flush=True)
