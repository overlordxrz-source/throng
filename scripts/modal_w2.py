"""Modal launcher for W2 (throng-toy w2/ on the comm-toy branch). First paid smoke (Cam Nov 9 ~17:50, cap $0.5):
a 2-minute A10G timing and a ~10-minute no-comm run of the per-agent population (inclusive fitness on).

    MODAL_PROFILE=roadtoagi111 modal deploy scripts/modal_w2.py
    MODAL_PROFILE=roadtoagi111 python -c "import modal; f=modal.Function.from_name('throng-w2','smoke'); print(f.spawn().object_id)"
Results: testbed/w2/ on the volume.
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")
PINNED_SHA = "489bcd0aa2456cd86375766e8ce84f37dbd0b352"  # comm-toy branch (w2 agents + navigator weights)
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_MOUNT = "/mnt/throng-runs"

app = modal.App("throng-w2")
volume = modal.Volume.from_name("throng-runs", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .run_commands(f"git clone {REPO_URL} /root/throng", f"cd /root/throng && git checkout {PINNED_SHA}",
                       f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"")
         .pip_install_from_requirements(_REQUIREMENTS).pip_install("jax[cuda12]"))


def _agents(tag, args):
    import os
    import subprocess
    os.chdir("/root/throng")
    os.makedirs(f"{VOLUME_MOUNT}/testbed/w2", exist_ok=True)
    out = f"{VOLUME_MOUNT}/testbed/w2/{tag}_{PINNED_SHA[:7]}.jsonl"
    p = subprocess.run(["python", "w2/agents.py", "--navs", "w2_out/nav/*.msgpack", *args, "--out", out],
                       capture_output=True, text=True)
    volume.commit()
    print(p.stdout[-1500:], p.stderr[-1500:], flush=True)
    return p.stdout[-1500:]


@app.function(image=image, gpu="A10G", timeout=25 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def smoke():
    import jax
    print("device", jax.devices()[0].device_kind, flush=True)
    t = _agents("smoke_timing_w64", ["--worlds", "64", "--updates", "8", "--inclusive", "--seed", "11"])     # ~2 min
    r = _agents("smoke_nocomm_w64", ["--worlds", "64", "--updates", "120", "--inclusive", "--seed", "12"])   # ~10 min
    return t + "\n" + r
