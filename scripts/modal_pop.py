"""Modal launcher for the separate-brains population (throng-toy scripts/pop_toy.py on the comm-toy branch).

    MODAL_PROFILE=roadtoagi111 modal run scripts/modal_pop.py::timing      # (e) re-price probe: L4 GPU vs 8-core CPU
Results go to the volume under testbed/pop/. Navigators come from testbed/toy/nav/ (plan nav).
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")
PINNED_SHA = "bc63ba0d672c2c121ea7dfafe73cc68a93ab8f38"  # comm-toy branch (pop_toy.py + selftest)
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_MOUNT = "/mnt/throng-runs"

app = modal.App("throng-pop")
volume = modal.Volume.from_name("throng-runs", create_if_missing=False)
_base = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .run_commands(f"git clone {REPO_URL} /root/throng", f"cd /root/throng && git checkout {PINNED_SHA}",
                       f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"")
         .pip_install_from_requirements(_REQUIREMENTS))
cpu_image = _base
gpu_image = _base.pip_install("jax[cuda12]")
NAVS = [f"{VOLUME_MOUNT}/testbed/toy/nav/nav_s{s}.msgpack" for s in range(260, 268)]


def _run(tag: str, args: list) -> str:
    import os
    import subprocess
    os.chdir("/root/throng")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_SHA
    os.makedirs(f"{VOLUME_MOUNT}/testbed/pop", exist_ok=True)
    out = f"{VOLUME_MOUNT}/testbed/pop/{tag}_{head[:7]}.jsonl"
    args = [x.replace("{VOL}", VOLUME_MOUNT) for x in args]
    proc = subprocess.run(["python", "scripts/pop_toy.py", *args, "--navs", *NAVS, "--out", out], capture_output=True, text=True)
    volume.commit()
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-3000:], flush=True)
        raise RuntimeError(f"{tag} exited {proc.returncode}")
    import jax
    return f"device={jax.devices()[0].device_kind}\n" + open(out).read()


@app.function(image=gpu_image, gpu="L4", timeout=30 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_l4(tag: str, args: list) -> str:
    return _run(tag, args)


@app.function(image=cpu_image, cpu=8.0, memory=8192, timeout=30 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_cpu8(tag: str, args: list) -> str:
    return _run(tag, args)


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=30 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_cpu2(tag: str, args: list) -> str:
    return _run(tag, args)


_T = ["--seed", "9", "--updates", "12", "--erode-wd", "1e-3", "--tag", "POP_TIMING"]


@app.local_entrypoint()
def timing():
    import json
    hs = {nm: f.spawn(f"timing_{nm}", _T) for nm, f in (("l4", run_l4), ("cpu8", run_cpu8), ("cpu2", run_cpu2))}
    for nm, h in hs.items():
        txt = h.get(); dev = txt.splitlines()[0]
        t = [json.loads(l)["t_update"] for l in txt.splitlines()[1:] if '"train"' in l][3:]
        print(f"[timing] {nm} {dev}: mean t_update (updates 4-12) {sum(t) / len(t):.2f} s", flush=True)


@app.local_entrypoint()
def timing_l4():
    import json
    txt = run_l4.remote("timing_l4", _T); dev = txt.splitlines()[0]
    t = [json.loads(l)["t_update"] for l in txt.splitlines()[1:] if '"train"' in l][3:]
    print(f"[timing] l4 {dev}: mean t_update (updates 4-12) {sum(t) / len(t):.2f} s", flush=True)
