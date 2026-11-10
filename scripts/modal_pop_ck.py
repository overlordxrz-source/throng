"""(Nov 10) CHECKPOINTED copy of modal_pop.py: same jobs, --ckpt on the volume, volume committed every 2 min.
Modal launcher for the separate-brains population (throng-toy scripts/pop_toy.py on the comm-toy branch).

    MODAL_PROFILE=roadtoagi111 modal run scripts/modal_pop.py::timing      # (e) re-price probe: L4 GPU vs 8-core CPU
Results go to the volume under testbed/pop/. Navigators come from testbed/toy/nav/ (plan nav).
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")
PINNED_SHA = "003ed2ca3bafb9595acc2e304c44dbb7d8ddd94d"  # comm-toy: pop_toy.py + checkpoint/resume (default path row-identical to 5c5b07a)
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_MOUNT = "/mnt/throng-runs"

app = modal.App("throng-pop-ck")
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
    import time
    os.makedirs(f"{VOLUME_MOUNT}/testbed/pop/ckpt", exist_ok=True)
    cmd = ["python", "scripts/pop_toy.py", *args, "--navs", *NAVS, "--out", out, "--ckpt", f"{VOLUME_MOUNT}/testbed/pop/ckpt/{tag}.ckpt"]
    logp = f"{VOLUME_MOUNT}/testbed/pop/ckpt/{tag}.log"; lg = open(logp, "a")
    p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT)
    while p.poll() is None:
        time.sleep(120); lg.flush(); volume.commit()
    lg.close(); volume.commit()
    tail = open(logp).read()[-4000:]
    proc = subprocess.CompletedProcess(cmd, p.returncode, tail, tail)
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


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=12 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_pop(tag: str, args: list) -> str:
    """one population run (~8 h at 3.15 s / update); 12 h timeout (Cam ~14:45: raise the timeout)"""
    return _run(tag, args)


# (e) Cam ~14:45: 5 POP + 3 control, registered lifetimes, erosion OFF, msg-grad all, fire 5 lit from 7,500
_POP = ["--updates", "9000", "--fifth-fire", "--lit-from", "7500", "--msg-grad", "all",
        "--eval-at", "1500", "3500", "5500", "7500", "--save-at", "7500", "9000"]
POP_JOBS = ([(f"pop_on_s{s}", ["--arm", "COMM", "--seed", str(s), *_POP, "--save-prefix", f"{{VOL}}/testbed/pop/params/pop_on_s{s}",
                                "--tag", "POP"]) for s in range(320, 325)]
            + [(f"pop_ctl_s{s}", ["--arm", "NOSIG", "--seed", str(s), *_POP, "--save-prefix", f"{{VOL}}/testbed/pop/params/pop_ctl_s{s}",
                                   "--tag", "POPCTL"]) for s in range(325, 328)])


@app.local_entrypoint()
def population():
    for (tag, _), txt in zip(POP_JOBS, run_pop.starmap(POP_JOBS)):
        print(f"[pop] {tag}: {len(txt.splitlines())} lines", flush=True)


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
