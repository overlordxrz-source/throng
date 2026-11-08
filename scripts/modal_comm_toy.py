"""Modal launcher for the CPU comm toy (scripts/comm_toy.py on the comm-toy branch). CPU containers only; Cam's
cap for toy runs on Modal: $2 total. One container per run; results to the volume under testbed/toy/.

    MODAL_PROFILE=rapidcrisp modal run scripts/modal_comm_toy.py --plan erosion
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")

PINNED_SHA = "75323c4fbcd328c7409d515e7eb3c16364bf1fa8"  # comm-toy branch
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_NAME = "throng-runs"
VOLUME_MOUNT = "/mnt/throng-runs"

app = modal.App("throng-comm-toy")
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
)


@app.function(image=image, cpu=8, memory=8192, timeout=3 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_toy(tag: str, args: list) -> str:
    import os
    import subprocess

    os.chdir("/root/throng")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_SHA, f"box did not clone the pinned SHA (got {head})"
    os.makedirs(f"{VOLUME_MOUNT}/testbed/toy", exist_ok=True)
    out = f"{VOLUME_MOUNT}/testbed/toy/{tag}_{head[:7]}.jsonl"
    cmd = ["python", "scripts/comm_toy.py", *args, "--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    volume.commit()
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:], flush=True)
        raise RuntimeError(f"{tag} exited {proc.returncode}")
    return open(out).read()


_W = "toy_out/ORACLE0.msgpack"
_BASE = ["--updates", "2000", "--warm-start", _W, "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99",
         "--ps-anneal", "1000", "1500", "--erode-from", "1500"]
PLANS: dict = {
    # exploratory (2026-11-08): weight-decay erosion calibration, wire OFF, 2 calibration seeds per strength.
    # The token-entropy force failed calibration (none < 0.5 bit); this is reported to Cam, not registered.
    "wdcal": [(f"wdcal_wd{wd}_s{s}", ["--arm", "NOSIG", "--seed", str(s), *_BASE, "--erode-wd", wd, "--tag", "WDCAL"])
              for wd in ("1e-4", "3e-4", "1e-3", "3e-3") for s in (60, 61)],
}


@app.local_entrypoint()
def main(plan: str, outdir: str = "toy_out_modal"):
    Path(outdir).mkdir(exist_ok=True)
    jobs = PLANS[plan]
    for (tag, _), text in zip(jobs, run_toy.starmap(jobs)):
        (Path(outdir) / f"{tag}.jsonl").write_text(text)
        print(f"[toy] {tag}: {len(text.splitlines())} records", flush=True)
