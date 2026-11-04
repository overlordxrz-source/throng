"""Modal launcher for the learner testbed (Cam's Step 2; registration in docs/RESEARCH_PROTOCOL.md).

Same image recipe as scripts/modal_app.py (repo cloned at a pinned SHA, CUDA jax), separate app so it can
never touch the training run. One container per (arm, seed); results are returned and also appended to
the volume under testbed/.

    # timing probe: one run, 2 updates, learning metrics suppressed (used to fix N before any result)
    modal run scripts/modal_testbed.py --timing
    # the registered 12 runs (arms A-D x seeds 0-2), N updates each
    modal run --detach scripts/modal_testbed.py --updates N
"""

from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")

PINNED_SHA = "a2d910bdd0ee6a49eabd478dc498676d30fd29a7"
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_NAME = "throng-runs"
VOLUME_MOUNT = "/mnt/throng-runs"
GPU = ["L4", "A10G"]  # fallback order: capacity for L4 alone queued >10 min

app = modal.App("throng-testbed")
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


@app.function(image=image, gpu=GPU, timeout=4 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_arm(arm: str, seed: int, updates: int, timing_only: bool = False) -> list[dict]:
    import json
    import os
    import subprocess

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir("/root/throng")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_SHA, f"box did not clone the pinned SHA (got {head})"
    import jax

    print(f"[testbed] arm={arm} seed={seed} updates={updates} sha={head} devices={jax.devices()}", flush=True)
    os.makedirs(f"{VOLUME_MOUNT}/testbed", exist_ok=True)
    out = f"{VOLUME_MOUNT}/testbed/{'timing' if timing_only else 'runs'}_{arm}{seed}.jsonl"
    cmd = ["python", "scripts/learner_testbed.py", "--arm", arm, "--seed", str(seed), "--updates", str(updates)]
    cmd += ["--timing-only"] if timing_only else ["--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    recs = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:], flush=True)
        raise RuntimeError(f"testbed {arm}{seed} exited {proc.returncode}")
    volume.commit()
    return recs


@app.local_entrypoint()
def main(timing: bool = False, updates: int = 0):
    import json

    if timing:
        for r in run_arm.remote("C", 0, 2, True):
            print(json.dumps(r))
        return
    assert updates > 0, "--updates N is required (fixed from the timing probe, before any result)"
    jobs = [(arm, seed, updates) for arm in "ABCDE" for seed in (0, 1, 2)]
    with open("testbed_results.jsonl", "a") as fh:
        for recs in run_arm.starmap(jobs):
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            print(f"[testbed] done {recs[-1]['arm']}{recs[-1]['seed']}", flush=True)
