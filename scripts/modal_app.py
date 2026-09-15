"""Modal provisioning for THRONG training runs.

Replaces the hand-provisioned-box assumption in run_bg.py / modal_train.py with
an app definition that can be launched and re-launched deterministically:
the image clones the repo at a **pinned commit SHA**, not a floating branch,
so the box provably runs what we think it does (same discipline as the §5.3
banner check in docs/WILL_RESTART_SEP2026.md, applied to provisioning itself).

Usage:
    # Cheap pre-flight: proves the image builds, the volume mounts, the repo
    # clones at the pinned SHA, and the expected startup banners print.
    # Runs on CPU with a tiny n_steps — no GPU is billed.
    modal run scripts/modal_app.py --test

    # Real run. --detach matters: the run must outlive this session.
    modal run --detach scripts/modal_app.py

    # Tail logs of a detached run:
    modal app logs throng-train

Before the real run: re-verify the pinned SHA below is still origin/master
(`git rev-parse origin/master`), and confirm the active Modal profile really
maps to the workspace holding the `throng-runs` volume (`modal profile
current`, then check via `modal workspace members list` — the local CLI's
profile-to-workspace label is a cache, not a guarantee).
"""

from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")

# Pinned at write-time via `git rev-parse origin/master`. Re-verify before
# a real launch — if origin/master has moved, decide deliberately whether to
# re-pin, not implicitly.
PINNED_SHA = "f367f02768795d4d13784e507b77d53edbd116da"
REPO_URL = "https://github.com/overlordxrz-source/throng.git"

VOLUME_NAME = "throng-runs"
VOLUME_MOUNT = "/mnt/throng-runs"

# Full run target (matches run_bg.py). n_updates = n_steps // ppo_rollout_steps
# (512) ~= 5859 updates. checkpoint_interval=2000 env steps (~4 updates, a few
# minutes) means the run is safely resumable across function invocations, so
# the per-call timeout below does not need to cover the whole 3M steps in one
# shot -- it only needs to bound a single invocation's worst-case hang.
N_STEPS_FULL = 3_000_000

# 23h: just under Modal's default 24h function-timeout ceiling. A hang inside
# this window is caught; a normal run instead ends the call via checkpointing
# and resumption on the next launch, not via hitting this ceiling.
TIMEOUT_SECONDS = 23 * 60 * 60

app = modal.App("throng-train")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO_URL} /root/throng",
        f"cd /root/throng && git checkout {PINNED_SHA}",
        # Fail the image build (not the run) if the pin didn't take -- a
        # provisioning bug this cheap to catch should never reach a GPU.
        f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"",
    )
    .pip_install_from_requirements(_REQUIREMENTS)
    # requirements.txt's plain `jax`/`jaxlib` resolve to the CPU-only PyPI
    # wheel. Reinstall with the CUDA extra so JAX actually sees the GPU --
    # confirmed via the CPU/GPU pre-flight test that without this, jax on the
    # A100 box reports zero devices and the run never gets past checkpoint
    # restore (misread at first as a restore-logic bug; it was a missing GPU).
    .pip_install("jax[cuda12]")
)


@app.function(
    image=image,
    gpu=None,
    timeout=600,
    volumes={VOLUME_MOUNT: volume},
)
def _tiny_cpu_smoke(n_steps: int) -> str:
    """CPU pre-flight: proves clone/volume/import/banners without a GPU."""
    import os
    import subprocess
    import sys

    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, "/root/throng")
    os.chdir("/root/throng")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"[preflight] cloned HEAD = {head}", flush=True)
    assert head == PINNED_SHA, f"clone did not land on pinned SHA (got {head})"

    assert os.path.isdir(VOLUME_MOUNT), f"volume not mounted at {VOLUME_MOUNT}"
    print(f"[preflight] volume mounted at {VOLUME_MOUNT}: OK", flush=True)

    from scripts.modal_train import build_cfg
    from jax_sim.train_entry import run_simulation

    cfg = build_cfg()
    print(f"[preflight] build_cfg() OK, checkpoint_dir={cfg['checkpoint_dir']}", flush=True)
    run_simulation(cfg, seed=42, n_steps=n_steps)
    return head


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=TIMEOUT_SECONDS,
    retries=0,  # a silent auto-retry would relaunch from a checkpoint mid-experiment
    volumes={VOLUME_MOUNT: volume},
)
def train(n_steps: int = N_STEPS_FULL) -> None:
    import os
    import subprocess
    import sys

    # Same GPU env run_bg.py / modal_train.py expect.
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/throng_jax_cache"
    os.makedirs("/tmp/throng_jax_cache", exist_ok=True)

    sys.path.insert(0, "/root/throng")
    os.chdir("/root/throng")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head == PINNED_SHA, f"box did not clone the pinned SHA (got {head})"

    from scripts.modal_train import build_cfg
    from jax_sim.train_entry import run_simulation

    log_path = os.path.join(VOLUME_MOUNT, "train.log")

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self._streams:
                s.flush()

    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print(
        f"modal_app.py — THRONG | pinned SHA {head} | n_steps={n_steps:_} | "
        "GPU=A100-80GB (single-device: CPU-offloaded rollout backward pass, "
        "no sharding -- A100 vs B200 costs ~1.35x more wall-clock for the "
        "PPO-bound path but is cheaper per unit compute)",
        flush=True,
    )
    # 2026-09-14: Volume.commit() was previously called only once, after the
    # ENTIRE run_simulation() call returns (i.e. after all n_steps complete
    # or the run crashes past this line). Modal Volume writes are not
    # guaranteed durable/visible until commit() runs, so every periodic
    # checkpoint saved during the run was uncommitted -- any stop, crash, or
    # preemption before natural completion could lose all progress since the
    # last full run, regardless of how many "[CKPT] Saved" lines printed.
    # Confirmed empirically 2026-09-14: after ~20 min and 15+ PPO updates
    # past a resume, `modal volume ls` still showed no checkpoint newer than
    # the resume point. Commit on the same cadence as the checkpoint itself.
    run_simulation(build_cfg(), seed=42, n_steps=n_steps, on_checkpoint_saved=volume.commit)
    volume.commit()


@app.local_entrypoint()
def main(test: bool = False, n_steps: int = 0):
    if test:
        steps = n_steps or 50
        print(f"[preflight] running CPU smoke test, n_steps={steps}")
        head = _tiny_cpu_smoke.remote(steps)
        print(f"[preflight] smoke test completed, cloned HEAD={head}")
        return
    steps = n_steps or N_STEPS_FULL
    train.remote(steps)
