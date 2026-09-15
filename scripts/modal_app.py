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
PINNED_SHA = "0a6301737b87fcaae1df4d4d47d3e12ce626b56f"
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
    # 2026-09-14: was 600s, sized for the old "zero real updates" smoke test.
    # Now that this actually runs real post-resume updates (see the n_steps
    # padding below), one CPU-only rollout+PPO-backward pair alone measured
    # >600s locally -- 600s guaranteed a timeout, not a check. Raised to
    # 1800s on that basis.
    #
    # 2026-09-15 (Cam): 1800s timed out at the same place twice, same
    # reasoning both times -- that's not a judgment call anymore, it's a
    # mis-scoped budget. Measured for real this run: blue rollout alone
    # 723.1s, blue PPO backward 879.0s -- 1602.1s before red's update even
    # starts, against an 1800s ceiling. Raised generously (5400s) rather
    # than fine-tuning a third guess -- this function is CPU-tier and rare
    # (one pre-flight per launch), so the cost of a wide margin is trivial
    # next to the cost of a preflight whose PASS means nothing because it
    # never actually reaches the finish line.
    timeout=5400,
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

    # 2026-09-14: n_updates = n_steps // T is an ABSOLUTE target step count,
    # not "steps to run from here" -- the training loop is `for ui in
    # range(start_update, n_updates)`. A flat n_steps=50 (n_updates=0) gives
    # an EMPTY range whenever start_update > 0, so every preflight against a
    # resumed checkpoint has been completing "successfully" without ever
    # executing a single loop iteration -- restore-path banners printed,
    # zero PPO updates, zero tripwire/crafting-bar code ever touched. Caught
    # by actually checking, not by trusting "smoke test completed" (Rule
    # 13). Resolve the real resume point the same way main_jax.py will, and
    # Pad enough steps for a genuine post-resume update -- just 1: CPU-only,
    # one rollout+PPO-backward pair alone measured >600s locally, so this
    # trades thoroughness for actually fitting in the timeout above. 1 real
    # update is still real coverage of the new code (the per-capita bar
    # reads b_alive_now every craft-active update; the tripwire's history
    # list gets its first real append) -- it just won't reach the
    # stability-window search, which only starts evaluating at update 7.
    _T = int(cfg.get("ppo_rollout_steps", 512))
    _resume_pin = cfg.get("resume_from_step")
    if _resume_pin is not None:
        _start_update = int(_resume_pin)
    else:
        import orbax.checkpoint as ocp
        _mngr = ocp.CheckpointManager(
            cfg["checkpoint_dir"], ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(create=False),
        )
        _latest = _mngr.latest_step()
        _start_update = int(_latest) if _latest is not None else 0
    _min_steps = (_start_update + 1) * _T
    steps = max(n_steps, _min_steps)
    if steps != n_steps:
        print(
            f"[preflight] n_steps={n_steps} would give an EMPTY update range "
            f"resuming from update {_start_update} -- padded to {steps} so at "
            f"least 1 real post-resume update actually runs.",
            flush=True,
        )
    run_simulation(cfg, seed=42, n_steps=steps)
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
    #
    # 2026-09-15 (Cam): that fix landed in this SHA but never reached the run
    # it was diagnosed on -- the run was already on an older pinned SHA and
    # died 7.5 hours later without ever writing a durable checkpoint, on
    # exactly this mechanism (checkpoint_dir resolved off the FUSE mount via
    # Path.resolve(), fixed separately in main_jax.py). "[CKPT] Committed"
    # printed the whole time; it was not evidence. Durability is a
    # precondition for running, not a property to fix while running --
    # verify it BEFORE trusting any further GPU-hours to this run. Snapshot
    # checkpoints/ now, and after the very first commit, confirm via a
    # volume.listdir() RPC (not the container's own FUSE view -- the same
    # class of blind spot that hid the original bug) that a new entry
    # actually landed. Halt immediately if it didn't.
    try:
        _ckpt_paths_before = {e.path for e in volume.listdir("checkpoints")}
    except Exception:
        _ckpt_paths_before = set()  # e.g. checkpoints/ doesn't exist yet on a fresh volume

    _durability_state = {"verified": False}

    def _commit_and_verify_durability():
        volume.commit()
        if _durability_state["verified"]:
            return
        # 2026-09-15: NOT volume.reload() -- that refreshes THIS container's
        # own local FUSE view, and fails outright ("there are open files
        # preventing the operation: path train.log is open") for the
        # entire run, since the Tee below holds train.log open the whole
        # time. volume.listdir() is a separate RPC against the backing
        # store's committed state, not a read through the local mount --
        # it doesn't need or want reload() first. Confirmed by the first
        # real launch on this fix: the FATAL branch below fired for this
        # reason, not a real durability failure, the first time this ran.
        try:
            _ckpt_paths_after = {e.path for e in volume.listdir("checkpoints")}
        except Exception as exc:
            print(
                f"[DURABILITY-GATE] FATAL: could not list checkpoints/ via the volume "
                f"RPC after the first commit to verify durability: {exc!r}. Halting "
                f"before burning more GPU.",
                flush=True,
            )
            raise SystemExit(1)
        _new_paths = _ckpt_paths_after - _ckpt_paths_before
        if not _new_paths:
            print(
                "[DURABILITY-GATE] FATAL: no new entry visible under checkpoints/ via "
                "volume.listdir() after the first commit -- the checkpoint save is not "
                "durable. This is the exact failure mode that silently lost the "
                "2026-09-14 19:12 EDT - 2026-09-15 02:51 EDT run (7.5 hours, 44 PPO "
                "updates): '[CKPT] Committed' printed every time and nothing ever "
                "reached the volume. Halting now before burning more GPU on a run that "
                "cannot save its output.",
                flush=True,
            )
            raise SystemExit(1)
        print(
            f"[DURABILITY-GATE] OK: {sorted(_new_paths)} confirmed visible under "
            f"checkpoints/ via volume.listdir() after the first commit -- durability "
            f"verified externally, not just trusted from the container's own print.",
            flush=True,
        )
        _durability_state["verified"] = True

    run_simulation(
        build_cfg(), seed=42, n_steps=n_steps,
        on_checkpoint_saved=_commit_and_verify_durability,
    )
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
