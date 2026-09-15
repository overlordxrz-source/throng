"""One-shot CPU-only diagnostic (Cam, 2026-09-15): separate a broken
checkpoint path from a broken Orbax write, without spending GPU-hours to
find out. Two real launches have now lost a durable checkpoint despite the
`.resolve()` fix landing and the `[CKPT-PATH]` banner resolving correctly --
which means either that banner is lying, or the path is fine and Orbax's
write itself isn't surviving the FUSE layer.

Writes a plain probe file and a real Orbax checkpoint into the exact
directory main_jax.py's CheckpointManager resolves to, commits, and prints
the resolved path for external verification. Run this, then check from
OUTSIDE the container:

    modal volume ls --json throng-runs checkpoints -r

Do not launch scripts/modal_app.py until this comes back with both the
probe file and a real (non-tmp) checkpoint step directory visible.

Usage:
    modal run scripts/diag_ckpt_durability.py
"""

from __future__ import annotations

import modal

VOLUME_NAME = "throng-runs"
VOLUME_MOUNT = "/mnt/throng-runs"
PROBE_STEP = 999_999  # far outside any real ppo range, unambiguous in a listing

app = modal.App("throng-ckpt-durability-diag")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "orbax-checkpoint", "jax[cpu]", "numpy"
)


@app.function(image=image, gpu=None, timeout=300, volumes={VOLUME_MOUNT: volume})
def probe(subdir: str = "checkpoints") -> str:
    import os
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    os.environ["JAX_PLATFORMS"] = "cpu"

    # Exact resolution main_jax.py uses post-fix (jax_sim/main_jax.py, ~line
    # 1042): os.path.abspath(), not Path(...).resolve() -- does not follow
    # symlinks, so it cannot be silently redirected off the mounted volume.
    ckpt_dir = os.path.abspath(os.path.join(VOLUME_MOUNT, subdir))
    print(f"[DIAG] resolved ckpt_dir = {ckpt_dir!r}", flush=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Step 1: plain probe file, same directory.
    probe_path = os.path.join(ckpt_dir, "_probe.txt")
    with open(probe_path, "w") as f:
        f.write("durability probe 2026-09-15\n")
    print(f"[DIAG] wrote probe file: {probe_path!r}", flush=True)

    # Step 2: one real Orbax save, same construction main_jax.py uses
    # (jax_sim/main_jax.py:1072-1073, max_to_keep=10 as of 2026-09-15).
    options = ocp.CheckpointManagerOptions(max_to_keep=10, create=True)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options=options)
    ckpt_mngr.save(PROBE_STEP, items={"dummy": jnp.array([1.0, 2.0, 3.0])})
    ckpt_mngr.wait_until_finished()
    print(f"[DIAG] Orbax save + wait_until_finished() done for step {PROBE_STEP}", flush=True)

    # List the container's OWN view before commit, for comparison against
    # the external listing after.
    print(f"[DIAG] local (in-container) listing of {ckpt_dir}: {sorted(os.listdir(ckpt_dir))}", flush=True)

    # Step 3: commit.
    volume.commit()
    print("[DIAG] volume.commit() returned without raising", flush=True)

    return ckpt_dir


@app.function(image=image, gpu=None, timeout=300, volumes={VOLUME_MOUNT: volume})
def probe_resolve_hypothesis() -> None:
    """Isolated A/B test of the retired .resolve() theory (Cam, 2026-09-15):
    the original diagnosis compared production checkpoints (ppo 2544-2556)
    that were ALSO all numerically below the pre-existing 2859/2862 fossils
    -- every one of those "confirming" observations is independently and
    fully explained by Orbax's own max_to_keep retention pruning, with no
    need to invoke .resolve() at all. That evidence was confounded, not
    controlling for the variable it claimed to isolate. This test controls
    for it: a fresh, fossil-free scratch directory (nothing in it can
    outrank anything else by step number, so retention can't be the
    explanation for any difference observed here), one write through the
    OLD str(Path(...).resolve()) construction, one through the CURRENT
    os.path.abspath() construction, a single volume.commit() covering both,
    then an external listing settles whether the resolved path specifically
    bypasses commit() tracking -- the actual, still-untested claim.
    """
    import os
    from pathlib import Path

    scratch = os.path.join(VOLUME_MOUNT, "_diag_resolve_test")
    os.makedirs(scratch, exist_ok=True)

    unresolved_dir = os.path.abspath(scratch)
    resolved_dir = str(Path(scratch).resolve())
    print(f"[DIAG-RESOLVE] unresolved_dir = {unresolved_dir!r}", flush=True)
    print(f"[DIAG-RESOLVE] resolved_dir   = {resolved_dir!r}", flush=True)
    print(f"[DIAG-RESOLVE] same string? {unresolved_dir == resolved_dir}", flush=True)

    os.makedirs(unresolved_dir, exist_ok=True)
    with open(os.path.join(unresolved_dir, "probe_via_unresolved.txt"), "w") as f:
        f.write("written via os.path.abspath() -- the current production path\n")
    print("[DIAG-RESOLVE] wrote probe_via_unresolved.txt", flush=True)

    os.makedirs(resolved_dir, exist_ok=True)
    with open(os.path.join(resolved_dir, "probe_via_resolved.txt"), "w") as f:
        f.write("written via str(Path(...).resolve()) -- the retired production path\n")
    print("[DIAG-RESOLVE] wrote probe_via_resolved.txt", flush=True)

    print(
        f"[DIAG-RESOLVE] local listing of unresolved_dir: "
        f"{sorted(os.listdir(unresolved_dir))}",
        flush=True,
    )

    volume.commit()
    print("[DIAG-RESOLVE] volume.commit() returned without raising", flush=True)


@app.local_entrypoint()
def main(resolve_test: bool = False, subdir: str = "checkpoints"):
    if resolve_test:
        probe_resolve_hypothesis.remote()
        print(
            "[DIAG-RESOLVE] done. Now check from OUTSIDE: "
            "modal volume ls --json throng-runs _diag_resolve_test"
        )
        return
    ckpt_dir = probe.remote(subdir)
    print(f"[DIAG] done. Resolved ckpt_dir was: {ckpt_dir!r}")
    print(
        f"[DIAG] now check from OUTSIDE: "
        f"modal volume ls --json throng-runs {subdir}"
    )
