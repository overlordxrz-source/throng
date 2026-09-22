"""Modal provisioning for THRONG training runs.

Replaces the hand-provisioned-box assumption in run_bg.py / modal_train.py with
an app definition that can be launched and re-launched deterministically:
the image clones the repo at a **pinned commit SHA**, not a floating branch,
so the box provably runs what we think it does (same discipline as the §5.3
banner check in docs/WILL_RESTART_SEP2026.md, applied to provisioning itself).

Usage:
    # Cheap pre-flight (2026-09-20, rescoped -- see _tiny_cpu_smoke's own
    # docstring for why): proves the image builds, the volume mounts, the
    # repo clones at the pinned SHA, build_cfg() loads, the checkpoint
    # restores, and the fossil guard evaluates against the real
    # checkpoint_dir. No GPU billed, and cheap enough (~1-5 min, no JAX
    # rollout) to run before every launch with no exceptions -- it does NOT
    # exercise training-loop logic (no rollout, no PPO update, no
    # tripwire/comms-warmup/dead-code-reset dynamics); only the real launch
    # verifies those.
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
PINNED_SHA = "5514f0287cc97a30117a3783a7faf31a6cf6a887"
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
    # 2026-09-20 (Cam): this function's scope was rewritten, which is why
    # the timeout dropped back down -- see the docstring below. Old history
    # for context: 2026-09-14 raised 600s->1800s once this started forcing a
    # real post-resume update; 2026-09-15 raised 1800s->5400s after that
    # timed out at the same place twice (measured: blue rollout 723.1s +
    # blue PPO backward 879.0s = 1602.1s before red's update even starts).
    # 2026-09-20: skipped three times running under deadline pressure, each
    # time with a locally sound reason -- which is exactly the failure mode
    # Cam flagged: "a check that's skipped whenever it's inconvenient isn't
    # a check." The real problem wasn't the reasons, it was the function:
    # even paid for in full, it never once exercised what actually broke
    # this session (the durability gate needs a checkpoint-interval of
    # updates; the tripwire/rate-limit/C0-split logic needs dozens of
    # updates post-unfreeze) -- 30-50 minutes of CPU time bought coverage of
    # exactly one real update of stage-0 logic, no more. Rescoped to what it
    # can actually verify cheaply (see docstring) so it can run every time,
    # no exceptions, rather than being a step that exists on paper. Model
    # init/restore compile can still take a few minutes; this is not "runs
    # instantly," just "affordable enough to never skip."
    timeout=600,
    volumes={VOLUME_MOUNT: volume},
)
def _tiny_cpu_smoke() -> str:
    """CPU pre-flight, rescoped 2026-09-20 (Cam) to what it can actually
    verify cheaply, so it can run before every launch with no exceptions.
    Rescoped again 2026-09-21 (hearths) to also restore the resume
    checkpoint through the real production grafting path.

    Proves: the image builds, the volume mounts, the repo clones at the
    pinned SHA, build_cfg() loads, the fossil guard evaluates against the
    REAL checkpoint_dir on the REAL volume (same CheckpointManager
    construction and find_fossil_checkpoints() call main_jax.py uses), and
    -- new for the hearth relaunch, since own_state_dim 22 -> 29 is exactly
    the class of shape mismatch a listdir-only check can't catch -- the
    resume checkpoint actually restores through graft_missing_param_
    subtrees + ensure_aux_head_params/ensure_predator_params (both teams)
    and produces the expected post-graft shape, not just that the restore
    call doesn't raise.

    Deliberately does NOT run any training-loop logic (no rollout, no PPO
    update, no tripwire, no comms-freeze/warmup/dead-code-reset dynamics).
    That was the old design (padding n_steps to force exactly one real
    post-resume update) -- paid for in full (30-50 CPU-minutes), it still
    never once exercised what actually broke this session: the durability
    gate needs a checkpoint-interval of updates to even attempt a save; the
    tripwire baseline split, the comms warmup, and the rate-limited dead-
    code reset all need dozens of updates post-unfreeze to do anything
    observable. One forced update bought real but narrow coverage (stage-0
    setup code only) at a cost high enough to make skipping this preflight
    tempting under time pressure -- which happened three times this
    session, each time for a locally sound reason. A check skipped whenever
    it's inconvenient isn't a check. This version is cheap enough that
    there's no longer a reason to skip it, and it says plainly what it does
    and doesn't cover instead of quietly implying more than it proves:
    verifying deep training-loop behavior still requires watching the real
    launch, not a substitute for it.
    """
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

    cfg = build_cfg()
    print(f"[preflight] build_cfg() OK, checkpoint_dir={cfg['checkpoint_dir']}", flush=True)

    import orbax.checkpoint as ocp
    from jax_sim.main_jax import find_fossil_checkpoints

    ckpt_dir = os.path.abspath(cfg["checkpoint_dir"])
    mngr = ocp.CheckpointManager(
        ckpt_dir, ocp.StandardCheckpointer(),
        # max_to_keep matches main_jax.py's real construction -- this
        # CheckpointManager only reads (all_steps()/latest_step()) so it
        # can't prune anything, but keep it faithful to what the real
        # launch constructs rather than assuming it's irrelevant.
        options=ocp.CheckpointManagerOptions(max_to_keep=10, create=False),
    )
    latest = mngr.latest_step()
    print(f"[preflight] checkpoint_dir latest step = {latest}", flush=True)
    resume_pin = cfg.get("resume_from_step")
    resume_target = int(resume_pin) if resume_pin is not None else (int(latest) if latest is not None else None)
    if resume_target is not None:
        fossils = find_fossil_checkpoints(mngr, resume_target)
        if fossils:
            all_steps = sorted(int(s) for s in mngr.all_steps())
            raise AssertionError(
                f"[preflight] FOSSIL-GUARD would refuse this launch: {ckpt_dir!r} contains "
                f"checkpoint(s) {fossils} numerically ahead of the resume target "
                f"({resume_target}). All steps present: {all_steps}. Fix before launching on "
                f"a GPU -- this is exactly the mechanism that silently lost two real launches "
                f"on 2026-09-15 (docs/THE-ECOLOGY-NEVER-RAN.md instance 6)."
            )
        print(f"[preflight] fossil guard OK -- no checkpoint ahead of resume target {resume_target}", flush=True)

        # 2026-09-21 (Cam, hearth relaunch): rescoped further to actually
        # restore the resume checkpoint through the real production
        # grafting path (graft_missing_param_subtrees + ensure_aux_head_
        # params/ensure_predator_params) -- not just list its step number.
        # own_state_dim changed 22 -> 29 for hearths; this is exactly the
        # class of shape mismatch a listdir-only check can't catch, and
        # it's cheap (CPU, no rollout) to verify before paying for a GPU.
        from jax.sharding import SingleDeviceSharding
        from flax.core import freeze, unfreeze
        import jax as _jax
        import jax.numpy as _jnp
        from jax_sim.network_jax import (
            AgentNetworkJax, PredatorNetworkJax,
            init_agent_params, init_predator_params,
            ensure_aux_head_params, ensure_predator_params,
            graft_missing_param_subtrees, sanitize_agent_params,
        )
        from jax_sim.obs_layout import make_obs_layout
        from jax_sim.main_jax import _normalize_config

        _cfg = _normalize_config(cfg)
        _layout = make_obs_layout(
            signal_dim=_cfg["signal_dim"], symbol_dim=_cfg["symbol_dim"],
            memory_slots=_cfg.get("memory_slots", 0), neighbor_k=_cfg["neighbor_k"],
            local_cells=(2 * _cfg["local_obs_radius"] + 1) ** 2,
            env_channels=int(_cfg.get("env_channels", 15)),
            own_state_dim=int(_cfg.get("own_state_dim", 29)),
        )
        _obs_dim = _layout.total_dim
        _model = AgentNetworkJax(
            hidden_dim=_cfg["hidden_dim"], n_heads=_cfg["n_heads"], n_layers=_cfg["n_layers"],
            obs_dim=_obs_dim, signal_dim=_cfg["signal_dim"], symbol_dim=_cfg["symbol_dim"],
            vocab_size=_cfg["vocab_size"], vq_beta=float(_cfg.get("vq_beta", 0.25)),
            vq_dead_code_reset=bool(_cfg.get("vq_dead_code_reset", True)),
            memory_slots=_cfg.get("memory_slots", 0),
            fwd_env_dim=_layout.loc_env_end - _layout.loc_env_start,
            cross_attn_enabled=bool((_cfg.get("phase9_canvas") or {}).get("cross_attn_enabled", False)),
            cross_attn_num_heads=int((_cfg.get("phase9_canvas") or {}).get("cross_attn_num_heads", _cfg["n_heads"])),
            env_channels=int(_cfg.get("env_channels", 15)), own_state_dim=int(_cfg.get("own_state_dim", 29)),
            n_actions=int(_cfg.get("n_actions", 8)),
            local_cells=(2 * _cfg["local_obs_radius"] + 1) ** 2, neighbor_k=_cfg["neighbor_k"],
        )
        _p12 = _cfg.get("phase12_red") or {}
        _red_hidden = _cfg.get("red_hidden_dim", _cfg["hidden_dim"])
        _model_red = PredatorNetworkJax(
            hidden_dim=_red_hidden, neighbor_k=_cfg["neighbor_k"], local_obs_radius=_cfg["local_obs_radius"],
            n_heads=_cfg["n_heads"], n_layers=_cfg["n_layers"], signal_dim=_cfg["signal_dim"],
            symbol_dim=_cfg["symbol_dim"], vocab_size=int(_p12.get("red_vocab_size", _cfg.get("vocab_size", 64))),
            vq_beta=float(_cfg.get("vq_beta", 0.25)), vq_dead_code_reset=bool(_cfg.get("vq_dead_code_reset", True)),
            memory_slots=_cfg.get("memory_slots", 0),
            cross_attn_enabled=bool(_p12.get("red_cross_attn_enabled", True)),
            cross_attn_num_heads=int((_cfg.get("phase9_canvas") or {}).get("cross_attn_num_heads", _cfg["n_heads"])),
            env_channels=int(_cfg.get("env_channels", 15)), own_state_dim=int(_cfg.get("own_state_dim", 29)),
            n_actions=int(_cfg.get("n_actions", 8)),
        )

        _cpu = _jax.devices("cpu")[0]
        _meta = mngr.item_metadata(resume_target)
        _target = _jax.tree_util.tree_map(
            lambda leaf: _jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=SingleDeviceSharding(_cpu)),
            _meta, is_leaf=lambda x: hasattr(x, "shape"),
        )
        _raw = mngr.restore(resume_target, args=ocp.args.StandardRestore(_target))
        _source = unfreeze(_raw)

        _rng = _jax.random.PRNGKey(0)
        _dummy_carry = _jnp.zeros((2, _cfg["hidden_dim"]))
        _dummy_obs = _jnp.zeros((2, _obs_dim))
        _b_template = unfreeze(init_agent_params(_model, _rng, _dummy_carry, _dummy_obs, _cfg["n_layers"]))
        _src_b = unfreeze(_source["b_params"])
        graft_missing_param_subtrees(_src_b, _b_template)
        _b_params = sanitize_agent_params(
            ensure_aux_head_params(_model, freeze(_src_b), _rng, _cfg["hidden_dim"], obs_dim=_obs_dim, n_layers=_cfg["n_layers"])
        )
        _dummy_carry_r = _jnp.zeros((2, _red_hidden))
        _r_template = unfreeze(init_predator_params(_model_red, _rng, _dummy_carry_r, _dummy_obs, _cfg["n_layers"]))
        _src_r = unfreeze(_source["r_params"])
        graft_missing_param_subtrees(_src_r, _r_template)
        _r_params = sanitize_agent_params(
            ensure_predator_params(_model_red, freeze(_src_r), _rng, _red_hidden, obs_dim=_obs_dim, n_layers=_cfg["n_layers"])
        )
        # Confirms the graft itself produced the right final shape (not just
        # that it ran without raising) -- the exact silent-lobotomy failure
        # mode Cam flagged would otherwise pass this check.
        _emb_kernel_shape = unfreeze(_b_params)["emb_own"]["kernel"].shape
        assert _emb_kernel_shape == (int(_cfg.get("own_state_dim", 29)), _cfg["hidden_dim"]), (
            f"[preflight] grafted emb_own kernel shape {_emb_kernel_shape} does not match "
            f"expected ({_cfg.get('own_state_dim', 29)}, {_cfg['hidden_dim']}) -- the pad is wrong"
        )
        print(
            f"[preflight] restore-through-grafting OK for both teams at step {resume_target} "
            f"(emb_own kernel: {_emb_kernel_shape})",
            flush=True,
        )
    else:
        print("[preflight] no checkpoint present yet (fresh start) -- fossil guard not applicable", flush=True)

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
    # died 7.5 hours later without ever writing a durable checkpoint. (Later
    # the same night: the actual mechanism turned out to be checkpoint
    # retention pruning, not the FUSE path -- see
    # docs/THE-ECOLOGY-NEVER-RAN.md instances 5's retraction and 6. The
    # durability-verification discipline below stands regardless of which
    # mechanism it was catching.) "[CKPT] Committed" printed the whole time;
    # it was not evidence. Durability is a precondition for running, not a
    # property to fix while running -- verify it BEFORE trusting any
    # further GPU-hours to this run. Snapshot checkpoint_dir now, and after
    # the very first commit, confirm via a volume.listdir() RPC (not the
    # container's own FUSE view -- the same class of blind spot that hid
    # the original bug) that a new entry actually landed. Halt immediately
    # if it didn't.
    #
    # 2026-09-20 (Will, self-caught): this hardcoded "checkpoints" here --
    # stale from before checkpoint_dir moved to checkpoints_r2541/ the same
    # night. Fired FATAL on the very first real checkpoint of the
    # relaunch (ppo=2544): it was checking the OLD, abandoned checkpoints/
    # directory, which this run never writes to, while the real save into
    # checkpoints_r2541/2544 had genuinely succeeded (confirmed durable via
    # `modal volume ls` five days later). A false alarm from watching the
    # wrong directory, not a real durability failure -- but it still halted
    # the run, exactly as a real one would have. Derive the path to watch
    # from the same config the run itself uses, so the two can't drift
    # apart again.
    _cfg = build_cfg()
    _ckpt_subdir = os.path.relpath(_cfg["checkpoint_dir"], VOLUME_MOUNT)
    try:
        _ckpt_paths_before = {e.path for e in volume.listdir(_ckpt_subdir)}
    except Exception:
        _ckpt_paths_before = set()  # e.g. checkpoint_dir doesn't exist yet on a fresh volume

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
            _ckpt_paths_after = {e.path for e in volume.listdir(_ckpt_subdir)}
        except Exception as exc:
            print(
                f"[DURABILITY-GATE] FATAL: could not list {_ckpt_subdir}/ via the volume "
                f"RPC after the first commit to verify durability: {exc!r}. Halting "
                f"before burning more GPU.",
                flush=True,
            )
            raise SystemExit(1)
        _new_paths = _ckpt_paths_after - _ckpt_paths_before
        if not _new_paths:
            print(
                f"[DURABILITY-GATE] FATAL: no new entry visible under {_ckpt_subdir}/ via "
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
            f"{_ckpt_subdir}/ via volume.listdir() after the first commit -- durability "
            f"verified externally, not just trusted from the container's own print.",
            flush=True,
        )
        _durability_state["verified"] = True

    run_simulation(
        _cfg, seed=42, n_steps=n_steps,
        on_checkpoint_saved=_commit_and_verify_durability,
    )
    volume.commit()


@app.local_entrypoint()
def main(test: bool = False, n_steps: int = 0):
    if test:
        print("[preflight] running CPU smoke test (SHA/volume/config/checkpoint-restore/fossil-guard only)")
        head = _tiny_cpu_smoke.remote()
        print(f"[preflight] smoke test completed, cloned HEAD={head}")
        return
    steps = n_steps or N_STEPS_FULL
    train.remote(steps)
