# THE ECOLOGY NEVER RAN

A log of confirmed instances where an ecological or training mechanism looked
live — present in config, read by code, producing plausible-looking numbers
on the dashboard — but was not actually operating as designed, for some
identifiable span of the project's history. Companion to
`RESEARCH_PROTOCOL.md` Part 3 (fossils): a fossil is a *policy* trained under
a defect; an entry here is the *defect* itself, dated as precisely as the
history allows, so nobody re-derives "wait, was this ever real?" from
scratch a second time.

---

## 1. Red curriculum state reset to stage zero on every process resume

**Introduced:** commit `8132d68`, "fix: value head init + red curriculum
(Cam)", 2026-05-27. `red_curriculum_idx = 0` and `red_sustain_count = 0`
were declared as plain Python locals in `_run_simulation_impl`, unconditionally,
on every call — never read from or written to the checkpoint.

**Fixed:** 2026-09-14, alongside persisting CtD competence-ramp state
(`jax_sim/main_jax.py`'s `training_state` checkpoint key). Found by noticing
that the two new ramp ratchets were built on the exact same unpersisted
pattern, then checking whether the pattern they were copied from had the
same defect. It did.

**Duration:** ~3.5 months (2026-05-27 to 2026-09-14) across however many
process resumes occurred in that window — every one of them silently
restarted red's curriculum stage at index 0 (the easiest stage,
`red_curriculum_stages[0]`) regardless of how far it had actually advanced
before the resume, and reset the sustained-survival streak counter to zero.

**Effect:** `red_curriculum_stages` is a difficulty ramp keyed on blue's
survival rate (`curriculum_survival_threshold`, default 0.80, sustained for
`curriculum_sustain_updates`, default 5, consecutive updates advances to the
next stage). A resume that silently drops back to stage 0 means: any
training time spent at an advanced difficulty stage before a resume is
*not* resumed at that difficulty — the curriculum re-climbs from scratch,
re-spending however many updates it takes to re-satisfy the sustain bar at
each stage along the way, every single restart. Whether this actually
suppressed the curriculum from ever reaching its harder stages in practice
(as opposed to just wasting re-climb time) has not been separately measured
and is not claimed here — only that the persistence gap existed and is now
closed. Read any historical dashboard `red_floor=` value from before
2026-09-14 as index-since-last-resume, not index-since-training-began.

**How it was found:** not by testing the curriculum mechanism directly —
by building the CtD competence ramp on the same unpersisted
Python-local-variable pattern, then applying `RESEARCH_PROTOCOL.md`'s "out
of scope is a decision, not an observation" principle to a defect noticed
in passing while extending it, rather than filing it away.

---

## 2. Checkpoint-restore schema-mismatch fallback never actually restored anything

**Introduced:** the `except ValueError` branch in `_run_simulation_impl`'s checkpoint-restore
block (`jax_sim/main_jax.py`) has existed for multiple phases, handling the case where a
checkpoint's param tree doesn't match the current architecture 1:1 (an old checkpoint missing
a head added since, or carrying state fields the current `model.init()` no longer produces).
Its innermost fallback, `ckpt_mngr.restore(step, items=target_dict)`, looked like a real
recovery path: present in code, reached under a named condition, followed by
`graft_missing_param_subtrees` to patch the result.

**Fixed:** 2026-09-14, while resuming from checkpoint 2541 per the calibration ladder's
codes_active rule. This is the first checkpoint in the project's history old enough (predates
`head_signal` reactivation, `nb_cross_attn`, `red_codebook`; carries `codebook_N.usage_ema`/
`dead_streak` fields the current `model.init()` doesn't produce) to actually need this
fallback — every prior resume was architecturally close enough that the unconstrained restore
one level up always succeeded and this path was never reached.

**Duration:** present, unexercised, since whichever phase introduced the schema-evolution
`except ValueError` branch (predates this session) through 2026-09-14 — an unknown but
plausibly multi-month span in which any checkpoint old enough to need it would have crashed
the run instead of resuming from it.

**Effect:** `ckpt_mngr.restore(step, items=target_dict)` goes through `CheckpointManager`'s
`"default"` item, which is bound to `StandardCheckpointHandler`. `StandardRestoreArgs`'s
`strict=False` does not loosen a key-set mismatch (only shape/dtype mismatches on keys present
on both sides — tested directly, raises the identical "do not match" error either way), and
orbax's own error message ("pass `partial_restore=True`") names a flag `StandardRestore`
doesn't expose at all; it only exists on the lower-level `PyTreeRestoreArgs`, which this
`CheckpointManager`'s handler registration refuses outright ("does not match with any
registered handler"). The fallback could never have succeeded for any checkpoint that actually
needed it — it would always hit this same crash, one level deeper than the outer schema-
mismatch handler it lived inside of, making the outer handler's recovery attempt itself only
partially real: catches the mismatch, prints "merging new heads manually," then crashes on the
next line for any checkpoint whose mismatch is a key-set difference rather than a pure
shape/dtype one.

**Fix:** bypass the Standard-bound `CheckpointManager` for this one restore and go straight to
`ocp.PyTreeCheckpointer()` against the on-disk `"default"` item, which does honor
`partial_restore`, with `restore_args` built via `ocp.checkpoint_utils.construct_restore_args`
(needed on top of `partial_restore` alone — verified locally that `partial_restore` without
explicit `restore_args` still raises `"Topology mismatch"` / `"sharding ... Got None"` against
a checkpoint saved on a different device topology than the restoring process). Regression test:
`tests/test_checkpoint_partial_restore_fallback.py`, a real Orbax round trip proving the strict
path fails first (so the fallback is genuinely exercised) and the fallback correctly restores
overlapping keys, drops checkpoint-only keys, and preserves target-only keys for
`graft_missing_param_subtrees` to fill.

**How it was found:** not by auditing the fallback in isolation — by resuming from a
calibration-picked checkpoint old enough to actually need it, per `RESEARCH_PROTOCOL.md`'s
"measure, don't assume" discipline applied to the resume point itself (2541, chosen by the
codes_active ladder rule, not by habit).

---

## 3. `--test` preflight never executed a training-loop iteration on a resumed checkpoint

**Introduced:** `scripts/modal_app.py`'s `_tiny_cpu_smoke` has passed a flat `n_steps=50` to
`run_simulation` since the file's creation. `jax_sim/main_jax.py`'s `n_updates = n_steps // T`
is an ABSOLUTE target update count, not "updates to run from here" — the loop is `for ui in
range(start_update, n_updates)`. With `n_steps=50` and `T=512`, `n_updates=0`; for any resumed
checkpoint (`start_update>0`), `range(start_update, 0)` is empty. The preflight printed every
restore-path banner (checkpoint restored, CTD-RAMP state, TRIPWIRE config) and returned
"[preflight] smoke test completed" — a real success message — without executing a single PPO
update, tripwire evaluation, or crafting-bar check.

**Fixed:** 2026-09-14, in the same session that built the tripwire and per-capita-bar logic
this gap would have silently failed to verify. `_tiny_cpu_smoke` now resolves the actual resume
point the same way `main_jax.py` will (`resume_from_step` if pinned, else the volume's own
`latest_step()`) and pads `n_steps` to guarantee at least 3 real post-resume updates run.

**Duration:** unknown start (present since the file's creation, 2026-09-14 per this session's
own commit history) through 2026-09-14 later the same day — every `--test` preflight run this
session against a resumed checkpoint (three of them, all reported as "passed") verified the
restore path only, never the loop body.

**Effect:** every "preflight passed" claim made earlier in this session about the comms
freeze, the tripwire thresholds, or the crafting-bar logic taking effect correctly was true
only for the banners printed before the loop starts. None of it proved the loop body itself
ran without error on the actual pinned SHA. The real detached launches were the first genuine
exercise of that code each time — Rule 13 in a form the reporting agent produced and repeated
without noticing.

**How it was found:** a local CPU smoke test run with a correctly-scaled `n_steps` (to verify
the stability-gated C0 capture and per-capita bar before trusting them) printed
`[JAX] Training PPO updates 2541 → -1` in an EARLIER attempt with a too-small `n_steps` —
noticing that "→ -1" meant zero updates were about to run, rather than assuming a completed
process meant work happened.

---

## 4. `red_detection_radius` — misidentified, then found to be disabled by design with an inverted comment

**Not a defect in the mechanism's operation** — a defect in this session's own citation of it,
caught before it reached a code change. Worth logging because the comment it was caught
against is itself wrong and could mislead the next person who reads it.

`red_detection_radius` gates whether **blue can see red** beyond a Chebyshev radius
(`jax_sim/observations_jax.py:206-209`, `_mask_loc_env_red_channel`) — introduced at value `8`
in commit `92479cc` ("Phase 5: forced communication architecture", 2026-05-19) specifically to
force blues who can't directly see a red to rely on neighbour alarm signals instead. It has
**nothing to do with red's ability to sense blue** — `limit_red_sensing=True` (the flag that
gates this masking) is passed only when building blue's own observations
(`jax_sim/main_jax.py:353`); red's observation build never receives it.

Commit `775ec7db` (2026-06-23, "Repo cleanup + config rename...") changed the value from `8` to
`0`, carrying forward a `# blues stay blind` design note from the P10.4 era and relabeling the
config comment to `# 0 = blind beyond 5×5 patch`. But the code's actual condition is
`if det_r > 0: apply the mask` (`observations_jax.py:207`) — at `det_r=0` the mask is **skipped
entirely**, so blue sees red **without any radius restriction**. The current comment asserts
the opposite of what the code does; the ORIGINAL 2026-05-19 comment had it right (`0 =
all-seeing (disables the mechanic)`). Source beats prose, confirmed here in both directions:
the mechanic has been off since 2026-06-23 (~3 months), and the reason it looks intentional
("blind") in the config is a comment that inverted during a rename, not a description of
current behavior. Plausible connection, not yet verified: `Alarm_Rate` has read ~0.001-0.003
in every live rollout this session — consistent with blue having no need to alarm-signal about
red sightings when it can already see every red on the grid directly.

This was caught here, not shipped: a subagent investigating predation cited
`red_detection_radius=0` as evidence that red can't sense blue, and the report to Cam repeated
that framing before the code was actually read. Corrected on Cam's explicit request to check
history before trusting either the comment or the earlier report.

**No code change made** — Cam's instruction was to hold any detection-radius change until
`catch_attempted`/`caught_b` data from a live rollout discriminates between "red can't find
blue" and "the catch path itself is broken." This entry documents the mechanism as found, nothing more.

---

## 5. Checkpoint saves silently wrote to a path `Volume.commit()` never tracked

**Introduced:** the line `ckpt_dir = str(Path(ckpt_dir).resolve())` in `_run_simulation_impl`'s
checkpoint init (`jax_sim/main_jax.py`), added at an unknown earlier point specifically to dodge
a historical Orbax `mkdir(exist_ok=True)` failure on a symlinked path (per its own comment: "Orbax
mkdir fails on symlinks (FileExistsError); use real volume path").

**Fixed:** 2026-09-14, same session as the volume-commit periodicity fix (`f367f02`) that this
defect immediately exposed. `.resolve()` silently resolves Modal's mounted volume path
(`/mnt/throng-runs/checkpoints`) to the volume's internal backing-store path
(`/__modal/volumes/vo-.../checkpoints`) — which bypasses the FUSE mount's write-tracking
entirely. Every checkpoint write went through the resolved path from the moment this line was
added; `Volume.commit()` (added later, this same session, believing it fixed durability) kept
returning successfully because it genuinely was committing the volume — the checkpoint files
just were never part of it.

**Duration:** unknown start (predates this session) through 2026-09-14. Every checkpoint saved
in that window that wasn't captured by the run's own single end-of-run `commit()` (the pre-fix
behavior) was lost on any interruption before natural completion — the same risk the periodicity
fix believed it had already closed, and had not.

**Effect:** compounds directly with instance found earlier: the periodicity fix (checkpoint saves
now commit on the same cadence as the save itself) was necessary but not sufficient, because the
thing being committed was never the checkpoint. Five real periodic saves during the 2541 resume
(steps 1302528, 1304064, 1305600, 1307136, 1308672) all printed `[CKPT] Saved` / `[CKPT]
Committed` with no error, and none were retrievable from the volume by any method — `modal volume
ls`, a direct path lookup, and a raw filesystem check from inside the live container all agreed
on the same stale 3-checkpoint listing. Not caught by the earlier fix's own verification, because
that verification checked whether `commit()` was being *called*, not whether the write it was
committing was in the volume at all.

**Fix:** `os.path.abspath()` instead of `Path(...).resolve()` for the path the
`CheckpointManager` actually writes through — absolute, but does not follow symlinks, so it
cannot be silently redirected off the mounted volume. For the already-absolute production value
this is a complete no-op. The protected-backup guard (`assert_checkpoint_dir_is_not_protected_backup`)
still needs a genuinely resolved path to catch a `checkpoint_dir` that reaches the backup via a
symlink — a separate local copy is resolved just for that one check, so the safety guarantee is
unchanged. Verified empirically before landing: `mkdir(parents=True, exist_ok=True)` on the
unresolved mount path raises no error on this Python/OS combination (the historical failure the
`.resolve()` call was added to avoid does not reproduce here), and a probe file written through
the unresolved path committed and became externally visible on the first try, while the resolved
path never did across five real attempts.

**How it was found:** Cam's one-measurement instruction — resolve the path the exact way
`main_jax.py` does, `os.path.exists` it, list its parent, and compare against the mount root —
run live against the actual running container via `modal container exec`, not reasoned about
from source. The corpus writer (`communication/analysis.py`'s `SignalCorpusWriter`) was checked
by the same standard and found NOT to share this defect: it checks `os.path.isdir("/mnt/throng-
runs")` and joins onto that raw path directly, with no `.resolve()` anywhere in its construction
— confirmed durable via the live volume listing (`signal_corpus.jsonl`, 623 MiB, modified
throughout the run, not stale from an earlier one).

**Postscript (2026-09-15):** this fix (`11f0bb2`) never reached the container running at the
time it was diagnosed — that run was already provisioned from the older, pre-fix pinned SHA
(`a0c48ed`), and per Cam's explicit ruling was deliberately not restarted onto the fix. It ran a
further 7.5 hours (ppo 2541→2584) and was lost in full to this exact bug when an unrelated
tripwire halted it — the emergency checkpoint at the halt included. See
`docs/RESEARCH_PROTOCOL.md` Part 2, "durability is a precondition for running," for the
general rule this instance forced into writing, and `scripts/modal_app.py`'s
`DURABILITY-GATE` check for the fix: verify durability externally (`volume.listdir()`) after
the first commit of any new launch, before trusting it with further GPU-hours.

---

*(Log format: mechanism, when it was introduced not-actually-working, when
it was fixed, how long the gap was, what it plausibly cost, and how it was
found. Append new confirmed instances below this line — suspicions belong in
a task note or `RESEARCH_PROTOCOL.md`'s Part 5 reporting discipline until
confirmed, not here.)*
