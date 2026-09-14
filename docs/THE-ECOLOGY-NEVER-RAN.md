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

*(Log format: mechanism, when it was introduced not-actually-working, when
it was fixed, how long the gap was, what it plausibly cost, and how it was
found. Append new confirmed instances below this line — suspicions belong in
a task note or `RESEARCH_PROTOCOL.md`'s Part 5 reporting discipline until
confirmed, not here.)*
