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

## Status as of 2026-09-20 — read this first if picking up cold

The run is **stopped, deliberately, and nothing should relaunch** until the items below are
addressed. This is not a crash or an infrastructure failure — everything in instances 1-6 is
fixed and confirmed working. The stop is because both of the ecology's pre-registered
selection pressures (`RESEARCH_PROTOCOL.md`'s "two pressures must be live, not merely
configured — predation must produce catches, and crafting must produce successes") were
checked directly against the corpus and log, on the actual live-run data, this same day:

- **Predation was structurally impossible**, independent of red's policy — see instance 7,
  below. **Fixed** (`is_big_green_mutation_rate: 0.03`, verified against the real
  reproduction path over 300 generations) but **not yet exercised on a real run** — the fix
  landed after the run was stopped for this finding, not before.
- **Crafting is real but rare** (103 successes across 41 stage-1 updates measured directly
  from the log, not zero — an earlier single-snapshot read had wrongly suggested zero) and
  **flat, not learning** (linear-fit slope ≈ 0.10 successes/update over the clean single-
  sourced window, statistically indistinguishable from zero) — consistent with the pressure
  being real but below the population's current learning threshold, not with the mechanism
  being broken. Verified to fire correctly under ideal conditions at production shapes
  (`tests/test_pair_craft_success_production_shapes.py`).
- **Informed agents (`can_see_recipe`) are measurably *worse*, not better, at holding a
  needed material when attempting to craft** (95% CI entirely below zero — see
  `RESEARCH_PROTOCOL.md` Part 2's 2026-09-20 entry). This bears directly on whether the
  receiver-necessity premise (recipe knowledge should confer an advantage) holds at all, and
  was logged as **measured, not diagnosed** at the time. It was in fact the second instance of
  instance 7's ratchet — see instance 8, below. **Fixed**
  (`can_see_recipe_mutation_rate: 0.03`, verified against the real reproduction path over 300
  generations, same as `is_big_green`) but likewise **not yet exercised on a real run**.
- **The mechanism behind the craft-material gap was tested and is not what it was suspected to
  be.** Cam's conjecture — informed agents search specifically for a needed material in a
  world where materials are zoned (each cell yields only 2 of 5), so they're more often
  empty-handed because they searched the wrong zone — was tested directly. Zone availability
  (the demanded material present anywhere in the agent's own zone at craft-attempt time): 
  **58.07%** measured across informed craft attempts (prior stated as "around 40%"). The
  specific causal cross-tab contradicts the conjectured mechanism: empty-handed *is higher*,
  not lower, for informed agents even when their zone does contain the needed material — the
  zoning mismatch is real but is not, by itself, the explanation for the measured gap. See
  `RESEARCH_PROTOCOL.md` Part 2's 2026-09-20 zone-availability entry for the full breakdown.
  **The zone/recipe mismatch fix itself (zone-local recipes vs. de-zoning) has deliberately not
  been implemented** — Cam: "Don't implement yet," pending review of this number.
- **Big-green catching requires 2+ coordinating red agents**, not that big-green is
  uncatchable — a red-policy coordination-rate question, separate from and downstream of the
  population fix above. **Known open item, explicitly not being fixed now** — see instance 8's
  closing note.
- **Both population-ratchet fixes landed this same day are untested on a real run.** Neither
  `is_big_green_mutation_rate: 0.03` (instance 7) nor `can_see_recipe_mutation_rate: 0.03`
  (instance 8) has been exercised against a live training run — both were verified only
  against the real `apply_auto_reproduce` function directly, over 300 synthetic generations
  each, with a mortality asymmetry standing in for the measured fitness disadvantage. The run
  was stopped before either fix could be landed and relaunched, and relaunch remains gated (see
  below), so this stays true going into the next session.
- **The zone-local-recipe proposal (Rule 12) is withdrawn.** Cam's own empty-handed mechanism
  (informed agents search the wrong zone) was tested and falsified — see the zone-availability
  entry above. What replaced it: **the collision arithmetic**, independently re-derived from
  the actual corpus and log (not taken on faith) — see below — showed the observed crafting
  success rate is statistically indistinguishable from what pure independent-chance collision
  between two randomly-moving agents would predict on its own, given their individually
  measured (not assumed) rates of holding a needed material and choosing to craft. **No
  coordination beyond chance has ever occurred at this mechanism**, and zone-local recipes
  would not move a co-location rate this low. **New ruling: give the world a crafting site** —
  a designated location where materials are deposited and persist, turning the two-body
  simultaneity problem into a one-body navigation problem and making recipe knowledge
  concretely actionable to transmit ("the site needs flint" vs. today's unusable "I have
  flint, come to me"). **Not yet implemented** — pending a full spec.
- **Collision-rate verification (2026-09-21, re-derived independently from the corpus/log,
  correcting the scoping used in the first pass):** the naive `step >= 1,323,008` filter used
  for a first attempt at this calculation is contaminated — `train.log` and the migration
  bundle's corpus contain **two separate relaunches that both resume-renumber from ppo≈2542**,
  so the same (step, ppo) pair appears more than once with different outcomes, and a step
  threshold alone pulls in an unrelated later relaunch (ppo 2782+). The true clean window,
  found by requiring file-order ppo continuity (not just a step floor), is the same one
  already established for the trend measurement above: **ppo 2584–2596 (13 updates, steps
  1,322,497–1,329,152)** — 100% genuine stage-1 (2-unit) recipes in this window, no stale
  full-recipe or stage-0 contamination. Measured directly from the corpus in that window:
  agents hold a recipe-needed material 17.80% of the time (not the ~13% estimated), and craft
  7.07% of the time overall, 9.17% of the time specifically when already holding a needed
  material (population-wide craft-action frequency is *not* the right multiplier once you've
  already conditioned on holding — using it inflates the estimate by roughly 2x). Recomputing
  Cam's exact model (Chebyshev radius-1 adjacency, `8/16,383` on a 128×128 torus — confirmed
  against `resolve_crafting`'s actual adjacency check) with these self-consistent, corpus-
  measured inputs: **expected 2.44–2.59 successes per 512-step rollout** (N=194–200) against
  **observed 2.00** (26 `craft_success` flags over 13 updates) — a Poisson z-score of −1.17,
  not significant. **Confirms Cam's conclusion, with corrected inputs**: the pressure has never
  exceeded what two independently-moving agents would produce by accident.
- **What already exists in the codebase resembling a site, checked before speccing a new
  mechanism (Cam's request):** no crafting-site or depot mechanism exists. The closest
  precedents, both in `jax_sim/grid_jax.py`, live and wired in but structurally different:
  `generate_contested_nodes`/`contested_res` (fixed random grid locations, `contested_n_nodes`
  of them, requiring `contested_min_harvesters` agents simultaneously *present* — not
  depositing anything — in the same step for a resource-yield bonus; wired in
  `jax_sim/main_jax.py` ~line 654-665, `agent_count` is a fresh per-step scatter-add headcount,
  nothing persists) and `update_scent_trails`/`scent_trails` (a genuine deposit-and-decay
  persistent grid field, but red-only, untyped — a single intensity scalar per cell, not a
  material — and unrelated to crafting). Neither is a site in the sense Cam means (a fixed
  place holding typed, persistent, depletable material); a real crafting depot would need to
  combine `contested_res`'s "fixed known locations, config-driven count" with
  `scent_trails`'s "deposit persists and decays/depletes over time" pattern, plus per-material
  typing neither currently has.

**Relaunch gate, verbatim (Cam, 2026-09-20): "Nothing relaunches until the zone-availability
number is in and the recipe fix lands. Running Gate C in a world that punishes using
information would produce a null that means nothing."** The zone-availability number is in
(above); the zone-local-recipe fix is withdrawn and replaced by the crafting-site proposal,
which has not landed. **Do not relaunch training.**

**Migration bundle:** the full state needed to resume on a new Modal account/workspace —
both live checkpoints, all 6 ladder fossils, both signal corpora, the complete `train.log`,
`config.yaml`, and the exact pinned commit — was pulled to local disk at
`~/throng_migration_2026-09-20/` (2.9 GB, verified: byte-identical to the independently-
trusted `~/throng_backup` copy where overlapping, valid JSONL, valid Orbax metadata on every
checkpoint). **Read `~/throng_migration_2026-09-20/MIGRATION.md` before uploading anything to
a new volume** — it has the upload order and the fossil-guard constraint (never put a
higher-numbered fossil in the live checkpoint directory; see instance 6, below, for what that
costs).

**Retractions on the record, so they aren't rediscovered:** instance 5's central causal claim
(`.resolve()` bypassing `Volume.commit()`) does not survive a controlled test and is retracted
in that entry's own postscript — the real mechanism was instance 6. `RESEARCH_PROTOCOL.md`
Part 2 carries a matching correction. An earlier report this same day that crafting produced
zero successes across stage 1 was also wrong (single-snapshot extrapolation) — see the
success-series entry above.

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

**Retraction (2026-09-15, same day, later):** the central causal claim above — that `.resolve()`
bypassing `Volume.commit()`'s tracking was the mechanism losing checkpoints — does not survive a
controlled test and should not be trusted. The "five real attempts" cited as evidence for the
resolved path failing (steps 1302528/1304064/1305600/1307136/1308672, i.e. ppo 2544-2556) were
**all** numerically below two fossil checkpoints (2859, 2862) already sitting in the same
directory from an abandoned lineage — see instance 6, below. Every one of those five saves is
fully and independently explained by Orbax's `max_to_keep` retention pruning them by step number,
with zero need to invoke path resolution at all. That evidence was never a controlled comparison;
it was correlational, and confounded by a variable nobody had identified yet.

Re-tested properly on 2026-09-15, in a fresh directory with no fossils (`_diag_resolve_test/`, so
retention pruning cannot be an alternative explanation for anything observed): one file written
via `str(Path(...).resolve())` (the retired construction, literally reproduced), one via
`os.path.abspath()` (the current one), one `volume.commit()` covering both, checked externally.
**Both files appeared.** `.resolve()` resolving `/mnt/throng-runs/_diag_resolve_test` to
`/__modal/volumes/vo-T496EY9LIBErLGI6JuMxzs/_diag_resolve_test` is real (confirmed, both dated
2026-09-15 11:13 EDT) — but a write through that resolved path commits and becomes externally
visible identically to a write through the unresolved one. There is no evidence, controlled or
otherwise, that `.resolve()` ever bypassed `Volume.commit()`'s tracking.

Verdict, per Cam's own framework (Part 2 of `RESEARCH_PROTOCOL.md`, "an instrument can fail on
its construction alone"): this was not a genuine latent bug that happened not to be the bug
losing checkpoints — there is no bug in `.resolve()` at all, here. It was a diagnosis built on a
confounded correlation, and a change to working code made under that mistaken theory. `git diff`
against `os.path.abspath()`: functionally identical for the production path (already absolute, no
`../` segments, both proven to commit correctly), so it is not reverted — there is no benefit to
reverting a change that is now proven harmless, only churn. But it fixed nothing, and did not
cause the loss it was credited with preventing on the SHA that carried it. The actual fix is
instance 6's guard, not this one. Docs and commit messages crediting `os.path.abspath()` /
`11f0bb2` with resolving checkpoint durability should be read with this retraction attached.

---

## 6. `max_to_keep` retention silently deleted every checkpoint a rolled-back resume saved

**Introduced:** not a code change at all — `ocp.CheckpointManagerOptions(max_to_keep=2, create=True)`
in `_run_simulation_impl`'s checkpoint init has read `max_to_keep=2` since before this session.
The defect is a configuration meeting a situation it was never checked against: Orbax's retention
prunes by **step number**, not save recency, keeping only the numerically highest `max_to_keep`
checkpoints it finds in the directory on each save. `resume_from_step=2541` in `config.yaml` is a
**deliberate rollback** ("2026-09-14 (Cam): resume_from_step pins an EARLIER-than-latest checkpoint
deliberately... Set in config.yaml, not a volume deletion — explicit, auditable, reversible" —
`jax_sim/main_jax.py`). Rolling back to 2541 while the directory still held `2859` and `2862`
(fossils from the lineage the rollback was rolling back *from*) meant every checkpoint the
resumed run went on to save — 2544, 2545, 2546, ... — was numerically *lower* than those fossils.
Orbax's own retention pruned each one on save, silently, before `Volume.commit()` ever ran.
`[CKPT] Saved` / `[CKPT] Committed` printed unconditionally regardless — neither line checks
whether the thing just saved is still there a moment later.

**Fixed:** 2026-09-15.

**Duration:** the rollback to 2541 was set 2026-09-14; the defect existed the entire time
`checkpoints/` held both the 2541-lineage and the higher-numbered 2859/2862 fossils from the
abandoned lineage — i.e. for both of the checkpoint-durability incidents logged in instance 5,
end to end. Not a new defect discovered on 2026-09-15; the same defect, finally isolated from the
`.resolve()` misdiagnosis that had been standing in front of it.

**Effect:** explains **both** lost runs without any reference to `.resolve()` or the FUSE mount at
all — the 2026-09-14 19:12 EDT–2026-09-15 02:51 EDT run (7.5 hours, 44 PPO updates, the entire
cascade-defusal sequence) and the first 2026-09-15 relaunch attempt (killed independently by an
unrelated bug in the new durability gate's own `volume.reload()` call, but its one real checkpoint
at ppo=2544 was *already* gone from the volume by the time that crash happened — confirmed absent
on a fresh `modal volume ls` immediately after, and again ~19 minutes later, ruling out
propagation delay). Both runs did everything right — saved on schedule, called `commit()` on
schedule — and lost their output anyway, because the directory they were pointed at was not the
clean, single-lineage directory everyone believed it was.

**Fix:** three parts, deliberately non-destructive (see the retraction on instance 5 for why
deletion was ruled out as a response to a deletion mistake made diagnosing this one):
1. `max_to_keep=2` → `10` (`jax_sim/main_jax.py`) — defense in depth, not the actual fix; ~26MB
   per checkpoint makes 10 negligible, and it removes the single-fault-tolerance failure mode
   where one bad save plus one bad predecessor loses everything.
2. A startup guard (`jax_sim/main_jax.py`, right after the resume target resolves, before the
   first rollout): enumerate every step Orbax recognizes in `checkpoint_dir` via
   `ckpt_mngr.all_steps()`, and refuse to launch (print the offending steps, `SystemExit(1)`) if
   any sits numerically ahead of the resume target. Same shape as `scripts/modal_app.py`'s
   `DURABILITY-GATE`: assert the environment is what you think it is before you run in it, rather
   than discovering the mismatch after GPU-hours are spent.
3. New lineage, new directory: `checkpoint_dir` now points at `checkpoints_r2541/`, seeded with
   only the 2541 checkpoint (restored from the local `~/throng_backup` copy — the volume's own
   2541 was deleted, see below). `checkpoints/` (2862, and this session's own diagnostic litter
   from before the mistake was caught) is left exactly as it was: inert once nothing points at
   it, and now itself evidence for this instance rather than live state.

**How it was found:** Cam's own three-outcome discriminator, run CPU-only before spending GPU
time on a third launch attempt: probe file + real Orbax save + `volume.commit()`, checked
externally. Both appeared — ruling out the path and ruling out Orbax's write pattern — which by
elimination pointed at "timing or state-dependent, what's different in the real run." The
discriminator itself was run against the *live* `checkpoints/` directory (matching production
exactly, per the instructions), which is what exposed the actual mechanism directly: the
diagnostic's own save (step 999999) triggered the same retention pass, and the before/after
external listing showed `2541` and `2859` gone immediately after — an unintended, costly, but
diagnostically decisive side effect, reported in full before any further action was taken.

**Incidental cost:** running the discriminator against the live directory (as specified, without
first isolating it into a scratch directory) deleted `checkpoints/2541` and `checkpoints/2859`
from the volume. `2541` was recoverable intact from `~/throng_backup/checkpoints/2541` (verified,
26MB, standard Orbax layout) and has been restored into the new `checkpoints_r2541/` lineage.
`2859` was not in that backup (which tops out at `2763`) and is not recoverable — low-stakes,
since it was one of the two fossils implicated in this instance to begin with, not a checkpoint
anyone was resuming from. The backup itself was, at the time this was noticed, the only surviving
copy of the entire pre-2859 record of this project outside the volume — six ladder checkpoints
(2490, 2493, 2538, 2541, 2760, 2763) have since been archived into a `fossils/` directory on the
volume, restoring off-laptop redundancy, deliberately separate from any directory a
`CheckpointManager` is ever pointed at live.

---

## 7. Blue's population composition was a one-way demographic ratchet — predation was
structurally impossible independent of red's policy, and no line of code was wrong

**Introduced:** `init_population` (`jax_sim/population_jax.py`, ~line 176) has always drawn
`is_big_green` as a one-time 20% random assignment at simulation start. `apply_auto_reproduce`
(~line 305-306) has always inherited it unchanged from an assigned parent at reproduction:
`parent_is_big_green = pop.is_big_green[assigned_parents]`, `new_is_big_green =
jnp.where(activate_mask, parent_is_big_green, pop.is_big_green)`. Both lines do exactly what
they say. Neither is a bug. `can_see_recipe` (~line 176-177, ~line 308-309) is built the same
way — a one-time 50% draw, inherited unchanged thereafter.

**The defect is structural, not a line-level error — the cleanest instance in this file for
exactly that reason.** `assigned_parents` (~line 239-241) is sampled *uniformly across the
currently alive population*, zero weighting by `is_big_green`: `parent_weights =
jnp.where(pop.alive, 1.0, 0.0)`. Combined with red's catch mechanic being small-blue-only in
practice (big-green requires 2+ coordinating reds post-`coop_threshold_step`, see instance
below in `RESEARCH_PROTOCOL.md`'s 2026-09-20 entries — small-blue has no equivalent
protection), small-blue's population is a one-way ratchet: any net predation pressure pushes
its alive-fraction down, which proportionally *lowers* its representation among new spawns
too (no compensating force), which pushes the fraction down further. At exactly 0 alive, the
parent-sampling weight for producing a new small-blue agent is exactly 0 — permanently, from
that population state on. Every mechanism involved was doing precisely what it was written to
do; the outcome is nevertheless a channel that spent its entire observable history unable to
apply the predation pressure it was designed to test.

**Duration:** unknown start (this is initial-population and reproduction logic, present since
before any session covered in this log) through 2026-09-20. Measured live-run state on
2026-09-20: `pop_split=small:0|big_green:194-200` across essentially every stage-1 update —
confirmed via the corpus and `train.log`, not inferred.

**Effect:** the entire red half of the ecology was decorative for as long as small-blue sat at
or near 0. `catch_attempted` (small-blue-only by construction) correctly read 0 or
near-0 — not an instrument failure (the earlier `catch_attempts=0` investigation, Part 2 of
`RESEARCH_PROTOCOL.md`, correctly identified an empty denominator; this instance is *why* the
denominator was empty). No amount of red-policy tuning could have produced catches during this
period, because the prey type red's mechanic can actually reach barely existed.

**Fix:** not a floor. Cam's explicit framing: "A floor is an external hand preventing an
outcome selection is producing; mutation is a rule of the world." Added
`is_big_green_mutation_rate` (default `0.03`, `config.yaml`) to `apply_auto_reproduce`: a
per-birth probability the offspring's `is_big_green` flips relative to its parent's,
independent of current population composition — present at 0% small-blue exactly as it's
present at 50%, the same way a real mutation rate doesn't care how rare the recessive allele
has become. `can_see_recipe` is the same structural pattern (fixed trait, one-time draw,
unchanged inheritance) under its own measured negative selection — see instance 8, below,
where the same fix was applied once the mechanism was understood well enough to justify it.

**How it was found:** Cam asked directly — "What transitions an agent to big_green, is it
reversible, and can small blue exist in steady state at all?" — after `pop_split=small:0` had
already been visible in the dashboard for three days without anyone asking what produced it.
Answered by reading `apply_auto_reproduce` directly, not by reasoning about red's policy.
Verified, not asserted: `tests/test_population_composition_lockin.py` proves the lock-in
against the real function (at 0 small-blue alive, every new spawn under
`is_big_green_mutation_rate=0.0` is confirmed big-green); `tests/test_is_big_green_mutation.py`
runs the real `apply_auto_reproduce` forward 300 real generations from a 100%-big-green start
under asymmetric (predation-like) mortality and confirms small-blue reappears (first
nonzero at generation 7) and stabilizes at a low nonzero tail frequency (0.74% mean, range
0-2% over the final 50 generations) rather than staying extinct or exploding to parity.

---

## 8. `can_see_recipe` was the second one-way demographic ratchet, under measured negative
selection with no reverse path

**Introduced:** same code, same day as instance 7 — `init_population` draws `can_see_recipe`
as a one-time 50% random assignment; `apply_auto_reproduce` inherited it unchanged from an
assigned parent (`parent_can_see = pop.can_see_recipe[assigned_parents]`, `new_can_see =
jnp.where(activate_mask, parent_can_see, pop.can_see_recipe)`) with the same uniform-by-alive-
count parent sampling instance 7 diagnosed for `is_big_green`. Instance 7 deliberately left this
trait alone: the fix needed a *reason*, not just a structural match, and at the time nobody had
checked whether `can_see_recipe` was actually under selection pressure, in which direction, or
by how much.

**The reason arrived the same day.** `RESEARCH_PROTOCOL.md` Part 2's 2026-09-20 entry measured
informed agents (`can_see_recipe=True`) holding a recipe-needed material at craft-attempt time
*less* often than uninformed agents — 11.89% vs 16.19%, 95% CI on the difference [-5.60, -2.99]
points, entirely below zero. Informed is measurably the *worse* trait to inherit right now.
Combined with the zero-selection-weighted parent sampling instance 7 already proved makes any
net-disadvantaged fixed trait a one-way ratchet, `can_see_recipe` was heading for fixation at
0% on exactly the same mechanism as small-blue — a second, independent ratchet that would have
killed the informed caste, and with it the entire receiver-necessity premise the ecology is
built to test, regardless of whatever else got fixed about recipe zoning.

**Duration:** structurally present since `init_population`/`apply_auto_reproduce` were written
(same "always," unknown-start caveat as instance 7); confirmed under active negative selection
as of the 2026-09-20 craft-material measurement above. Not yet observed to reach 0% on a live
run — caught and fixed from the mechanism and the measured fitness gap, the same way instance 7
was fixed before small-blue's reappearance was ever tested live, not after a `pop_split` read
showed it gone.

**Effect:** none yet observed on a live run (no run has stayed up long enough post-diagnosis to
reach fixation), but the same argument that made instance 7's fix non-optional applies here:
absent mutation, a fixed trait under sustained net-negative selection with zero reverse path
converges to 0% given enough generations, deterministically, independent of policy quality on
either side.

**Fix:** identical medicine to instance 7, per Cam's direct instruction ("Apply the same
medicine... Same rate, same test"): added `can_see_recipe_mutation_rate` (default `0.03`,
`config.yaml`) to `apply_auto_reproduce` — a per-birth probability the offspring's
`can_see_recipe` flips relative to its parent's, independent of current population composition.
Uses its own RNG split (`k6`, distinct from `is_big_green`'s `k5`) so the two mutation rolls
don't interfere; confirmed independent in
`tests/test_is_big_green_and_can_see_recipe_mutation_are_independent`.

**How it was found:** not found — anticipated. Cam named the structural parallel directly
("`can_see_recipe` is the second ratchet... same structure as `is_big_green`... Apply the same
medicine") once the craft-material measurement gave the fix a justified direction (informed is
disadvantaged, so mutation should be introducing informed agents into an uninformed-trending
population, not the reverse). Verified, not asserted, the same way as instance 7:
`tests/test_can_see_recipe_mutation.py` runs the real `apply_auto_reproduce` forward 300 real
generations from a 100%-uninformed start, under a mortality asymmetry standing in for the
measured fitness disadvantage (informed agents die at 0.25/generation vs uninformed at 0.05,
modeling the measured craft-material gap), and confirms informed reappears (first nonzero at
generation 4) and stabilizes at a low nonzero tail frequency (0.43% mean, range 0-1.5% over the
final 50 generations) rather than staying extinct. A `mutation_rate=0.0` regression guard
confirms the original lock-in reproduces exactly when the fix is disabled.

**Known open item, explicitly not being fixed now:** red's catch mechanic requires 2+
coordinating reds post-`coop_threshold_step` (instance 7's cross-reference) — a red-policy
coordination-rate sparsity, separate from and downstream of both population ratchets above.
Cam: "Red stays untouched... note it as a known open item, don't fix it now."

---

*(Log format: mechanism, when it was introduced not-actually-working, when
it was fixed, how long the gap was, what it plausibly cost, and how it was
found. Append new confirmed instances below this line — suspicions belong in
a task note or `RESEARCH_PROTOCOL.md`'s Part 5 reporting discipline until
confirmed, not here.)*
