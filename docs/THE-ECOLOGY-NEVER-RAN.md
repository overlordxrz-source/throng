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

*(Log format: mechanism, when it was introduced not-actually-working, when
it was fixed, how long the gap was, what it plausibly cost, and how it was
found. Append new confirmed instances below this line — suspicions belong in
a task note or `RESEARCH_PROTOCOL.md`'s Part 5 reporting discipline until
confirmed, not here.)*
