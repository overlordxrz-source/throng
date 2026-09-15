# THRONG Research Protocol

**Read this at the start of every session, before touching code or interpreting any result.**
It is the distilled output of the Sep 2026 audit, in which thirteen defects were found that
each had one property in common: something reported success while doing nothing, and a
measurement was taken anyway.

---

## Part 1 — Code integrity: prove the work happened

A success message, a green test, an absent error and a plausible wall-clock time are all
compatible with nothing having occurred. Before believing any step did what it claims,
**measure an effect**.

- **Iteration counts.** Assert the loop body ran. `n // batch` is zero whenever `n < batch`,
  the loop silently skips, and the success line still prints. Degenerate cases must raise.
- **Coefficients.** A term multiplied by zero is dead. A live consumer reading state whose
  trainer is disabled is worse than dead — it is misleading.
- **Gradients.** Prove `‖∂loss/∂param‖ > 0` on real data at production shapes, per term, for
  each team. Keep *structural* (a raw gradient reaches the parameters at all) separate from
  *effective* (the coefficient-weighted, optimizer-applied delta matches config).
- **Declared but never written.** A state field can exist, be initialized, be read by
  consumers, and have no write site anywhere. Every state grid needs a proven write site.
- **Read but never used.** A config value can be loaded into a local that nothing references.
  Checking "is this key read?" misses it. Check that it reaches an effect.
- **In-bounds is not correct.** Index arithmetic that lands inside the right block is the
  hardest error to see. Derive offsets from the canonical layout function, never by hand.
- **Fixtures from production config.** Zero-arg constructors test dataclass defaults. A test
  that cannot fail when the live layout changes is decoration.
- **"Out of scope" is a decision, not an observation.** Noticing a defect while building on
  the exact thing it affects is not a neutral fact to log and move past — the scope question
  is already answered by the fact that you're standing inside it. `red_curriculum_idx` and
  `red_sustain_count` (plain Python locals, reset to zero on every process resume, never
  checkpointed) were noticed this way and scoped out once; the CtD competence ramp was then
  built on the identical unpersisted pattern, silently inheriting the same defect a second
  time (`THE-ECOLOGY-NEVER-RAN.md` #1). The noticing is the hard part. Don't spend it and then
  not act on it.

## Part 2 — Experiment integrity: is the experiment running at all?

Code that works is not an experiment that runs. These are separate failures and the second is
more expensive, because it produces publishable-looking nulls.

- **A configured mechanism is not a live mechanism.** Presence in `config.yaml` proves
  nothing. Every selection pressure must be confirmed by a *measured effect in a live
  rollout* before any result gathered under it is interpreted. Predation must produce
  catches. Crafting must produce successes.
- **Distinguish "tested and failed" from "never tested."** A null measured while the
  mechanism was inert is not evidence about the hypothesis. It is evidence about the
  apparatus. Six phases of receiver-side nulls turned out to be the latter.
- **Instrument failure modes before running.** A single `failed` counter cannot tell you
  whether the failure was logistical or informational. Partition every failure into causes
  that map onto different conclusions, or the result will be uninterpretable.
- **Pre-register bars, and never move them after seeing data.** Tightening on *external*
  evidence before a measurement is legitimate; loosening after is not. A marginal pass of a
  bar set in advance is suspicious, not successful. **Confirmed instance (2026-09-14):** Gate
  A's stop condition 3 was pre-registered as "dead-code resets still firing every update at
  production scale." Resets fired on nearly every update of the real launch, including a
  single update resetting 81% of total codebook capacity (51/56/48 of 64 per slot). Reported
  Gate A as clear by substituting a magnitude comparison against the pre-fix pathology (100-157
  fixed count every update, no variation) for the pre-registered criterion (fires every
  update) — a distinction that may turn out to be correct, but was introduced after seeing the
  data, by the same person who wrote this rule. Caught by Cam, not self-caught.
- **A threshold fitted to the observed distribution is not a threshold.** Adaptive cutoffs
  derived from the data they judge are p-hacking with extra steps.
- **Retiring an instrument is not the same act as moving a bar, but it is easy to disguise
  one as the other.** Moving a bar is the same instrument with a new threshold picked after
  it failed. Retiring an instrument requires independent evidence it doesn't track what it
  proxies — not a threshold failing, not a checkpoint that happens to pass or fail under it.
  **Confirmed instance (2026-09-14):** `rel_spread` (z_e cross-agent continuous variance,
  relative to peak magnitude) was proposed as a proxy for codebook-channel health, then
  retired within the same session, on a calibration ladder built specifically to test it:
  step 2538 has the ladder's LOWEST `rel_spread` (1.18%) sitting next to near-top
  `codes_active` (30|39|28/64); step 2862 (terminal collapse, `codes_active`=1|3|3/64) has a
  HIGHER `rel_spread` than both 2760 and 2763. An instrument that goes up as the channel
  dies is not measuring the channel. That contradiction is visible without reference to any
  threshold and without any checkpoint passing or failing a gate — the retirement is
  licensed by the ladder's internal structure, not by 2862's number being inconvenient. Any
  future instrument swap must meet that same standard, stated in writing, before it is
  accepted: name the independent contradiction, not the failed threshold.
- **A threshold compared against a single stochastic sample of the quantity it gates is a
  design error, identifiable without running anything.** It doesn't need a run to fail
  before it's flagged — the error is visible from the shape of the comparison itself.
  **Confirmed instance (2026-09-14, Cam's own — logged by the same standard as the bar-
  moving entry above):** the first comms-freeze tripwire draft set `C0` (the stage-0
  reference for `codes_active`) from a single production update — the first one after
  resume — then applied a zero-tolerance floor against it. `codes_active` is a count over a
  rollout batch; consecutive samples land below one another roughly half the time on noise
  alone, so a zero-tolerance floor anchored to one sample has an expected time-to-false-halt
  of 2-3 updates. Compounded by the specific resume point (2541, trained under the pre-fix
  ecology, resuming into the fixed one): update 1 is a distribution-shock sample, the single
  worst update in the run to anchor a floor to. Caught and fixed before any run — `C0`
  replaced with the per-slot median of updates 3-7 after resume (updates 1-2 discarded
  outright, tripwires arm at update 8), and the zero-tolerance floor replaced with a
  noise-tolerant `<0.85*C0` for 3 consecutive updates. The `<0.6*C0`-for-5 "slow" trip was
  also retired in the same pass — not because it was wrong, but because it was redundant
  with the fast trip *and* blind to the actual decay mode finding 3 identified (a gradual
  decline that crosses 0.75 and stays there for hundreds of updates without ever forming a
  5-update streak below 0.6) — replaced with a `<0.75*C0`-for-15 drift trip verified against
  the real 2541→2763 history (slot1: 35→12, 0.34x over ~220 updates) before being trusted.
  **Second instance of the same error, same parameter, live on the GPU (2026-09-14, Cam's
  own again):** the fixed-window fix above still compared against a fixed calendar window
  (updates 3-7) rather than a *stable* one. Live data: slot1 sampled 14|13|13|30|52 across
  those five updates while still climbing out of the post-resume transient toward its
  settled ~50s — the median (14) landed on the floor of a recovery, not a baseline. Indexed
  to that C0, slot1's fast trip would fire at 7 against an operating level near 52: a 70%
  real collapse would have passed silently. Caught by re-deriving the rule from the raw
  per-update samples before trusting the number, not by the tripwire firing. Fixed:
  fixed-window capture replaced with stability-gated capture — `C0` = median of the first 5
  *consecutive* updates (window start ≥3) in which every internal update-over-update
  transition is ≤20% relative change on all three slots, forced to the median of updates
  35-40 (logged loudly) if nothing stabilizes by the same 40-update budget as the stage-0
  hard escape. Every candidate window is printed as evaluated, so the stabilization is
  visible in the log rather than asserted. Verified against the real data before landing:
  the rule rejects the 3-7 window it replaces (fails at 4→5, −24%; 5→6, +130%; 6→7, +73%)
  and accepts the first window where all three slots move under 8% between consecutive
  updates.
- **An instrument can fail on its construction alone, independent of any threshold or any
  update ever passing or failing it — check the mechanism, not just the outcome.**
  **Confirmed instance (2026-09-14, Cam):** the CtD crafting ramp's stage-advance bar was
  `successes / (successes + futile attempts)`, i.e. success rate. Live data: `success` held
  flat (40→35 update-over-update) while `futile_wrong_mats` nearly tripled (2875→7432), so
  the rate *fell* (1.3%→0.4%) — not because blue got worse at crafting, but because blue
  tried harder without materials. The denominator is under the agent's own control; an
  agent that attempts more looks less competent by this metric regardless of what it
  actually achieves. That is a construction defect, provable by reading the formula against
  what the policy can influence, and does not require the bar to ever have passed or failed
  to be disqualified — same standard as the `rel_spread` retirement (Part 2, above): name
  the mechanism-level defect, not a failed threshold. Replaced with a per-capita bar —
  `successes / living_blue_population ≥ 10%`, sustained 3 updates — which the policy cannot
  depress by attempting more.
- **Source beats prose.** Where documentation and code disagree, the code wins and the
  disagreement is itself a finding to record.
- **Durability is a precondition for running, not a property to fix while running. A run
  that cannot save its output has zero expected value no matter how well it's going.**
  Restarting costs only elapsed time; losing unsaved progress costs the progress. Weigh them
  accordingly, and weigh them *before* launching, not after a convenient reason not to
  restart presents itself. **Confirmed instance (2026-09-14 to 2026-09-15, Cam's own):** the
  checkpoint-durability bug (`Path(...).resolve()` silently bypassing `Volume.commit()`'s
  write-tracking, docs/THE-ECOLOGY-NEVER-RAN.md instance 5) was diagnosed and fixed
  (`11f0bb2`) while a run built on the *pre-fix* pinned SHA was already in flight. Cam ruled
  "leave it, this run" to avoid discarding 20 updates of ramp-floor convenience and a
  hard-won post-cascade codebook state — weighing durability against convenience without
  weighing what durability actually gates: whether the run's output exists at all. The run
  trained for 7.5 hours (44 PPO updates, ppo 2541→2584, including the entire cascade-defusal
  sequence the fix had just been proven against) and was lost in full when a second, unrelated
  tripwire halt fired — every checkpoint saved during that window, including the emergency
  save at the halt itself, silently failed to commit, for exactly the bug already diagnosed
  and already fixed in the repo. A fix that exists in the repo but not in the pinned SHA of
  the running container is not a fix. Once durability is confirmed broken, the only correct
  ruling is: stop, relaunch on the fix, resume the small loss now, before it becomes a large
  one. Verify durability *externally* before trusting further GPU-hours to any run — see the
  `DURABILITY-GATE` check in `scripts/modal_app.py` (`volume.listdir()` after the first
  commit, not the container's own print, not its own FUSE view), added directly in response
  to this instance.

## Part 3 — Fossils

**A fossil is state carried forward through grafts and resumes that was trained under
conditions that no longer exist.** It is not random, not obviously broken, and it is
consulted every step as though it were current. Three confirmed instances:

- `head_confidence_1/2` — trained once in the Phase 9.1 era, frozen through every resume, and
  read by the epistemic gate to make real behavioural choices against an architecture that had
  since changed underneath it.
- **Blue's crafting-avoidance policy** — ~1.4M steps spent in a world where 3 of 5 crafting
  materials never spawned (flint/clay/vine grids declared but never written), so ~95.5% of
  recipes were unsatisfiable from the instant they were drawn. Blue correctly learned crafting
  was futile, because under those conditions it was. Confirmed as a fossil, not a ceiling, by
  the small-scale verification of the spawn fix (2026-09-13): the very first post-resume
  update — before dispersal, while the population was still randomly scattered rather than
  arranged by the trained policy — produced 27 real craft successes. Every update after, as
  agents re-converged on trained (avoidant) behavior, produced zero.
- **Red's hunting-avoidance policy** — trained under a reward signal where `reward_red_catch`
  was read from config and never wired into the advantage computation (Rule 13 #11); the only
  live reward was two dense, blue-position-independent per-step terms. Red correctly learned
  hunting was pointless, because it was never rewarded for it. The same first post-resume
  update produced 308 catches from the scattered starting population; every update since,
  under the (now-connected but not-yet-effective) reward, produced 0-6.

When an ecological defect is repaired, **the policy trained under the defect is a fossil of
it.** Expect it to behave as though the old world still holds, and expect no gradient to
correct it if the new opportunity is too rare to be discovered by exploration. This is the
sparse-reward coordination trap, and the project has already solved it once: the Phase 16 CtD
gate made Big Green solo-catchable so agents learned the value of the noun before facing the
cooperative friction that required the verb.

**Overwritten versus bypassed.** A competence ramp that temporarily eases a mechanism will
produce successes during the easy period — that is expected and is not evidence the fossil is
fixed, since it was engineered to be possible. The test is what happens *after* the ratchet
back to full difficulty: track **attempt rate**, not just success rate, in the window right
after. A genuinely repaired policy keeps attempting the mechanism at roughly the same rate it
did during the ramp, because it has learned the mechanism has positive expected value, even
though success gets harder again. A policy that only exploited the temporary window —
bypassed, not overwritten — shows attempt rate collapse back toward the pre-ramp fossil
baseline within a few updates of the ratchet, with nothing else in the environment having
changed. That collapse, or its absence, is the tell.

## Part 4 — Change discipline

- **One experimental variable at a time.** Corrections to confirmed defects may be bundled;
  treatments may not. State which is which in the log before the run.
- **Rule 12 stands:** ecological parameters need Cam's sign-off. Restoring a mechanism the
  design already specifies is implementation, not tuning — but say so explicitly when you do
  it.
- **Never adjust an ecological parameter to make a gate open.** That destroys the only clean
  test available.
- **Kill the class, not the instance.** Four hand-written mask sites produced two bugs; one
  shared constant and one shared function produced none.

## Part 5 — Reporting

Lead with what changes the reader's mind, then the supporting detail. The most consequential
finding of the audit arrived as an aside in the middle of a task narration.

Separate **confirmed defects** from **suspicions** and label which is which. Report what you
checked and found clean, so coverage is visible rather than assumed. When you find one
instance of a defect class, ask how many siblings it has — they cluster, because they come
from the same habits.

**Confirmed instance (2026-09-14, Cam's correction):** the dead-code-reset suppression fix was
reported as having resolved the *original* codebook collapse. What was actually measured: 42
of 64 codes in one slot sat past the dead-streak threshold during this resume's pre-stability
window, fired in one shot at arming, and `codes_active` recovered immediately and held. That is
real, and it's MEASURED. What was reported alongside it — that this same mechanism *caused* the
original collapse — is CONJECTURED: the natural comparison is the previous (uncontrolled,
collapsed) run, but that run resumed from a different checkpoint (2763, not 2541) with a
materially different starting vocabulary. A suggestive pair, not a controlled one, and it was
written up without that confound named. Compounding it: the calibration ladder itself (logged
earlier the same day) shows `codes_active` declining gradually across ~220 ordinary training
updates between those two checkpoints — far too slow to be the resume-cascade mechanism — which
the causal claim, if accepted uncritically, would have implied was somehow explained by a
fast-onset resume artifact. Two distinct mechanisms are more likely than one: a fast resume
cascade (now defused) and a slow decay under absent communication pressure (still unaddressed,
the actual subject of the receiver-necessity thesis this whole audit sits inside). **Tag every
causal claim MEASURED or CONJECTURED explicitly, in the same sentence that makes it** — a strong
result on one question is not evidence on an adjacent one just because it arrived in the same
report.
