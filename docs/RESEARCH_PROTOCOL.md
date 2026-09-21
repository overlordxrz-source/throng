# THRONG Research Protocol

**Read this at the start of every session, before touching code or interpreting any result.**
It is the distilled output of the Sep 2026 audit, in which thirteen defects were found that
each had one property in common: something reported success while doing nothing, and a
measurement was taken anyway.

**Picking this up cold, on a new account or after a gap:** `docs/THE-ECOLOGY-NEVER-RAN.md`
has a "Status as of 2026-09-20" section at the top with the current blocker (both ecology
pressures were checked and one was structurally dead until that day's fix), what's fixed vs.
still open, and where the migration bundle (`~/throng_migration_2026-09-20/MIGRATION.md`) is.
Read that first if the question is "where do things stand," read this document for the
standing rules that produced those findings.

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
  **Correction (2026-09-15):** the mechanism named above (`Path(...).resolve()` bypassing
  `Volume.commit()`) does not survive a controlled test and has been retracted — see
  docs/THE-ECOLOGY-NEVER-RAN.md instance 5's retraction and instance 6. The actual mechanism
  losing both this run and a same-day relaunch attempt was Orbax's `max_to_keep` retention
  pruning by step number, silently deleting every checkpoint a *deliberately rolled-back*
  resume saved because a higher-numbered fossil from an abandoned lineage sat in the same
  directory. The principle this bullet states held regardless — durability was still the
  actual precondition being violated, and external verification was still what caught it — but
  the specific fix credited above was not the fix; read the two entries it points to for what
  was.
- **A safeguard's own arming condition needs the same scrutiny as the thing it guards.**
  Nesting a check inside an unrelated gate silently disables it without changing anything
  observable — no error, no different log line, nothing to notice short of asking whether the
  gate is actually correct for what's nested inside it. **Confirmed instance (2026-09-20,
  self-caught, answering Cam's direct question "is that tripwire armed"):** the PRESSURE
  tripwire (halts if `futile_uncoordinated` is still 0 after 10 updates in stage 1 — the check
  that the coordination-pressure ramp is actually applying pressure) had its own counter,
  `comms_stage1_updates_elapsed`, nested inside the *codes_active* baseline-comparison gate
  (`_active_c0 is not None and _active_u > _active_captured_at`) introduced by the same
  session's C0_stage0/C0_stage1 split. PRESSURE has nothing to do with codes_active or any C0
  baseline — it was disabled for as long as `comms_c0_stage1` hadn't captured, which on the
  live run was every update since the unfreeze. Didn't change that run's outcome
  (`uncoordinated_seen` had already latched `True`), but the halt condition this counter exists
  to detect could never have fired while gated this way — a safeguard that only works when
  another, unrelated safeguard happens to have already succeeded is not armed, whatever its own
  logic says. Fixed by moving the counter and its halt check outside the C0 gate entirely
  (`jax_sim/main_jax.py`). Found by direct interrogation ("what is X actually reading, is Y
  actually armed") after a report cited the metric, not by the tripwire firing or by code
  review before landing the refactor that caused it.
- **RETRACTED IN FULL, 2026-09-21 (Cam) — struck from the record, not softened: the
  informed-vs-uninformed material-holding comparison below measures nothing coherent and
  must not be cited.** Original claim, entered 2026-09-20 as "the most important number
  available": split blue's stage-1 corpus records by `can_see_recipe`, informed agents held a
  needed material 11.89% of the time (n=2,700) vs uninformed 16.19% (n=24,973), difference
  −4.30 points, 95% CI [−5.60, −2.99], z=−6.46. **Why it's void, not merely uncertain:**
  `docs/THE-ECOLOGY-NEVER-RAN.md` instance 10 (found 2026-09-21, while deleting the code that
  did it) established that `can_see_recipe` was globally re-rolled for the *entire living
  population* every ~1000 steps, for the entire measured history this corpus covers — not a
  one-time inherited trait as this entry originally assumed. That means "informed" and
  "uninformed" were never stable groups: an agent's `can_see_recipe` label at the moment of a
  craft attempt is a snapshot, but the material-holding *outcome* being measured was produced
  by that agent's gathering behavior over the preceding steps, under whatever `can_see_recipe`
  value it had *then* — which could differ from the label at craft-attempt time. Membership
  churned inside the very window the comparison treats as fixed. The comparison is not weakened
  or confounded, it is measuring an undefined quantity, and is struck rather than caveated.
  **What survives:** the hearth design's collision-rate evidence (`THE-ECOLOGY-NEVER-RAN.md`'s
  2026-09-21 entry, 2.00 observed vs 2.44–2.59 expected, z=−1.17) never depended on
  `can_see_recipe` and is untouched. **What does not survive, and must not be implied by
  anything written about this project going forward:** there is no evidence information
  *wasn't* helping. There is only evidence that coordination was at chance. Don't let the
  absence of the struck finding read as its opposite.
  **Also measured the same session, directly bearing on what "success" the pressure could
  even reward, and independent of the `can_see_recipe` defect above (this part is not
  retracted):** the literal informed-vs-uninformed *pair-success* rate (not the material-
  choice proxy above) cannot be measured from this corpus at all —
  `sample_frac=0.08` means P(both members of a real successful pair are independently sampled
  at the same step) = 0.08² = 0.64%; across 481,631 stage-1-window corpus records (14,592
  distinct steps) exactly 1 fully-sampled successful pair was found, against an expected ~13
  if even ~2,000 true successes existed in that window. This is an instrument-coverage
  ceiling, not evidence of anything about the ecology — record it so nobody re-attempts the
  same query expecting a different answer without first changing the corpus's sampling
  design (e.g. a targeted denser sample keyed to craft attempts specifically).
- **Read the mechanism before ruling on the policy it constrains.** `pop_split=small:0` and
  `catch_attempts≈0` together look like "red can't catch anything" — but `blue_caught≈1 per
  rollout, occasionally` was already visible in the same dashboard block, which a "predation is
  dead" read doesn't explain. **Confirmed instance (2026-09-20, Cam's own — "different fixes;
  I'm not ruling on red until I know which"):** `jax_sim/grid_jax.py`'s `apply_catches()` gates
  big-green catches on `coop_active = step >= coop_threshold_step` (`coop_threshold_step:
  100_000`, `config.yaml`) — past that step (current env step ~1.33M, 13x past it, for the
  entire observable history of this checkpoint lineage), catching a big-green requires **2 or
  more red agents simultaneously within `catch_radius=1` of the same target, both choosing
  STRIKE on the same step**; a lone striking red "mauls" instead of catching, and is
  *penalized* for it (`reward_big_green_solo_penalty: -1.0` vs `reward_big_green_success: 8.0`
  for a real coop catch). With the population at ~100% big-green (instance 7,
  `THE-ECOLOGY-NEVER-RAN.md`), every catch opportunity red has had is gated by this bar, and
  the observed ~1-per-rollout rate is red *occasionally clearing* a genuinely demanding
  2-agent coordination requirement across 250 largely-undirected agents — not a sign the
  target is unreachable. This is a red-policy coordination-rate question, separate from and
  downstream of the population-ratchet fix (instance 7) — worth revisiting once small-blue's
  mutation-restored presence (also instance 7) gives red an easier target to practice on
  again, before concluding anything further about red's own policy specifically.
- **A specific causal conjecture for a measured gap needs its own direct test — a plausible
  mechanism is not evidence until it's checked against the data it's supposed to explain.**
  The −4.3-point informed-vs-uninformed material-holding gap above (Cam's own conjecture,
  2026-09-20): informed agents search specifically for the recipe's needed material in a world
  where materials are zoned (`jax_sim/grid_jax.py`'s `material_zone_masks` — wood/stone split
  on x, flint/clay/vine split on y, every cell yields exactly 2 of 5 materials), so they're
  more often empty-handed because the zone they're standing in and the material their recipe
  demands don't line up. Stated prior: "around 40%."
  **Tested directly, as specified:** for each informed agent at each craft attempt, whether the
  demanded material is present anywhere in that agent's zone. **Measured: 58.07%** of informed
  craft attempts occur in a zone that does contain the demanded material — well above the
  stated 40% prior. **The specific causal mechanism is not supported by the cross-tab:**
  splitting empty-handed rate by zone-availability, informed agents are *more* often
  empty-handed when their zone *does* contain the needed material, not less — the reverse of
  what the conjecture predicts. The zoning/recipe mismatch is real (58.07% means a substantial
  minority of informed attempts happen in a zone that structurally cannot supply the recipe),
  but it is not, by itself, the explanation for why informed agents are worse than uninformed
  at holding a needed material. Reported as measured, not reconciled — the zone-availability
  number and the causal cross-tab are both facts about this corpus; a full explanation of the
  original gap remains open.
  **Ruling on what to do about the mismatch (Cam, Rule 12, 2026-09-20):** favor zone-local
  recipes (constrain what a recipe can demand to materials actually reachable from the zone a
  recipe-holder starts in or is assigned) over removing zoning entirely — this preserves the
  receiver-necessity structure (knowledge should still need to travel) rather than flattening
  the world to make the mismatch disappear. **Not yet implemented** — explicit instruction:
  "Don't implement yet. Give me the zone-availability number first; it decides the magnitude
  and it may change which option is right." The number above is that input; the fix itself is
  still pending review against it.
  **Relaunch gate:** "Nothing relaunches until the zone-availability number is in and the
  recipe fix lands. Running Gate C in a world that punishes using information would produce a
  null that means nothing." The number is in; the fix has not landed; no relaunch.
  **Superseded, 2026-09-21:** Cam's own empty-handed mechanism above was falsified cleanly by
  the cross-tab. The zone-local-recipe fix is withdrawn; see the next entry.
  **Further struck, same day:** the gap this entry was built to explain is itself retracted
  (see the entry above — `can_see_recipe` churned globally throughout the measured history, so
  no informed/uninformed behavioral comparison from that corpus is coherent). This entry's
  causal cross-tab (empty-handed rate split by zone-availability) inherits the identical
  defect — it relates a snapshot `can_see_recipe` label to a behavioral outcome (holding
  material) produced by earlier gathering under a possibly-different label — and is void for
  the same reason. What is NOT affected: the raw geometric fact that materials are zoned
  2-of-5 per cell (`material_zone_masks`, read directly from code, no corpus measurement
  involved) stands untouched; only the *behavioral* zone-availability percentage and its
  cross-tab are struck.
- **The zone-local-recipe fix is withdrawn; the mechanism itself was replaced (Hearths), and
  three gates were pre-registered, with fixed thresholds, before implementing anything.**
  Independently re-deriving the collision arithmetic (not taking Cam's hand estimate on faith —
  see `docs/THE-ECOLOGY-NEVER-RAN.md`'s 2026-09-21 collision-rate entry for the full method and
  numbers) confirmed observed crafting success is statistically indistinguishable from pure
  accidental co-location given agents' own measured rates of holding the needed material and
  choosing to craft. No coordination beyond chance ever occurred, and zone-local recipes would
  not have moved a co-location rate that low — the mechanism itself, not its material
  availability, was the problem. Ruling: pair-adjacency crafting is replaced entirely by
  **Hearths** (`jax_sim/grid_jax.py`'s `resolve_hearth_deposits`, `jax_sim/ctd_ramp.py`'s
  `HEARTH_RAMP_STAGE_N`) — four fixed locations, CRAFT-to-deposit with decay instead of
  same-step adjacency, curriculum N ramping 1→2→3 on the existing CtD machinery, hearth
  positions visible to everyone and hearth needs gated by `can_see_recipe`. Full mechanism,
  reward sizing, and implementation status in `THE-ECOLOGY-NEVER-RAN.md`'s "Status as of
  2026-09-21" section and instance 9.
  **Three gates, pre-registered verbatim (Cam, 2026-09-21) before implementation, thresholds
  fixed, not evaluated yet — nothing has run:**
  - *Gate 0*: "Within 50 updates at N=1, per-capita deposit rate must rise measurably above its
    value in the first 5 updates. If solo deposits don't increase when a single agent can earn
    reward alone, agents cannot learn one-body navigation and nothing downstream matters."
  - *Gate 1*: "At N=2, completion rate must exceed the random-collision baseline computed
    exactly the way we just computed it — measured holding rate, measured deposit rate
    conditional on holding, real geometry. We now have that method and it worked; reuse it
    verbatim."
  - *Gate 2*: "Measure whether uninformed agents deposit the correct material at a rate above
    chance (1/5). If they do, information reached them... just a counter." Two registered
    controls against the spatial-following confound: accuracy conditional on whether an
    informed agent was nearby recently, and a channel-ablation run (zero the wire, see if
    accuracy falls).
  **Relaunch gate, verbatim: "I want Gate 0 checked on the first fifty updates and nothing
  assumed past it."** Instrumentation for all three gates is landed and the mechanism is
  tested at production shapes and end-to-end under real JIT compilation (see
  `THE-ECOLOGY-NEVER-RAN.md`), but none of the three has been evaluated against a real run —
  none has happened. No relaunch.

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
