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
  none has happened.
  **Update, 2026-09-21 (later): three launch-blocker items were required before relaunch and
  are now verified** — the own_state_dim 22→29 checkpoint restore pads rather than drops or
  reinitializes the affected tensors (bit-identical output verified against a real pre-hearth
  checkpoint, `tests/test_hearth_checkpoint_pad.py`), Stage A is confirmed N=1 (not N=2), and
  comms freeze/unfreeze plus the Adam-spring defusal survived the `HEARTH_RAMP_STAGE_N` rename
  intact — see `THE-ECOLOGY-NEVER-RAN.md`'s 2026-09-21 (later) status entry for the full
  verification. **Launch is authorized**, Gate 0 armed from update one, nothing past it assumed.
  **Gate 0 verdict (2026-09-22, decided offline from the corpus already paid for — zero GPU):
  FAILED, unconfounded.** Single-tile run, ppo 2542–2591 (50 updates — exactly the Gate 0
  window), corpus scoped to one `launch_id` (`2a5d48f7405f`, no contamination). Mean distance
  from each living blue to its nearest hearth: flat at the random-walk expectation the whole
  run — 21.20 overall vs an exact expectation of 21.336 (computed over every cell, not
  sampled); first-5-update mean 21.30, last-5 21.37; slope +0.008/update. n≈3,600 agent-steps
  per update, so this is a measurement, not a guess: they never oriented on hearths at all.
  Attempts vs chance, computed from measured inputs (mean living pop 189.5, P(CRAFT)=7.24%,
  P(on a single tile)=4/16384): expected 1.71/rollout, 85.7 over 50 updates; observed 21
  (0.42/rollout), z=−6.99 — well *below* chance, not near it. (Cam's hand estimate was
  2.4 expected vs 0.86 observed; corrected by the real P(CRAFT) and population, same
  direction, larger gap.) Starvation confound checked and ruled out: living blue ranged
  148–200 (mean 189.5), at or below the 150 floor on 1 update of 50, never 2 in a row; energy
  0.410–0.546, never below 0.35. The registered starvation halt never fired — the run stopped on
  the PRESSURE tripwire (zero deposits for 11 straight stage-1 updates), not on starvation. A
  stressed but functioning population that didn't learn — not a dying one. **Decision rule
  applied as pre-registered: distance flat → widening is justified; relaunch with the 7×7
  hearth (`hearth_radius: 3`).**
- **A streak counter, or a baseline, is only meaningful within the regime it accumulated in.**
  Same class of error as capturing C₀ across a regime change, one level up. **Confirmed
  instance (2026-09-22, Cam's diagnosis):** the PRESSURE counter reached 11 under single-tile
  hearths, the checkpoint carried it into a relaunch with 49× the hearth area, and it re-fired
  on the very first update — before the new geometry could show anything either way. The run
  measured nothing, and cost a launch. **Rule:** training_state carries an *ecology
  fingerprint* — a fixed-order vector of the config values that define the world agents live in
  (`jax_sim/main_jax.py`'s `ecology_fingerprint_vector`: hearth radius, curriculum N schedule,
  hearth rewards and decay, catch radius, coop threshold, catch rewards). On resume, any
  difference — or a checkpoint that predates the fingerprint — resets every comms-freeze
  tripwire streak and every C₀ capture, and prints each changed field by name
  (`[ECOLOGY-FINGERPRINT] hearth_radius: 0.0 -> 3.0`). Keys that only describe how fast we get
  there (learning rate, batch size) are deliberately excluded, so tuning doesn't reset the
  instruments. **The hearth ramp restarts at stage 0 as well** (decided 2026-09-26). An earlier
  draft kept the stage on the grounds that it's "what the population learned." For checkpoint
  2591 that was false: it sat at stage 1 only because the stage-0 hard escape fired at 0/188
  deposits (`train.log`, ppo 2581). Resuming the 7×7 hearth there would have tested it under
  N=2 with no solo bootstrap, confounding geometry with coordination. A stage reached under
  the old world is not evidence under the new one. A population that genuinely learned
  re-clears stage 0 in a few updates (floor + 3-update bar), so the reset is cheap when
  unneeded. Comms re-freeze follows from the stage; the red ramp is untouched. Proven end to end through the
  real Orbax save/restore (fresh → radius changed → reset and named; same world → restored
  as-is), and pinned in `tests/test_ecology_fingerprint.py` — including the float32
  round-trip case, caught in review before landing: training_state stores the vector at
  float32, so comparing against a fresh float64 vector would have flagged 0.3 → 0.30000001 as a
  world change on *every* resume, making the safeguard cry wolf permanently.
- **A gate that cries wolf will eventually be ignored — and the fix for a false positive must
  not create a false negative.** **Confirmed instance (2026-09-22):** the widened-hearth
  relaunch resumed at 2591 and halted on its first update; the emergency save re-targeted 2591
  (it saved at `ui`, not `ui + 1` like the periodic save), Orbax declined — verified on the
  installed version: `save()` to an existing step returns `False`, no exception, mtime
  unchanged — and the durability gate called "nothing new appeared" a FATAL. On the volume:
  `checkpoints_hearth/2591` mtime unchanged at its original 02:49 save, no orphaned
  `*.orbax-checkpoint-tmp-*`, `_METADATA` downloaded and parses. Nothing was lost. Fixed at
  three layers: (1) the emergency save targets `ui + 1`; (2) both save sites check `save()`'s
  own return value instead of printing "saved" unconditionally, and pass it to the gate;
  (3) the gate (extracted to `scripts/durability_gate.py`, so every branch is tested against a
  fake volume rather than first exercised on a paid run) treats "no write attempted" as neither
  FATAL nor verified. **Two traps found in review, both now pinned in
  `tests/test_durability_gate.py`:** a first draft softened "no new entry and no tmp dir" to a
  warning — which would have waved through both real historical losses (the 7.5-hour run and
  instance 6), neither of which left a tmp dir. And writing the tmp test exposed a hole that
  predated today: the gate counted an orphaned tmp dir as a "new entry," so a write that
  started and never finished would have been reported as durability VERIFIED. Both closed.
- **Gate 0 on the widened hearth (2026-09-27/28): failed again, unconfounded by the escape
  landing inside the window (Cam's registration error, corrected) — and this time the failure
  mode itself changed.** 50-update window, ppo 2591–2641, hearth_radius=3. Apparatus check
  passed decisively (attempts far above the ≥8.6/rollout bar — 3,063 attempts over 38 stage-0
  updates alone). Distance-to-hearth stayed flat within stage 0 (21.438 → 21.466 first-5 vs
  last-5, well inside the update-to-update std) — the literal full-window comparison is
  confounded by the stage-0 hard escape landing at ppo 2631 (40 updates after the fingerprint
  reset, same ceiling as the single-tile and 7×7 runs before it — three escapes at three
  geometries), putting stage-0 data in "first 5" and stage-1 data in "last 5"; the stage-0-only
  reading is the one that stands. **Diagnosis (Cam): the hearth task is a conjunction — on a
  hearth AND holding the needed material — and only the first conjunct had been measured or
  fixed.** Three offline measurements, from the corpus already paid for, before touching code:
  (a) **per-material satisfiability — clean.** All five materials observed held in the
  thousands (wood 16,730 / stone 18,761 / flint 8,495 / clay 6,869 / vine 8,588 agent-step
  observations); spawn code confirmed non-degenerate zones and non-zero regen for all five — the
  historical `flint_grid`/`clay_grid`/`vine_grid` zero-init bug (instance 4)
  has not recurred. `hearth_deposited=True` appears zero times in the *entire* corpus across
  every launch ever recorded, including the single-tile run known to have real deposits — traced
  to expected sampling sparsity (corpus_every_n_steps=4 × ~0.83% true conversion → expected
  ≈0.6 sampled deposits in-window), not a broken field; deposits-by-material is unmeasurable at
  this corpus density, a genuine instrument gap, not a red flag. (b) **corrected conversion —
  decisive.** Old instrument's denominator was CRAFT-presses (agent-controlled, spammable — the
  same flaw as the retired success/attempts bar). Rebuilt denominator = actual on-hearth-tile
  agent-steps (1,071, reconstructed positionally at radius 3 from the four fixed hearths):
  holding nothing 33.89%, holding *some* material 66.11%, holding the *needed* material 0.65%.
  Arriving with something is solved; arriving with the right thing is 15× rarer than that.
  (c) **informed vs uninformed P(holding needed | on hearth) — inconclusive, honestly reported
  as such.** Informed 4/879 (0.46%), uninformed 3/192 (1.56%) — opposite Cam's hypothesized
  direction, but 7 total successes is nowhere near enough to distinguish signal from noise
  either way; not rounded to a conclusion in either direction.
- **The hearth curriculum's stage-A bootstrap is now split into two rungs, isolating the two
  conjuncts instead of asking one stage to solve both at once (Cam's design, pre-authorized on
  (a) coming back clean).** `jax_sim/ctd_ramp.py`: `HEARTH_RAMP_STAGE_N = (1, 1, 2, 3)` and a new
  parallel `HEARTH_RAMP_ACCEPT_ANY = (True, False, False, False)`. **A0** (N=1, accept_any=True):
  any material completes it — pure navigation plus gathering, no recipe knowledge required or
  rewarded; registered prediction: since ~66% of on-hearth agent-steps already hold something,
  A0 should exit on merit within ~15 updates, and if it instead times out at the 40-update hard
  escape, that is a finding about the learner (reward scale, credit assignment, hearth-position
  observability), not the ecology. **A1** (N=1, accept_any=False): identical to the old stage
  0 — learn to carry the *right* thing; the first rung in this project's history where
  `can_see_recipe` is worth something and not everyone has it, so Gate 2 becomes a meaningful
  measurement here and nowhere earlier. **B**/**B+** (N=2/N=3): unchanged from the old stages 1
  and 2. Exit bar lowered 10%→5% per-capita (still sustained 3 updates) for every rung; the
  40-update hard escape is now generic over every rung, not hardcoded to the first.
- **Three stage-index bugs caught in review before landing, from inserting a stage in the
  middle of an existing schedule instead of appending one.** The comms-freeze tripwire's
  baseline-selection logic, the C0_stage1 search trigger, and the PRESSURE tripwire all
  hardcoded `craft_ramp_stage_outer == 0` to mean "the frozen/solo regime" and `== 1` to mean
  "the first live-coordination regime" — true when N=1 was exactly one stage, silently wrong the
  moment A0 is inserted before it (A1, now index 1, is still N=1/frozen, not the coordination
  stage the old index 1 meant). Fixed by deriving both from `HEARTH_RAMP_STAGE_N` itself: the
  frozen regime is wherever `N==1` (spans A0+A1), and the coordination regime is
  `HEARTH_COORD_STAGE_IDX` — a module-level constant (first index where N>1), not a per-function
  local, after an earlier draft of the fix defined it inside the wrong one of two functions and
  crashed on first execution. **A fourth, independent bug in the same review pass:** the ecology
  fingerprint's new `accept_any` and `craft_ramp_success_bar` fields were first *inserted* into
  the middle of `ECOLOGY_FINGERPRINT_LABELS`, between the existing N-schedule and reward fields.
  `ecology_fingerprint_diff`'s `zip()` over `(labels, old, new)` silently truncates to the
  shorter array on a length mismatch — a checkpoint saved under the pre-split 16-field schema
  (every real checkpoint on the volume, including the one this redesign resumes from) would have
  had its post-insertion fields compared against the wrong labels entirely, not merely skipped.
  Fixed two ways: new fields are now appended, never inserted, so a shorter old vector's
  pre-existing fields stay correctly aligned; and `ecology_fingerprint_diff` now detects a bare
  length mismatch explicitly and reports it by name instead of silently truncating, so the next
  schema change doesn't have to rely on getting the append-only convention right by memory
  alone. All four pinned in `tests/test_ecology_fingerprint.py` and `tests/test_ctd_ramp.py`.
- **Three offline checks on A0/A1 (Cam, zero GPU), one confirmed apparatus bug, one confound
  reconciled, one decisive vision check.** (1) A0's 10-update clearance: pop_frac rose 1.50% →
  9.00% → settling 5–7%, but mean distance-to-hearth stayed flat (21.00→21.22, no decline) the
  entire window — what changed was CRAFT-issuance while already on a hearth, not navigation.
  (2) End-to-end match rate between (inventory==logged need) and the simulator's own
  `hearth_deposited`, pooled across every launch: 99.17% overall, 99.34% informed — short of the
  ~100% bar. Root cause found and confirmed by hand on both mismatches: `main_jax.py` recorded
  `grid.hearth_need` to the corpus **after** that step's own reassignment (the line that applies a
  completion's reroll), so on the rare step a hearth completes, the corpus logged what it wants
  *next*, not what it just accepted. Fixed by capturing the pre-resolution value
  (`_hearth_need_pre_resolution`) before the reassignment and logging that instead. Confirmed to
  **not** implicate the informed<uninformed finding: `resolve_hearth_deposits` itself reads
  `grid.hearth_need` before this line touches it (the actual deposit decision was never affected),
  and Gate 2's dashboard numbers come from `hearth_deposited`/`can_see_recipe` directly, never
  from this field. (3) Need-rotation vs. delivery time: initial pooled measurement (n=5, median
  40 steps) was corrected on review — per-hearth breakdown showed all 5 detected rotations came
  from a single hearth (slot 3), with zero detected at the other three; Cam's suspicion of
  clustering was right. But the alternative derived from a uniform-across-4-hearths assumption
  (~1,707 steps) isn't supported by the data either — the true distribution is far from uniform,
  and corpus sparsity (only ~25% of steps logged) leaves the other three hearths' true rate
  unmeasured either way. Left unresolved, non-blocking, flagged for a better-powered check later.
- **Vision check, decisive: `emb_own`'s hearth-dimension rows are alive, not dead.** Real
  checkpoint (step 2670), restored through the production grafting path. `emb_own` kernel
  (29, 256): hearth rows (13–24, position + need) median L2 norm 1.159 vs. 1.679 for non-hearth
  rows — 69% ratio, nowhere near the 5%-of-median dead threshold. `gwt_comms_1` (2738, 256): zero
  rows below 5% of median across the entire kernel. Per Cam's own pre-registered branching, this
  selects the shaping lever, not re-initialization.
- **Potential-based hearth-approach shaping added** (`jax_sim/ctd_ramp.py`'s
  `hearth_shaping_term`, mirroring `red_shaping_term` exactly): dense gradient toward the nearest
  hearth whose *current* need matches the material the agent holds, `F_t = beta*(gamma*Phi(s')
  -Phi(s))`, `Phi(s)=-d(s)/D_max`, zeroed on the step a deposit happens (need rerolls on
  completion, so the nearest matching hearth can teleport to a farther one the instant a deposit
  lands — same reason red's is zeroed on catches). An agent holding nothing, or holding a
  material no hearth currently wants, gets `Phi=0` (no artificial reward for an undefined target).
  Config: `ctd_competence_ramp.hearth_shaping_enabled` (now `true`), `hearth_shaping_beta` (2.5,
  matching red's own value as a starting point, not a derived number). Both fields appended to
  the ecology fingerprint, so enabling shaping forces the same stage-0/A0 reset a tripwire streak
  reached without shaping isn't evidence of anything once shaping exists. Not gated by curriculum
  stage — distance-to-hearth has been flat in every stage measured so far (single-tile, 7×7
  stage 0, A0, A1), so shaping applies whenever hearths exist, not just during one stage.
  **Pre-registered for the next run:** within 30 updates of A0 starting, mean distance-to-hearth
  must fall ≥10% below its A0 update-1 value, exceeding update-to-update noise. If not, and this
  was the re-init branch, add shaping and continue without stopping (moot here — shaping is
  already the branch taken). If it still doesn't decline after shaping, stop and report: that
  would mean goal-directed navigation genuinely isn't learnable here, a different conversation
  from anything this log has covered.
- **Two scope bugs caught by running the smoke test, not by inspection, both the same class as
  the `HEARTH_COORD_STAGE_IDX` bug two entries up:** a Python-level helper (`hearth_shaping`'s
  enable flag for the dashboard print) was first defined inside `make_sim_step`, invisible from
  `_run_simulation_impl`'s own outer training loop where the print lives — crashed with a plain
  `NameError` on first execution. Fixed by reading the same config value again in the outer
  scope specifically for the print, leaving `make_sim_step`'s own copy to gate the traced
  computation. A reminder that this file's two-function split (JIT-traced step builder vs. outer
  Python loop) doesn't share locals, and that fact doesn't announce itself until something tries
  to cross it.
- **Shaping run's pre-registered check: failed the threshold, but not flat — the first real
  directional signal this investigation has measured.** A0 update-1 (ppo 2671) mean
  distance-to-hearth 21.600; last-5 of the 30-update window (ppo 2696–2700) mean 20.868 — **-3.39%**,
  short of the required -10%. But the decline itself (0.732) **does** exceed the update-to-update
  std (0.385) — a real decline, not noise, unlike single-tile, 7×7 stage 0, and A0's own first
  clearance, all of which were flat within noise. A0 itself re-cleared via `bar met` at ppo 2681 (11
  updates, inheriting a streak of 2 from before the Modal pause); A1 then showed the familiar
  pattern sharpened further — attempts jumped from ~10–20/update in A0 to 84–377/update in A1,
  while deposits collapsed to 0–2/update, the conjunction's second half still the bottleneck.
  Population dipped to 151 at ppo 2696 (never crossed the 150 floor); energy never below 0.419. No
  halt trigger fired. **Per the pre-registered rule, this is a stop-and-report point** (shaping was
  the branch already taken, so "add shaping and continue" doesn't apply): run stopped, reported,
  awaiting Cam's read rather than continuing on an unmet threshold. Last verified checkpoint: 2700
  (Lightning AI Studio `throng-train`, `/teamspace/studios/this_studio/throng-runs/`).

- **Cam's follow-up on the shaping run (2026-09-29): the -3.39% "first real directional signal"
  above is retracted as evidence; the offline checks (zero GPU) say the shaping was too weak to
  test anything.** All numbers from the Lightning corpus + Modal corpus tail (updates 2669-2700,
  ~3,700 uniformly-sampled alive agents/update; `hearth_dist` reproduced from the corpus tracks the
  dashboard's own `mean_dist_to_hearth` at r=0.96, mean difference -0.02, so the instrument is
  sound). Scripts were scratchpad-only; the definitions are in the bullets.
  1. **Baseline was a single draw (third time this session).** 21.600 (ppo 2671) is the highest
     value in the window; window mean 21.07, update-to-update sd 0.39, range 20.33-21.98. First-5
     mean (2671-75) 21.386 vs last-5 (2696-2700) 20.868: -2.4%, |diff|/SE = 3.6 on update-level SE
     -- but that is the all-agent metric (includes ~30% empty-handed agents and the A0->A1 stage
     switch inside the window), it sits inside the earlier "flat" range (20.47-21.44), and it does
     **not** appear where the shaping acts (below). Not evidence of learned approach.
  2. **Aligned instrument (holders only; nearest hearth wanting the carried material -- exactly
     the shaping target).** No approach in either stage. Within A0 (2671-75 vs 2677-81): 39.5 ->
     38.3 (-3.0%, 0.5 SE); against a position-conditioned null (same positions, random matching
     hearth): excess +0.85 -> -0.16, 0.6 SE. Within A1 (2682-86 vs 2696-2700): 37.0 -> 49.9
     (+34.8%) -- *farther*, not closer -- excess vs the null +0.27 -> +8.28. **That A1 rise is not
     interpretable as avoidance:** hearth needs sat on essentially one configuration from 2684 on
     (one hearth changing at a time: [4,4,4,3] -> [4,4,3,3] -> [4,4,3,0] -> [4,4,2,0]), so ~17
     "updates" are one geometry; the update-level SEs there badly overstate independence. Stated
     narrowly: no evidence of approach anywhere, and the only stage with enough need-rotation to
     average over geometry (A0) is flat. Holders-only nearest-ANY (A0's rule) is also flat:
     21.31 -> 21.09 (-1.0%, 0.7 SE). Stage-aware literal metric (A0-any vs A1-matching) is
     meaningless across the switch (21.3 vs 49.9 is definition change, +134%) -- use the
     stage-invariant matching-hearth metric for cross-stage comparison.
  3. **Shaping strength -- it was effectively off.** Real `hearth_shaping_term` driven with each
     sampled holder's actual position/holding/needs and the measured action mix (A0 move-fraction
     0.43, A1 0.41), 20 draws/record: at beta=2.5 mean|F| = 0.0030 (A0) / 0.0034 (A1) over all
     agent-steps, std(F) ~0.0105, only 32-37% of agent-steps have a defined target. Blue reward
     over the last 30 updates: mean 0.024, std 0.096 (rms 0.102). So mean|F| is ~3-14% of the
     |mean reward| lower bound and a few percent of any realistic mean|r| (mean|r| itself is not
     logged -- the new dashboard line reports it); the logged signed `shaping_mean`
     (0.0007-0.0011) is 2-4% of reward and is fully explained by potential-based shaping's
     (1-gamma) drift for holders (MC no-navigation mean: 0.0005 A0 / 0.0008 A1, matching the log)
     -- i.e. zero net approach, consistent with item 2. **Under Cam's pre-authorized rule (<10%):
     shaping was configured but never really on.**
  4. **Row norms, on the record (were in this log but not in the chat report).** Branch decision,
     ckpt 2670: `emb_own` (29,256) hearth rows 13-24 median L2 1.159 vs 1.679 non-hearth (69%);
     `gwt_comms_1` (2738,256): 0 rows under 5% of median. Re-measured ckpt 2700 (post-shaping,
     raw restore): `emb_own` 1.225 vs 1.721 (71%), `gwt_comms_1` blue 0 rows <5% (min 0.088) --
     alive before and after, barely moved. (Red's own `gwt_comms_1` has 95 near-zero rows; red's
     comms are decoupled, unrelated to hearths.)
  5. **Discrepancy flagged, NOT changed:** `hearth_shaping_term` targets the nearest hearth whose
     need matches the held material in *every* stage, but A0's deposit rule accepts ANY hearth
     (`accept_any_material`). In A0 shaping therefore pulls toward a matching hearth even when a
     nearer one would score, and gives holders with no matching hearth (~40% of A0 holders) no
     signal at all. Harmless in A1 (rule and shaping agree); in A0 it weakens shaping. Left as is:
     changing it would be a second variable alongside beta.
  **Decision (pre-authorized): scale beta.** 2.5 -> 10.0 (mean|F| ~0.0136, i.e. 17-34% of any
  plausible mean|r|, bracketing the 25% target), plus log-only instrumentation:
  `ShapingStrength` (mean|F|, mean|r|, ratio, std ratio -- same (T,N) arrays PPO averages) and
  `AlignedDist` (holders->matching hearth, holders->any hearth) on every dashboard. Retune beta
  from the first dashboard's measured ratio, not from the estimate above. `hearth_shaping_beta`
  is in the ecology fingerprint, so this resumes from 2700 with the designed stage-0/A0 reset.
  **Pre-registered for the relaunch:** aligned metric (holders -> matching hearth; stage-invariant),
  first-5 mean vs last-5 mean over 30 updates, decline >= 10% AND exceeding twice the SE of the
  difference. If shaping is meaningful (first-dashboard ratio in range) and the aligned metric is
  flat: stop and report -- the "can this learner do goal-directed navigation" conversation.
  Caveat carried forward: SEs from update-level means overstate independence when needs are
  static; prefer to read the A0 (rotating-needs) stretch and the position-conditioned excess.
  **Measured on the first relaunch dashboards (beta=10, resumed from 2700, stage-0/A0 reset as
  designed):** update 1 after resume is not representative (agents restart empty-handed: only 6%
  of alive-steps had a holder with a matching hearth, vs 26-34% by update 3-5; the same dip is
  visible in the corpus at 2671 and 2674). Steady state (ppo 2704-2706): mean|F| 0.0068-0.0081,
  mean|r| 0.069-0.075 (first direct measurement of mean|r|: ~0.07), **ratio 9.9-10.8%**,
  std(F)/std(r) 33-36%. The offline estimate (mean|F| ~0.0136) was ~2x high -- measured beats
  estimated, per Cam's "verify on the dashboard". Retuned beta 10 -> 24 (25/10.2 x 10) to reach
  the ~25% target, resetting the pre-registered 30-update clock to the second relaunch. **Flagged
  for Cam:** the two yardsticks disagree by ~3x -- at beta=24 std(F)/std(r) will be roughly 80%,
  i.e. shaping would dominate advantage variance even though its *mean magnitude* is 25% of
  reward (most steps are unshaped: only ~30% of alive-steps have a target, ~40% of those move).
  If 25% of mean|r| was meant as "PPO can feel it", beta ~10 (ratio ~10%, std ratio ~35%) may
  already qualify; his call.
  **beta=24 relaunch (resumed from 2706, `hearth_shaping_beta: 10.0 -> 24.0` fingerprint reset
  fired as designed): steady-state dashboards (ppo 2709-2711): mean|F|/mean|r| = 19.3 / 18.3 /
  15.7%, std(F)/std(r) = 54-60%, mean|r| 0.085-0.096 (rises with beta because r *includes* the
  shaping, so the ratio saturates below the naive linear scaling).** Not retuned a third time:
  both yardsticks are now well above the 10% line and the choice between them is Cam's.
  **Pre-registered window, fixed before any outcome data: 30 updates starting at the first update
  after this resume (ppo 2707-2736); first-5 = 2707-2711, last-5 = 2732-2736; metric =
  `AlignedDist holders->matching_hearth` (stage-invariant), decline >= 10% and > 2x SE of the
  difference of the two 5-update means.** Known bias, stated in advance: the first 1-2 updates
  after every resume are empty-handed (holder coverage 3-12% of alive-steps vs 25-45% steady
  state), so the early mean is noisier -- report the literal first-5 AND a version dropping
  2707-08.

- **Registration (Cam, timestamped 2026-09-29T13:55Z, run at ppo 2712, before the window closes
  at 2736) -- supersedes the window stated in the entry above.**
  * **Primary:** first-5 = ppo 2709-2713, last-5 = ppo 2732-2736. **Secondary:** the literal
    2707-2711 first-5 (same last-5). Metric: `AlignedDist holders->matching_hearth`. Pass = decline
    >= 10% AND > 2x SE of the difference. **SE is computed from update-level means (five against
    five), never from individual records** -- records within an update are correlated and needs sit
    in one geometry for long stretches, so a per-record SE badly overstates certainty.
  * **Why excluding 2707-08 is a registration and not a moved bar:** the reason is holder coverage
    (3-12% of alive-steps at 2707-08 vs 25-45% steady state), which is independent of the distance
    outcome, decided before any last-5 outcome data exists. Disclosure: the per-update
    `AlignedDist` values for 2707-2711 had already been printed to the log/report when this was
    registered; the last-5 window (2732-2736) is entirely unseen.
  * **Shaping-strength yardstick, settled by Cam: variance.** PPO normalizes advantages, so mean
    shifts are absorbed by the value baseline and the gradient feels variance. Shaping is
    ~55-60% of per-step reward variance at beta=24 (std(F)/std(r)) -- very much felt. The
    mean/variance gap is structural for potential-based shaping (signed +/-beta steps cancel in
    mean, not in variance); the "25% of mean|r|" target used the wrong yardstick. Both readings
    clear the 10% line. **beta=24 stands; nothing changes until the window closes.**
  * **A0 mismatch (shaping toward matching hearth vs. A0's accept-any deposit): left as is, noted
    as a later refinement.** It doesn't invalidate the test: shaping and the aligned metric both
    use `need == held`, and the question is whether shaping toward the matching hearth produces
    approach to it, on which A0's deposit rule has no bearing.
  * **Stage record:** the beta change (10 -> 24) reset the fingerprint as designed: at ppo 2706
    the ramp went "stage 0 -> stage 0, starting now at ppo=2706" (the first beta change, 2.5 -> 10
    at ppo 2700, had taken it A1 -> A0). Stage at 2707 = A0; at ppo 2712, stage 0/3 [A0],
    streak 1/3, no transition inside the window so far. Any transition will be logged here.
    **Transition logged 2026-09-29: A0 -> A1 at ppo 2725 (step 1_395_200, "bar met")**, inside the
    registered window. A0 spanned 2707-2725 (19 updates; 9-19 deposits/update in the final stretch,
    population 165 at 2720 recovering to 197 at 2725, floor 150 never approached). 2726 onward is
    A1, so the first-5 (2709-2713) is A0 and the last-5 (2732-2736) is A1; the registered metric is
    stage-invariant by construction (matching-hearth distance for holders in both stages).
  * **For the record:** (1) blindness is falsified -- `emb_own` hearth rows alive at ~70% of the
    non-hearth median at both 2670 and 2700; (2) the beta=2.5 shaping was effectively zero, so
    **navigation had never actually been tested before this run**.
  * **Decision tree:** pass -> navigation is learnable with a real gradient; next question is
    whether informed agents approach the matching hearth more than uninformed ones (where the
    channel finally has a job). Fail with shaping this strong -> the genuine "can this learner do
    goal-directed navigation" conversation, earned. Report at 2736, at any transition, or on a trip.
  * **Ops:** Lightning Studio `auto_shutdown`/`auto_sleep` disabled 2026-09-29T13:55Z with the
    user's explicit permission (the earlier auto-mode block had been on unauthorized settings
    changes).

- **Window result (beta=24, ppo 2707-2735): registered check FAILED -- no approach to the matching
  hearth -- and the run crashed at 2736, one update before the window closed. Stop-and-report
  per the registered decision tree ("fail with shaping this strong").**
  * **Crash:** 2026-09-29 ~15:16Z during the red PPO update of ppo 2736,
    `XlaRuntimeError: INTERNAL: Failed to launch ptxas`, raised inside a small diagnostic
    reduction (`_head_grad_norm`, `jnp.sum(g**2)`) that had run every update before; disk was fine
    (355 GB free) and the Studio was found Stopped afterwards with `auto_sleep`/`auto_shutdown`
    both False. **Cause unconfirmed** -- looks host-level (process spawn failed on an
    already-compiled op), not code; not investigated further. Last committed checkpoint **2733**
    (2734-35 dashboards exist in the log but have no checkpoint; 2736 never printed). Artifacts
    saved locally: `~/throng-migration-bundle/` (ckpt 2700, nohup_2735.out, train.log, corpus).
  * **Registered comparison, last window 4 of 5 (2732-2735), update-level means, SE from
    five-vs-four update means:** primary first-5 (2709-13) 38.63 -> last-4 45.54, **+17.9%**
    (farther), SE 3.11, 2.2 SE the *wrong way*. Secondary literal first-5 (2707-11) 36.64 -> 45.54,
    +24.3%. **A pass is arithmetically impossible even without 2736:** the five-update last mean
    cannot fall below 4*45.54/5 = 36.4 while the pass line is 34.8 (distance cannot be negative).
    So the missing update does not change the verdict.
  * **Position-conditioned null (same positions, randomised matching hearth; corpus reproduces the
    dashboard: 38.75 / 45.47):** A0, 2709-2725, needs rotating through 17 configurations: mean
    excess **-0.56** (about zero; A0-late 2721-25 vs first-5: +0.46, 0.2 SE) -- no approach in the
    only stretch with enough need-rotation to average geometry. A1, 2726-2735: excess +3.9 (up to
    +6), but needs sat on two configurations ([4,4,3,4], then [4,4,1,4]) -- the single-geometry
    confound again, so "farther" is not interpretable as avoidance. Stated narrowly: **with shaping at
    ~55-75% of per-step reward variance (std(F)/std(r); mean|F|/mean|r| 20-30% steady state),
    holders showed no movement toward the hearth wanting what they carry.**
  * **Context, not evidence of learning:** A0 cleared its bar at 2725 (deposits 4-19/update),
    A1 then collapsed to 0-2 deposits/update with 3-83 attempts (wrong material), holder coverage
    dipped to 13-18% at 2726-31 then recovered to ~37%; population minimum 165 (2720, floor 150
    never approached).
  * **Status:** run stopped (crash), not relaunched -- the pre-registered branch says this is the
    genuine "can this learner do goal-directed navigation" conversation and needs Cam's read
    before any more compute. Studio stopped 2026-09-29 to avoid idle spend (files persist).

- **Cam's three offline checks after the failed window (2026-09-30, zero GPU): (a) shaping direction
  PASSES; (b) positive control shows almost no directional movement; (c) not run, gated off by
  (b) per his decision tree.** Hypothesis under test: a direction/position error in the shaping
  would produce exactly the observed pattern (A0 flat as the target rotates, A1 rising as a frozen
  target is pushed *away* from); the earlier Monte Carlo validated magnitude, not sign.
  * **(a) Direction test -- pass, and the test is non-vacuous.** `tests/test_hearth_shaping_direction.py`
    (6 tests, production shapes: grid 128, N=200, beta=24, gamma=0.999; positions from the REAL
    `apply_moves`, so the torus wrap comes from the environment, not my arithmetic): toward the
    matching hearth -> F>0, away -> F<0, across the x and y wrap boundaries, 1,500 random
    (position, action, material, needs) draws vs an independent numpy torus-Chebyshev, batch of 200
    == one-at-a-time, empty-handed and deposit steps -> 0. **Mutation-checked:** replacing the
    function with a sign-flipped, s'==s (Cam's exact hypothesis), or s/s'-swapped version makes 4
    of the 6 fail. **Real call site:** a recording wrapper around `hearth_shaping_term` inside the
    real training loop (`jax.debug.callback`, 160 real steps, grid 32 / 24 agents on CPU -- a
    production-size CPU rollout was not attempted; the code path is shape-agnostic and production
    shapes are covered by the unit tests): s != s' for 38.3% of agent-steps (so not the same
    position); pre-move position at t+1 equals post-move position at t for 97.5% of agent-steps
    (s really is pre-move; the rest are deaths/respawns); every F equals an independent recompute
    from the loop's own arguments; closer to the matching hearth -> F>0 16/16, farther -> F<0
    13/13; 48 torus-wrapping moves in the sample. Source check agrees: `_b_pos_before_move` is
    captured before `apply_moves`, `b_pop` is replaced with the post-move positions after it.
    **The shaping sign is correct; the 'F collapses to (gamma-1)Phi(s)' failure mode is ruled out.**
    So the failed window stands as run (`registration` above); no fix, no re-run triggered by (a).
  * **(b) Positive control -- blue barely moves directionally even relative to its predator.**
    Corpus 2673-2735 (221,885 records, launch 8711 duplicate excluded). Being caught costs blue
    -2 and its life (`b_penalty = -1*caught`, times |reward_blue_caught|=2 -- checked, sign right),
    so fleeing red is the natural directional behavior. Statistic: for records whose action is a
    move, m.u where u is the unit vector to the nearest red (bearing = atan2(axis0 offset, axis1
    offset), action deltas from `apply_moves`); +1 straight toward, -1 straight away; update-level
    SE. Baseline (red >20 away, unsensed): -0.003. Adjacent (d<=1): **-0.071** (SE 0.011; shift vs
    baseline -0.068, z=3.7; 54% of moves have an away component vs 48-49% baseline) -- a real but
    tiny flee. d=2 (edge of the 5x5 patch, the last distance red is visible at): +0.020 (no flee).
    d=3-20 ~0 (expected: blue is blind beyond the patch). Caveats stated: nearest-red only;
    possible one-step misalignment between action and the bearing's position snapshot (a real fleer
    would then appear at d=2 with negative m.u -- it doesn't); "toward food when energy is low"
    could not be run (corpus has local resource amount, not direction). **Read:** nothing
    approaching robust goal-directed movement is visible even for the strongest, longest-trained
    signal in the world; this is weak-to-moderate evidence, not proof, that the learner as built
    does not navigate directionally.
  * **(c) Not run:** gated on (a) passing AND (b) showing navigation; (b) did not. Checkpoint 2733
    is staged locally for it if Cam wants the hearth_rel-vs-red-position sensitivity comparison
    anyway (it would also help interpret (b)).
  * **Decision-tree position:** (a) passes, (b) no directional behavior -> the branch Cam described
    as "the learner can't do goal-directed movement in this architecture ... the real
    conversation", pending his concrete proposal. Nothing relaunched.
  * **Migration status:** the new Lightning account's key authenticates but is forbidden (403) on
    the teamspace (`overlordxrz-org/default-project`), so the Studio could not be created; the old
    account is out of balance ("insufficient balance", explains the stop after the ptxas crash).
    Bundle staged locally at `~/throng-migration-bundle/` (checkpoint 2700 + **2733**, corpora
    through 2735, logs). The Mac's disk is ~98% full (520 MB free).

- **RETRACTED HYPOTHESIS (Cam, 2026-09-30): "a direction/position error in the hearth shaping."**
  Withdrawn, cleanly falsified. Evidence: 6 direction tests at production shapes (toward->F>0,
  away->F<0, both torus wrap edges, 1,500 random draws vs an independent numpy torus-Chebyshev);
  mutation-checked (sign-flip, s'==s, swapped s/s' each fail 4 of 6); real-loop call-site capture
  (s != s' for 38.3% of agent-steps, pre-move position at t+1 == post-move at t for 97.5%, every F
  equals an independent recompute, 16/16 closer->F>0, 13/13 farther->F<0). The "F collapses to
  (gamma-1)Phi(s)" variant is ruled out by the same capture. The failed beta=24 window stands as run.
- **Cam's proposal: the Phase 11.3 epistemic gate executes imagined actions that are trained as
  if the reactive policy sampled them. Code checks (2026-09-30), both done before any spend.**
  * **Check 1 -- CONFIRMED.** `jax_sim/main_jax.py`: `b_actions = where(b_gate_imagine,
    b_a_imagined, b_actions_reactive)` -- the ENVIRONMENT executes the imagined action on gated
    steps. The stored PPO action is that executed `b_actions`; the stored log-prob is
    `log_softmax(b_action_logits)[b_actions]`, i.e. log pi_reactive of the executed action.
    `rl_jax.py`'s ratio is `exp(logpi_new(a_exec) - logpi_old(a_exec))`: no imagination mask, no
    gate flag, no behaviour-policy (mu/pi) correction anywhere in the update. So on gated steps
    (mean 77.3% of alive-steps over 2707-2735; range 72.3-80.7%) the gradient is fed actions from
    a behaviour policy mu != pi, with ratios relative only to pi_old -- the samples are trained as
    if on-policy. (Note the honest limit of this finding: it establishes the estimator is biased
    for on-policy PG, not that the bias is what suppressed navigation -- the ablation tests that.)
  * **Check 2 -- what the imagination selects on.** `jax_sim/imagination_jax.py`: for each of the
    12 actions, hold that ONE action constant for K=5 steps through the learned
    `carry_forward_dynamics`, sum gamma-discounted `value_from_carry` along the rolled carry, take
    argmax (disabled actions masked). Two structural notes: (i) the k=0 term is
    `value(carry_0)`, identical for every action, so only the k=1..4 terms can differ across
    actions; (ii) the score is only as informative as the carry-dynamics model's action
    conditioning (carry_fwd MSE ~0.0002 on the dashboard, i.e. next carry ~ current carry).
    Observed: imagination_agree averaged **8.0%** over 2707-2735 (chance for 12 actions = 8.3%)
    -- indistinguishable from chance, and the imagined-action histogram is Stay-heavy (39-55%).
    The gate itself is batch-relative (`conf < mean(conf)*1.0`), so ~half+ of agents imagine by
    construction, independent of absolute confidence. **Read: the imagined action is a
    near-deterministic but uninformative function of state.**
  * **Baselines for the mechanism test (window 2707-2735, beta=24, gate on):** PPO clip fraction
    mean **0.384** (min 0.277, max 0.642, sd 0.100); conf_gate_imagine_frac 77.3%.
- **REGISTRATION -- gate-off ablation (timestamped 2026-09-30T16:34Z, before launch; resume from ckpt 2733, so
  update 1 = ppo 2734).** One variable: `phase9_canvas.imagination_force_reactive: true` -- the
  reactive action is always executed; imagination and the gate are still computed for telemetry
  (`imagination_agree`, `would_imagine`), never override the action, never charge the
  metabolic tax. Everything else identical: beta=24, same config, same code otherwise.
  * **Fingerprint:** `imagination_force_reactive` appended (and `hearth_ramp_hold_stage`, below),
    so resuming from 2733 resets tripwires and the curriculum by the designed mechanism (the
    fingerprint length changes -> flagged as a schema change -> reset).
  * **Curriculum held at A0 for the whole window** (`ctd_competence_ramp.hearth_ramp_hold_stage: 0`,
    a new key; -1 = off is the default): stage advance is suppressed (logged as HELD with the reason
    it would have fired). A0 is the like-for-like regime where the clean null was measured
    (17 need-configurations, excess -0.56). Disclosure: this second appended field goes beyond
    Cam's "add the gate setting" -- same run, same reset semantics, so appended together.
  * **Apparatus (update 1):** `conf_gate_imagine_frac` must read 0.0% AND executed==reactive must
    read 100%. **Enforced in code:** the loop raises `SystemExit(1)` with an `[APPARATUS] FATAL`
    line on the first dashboard if either fails, when the flag is on.
  * **Mechanism (updates 2-5 = ppo 2735-2738):** mean PPO clip fraction <= 0.20 (baseline 0.384). If
    it stays ~0.33+ with the gate off, the off-policy mechanism is wrong even if something else
    improves.
  * **Primary:** the registered hearth test. Metric `AlignedDist holders->matching_hearth`.
    Window 30 updates = ppo 2734-2763. **First-5 (post-transient) = 2736-2740** (drops the first two
    empty-handed post-resume updates 2734-35, same rule as before, decided before any data);
    **last-5 = 2759-2763.** Pass = decline >= 10% AND > 2x SE of the difference of the two
    5-update means (SE from update-level means, never records). Secondary literal first-5 = 2734-2738.
  * **Secondary (red-flee positive control, from the corpus, last 15 updates 2749-2763):** the
    d<=1 mean m.u (baseline -0.071, SE 0.011) should strengthen well past baseline -- operationalised
    here as <= -0.14 (2x); and a nonzero effect at d=2 (baseline +0.020, SE 0.008) -- operationalised
    as < 0 with |z| > 2 against the far-red baseline. These two thresholds are my operationalisation
    of "well past" and "nonzero", fixed now, before data; Cam may tighten them before results.
  * **Known side effect, not a second variable in intent:** with imagination not executed, the
    imagination metabolic tax (mean ~0.0019/step) is no longer charged, so the energy economy shifts
    slightly; energy and population will be watched for a knock-on. Halt rules and tripwires stay armed.
  * **(c) skipped** per Cam: the ablation answers the same question more decisively.

- **Pre-launch disclosures for the gate-off ablation (2026-11-04T05:54Z, before any outcome data).**
  * **Platform moved back to Modal** (profile `christopherchungus`, fresh workspace). Lightning's
    API has returned Cloudflare 521 for over a day on both accounts (plain curl to
    `lightning.ai/v1/...` too) -- a Lightning-side outage, not ours. Volume `throng-runs` created
    fresh; checkpoint 2733 + both corpora + train.log uploaded from `~/throng-migration-bundle/`
    (29/29 files, byte-identical). CPU preflight passed: clone at `385527d`, latest step 2733,
    fossil guard clean, restore-through-grafting OK for both teams (emb_own (29,256)).
  * **Not a treatment, disclosed anyway:** the Modal image installs `jax[cuda12]` unpinned and
    resolved the 0.10.2 CUDA plugin; Lightning ran jax 0.10.0. First thing to check if the ptxas
    crash recurs. GPU is A100-80GB (Lightning ran L40S).
  * **Duplicate ppo numbers on the volume:** the uploaded corpora/train.log run through 2735 (the
    crashed gate-on run), but this resumes from 2733, so ppo 2734-2735 will exist twice -- gate-on
    (old) and gate-off (new). Every corpus/log analysis of this window must filter on the new
    run's `launch_id` (printed at launch) or start after the new launch banner; "last occurrence
    wins" protects only the dashboard parse. Same trap as the `8711e4a20e62` duplicate.
  * **Guard false-halt risk checked:** only the gate line and its else-branch assign `b_actions`
    (grep), so `executed==reactive` must be exactly 100% under the ablation.
  * **Ablation not run end-to-end locally** (the CPU smoke was killed twice by session restarts and
    a third time for RAM before its first dashboard); update 1 on the real run is the end-to-end
    test, enforced by the in-code apparatus halt.

- **Cam's approval + interpretation, fixed before results (2026-11-04T05:58Z).** Approved as registered,
  thresholds included. Read of check 2: carry-forward MSE ~0.0002 makes the 4 action-dependent
  terms differences of near-identical values -> deterministic noise; and a 77% gate picking Stay
  39-55% of the time freezes about a third of executed moves. Both mechanisms predict a navigation
  null and the ablation removes both, so a pass will not say which mattered (accepted; separate
  later only if it matters). Disclosure 2 (no metabolic tax) left as is: better-fed creatures
  weaken the pull toward hearths, so it biases toward FAIL, making a pass more credible.
  **Interpretation:**
  * clip <= 0.20 AND primary passes -> the gate was the blocker; keep force_reactive on, next run
    resumes the A0->A1 progression.
  * clip <= 0.20 AND primary fails -> estimator fixed, navigation still absent: the honest learner
    question; Cam brings a proposal.
  * clip stays high -> the mechanism story is wrong whatever distance does; stop and find what
    else produces off-policy-looking updates BEFORE reading the primary.
  **Operational changes before update 1 (no outcome data existed):** the first launch
  (ap-1KfT8fhCEO6GwFHfFc9T7F) confirmed the fingerprint reset (schema 24 -> 26 fields) and A1 ->
  A0, then was stopped while still compiling -- no update completed, volume still held only
  checkpoint 2733. `checkpoint_interval` 2000 -> 512 (every update instead of every 3; durability
  only, not in the fingerprint), re-pinned to `bcaf1ce`, relaunched. Reporting cadence per Cam:
  update-1 apparatus line, then updates 2-5 clip, then nothing until 2763.

- **OPEN (logged 2026-11-04, Cam ruling 4): red catch reward reads ~6.5 per caught blue against `reward_red_catch:
  3.0`.** Over 2707-2735, red's per-update reward mean tracks blue_caught at r=0.95 with slope ~6.5. Likely more than
  one red credited per caught blue (`r_caught_any` is per red), not yet separated. Does not touch blue navigation;
  running pin left alone; to be checked after ppo 2763.

- **Gate-off ablation -- apparatus and mechanism results (Modal `ap-R21DLvT8DELG11wxilbRam`, pin `bcaf1ce`,
  `launch_id=d6efb75d25c5`; every corpus/log analysis of 2734-2763 filters on this id).**
  * **Apparatus (update 1, ppo 2734): PASS.** `conf_gate_imagine_frac=0.0%`, `executed==reactive=100.0%`, stage
    0/3 [A0], beta=24. Held on every update through 2740.
  * **Mechanism (updates 2-5 = ppo 2735-2738): PASS.** PPO clip fraction 0.067 / 0.060 / 0.052 / 0.059, **mean
    0.060** against the <= 0.20 bar (baseline 0.384 over 2707-2735, gate on). Update 1 (excluded from the registered
    mean) read 0.078; 2739-2740 read 0.044 / 0.046. The off-policy story predicted exactly this.
  * Per Cam: nothing further reported until the window closes at ppo 2763.

- **Gate-off ablation -- window closed (ppo 2734-2763, `launch_id=d6efb75d25c5`).** No tripwire, halt or apparatus
  failure on any update; `executed==reactive` 100.0% and curriculum HELD at A0 throughout; blue 194-200, red 250.
  App stopped at ppo 2764 (checkpoints 2734-2764 on the volume); Modal metered $4.98, all credits.
  * **Primary: FAIL.** `AlignedDist holders->matching_hearth`, update-level means: first-5 (2736-2740) 39.12
    (sd 1.87), last-5 (2759-2763) 40.07 (sd 2.03); diff -0.95, SE 1.23 -- a 2.4% *increase*, z -0.77. Secondary
    literal first-5 (2734-2738) 39.95 -> 40.07, -0.3%, z -0.09. Not registered, read for context only:
    `holders->any_hearth` 21.91 -> 21.53 (1.7% decline, z 4.2) -- far below the 10% bar; holder coverage rose
    ~41% -> ~55% of alive-steps.
  * **Secondary red-flee: FAIL, and weaker than baseline.** Corpus records with this launch_id, updates 2749-2763
    (55,854 records; update u = steps [(u-1)*512, u*512)): d<=1 mean m.u **+0.009** (SE 0.016, 15 updates) against
    the <= -0.14 bar and the -0.071 baseline; d=2 -0.021 (SE 0.013) vs far>20 -0.012 (SE 0.023), z -0.36 (bar |z|>2).
    Off-by-one update assignment (u = step//512) gives the same read (d<=1 +0.012, d=2 z +0.40).
    **Instrument check:** the analysis script was rebuilt (the original was lost with the scratchpad; now
    `~/throng-smoke/scripts/red_flee.py`) and run on the old corpus 2673-2735 excluding `8711e4a20e62`: 221,885
    records (same count), d<=1 -0.0707 SE 0.0106, d=2 +0.0205 SE 0.0081 -- reproduces the published baseline.
  * **Exploratory, not registered: the old flee came from the executed imagined actions.** Last gate-on launch
    (`3475709db423`, 2700-2735): d<=1 -0.075 (SE 0.013). First gate-off stretch (2736-2748): d<=1 **+0.037**
    (SE 0.013). The flee vanished the moment imagination stopped being executed, before 13 updates of training could
    plausibly erase a learned behaviour. Read: the reactive policy carries no adjacent-red flee at all; the weak
    -0.07 "positive control" was the imagination path's carry-dynamics argmax, not the learner.
  * **Interpretation (Cam's tree, fixed 05:58Z):** clip <= 0.20 (0.060) AND primary fails -> "estimator fixed,
    navigation still absent: the honest learner question; Cam brings a proposal." Nothing relaunched.
  * **OPEN item closed -- red catch ~6.5 per caught blue.** Source: a caught small blue pays each red in
    Chebyshev range `phase16.reward_small_blue: 3.0` (`r_caught_small`, a count) **plus** `reward_red_catch: 3.0`
    (`r_caught_any`), wired additively alongside the phase16 terms per Cam's earlier ruling (comment at
    `main_jax.py` ~1155) = 6.0 per credited red; every red within radius 1 of the caught blue is credited, so
    multi-red catches lift the per-caught-blue slope above 6. Not a bug against the ruling; whether 6.0 is the
    intended magnitude is Cam's call.

- **Amendment to the entry above (same day, before reporting to Cam) -- two claims stated more strongly than the
  data supports.**
  * **The exploratory flee note is narrowed.** The flee vanishing at the resume is *consistent with* the flee living
    in the executed imagined actions, but other things changed at that same resume and are not separated from it:
    the fingerprint reset A1 -> A0 (accept-any material; holder coverage rose, so hearth pull may override flee),
    the imagination metabolic tax stopped being charged, and the platform changed (L40S/jax 0.10.0 -> A100/0.10.2).
    And gate-off d<=1 = +0.037 (SE 0.013, z ~2.8) is movement *toward* adjacent red, not an absent flee -- a separate
    effect, **unexplained** (candidates, untested: hearth attraction where reds sit near hearths; action/position
    snapshot misalignment). "The reactive policy carries no flee" is withdrawn.
  * **Red-catch: explained from source, not fitted.** 3.0 + 3.0 = 6.0 per credited red is read from the code; the
    multi-red credit factor needed to reach ~6.5 (~1.08 credited reds per caught blue) was not measured, and the
    units of the original slope (per-catch totals vs per-red means) were not re-checked. Status: likely explained,
    not closed.

- **REGISTRATION -- Cam's learner diagnostic, Step 1 (2026-11-04, before any entropy / grad-norm / informed-fraction
  value from 2734-2763 was looked at).**
  * **Already seen, disclosed:** per-update AlignedDist (both), holder coverage, Clip, the red-flee reads, and
    VF_loss / value means for 2734-2736 (dashboard lines read earlier). **Not yet seen:** Entropy, `ppo ent=`,
    `grad_norms`, `adv`/`returns` stats, action counts, and the corpus informed fraction among holders.
  * **Code facts (read from source, not results) that change the premises of Step 1:**
    1. **No global clip.** `_minibatch_step` clips three groups independently to norm 2.0: `head_action`,
       `head_value`, and everything else ("trunk"). Logged `grad_norms (last mb)` are **post-clip**, last minibatch
       only; pre-clip size is not logged. Aux losses use a separate optimizer. Within `ppo_loss`, the trunk gradient is
       pg + 0.25*vf (Huber, delta 0.5, value-clipped at +/-0.2; `detach_value=False`, so the value loss reaches the
       trunk) + entropy + logit penalty + vq. Comms subtree is zeroed before the clip at stage 0 (held all window).
       `ppo_epochs: 1`.
    2. **Entropy ceiling is not ln 12.** Actions 6/7/8 (Push/Guard/Build) are logit-masked -> 9 live actions,
       ceiling ln 9 = 2.197. Logged `Entropy` = action entropy + alarm entropy (2-way head, live for blue except
       feral agents, forced silent). So action entropy lies in [logged - 0.693, logged].
    3. **Hearth position is visible 100% by construction; which hearth matches is not.** `hearth_rel` gives the
       relative vector to all 4 hearths globally. `hearth_need` is masked by `can_see_recipe`, and
       `hearth_shaping_term` targets the nearest *matching* hearth with no A0 accept-any branch -- so for an
       uninformed holder both the shaping target and the registered primary metric are unobservable.
    4. **Advantages are truncated Monte Carlo, no bootstrap.** `compute_gae` discards lambda ("no value
       bootstrapping"); returns are discounted sums (gamma 0.999) to the end of the 512-step rollout, carry starts at
       0, nothing injects gamma*V(s_T) before the call (checked: rl_jax 388, main_jax 3133/3302, rollout -> batch
       unchanged); advantage = return - V, normalised, clipped +/-10. Prediction: under pure MC, potential shaping
       telescopes to beta*(gamma^(T-t)*Phi(s_T) - Phi(s_t)); a move's credit carries the variance of every later step
       in the rollout, so shaping gives no variance reduction, and the return scales with steps remaining.
  * **Free reads now, bars fixed here:**
    - **Entropy** per update; first-5 (2736-2740) vs last-5 (2759-2763), update-level SE. Action head called
      **near ceiling** if logged >= 2.670 (lower bound 1.977 = 0.9 * ln 9); **clearly below** if logged < 1.977;
      in between = indeterminate from logs. Executed-action marginal entropy from `Actions (blue)` counts, labelled
      an *upper bound* on mean conditional entropy (only "far below ceiling" is informative).
    - **Clip firing:** fraction of updates on which each group (actor/critic/trunk) reads >= 1.99 post-clip.
    - **Informed fraction among holders** (corpus, this launch_id, 2734-2763): records with inventory > 0 and
      `can_see_recipe=True`. If < 0.5, the matching-hearth task as shaped and measured is unobservable for most
      holder-steps.
    - **Synthetic, real `compute_gae`:** signal-to-noise of a one-cell toward-vs-away move under shaping only
      (beta 24, gamma 0.999, T 512, random-walk continuation), against a bootstrapped-GAE (lambda 0.95) mutant that
      must differ. No network, seconds on CPU.
  * **Deferred (needs an offline recompute from a checkpoint):** approx KL, explained variance, per-loss trunk
    gradient split, per-move advantages (toward vs away hearth; flee vs toward-red at d<=1), advantage vs
    time-in-rollout. Not built yet: a production-population CPU rollout has the memory profile that overloaded the
    operator's machine, and Modal spend is frozen until the Step-2 testbed passes. Sent to Cam first.
  * **Interpretation mapping to Cam's branches:** fact 4 is the "reward/GAE path" branch independently of the
    deferred advantage check; facts 1-2 make "optimizer path" readable only partially from the logs; fact 3
    answers "visibility" -- positions yes, target identity only for informed agents.

- **Learner diagnostic Step 1 -- free-read results (launch `d6efb75d25c5`, 2734-2763; bars as registered above).**
  * **Entropy: not near ceiling, and rising.** Logged (action+alarm) first-5 1.452 -> last-5 1.992, diff +0.540,
    SE 0.032; monotone-ish climb 1.11 (2734) -> 2.03 (2763). First-5 < 1.977 -> action head **clearly below
    ceiling** at the start; last-5 is in the indeterminate band. Executed-action marginal entropy 1.47 -> 2.08
    (upper bound, of 2.197). Read: the policy is *not* frozen near uniform with tiny updates -- it moves every
    update, toward more randomness. Exploratory reading, not tested: relaxation of a policy that was peaked
    under gate-on training (77% of executed actions came from imagination), with no strong task gradient
    opposing the entropy bonus (`ppo_entropy_coef` 0.02).
  * **Clip firing: never.** Post-clip last-mb norms never reach 1.99 in any group on any update (means: trunk
    0.229, actor 0.067, critic 0.011), so post-clip = pre-clip here and clipping dilutes nothing. Cam's "clipping
    share" branch is ruled out for this window. (Adam normalises step size, so small raw norms do not by
    themselves mean small steps.)
  * **Target identifiability:** informed fraction among holders **0.562** (112,153 records; per-update range
    0.376-0.806) -> above the 0.5 bar: the matching hearth is observable for a majority of holder-steps.
    Overall informed fraction 0.567 -- far from the 0.25 the comm draft assumes (mutation drift; the draft's
    freeze is needed).
  * **Synthetic, real `compute_gae` (`~/throng-smoke/scripts/mc_snr.py`; 4000 random walkers, 128 torus, one
    hearth, shaping only, V=0 for both estimators):** per-sample SNR of toward-vs-away credit **0.161** (t<128:
    0.132; t>=448: 0.549) vs **1.038** for bootstrapped GAE lambda=0.95 (mutant differs, as required) and 134 for
    TD(0). ~6.5x lower SNR = ~40x more samples for the same gradient signal, before adding the noise of every
    other reward stream in production. Confirms code fact 4 as a large effect.
  * **Read against Cam's branches:** optimizer path -- no clipping problem; entropy not stuck at ceiling (the
    policy moves, but toward uniform). Visibility -- not the blocker. Reward/GAE path -- **the lead suspect**: the
    estimator throws away the bootstrap, so shaping's dense credit is buried under 512 steps of return noise.
    Still deferred: KL, EV, per-loss trunk split, per-move advantages (need a checkpoint recompute).
  * **Constraint for Step 2:** the testbed only tests this path if it uses the production `compute_gae` with
    T=512, gamma 0.999; a short rollout or a bootstrapped estimator could pass and say nothing about production.
    Natural design: run it twice, production estimator vs bootstrapped GAE (needs V(s_T) at the rollout
    boundary), same bar -- both free on CPU.

- **Amendment to Step 1 results (same day, before reporting to Cam).**
  * **The first synthetic measured a setup artifact.** Shaping-only reward is policy-invariant (every policy's
    value is -beta*Phi(s0); config comment ~line 479), so its "credit" came from the truncation term and from
    discounting at gamma*lambda vs gamma -- not from any task signal; and its lambda=.95 "mutant" had V=0, i.e. no
    bootstrap. Replaced by `~/throng-smoke/scripts/mc_snr_deposit.py`: a deposit reward on reaching the hearth
    (respawn after), with and without shaping; the mutant is GAE lambda=.95 **with** bootstrap, critic = mean MC
    return per distance bin, gamma*V(s_T) at the boundary. Per-sample toward-vs-away SNR:
    deposit only: real MC 0.015 (V=0) / 0.031 (fitted V) vs bootstrapped **0.133** (~4-9x);
    deposit + shaping: real 0.159 / 0.139 vs bootstrapped 0.97-0.98 (~6-7x) -- the shaping part of this comparison
    exists only because the fitted critic does not exactly absorb the potential, so it is illustrative, not clean.
    Deposit-only is the artifact-free row, and it agrees in direction: the production estimator costs roughly
    4-9x in SNR (~16-80x in samples). Deposit magnitude does not change SNR (scale-invariant; 0.3 and 2.7 identical).
    Single cell on a 128 torus: random walkers almost never deposit, so absolute values are small by design.
  * **Step 2 as specified ("shaping only") is degenerate:** with a correct estimator and a converged critic there
    is nothing to learn. The testbed needs the deposit reward (shaping optional on top), and should compare the
    production `compute_gae` against lambda + bootstrap.
  * **Exploratory -- the critic looks poorly fit, which matters for any bootstrap fix.** Value mean trails return
    mean all window (4.6 vs 9.4 at 2734; 5.0 vs 6.4 at 2763); values std ~2.5 vs returns std 6-7.5; critic grad norm
    ~0.01; value clip +/-0.2 in raw (unnormalised) return units, Huber delta 0.5, vf_coef 0.25. GAE with a bad V
    trades noise for bias, so the deferred EV check rises in priority and Step 2 needs a value-tracking bar.
  * **Corrections:** clipping -- only the last of ~200 minibatches per update is logged, so "ruled out" becomes
    **very unlikely**. Holder definition -- corpus `inventory>0` (60% holders) is not the shaping's held material
    (dashboard held_any n ~87-96%); cite the overall informed fraction **0.567** (`can_see_recipe` is independent
    of holding). Matching n ~55% vs any n ~95% of alive-steps: ~40% of alive-steps hold a material no hearth
    currently wants and get no shaping.
  * **What pushes entropy up, concretely:** the entropy bonus (`ppo_entropy_coef` 0.02) and the L2 logit penalty
    (0.01 * mean logit^2) in `ppo_loss`. `action_entropy_coef: 0.05` in config is **not wired** anywhere (grep:
    no reader in jax_sim/ or scripts/). **The starting peakedness was one direction:** executed East share 60%
    (2734) -> 45% (2740) -> 35% (2749) -> 26% (2763); other moves flat ~7-10%. Supports, without testing, the
    reading that a gate-on-era "drift East" bias (blind drift on a torus, not navigation) is relaxing.

- **Derivation (Cam, 2026-11-04), logged as a derivation, not a result -- with one correction.** Under truncated MC
  (lambda=1, no bootstrap) potential shaping telescopes: shaping's part of G_t = gamma^(T-t)*Phi(s_T) - Phi(s_t).
  The action's effect on Phi(s_(t+1)) cancels inside the sum; Phi(s_t) is fixed before the action (absorbed by V).
  **Correction:** Cam's last step, "gamma^512 makes the end term negligible", does not hold numerically:
  0.999^512 = 0.599 (0.774 at 256 steps left, 0.938 at 64). And the action's displacement is not cancelled in
  s_T: under a continuation that does not depend strongly on position, a one-cell move shifts the whole remaining
  path, so it reaches Phi(s_T) with weight 0.6-1.0 of the immediate shaping. So per-action credit survives *in
  expectation* under MC; what it loses is precision (it carries the variance of the rest of the rollout). Existing
  evidence (`mc_snr_deposit.py`, random-walk policy): adding shaping raised production-MC per-sample SNR from
  0.015-0.031 to 0.14-0.16. Both readings are registered below; arm B decides between them.
  Also corrected: `ppo_value_coef` is **0.1** (Step 1 entries above say 0.25 -- wrong; the config value is 0.1).

- **REGISTRATION -- learner testbed, Step 2 (2026-11-04, before any learning metric from it was read; one timing
  smoke ran with metrics suppressed).** Script `scripts/learner_testbed.py`.
  * **Setup:** real `AgentNetworkJax` (production dims, fresh init), real `ppo_update` / `_minibatch_step` /
    `ppo_loss` (production coefficients from config via `_normalize_config`: lr 3e-4, clip 0.2, vf 0.1, ent 0.02,
    vq 0.1, minibatch 512, 1 pass), production sampling (masked logits + alarm head, joint log-prob), T=512,
    gamma 0.999. World: 15x15 torus, one hearth at the centre (all four `hearth_rel` slots point at it),
    hearth_radius 1 (production is 3; on 15x15 that would cover 22% of cells and leave no distance range),
    agents always hold wood and are informed; deposit = Craft within radius -> reward 2.7 (0.3 deposit + 2.4 solo
    completion), respawn at a random cell outside the hearth, still holding, no done. 32 independent copies of
    the one-agent world per batch. No reds, comms, neighbours, food, energy or aux losses (separate optimizer,
    not part of `ppo_loss`). Comms subtree frozen as at stage 0.
  * **Arms:** A production MC, deposit only. B A + shaping (beta 24). C GAE lambda 0.95 + bootstrap gamma*V(s_T),
    deposit only (production function swapped for the update only; jax_sim/ unchanged). D C + shaping.
    3 seeds per arm (0, 1, 2), same N updates for every run.
  * **Metric (per run):** mean toroidal Chebyshev distance to the hearth centre over all agent-steps of an
    update. **Decline** = mean of updates 1-3 minus mean of the last 3 updates. Secondary: deposits per env
    per update (same windows).
  * **An arm learns** if all 3 seeds show decline >= 0.5 cells (~10% of the ~5.0 random-walk expectation).
  * **"X faster than Y"** = every seed of X has a larger decline than every seed of Y (min_X > max_Y: outside
    seed spread). "X ~ Y" = the seed ranges overlap.
  * **Predictions:** C faster than A; D faster than C (Cam). B vs A: Cam predicts B ~ A; the correction above
    predicts B faster than A. Whichever holds is the read on the derivation.
  * **Critic bar (C and D):** explained variance of V against that arm's own targets, mean over the last third
    of updates, >= 0.2 in every seed. If C or D fails it, C/D are not read: the +/-0.2 raw value clip is fixed
    first and C/D re-run.
  * **N** is fixed at launch, before any metric is read, by the platform: the timing smoke measured ~9.6 min per
    update on the operator's CPU (72-107 s rollout + 32 minibatch backward passes at ~15 s each; ~1 GB RSS).

- **Registration addendum -- Step 2, before launch (Cam's approval + addition, 2026-11-04).**
  * **Platform and spend:** Modal GPU (L4, A10G fallback), app `throng-testbed` (`scripts/modal_testbed.py`),
    pinned `a2d910b`. Cam approved spend under a **$10 cap** (operator approved first). Billing summary before
    launch: metered $4.87.
  * **Arm E -- frozen-policy control,** 3 seeds (0, 1, 2): A's reward and estimator, lr=0. Adam at lr=0 applies an
    exactly-zero update, so E skips the PPO step (identical parameters, cheaper). Measures the distance-change
    noise floor from deposit/respawn dynamics and starting-weight biases.
  * **Revised "learns" rule:** an arm learns only if **all 3 seeds** show decline >= 0.5 cells **and** every one of
    its seeds has a larger decline than **every E seed**. "Faster than" and the critic bar are unchanged.
  * **N = 80 updates per run, all 15 runs** (fixed from the GPU timing probe: ~18 s/update warm, 35 s for the first;
    metrics suppressed). Windows: first = updates 1-3, last = updates 78-80. Estimated cost ~$6-7.
  * **Config audit:** the testbed is exempt (Cam). Production launches stay blocked until the allowlist is
    settled; triage sent as `docs/drafts/2026-11-04_config_audit_triage.md`.

- **Learner testbed Step 2 -- results (app `ap-JsKvgr4kSzMRdqDJ0oYOa0`, pin `a2d910b`, 15 runs x 80 updates;
  raw `docs/data/testbed_2026-11-04.jsonl`, registered analysis `docs/data/testbed_2026-11-04_analysis.txt`).**
  Spend: metered $4.87 -> $10.99 = **$6.12** (includes the timing probe), under the $10 cap.
  * **GPU types:** no run landed on L4. All A and B seeds and C0 reported "NVIDIA A10"; C1, C2, D0 reported "NVIDIA
    A10G". D1, D2, E0-E2 ran in reused containers (Modal reuses warm containers for later inputs), which can only be
    narrowed to those same two types (`docs/data/testbed_2026-11-04_gpu_map.txt`).
  * **E (frozen, noise floor):** declines -0.006 / -0.046 / -0.018; deposits flat (~0.7-1.6/env). The control works:
    deposit/respawn dynamics and starting-weight bias move distance by ~0.
  * **A (production MC, deposit only): LEARNS.** Declines 1.966 / 0.966 / 1.949 (all >= 0.5, all > max E);
    deposits/env 0.7-1.5 -> 5.6-13.6.
  * **B (MC + shaping): LEARNS.** Declines 0.753 / 1.813 / 1.042.
  * **B vs A: B ~ A** (seed ranges overlap). **Cam's prediction held; my correction's prediction (B faster) failed.**
    Caveat: testbed D_max = 7, so one cell of shaping = 24/7 ~ 3.4 vs production 24/64 ~ 0.375, with deposit 2.7 in
    both -- B ran at ~9x production's shaping-to-deposit ratio, which limits what B says about production beta. Open,
    not explained: under MC, shaping raised per-sample SNR in the random-walk synthetic but showed no benefit here.
  * **C and D: critic bar FAIL -> not read**, per the registration. EV over the last third: C -1.925 / -1.380 /
    -1.470; D -0.069 / -0.050 / -0.056 (bar >= 0.2). The script's LEARNS column for C/D is not a verdict.
  * **The critic is broken in every arm, not only C/D (exploratory).** EV against each arm's own targets is ~0.00
    throughout A and B; C0's EV went 0.18 (update 20) -> -2.15 (update 80) as returns grew. Consistent with the
    +/-0.2 raw-units value clip: V can move at most 0.2 per update while returns grow by far more. A learned with
    essentially no working baseline -- a production fix in its own right, not only a C/D prerequisite.
  * **Headline:** the production estimator, real network and real `ppo_loss` learn one-body navigation in the
    testbed. Truncated MC costs signal-to-noise but is not by itself fatal at this scale, so it does not alone explain
    production's flat distance. Cam's branch: "passes -> add one ecology factor at a time until it breaks." Obvious
    differences from production, listed for Cam (not a sequence): grid 15 vs 128 (D_max 7 vs 64), matching-hearth
    identity (informed fraction, rerolling needs), the other reward streams (red catches, food, energy), population
    and neighbours, the comms freeze.
  * **Two problems for any C/D re-run (Cam's call; nothing launched):** (1) the registered metric cannot see speed --
    deposits respawn agents at random cells, so distance saturates near ~3 and end-of-run decline can read "~" even
    when one arm learns far sooner. A speed metric (cumulative deposits, or first update below a distance threshold)
    would have to be registered for the re-run, disclosing that the A/B and C/D curves have now been seen.
    (2) Cost: ~$3.9 of the cap remains; 6 C/D runs at 80 updates are ~$4.3.

- **Cam's ruling after Step 2 (2026-11-04).** Budget: the $3.90 was the testbed sub-cap's remainder; the overall cap
  has ~$29 left; **new $10 sub-cap** for Step 0 + Step 1. Order: config cleanup (done, `71eb97c`: 28 dead keys
  deleted, 16 pending keys set to what the code does and allowlisted, fingerprint byte-identical, tests unchanged,
  audit now a launch gate in `modal_app.py`), then Step 0, then Step 1. Report when the ladder is complete.
  Config note for Cam: `gumbel_tau` and `action_entropy_coef` are still read by the legacy PyTorch path in
  `agents/` (unreachable from every current launcher); `gumbel_tau: 0.0` would divide by zero there if revived.

- **REGISTRATION -- Step 0, critic fix (before any run of it).** Change (`jax_sim/rl_jax.py`): the value loss no
  longer clips predictions to old_values +/- 0.2 in raw return units; the Huber error is measured in units of a
  running return std (`RunningReturnStd`, parallel Welford over each update's targets, one per team, updated
  before use, not checkpointed). Value head stays in raw units, so GAE/bootstrap/imagination see unchanged value
  semantics. Test `tests/test_value_target_normalization.py` (3 tests; the old production code fails 2 of them --
  mutation-checked by restoring the old block in `ppo_loss`); full suite 153 passed / 56 skipped.
  * **Verification:** 2 seeds (0, 1) of arm A at 15x15, N=80, testbed now passing `return_rms` exactly as
    production does. **Bar (both seeds):** EV, mean over updates 36-40, >= 0.2; **and** learning no worse than the
    old A seeds = registered decline (updates 1-3 vs 78-80) >= 0.966 (the smallest old-A decline). Pass -> the fix
    lands in production. Also in this plan: a one-update timing probe at 128x128 (metrics suppressed) for the
    ladder's cost.

- **REGISTRATION -- Step 1, map-size ladder (a SCAN; runs only if Step 0 passes; break point to be confirmed later
  with 3 seeds).** Disclosure: the 15x15 curves of arms A-E (Step 2) have been seen.
  * **Design:** sizes 15, 32, 64, 128; one hearth at the centre, hearth_radius 1, deposit reward only (2.7), no
    shaping; 32 envs; N=80; T=512, gamma 0.999. Arms: **A** production MC (with the Step 0 critic fix), **C'**
    GAE lambda 0.95 + bootstrap with the fixed critic (testbed arm C at the new pin), **E** frozen (lr=0). Seeds
    0, 1. 24 runs, `scripts/modal_testbed.py --plan ladder`.
  * **Metric:** deposit rate = mean deposits per env per update over updates 71-80. Ratio = a run's rate / E's rate
    at the same size (mean over both E seeds). Floor if E's pooled window has zero deposits: 0.5 deposits over the
    pooled window (2 seeds x 10 updates x 32 envs). **An arm learns at a size if both seeds reach >= 2x E.**
    Reference (random walk, 1/G^2 per step): expected E rate ~2.3 / 0.50 / 0.12 / 0.03 deposits per env-update at
    15 / 32 / 64 / 128.
  * **Prediction (Cam):** A breaks once typical start distance exceeds the ~sqrt(512) ~ 22-cell random-walk reach
    -> fails at 64 and 128; C' holds further out. Typical start distance (mean toroidal Chebyshev to the hearth,
    uniform respawn outside radius 1): 5.1 / 10.8 / 21.4 / 42.7 at 15 / 32 / 64 / 128 -- so 64 sits right at the
    predicted boundary.
  * EV is reported per run, descriptive only (not a gate). Spend baseline before Step 0: metered $10.99.

- **Step 0 -- result: FAIL as registered (pin `f7b0392`; raw `docs/data/testbed_step0_2026-11-05.jsonl`).**
  Spend $1.21 (metered $10.99 -> $12.20; 2 runs + the 128x128 probe). All on "NVIDIA A10".
  * A0: EV(36-40) +0.004, decline 1.859, deposits/env (78-80) 10.64. A1: EV(36-40) +0.001, decline **0.679**
    (< 0.966), deposits 3.61. Both fail the EV bar; A1 also fails the learning bar. **The fix does not land; the
    ladder (conditional on Step 0) is not launched.** Nothing is in production: `modal_app.py` is still pinned to
    `bcaf1ce`, but master now carries the unvalidated value-loss change -- Cam to rule (keep pending vs revert).
  * Same seeds as Step 2's A0/A1 (same init seed; GPU runs are not bitwise deterministic, so the pairing is
    weak): decline 1.966 -> 1.859 and 0.966 -> 0.679; deposits 13.59 -> 10.64 and 5.60 -> 3.61 -- both seeds
    somewhat worse with the fix. Reported, not explained.
  * **The EV bar was mis-specified for an MC arm (my registration error).** Against truncated-MC targets the
    return at step t sums the rewards left in the rollout, so it scales with steps remaining, which the critic
    cannot observe. `scripts/ev_ceiling_truncated_mc.py` (Bernoulli deposits at the observed rates, gamma .999,
    T 512): a critic that knows only t reaches EV 0.12 / 0.33 / 0.60 / 0.90 at 1 / 3.6 / 10.6 / 60 deposits per
    rollout, while a t-blind critic that knows the next 20 steps' rewards *exactly* is capped at 0.09 / 0.07 /
    0.05 / 0.02. So EV >= 0.2 is unreachable for arm A by construction, whatever the critic; EV ~0 in every MC arm
    (Step 2 and here) says nothing about the clip. The fix can only be tested on bootstrapped targets (arm C'),
    where this time dependence is mostly absent -- which also means C's EV of -1.4 to -1.9 in Step 2 *was* a real
    critic failure.
  * **Cost data for the ladder:** the 128x128 probe ran 18.5 s/update warm (34.7 s first) -- the same as 15x15
    (the testbed env cost does not grow with grid size), so ~$0.41 per 80-update run at every size.

- **Cam's ruling after Step 0 (2026-11-05).** The EV bar on arm A was his design error too; the ceiling script
  stands. Named precisely: **production treats rollout truncation as termination, with no bootstrap at the cut.**
  (1) Re-test the critic fix on C'. (2) **Master reverted** (`3cbf1b7`, jax_sim/ byte-identical to `71eb97c`); the
  fix lives on branch **`critic-fix`** (`71a8d37`) as an opt-in path (`value_clip=False` + `RunningReturnStd`);
  its default path is the production formula exactly (tested: `test_default_path_is_the_unchanged_production_
  formula`; suite 154 passed / 56 skipped). The worse same-seed arm-A runs under the fix are disclosed as
  unexplained. (3) The ladder waits for C'; A stays production (old critic); E trimmed to 1 seed per size.
  Stage sub-cap raised to **$15** (metered baseline $10.99 -> guard at $25.99), with M's knowledge.

- **REGISTRATION -- Step 0 re-test on C' (before launch).** Pin `71a8d37` (critic-fix branch), plan `cretest`.
  15x15, N=80, 32 envs, deposit only; arm C (GAE lambda 0.95 + bootstrap) with `--critic-fix`, at **vf_coef 0.1**
  and **vf_coef 0.5**, seeds 0 and 1 (4 runs, ~$1.60).
  * **Bar, per vf, every seed:** EV mean over updates 36-40 >= 0.2; **and** decline (updates 1-3 vs 78-80) >= 0.966
    (lowest Step 2 arm-A seed, production critic) **and** > every Step 2 E seed (max -0.006).
  * **Decision:** both vf pass -> take 0.1 (smaller change); one passes -> that one; **neither -> stop and report**
    (the critic problem is then something else -- trunk sharing, lr -- Cam rules).
  * Analysis: `scripts/analyze_ladder.py cretest`, committed before results.

- **REGISTRATION -- Step 1 ladder, as trimmed (supersedes the arm/seed list above; metric and bars unchanged).**
  Runs only if the C' re-test passes, at the same pin. Sizes 15/32/64/128; **A** = production critic, seeds 0, 1;
  **C'** = critic fix at the vf chosen by the re-test, seeds 0, 1; **E** = frozen, **seed 0 only**. 20 runs,
  ~$8.20. E's rate at a size is its one seed; floor 0.5 deposits over its window (1 x 10 x 32). Learns at a size if
  both seeds >= 2x E. Prediction (Cam) stands: A breaks beyond ~22 cells (64 and 128); C' holds further out.

- **C' re-test -- result: FAIL for both vf (pin `71a8d37`; raw `docs/data/testbed_cretest_2026-11-05.jsonl`).**
  All 4 runs on "NVIDIA A10". Spend $2.33 (metered $12.20 -> $14.53; ~$0.58/run incl. a fresh image build);
  stage total $3.54 of $15. **Stopped per the ruling; the ladder is not launched.**
  * EV(36-40): vf 0.1 -1.933 / -1.229; vf 0.5 -0.750 / -0.721 (bar 0.2). Learning bars pass everywhere: declines
    1.119 / 1.134 / 1.332 / 1.277; deposits/env (78-80) 47.5 / 52.3 / 59.1 / 56.3 (A in Step 2: 5.6-13.6).
  * **Shape (descriptive, not registered):** EV rises to 0.13-0.39 by updates 15-20, while deposits ramp from ~1
    to ~50 per rollout, then collapses once deposits plateau: updates 30/40/60/80 = -0.04/-2.38/-3.28/-8.40 (vf 0.1,
    seed 0); vf 0.5 degrades slower (-2.22 / -2.71 at 80). Step 2's C (old clip) also ended at -1.4 to -1.9.
  * **Candidates, for Cam's ruling (none tested):** (a) bootstrap feedback -- C's targets are built from V itself,
    and with the clip off V can chase its own targets; (b) shared trunk -- the policy sharpens (entropy ~1.2-1.7)
    and reshapes the features V reads, consistent with vf 0.5 holding up longer than 0.1; (c) the running std is
    pooled over all history (plain Welford, no decay) while the return scale grows ~50x, so it lags the current
    scale; (d) instrument: EV is measured against the arm's own bootstrapped targets, which move with V -- an EV
    against independent targets (e.g. the MC returns of the same rollout) would separate "V is wrong" from
    "the target moved".

- **Cam's ruling after the C' re-test (2026-11-05).** Headline is behaviour: C' made 47-59 deposits/env against A's
  6-14, every seed above the A floor -- **logged as exploratory** (the deposit comparison was not registered),
  pointing at the estimator as the brake. His read of the EV collapse: mostly the metric, partly real -- at plateau
  the agent is ~10 steps from a deposit everywhere against a ~1000-step horizon (gamma 0.999), so the
  state-dependent part of V^pi nearly vanishes, Var(target) collapses and EV blows up on small V jitter; (c) is a
  real defect either way (rescaling the normaliser without rescaling the output layer shifts every prediction when
  the stats move -- what PopArt solves). On (d): MC returns from the same truncated rollout hit the steps-remaining
  ceiling, so they are not independent; the independent check, if needed later, is long-rollout V^pi from a saved
  checkpoint. **Nothing lands in production** until the critic is understood. **gamma 0.999 parked** as the next
  lever (not mixed in now).
  * **Free check:** the 4 re-test runs logged only EV, not Var(target) / Var(target - V), so it cannot be done from
    them. Both are now logged per update (critic-fix `d0d74ef`), with mean V and mean target.

- **REGISTRATION AMENDMENT -- ladder (prospective, before any ladder data; follows the 15x15 C' curves, disclosed).**
  * **Critic quality is a diagnostic, not a gate, for behavioural claims.** The ladder's verdicts rest on deposits
    >= 2x E, which a biased critic cannot fake.
  * **C' runs at vf_coef 0.5,** chosen on EV stability and deposits; **neither vf variant passed its registered
    critic bar** (disclosed). A C' failure at a size reads as **ambiguous** between critic and size; a C' success
    does not.
  * A stays on the production critic. Pin `d0d74ef`; `--plan ladder --vf 0.5`; 20 runs; stage cap $15 (guard at
    metered $25.99; now $14.53).
  * **Free check, reading fixed now** (`scripts/analyze_ladder.py`, `critic_check`): per C' run, Var(target) and
    Var(target - V), ramp window updates 11-20 vs plateau 71-80. **Artifact** = Var(target) plateau/ramp < 0.5 and
    Var(resid) plateau/ramp <= 2. **Diverging** = Var(resid) plateau/ramp > 2 (then PopArt comes first). Otherwise
    neither. Cam's prediction: artifact.

- **Ladder -- results (app `ap-gtURfJtI9gaApfKnS6sttf`, pin `d0d74ef`, 20 runs x 80 updates; raw
  `docs/data/testbed_ladder_2026-11-05.jsonl`, analysis `..._analysis.txt`).** Spend $8.27 (metered $14.53 ->
  $22.80); stage total **$11.81 of $15**. GPUs: 15 all A10; 32 all A10G; 64 and 128 mixed (per run in the data).
  * **Deposit rate over updates 71-80 as a multiple of E (bar: both seeds >= 2x):**

    | size (typical start) | E rate | A (production critic) | A learns | C' (bootstrap + fix, vf 0.5) | C' learns |
    |---|---|---|---|---|---|
    | 15 (5.1) | 0.847 | 15.7x / 6.7x | **yes** | 69.8x / 67.7x | **yes** |
    | 32 (10.8) | 0.147 | 1.62x / 1.17x | **no** | 147.7x / 150.3x | **yes** |
    | 64 (21.4) | 0.025 | 1.62x / 1.38x | no | 0.87x / 3.62x | no (ambiguous: critic or size) |
    | 128 (42.7) | 0.0094 | 0.67x / 2.00x | no | 0.33x / 0.00x | no (ambiguous) |

  * **Cam's prediction (A breaks beyond ~22 cells, fails at 64 and 128): wrong in the direction that matters --
    A already fails at 32**, where typical start distance is ~11 cells, half the random-walk reach. **C' holds
    further out: it learns at 32** (deposits ~0.2 -> ~21 per env-update, both seeds), breaks at 64.
  * At 64, C'1's 3.62x rests on ~29 deposits in its window vs E's ~8 (E is one seed, and its rate at 64/128 is a
    handful of deposits) -- with C'0 at 0.87x, the both-seeds rule fails it. Cam's planned 3-seed confirmation of
    the break point applies here.
  * **Largest size where navigation learns: 15 with the production learner; 32 with C'** (but C's critic -- below).
  * **Free check (registered reading): DIVERGING where C' learned, so Cam's "artifact" prediction fails.** At 15
    and 32 Var(target) did **not** collapse -- it rose (x1.8 at 15, x25-34 at 32) -- and Var(target - V) rose more
    (x7-8 at 15, x44-97 at 32). At 64/128, where nothing was learned, Var(resid) = Var(target) (V explains nothing;
    both shrink with no deposits), so those "artifact"/"neither" labels are degenerate, not evidence.
    Context for the reading: at 15 the plateau target is ~100 (mean V 96 vs target 100) with Var(target) ~2.5, so
    the state-dependent part *is* small relative to the level (Cam's mechanism, in spirit), but V's per-state error
    grew to ~3x the target spread instead of staying flat. Untested candidate, matching his (c): `RunningReturnStd`
    pools all history, and the target mean climbs 0 -> ~100 during learning, so the pooled std is dominated by that
    climb and an error of a few units reads as a small fraction of a std -- a weak value gradient exactly when V needs
    to resolve fine structure. Per the ruling, **PopArt comes first**; nothing lands in production.

- **Cam's ruling after the ladder (2026-11-05).** Logged: **random-walk reach (sqrt(512) ~ 22) was the wrong
  threshold for MC** -- A breaks at 32 (~11-cell starts) because credit SNR collapses well before reach runs out.
  Suggestive only, not a finding: C' breaks at ~21 cells, about where he had predicted A would. PopArt first, as
  ruled. 3-seed confirmation at 64 skipped (superseded by the PopArt arm). Production untouched. Stage sub-cap $5.

- **REGISTRATION -- Stage 3 (before launch).** Pin `b8d6627` (critic-fix branch), plan `stage3`, 6 runs, one
  launch; analysis `scripts/analyze_ladder.py stage3` committed before results. Spend baseline: metered $22.20
  (the billing summary moved down from $22.80 between reads -- reported as read); guard at $27.20.
  * **Checkpoints were not saved** by the ladder (the testbed never wrote params), so per the ruling C'@32 is re-run
    with saving: same config as the ladder's C' (critic fix, vf 0.5, 80 updates), seeds 0, 1; final params written
    to the volume (`testbed/params/`).
  * **Observation check (done):** `tests/test_testbed_obs_matches_production.py` builds the production observation
    with `build_observations_jax` and asserts the testbed's 8-dim `hearth_rel` block is identical at 15/32/64/128
    (mutant without the toroidal wrap is caught). Difference from production, by design: the testbed's one hearth
    fills all four slots; production has four distinct hearths.
  * **(1) Transfer test, eval only:** each run's FROZEN final C'@32 policy, fresh world and carries, 10 rollouts
    (512 steps x 32 envs) at 64 and at 128. Rate = mean deposits/env per rollout over the 10. Floor = the ladder's
    E rate at that size (64: 0.0250, 128: 0.0094). **Transfers at a size iff both seeds >= 2x E.** A seed whose
    re-run did not itself learn at 32 (>= 2x the ladder E rate 0.1469 over updates 71-80) voids its transfer
    reading. Prediction (Cam): transfers at 64, probably at 128.
  * **(2) PopArt on C'** (`PopArt`, EMA beta 0.1 per update, initialised from the first batch, output layer rescaled
    to preserve predictions; tested in `tests/test_popart.py`, mutation-checked), at 32 and 64, seeds 0, 1, vf 0.5,
    N=80. **Critic fixed iff Var(target - V) grows <= 2x from ramp (11-20) to plateau (71-80) at 32, both seeds.**
    **Extends reach iff both 64 seeds reach >= 2x the ladder E rate at 64.** Prediction (Cam): critic fixed; 64
    uncertain.
  * **How it reads (Cam):** transfer passes -> the path to the full world is a size curriculum (32 -> 64 -> 128),
    not learner surgery. Either way the communication experiment moves into the testbed at 32x32 (draft only).

- **Stage 3 -- results (pin `b8d6627`; raw `docs/data/testbed_stage3_2026-11-05.jsonl`, analysis `..._analysis.txt`;
  final params on the volume under `testbed/params/`).** All 6 runs on "NVIDIA A10". Spend **$3.38** (metered
  $22.20 -> $25.58) of the $5 sub-cap.
  * **Re-run sanity:** C'@32 learned again in both seeds (22.0 / 23.1 deposits per env-update = 150x / 157x the
    ladder E) -> both transfer readings valid.
  * **(1) Transfer: PASSES at 64 and at 128.** Frozen C'@32 policies: 64 -> 1.67 / 1.16 deposits per env-rollout =
    **66.8x / 46.5x** the ladder E; 128 -> 0.34 / 0.19 = **36.6x / 20.3x**. Cam's prediction (64 yes, 128 probably)
    held at both. The compass-following policy learned at 32 generalises to maps the learner cannot learn on
    directly (C' fails at 64/128 from scratch). Per Cam's reading: **the path to the full world is a size curriculum
    (32 -> 64 -> 128), not learner surgery.** (Absolute rates fall with size, as they must: longer walks between
    deposits.)
  * **(2a) PopArt critic: FAIL as registered** -- Var(target - V) grew x4.44 and x20.52 from ramp to plateau at 32
    (bar <= 2x). **But the registered rule was mis-specified for this outcome (disclosed, verdict unchanged):** it
    assumed the artifact branch, where Var(target) falls; here Var(target) itself grew x4.27 and x34.52, so the
    residual grew *with* the target. Relative residual Var(resid)/Var(target): PopArt 0.88 -> 0.51 -> 0.58 and
    0.98 -> 0.53 -> 0.49 (updates 15 / 40 / 80), against the critic-fix re-run's 0.98 -> 0.37 -> **2.18 / 2.28** in
    the same stage. EV at plateau: PopArt **+0.43 / +0.42** (stable 0.26-0.51 from update 20 on) vs the fix's
    **-1.18 / -1.28**. Read, not registered: PopArt stops the divergence; the fix variant diverges.
  * **(2b) PopArt reach: EXTENDS to 64 as registered** (3.50x / 476.9x the ladder E). Disclosure: seed 0 barely
    clears (0.0875 per env-update, ~28 deposits in its window, EV ~0 -- it did not visibly learn); seed 1 learned
    (deposits 0 -> 6.1 -> 12.5 by updates 40 / 60 / 80, EV +0.73). One clear learner, one marginal.
  * Exploratory: PopArt at 32 delivered more than the fix variant (27.1 / 26.7 vs 22.0 / 23.1 per env-update).

- **Cam's ruling after stage 3 (2026-11-05).** Transfer at both sizes is "the result of the project so far":
  navigation is a general skill; the route to the full world is a size curriculum. **Critic:** the ramp-to-plateau
  residual rule is **retired on the record** (his mis-specification: it assumed target variance would shrink).
  PopArt is **accepted for testbed use** on the bar registered before that rule, EV >= 0.2 at updates 36-40:
  **0.549 / 0.472** at 32x32 (pass); plateau EV +0.42 / +0.43. **Disclosure:** the critic-fix variant ALSO passes
  that earlier bar at 32 in the same stage (0.558 / 0.591 at 36-40) before collapsing to -1.18 / -1.28 by update 80,
  so the 36-40 window alone does not separate the two critics; PopArt's stability through the plateau does. Both
  bars and both outcomes stated. Production landing waits for the curriculum work. **Order:** communication first
  (the thesis test, at 32x32 where navigation works), curriculum after. Stage sub-cap **$10** with M's knowledge.

- **REGISTRATION -- testbed communication experiment (before launch).** Code on branch `comm-testbed` (`b88fe7f`):
  `scripts/comm_testbed.py`; token policy in `rl_jax` (`token_logits`, token log-prob in the PPO ratio, token entropy
  bonus); analysis `scripts/analyze_comm.py` committed before results. Launcher plan `comm`.
  * **Launch gates (met):** `tests/test_token_policy_gradient.py` passes -- the shared, listener-dependent reward
    reaches `head_signal_slot*` through the real `ppo_loss` and raises the taken token's log-prob; mutant 1 (no token
    log-prob = today's design) and mutant 2 (shared reward zeroed) give exactly zero gradient; with the token
    log-prob removed from the real `ppo_loss` the main test fails. Suite 162 passed / 56 skipped on the branch.
  * **World:** 32x32, the four production hearths (8,8) (8,24) (24,8) (24,24), one active; Craft within radius 1
    of it = deposit (2.7 to the depositor, depositor respawns, active re-rolls uniformly among the other three; also
    every 256 steps without a deposit). Two agents per world, one shared network: sender (one-hot of the active
    hearth in its 4 need dims, `can_see_recipe` 1) and listener (those dims 0). All see the four `hearth_rel`
    vectors. Listener deposit also pays the sender 2.7 (shared reward). 32 worlds = 64 agents. **No local patch, so
    agents cannot see each other: spatial following is impossible by construction** (disclosed; the only route for
    information is the wire, or the listener's own memory of where it last deposited).
  * **Learner:** C' (GAE 0.95 + bootstrap), PopArt (beta 0.1, stats fitted from the first rollout), vf 0.5,
    production lr/clip/entropy coefficients, `freeze_comms=False`; token tau **0.08** (registered calibration rule:
    mean per-slot token entropy ~ half of ln 64 at the warm-start params -> 2.12 / 1.93 nats at tau 0.08), token
    entropy coefficient 0.01. **Warm start** from the stage-3 PopArt@32 policy of the same seed (all params;
    comms subtree there was never trained -- frozen at init). **Disclosure: the warm start does not transfer to the
    four-hearth observation** -- in a local 8-world smoke the warm-started agents made **0** craft attempts in 512
    steps (that policy was trained with one hearth in all four `hearth_rel` slots). It is a weak initialisation, not
    a ready compass; the ORACLE arm is what guards against an uninterpretable null.
  * **Arms and seeds:** COMM (wire live) seeds 0, 1; NOSIG (wire zeroed both ways) seeds 0, 1; ORACLE (wire zeroed,
    listener also gets the one-hot) **seed 0 only** -- cut from 2 per Cam's rule (oracle seeds first, never the
    primary), because the GPU timing probe measured **~44 s/update** (66 s first), so 6 runs x 150 updates would
    exceed $10. **N = 120 updates** per run. Estimated ~$8-8.5 incl. MI and swap eval. Spend baseline: metered
    $25.36; guard at $35.36.
  * **Apparatus (enforced in code, halts the run):** every 64 steps -- sender one-hot == active index; listener need
    dims exactly 0 (ORACLE: one-hot); broadcast == codebook[token] per slot; NOSIG/ORACLE wire input exactly 0; in the
    swap eval, the listener receives exactly the sender's T' broadcast.
  * **PRIMARY -- token swap (frozen final COMM params, per COMM seed):** 10 paired eval rollouts (same initial world
    and keys); in the swap arm the listener receives the token its sender would emit with the active index replaced
    by T' = (T + 1) mod 4 (sampled with the same key; the sender still acts on the true T). Delta_k = mean m.u toward
    hearth T' over listener moves (swap) - (control). **A seed passes iff mean Delta >= 0.05 and > 2 x SE across the
    10 pairs; the primary passes iff both COMM seeds pass.** Mirror (reported): m.u toward the true T falls under swap.
  * **APPARATUS ARM -- ORACLE:** listener correct-hearth fraction (deposits at the active hearth / Craft attempts
    within radius 1 of any hearth) over updates 111-120 **>= 0.50 with >= 50 attempts**; chance is 0.25. If it fails,
    a COMM null is uninterpretable.
  * **SECONDARY:** listener correct fraction and deposits per world, COMM vs NOSIG, last 10 updates (COMM > NOSIG
    outside seed spread); MI(token; active) per slot over the last 10 updates' sender steps, Miller-Madow corrected,
    vs a block-permutation null over rotation epochs (1,000 shuffles), pass = above the null's p99 in >= 1 slot --
    secondary only (high MI does not show the listener uses the token). NOSIG correct > 0.25 is logged as memory /
    search order (a listener that remembers its last deposit hearth can reach 1/3 with no information), not the
    channel.
  * **Attempt 1 cancelled -- infrastructure, not the experiment (app `ap-yIyojumc8GkNZuOBkQRNYJ`).** All 5 runs
    received a client-side cancellation at 19:48:39Z, at update 34 of 120 (~21 s/update, faster than the probe). No
    error, apparatus halt or traceback in any container; the spend guard had not fired (metered $27.05 vs $35.36);
    the Mac's sleep was blocked (caffeinate). Cause: with `modal run --detach`, only the last-triggered call survives
    a local disconnect, and the launcher's `starmap` ran from the local process. **The partial results were not
    analysed** (line counts only); they stay on the volume as `testbed/comm/comm_*_run_b88fe7f.jsonl`. Spend ~$1.7.
  * **Attempt 2 = the same registered run, unchanged design**, launched through `orchestrate_comm` -- a CPU
    function inside Modal that runs the starmap, so a local disconnect cannot cancel it. Output files carry an
    `_a2` suffix; combined results at `testbed/comm/comm_a2_results.jsonl`.

- **Testbed communication experiment -- results (attempt 2, app `ap-lFdp8aDMbEuGhU2bK7mD7s`, pin `b88fe7f`; raw
  `docs/data/testbed_comm_a2_2026-11-05.jsonl`, analysis `..._analysis.txt`).** All 5 runs on "NVIDIA A10G"; no
  apparatus halt. Stage spend ~**$7.8 of $10** (metered $25.36 -> $33.11-33.30; the summary moves between reads),
  incl. attempt 1's ~$1.7. About $6 remains under the overall cap.
  * **APPARATUS (ORACLE): PASS.** Listener correct-hearth fraction **0.771** (4,350 attempts) over updates 111-120,
    against chance 0.25; listener deposits 10.5 per world-update. A listener that is told the active hearth learns to
    pick it, so the comm null below is interpretable by the registered rule.
  * **PRIMARY (token swap): FAIL.** Delta = **-0.0026** (SE 0.0026) and **+0.0001** (SE 0.0001); bar >= 0.05 and
    > 2 SE in both seeds. Mirror toward the true T: +0.0021 / -0.0000. The swapped token differed from the true
    one on 15.8% / 9.4% of sender steps (31% at the warm start), so the sender's token depends on T a little, and
    the listener's movement does not respond to it.
  * **SECONDARY.** Listener correct fraction COMM 0.022 / 0.035 vs NOSIG 0.056 / 0.021 -- COMM not above NOSIG;
    listener deposits per world-update COMM 0.88 / 0.93 vs NOSIG 1.93 / 0.95. **MI(token; active): above the
    block-permutation null's p99 in every slot of both COMM seeds, but tiny** -- 0.0075 / 0.0071 / 0.0008 bits (seed
    0) and 0.0026 / 0.0017 / 0.0006 (seed 1), of a 2-bit ceiling. Tokens carry a trace of T that listeners do not use.
  * **Not registered -- the comm arms never produced a competent informed sender.** Informed senders, who see T in
    every arm, delivered 0.8 / 1.7 (COMM) and 6.6 / 0.5 (NOSIG) per world-update by update 120, against **11.0** for
    the ORACLE world (where both agents are informed); NOSIG0's sender was still climbing steeply (1.6 at 90 -> 6.6 at
    120). Uninformed listeners settled into **camping** -- parking at one hearth and spamming Craft until it becomes
    active (attempts grew to 1,100-1,700 per update, vs 449 for the ORACLE listener), which is why the per-attempt
    correct fraction sits far *below* 0.25 in COMM/NOSIG: that chance level does not apply to camping. Reading: the
    ORACLE check shows a listener *can* use T; it does not show that senders here ever became competent enough, or
    that listener deposits were frequent enough, for the shared reward to shape the token. With half of the shared
    network's experience uninformed, 120 updates from a non-transferring warm start was not enough for that. **The
    null is about this regime (weak senders, rare shared reward), not evidence that the channel cannot work.**
  * Interpretation per the registration: apparatus passes, primary fails, MI passes as secondary -> "information is
    (barely) in the tokens but listeners don't use it". Next step is Cam's call.

- **Cam's ruling (2026-11-07).** Record the null as **"this regime", not "channel impossible"**. His warm-start
  choice (the PopArt@32 policy) was wrong for the four-hearth world -- logged as his. Free analysis first; then one
  final run on the last ~$5, guard stopping at ~$1 remaining.
- **Free analysis ($0, attempt-2 logs) -- confirms the diagnosis, so the plan stands:**
  * **Senders camp too.** Craft attempts per sender deposit, updates 111-120: COMM 46.3 / 18.5, NOSIG 5.5 / 58.2,
    vs **ORACLE 1.3**. Camping bleeds through the shared network into the role that holds the answer.
  * **The shared reward fired rarely:** listener deposits per update across all 32 worlds, COMM0 0 / 7 / 28 / 31 / 19
    and COMM1 0 / 9 / 30 / 40 / 26 at updates 10 / 30 / 60 / 90 / 120 (~0.9 per world per rollout), vs ~335 for the
    ORACLE listener.
  * **Token entropy did not collapse** (3 slots summed; max 12.48): dips from ~6.4 to ~4 by updates 10-20, then
    recovers to 4.8-6.0 by 120, with the same shape in NOSIG and ORACLE, where tokens are unused. The 31% -> 9-16%
    fall in swap difference is the sender's tokens depending less on T, not tokens going constant. Per-slot entropy
    was not logged (sum only). The 0.01 token entropy bonus was already on and is kept (disclosed).

- **REGISTRATION -- final comm run (before launch).** Branch `comm-testbed` `ceadaf8`; plan `comm_final` via the
  in-Modal orchestrator; analysis `scripts/analyze_comm.py <final> <attempt-2 file>` committed before results.
  * **Unchanged from the registered comm design:** world, wire, shared reward, token policy (tau 0.08, entropy coef
    0.01), C' + PopArt, vf 0.5, apparatus halts, N = 120, primary (token swap, both COMM seeds, Delta >= 0.05 and
    > 2 SE over 10 pairs), secondaries (COMM vs NOSIG; MI vs its null).
  * **Changes (three, all Cam's):** (1) **everyone warm-started from ORACLE0's final params**
    (`comm_ORACLE0_run_b88fe7f_a2.msgpack`); local smoke: informed senders delivered 17 per world in update 1
    (ORACLE0's end rate 10.28) -- competent from the start. (2) **Camping closed** by a cost of **0.1 per Craft that
    is not a deposit** (any agent, anywhere): camping at the observed 18-58 attempts per deposit costs 1.8-5.8
    against the 2.7 deposit; ORACLE-style 1.3 attempts per deposit costs ~0.03. (3) **Auto-halt after update 2**
    (enforced in code, the run exits and the orchestrator stops the app): mean sender deposits per world over
    updates 1-2 **>= 7.2** (70% of ORACLE0's 10.28) **and** token entropy (summed) at update 2 **>= 3.0** (lowest
    seen in any run: 3.9).
  * **Camping check, registered:** NOSIG listener attempts per update over updates 111-120 **<= 870** (2x ORACLE0's
    435) -- "loophole closed". Attempts per listener deposit reported alongside.
  * **Arms:** COMM seeds 0, 1 (primary); NOSIG seed 0. ORACLE is not re-run: the apparatus result is attempt 2's
    ORACLE0 (0.771), whose final params are now everyone's starting point. Testbed agents still cannot see each other,
    so "following the sender" is impossible by construction; NOSIG measures what a listener gets with no channel.
  * **Wire check done before launch:** at ORACLE0's params, changing the received signal moves the listener's action
    logits by up to 0.23 (mean action-distribution TV 0.012; policy entropy 2.13) -- the channel input reaches the
    policy; the effect is small at the start.
  * Cost ~$3.7 (3 runs at ~$1.2). Spend baseline at launch recorded below; guard stops all testbed apps at baseline
    + $5.00 (~$1 of the overall cap left).

- **RESULTS -- final comm run (2026-11-07).** App `ap-EZ9CS4UzLQooSI96isTbsA`, all three runs completed (A10, ~74
  min each); data `docs/data/testbed_comm_final_2026-11-07.jsonl`, analysis output
  `docs/data/testbed_comm_final_2026-11-07_analysis.txt` (registered script, unmodified). Spend: metered $32.94 ->
  $37.61-37.78 (the figure moved between reads; **~$4.7-4.8, vs the $3.7 estimate** -- runs took ~74 min each); ~$2.2-2.4 of
  the $40 cap left; guard ($37.94)
  did not trip.
  * **Auto-halt: passed, run continued** (MEASURED from the train records; the halt record is written only on HALT):
    mean sender deposits/world over updates 1-2 = 14.6 (COMM0) / 14.5 (COMM1) / 15.0 (NOSIG0) vs the 7.2 bar; token
    entropy at update 2 = 5.37 / 5.36 / 5.37 vs the 3.0 floor.
  * **PRIMARY (token swap): FAIL.** Delta +0.0043 (SE 0.0026) COMM0, -0.0040 (SE 0.0026) COMM1; bar 0.05 and > 2 SE.
  * **Camping check: OPEN.** NOSIG0 listener attempts/update 941 (bar <= 870); COMM0 1,140, COMM1 1,057.
  * **Apparatus:** attempt 2's ORACLE0, 0.771 (registered; not re-run).
  * **Secondaries:** COMM listener correct 0.095 / 0.146 vs NOSIG 0.145 -- no COMM advantage. MI above the null in
    most slots but tiny: max 0.0038 bits (COMM1 slot 1).
  * **Not registered -- what the three changes did (MEASURED):**
    - *Competent senders: fixed.* Sender deposits/world 15.7 / 23.4 (COMM), 22.9 (NOSIG), updates 111-120, vs
      ORACLE0's 10.28 and attempt 2's 0.5-6.6.
    - *Shared-reward frequency: up ~5x but still a third of ORACLE.* Listener deposits 3.4-4.8 per world-update
      (~110-155 per update) vs attempt 2's ~20-40 and ORACLE0's ~335.
    - *Camping: changed form, not closed.* Attempts per listener deposit fell from attempt 2's 18-47 to **6.9-10.6**.
      Listeners now search the four hearths rather than park at one. At ~0.145 success per attempt a Craft returns ~0.39 gross against
      an expected cost of 0.085 (0.1 on the 85.5% that miss), so searching pays. **Registration error (mine):** the 0.1 cost
      was sized against the sender camping rates (18-58 attempts per deposit) and never checked against a searching
      listener. Break-even for blind search is p x 2.7 / (1 - p) ~ **0.45** at p = 0.145 (it would also tax the senders' misses).
    - *Senders' tokens stopped depending on T.* The swap changed the sender's token on only **3.2% (COMM0) / 7.5%
      (COMM1)** of sender steps at update 120. **Start point measured afterwards ($0, local CPU, `ceadaf8`, COMM seed 0, ORACLE0
      warm start + 1 update, 3 swap pairs): 16.0-16.4%.** So the tokens carried some T at the start and lost most of
      it under training with the live wire (one local seed; the swap there moved the listener by <= 0.0006). So the primary tested whether
      listeners react to a token that almost never changes; it is close to a no-op on the sender side.
    - *Token entropy collapsed only where the wire is live (n = 1 NOSIG control; COMM1's drop is in its last 20 updates).* COMM0 5.6 -> 2.0 -> 0.6 (updates 60 / 80 / 120); COMM1
      5.1 -> 2.6 (100 / 120); NOSIG0 rose 5.4 -> 6.05. The 0.01 bonus did not hold it; the floor was checked only at
      update 2. Cause CONJECTURED, not tested: with the wire live, tokens change what the partner does, so PPO has a
      gradient on them -- here it sharpened them toward a constant, not toward T.
  * **Reading, per the registration:** apparatus passes, primary fails, camping still open. In this testbed a
    language did not form even with competent senders. The sender side never put T into the tokens: the swap was
    near a no-op, MI is under 0.004 bits, and in the live-wire arms the tokens collapsed. MEASURED. The cause is
    CONJECTURED: blind search at ~7 attempts per deposit is good enough that the channel's marginal value per
    listener is small, and the sender's credit for it is diluted. Budget for this line is spent; next step is Cam's.

- **Cam's ruling (2026-11-07, after the final run).** The collapse is the finding: the classic emergent-comm
  chicken-and-egg (a listener that doesn't yet use tokens feels them as noise, so the cheapest group fix is for
  senders to go quiet). Known remedies: positive signalling / positive listening (Eccles et al. 2019). GPU spend
  stays $0: (1) free log check, (2) CPU toy with the same env and `ppo_loss` code, gate first, four arms, (3) a
  real-network confirmation is only drafted if a toy arm passes, for M to fund. M added $40 (new Modal account,
  profile `rapidcrisp`); Cam's allocation: comm confirmation sub-cap $12, size curriculum ~$10, ~$20 reserve for a
  3-seed replication. Toy on Modal allowed under $2 total if the laptop is too slow (it isn't: ~0.2 s/update).
- **Step 1 -- free check (MEASURED; does not contradict the hypothesis, so no early report).**
  * *Receiving range is undefined in the testbed:* the wire always reaches the partner (no distance), so the
    "entropy falls faster in range" half cannot be asked of these runs.
  * *Correction to the reading of the logs:* the logged `token_entropy` is the mean over BOTH roles (both agents
    broadcast and both hear). Per-role entropy and listener sensitivity were measured afterwards on the saved params
    (`scripts/probe_sensitivity.py` on `comm-toy`, 96-step COMM rollouts, 32 worlds, same obs builder): listener action TV
    when the heard broadcast is replaced by a random token triple -- **ORACLE0 (start) 0.0156 -> COMM0 0.0046,
    COMM1 0.0044 (NOSIG0 0.0060)**; replaced by the sender's T' broadcast -- 0.0022 -> 0.0005 / 0.0001 (0.0006).
    Token entropy sender / listener: start 5.32 / 5.41; COMM0 0.39 / 1.13; COMM1 2.51 / 2.83; NOSIG0 6.06 / 6.11.
    Sender token changes under T -> T': 18.1% -> 3.4% / 8.4% (NOSIG0 12.6%).
  * Reading: listeners never came to use the tokens (sensitivity did not grow; it fell ~3x, and ~2.6x in NOSIG
    too, so part of the fall is generic drift, n = 1 per arm). Both roles' tokens collapsed in COMM, not only the
    senders' -- consistent with a shared comms pathway. The prediction "sensitivity stays flat" is not met
    literally (it fell), but the contrary outcome (sensitivity growing, i.e. listeners starting to use tokens) did
    not occur. Note the magnitude: at the start, a random token moves the listener's action distribution by TV
    ~0.016 -- the "noise" the hypothesis needs is small.

- **REGISTRATION -- CPU comm toy (before any registered arm is run).** Code `scripts/comm_toy.py`, analysis
  `scripts/analyze_toy.py`, branch `comm-toy` (from `comm-testbed` `ceadaf8`) at **`74e4d33`**.
  * **Toy = the testbed with the size cut, the same mechanisms.** Identical: world rules (four production hearths,
    sender informed / listener not, one active hearth, Craft within radius 1 deposits 2.7, respawn away from
    hearths, re-roll on ANY deposit or after REROLL steps, shared 2.7 to the sender on a listener deposit), the
    symmetric one-step wire, the token policy (`token_logits`, tau 0.08, log-prob in the ratio, token entropy coef
    0.01), VQ (`vector_quantize_signals`, beta 0.25, vq_coef 0.1), **`rl_jax.ppo_loss` itself**, C' (`gae_bootstrap`,
    gamma 0.999, lambda 0.95), PopArt (beta 0.1), vf 0.5, ent 0.02, Adam 3e-4, the per-group gradient clip at 2.0,
    minibatch 512, one epoch, a production-shaped comms pathway (separate feed-forward branch from the obs ->
    per-slot heads -> codebooks, independent of the recurrent trunk), the /2 action-logit temperature, the paired
    token-swap eval with its apparatus check. Changed: G = 12, REROLL = 96 (256 x 12/32), T = 128, 64 worlds,
    MLP(64) + GRU(64) trunk, 59-dim obs, no alarm head, jitted `lax.scan` rollout. Craft cost **0.45** (Cam).
  * **Disclosed pre-registration changes and peeks:**
    (a) *v1 toy took z_e from the recurrent trunk; its ORACLE's token entropy collapsed to 0.35 with no wire at
    all* (real ORACLE: 5.3). Fixed by giving the toy production's separate comms pathway (`network_jax`
    `gwt_comms_1 -> head_signal_slot*`); the v2 ORACLE holds ~9.5. (b) *Wire-input weight rescale 0.06* (`--wire-init-scale`):
    at ORACLE0-toy params the listener's random-token sensitivity was TV 0.31 vs **0.0156** on the real network (the
    toy feeds the raw wire to a dense layer whose wire rows never trained). The wire was zero all through ORACLE
    training, so those rows of `enc_0` and `gwt_comms_1` are still at init (verified bit-identical); after the warm
    start they are multiplied by 0.06, giving TV 0.0156. The input stays at full scale -- a first try used an input
    gain instead, rejected before any run because under Adam it would also slow the listener's learning ~16x
    (a bias toward the null being tested). Calibrated on the pre-treatment state only.
    (c) *An unregistered 300-update baseline COMM seed 0* ran by mistake (a flag was added before its code); only its
    last two swap pairs were seen (Delta ~ +0.003 / -0.005, at gain 1). (d) A 12-update plumbing test of NOSIG, BASE and
    BOTH (seed 0, unscaled wire, coefficients 1.0 / 0.19 from an earlier calibration) to check the analysis script;
    its full output was seen: precondition 12.44 vs NOSIG 1.61, NOSIG break-even cost 0.40, BOTH MI ~0.5 bits per
    slot, swap-diff 0.69 and 6,420 listener Crafts per update (PL drove Craft spam), primary Delta +0.002 (BASE) /
    +0.005 (BOTH) on 2 pairs. Nothing was changed because of it; the coefficients follow the rule.
  * **Warm start (as the final run):** ORACLE-toy seed 0, 1,500 updates from scratch (`toy_out/ORACLE0.msgpack`, committed, sha256
    `f5d01c19...4c32bdd84`;
    end: listener 12.4 deposits/world-update, 98% correct). Every arm below starts from it.
  * **Runs:** 1,000 updates each, seeds 0/1/2, all from ORACLE0-toy, `--wire-init-scale 0.06`, cost 0.45: NOSIG;
    BASE (COMM, no aux); PS (`--ps 1.99`); PL (`--pl 0.114`); BOTH (both). Coefficients by the registered rule c = 0.25 x
    ||grad PPO|| / ||grad aux|| (full-parameter norms, pre-clip, 4 minibatches of one COMM rollout at ORACLE0-toy,
    rescaled), computed once, not tuned. PS = sum over slots of the batch MI(token; active hearth) on sender rows;
    PL = mean L1 between the listener's action distribution with the wire and (stop-grad) with the wire zeroed.
    Self-test (`--selftest`): each term's gradient reaches only its own role's rows, one ascent step raises it, and
    mutants without the role mask are caught.
  * **(1) Precondition, channel value:** ORACLE0-toy listener deposits/world (its last 50 updates) >= 2 x the
    NOSIG mean (last 50). Fail -> stop, toy unrepresentative. Reported alongside, never retuned: NOSIG listener
    p(deposit | Craft) and the break-even cost p x 2.7 / (1 - p) vs 0.45.
  * **(2) Gate, live-wire collapse:** sender token entropy, mean of the last 50 updates / mean of updates 1-5:
    BASE seed-mean <= 0.6, every NOSIG seed >= 0.9, and max BASE < min NOSIG (real: COMM 0.11 / 0.49, NOSIG 1.12).
    Fail -> stop, toy unrepresentative; PS/PL/BOTH are not run.
  * **(3) Primary, per arm:** token swap, 10 pairs, Delta = mu(T')_swap - mu(T')_control; a seed passes with
    Delta >= 0.05 and > 2 SE; **an arm passes iff all 3 seeds pass.** Prediction (Cam): BASE fails, BOTH passes.
  * **Run order:** NOSIG and BASE (3 seeds each) first, then `analyze_toy.py` on those six; PS / PL / BOTH run only
    if the precondition and the gate both pass.
  * **Secondaries:** listener deposits vs NOSIG; MI(token; T) per slot vs a plain permutation null (the real
    testbed permuted in epoch blocks; a plain null overstates significance when tokens are autocorrelated); per-role token
    entropy, swap-diff and listener sensitivity over training (the step-1 questions, now per update).
  * **Next step:** an arm passes -> draft one real-network confirmation run with a cost estimate for M (nothing
    launches). None passes -> report.

- **RESULTS -- CPU comm toy (2026-11-07).** `comm-toy` `74e4d33`, 15 runs of 1,000 updates (laptop CPU, ~30 min
  total, $0). Analysis output `docs/data/toy_comm_2026-11-07_analysis.txt` (registered script, unmodified).
  * **(1) Precondition: PASS.** ORACLE-toy listener 12.44 deposits/world-update vs NOSIG 2.07 (6x).
    Craft-cost check: NOSIG listener p(deposit | Craft) 0.19-0.22 -> break-even 0.61-0.75, so **blind search is
    still profitable at 0.45** in the toy (reported, not retuned).
  * **(2) Gate: PASS -- the toy reproduces the live-wire collapse.** Sender token entropy last-50 / first-5: BASE
    0.979 / 0.013 / 0.000 (mean 0.33), NOSIG 1.011 / 1.010 / 1.024. Two of three BASE seeds collapsed to ~0;
    NOSIG never did (real: COMM 0.11 / 0.49, NOSIG 1.12). The third clause held narrowly (0.979 < 1.010).
  * **(3) Primary: no arm passes; the prediction "BOTH passes" fails.** Delta per seed:
    BASE +0.002 / +0.017 / +0.000 (all fail); **PS -0.029 / +0.410 / +0.342 (2 of 3 pass, SE <= 0.006)**;
    PL -0.000 / -0.000 / +0.000 (all fail); BOTH +0.077 / -0.098 / +0.168 (2 of 3 pass).
  * **Secondaries (MEASURED):**
    - *PS made the tokens a code for T, and listeners came to use it.* MI(token; T) 1.96-1.99 bits per slot of a
      2-bit ceiling, in every PS seed (BASE <= 0.024). The swap changed the sender's token on 100% of steps.
      **PS listeners delivered 4.8 / 5.2 / 7.0 per world-update vs NOSIG 1.5-2.9 (BASE 0.3-2.1).**
    - *Not registered -- listener deposits under the swap* (same pairs; `scripts/swap_detail.py` on the run data
      committed at `comm-toy` `451e6f1`, `toy_out/*.jsonl.gz`): PS 4.53 -> 1.30, 4.97 -> 1.63,
      6.77 -> 2.33; BOTH 2.21 -> 0.83, 0.06 -> 0.04, 2.06 -> 0.74; **BASE0 1.94 -> 1.95**. A wrong token costs PS
      listeners 65-71% of their deposits in all three seeds, including PS0, whose registered movement metric was
      negative (Delta -0.029; it also moves *away* from the true hearth under the swap, mu(T) -0.09 -> -0.15). So PS0's
      listeners do depend on the token; the registered movement-toward-T' metric doesn't capture how. That reading
      is CONJECTURED.
    - *PS listener competence came late.* Deposits rose mainly after update ~400-700 (PS0: 0.06 at 700 -> 5.0 at
      1,000) and were still climbing at 1,000.
    - *PL alone broke listeners.* Deposits 0.01-0.07, Crafts ~20-44 per update; MI <= 0.008. Rewarding "act
      differently from the no-wire policy" without an informative sender pushed listeners off the task. In BOTH it
      cost one seed (BOTH1: deposits 0.04).
    - *Step 1 inside the toy:* in BASE the listener's random-token sensitivity rose (0.014-0.023 -> 0.06-0.21),
      unlike the real end params (fell). In the collapsed seeds this probe is off-distribution (a listener that
      only ever hears one token is tested on others), so it is not a clean answer to "flat or growing".
  * **Reading per the registration:** no arm passes -> report; nothing is drafted for funding. Not registered:
    positive signalling alone produced an informative code and listeners that measurably depend on it, in 3/3
    seeds by deposits-under-swap, and 2/3 by the registered movement metric. Next step is Cam's.

- **Cam's ruling (2026-11-07, on the toy).** Registered primary failed (2/3); PL dropped; step 1 inconclusive rather
  than contradicting. On the record: **PS makes the channel informative by design; listener use is the emergent
  part, and that is the claim tested.** Before registering, check that swaps are in-distribution; then a free,
  registered replication on fresh seeds.
- **Swap check (exploratory, on the original PS seeds 0-2; `comm-toy` `a23eaf0`).** The three runs were regenerated
  with identical arguments plus saved params; every training record is identical to the originals. New
  *empirical* swap mode: the listener hears a full token triple drawn from the triples its sender actually emitted
  for T' in the paired control rollout. Results (u1,000; empirical / constructed):
  * The constructed swap tokens were already in-distribution: **98.5-99.5%** had been emitted by the sender for T'
    in the control rollout. (The comms pathway is feed-forward from the obs and slots are sampled independently in
    normal play too, so the constructed token is a normal T' token.) Off-distribution confusion is ruled out.
  * Deposit drop under the swap: PS0 72% / 71%, PS1 67% / 67%, PS2 66% / 66%. Delta (movement toward T'): PS0
    -0.026 / -0.029, PS1 +0.413 / +0.410, PS2 +0.348 / +0.342. **Unchanged, so the swap construction does not
    explain PS0's negative score.**
  * *Redirection (new measure, crafts at the T' hearth per world, empirical swap):* PS0 0.38 -> 14.16, PS1 0.56
    -> 14.40, PS2 0.74 -> 20.27. **PS0's listeners do go to the hearth the swapped token names (37x).** The
    movement-toward-T' score misreads how PS0's listeners travel: its control score toward the TRUE hearth is
    also negative (-0.09) while it delivers 4.5 per world. Why is CONJECTURED (not investigated).

- **REGISTRATION -- PS replication in the CPU toy (before any run).** `comm-toy` `a23eaf0`; launcher `run_rep.sh`;
  analysis `scripts/analyze_rep.py`. Everything as the toy registration (ORACLE0-toy warm start sha256
  `f5d01c19...4c32bdd84`, `--wire-init-scale 0.06`, cost 0.45) except as listed.
  * **Arms, all seeds fresh, 2,000 updates, eval (MI + 10 swap pairs, both swap modes) at 1,000 and 2,000:**
    PS x5 (seeds 10-14, c_ps 1.99); PS-anneal x3 (seeds 20-22, c_ps 1.99 decaying linearly to 0 over updates
    1,000-1,500, then 0 to 2,000); NOSIG x3 (seeds 30-32). **c_ps is 1.99, not Cam's 1.96:** 1.96 was the
    pre-rescale calibration; 1.99 is what the toy ran. Craft cost stays 0.45. Known limit: blind search still
    pays (break-even 0.61-0.75 in the toy), logged again per NOSIG seed.
  * **P1 reliance (all 5 PS seeds), empirical swap, update 2,000:** listener deposits drop >= 30% (1 - mean
    swap / mean control over the pairs). **Added before running:** the paired drop must be > 2 SE, and the seed's
    listener deposits (last 50 training updates) must be >= the NOSIG mean. Reliance on a channel that leaves
    listeners worse off than no channel is not success.
  * **P2 meaning (>= 4 of 5 PS seeds), empirical swap, update 2,000:** Delta = mu(T')_swap - mu(T')_control
    >= +0.05 and > 2 SE. **Disclosed: this threshold was set after a 2/3 outcome; the seeds are fresh.**
  * **P3 self-sustaining (>= 2 of 3 anneal seeds), update 2,000 (500 updates at c_ps = 0):** MI >= 1.0 bit
    (Miller-Madow, mean over the 3 slots, sender tokens of updates 1,991-2,000) **and** P1's reliance conditions.
  * **Secondary (registered, not decisive):** redirection. Listener Crafts at the T' hearth under the empirical
    swap >= 2x control with a paired increase > 2 SE, in >= 4 of 5 PS seeds. Added because the movement score
    misread PS0. If P2 fails and redirection passes, that is reported as such and the call is Cam's. Also
    reported: the constructed-swap results, token membership rates, everything at update 1,000, per-role entropy,
    and the NOSIG break-even.
  * **Decision:** P1 and P2 pass -> draft the real-network PS confirmation (sub-cap $12; set up the `rapidcrisp`
    volume and params first; an MI floor instead of the token-entropy halt; P3 decides whether it has an anneal
    arm). Nothing launches until it is approved. Otherwise -> report; nothing goes to Modal.

- **RESULTS -- PS replication in the CPU toy (2026-11-07).** 11 runs, laptop CPU, 39 min, $0. Data and output:
  `comm-toy` `4a01b75` (`toy_out/rep_*.jsonl.gz`); `docs/data/toy_ps_replication_2026-11-07_analysis.txt`
  (registered script, unmodified). All swap tokens in-distribution (empirical mode by construction; constructed
  97.9-100% seen for T').
  * **P1 reliance: PASS (5/5).** Deposit drop under the empirical swap 70 / 91 / 70 / 67 / 88% (paired SE <= 0.15);
    PS listeners 8.2 / 9.1 / 8.8 / 7.7 / 9.2 deposits per world-update vs NOSIG 2.10 (0.41 / 3.12 / 2.76)
    and ORACLE-toy 12.4.
  * **P2 meaning: PASS (4/5).** Delta +0.363 / +0.249 / +0.234 / +0.385 / **-0.119** (s14), SE <= 0.008.
    (Threshold set after the 2/3 outcome; fresh seeds.)
  * **P3 self-sustaining: PASS (3/3).** After 500 updates at c_ps = 0: MI 1.99 / 1.94 / 2.00 bits; drop 67 / 72 /
    74%; listeners 7.8 / 5.8 / 9.0. The anneal seeds' Delta was +0.49 / -0.01 / +0.58 (not part of P3).
  * **Secondary redirection: 5/5.** Crafts at the swapped hearth T', control -> swap, per world: 0.59 -> 22.1,
    0.33 -> 49.7, 0.85 -> 22.7, 0.44 -> 22.1, **0.41 -> 5.7 (s14)**. So s14's listeners also go where the
    swapped token points, as PS0 did, while its movement score is negative. This is the second instance of the
    movement metric misreading a seed that redirects.
  * At update 1,000 (pre-anneal / mid-run), every PS and PSA seed already showed a 59-72% drop; MI was >= 1.97
    bits. Between 1,000 and 2,000 listener deposits kept rising (PS 2.3-6.9 -> 7.5-8.7).
  * Known limit, logged: blind search still pays at 0.45 (NOSIG break-even 0.25 / 0.69 / 0.66).
  * **Reading (scope).** PS makes the code informative by design, and that part is not a finding. The emergent and
    replicated part: listeners came to depend on the code and to act on its meaning, in 5/5 fresh seeds. With the
    nudge removed, **the code persisted (3/3, MEASURED)**. That the shared reward *maintains* it through listener use
    is CONJECTURED: sender token entropy ended at 0.00-0.14 (tau 0.08), and a saturated policy gets almost no
    gradient, so it could persist from inertia alone. A wire-off control tests this (below). This is the CPU toy,
    not the real network.
  * **Exploratory control (not registered; `comm-toy` `bfedcf2`): PS-anneal with the wire OFF** (NOSIG arm, same
    schedule, seeds 40-42). MI at update 1,000 -> 2,000: 2.000 -> 1.970, 1.997 -> 1.995, 2.000 -> 2.000. The
    wire-on PSA seeds went 1.985 -> 1.991, 1.979 -> 1.941, 1.986 -> 1.999. **The code persists just as well with no
    listener**, even though those senders' token entropy stayed ~2.2 rather than saturating. So in this toy nothing
    erodes an unused code within 500 updates: **P3, as registered, measured inertia, not self-maintenance.** The
    "self-sustaining" question stays open. A test that can tell use from inertia needs a force that erodes unused
    codes, e.g. a perturbation after the anneal, or a token-entropy pressure under which the wire-off code decays.
    The question is whether the wire-on code resists it.
  * **Decision per the registration:** P1 and P2 pass -> draft the real-network PS confirmation for M; P3 passed ->
    it includes an anneal arm. Draft: `docs/drafts/2026-11-07_real_network_PS_confirmation_DRAFT.md`. Nothing
    launches until it is approved.

- **Cam's ruling (2026-11-08, on the replication).** *P3 passed as registered; its interpretation is retracted:
  the persistence is inertia, not self-maintenance* (the wire-off control). **Reliance 5/5 and meaning 4/5 on fresh
  seeds stand as the first replicated channel result** (CPU toy; scope line as written above). M's roadmap
  ("THRONG North Star Roadmap", Nov 8) now holds the plan, caps and standing rules. Where its next-30-days list and
  this ruling differ, the ruling is newer and is followed, flagged to M: the roadmap says craft cost >= break-even,
  swap primary, and an anneal arm if P3 passed.
  1. **Paid run: redirection is the meaning primary, movement secondary.** Registered before launch, with the
     power argument (movement misread 2 of 8 toy seeds that redirected; ~0.75^2 = 56% joint pass on two seeds) and
     the disclosure that it was chosen after the toy.
  2. **Craft cost stays 0.1**, so PS is the only difference from the final run's COMM and NOSIG.
  3. **Step 0, jit-port of the testbed rollout: approved.** Equivalence bar fixed before running, then one timing
     run (<= $0.50); if the speed-up is under ~5x, report before drafting arms.
  4. **Erosion test (CPU toy, free, in parallel).** An eroding force, calibrated on separate seeds until a wire-off
     code falls below 0.5 bits within 500 updates; then register wire-on vs wire-off, 3 fresh seeds each. Pass:
     wire-on keeps MI >= 1 bit and reliance in >= 2 of 3 while wire-off decays. It does not block the confirmation.
  * Order: step 0 and erosion in parallel; the confirmation on `rapidcrisp` (cap $15) once step 0 passes.

- **REGISTRATION -- step 0 equivalence bar (fixed before any comparison is run).** Branch `comm-jit` (from
  `comm-testbed` `ceadaf8`).
  * **Reference:** `scripts/comm_testbed.py`'s Python-loop rollout, unchanged except a test-only `--dump-equiv`
    hook. The hook injects the draws the Python world takes from numpy, from a file: initial positions and active
    hearths; per step, the (N, 2) respawn candidates and the re-roll increments, used in world-index order. The
    injected draws have the same distributions: uniform over cells not within radius 1 of a hearth, and uniform
    on {1, 2, 3}. It dumps one 512-step rollout: obs, actions, rewards, tokens, log-probs.
  * **Candidate:** `scripts/comm_testbed_jit.py`, the same world as a `lax.scan`, consuming the same injected
    draws and the same JAX key schedule (`key, k_obs, k_act = split(key, 3)` per step).
  * **Setup:** COMM arm (wire live), ORACLE0 params, 32 worlds, 32x32, CPU, **3 fixed seeds (0, 1, 2), 512 steps
    each.**
  * **PASS (exact):** actions, rewards and positions identical at every step for all 64 agents, all 3 seeds. Obs
    and tokens also compared.
  * **Pre-stated float fallback** (used only if exact identity fails, and only for float-level causes): (a) with
    the dynamics teacher-forced (the jit step fed the reference's state, actions and draws), positions, rewards,
    active hearths and the wire must match exactly for all 512 steps; and (b) at every step, the action logits
    on identical inputs must agree within max |d logit| <= 1e-4 (float32); and (c) the first free-running
    divergence must occur at a step whose obs are bitwise identical, with |d logit| <= 1e-4. Anything else fails.
  * **Then one timing run on `rapidcrisp` (<= $0.50):** a few COMM updates at the real settings. Speed-up vs the
    measured ~37 s/update on an A10. Under ~5x -> report before drafting arms.

- **Cam's rulings (2026-11-08, second set).**
  1. **Update-2 halt is an apparatus check only.** Halt if the PS gradient does not reach the message head, if
     messages are constant, or if the listener's wire input is zero. No MI criterion at update 2.
  2. **MI floor from a CPU smoke** of the real network (free, or Modal CPU within the $2 toy allowance), PS arm,
     2 seeds. Pick a check update K <= 25% of the run where both seeds are clearly above the shuffled-token null.
     Floor = half the lower seed's MI at K, and above the null's p95. K and the floor are registered before the
     paid launch. If MI is still near zero at the last smoke update, report it and don't guess.
  3. **Erosion calibration:** the weakest setting that gets wire-off MI under 0.5 bits within 500 updates **on
     every calibration seed**. If none does, report it; the window is not extended.
  4. **Modal for the toy approved:** the timing run plus the 6 erosion runs in parallel CPU containers, within the
     $2 toy cap. The two jobs already running finish on the laptop.
  5. The roadmap is the reference: saved as `docs/THRONG_North_Star_Roadmap.md`. The copy received still says
     "halt at update 2 on an MI floor"; that line is marked superseded by ruling 1 until the updated file arrives.

- **Erosion calibration, force 1: token-entropy bonus -- FAILED to calibrate (reported per ruling 3; the window is
  not extended).** `comm-toy` `75323c4`, wire off (NOSIG), PS 1.99 annealed over 1,000-1,500, the bonus switched on
  from 1,500, one calibration seed per strength (50-53). Wire-off MI at update 2,000 (plug-in, mean over slots):
  0.03 -> 1.84 bits, 0.1 -> 1.42, 0.3 -> 1.44, **1.0 -> 1.49** (it plateaued at ~1.5 from update ~1,600). None got
  below 0.5 bit. Sender token entropy rose to ~5 at the strong settings while MI held at ~1.5: senders spread over
  more tokens *within* each hearth's own set, so the code stays informative. A token-entropy bonus does not erode
  the mapping.
  * **Exploratory, not registered: force 2, decoupled weight decay on the message head** (`gwt_comms_1`,
    `head_signal_slot*`; the other force Cam named), the same schedule, 4 strengths (1e-4, 3e-4, 1e-3, 3e-3 per
    minibatch step) x 2 calibration seeds (60, 61), on Modal CPU within the $2 toy allowance. Reported to Cam with
    the step-0 timing. Nothing is registered with it until he rules.

- **RESULTS -- step 0 (2026-11-08).**
  * **Equivalence: PASS (exact), on the pinned commit `comm-jit` `1d3b951`**
    (`docs/data/step0_equiv_final_1d3b951.txt`): actions, rewards and positions are bitwise identical at all 512
    steps for all 64 agents on seeds 0/1/2, and so are obs, tokens and log-probs (558-585 deposits per rollout,
    so deposits, respawns and re-rolls were exercised). *Disclosed: the first attempt failed the registered bar on
    rewards only* -- 1 float32 ulp (2.4e-7) at 2-3 steps per seed, because the reference sums rewards in float64
    and casts while the port summed in float32. The candidate was fixed to reproduce the reference's float64
    rounding (a reward table built in float64 in the reference's order); the bar was not changed. Attempt files
    are on the branch (`docs/step0_equiv_attempt{1,2}.txt`).
  * **Timing: speed-up ~1.07x -- under the ~5x gate, so this is reported before any arms are drafted.** A10, COMM,
    6 updates: **34.3-34.9 s per update** (update 1: 63 s, with compile), vs ~37 s for the Python path. Split:
    **rollout 8.3 s, PPO update 25.5 s**. The jit port removed most of the rollout cost, but the rollout was never
    the bottleneck: `ppo_update` (64 minibatches of 512, each copied host->device, with per-minibatch host syncs
    for metrics) is ~74% of the update. Apparatus check at update 2 passed (496 distinct sender messages,
    listener wire nonzero 100%).
  * **Spend on `rapidcrisp`: $1.16 metered.** That is the timing run (~$0.2) plus **~$0.9 wasted** on the
    exploratory weight-decay calibration on Modal CPU: 8 containers x 8 cores cost ~$0.20/min, far above my
    estimate. I stopped it before it finished (it writes results only at the end); nothing was returned. It now
    runs on the laptop for free. Toy allowance left: ~$1.

- **Erosion calibration, force 2: weight decay on the message head -- calibrated (exploratory; awaiting Cam's
  ruling before registration).** `comm-toy` `f005323`, laptop, wire off, the same schedule (PS 1.99 annealed over
  1,000-1,500; decay from 1,500), calibration seeds 60/61. Wire-off MI at update 2,000 (plug-in, mean over
  slots), and the first update below 0.5 bit:
  1e-4: 0.71 (never) / 0.05 (1,896); **3e-4: 0.15 (1,788) / 0.01 (1,700)**; 1e-3: 0.003 (1,594) / 0.004 (1,584);
  3e-3: 0.002 (1,536) / 0.003 (1,538). **The weakest strength that gets every calibration seed below 0.5 bit within
  500 updates is 3e-4** (per minibatch step, decoupled, on `gwt_comms_1` + `head_signal_slot*`). Senders' own
  deposits are unaffected (18.6-21.2 per world), so it erodes the code, not the agents.

- **Cam's rulings (2026-11-08, third set; replaces the second set's open items).**
  1. **Laptop:** M needs it. Only code, unit tests and checks under ~10 min run there; everything else on Modal.
  2. **Equivalence accepted** (`1d3b951`; the candidate was fixed, the bar was unchanged).
  3. **Speed, option 1 first:** scan the minibatches in one jitted call, with the batch on the GPU and the same
     minibatch permutation. The bar is registered first. Check and timing on Modal GPU (<= $0.30), on A10 and L4;
     use whichever is cheaper per update. Under 1.3x, or failing the bar -> stop and run at current speed.
  4. **Smoke on Modal GPU** (cheaper per update than CPU for the real network): PS arm, 2 fresh seeds the
     confirmation won't use, <= 120 updates each, spend guard $4. It sets K, the MI floor (as ruled) and N = 2x
     the update where MI levels off. If MI hasn't risen or levelled off by 120: send the curve; Cam sets N. All
     three are registered before the paid launch.
  5. **Confirmation:** PS x2 + a fresh NOSIG at matched length N; cap $15; spend guard; apparatus halt at update
     2; MI halt at K.
  6. **Erosion: weight decay 3e-4 on the message head approved.** Registered as proposed, plus never-used seeds
     and a manipulation check: if wire-off is not below 0.5 bit at update 2,000 in >= 2 of 3 seeds, the test is
     uninformative (reported; neither pass nor fail). The 6 runs go on Modal CPU.
  7. **Modal CPU sizing:** list ~$0.047 per core-hour + ~$0.008 per GiB-hour, so 2 cores / 4 GiB is ~$0.13/h.
     cpu and memory are set explicitly; one container is probed for 2 min; fan out only if it costs < $0.01/min
     per container. **The toy/CPU allowance is now $4 total, including the ~$0.9 spent.**
  * Roadmap re-saved (`docs/THRONG_North_Star_Roadmap.md`). Its item 4 still says the smoke is an "overnight CPU
    smoke run"; ruling 4 (Modal GPU) is newer and is followed.

- **REGISTRATION -- erosion test, CPU toy (before any run).** `comm-toy` `8469c90`; analysis
  `scripts/analyze_erosion.py`; launcher `scripts/modal_comm_toy.py` (plans `erosion_probe` + `erosion_rest`;
  Modal CPU, 2 cores / 4 GiB per container).
  * **Arms (never-used seeds), 2,000 updates, everything as the replication's PS-anneal arm** (ORACLE0-toy warm
    start, `--wire-init-scale 0.06`, cost 0.45, c_ps 1.99 annealed linearly to 0 over 1,000-1,500) **plus weight
    decay 3e-4 per minibatch step on the message head (`gwt_comms_1`, `head_signal_slot*`) from update 1,500**:
    wire ON (COMM) seeds 70/71/72; wire OFF (NOSIG) seeds 80/81/82. Eval (MI + 10 swap pairs) at 1,500 and 2,000.
  * **Manipulation check:** wire-off MI (Miller-Madow, mean over slots, updates 1,991-2,000) < 0.5 bit in >= 2 of 3
    seeds. If not, the test is **uninformative** (reported; neither pass nor fail).
  * **Wire-on seed passes:** MI >= 1.0 bit at 2,000 **and** reliance at 2,000. Reliance means the empirical-swap
    deposit drop is >= 30%, the paired drop is > 2 SE, and the listener's deposits over the last 50 updates are >=
    the wire-off arm's mean (the matched no-channel level).
  * **PASS:** the manipulation check holds and >= 2 of 3 wire-on seeds pass. **FAIL:** it holds and < 2 pass.
  * **Spend:** within the $4 toy/CPU allowance (~$0.9 spent). Probe one container for >= 2 min and fan out only if
    it costs < $0.01/min (list estimate for 2 cores / 4 GiB: ~$0.002/min).

- **Speed option 1 (scanned PPO update) -- on hold for Cam's bar definition (not run on GPU).** A dev check on the
  laptop (CPU, 2 worlds, PS on) gave identical minibatch order but a **max per-leaf relative parameter
  difference of 9.7e-4** (> 1e-5). Diagnosis: the largest absolute difference anywhere is 2.9e-6 (~1% of one lr
  step), in the attention key biases. Those biases get exactly zero true gradient (a constant added to every key
  leaves the softmax unchanged), so they receive only float noise, which Adam's first step amplifies to about ±lr.
  Their values are tiny, so the per-leaf ratio inflates. Relative to the whole parameter vector the difference is
  ~3e-6. Cam's bar does not say per-leaf or global; choosing the passing definition after seeing this would be
  post hoc, so it goes to Cam. Per his rule the confirmation path runs at current speed.
- **c_ps for the real network = 2.015** (registered rule: 0.25 x |grad PPO| / |grad PS| at ORACLE0, 4 minibatches of
  one COMM rollout, seed 0; ratios 6.61 / 8.86 / 8.86 / 7.92). Laptop, 8.8 min.
- **Smoke (Cam ruling 4), launched:** `comm-jit` `0a82f1e`, Modal GPU, PS arm (c_ps 2.015), seeds 20 / 21 (the
  confirmation will not use them), 120 updates each, loop PPO path, update-2 apparatus halt, per-update MI with a
  block-permutation null (p95). K, the floor and N are derived from it by Cam's rule and registered before the
  paid launch. Spend guard: the smoke's  on top of the toy allowance.

- **Cam's ruling (2026-11-08): the scanned-PPO bar was ill-posed** (a per-leaf relative bar blows up on near-zero
  parameters; Adam's first step turns float noise into ±lr steps on zero-gradient leaves). The hold was right.
  Guard reset to the approved sum: meter $1.31 + smoke $4 + remaining toy allowance ~$3.0 + speed check $0.30 =
  **$8.60** (a runaway guard; each job keeps its own allowance). c_ps 2.015 accepted. The user reports $41 left on
  the account.
- **REGISTRATION -- scanned-PPO equivalence bar (before the GPU check).** `comm-jit` (commit recorded with the
  results), `--check-ppo-scan`, COMM, ORACLE0 params, **PS on (c_ps 2.015) so mutant (iii) is meaningful, fresh
  seeds 40/41/42**; the loop path is the reference, and both run on the same GPU in one process from identical
  params / optimizer state / PopArt state / key.
  * (a) per-minibatch total losses equal within 1e-6 relative;
  * (b) per-minibatch gradients (post-clip, as applied), every leaf, elementwise
    |dg| <= 1e-4 |g_ref| + 1e-6 max|g_ref| (the max taken per leaf per minibatch);
  * (c) parameters after the full update, every element, |dtheta| <= 0.05 lr (lr 3e-4 -> 1.5e-5);
  * identical minibatch permutation;
  * (d) mutants must each fail at least one of (a)-(c): (i) one minibatch dropped, (ii) minibatch order shuffled,
    (iii) PS term zeroed. If any mutant passes, the bar is too loose and that is reported.
  * No leaves are excluded. PASS = all three seeds pass (a)-(c) and every mutant fails on every seed.
  * Timing, each GPU type in a single container: A10G (scan) and L4 (loop and scan), 3 updates each, compile
    excluded (update 1 dropped). The scan path is used only if it passes **and** is >= 1.3x faster; the GPU is
    whichever is cheaper per update at list price. Budget <= $0.30.
  * Disclosed: two laptop dev checks. The first (per-leaf ratio, before this bar) is reported above. The second,
    of this bar's code, hit the 10-minute laptop limit and was killed with no output.

- **RESULTS -- erosion test (2026-11-08): FAIL as registered.** Modal CPU (6 x 2-core containers, ~$0.005/min
  each), data `comm-toy` `5d35f22`, output `docs/data/toy_erosion_2026-11-08_analysis.txt` (registered script,
  unmodified).
  * **Manipulation check holds (3/3):** wire-off MI at 2,000 = 0.099 / 0.009 / 0.002 bit (2.0 at 1,500).
  * **Wire-on: 0/3 pass.** MI at 2,000 = 0.000 / 0.001 / 0.000. The plug-in trace reaches ~0 by update 1,750,
    *as fast as or faster than wire-off*. Reliance at 2,000 is gone (deposit drop +4% / -0% / -2%).
  * **Not registered (the registered eval at 1,500, before erosion), so the test is valid:** the wire-on channel
    was in use when the decay began. Deposit drop under the empirical swap 72 / 67 / 74%; crafts at the swapped
    hearth 0.8 -> 19.0, 1.1 -> 16.2, 0.6 -> 21.4. Listener deposits fell from 6.5 / 3.9 / 8.9 (updates 1,400-1,500)
    to 3.5 / 1.0 / 4.0 (1,900-2,000), back to about the no-channel level (wire-off 2.6).
  * **Reading.** MEASURED: in the toy, listeners' use of the code gives it no measurable protection against an
    eroding force that kills an unused code within 500 updates. The code was worth ~3-5 deposits per
    world-update to the listener, yet it decayed as fast as an unused one. CONJECTURED (not tested): the reward
    gradient reaching the sender's message head through the token log-prob is weak next to the decay. The sender's
    advantage is dominated by its own deposits, so the shared-reward credit for any one token is diluted. Rung 3
    ("the code survives without the nudge") is not shown; the anneal arm stays out of the paid run. The
    confirmation is not blocked (Cam).

- **Cam's ruling (2026-11-08, erosion).** FAIL accepted as registered; rung 3 isn't shown in the toy; the anneal arm
  stays out of the paid run. On the record: **wire-on eroded as fast as or faster than wire-off, which echoes the
  real network's live-wire collapse.** Next: a diagnostic, then a candidate fix designed only after it (message
  actions get their own credit: a second value head on the shared reward, with the token log-prob trained on that
  advantage only; fallback COMA-style counterfactual credit), then the same erosion test. Paid confirmation
  unchanged.
- **REGISTRATION -- erosion diagnostic (before the runs).** `comm-toy` `5f3da12`, Modal CPU (plan `diag`), the
  erosion test's settings with 1,800 updates: wire ON seed 90, wire OFF seed 91 (fresh). Logged at every update
  1,500-1,800 on the message head (`gwt_comms_1` + `head_signal_slot*`):
  * (a) ||g|| = norm of the PPO-loss gradient (mean of 4 minibatches of the update's batch, before the update) vs
    the decay pull wd·||theta||; plus the realised per-update displacements: optimizer ||Delta_opt|| vs decay
    ||Delta_wd|| (comparable units, since Adam rescales g).
  * (b) cos(-g, theta_1500 - theta), and cos(Delta_opt, theta_1500 - theta).
  * (c) shared share = Var(S_shared) / (Var(S_shared) + Var(S_own)): S = discounted (gamma·lambda) forward sums of
    each reward component on sender rows, no baseline.
  * (d) reliance every 10 updates (4 empirical-swap pairs: deposit drop, crafts at T') vs plug-in sender MI every
    update.
  * **Predictions, operationalised (means over updates 1,501-1,800, wire-on run):**
    - **Dilution:** cos(-g, back) > 0 **and** the optimizer's displacement component toward theta_1500
      (Delta_opt · back-hat) < 0.1 x ||Delta_wd|| **and** shared share < 0.2.
    - **Live-wire collapse:** cos(-g, back) < 0.
    - **Listener-first:** reliance (deposit drop) falls below 50% of its update-1,500 value at an earlier probe
      than the first update at which MI falls below 50% of its 1,500 value. Within 10 updates of each other
      counts as simultaneous.
  * The wire-off run is the reference for all of the above (no listener, so no use-gradient).

- **RESULTS -- scanned-PPO check (2026-11-08): FAIL on the bar and on speed -> confirmation runs at current speed
  (loop path), per Cam's rule.** `comm-jit` `24678d3`, Modal (one A10G container: 3 checks + scan timing; one L4
  container: loop + scan timing).
  * **Bar, seeds 40/41/42: candidate fails (a), (b) and (c) on all three.** Max per-minibatch loss diff 3.0e-3 /
    6.4e-3 / 1.2e-3 relative; gradients exceed the allclose tolerance (worst excess 0.011-0.031); params after
    the update differ by **11 / 23 / 19 x lr**. The minibatch permutation is identical. All three mutants fail
    (the bar discriminates). On GPU the scanned path diverges far beyond float noise, while the laptop dev check
    differed by ~0.01 lr. Not investigated: whether the loop path is even deterministic run-to-run on GPU
    (nondeterministic reductions amplified over 64 Adam steps would look like this). The scanned path is not
    used.
  * **Speed: no gain anyway.** A10G: scan 33.9-34.3 s/update (PPO 25.0) vs loop 34.6 (PPO 25.5). L4: loop
    43.5-43.8 s, scan 42.3-42.6 s.
  * **GPU choice (cheaper per update at list price, ~$1.10/h A10G, ~$0.80/h L4):** loop path ~$0.0106/update
    on A10G vs ~**$0.0097 on L4** (about 9% cheaper, 26% slower wall-clock) -> **L4** for the confirmation.

- **Cam's ruling (2026-11-08, speed check).** FAIL accepted; the `comm-jit` scanned path stays on its branch,
  unmerged, and all speed work stops before the confirmation (the failing mutants show the bar works). Whether
  the loop path reproduces itself on GPU is open and not blocking (check it, <= $0.10, only if a later result
  depends on exact reproduction). **GPU for the confirmation: A10G**, the smoke's hardware, so the confirmation
  matches the run that set K / floor / N (L4 would save < $1 for ~26% more wall-clock). The later speed levers are
  the lean core brain and the minibatch count, each validated separately, after the confirmation. Smoke and
  diagnostic proceed as registered; K, the floor and N go to Cam with the smoke curves before the paid launch.

- **RESULTS -- MI-floor smoke (2026-11-08).** `comm-jit` `0a82f1e`, A10G, PS (c_ps 2.015), seeds 20/21, 120
  updates each, ~36 s/update; data `docs/data/smoke_PS2{0,1}_0a82f1e.jsonl.gz`. Apparatus check at update 2
  passed on both (PS gradient on the message heads 0.93 / 1.13; 593 / 613 distinct sender messages; listener wire
  nonzero 100%).
  * **Sender MI (Miller-Madow, mean over slots) vs block-null p95,** s20 / s21:
    u1 0.006 / 0.007 (null 0.0008 / 0.0009); **u2 0.238 / 0.244 (0.004 / 0.005)**; u3 1.596 / 1.603 (0.024 /
    0.024); u5 1.83 / 1.98; u10-50 1.84 / 1.99 (flat); u60-90 dips to 1.73-1.80 / 1.90-1.97; u100-120
    1.99 / 1.99. Null p95 stays 0.01-0.10.
  * **Listener deposits per world** (no swap eval in the smoke): s20 1.9 (u1) -> 3.2 (u20) -> 4.2 (u40) -> 1.0
    (u60) -> 4.9 (u120); s21 1.9 -> 1.2 -> 3.4 (u40-50) -> 0.7 (u80-90) -> 1.8 (u120). References: the final
    run's NOSIG0 4.26, COMM 3.4-4.8.
  * **Instability:** sender deposits crashed from ~23 to ~2.5 (s20, u55-65) and to ~1.6 (s21, u80-90), then
    recovered, with the MI null rising during the crash. The final run (no PS) also had sender-deposit drops
    (COMM1 8.4 at u60).
  * **Cam's rule gives N ~ 10:** MI levels off at update ~5 (>= 90% of its plateau by u3-5), so N = 2 x 5 = 10.
    That measures the sender-side code that PS creates by design, not listener use, which in the toy took
    hundreds of updates. **Not applied; sent to Cam with the curve.** If N stood: K <= 2.5 -> K = 2, floor =
    0.5 x 0.238 = 0.119 bit (> null p95 0.005). For a longer N, K = 5 (both seeds ~80x their null): floor =
    0.5 x 1.833 = 0.92 bit.
  * **Budget arithmetic for Cam's N:** measured ~$1.31/GPU-h, so ~$0.013 per update per arm. Three arms (PS x2 +
    NOSIG) at N updates cost ~$0.039 N, so under the $15 cap with a 30% margin N <= ~300.

- **RESULTS -- erosion diagnostic (2026-11-08).** Modal CPU, data/output `comm-toy` `55dcb22`,
  `docs/data/toy_erosion_diag_2026-11-08.txt`. *Disclosed: the analysis script (`scripts/analyze_diag.py`) was
  committed after the runs finished but before their data were read; the metrics and predictions were registered
  before the runs.* Means over updates 1,501-1,800, wire ON (seed 90) / wire OFF (seed 91, reference):
  * (a) raw ||g|| 1.26 / 0.47 vs wd·||theta|| 0.0017 / 0.0011 (raw units; Adam rescales). **Realised per-update
    displacement on the message head:** optimizer 0.053 / 0.053 vs decay 0.055 / 0.035. The optimizer's component
    toward theta_1500 is **+0.056 x** the decay displacement on / -0.005 x off.
  * (b) cos(-g, theta_1500 - theta) **+0.0013** / +0.0002; cos(Delta_opt, back) +0.063 / -0.003.
  * (c) shared-reward share of the token advantage variance **0.34** / 0.09.
  * (d) wire on: reliance (deposit drop) held at ~0.70 through update ~1,750 while MI fell 2.0 -> 0.29; MI < 50%
    at **1,712**, reliance < 50% at **1,790** (MI first). Wire off: MI < 50% at 1,631, plateau ~0.34.
  * **Predictions:** dilution **not supported** (cos > 0 marginally and the pull-back < 0.1 x decay, but the
    shared share is 0.34, not < 0.2); live-wire collapse **not supported** (cos ~ 0, not < 0); listener-first
    **not supported** (MI fell first; listeners kept relying on the code until MI was ~0.3 bit).
  * **Reading (MEASURED, n = 1 per arm):** with the wire on, the PPO gradient on the message head is essentially
    orthogonal to the direction that would restore the code. The optimizer moves the head as much as the decay
    does, but only ~6% of that goes toward the old code. The shared reward is a third of the token's advantage
    variance, so its *share* is not tiny. The problem is that the token-credit gradient carries almost no
    restoring direction: it acts like noise relative to "keep this mapping". CONJECTURED: per-token credit is
    too noisy (low SNR), not too small. That is consistent with Cam's candidate fix (a separate value head on the
    shared reward, so the token's advantage is not the sender's full, own-deposit-dominated advantage), which
    would raise the SNR rather than the share. The wire-on code started decaying ~80 updates later than wire-off
    (a hint of protection, n = 1) but ended lower (0.09 vs 0.34 bit).

- **Cam's rulings (2026-11-08, smoke and diagnostic).** (1) His N rule measured the sender's code (built by PS by
  update 5), not listener use. (2) **Confirmation:** PS x2 on fresh seeds, N_max = 450, A10G, guard $15; swap
  probes every 50 updates. **Primary:** the draft's redirection bar on the mean of the probes at 350/400/450, in
  both seeds (averaging because of the crash-and-recover cycles). **Halts:** update-2 apparatus; MI at K = 5 >=
  0.92 bit; MI under the floor at two probes in a row. No futility stop on redirection. Reported: sender-delivery
  crashes, listener deliveries and reliance at every probe. NOSIG moves to the replication stage (fresh,
  matched); the primary is within-run, so "matched NOSIG now" is withdrawn. (3) Diagnostic accepted with the slip
  disclosed; reading: **the message-head credit is noise-dominated, not too small and not anti-restoring.**
  (4) **Credit fixes in the toy** (registered first, Modal CPU, in parallel). A: a separate value head V_shared(s)
  on the shared reward, with the token's advantage from that stream only (movement keeps the full advantage).
  B: a message-conditioned critic Q_shared(s, m) with the COMA counterfactual advantage. Same erosion protocol
  (wd 3e-4), fresh seeds, wire-on x3 + wire-off x2 per fix. Pass = the erosion pass on wire-on while wire-off
  still erodes; a maintained wire-off is an artifact (= fail of that fix). Gradient tests and mutants for both
  heads before any run. (5) **Toy/CPU allowance raised to $6 total.**

- **REGISTRATION -- real-network PS confirmation (before launch).** `comm-jit` `326c8cb` (the jitted rollout,
  exact-equivalent to the Python path; loop PPO path); launcher `scripts/modal_comm_jit.py::confirm`; analysis
  `scripts/analyze_confirm.py` (master).
  * **Arms:** PS x2, **fresh seeds 50 / 51**, A10G, everything as the final comm run except PS: ORACLE0 warm
    start, tau 0.08, token entropy 0.01, C' + PopArt, vf 0.5, **craft cost 0.1**, 32 worlds, T = 512, **c_ps
    2.015**. N_max = **450** updates. NOSIG moves to the replication stage (Cam).
  * **Probes every 50 updates** (50 ... 450): 10 empirical-swap pairs (a full token triple from the sender's own
    tokens for T' in the paired control rollout), plus MI vs a 1,000-perm block null.
  * **PRIMARY (both seeds):** redirection on the probes at 350 / 400 / 450, pooled to 30 pairs. Listener Crafts at
    the swapped hearth per world: mean(swap) >= 2 x mean(control) **and** the paired increase > 2 SE. **Chosen
    after the toy** (Cam's ruling): the movement score misread 2 of 8 toy seeds that clearly redirected, and with
    two seeds a joint pass on movement, if meaning is present, is only ~0.75^2 = 56%. Movement Delta is reported
    alongside.
  * **Halts:** the update-2 apparatus check (PS gradient on the message heads > 0, > 1 distinct sender message,
    listener wire nonzero); **MI at K = 5 >= 0.92 bit** (Cam's rule on the smoke: half the lower seed's MI at 5,
    above the null p95); MI under 0.92 at two consecutive probes. **No futility stop** on redirection.
  * **Reported at every probe:** sender deliveries (crash = the window minimum), listener deliveries, reliance
    (deposit drop under the swap), movement Delta, MI vs null, swap-token membership.
  * **Cost and guard:** estimated ~5 h per seed (450 x ~36 s + 9 probes) at the measured ~$1.31/GPU-h, so ~$6.6
    per seed and ~$13 total. Meter at launch $5.00. Runaway guard = meter + $15 confirmation + the remaining toy
    allowance (~$4.4) = **$24.40**; the confirmation itself is held to $15 by monitoring its app.

- **Reproducibility note (2026-11-08).** The first credit-fix commit (`ce3b7c1`) broke the toy's bit-reproducibility
  on the default (joint) path: EV differed at 1e-7 from update 2, against the original PS0 run. A bisect showed
  every earlier version reproduces PS0 exactly, including those the erosion test (`8469c90`) and the diagnostic
  (`5f3da12`) ran on, so their results are unaffected. Cause: an unconditional extra `log_softmax` in the rollout,
  which changed the compiled graph. It is now computed only for credit A/B (`14fd2c7`), and the joint path again
  reproduces PS0 bit for bit over updates 1-30.
- **REGISTRATION -- credit fixes A and B under the erosion protocol (before any run).** `comm-toy` `14fd2c7`;
  analysis `scripts/analyze_credit.py`; launcher plan `credit` (Modal CPU, 2 cores / 4 GiB, ~$0.005/min per
  container; 10 containers).
  * **Fix A:** a separate value head V_shared(s) (PopArt-normalised) on the shared (listener-delivery) reward.
    The token's PPO surrogate uses only that stream's GAE advantage (standardised over sender rows; listener rows
    0). Movement keeps the full advantage via `ppo_loss`.
  * **Fix B:** a message-conditioned critic, additive over slots, q_i(s, m_i), regressed (on sender rows) on the
    shared lambda-return, in units of one delivery. The token's surrogate uses the COMA counterfactual advantage
    sum_i [q_i(s, m_i) - sum_m pi_i(m) q_i(s, m)] (stop-grad, standardised over sender rows). Movement keeps the
    full advantage.
  * **Both:** joint token ratio, PPO clip 0.2, token entropy 0.01, critic loss weight vf 0.5, own clip group for the
    new heads. **Self-tests pass, mutants caught:** the token term reaches only the message heads; the critic loss
    reaches only its head; the movement loss touches neither; the token stream is unchanged when only the sender's
    own reward changes (the full-reward mutant is caught); for B, the baseline has E_pi[A] = 1.9e-7 (the
    no-baseline mutant 0.92).
  * **Protocol (as the erosion test):** 2,000 updates; ORACLE0-toy warm start (new heads at init);
    `--wire-init-scale 0.06`; cost 0.45; PS 1.99 annealed to 0 over 1,000-1,500; weight decay 3e-4 on the message
    head from 1,500; eval at 1,500 and 2,000. **Fresh seeds:** A on 100/101/102, off 103/104; B on 110/111/112, off
    113/114.
  * **Pass per fix:** >= 2 of 3 wire-on seeds keep MI >= 1.0 bit at 2,000 **and** reliance (empirical-swap deposit
    drop >= 30%, paired drop > 2 SE, listener deposits over the last 50 updates >= the fix's wire-off mean),
    **and** wire-off still erodes: MI < 0.5 bit at 2,000 in **both** wire-off seeds. A wire-off code that is
    maintained is an artifact, so it counts as a FAIL of that fix (Cam). Reference: the joint baseline went 0/3.
  * **Spend:** toy/CPU allowance $6 total; ~$1.6 used before this; estimate ~$2 for the 10 runs.

- **Pin check for the confirmation (Cam, 2026-11-08; done before reading any probe result).** Seeds 50 / 51 run the
  real-network testbed on **`comm-jit` `326c8cb`** (the containers assert the cloned SHA equals the pin; the output
  files are `confirm_PS50_326c8cb.jsonl` / `confirm_PS51_326c8cb.jsonl`). The toy drift and its bisect (`5f3da12`
  vs `ce3b7c1`) are on the `comm-toy` branch, whose code the confirmation does not use. **The registered step-0
  equivalence was re-run on `326c8cb` itself: PASS** (actions, rewards, positions, obs, tokens and log-probs
  bitwise identical to the Python path, 3 seeds x 512 steps; `docs/data/step0_equiv_326c8cb.txt`). The three
  commits after `1d3b951` add opt-in code only (PS in `ppo_loss`, the unmerged scan path, check modes, halts);
  the training rollout is unchanged. Nothing from the probes had been read when this was checked.

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
- **No threshold is ever registered against a single-update baseline.** First-5 mean, minimum,
  and the SE of the difference between two 5-update means (or better) sets the noise bar. A high
  first draw followed by an ordinary mean is regression to the mean and reproduces a "decline"
  with nothing learned. Confirmed instance (2026-09-29): the shaping-run check was registered
  against ppo 2671's 21.600, the highest value in the whole record (third time this session:
  twice with C0, once here).
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
