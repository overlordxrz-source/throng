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

- **RESULTS -- credit fixes under the erosion protocol (2026-11-08): A FAIL, B FAIL as registered (0/3 wire-on
  each).** Modal CPU, data `comm-toy` `16d6807`, output `docs/data/toy_credit_fixes_2026-11-08_analysis.txt` (registered
  script, unmodified). *Disclosed:* B wire-on seed 111's container restarted at update 133 and re-ran from update 1
  on the same seed. The partial attempt is identical to the complete run over updates 1-133 (deterministic) and
  was dropped (`credit_B_on_s111.clean.jsonl`); the raw file is kept.
  * **Wire-off still erodes, both fixes:** A 0.404 / 0.056, B 0.255 / 0.193 bit at 2,000.
  * **Wire-on, MI at 2,000 / reliance (deposit drop) at 2,000:** A 0.107 / +1%, **0.619 / +69%** (listener 7.7
    deposits per world), 0.318 / +3%. B **0.679 / +60%** (6.3), 0.487 / -14%, 0.270 / +35%. Every seed had 60-76%
    reliance at 1,500. No seed met MI >= 1.0.
  * **Not registered (descriptive):** relative to the joint baseline (wire-on MI at 2,000 ~0.000, reliance ~0%),
    both fixes slow erosion, and 2 of 6 wire-on seeds still had a working channel at 2,000 (A s101, B s110) with
    MI below the 1-bit bar. But wire-off also eroded more slowly than under the joint baseline (0.10 / 0.01 /
    0.002), so part of the slowing is not use-dependent. CONJECTURED: removing the noisy token gradient slows
    drift for everyone. The use-dependent gap is modest: mean wire-on vs wire-off MI at 2,000 is 0.35 vs 0.23
    (A) and 0.48 vs 0.22 (B).
  * **Reading:** neither fix gives the code a protection that meets the registered bar. Per Cam's plan, the next
    step is his call (a stronger fix, COMA-style per-message credit with a better critic, a longer or weaker
    erosion window, or accepting that rung 3 needs a different pressure).

- **Cam's rulings (2026-11-08, credit fixes).** (1) A FAIL and B FAIL accepted as registered; rung 3 is not shown;
  the restart disclosure is fine (deterministic, exact match); the unregistered slowdown stays exploratory.
  (2) **Bar change, prospective only:** future rung-3 tests take *reliance* as primary (swap drop >= 30%, paired
  drop > 2 SE, listener deliveries >= NOSIG level), with MI >= 0.5 bit secondary. MI >= 1 was stricter than the
  property of interest. This does not rescue B. (3) **Next batch** (toy, Modal CPU, fresh seeds, registered first,
  in parallel): **T1** B replication on the new bar (wd 3e-4, wire-on x5, wire-off x2; pass = reliance in >= 4 of
  5 wire-on and wire-off MI < 0.5); **T2** generational turnover (no decay, no credit fix; PS annealed 1,000-1,500;
  then every 250 updates up to 3,000, reinitialise one side, alternating sender message head / listener
  wire-input layers; wire-on x3, wire-off x2; metrics before each reset: reliance, MI and code continuity
  (token -> fire agreement with the pre-reset mapping, chance 25%); pass = reliance and continuity >= 0.8 before
  each of the last 3 resets in >= 2 of 3 seeds; manipulation = wire-off MI < 0.5 after the first sender reset);
  **T3** turnover + newborn imitation (each reset sender head first imitates the previous head's (fire, token)
  pairs from its last 50 updates, then learns normally; seeded iterated learning, Lu et al., ICML 2020).
  (4) Toy/CPU allowance $10 total.
- **M's decision (2026-11-08): T3 approved.** The newborn copying drive is a **permanent part of the agents' nature,
  not a scaffold**. It is registered and run in parallel with T1 and T2: wire-on x3, wire-off x2, the same pass bar
  as T2.

- **REGISTRATION -- T1 / T2 / T3 (before any run).** `comm-toy` `69ad45e`; analysis `scripts/analyze_turnover.py`;
  launcher plan `tbatch` (17 Modal CPU containers, 2 cores / 4 GiB). Everything else as the erosion protocol
  (ORACLE0-toy warm start, `--wire-init-scale 0.06`, cost 0.45, PS 1.99 annealed to 0 over 1,000-1,500).
  * **T1 (credit B on the reliance bar):** wd 3e-4 on the message head from 1,500, 2,000 updates. Wire-on seeds
    120-124, wire-off 125/126. **Pass:** reliance at 2,000 in >= 4 of 5 wire-on, **and** wire-off MI < 0.5 at 2,000
    in both. Reliance = empirical-swap deposit drop >= 30%, paired drop > 2 SE, listener deposits (50 updates
    before) >= the wire-off mean over the same window. Secondary: MI >= 0.5.
  * **T2 (generational turnover):** no decay, no credit fix, 3,000 updates. **Resets at 1,750 / 2,000 / 2,250 /
    2,500 / 2,750**, alternating sender / listener / sender / listener / sender:
    - **sender reset** = the message head (`gwt_comms_1` + `head_signal_slot*`) re-initialised from a fresh init;
      the codebooks are kept;
    - **listener reset** = the wire-input rows of `enc_0`, re-initialised at the same 0.06 scale;
    - Adam moments are zeroed for the reset parameters.
    Wire-on seeds 130-132, wire-off 133/134. Eval (10 empirical swap pairs + MI) at 1,500 and just before every
    reset, plus at 3,000. **Code continuity** at an eval point = mean over slots of the accuracy of decoding the
    current tokens (updates u-9..u) to fires with the per-slot token -> fire map recorded at the previous eval point,
    i.e. across the reset in between. Chance 25%; tokens unseen in the old map count as misses.
  * **T3 (turnover + newborn imitation; approved by M as a permanent part of the agents):** as T2. After each
    **sender** reset, the new head first trains by supervised cross-entropy on (obs, token) pairs from the previous
    head's last 50 updates: 500 Adam steps, lr 1e-3, minibatch 512, only the message head moves. Then it learns
    normally. Wire-on seeds 140-142, wire-off 143/144.
  * **T2 / T3 pass (Cam: "reliance and continuity >= 0.8 before each of the last 3 resets in >= 2 of 3 seeds"),
    operationalised:** at 2,250, 2,500 and 2,750, all three must hold:
    - reliance (the bar above);
    - **reliance retention >= 0.8**, i.e. drop(u) / drop(1,500), because an absolute deposit drop of 0.8 is above
      what even the peak toy channel reached (0.67-0.76);
    - **continuity >= 0.8**.
    The arm passes if >= 2 of 3 wire-on seeds do. *Interpretation flagged to Cam: "reliance >= 0.8" is read as
    retention plus the absolute rung-3 bar.*
  * **Manipulation check:** wire-off MI < 0.5 at 2,000 (after the first sender reset) in both wire-off seeds. This
    is a gate for T2. **For T3 it is reported but not a gate, registered in advance:** newborn imitation passes the
    code on whether or not anyone listens, so a maintained wire-off code is the expected effect of copying, not an
    artifact. T3 is read against T2 on the wire-on bar.
  * **Spend:** within the $10 toy/CPU allowance (~$3.6 used before; estimate ~$4.7 for the 17 runs).

- **Cam's rulings (2026-11-08, T-batch and architecture).**
  1. **Pre-data amendment to T2 / T3** (no T data had been read): "reliance" = the absolute rung-3 reliance bar,
     and ">= 0.8" applies to continuity only. Retention (drop / drop at 1,500) is reported as a secondary, not a
     gate. The analysis was amended and committed before any T data was read (`comm-toy` `b32fded`).
  2. T3's wire-off is reported, not gated; T3 wire-on vs wire-off shows what use adds beyond copying (secondary).
  3. Reset details accepted.
  4. **The architecture doc is accepted as the target and is now in the roadmap.** When the production learner
     fixes land, also **disable the culture-grid writers and the ToM and symbol heads** (no loss; they write noise
     into the world).
  5. **T4 approved** (below).
  6. **Toy/CPU allowance $15 total; guard raised by $5** (to $33.40).
- **REGISTRATION -- T4, a MAPPO-style centralized critic for message credit (before any run).** `comm-toy`
  `07e8c34`; analysis `scripts/analyze_t4.py`; launcher plan `t4` (10 Modal CPU containers).
  * **Critic (training time only; actors unchanged; separate params and Adam):** an MLP 128 x 2 on global features
    of the world at that step:
    - the sender's obs and the listener's obs (positions are inside both);
    - the listener's GRU state;
    - the active-fire one-hot.
    **T4a:** V_shared(g), PopArt-normalised, on the shared (listener-delivery) reward. The token's advantage is its
    GAE, standardised over sender rows. **T4b:** per-slot q_i(g, m_i) over the 64 codes, regressed on the shared
    lambda-return (units of one delivery). The token's advantage is the COMA counterfactual
    sum_i [q_i(g, m_i) - sum_m pi_i(m) q_i(g, m)]. Movement keeps the full advantage in both. Listener rows get no
    token credit (they receive no shared reward).
  * **Self-tests PASS, mutants caught:**
    - the actor param tree is identical to the baseline network;
    - the token term reaches only the message heads;
    - the critic reacts to the listener and fire inputs (max change 1.19 / 1.84) while a listener-blind (local)
      mutant does not (0.0);
    - the token stream is unchanged when only the sender's own reward changes, and the full-reward mutant is caught;
    - T4b's baseline has E_pi[A] = 2.4e-8 (the no-baseline mutant 0.17).
    The joint path still reproduces PS0 bit for bit; the A / B self-tests still pass.
  * **Protocol:** the erosion test's (2,000 updates; PS 1.99 annealed over 1,000-1,500; wd 3e-4 on the message head
    from 1,500; eval at 1,500 and 2,000). **Fresh seeds:** T4a on 150-152, off 153/154; T4b on 160-162, off 163/164.
  * **Pass per arm:** the reliance bar (empirical-swap drop >= 30%, paired drop > 2 SE, listener deposits >= the
    arm's wire-off mean) in >= 2 of 3 wire-on at 2,000, **and** wire-off MI < 0.5 bit at 2,000 in both. Secondary:
    MI >= 0.5; retention vs 1,500.

- **RESULTS -- T4 (2026-11-08): T4a FAIL, T4B FAIL as registered.** Modal CPU, data `comm-toy` `b5de53b`, output
  `docs/data/toy_t4_2026-11-08_analysis.txt` (registered script, unmodified; all 10 runs complete, no restarts).
  * **T4a (central V_shared baseline):** wire-on reliance at 2,000 is +55% / -2% / +19% (1/3 pass); MI 0.22 /
    0.01 / 0.07. Wire-off MI **0.83** / 0.34, so the manipulation failed in one seed.
  * **T4b (central Q_shared counterfactual):** wire-on reliance at 2,000 is **+66% / +62% / +71%, all three pass,
    retention ~1.00**. Listener deposits 7.4 / 7.5 / 6.7 vs the NOSIG level 3.2; MI 1.29 / 0.86 / 1.03. **But
    wire-off MI 1.29 / 0.70:** the code survives the decay with no listener at all, which counts as an artifact
    (Cam's rule) -> FAIL.
  * **Reading.** MEASURED: under T4b the channel stays fully intact through erosion, but wire-on and wire-off codes
    survive alike (mean MI 1.06 vs 0.99), so there is no evidence that *use* is what protects the code.
    CONJECTURED (not tested): the sender only ever emits its current code, so the counterfactual critic's values
    for never-sent tokens are extrapolations. If they sit below the sent token's value, the counterfactual
    advantage always favours the token in use, which entrenches any existing code whether or not anyone listens
    (a known weak spot of COMA-style baselines when coverage is narrow). A test: the same arm with forced
    exploration of off-code tokens (or a critic trained with token dropout), with the wire-off control.

- **Cam's ruling (2026-11-08, T4).** T4a / T4b FAIL accepted; T4b's wire-on result is not evidence for rung 3 (the
  wire-off control kept its code). Two hypotheses: **H1 entrenchment** (counterfactual Q for never-sent tokens is
  extrapolated low, so the advantage always favours the token in use) and **H2 slow decay** (central critics remove
  gradient noise that helped erode codes, so wd 3e-4 is too weak; T4a's wire-off 0.83 fits H2). Free check on saved
  T4b outputs first; then T5a (forced off-code exploration, if H1), T5b (wd recalibrated on wire-off, if H2), or
  both. Relabel-only token dropout is ruled out. Bar unchanged. T5 cap $3 inside the $15 toy allowance, with a
  price probe. Build-order steps 2, 3 and 6 use the local critic (they test code formation, not survival); their
  registrations go to Cam with prices and do not wait for T5. T1-T3 are analysed first.
- **RESULTS -- T1 and T2 (2026-11-08), registered script (amended pre-data), `docs/data/toy_t1_t2_2026-11-08_analysis.txt`,
  data `comm-toy` `63ada8d`.**
  * **T1 (credit B on the reliance bar): FAIL.** Wire-on reliance at 2,000 is 0/5 (drops +9 / +3 / +6 / +4 / +53%;
    listener deposits 1.4-2.9, all below the NOSIG level 3.3). Wire-off MI 0.518 / 0.410: seed 125 is just above
    0.5. B's earlier partial channel (seed 110) does not replicate.
  * **T2 (generational turnover): FAIL (0/3); the manipulation check holds** (wire-off MI 0.09 / 0.08 at 2,000).
    After the first sender reset (1,750) the wire-on code is gone too (MI <= 0.2 from 2,000 on) and never re-forms
    within the 250-update windows. Without the nudge, a fresh message head does not rebuild a code through use.
    Continuity therefore alternates between 0.00 and about chance (0.23-0.31). Retention (secondary) 0.00-0.45.
- **T4 free check (Cam's item 3), partial.** (a) **wire-off MI trajectory**, 10-update means, 1,500 -> 2,000:
  * T4b s163: 1.99 -> 1.31 by 1,600, then flat at 1.26-1.45 (slope over 1,901-2,000: +0.07 bit / 100 updates).
  * T4b s164: 2.00 -> 0.53 by 1,700, then 0.73-0.81 (+0.01).
  * For comparison, T4a s153 / s154: still falling at -0.32 / -0.33 bit / 100.
  So **T4b's wire-off codes stopped decaying partway rather than still falling** (against H2 for T4b); T4a's fit
  H2. **(b) and (c) cannot be computed from the saved outputs:** the T4 runs saved neither the actor nor the
  critic parameters, and no Q values or message advantages were logged. Per the ruling ("anything else: report it
  and don't launch") nothing was launched. The runs are deterministic (the B s111 restart reproduced exactly), so
  a bit-identical regeneration of the T4b seeds with parameters saved at 1,500 and 2,000 (~$0.8) would give the
  exact states; that needs Cam's OK.

- **RESULTS -- T3 (turnover + newborn imitation), 2026-11-08: FAIL (0/3 wire-on).** Registered (amended) script,
  `docs/data/toy_t3_2026-11-08_analysis.txt`, data `comm-toy` `2ce9808`. *Disclosed:* wire-on seed 142's container
  restarted 3 times and re-ran from update 1. All 4 attempts are identical over their shared updates
  (deterministic); the final, complete attempt was kept (`t3_on_s142.clean.jsonl`; the raw file is kept). The
  verdict is the same on the raw and the cleaned data.
  * **Wire-off (reported, not gated):** imitation carried the code through every reset. MI 1.90 / 1.97 at 2,000 and
    1.64-1.95 at every later eval point. Copying alone preserves an unused code, as expected.
  * **Wire-on:** the code was **lost**, not carried. Continuity across the first sender reset was 0.08 / 0.07 /
    0.17 (below the 0.25 chance level), even though imitation reproduced the old head at 0.82 / 0.78 / 0.56
    exact-triple accuracy. By 2,250-2,750, MI was 0.00 in seeds 141 / 142 and 0.33-0.74 in seed 140.
  * **Seed 140 (not registered):** its listeners kept relying on the channel at 2,250 / 2,500 (deposit drop 64% /
    63%, retention 0.96 / 0.95, deliveries above NOSIG) with continuity only 0.26 / 0.40. The pair moved to a *new*
    code instead of keeping the old one. Reliance passed the absolute bar at two of the three checkpoints;
    continuity failed all three.
  * **Reading (MEASURED, toy):** with the wire live, the code was not stable even when each newborn started as a
    copy. With the wire off, it was. Same direction as the real network's live-wire collapse and the erosion test
    (wire-on eroded as fast as or faster than wire-off). CONJECTURED: once a listener's responses feed back into
    the sender's training, the copied mapping drifts within 250 updates.

- **Cam's rulings (2026-11-08, T1-T3 and next steps).**
  1. T1 / T2 / T3 FAIL accepted; seed 142's restarts accepted as disclosed.
  2. **The finding that matters most:** copied codes survive every reset with the wire off but are pulled apart with
     the wire on. With the erosion result and the real network's live-wire-only collapse: **the live wire is an
     active destabilising force, not just a missing restoring one.** Finding it is now the top priority, ahead of
     T5.
  3. **Step A (free, code only):** list every gradient path from the wire into the sender's message head, the
     codebook or any shared parameter. Say whether sender and listener share parameters in the toy. Add a unit test:
     with the token's own PPO loss zeroed, the gradient of the listener's losses w.r.t. the sender's message
     parameters, wire-on vs wire-off; a nonzero difference is a leak.
  4. **Step B (diagnostic, cap $1.5, price probe first):** regenerate the T3 wire-on seeds to 1,750; after the first
     sender reset and imitation, branch each for 250 updates: (i) as T3; (ii) listener frozen; (iii) stop-gradient
     on the wire if step A finds a leak, else the sender's tokens frozen after imitation while the listener learns.
     Readouts at 2,000: continuity, reliance, copy accuracy. Predictions written before launch. Cam's (~50%): the
     pair chase each other, so (ii) lifts continuity above the copy level and (iii) keeps reliance >= 30%.
  5. **Approved:** the T4b regeneration with params saved at 1,500 and 2,000 (~$0.8), then checks (b) and (c) as
     registered. T5 waits for step B.
  6. **Every toy run from now on saves actor and critic params at 1,500, 1,750 and 2,000.**
  7. **Step 3 approved as drafted; launch now.** Step 2 launches once step A reads out. If step A finds a gradient
     path through the VQ wire, Cam is told first (one-hot symbols as actions would then be a fix as well as an A/B,
     and step 2 would gain a registered no-nudge survival phase). Step 6 launches when step 2 reads out, on its
     winning channel. For step 6, chance on held-out pairs = both parts correct (1/16), checked against a
     shuffled-message null.
  8. Noted, unregistered and not scored: seed 140's pair moved to a new code and kept relying on it with no nudge,
     while a fresh head in T2 never rebuilt a code, so copying plus use did something there.
  9. Spend: step B + T4b regeneration + steps 2 and 3 ~ $6.7 of ~$10.7 left; step 6's $1.7 fits in $15.

- **RESULTS -- Step A (gradient-path audit, 2026-11-08, $0).** `comm-toy` `037266c` (`--selftest-leak`).
  * **Shared parameters:** sender and listener share **every** parameter (one ToyNet; the role is set only by the
    informed flag in the obs).
  * **Paths into the message parameters** (`gwt_comms_1`, `head_signal_slot*`, `codebook_*`), from the code:
    1. **The listener's own token policy gradient.** The listener's tokens are sampled from the same shared head
       and enter the joint PPO ratio with the listener's advantage; its input includes the heard wire.
    2. **The listener's VQ commitment and codebook loss.** It pulls the shared head **and the codebook** (the wire
       content itself) toward the listener's z_e, computed from an obs that includes the wire.
    3. **The listener's token-entropy bonus,** by the same route.
    4. **Shared normalisation:** advantages are standardised over both roles and PopArt is shared, so the
       listener's rewards rescale the sender's token credit (a scale coupling, not a gradient).
    5. **None** from the movement or value losses (the message pathway is a separate branch).
  * **Unit test** (token PPO term zeroed, listener rows of one wire-on rollout; gradient w.r.t. the message
    parameters with the heard wire as recorded vs zeroed). At a **trained code** (PS0 final):
    - VQ: 0.026 vs 0.179 (difference 0.179);
    - token entropy: 0.054 vs 0.156 (0.166);
    - the listener's token PG: 0.25 vs 196 (the zeroed wire is far off-distribution);
    - movement / value: 0.0 / 0.0 / 0.0.
    At ORACLE0 (start): differences 0.04 / 0.12 / 0.28 / 0.0. **Leaks exist**, through all three listener message
    losses. The codebook gets 0.012 (trained) / 0.097 (start) from the listener's VQ loss.
  * **Forward coupling (not a gradient) -- the largest effect.** `gwt_comms_1` reads the full obs, including the
    heard wire, so the **sender's token depends on what the listener just broadcast**. At the trained code,
    replacing the sender's heard wire with random tokens changes the sender's own argmax token in **53%** of slots
    (TV 0.55), and zeroing it gives TV 0.90. At the warm start: 15%, TV 0.012. The dependence is learned with the
    code. With the wire live, the sender's fire -> token mapping is partly a function of the listener's broadcast,
    so any drift in the listener's tokens (learning, or a wire-input reset) moves the sender's code. With the wire
    off this input is always zero. CONJECTURED: this is the active destabilising force behind T3 / erosion / the
    real network's live-wire collapse; production's comms pathway also reads neighbour signals.
  * Per the ruling, **step 2 is held** and Cam is told (a gradient path through the VQ wire exists). Step B's arm
    (iii) needs his choice, because a stop-gradient alone would not cut the forward coupling.

- **REGISTRATION -- build-order step 3, no assigned sender (approved as drafted; before any run).** `comm-toy`
  `5100eb0`; analysis `scripts/analyze_step3.py`; launcher plan `step3` (10 Modal CPU containers, ~$2.0).
  * **PROX (wire on), seeds 170-174:** both agents identical. An agent sees the active fire's one-hot (and its
    informed flag is 1) only while within Chebyshev radius 2 of it. Both always speak and listen. The shared reward
    is symmetric: either agent's delivery also pays its partner 2.7. PS (c_ps 1.99, **on**, no anneal) acts on
    whichever rows are informed. Local (joint) critic. Otherwise the replication's PS recipe: ORACLE0-toy warm
    start, wire-init 0.06, cost 0.45, 2,000 updates, eval at 1,000 and 2,000; params saved at 1,500 / 1,750 /
    2,000.
  * **ASSIGNED (today's sender / listener; the concurrent reference), seeds 175-177. PROX_NOSIG (wire off), seeds
    178 / 179** (the NOSIG level).
  * **Swap (PROX):** every currently informed speaker's broadcast is replaced by a full token triple drawn from the
    triples informed agents emitted for T' in the paired control rollout. An apparatus check confirms the partner
    hears it. **Redirection:** Crafts at the T' hearth by agents **uninformed at that step**, per world.
  * **PASS:** at 2,000, PROX redirection (swap >= 2 x control, paired > 2 SE) in >= 4 of 5 seeds, **and** the PROX
    mean redirection ratio >= 0.5 x the ASSIGNED mean ratio ("on a par").
  * **Co-reported:** reliance (deposit drop under the swap), total deliveries vs PROX_NOSIG, the informed fraction
    (~3.5% of agent-steps at the warm start), and the MI of informed agents' tokens.
  * *Note:* step A found a forward coupling (sender tokens depend on the heard wire) and gradient leaks through the
    listener's message losses. Both act in PROX too (everyone speaks and listens). Step 3 is run as approved; its
    reading takes this into account.

- **REGISTRATION -- step B, live-wire branch diagnostic (before any run).** `comm-toy` `a746b64`; launcher plans
  `stepB_pre_s140/141/142` then `stepB_branch` (Modal CPU, 2 cores / 4 GiB; cap $1.5; price probe = the first
  regeneration container alone).
  * **Phase 1:** regenerate the T3 wire-on seeds 140 / 141 / 142 with T3's exact arguments to update 1,750
    (including the 1,750 sender reset + imitation). Then pickle the **full** training state (params, Adam, PopArt,
    env state, JAX key, numpy RNG, MI buffer, turnover maps, imitation buffer, and the elder's (obs, token) pairs).
    Fidelity check, reported: the regenerated records 1-1,750 vs the original T3 runs. The resume path was verified
    locally: resuming from a snapshot is bit-identical to a straight run.
  * **Phase 2, three arms per seed, 1,751-2,000, readouts at 2,000:**
    - **(i) as T3** (branch `none`; it must reproduce T3's updates 1,751-2,000 if phase 1 is bit-identical);
    - **(ii) listener frozen:** listener rows act with the params snapshotted at 1,750 (after the reset and
      imitation) and are masked out of every loss, so the listener's policy, tokens and value do not change; the
      sender learns normally;
    - **(iii) stop-gradient on the wire** (step A found leaks): the message parameters (`gwt_comms_1`,
      `head_signal_slot*`, codebooks) get gradient from **sender rows only**, rescaled to the all-row mean. The
      listener's token PG, VQ and token-entropy terms cannot reach them; all other parameters train as usual.
  * **Readouts at 2,000:**
    - continuity vs the elder's map at 1,750;
    - reliance (empirical swap, 10 pairs: drop, paired SE, deliveries);
    - copy accuracy: the head's exact-triple and per-slot agreement with the elder's (obs, token) pairs.
  * **Not run (disclosed):** an exploratory wire-blind arm (iv). At a trained code, zeroing the sender's heard wire
    alone changes its tokens massively (step A: TV 0.90), so switching the message head to wire-blind at 1,751
    would scramble the code by construction and make its continuity meaningless. Testing the forward coupling
    needs a head that is wire-blind from the start; proposed to Cam separately.
  * **Predictions (written before launch).**
    - *Mine:* (i) reproduces T3 (continuity ~0.1-0.3). **(ii) listener frozen holds the code best:** continuity
      clearly above (i) in >= 2 of 3 seeds, and reliance >= 30% in >= 2 of 3. With the listener's broadcast frozen,
      the forward coupling still exists but its input stops drifting. **(iii) stop-gradient helps less than (ii):**
      continuity above (i) by a smaller margin, because the forward coupling and the listener's own re-learning
      remain.
    - *Cam's (~50%):* the pair chase each other, so (ii) lifts continuity above the copy level and (iii) keeps
      reliance >= 30%.
  * **Registered reading (Cam):** if (ii) holds the code, the listener's re-learning drives the drift. If (ii)
    still drifts, the sender's own learning does it (through a leak or the token's credit). If (iii) loses
    reliance, listeners abandon an imperfect code. Anything else is reported.

- **RESULTS -- step B (2026-11-08).** Modal CPU (~$0.6), data `comm-toy` `9f8521d`, output
  `docs/data/toy_stepB_2026-11-08_analysis.txt` (registered script, unmodified).
  * **Fidelity.** The phase-1 regenerations are **bit-identical to the original T3 runs over updates 1-1,750** (all
    three seeds). The resumed as-T3 arm (i) first differs from the original at 1,751 by ~1e-6 (token entropy),
    float-level noise from the reset / imitation step running in a different process. It then diverges chaotically
    within ~10 updates (e.g. s141 listener deposits at 2,000: 1.59 vs 0.47 in the original). So arm (i) is a valid
    as-T3 continuation, not a bitwise replay, and **the post-reset dynamics are highly sensitive to tiny
    perturbations.** All three arms start from the identical snapshot, so the registered comparisons stand. (The
    local resume test was bit-identical; it had no imitation step between snapshot and resume.)
  * **Readouts at 2,000** (s140 / s141 / s142); copy level right after imitation at 1,750 (exact triple): 0.82 /
    0.78 / 0.56.
    - **(i) as T3:** continuity 0.069 / 0.081 / 0.168; reliance drop +67% / -2% / +1%; copy accuracy (per slot)
      0.23 / 0.10 / 0.02.
    - **(ii) listener frozen:** continuity 0.041 / 0.000 / 0.030; reliance +14% / -2% / +1%; copy 0.28 / 0.25 /
      0.02.
    - **(iii) stop-gradient:** continuity 0.090 / 0.000 / 0.169; reliance +55% / +6% / -3%; copy 0.16 / 0.10 /
      0.02.
    - Exact-triple copy accuracy at 2,000 is 0.000 in every arm and seed.
  * **Registered reading:** (ii) does **not** hold the code: continuity is above (i) in 0/3 seeds and reliance
    >= 30% in 0/3. So the drift is driven by **the sender's own learning**, not the listener's re-learning. (iii)
    makes no material difference (continuity above (i) in 2/3 by <= 0.02; reliance >= 30% in 1/3). **Both
    predictions failed** (mine: (ii) holds best; Cam's: (ii) above the copy level and (iii) keeps reliance).
  * **Reported as "anything else":**
    1. **Arm (ii) did not remove the forward coupling.** The frozen listener params are post-reset, and the message
       head is shared, so the frozen listener still broadcasts from the *new* head. The sender hears a wire unlike
       the elder era's, and its message head reads that wire (step A).
    2. **The head itself changes:** copy accuracy, measured on the elder's *fixed* (obs, token) inputs, fell from
       0.56-0.82 (exact) to 0.02-0.28 (per slot) in every arm. So the drift is in the weights, not only in the
       inputs.
    3. **The copy is overwritten within 250 updates in every arm**, while in T3 with the wire off the copied code
       held through every reset.
    CONJECTURED: the sender's message head is pushed by its own losses (token PG on its shared-reward-diluted
    advantage, VQ commitment toward codewords for z_e computed from the changed heard wire) once its inputs include
    a live wire. A wire-blind-from-the-start message head (proposed) would separate the input path from the
    credit path.

- **RESULTS -- build-order step 3, no assigned sender (2026-11-08): FAIL (PROX redirection 0/5).** Registered
  script, `docs/data/toy_step3_2026-11-08_analysis.txt`, data `comm-toy` `39e4acb` (all 10 runs complete, 2,000
  updates).
  * **PROX:** Crafts at the swapped fire by uninformed agents, control -> swap: 0.30 -> 0.28, 0.28 -> 0.31, 0.36 ->
    0.30, 0.29 -> 0.29, 0.28 -> 0.28 (ratio 0.84-1.12; no paired increase). Reliance drop +3% to +6%. MI of
    informed agents' tokens **1.99-2.00 bits** (PS makes the code informative). Informed fraction 0.17-0.21.
  * **ASSIGNED reference (rung 2 replicates again):** redirection ratios **25 / 44 / 33x**, reliance drop **71 / 73 /
    73%** (3/3).
  * PROX mean ratio 0.98 vs 0.5 x the ASSIGNED mean (16.9): not on a par.
  * **Not registered:** PROX deliveries with the wire on are 20.8 / 15.5 / 19.1 / 20.5 / 21.0 per world vs 14.7 /
    15.4 with it off, so the wire carries something useful, but not "which fire" in a form uninformed partners act
    on under the swap. CONJECTURED: on a 12 x 12 map, a radius-2 view finds the lit fire quickly (informed ~20% of
    the time), so the message adds little "where" information; the wire may carry timing or proximity
    information instead. Not tested.

- **RESULTS -- T4 check 3 (b)/(c) on the regenerated T4b seeds (2026-11-08): neither H1 nor H2 -> T5 not launched
  (per the ruling).** Data `comm-toy` `e61c23a`, `docs/data/toy_t4b_check3_2026-11-08.txt`.
  * **Regeneration fidelity:** s161 (on) and s164 (off) are **bit-identical** to the original T4b runs. s160 (on)
    and s163 (off) differ at the float level from updates 1 / 3 (~1e-6) and then diverge, so they are replicates,
    not replays. CONJECTURED cause: Modal CPU containers landing on hosts with different instruction sets (the
    same mechanism would explain step B's 1e-6 offset); not confirmed. s163's container also restarted once (the
    final complete attempt was used). **Regenerated s163 eroded to MI 0.0 / 0.91 / 0.38 at 2,000 (mean 0.43),
    where the original kept 1.29**, so T4b's "wire-off code survives" is not robust across replicates.
  * **(b) Q over all 64 tokens per slot, 500 sampled sender states,** sent-token top-Q share / Q spread (std over
    tokens):
    - wire-off s163: 0.248 / 2.08 at 1,500 -> **0.009 / 1.00** at 2,000;
    - wire-off s164: 0.241 / 1.97 -> **0.108 / 2.75**;
    - wire-on s160: 0.269 / 1.82 -> 0.001 / 3.61;
    - wire-on s161: 0.308 / 4.80 -> 0.555 / 6.23.
  * **(c) mean counterfactual advantage of the sent token:** wire-off +0.0000 / -0.0002 at 2,000 (+0.0000 /
    +0.0000 at 1,500); wire-on -0.0093 / +0.0184.
  * **Registered reading:**
    - **H1 (entrenchment)** needs the sent token to top Q in >= 80% of states in both wire-off seeds: measured
      0.9% and 10.8%, and the wire-off sent-token advantage is ~0. **Not supported.**
    - **H2 (slow decay)** needs wire-off MI still falling (check (a): flat for T4b) **and** a wire-off spread under
      a quarter of wire-on's (measured 1.88 / 4.92 = 0.38). **Not supported.**
    -> **"Anything else": reported; T5 is not launched.**

- **RESULTS -- real-network PS confirmation (2026-11-08): PRIMARY FAIL (1/2) as registered.** `comm-jit` `326c8cb`,
  A10G, seeds 50 / 51, 450 updates each, both complete; no halts (the update-2 apparatus check and the update-5 MI
  check, 1.83 bits vs 0.92, passed on both). Data `docs/data/confirm_PS5{0,1}_326c8cb.jsonl.gz`; output
  `docs/data/confirm_PS_2026-11-08_analysis.txt` (registered script, unmodified).
  * **Seed 51: PASS, strongly.** Redirection on the probes at 350 / 400 / 450 (30 pairs): Crafts at the swapped
    fire per world 2.12 -> **39.78 (18.7x)**, paired +37.65, SE 1.59. **Reliance rose steadily across probes:**
    +26 / +40 / +33 / +40 / +48 / +50 / +56 / **+61** / +53% (updates 50-450); movement Delta +0.05 to +0.40.
    Listener deliveries per world (training window) climbed from 2.4 to 5.2-5.9, above the final run's no-signal
    level (4.26). Sender deliveries never crashed (window minima >= 11.8 from update 150).
  * **Seed 50: FAIL.** A weaker channel formed early (redirection 1.5-2.0x and reliance 12-23% at probes 50-200).
    It faded by 250-300, and then **a sender crash** (window minimum sender deliveries 1.8 at 350 and 2.2 at 400;
    window mean 9.7 at 400) took the listener with it: deliveries 0.8 at probes 350 / 400. Over 350-450,
    redirection is 5.05 -> 5.56 (1.10x).
  * MI of sender tokens ~2.0 bits at every probe in both seeds (null p95 <= 0.10). Swap tokens were 100%
    in-distribution.
  * **Reading.** As registered, the confirmation fails (the bar needed both seeds). MEASURED: **seed 51 is the first
    real-network channel that listeners depend on and act on** (rung-2 behaviour, 18.7x redirection, 61% reliance
    at 400), with the same PS recipe that passed in the toy. Seed 50 shows the risk the averaging was meant to
    absorb: a sender crash-and-recover cycle that landed on the primary window. Per the roadmap, nothing is called
    a result before a replication passes. Next step is Cam's.

- **Cam's rulings (2026-11-08 evening).**
  1. Accepted as registered: step B (the sender's own learning drives the drift; both predictions failed), step 3
     FAIL 0/5, the T4 check (neither H1 nor H2), and the confirmation primary FAIL 1/2. Restart and regeneration
     disclosures accepted. Bit-identity across Modal hosts is no longer required (rule-6 snapshots remove the need
     to regenerate).
  2. **T5 dropped** (T4b's wire-off survival did not replicate).
  3. **C1, the wire-blind message head from the start** (the leading suspect is step A's forward coupling). Fresh
     seeds, cap $2.5, price probe first. Wire-blind = the sender's tokens cannot depend on what it hears by any path,
     including a shared trunk. Apparatus check before launch: token TV under random heard input is exactly 0; a
     mutant with one live wire row must fail it. **C1-T3** (T3 protocol, wire-on x3 / wire-off x2): the coupling is
     the destabiliser if, in >= 2 of 3 wire-on seeds, continuity across the first reset is >= 0.5 **and**
     exact-triple copy accuracy on the elder's inputs at 2,000 keeps >= half its post-imitation value. **C1-E**
     (erosion protocol, same counts): the rung-3 bar, unchanged. **Formation gate:** at 1,500 the wire-blind head
     must meet the P1 reliance bar in >= 2 of 3 wire-on seeds in each protocol; otherwise stop and report.
     Predictions first; Cam's (~55%): C1-T3 passes its reading, and C1-E still fails the rung-3 bar (erosion also
     tests credit).
  4. **Step 2: option 2.** It launches when C1 reads out: the wire-blind head in both arms if C1-T3 confirms the
     coupling, the normal head otherwise. Step 6 follows step 2.
  5. **Step 3:** follow-up waits for C1. A free check: how often the uninformed partner finds the lit fire on its
     own before a swap could matter, and how often the informed agent just delivers itself.
  6. **Confirmation:** seed 51 is recorded as the strongest positive evidence yet, not as a result. Free checks
     before any replication: a seed-50 post-mortem (does the 250-300 fade start with sender MI or listener
     reliance; before the 350-400 crash, the PS loss's share of the gradient, grad norm, KL, clip fraction, policy
     entropy, value loss and EV), and step A's random-heard-input test on the real net (seed 51 final; seed 50 near
     update 200). The replication (PS x3 + matched NOSIG, ~$24) waits for C1 and these checks, and for M to top up;
     no paid GPU launch until then.
  7. Spend: ~$8.4 left in the toy allowance; C1 ($2.5) + step 2 ($2.4) + step 6 ($1.7) = $6.6. Guard stays at
     $33.40.

- **REGISTRATION -- C1, the wire-blind message head from the start (before any run).** `comm-toy` `f6f574e`; analysis
  `scripts/analyze_c1.py`; launcher plan `c1` (Modal CPU, 2 cores / 4 GiB; cap $2.5; price probe = c1t3_on_s180
  alone first; ~$0.2 per run, 10 runs ~ $2.0).
  * **Wire-blind:** the message pathway (`gwt_comms_1` -> slot heads -> codebooks) is a feed-forward branch from the
    obs that shares nothing with the trunk. Its input has the heard-wire dims zeroed, for both roles, from update 0
    (the warm start's wire rows are simply ignored). The listener's *policy* still reads the wire through the trunk
    (`enc_0`).
  * **Apparatus check, passed before launch** (`--selftest-wireblind`): the sender's token distribution under
    random heard input vs recorded is **exactly** unchanged (max |dp| = 0.0) at both a trained code (PS0 final) and
    ORACLE0-toy. The one-live-row mutant changes it (max |dp| 0.42 / 0.0037) and fails, as required. The normal head
    changes its argmax in 43% / 14% of slots. The default path still reproduces PS0 bit for bit.
  * **C1-T3:** T3 protocol (PS 1.99 annealed over 1,000-1,500; sender / listener resets from 1,750; newborn
    imitation), **run to 2,000** (the readout update; later resets are not needed). Wire-on seeds 180-182, wire-off
    183 / 184.
  * **C1-E:** erosion protocol (wd 3e-4 on the message head from 1,500; 2,000 updates). Wire-on 185-187, wire-off
    188 / 189.
  * **Bars (Cam):**
    - **formation gate:** P1 reliance at 1,500 in >= 2 of 3 wire-on seeds per protocol, else stop and report;
    - **C1-T3 reading:** the coupling is the destabiliser if, in >= 2 of 3 wire-on seeds, continuity across the
      first reset (2,000 vs the 1,750 map) is >= 0.5 **and** exact-triple copy accuracy on the elder's inputs at
      2,000 is >= half its post-imitation value (same 4,096 sampled rows);
    - **C1-E:** the rung-3 bar (reliance at 2,000 in >= 2 of 3 wire-on; wire-off MI < 0.5 in both).
  * **Predictions (before launch).**
    - *Mine:* formation gate passes in both (~80%: PS builds the code, and listeners read it through the trunk).
      **C1-T3 confirms the coupling** (~60%). With the head's inputs identical to the elder era, the copy should
      hold as it did with the wire off in T3; the risk is that the sender's own token credit still overwrites it.
      **C1-E fails the rung-3 bar** (~65%): erosion also tests credit, and the diagnostic showed the credit carries
      no restoring direction.
    - *Cam's (~55%):* C1-T3 passes its reading; C1-E still fails the rung-3 bar.

- **RESULTS -- free checks (2026-11-08 evening).**
  * **Seed-50 post-mortem** (saved confirmation logs):
    - **The 250-300 fade starts with the listener, not the sender.** Sender MI stays 1.995-1.998 bits through every
      window, and so does the PS term (4.13 nats). Reliance fell from +16% (probe 200) to -2% (probe 250). Seed 50's
      listener per-attempt correctness never rose (0.12-0.15); seed 51's rose 0.17 -> 0.32.
    - **The crash (onset update 341; sender deliveries 21 -> 4 within 6 updates)** was preceded by rising
      **policy entropy** (window means 1.65 at 301-325, 1.87 at 326-350; 2.19 at 342) and a slight EV dip
      (1.000 -> 0.99). Token entropy and MI were unchanged.
    - **Not answerable from the logs:** the PS loss's share of the gradient, grad norm, KL, clip fraction and value
      loss were never written to the real-testbed records (`ppo_update` computes most of them; they were not
      emitted). Adding them is a free change before any replication.
  * **Step A's random-heard-input test on the real network** (`comm-jit` `--coupling-test`, one rollout, 4,096
    sender rows). Only *final* checkpoints were saved for the confirmation, so there is no seed-50 checkpoint near
    update 200; both finals were used. Replacing the sender's heard wire with random tokens changes its own token
    argmax in **97.6% (seed 51) and 94.6% (seed 50) of slots** (TV 0.98 / 0.95); zeroing it changes 99.7%. At the
    ORACLE0 warm start: 16.7% (TV 0.021). **On the real network, the trained sender code is almost entirely a
    function of (fire, what the partner just said)**, including in seed 51, whose channel works. The forward
    coupling exists, and is learned, on the real network too.
  * **Step 3's world** (saved PROX params at 2,000, 4 rollouts each, 3,700-5,100 delivered fire periods per seed):
    **96-98% of deliveries are by the agent that became informed first**, with a **median of 1 step** from becoming
    informed to delivering (seeing the fire at radius 2 means already being at it; the delivery radius is 1).
    Partners deliver 2-4% of the time and find the fire on their own beforehand in 6-13% of periods. **The world
    gives almost no reason to listen**, so per Cam's rule the redesign is the world, not the agents.

- **RESULTS -- C1, the wire-blind message head (2026-11-08): formation gate PASS (3/3 both protocols); C1-T3
  coupling CONFIRMED as the destabiliser (3/3); C1-E PASS -- the first pass of the rung-3 bar.** Registered script,
  `docs/data/toy_c1_2026-11-08_analysis.txt`, data `comm-toy` `09d16e4`. *Disclosed:* seed 180's container (the probe)
  restarted once; the final complete attempt was used (raw file kept). ~$2 Modal CPU.
  * **Formation (1,500):** C1-T3 drops +65 / +67 / +73%, listener deposits 5.1 / 7.9 / 9.3 vs NOSIG 2.3. C1-E +69 /
    +73 / +65%, 5.7 / 7.1 / 6.5 vs 2.1. MI 1.98-2.00. **A wire-blind head forms the code as well as the normal one.**
  * **C1-T3:** continuity across the first sender reset **0.868 / 0.957 / 0.853** (T3 with the normal head: 0.07 -
    0.17). Exact-triple copy accuracy post-imitation -> 2,000: 0.781 -> 0.500, 0.492 -> 0.417, 0.384 -> 0.309 (all
    >= half; T3: -> 0.000). Wire-off continuity 0.957 / 0.961. **Removing the forward coupling lets a copied code
    survive the reset.**
  * **C1-E (rung-3 bar):** wire-on reliance at 2,000 **+39%** (paired 2.03, SE 0.08; deliveries 5.27 vs NOSIG 2.50),
    **+73%** (3.81, SE 0.07; 5.48), +17% (fails); **wire-off MI 0.001 / 0.022** (eroded). **PASS (2/3, wire-off
    erodes).** With the coupling removed, listener use kept the channel alive through erosion while the unused
    code died. Secondary: wire-on MI at 2,000 is 0.18 / 0.41 / 0.03 (below 0.5). The code got sparser but stayed
    relied on.
  * **Predictions:** C1-T3 passes (mine ~60%, Cam's ~55%) -- right. C1-E fails (mine ~65%, Cam's) -- **wrong.**
  * **Not registered, reported:** in C1-T3, reliance at 2,000 is -4% / -15% / +2% despite continuity 0.85-0.96.
    The sender kept its mapping across the reset, but listeners were not relying on it at that checkpoint. Not
    explained yet.
  * **Reading:** the live wire's forward coupling (the message head reading the heard wire) is the active
    destabiliser behind T3's lost copies, and plausibly behind the erosion failures and the real network's
    collapse (step A on the real net: 95-98% token dependence on the heard wire). Rung 3 passes in the toy with a
    wire-blind head. This is the toy only; nothing is called a result before a replication. Per the ruling, step
    2 now launches with the wire-blind head in both arms.

- **Cam's rulings (2026-11-08, free checks).**
  1. All three accepted. Approved (free): add the PS loss's gradient share, grad norm, KL, clip fraction and value
     loss to the real testbed's logs, with the default path bit-identical. **Every future paid run saves a
     checkpoint every 50 updates.**
  2. The real-net coupling (95-98% vs 17% at ORACLE0) means the toy's suspect exists there too, and stronger. The
     replication waits for C1. If C1-T3 confirms the coupling (**it did, 3/3**), the replication gets a wire-blind
     message pathway (with its apparatus check) against the current pathway; Cam registers it.
  3. Seed 50 noted (the fade began with listeners who never learned to decode; the crash followed rising action
     entropy). No ruling until the new logs exist.
  4. **Step 3b, a redesign of the world:** a **lookout** -- a cell or small area far from all 4 fires. Whoever
     stands there sees which fire is lit; nobody else does, and no roles are assigned. The lookout-to-fire distance
     and the fire's lit duration are set (by a free calculation from movement speed) so that walking there yourself
     loses to telling a nearer partner. A value-of-information check comes first: LOOKOUT-ORACLE (the agent not at
     the lookout is fed the lit fire) vs NOSIG must give >= 1.5x deliveries; if not, retune the distances, not the
     agents. Arms: LOOKOUT x5, NOSIG x2, ASSIGNED x3 in the same world. Pass: redirection in >= 4 of 5 LOOKOUT at
     >= 0.5x the assigned mean. Report, don't gate, the share of time at the lookout. Priced with the draft; it waits
     for M's top-up if it doesn't fit after C1, step 2 and step 6.
  5. Rejected: making the informed agent unable to deliver (a role by rule; the lookout puts the pressure in the
     geography instead).

- **REGISTRATION -- build-order step 2, channel A/B (launches now per Cam's ruling; C1-T3 confirmed the coupling, so
  both arms use the wire-blind head).** `comm-toy` `ec4be03`; analysis `scripts/analyze_step2.py`; launcher plan
  `step2` (12 Modal CPU containers, ~$2.4).
  * **Base (both arms):** the replication's PS recipe (ORACLE0-toy warm start, wire-init 0.06, cost 0.45, c_ps 1.99
    **on**, local critic, 2,000 updates, eval at 1,000 and 2,000), **plus the wire-blind message head** (C1).
  * **S2_VQ (seeds 190-194):** today's channel (3 slots x 64 learned VQ codes, VQ loss 0.1).
  * **S2_SYM (seeds 195-199):** 3 symbols from a vocabulary of **8**, broadcast one-hot. Implemented as a
    codebook-distance policy over **fixed orthogonal codewords** c_k = 0.3 e_k (k < 8), with the other 56 rows pushed
    unreachably far. -||z - c_k||^2 / tau = (2 x 0.3 / 0.08) z_k + const, i.e. a linear symbol head (temperature tau
    / 0.6) on the same message pathway, and the broadcast is 0.3 x one-hot. No VQ loss, so the codewords never
    change. Everything downstream (the joint PPO ratio, the PS term, the swaps, MI) is unchanged.
  * **S2_NOSIG (seeds 200 / 201):** wire off (the NOSIG level).
  * **Apparatus self-test PASS** (`--selftest-sym`): probability on unreachable symbols 0.0; broadcasts are exactly
    0.3 x one-hot; the constant-codebook mutant is caught; after one real update the codebooks are bit-identical and
    the symbol head moved. The default path still reproduces PS0 bit for bit.
  * **Bars (as drafted):** SYM is **adopted** iff its reliance passes in >= 4 of 5 seeds **and** its mean deposit
    drop is >= VQ's mean - 0.10. Otherwise VQ stays; if both arms fail their own reliance bar, the A/B is
    uninformative. Redirection is co-reported (4 of 5). Secondary: MI, formation speed (first update with plug-in MI
    > 1 bit), listener deliveries, message entropy.

- **Done (free, Cam's item 1):** the real testbed (`comm-jit` `c242635`) now has `--log-diag` (PS gradient share,
  grad norm, approx. KL, post-update clip fraction, value / policy / VQ loss), computed in separate passes on a fixed
  2,048-row sample. **Training is bit-identical with and without it** (params after 2 updates compared byte for
  byte). `--ckpt-every 50` checkpoints every paid run. Smoke values: grad norm 1.60, PS share 0.33, KL 0.020, clip
  fraction 0.33.
- **Ready for the replication (free): a wire-blind message pathway on the real network** (`AgentNetworkJax
  wire_blind=1`: the heard-neighbour block of the message pathway's input is zeroed; the pathway shares nothing else
  with the trunk). **Apparatus check on seed 51's trained weights:** max |dp| of the token distribution under random
  input on all 6 neighbour slots = **0.000000**. The one-live-dim mutant gives 1.0 (caught); the normal pathway
  changes 97.6% of tokens. The gradient tests still pass (12/12).

- **RESULTS -- build-order step 2, channel A/B (2026-11-08): VQ STAYS.** Registered script,
  `docs/data/toy_step2_2026-11-08_analysis.txt`, data `comm-toy` `3723769` (all 12 runs complete; wire-blind head in
  both arms).
  * **S2_VQ:** reliance **5/5** (drop +71 / +71 / +71 / +67 / +91%; listener deliveries 6.8-9.5 vs NOSIG 3.1);
    redirection 5/5; MI 1.98-2.00; MI > 1 bit by update 8-9. **Rung 2 replicates again, now with the wire-blind
    head.**
  * **S2_SYM** (3 x 8 symbols): reliance **1/5**. Drops of +69 / +60 / +57 / +65% in four seeds and redirection 4/5:
    listeners do depend on the symbols. But their deliveries are 6.9 / 2.3 / 2.2 / 0.4 / 1.2, **below** the NOSIG
    level in four seeds. Seed 198 never formed a usable code (drop 0%, MI 1.33, message entropy 3.9). MI 1.33-1.99;
    MI > 1 bit by update 10-11. Mean drop +50% vs VQ's +74% (-0.10 margin not met).
  * **Reading:** in this toy, 3 x 8 one-hot symbols chosen as actions do not match the learned VQ wire: listeners
    rely on symbols but deliver less with them than with no channel. CONJECTURED (not tested): the fixed, low-norm
    one-hot inputs are harder for the listener to decode than learned codewords, or the symbol-policy temperature
    (tau / 0.6) is mis-sized. Per the ruling, **step 6 follows on the winning channel (VQ)**. Note for Cam: with 64
    codes per VQ slot, 16 (fire x material) meanings do **not** outnumber one slot's codes, which the step-6 design
    requires.

- **REGISTRATION -- build-order step 6, compositionality pilot (launches now: step 2 read out, winner VQ).**
  `comm-toy` `e48e98f`; analysis `scripts/analyze_step6.py`; launcher plan `step6` (5 Modal CPU containers, ~$1.7).
  * **World:** the target is a **(fire, material) pair**, 4 x 4 = 16 meanings. The sender sees the fire one-hot
    (need dims) and the material one-hot (in the 4 former noise dims; the obs width and warm start are unchanged).
    Craft is replaced by **4 Craft-material actions (5, 9, 10, 11)**; a delivery needs the active fire **and** the
    target material's action. The pair re-rolls to a different training pair on a delivery or after 96 steps.
    **Held out: pairs 1 = (fire 0, material 1) and 11 = (fire 2, material 3)** (one per fire family; fixed before
    running), never active in training (asserted every rollout).
  * **Channel: step 2's winner, VQ, with the wire-blind head**, and the **slot vocabulary cut to 4** (codes 4-63
    unreachable; codes 0-3 stay learned VQ codewords), so the 16 meanings outnumber one slot's codes. *Disclosed:*
    the draft specified the cut for symbols; it is applied to VQ here because the design requires meanings to
    outnumber one slot's codes, which 64-code VQ slots would not.
  * Base: the replication's PS recipe (PS on, labels = the 16 pairs), ORACLE0-toy warm start, wire-init 0.06,
    cost 0.45 (any of the 4 Craft actions when not delivering), **3,000 updates**, eval at 1,000 / 2,000 / 3,000.
    Seeds 210-214.
  * **Apparatus self-test PASS:** every delivery is at the active fire with the target material's action; material
    dims are present for informed rows only; no held-out pair is active. **The leak mutant trips the training
    assertion at update 1.** The default path still reproduces PS0 bit for bit.
  * **Gates (Cam):**
    1. topographic similarity at 3,000 above a 1,000-permutation null (p95) in >= 4 of 5 seeds;
    2. held-out generalisation: listener success per attempt (both parts right) above chance 1/16 (binomial p <
       0.01) **and** above a **shuffled-message null** (each listener hears another world's sender; two-proportion
       p < 0.01) in >= 3 of 5 seeds.
    PASS = both.

- **Cam's rulings (2026-11-08 night).** C1 accepted: C1-T3 confirms the coupling (3/3) and C1-E passes the rung-3 bar as
  registered (a registered pass, **not yet a result** until a replication passes). Free checks on C1 first:
  - C1-E's redirection at 2,000;
  - wire-on MI and reliance slopes over 1,800-2,000;
  - whether C1-T3's 2,000 probe came after a listener wire-row reset.

  Approved:
  - C1-E wire-on x3 continued under erosion to 3,000 (reliance bar at 3,000 in >= 2 of 3);
  - C1-T3 x3 continued to 2,750 (reliance at 2,250 and 2,750, after 250 updates of listener relearning, in >= 2 of
    3; continuity >= 0.8 across each reset), together capped at $0.6;
  - **a C1-E replication** (fresh seeds, wire-on x5 / wire-off x2, cap $1.5; pass = the reliance bar at 2,000 in
    >= 3 of 5, pooled mean drop >= 30%, wire-off MI < 0.5 in both; redirection reported);
  - **step 3b's value-of-information check** ($0.8): the non-lookout agent is fed the lit fire only while the other
    agent stands at the lookout; report occupancy.

  Step 2: VQ stays, SYM parked. Step 6's vocabulary cut accepted (pre-data, design-consistent). Step 3b layout
  accepted. Toy allowance $18; guard $36.40.

  North star: all four ideas agreed. When hearing reconnects, tie it to age, as songbirds do (learn, then
  crystallise; a trait of their nature, so M's OK). Order: lookout -> two predator kinds -> what x where. Build the
  dictionary and the "speak to them" test now. The culture test is drafted after the C1-E replication. Cam's own
  proposal: separate brains in the toy next (2, then 4 with rotating partners; Rita et al., ICLR 2022).
- **Cam (2026-11-08): M won't top up; use this account's remaining credit, then move to a new account.**
  - Check the real remaining credit first; guard at $41.0 metered (or $1.5 under the real limit if lower).
  - **The real-net replication is shrunk to fit:** wire-blind PS x2 on fresh seeds, cap $13, otherwise as registered
    (450 updates, `--log-diag`, `--ckpt-every 50`, the confirmation's probes / halts / primary / bar). Pass = primary
    in 2 of 2; 1 of 2 = partial (the third seed and NOSIG run on the next account). The final run's 4.26 is the
    historical NOSIG reference.
  - Toy items first (~$2.9), then the replication.
  - **Before any account switch, copy every checkpoint, snapshot and log off Modal.**

- **C1 free checks (Cam's item 2; $0, exploratory; `comm-toy` `86bd9c6`, `scripts/analyze_c1_free.py`, output
  `docs/data/c1_free/analysis_c1_free.txt`).** These are eval-only re-probes of the saved C1 actor params (local,
  ~30 s each).
  - **Re-probe validation:** controls and constructed swaps are bit-identical to the in-run probe. The empirical
    replacement triple is a different draw (the eval RNG is seeded with the update number, and in-run MI consumes it
    first). The re-probes reproduce the in-run 2,000 drops: C1-E 39 / 72 / 16% vs in-run 39 / 73 / 17%.
  - **(c) C1-T3's 2,000 probe came after the listener wire-row reset: yes, a probe artifact.** The final
    `evaluate(params, a.updates)` runs after the loop, and the loop's last step had already applied the 2,000
    listener reset (newborn `enc_0` wire rows). The u2000 params were saved before that reset.
    - Re-probed pre-reset at 2,000, **reliance passes in 3 of 3 wire-on seeds:** drop +67 / +69 / +73%, paired
      5.1 / 5.0 / 6.9 (SE 0.10-0.16). Listener deliveries 7.9 / 7.8 / 9.7 vs NOSIG 3.06. Redirection ok 3/3
      (crafts at T' 0.5-0.7 -> 18.6-23.2).
    - The in-run post-reset numbers were -4 / -15 / +2%.
    - The registered C1-T3 reading is unaffected. Continuity is recorded before the reset. `copy_acc` is emitted
      after it, but the listener reset touches only `enc_0`'s wire rows, which the wire-blind message pathway never
      reads.
  - **(a) C1-E redirection at 2,000 (in-run): 2 of 3.**
    - s185: crafts at T' 3.10 -> 15.39.
    - s186: 0.49 -> 15.67.
    - s187: 3.12 -> 3.87 (fails, as its reliance does).
  - **(b) C1-E wire-on slopes: the code is decaying in 2 of 3 seeds.**
    - Reliance (re-probes) at 1,500 / 1,750 / 2,000:
      - s185: 69 / 56 / 39%;
      - s186: 72 / 68 / 72% (stable);
      - s187: 66 / 53 / 16%.
    - Over 1,800-2,000, per 100 updates:
      - plug-in MI: s185 -0.020 (0.25 -> 0.22), s186 -0.034 (0.45 -> 0.40), s187 -0.057 (0.13 -> 0.04);
      - listener deliveries: s185 +0.12, s186 -0.15, s187 -0.24.
    - Wire-on MI(MM) at 2,000 is 0.18 / 0.41 / 0.03, so even the passing seeds sit below the 0.5-bit "eroded" line.
    - Per-slot sender MI is low while listeners still rely on the code. The wire carries the sampled tokens'
      codebook vectors, so per-step information is capped by the token MI. The likely reading is that the
      listener's GRU integrates a weak per-step signal over many steps, and the swap feeds it a misleading T' triple
      at every step. **Not measured yet:** the dictionary work computes the information carried in the whole triple.
  - **My prediction (registered before any continuation):** C1-E's 2,000 pass is a snapshot of a decaying code.
    - s185 crosses 30% by ~2,100-2,300; s187 is already below.
    - The 3,000 bar (>= 2 of 3) **fails** (~75%).
- **Item 3 cannot run as written: C1 saved actor params only, not full-state snapshots.** There is no optimizer,
  PopArt, environment or RNG state at 2,000. Options for Cam:
  - **(A) Faithful regeneration.** Same seeds, rerun from 0 in one process (deterministic; a different host can
    differ at the float level, as in step B), with extra eval points.
    - C1-T3 wire-on x3 to 2,750 with `--reset-until 2500`. With `_T3` as written, the 2,750 probe would come after
      the 2,750 sender reset: the same artifact as (c).
    - NOSIG level declared now as C1-T3 wire-off at 2,000 (3.06), or regenerate the wire-off pair too (+$0.55).
    - ~$0.83.
  - **(B) For C1-E: fold it into item 4.** Run the replication's 5 + 2 fresh seeds to 3,000 with evals at 2,000 /
    2,500. A mid-run eval leaves training bit-identical: tested locally, 60/60 train rows identical with vs without
    `--eval-at 30`. So the registered 2,000 readout is unchanged, and the 3,000 question gets 5 seeds instead of 3.
    About +$0.7 over item 4.
  - **(C) Cheap but confounded:** warm-restart from the u2000 actor params (fresh Adam moments, PopArt
    re-initialised, new environment state), ~$0.5. The optimizer restart perturbs exactly the slow drift under test.
    I don't recommend it.
  - **Recommendation:** A (T3, declared NOSIG level) + B (E) ~ $0.83 + $2.1 = $2.9, plus 3b's VOI check $0.8 =
    **$3.7** (vs $2.9 approved). This waits for the credit check and Cam.

- **RESULTS -- step 6, compositionality pilot (registered; `comm-toy` `e48e98f`; `scripts/analyze_step6.py`
  unmodified; data `docs/data/step6/`): FAIL.**
  - **Gates:**
    - Gate 1 (topographic similarity) 5/5: passes, but trivially.
    - Gate 2 (held-out generalisation) 0/5: success on the held-out pairs is 0.000-0.025 per attempt vs chance
      1/16, and equals the shuffled-message null (only s214 differs from the shuffled null, and it is still below
      chance).
  - **Why gate 1 is hollow:** the sender uses only **4-7 distinct messages for 14 meanings**. rho is identical at
    1,000 / 2,000 / 3,000 in every seed, so the code froze early. Meanings that share a fire share a message, which
    alone gives rho > 0.
  - **Diagnostic (not registered): the sender's token policy goes deterministic within 10-15 updates.** Token
    entropy < 0.01 nats from update 10 / 11 / 10 / 10 / 15. MI(token; pair) is 0.65-0.91 bits at that point and
    only 1.2-1.4 bits at 3,000, against log2 14 = 3.81 needed. Listener correct per attempt at the end: 0.34 /
    0.34 / 0.07 / 0.23 / 0.27.
  - **Comparison (free):** step-2 VQ and C1 (full vocabulary) start at 9.5 nats of sender token entropy
    (3 slots), fall to 0.8-1.7 by update 20 and **never** go below 0.01. Step 6 starts at **1.09 nats at update 1**,
    before any learning: the warm-started z_e already sits deep in single cells of the 4 retained codewords. The
    early collapse is therefore specific to step 6's vocabulary cut against the warm-start codebook.
  - **Two candidate causes, not separated yet:**
    - **(i) Codebook geometry.** Vocab 4 at tau 0.08 leaves no sampling exploration, so PS moves z_e across a cell
      boundary only rarely (MI 0.66 -> 1.25).
    - **(ii) The material enters through the old noise slots.** In COMP the material is written into O_NOISE,
      which carried pure Gaussian noise throughout ORACLE0 training. The warm-started pathway has no learned use for
      it, and a saturated softmax gives PS little gradient to build one.
    - Evidence on (ii): in s212 / s213 / s214 the 14 meanings map to exactly 4 messages, consistent with fire only
      (rho > 0). In s210 / s211, 5 / 7 messages: the material splits a fire's message at most occasionally.
  - The failure is upstream of compositionality: there is no complete code to compose.
  - **Candidates for a rerun (not run; Cam's call):**
    - for (i): a token temperature matched to the vocab-4 codebook geometry, or a token-entropy floor early;
    - for (ii): the material on fresh dims (not the old noise slots), or no warm start for the comp pathway.
    - Either way, with "distinct messages >= 12 of 14" as a precondition gate before compositionality is read.
  - Spend ~$0.65 (metered $24.13 -> $24.78; 5 x 3,000 updates; no restarts).
- **North star: dictionary + "speak to them" (Cam's item 9.3; exploratory, $0, local; `comm-toy` `1873439`
  `--speak-test`, `scripts/analyze_speak.py`; data `docs/data/speak/`).**
  - **Method.** Frozen saved actor params. The dictionary maps fire -> the sender's modal triple ("dictionary") or
    each slot's modal token ("slot dictionary"), built from 4 frozen control rollouts on their own keys.
  - We then "speak": the listener hears our word for X = T + s instead of its sender, over 10 paired episodes. It
    is compared with the control (real sender), the sender's own triples for X ("own words") and babble (uniform
    random triples).
  - Readouts: listener Crafts at X, and **specificity** = Crafts at X / Crafts at any fire. Specificity is needed
    because babble makes listeners Craft everywhere: raw counts under babble reach 10-20 per world.
  - "Understood" (descriptive): for every s in 1..3, dictionary Crafts at X >= 0.8x own words, specificity >= 0.5,
    and specificity >= babble + 0.25.
    - **Disclosure:** this is my own descriptive rule, **revised after the first pass**. The first version had only
      the count (>= 0.8x own words, babble < 0.25x). The babble counts (10-20 per world) showed that a count is not
      specific, so I added specificity and the rows were re-run with Crafts at any fire logged. No registered bar is
      involved.
    - **Not broken down by target fire:** fire 0's modal-triple purity is 0.1-0.8% in every checkpoint (shared warm
      start). The per-slot dictionary covers it, but an average over targets could hide one word that fails.
  - **Understood in 12 of 14 checkpoints:**
    - C1-T3 x3 (u2000, pre-reset);
    - step-2 VQ s190-193;
    - C1-E x3 at u1500;
    - C1-E s185 / s186 at u2000.
    - **Saying "X" sends listeners to X:** ~18-27 Crafts at X per world (control 0.4-0.8), specificity 0.73-0.80
      (babble 0.26-0.33). Our dictionary word works as well as the sender's own words; the per-slot dictionary
      works as well as the modal triple.
  - **Not understood:**
    - C1-E s187 u2000 (eroded: triple MI 0.12 bit; no condition moves listeners);
    - step-2 VQ s194, an odd listener: babble itself sends it to T+2 (57.5 Crafts, specificity 0.83). That is where
      the criterion fails, even though our "T+2" gives 89 Crafts at specificity 0.98. For T+1 / T+3, our words and
      its own words are specific (0.67 vs babble 0.07) but few (~4.9 Crafts).
  - **Explains C1-E's puzzle.** In the eroded C1-E seeds s185 / s186 at u2000:
    - per-slot MI is 0.0-0.54 and triple MI 0.57 / 0.63 bit;
    - **our dictionary works *better* than the sender's own words** (18-20 vs 14.5-15.6 Crafts at X; specificity
      0.57-0.62 vs 0.55-0.57 in s185, 0.79 vs 0.78 in s186).
    - Erosion degrades the *sender's* use of the words. The *listener* still understands the original vocabulary.
      Reliance survives because the sender still sometimes says the right word, and the listener integrates those
      uses over steps.
  - **Caveat:** the triples are diverse (363-4,076 distinct triples, modal-triple purity often < 1%) while each slot
    is nearly pure. The words are per-slot tokens, and the three slots are redundant copies, not a composition.

- **REGISTRATIONS -- this account's remaining runs (before any launch; Cam's night rulings). Launch waits for the
  verified remaining credit.**
  - **Code:**
    - `comm-toy` `1f8e51d` (launcher PINNED_SHA); `comm-jit` `c242635` (jit launcher PINNED_SHA; plan `replicate`).
    - Default toy path re-verified bit-identical to PS0 after the speak-test and lookout additions (30 updates,
      common keys).
  1. **C1-E replication (item 4), plan `c1rep`.**
     - Wire-blind head, erosion protocol exactly as C1-E. Fresh seeds: wire-on 230-234, wire-off 235 / 236.
       2,000 updates.
     - Analysis `scripts/analyze_c1rep.py`, registered: reliance as C1. **PASS = reliance at 2,000 in >= 3 of 5
       AND pooled drop >= 30%** (1 - sum swap / sum control over all 50 wire-on empirical pairs) **AND wire-off MI
       < 0.5 in both.** Redirection reported.
     - **Each run also saves a full-state snapshot at 2,000** (`testbed/toy/snap/c1rep_*_u2000.pkl`). If Cam
       approves extending to 3,000, the continuation resumes from the real state.
       - Checked locally: snapshot at 25 + resume to 40 reproduces a straight 40-update run bit for bit, training
         rows and final swap probe, under erosion + wire-blind.
       - The registered 2,000 readout is unaffected (the snapshot consumes no RNG).
     - Price ~$0.6-1.4 (step 6 metered ~$0.045 / 1,000 updates; C1's estimate was ~$0.1).
     - **My prediction:** given C1-E's decay (item 2), reliance at 2,000 passes in 3 of 5 (~45%) and the replication
       fails on the count (~55%). The pooled drop >= 30% is likely (~65%).
  2. **Step 3b value-of-information check (item 6), plan `s3b_voi`.**
     - World: G 24, fires at (3,3) (3,9) (9,3) (9,9), lookout Chebyshev 1 around (18,18), reroll 10, warm start
       ORACLE0, PS off, wire off in both arms.
     - LOOKOUT-ORACLE (`--lookout-oracle`: an agent is informed while it **or its partner** stands at the lookout)
       s220 / s221 vs NOSIG (informed only while standing there itself) s222 / s223. 2,000 updates.
     - **Apparatus** (`--selftest-lookout`): PASS for both arms. Geometry: lookout -> every fire 16 steps, cluster
       centre (6,6) -> every fire 4. The informed bit and the need dims are nonzero exactly where the predicate,
       recomputed from positions, says. The everyone-at-the-lookout mutant FAILS in both arms.
     - Analysis `scripts/analyze_step3b_voi.py`. Deliveries by either agent per world, mean over 1,801-2,000.
       **QUALIFIES iff ORACLE >= 1.5x NOSIG AND every ORACLE seed > every NOSIG seed AND ORACLE >= 0.5 per world.**
       - The last two conditions are **my additions; Cam can strike them.**
       - Why the floor: in a 20-update local smoke, both arms deliver ~0 per world, and the warm-start policy stands
         at the lookout in <= 0.1% of agent-steps. If neither arm learns the world, a 1.5x ratio of two near-zeros
         means nothing.
     - Occupancy reported (agent-steps at the lookout; world-steps with either agent there). Price ~$0.4-0.8.
     - **My prediction:** does not qualify at 2,000 (~60%). Discovering the lookout and holding a lookout / runner
       division of labour from a 0.1% start is a hard exploration problem. NOSIG stays near blind search.
  3. **Real-net replication, shrunk (Cam's message 2), plan `replicate`.**
     - The confirmation's recipe exactly (450 updates, tau 0.08, craft cost 0.1, c_ps 2.015, ORACLE0 warm start,
       apparatus check at 2, MI halt K = 5 / floor 0.92 / two-probe halt, empirical-swap probes every 50 x 10
       pairs), **plus the wire-blind message pathway (`--wire-blind 1`, apparatus PASS earlier), `--log-diag`, and
       `--ckpt-every 50`** (params to `testbed/comm_jit/ckpt/`).
     - Fresh seeds 52 / 53 (unused on the real net: not in the protocol, not on the volume).
     - Primary and analysis unchanged: `scripts/analyze_confirm.py` (redirection on probes 350 / 400 / 450,
       swap >= 2x control and paired > 2 SE). **PASS = primary in 2 of 2; 1 of 2 = partial** (the third seed and
       NOSIG on the next account; historical NOSIG reference 4.26).
     - **Price (from Modal's per-app billing report):** the confirmation's app cost **$11.26 for its two runs**
       (A10G $10.86, memory $0.32, CPU $0.08), so **$5.63 per run** including evals and checks. `--log-diag` adds
       un-jitted gradient passes inside `t_update` (unmeasured), which leaves ~15% headroom for two runs under $13.
     - **Price probe and go rule:**
       - `rep_PS52` runs alone until ~10 updates are logged.
       - Projected per run = $5.63 x (its mean t_update over updates 3-10 / 35.2 s, the confirmation's).
       - `rep_PS53` launches only if 2 x the projection <= $12.5, so both runs complete through the 450 probe:
         one total-metered guard firing near update 430 would kill both primary windows.
       - Otherwise I stop and put options to Cam (drop `--log-diag`, which the confirmation didn't have; or one seed).
         The guard does not decide.
     - Spend guard on the jit app at baseline + $13.
     - **My prediction:** primary 1 of 2 (~45%), 2 of 2 (~35%), 0 of 2 (~20%). Wire-blind removes the coupling but
       not the sender crash seen in seed 50.
  - **Budget arithmetic.** Metered $24.78 now, + toy (1 + 2) ~$1.0-2.2, + real net <= $13 = **~$38.8-40.0**, plus
    item 3 if approved (~$0.4-1.4). That leaves < $1-2 of slack under $41. The real limit is still unverified, so
    the guard stays where it is until it is.

- **Backup pass 1 (Cam's rule before any account switch), 2026-11-08 ~18:00.**
  - The whole `throng-runs` volume on `rapidcrisp` is copied to `~/throng-backup/rapidcrisp_throng-runs/testbed/`:
    895 MB, 285 files. Verified against the volume listing by name and size, 0 mismatches:
    - `comm_jit` 11, `params` 3 (including the real-net ORACLE0 warm start), `toy` 106, `toy/params` 162, `toy/snap` 3.
  - Egress within the free allowance.
  - **Pass 2 is required after the last run on this account** (c1rep snapshots, s3b params, real-net checkpoints).

- **Cam's rulings, Nov 9 (pasted by the user; M's decision included).**
  1. **Credit verified.** M reported $18 left at 21:50 with the meter at ~$24.3, so the limit is ~$42.3 (~$17.5 left
     at $24.78). The user read **$17.8 left** on the billing page on Nov 8 ~19:30 with the meter at $24.73, which
     agrees (limit ~$42.5). **Guard stays at $41.0 total metered**, restarted before each launch.
  2. **Free checks accepted.** C1-T3's ~0 was the post-reset probe; the pre-reset re-probe passes 3/3. C1-E's 2,000
     pass stands as registered, but in 2 of 3 seeds it is a decaying code: **rung 3 is not shown until the 3,000
     readout.**
  3. **Item 3: B approved, A deferred to the next account, C rejected.** The C1-E replication continues all 7 seeds to
     3,000. The 2,000 readout stays the registered primary.
     - **Registered bar at 3,000:** reliance in >= 3 of 5 wire-on seeds AND wire-off MI (Miller-Madow) < 0.5 bit in
       both. No pooled-drop term. NOSIG level = the wire-off mean over the 50 updates before 3,000. Redirection
       reported. Added to `analyze_c1rep.py`.
     - **Implementation (disclosed):** the 7 runs go **straight to 3,000** instead of 2,000 + resume. Evidence that this
       is the same experiment:
       - in `comm_toy.py`, `--updates` is used only as the loop bound and for the final eval;
       - local check: a 30-update run with `--eval-at 20` reproduces a 20-update run row for row
         (31/31 train / MI / swap rows identical up to update 20);
       - mid-run evals and snapshot + resume were already shown bit-identical.
     - The 2,000 full-state snapshot is still written. Eval at 1,500 / 2,000 / 2,500, final at 3,000. Params saved
       at 1,500 / 1,750 / 2,000 / 2,500 / 3,000.
  4. **Approved, free: the listener rows' share of the message-head gradient in C1-E** (u1500 and u2000, wire-on
     s185-187), plus its cosine with the sender rows' gradient. Registered before computing:
     - Gradient = the first-epoch training gradient of `total_loss` at the saved params on fresh rollouts (4
       rollouts, every minibatch of a random permutation, ratio = 1, pre-clip). Every term is a row-masked mean, so
       g_all = c_S + c_L exactly, with c_X = (X's row fraction) x the gradient with alive = X's rows. The split
       residual is reported. c_ps = 0 at both checkpoints under the erosion protocol.
     - Message head = the step-B (iii) key set (gwt_comms_1, head_signal_slot0-2, codebook_0-2). Also reported
       without the codebooks.
     - **Share = ||c_L|| / (||c_L|| + ||c_S||)**, which is bounded in [0, 1] even when the cosine is negative. Also
       reported: ||c_L||^2 / (||c_L||^2 + ||c_S||^2), cos(c_L, c_S), and the projection c_L . g_all / ||g_all||^2.
     - **PopArt stats are not in the actor-only saves.** They are fitted by OLS of the GAE returns on the raw value-head
       output, iterated 5x. Approximate, disclosed. Exact cross-check later on the c1rep u2000 full-state snapshots
       (which hold the PopArt stats): same tool, exact vs fitted.
     - **Trigger (Cam):** if the share >= 30%, register for the next account: C1-E with the message head trained by
       sender rows only (step B's (iii) flag), wire-on x3 and wire-off x2. I read "the share" as **>= 0.30 in at least
       2 of the 3 seeds at either checkpoint**; Cam can amend.
     - Tool: `comm_toy.py --grad-share` (eval-only; exits before training; the default path is untouched).
  5. **Speak test accepted as descriptive** (with the specificity disclosure). **It meets translation-track step 2's
     gate in the toy** (our word on par with the real sender's). Recorded: in the eroded C1-E seeds our dictionary word
     beats the sender's own (18-20 vs ~15 Crafts at X), so **erosion is on the production side**; the listener keeps
     the vocabulary.
  6. **Step 6 FAIL accepted. Rerun one arm on the next account:**
     - keep ORACLE0 for navigation;
     - re-initialise the message head and the material input rows;
     - set tau so the starting sender token entropy matches step 2's (~9.5 nats summed over slots, update 1);
     - **precondition gate:** >= 12 distinct messages (of 14 training meanings) at update 1,000 in >= 4 of 5 seeds,
       otherwise stop and report before reading compositionality.
  7. **3b VoI: my two added conditions accepted.** It runs on this account after the replication launches if the probe
     leaves room, otherwise on the next account. If it fails on occupancy, the next design gives agents a reason to
     pass by the lookout, not a bigger reward for information.
  8. **Real-net PS x2 (seeds 52 / 53) approved as planned, with the probe gate.** Order: real-net probe, then the C1-E
     replication, then VoI.
  9. **North star.**
     - Crystallization on the production side, tied to the message head's plasticity (hearing stays plastic), and
       learning-speed variety across agents: agreed. **M approved both traits as permanent parts of their nature:
       message-head plasticity that settles with age, and varied learning speeds across agents. Design them into the
       separate-brains stage** (after the toy repeat).
     - **Consequence to plan for (Cam):** with crystallization plus copying, wire-off codes will persist too (T3 showed
       copying alone keeps them), so the wire-off control stops discriminating. The bar moves to **reliance across
       generations, plus adapting when the world changes** (e.g. the fires move): a living code persists while it is
       useful and changes when the world does.
     - Order: the C1-E replication, then separate brains with rotating partners and varied learning speeds.

- **Real-net probe, first attempt (2026-11-08 19:34): out of GPU memory, ~$0.1.**
  - `rep_PS52` on c242635 crashed in update 1, **in `--log-diag`, not in training**: the PPO update completed
    (64 minibatches); the eager (un-jitted) gradient of the loss on the 2,048-row diag sample then exhausted the A10G
    (`RESOURCE_EXHAUSTED`, every allocator chunk in use, so not fragmentation).
  - **Apparently never run on the GPU before:** its verification (training bit-identical, smoke values) was recorded as
    free work, i.e. a local CPU run. A first-update GPU smoke of every new flag belongs in the config audit from now on.
  - **Fix (comm-jit 7366052, launcher re-pinned):** the same loss on the same sample, jitted once; the KL forward is
    jitted; an unused eager loss evaluation was removed. **Update-1 self-check:** the jitted vs the eager gradient on a
    256-row sub-sample, relative difference <= 1e-3, otherwise the run stops (exit 4). The training path is untouched.
  - A local CPU run of the real net was too slow for the laptop rule (> 10 min at update 1) and was stopped. The
    relaunched probe is the GPU test.

- **Real-net probe, second attempt (7366052): out of memory again, ~$0.1.** The update-1 self-check PASSED (jitted vs
  eager gradient on 256 rows: rel. diff 2.0e-5), but the jitted gradient on the full 2,048-row sample needs **one 15.3
  GiB buffer** (memory scales with rows: ~7.6 MB per row in this network; training's 512-row minibatches fit).
  - **Fix (comm-jit 80e3d6e, re-pinned): the diag gradient is the mean of the gradients on the sample's four 512-row
    chunks; the KL forward is chunked the same way.** **Definition change, disclosed:** the PS term is a batch
    statistic, so `ps_grad_share` is now the mean training-minibatch gradient's share (what the optimizer sees per
    step), not the 2,048-row gradient's. `grad_norm` is likewise the norm of the mean minibatch gradient. Neither is a
    bar. Training path untouched.
  - Cumulative cost of the two failed starts ~$0.2. **If the third attempt fails, I stop and put "drop --log-diag" to
    Cam** rather than iterate on paid GPU.

- **Item 4 result (free): the trigger fires.** `toy_out_modal/grad_share/analysis_grad_share.txt` (copy in
  `docs/data/grad_share/`). Listener rows' share of the C1-E message-head gradient, ||c_L|| / (||c_L|| + ||c_S||):

  | seed | u1500 | u1750 (extra) | u2000 | cos(c_L, c_S) at 1500 / 1750 / 2000 |
  |---|---|---|---|---|
  | s185 | 0.61 | 0.43 | 0.41 | -0.01 / -0.06 / -0.15 |
  | s186 | 0.63 | 0.45 | 0.57 | -0.03 / -0.32 / -0.22 |
  | s187 | 0.62 | 0.49 | 0.40 | +0.03 / +0.05 / -0.01 |

  - **>= 0.30 in 3/3 seeds at both registered checkpoints.** The split is exact (residual ~2e-7). Without the
    codebooks the shares are the same to 3 decimals: the listener-row gradient lands on gwt_comms_1 and the slot heads.
  - **Descriptive only:** listener rows contribute 0.7-1.7x the sender rows' gradient magnitude on the message head,
    cosine with the sender rows' gradient -0.32 to +0.05.
  - **Split-half noise check (exploratory, after the result; `splithalf.txt`):** c_L and c_S each recomputed from two
    independent halves (2 rollouts each). The halves barely agree: split-half cosine of c_L -0.36 to +0.41, of c_S
    -0.35 to +0.23, of g_all -0.07 to +0.29. **At this sample size the message-head gradient is noise-dominated on both
    sides**, so the share measures the relative size of the two sides' sampling noise, and a near-zero cosine is what
    independent noise gives. **No mechanism claim** ("the listener pushes against the sender") is supported. What
    stands: about half of the per-step random walk on the code-carrying weights comes from listener rows, whose tokens
    nobody reads under the wire-blind head. The stop-grad arm removes exactly that half; it remains a fair test.
  - Caveat: PopArt stats are fitted (actor-only saves). The exact cross-check on the c1rep u2000 snapshots (seeds
    230-234) validates the fitting method; it does not re-measure seeds 185-187.
  - **Registered for the next account (Cam's trigger):** C1-E with the message head trained by sender rows only (step
    B (iii): the MSGK keys take the sender-row gradient rescaled by the sender row fraction, i.e. exactly c_S), wire-on
    x3 / wire-off x2, fresh seeds, the C1-E protocol otherwise, the C1-E replication's 2,000 bar (and its 3,000 bar).
    **Code needed:** the (iii) flag currently runs only as a branch from a resumed snapshot (`--resume --branch
    stopgrad`); a from-scratch flag plus its apparatus check (listener-row message-head gradient = 0; mutant caught).

- **Real-net probe, third attempt (80e3d6e, app ap-IUr76MOVkalziEh1gJv8WC): running.** Diag self-check PASS (rel. diff
  2.0e-5), apparatus PASS at update 2, no OOM. **Analyse only `rep_PS52_80e3d6e.jsonl`** (the volume also holds the two
  failed starts' files).
  - **Go rule: PS53 NOT launched.** t_update over updates 3-10 = 39.5 s, plateau 39.7 s from update 10 (the
    confirmation: 35.2-35.5 s over the same windows), so the projection is $5.63 x 39.7 / 35.2 = **$6.35 per run; 2x =
    $12.7 > $12.5.** With the failed starts ($0.135 by the per-app billing report) option (a) totals ~$12.8 of Cam's $13.
  - Options put to Cam: (a) PS53 as configured (~$12.8 of $13; total metered ~$39 of $41); (b) PS53 without
    `--log-diag` (~$12.1): the same experiment, because the diag branch draws its own sample from
    `np.random.default_rng(u)`, calls pure jitted functions, and writes nothing back into params, optimizer, PopArt or
    any RNG that training uses; (c) PS53 dropped (PS52 alone = partial at best). **Recommend (b).**
- **C1-E replication launched** (plan `c1rep`, 7 runs straight to 3,000; pin 1f8e51d), after the probe, per Cam's order.
- **3b VoI launched** (plan `s3b_voi`, 4 runs). Room: metered $25.1 + PS52 rest ~$6.2 + c1rep ~$1.0 + PS53 if
  approved ~$6.35 + VoI ~$0.4 = ~$39 < $41.
- **If the Mac sleeps the local guard stops.** Worst case without it: the runs end on their own (PS52 projected ~$6.35;
  its function timeout is 8 h = ~$9; toy runs ~$0.05 per 1,000 updates), so metered stays < ~$36 without PS53.

- **Cam, Nov 9 early.**
  1. **PS53: option (b).** It launches now without `--log-diag`. PS52 keeps the diagnostics, enough for a crash
     post-mortem on one seed. Same registered experiment and bar (primary in 2/2 = PASS, 1/2 = partial).
  2. Accepted: the OOM fix and the `ps_grad_share` redefinition (not a bar); C1-E straight to 3,000; VoI launched.
  3. Grad share: the trigger fired (accepted); withdrawing the "opposing push" reading was right. **The stop-grad arm
     runs first on the next account**, with its from-scratch flag and **apparatus check: only sender rows send
     gradient into the message head, and a mutant that leaks one listener row must fail.**
  4. **Separate-brains stage:** if the code weights mostly random-walk, crystallization (message-head plasticity
     falling with age) damps that walk directly, and separate brains remove the listener half by construction. **Log
     gradient noise and code drift** so each trait's effect can be measured.
  5. **Next account, in order:**
     1. backup pass 2;
     2. move the spend guard into a small scheduled Modal function, so it doesn't depend on the Mac (check
        feasibility and price first);
     3. the stop-grad arm;
     4. C1-T3 to 2,750 (option A);
     5. the step-6 rerun;
     6. the 3b arms, if VoI qualifies;
     7. the third real-net seed plus NOSIG;
     8. a 2-minute real-net timing on an A100 vs the A10G (cost per run, future runs only).
  6. Each batch is analysed on landing with the registered scripts.
  - Launcher: `REPLICATE` adds `--log-diag` for seed 52 only.

- **Step 3b VoI result (registered analysis, `analyze_step3b_voi.py`; 4 runs complete, 2,000 updates each): DOES NOT
  QUALIFY.**
  - Deliveries per world at 2,000: ORACLE 0.00 / 0.00, NOSIG 0.00 / 0.00 (ratio 0.76; every-ORACLE > every-NOSIG
    false; floor false). `docs/data/s3b_voi/`.
  - **Two failures, not one** (descriptive, from the train rows):
    - **Occupancy at chance:** lookout agent-steps 0.015-0.016 at the end. The lookout is 9 of 576 cells (0.0156),
      so this is uniform wandering; no lookout behaviour emerged. The ORACLE arm is informed only 3% of the time
      (either agent at the lookout), so it cannot show value either.
    - **The base task collapsed:** Craft attempts fall from ~1-4 per world (updates 1-10) to ~0 by update 11-100 in
      all four runs, and deliveries stay ~0.005 per world throughout. With craft cost 0.45 and success rare in the
      24-grid, the warm-started policy learns not to Craft. Informed or not, nobody delivers.
  - **Reading for Cam's rule** ("if it fails on occupancy, give agents a reason to pass by the lookout"): it fails on
    occupancy, but a lookout reason alone would not rescue it, because informed agents don't deliver either. The next
    world design has to keep the base task alive (e.g. a lower craft cost in the 24-grid, or fires nearer the
    spawn area) *and* route agents past the lookout. That is Cam's design call. No 3b arms on this account.
  - Cost ~$0.35 (per-app billing report not yet posted).

- **Cam, Nov 9 (VoI follow-up).** (1) The export for M goes to M's Drive folder `My Drive/CascadeProjects/throng/for_m/`
  with a combined `for_m_all.json`. Built in `docs/data/for_m/` (+ `docs/data/for_m.zip`). **This session cannot
  write the synced Drive folder** (macOS denies access to `~/Library/CloudStorage`, "Operation not permitted"), so the
  user copies it. (2) and (3) below.
- **Culture tests (Cam, recorded):** toy seeds converge on the same words because they share the ORACLE0 start, though
  ORACLE0 itself has no code. A shared starting brain acts like a shared body that makes some words easier. **The
  separate-brains and culture tests need different starting weights**, or newborns' words will match elders' by
  default.
- **REGISTERED (before data), for the next account: step 3b VoI, retuned world (Cam: retune the world, not the
  agents).**
  - World: G 24, fires (3,3) (3,9) (9,3) (9,9), lookout Chebyshev 1 around (18,18), reroll 10, as before, plus:
    - **spawn box 1,1-11,11** (`--spawn-box 1,1,11,11`, new flag; start and respawn cells restricted to the fire
      cluster's region; 85 cells);
    - **craft cost 0.381 = half the blind-search break-even.**
  - **Free calculation** (`throng-toy/scripts/calc_s3b_world.py`; scripted agents in the exact world rules, 1,024
    worlds x 128 steps):
    - Blind searcher (tours the fires, one Craft at each): P(success per Craft) q = 0.220, so break-even
      **c\* = R q / (1 - q) = 2.7 x 0.220 / 0.780 = 0.762**.
    - At c = 0.381: blind net reward +48 per world per episode (11.9 deliveries); informed scripted 30.5 deliveries,
      2.56x blind.
    - Geometry rule kept: lookout to any fire 14 steps (+1 Craft) > reroll 10, so walking loses. A told partner in the
      spawn box needs 7.4 steps on average (max 16, so far corners can miss).
    - Every spawn cell is >= 12 steps from the lookout.
  - Arms as registered: LOOKOUT-ORACLE (informed while self or partner at the lookout) x2 vs NOSIG x2, 2,000 updates,
    ORACLE0 warm start, PS off, wire off. Analysis `analyze_step3b_voi.py`, with one extra condition.
  - **QUALIFIES iff ORACLE >= 1.5x NOSIG AND every ORACLE seed > every NOSIG seed AND ORACLE >= 0.5 per world AND
    (new, Cam) NOSIG >= 0.5 per world** (base-task floor; deliveries by either agent per world, mean over 1,801-2,000).
  - **Smoke gate before paying (<= 10 min, laptop):** both arms, 50 updates. **Pass iff mean Craft attempts per world
    over updates 41-50 >= 0.5x the mean over updates 1-10, in both arms.** Otherwise stop and report.
  - **Risks flagged before data:**
    - **Occupancy by chance is now ~0.** Every spawn cell is >= 12 steps from the lookout, so ORACLE gains only if
      agents learn to walk out. That is the real question, but a hard exploration problem.
    - **Warm-start navigation may not transfer to G 24.** The calculation shows the *old* VoI world was not over
      the blind break-even either (cost 0.45 vs c\* 0.78 with spawns anywhere), so its collapse was not the cost.
      Candidate cause: ORACLE0 learned on G 12, and the observation scales positions and fire offsets by G (pos / G,
      offset / (G/2)), so on G 24 every fire looks half as far. A free diagnostic (ORACLE arm, everyone informed,
      G 24 vs G 12, a few updates) follows and is reported as such; if navigation is broken, the remedy is Cam's call
      (it touches the observation, i.e. the body).

- **Retuned 3b world: registered smoke gate FAILS (laptop, $0); stopped as registered.**
  - Smoke (seeds 240 ORACLE / 241 NOSIG, 50 updates, the registered world): Craft attempts per world, updates 1-10 →
    41-50: ORACLE 0.31 → 0.00, NOSIG 0.38 → 0.00. Deliveries per world 0.04 → 0. The scripted blind searcher in the
    same world makes 11.9.
  - **Diagnostic (reported as such): the ORACLE0 warm start cannot navigate on G 24.** ORACLE arm (everyone always
    informed), seed 242, 3 updates, deliveries per world:
    - G 12 (the warm start's own world): 24.4 / 24.0 / 22.6;
    - G 12 with reroll 10: 23.8 / 23.1 / 22.1;
    - **G 24, same fire coordinates, spawn box: 0.12 / 0.03 / 0.00.**
  - **This is also the cause of the first VoI's collapse**, not the craft cost (the old world's blind break-even was
    0.78 > 0.45). ORACLE0 learned on G 12, and the observation scales positions and fire offsets by G (pos / G,
    offset / (G/2)), so on G 24 every fire looks half as far as it is.
  - Options for Cam (no 3b run until he picks):
    - **(a) A G 24 warm start:** train an ORACLE0-style navigator in the 3b world itself (ORACLE arm, PS off), ~1
      toy run, then the VoI on top. The agents learn their own world; the cleanest.
    - **(b) A constant observation scale** (positions and offsets divided by 12 on any grid): the world is unchanged,
      but the sensor changes and offsets beyond 6 exceed the trained range.
    - **(c) Shrink the layout into G 12:** impossible with the geometry rule (torus max 6 cells per axis; the lookout
      can't be 14+ steps from the fires).
    - Recommend (a). Its smoke gate is the same: Craft attempts don't collapse by update 50, and the ORACLE navigator
      delivers >= 0.5x the scripted informed level (30.5 per world) before the VoI starts.

- **Cam, Nov 9 (3b follow-up).** (1) No Drive copy needed; Cam read `for_m` from the repo with M's permission. (2) Option
  (a) approved: a G 24 navigator warm start. Gate: **navigator >= 0.5x scripted informed (>= 15.3 deliveries per world)
  and no Craft collapse by update 50.** Launch now only if the projected account total stays <= $40.0 metered,
  otherwise first on the next account. The VoI rerun on top goes to the next account either way. Break-even numbers
  and craft cost 0.381 accepted. (3) Design requirement (architecture section below). (4) Culture note acknowledged.
- **REGISTERED: the G 24 navigator** (plan `nav24`):
  - ORACLE0's own recipe moved to the 3b world: arm ORACLE (everyone always informed; no lookout flag), **from scratch
    (no warm start)**, PS off, G 24, fires (3,3) (3,9) (9,3) (9,9), spawn box 1,1-11,11, reroll 10, craft cost 0.381,
    seed 250, 2,000 updates (params auto-saved at 1,500 / 1,750 / 2,000).
  - **Gate:** (i) no Craft collapse by 50: mean Craft attempts per world over updates 41-50 >= 0.5x updates 1-10 (local
    smoke first, then the same rows in the paid run); (ii) **deliveries by both agents per world, mean over updates
    1,951-2,000, >= 15.3.** The saved u2000 params become the 3b warm start only if both hold.
  - Budget rule: launch only when the projection (metered + remaining real-net + remaining c1rep + ~$0.2) <= $40.0.
    At 21:03 it was ~$39.7, within the c1rep estimate's error, so the decision waits for c1rep to land (~21:50)
    and pin its cost.
- **Architecture requirement for our own world (W2+), Cam:** senses in **absolute local units with a limited
  range**, not scaled by grid size, so worlds can grow without breaking navigation. Option (b)'s direction long-term;
  no work now. (Root cause found in 3b: ORACLE0's pos / G and offset / (G/2) features made fires on G 24 look half
  as far.)

- **G 24 navigator: registered smoke gate (i) FAILS (laptop, $0). Not launched.**
  - Seed 250, from scratch, craft cost 0.381, 50 updates: Craft attempts near a fire per world 1.02 (updates 1-10) →
    0.03 (41-50); deliveries 0.24 → 0.01.
  - **Why:** a random beginner Crafts anywhere and pays 0.381 per failed Craft. On G 24 only 36 of 576 cells (6%) are
    next to a fire, versus 25% on G 12 where ORACLE0 learned, so the policy learns to stop crafting before it learns
    to navigate. The blind break-even (0.762) assumes a searcher that Crafts only at fires, which a beginner isn't.
  - **Exploratory smokes, same seed and world (after the gate failed; disclosed):**
    - cost 0: attempts 2.73 → 12.78, deliveries 0.55 → 2.16 (learning);
    - cost 0.1: attempts 1.62 → 3.30, deliveries 0.37 → 0.82 (learning).
  - **Proposal for Cam:** train the navigator at a **density-matched craft cost**: ORACLE0's 0.45 x (near-fire cell
    share G 24 / G 12 = 0.0625 / 0.25) = **0.11**. Same gates (no collapse by 50; >= 15.3 deliveries per world at
    2,000), then the VoI on top at 0.381 as accepted. Alternatively cost 0 for the navigator only. Next account
    (the gate failed, so nothing launches here).

- **Cam, Nov 9 (navigator).** Smoke FAIL and diagnosis accepted.
  - **REGISTERED for the next account: the G 24 navigator at the density-matched craft cost 0.11**, not 0 (which
    risks teaching craft-spam that then collapses at 0.381). Otherwise as the `nav24` registration: arm ORACLE, from
    scratch, PS off, G 24, the four fires, spawn box 1,1-11,11, reroll 10, seed 250, 2,000 updates. Gates unchanged:
    no Craft collapse by update 50 (local smoke first), and >= 15.3 deliveries per world (both agents, mean over
    1,951-2,000).
  - **The VoI on top stays at 0.381, with its own <= 10-minute smoke first.** That smoke must also show **no Craft
    collapse when the navigator moves from 0.11 to 0.381**. Same rule: mean Craft attempts per world over updates
    41-50 >= 0.5x updates 1-10, in both VoI arms, warm-started from the navigator's u2000 params. If it collapses,
    report before anything paid.

- **C1-E replication: RESULT (registered analysis `analyze_c1rep.py`; 7/7 runs complete, 3,000 train rows each, no
  restarts; `docs/data/c1rep/analysis_c1rep.txt`).**
  - **At 2,000 (the registered primary): FAIL.**
    - Reliance 2/5 (s231 +52%, s234 +69%; s230 +12%, s232 +13%, s233 +11%). The bar needs 3.
    - **Wire-off s236 MI 0.93** (>= 0.5; s235 0.001).
    - Pooled drop 37% (would pass), redirection 4/5.
    - My registered prediction was pass ~45%.
  - **At 3,000 (the registered continuation bar, Cam Nov 9): PASS.**
    - Reliance 3/5: s231 +80%, s232 +58%, s234 +51%; s230 +4%, s233 -1%.
    - Wire-off MI 0.001 / 0.133 (< 0.5 in both).
    - Redirection 3/5; pooled drop 47%.
  - **Per seed (descriptive), reliance drop at 1,500 / 2,000 / 2,500 / 3,000:**

    | seed | 1,500 | 2,000 | 2,500 | 3,000 | reading |
    |---|---|---|---|---|---|
    | s231 | +73 | +52 | +66 | +80 | held |
    | s234 | +70 | +69 | +45 | +51 | held |
    | s232 | +72 | +13 | +32 | +58 | **recovered** |
    | s230 | +76 | +12 | +5 | +4 | died |
    | s233 | +70 | +11 | -1 | -1 | died |

    Formation at 1,500: 5/5 reliance and redirection, MI ~2 bits.
  - **Reading.**
    - The registered primary fails. The C1-E 2,000 pass (3/3) does not replicate on fresh seeds.
    - Under the registered 3,000 bar, 3 of 5 seeds hold a code the listener relies on, 1,500 updates after PS was
      removed and with the erosion pressure still on. One of them (s232) recovered after nearly losing it.
    - **Two cautions before calling rung 3:**
      - The wire-off control discriminates only late: s236's PS-made code was still at 0.93 bits at 2,000 and
        decayed by 2,500.
      - At 3,000 the reliant seeds carry little sender MI (0.09-0.26 bits per slot, Miller-Madow) yet listeners lose
        51-80% of deliveries under the swap: a weak but consistently read code.
    - Rung 3 is Cam's call.
- **Grad-share PopArt cross-check (registered method validation; `docs/data/grad_share/xcheck.txt`):** c1rep u2000,
  seeds 230-234, exact PopArt (snapshot) vs fitted (OLS, the C1-E method).
  - **The fitted sigma is 9-12x too small** (16-20 vs 143-231).
  - The share still agrees within 0.19 (mean |diff| 0.08), and every exact share is >= 0.30 (0.37-0.58).
  - **So the C1-E 185-187 shares are approximate to about +-0.2, and the trigger stands.** The cosine is not robust to
    the fit (s231: exact -0.22, fitted +0.19), in line with the split-half finding.

- **Cam, Nov 9 ~02:40: C1-E replication ruling.**
  1. **Verdict: toy self-sustaining code MET, FRAGILE.** The u3000 readout decides, as agreed at 00:40. It passed
     at the minimum (reliance 3/5, pooled 47%, wire-off MI 0.001 / 0.13, redirection 3/5). Also recorded:
     - the u2000 primary FAILED (2/5), so the original C1-E pass at 2,000 does not replicate;
     - 2/5 seeds lost the code for good, so **a used code and silence are both stable end states**;
     - **this is not ladder rung 3**, which needs reliance across >= 3 generations of separate brains.
     - **New protocol rule: erosion tests read no earlier than 1,500 updates after PS ends** (s236 kept 0.93 bits
       through 500 erosion updates with no listener).
  2. **Free checks on the u3000 params** (laptop if each < 10 min, else Modal CPU; cap $0.30). Report-only except (a).
     - **(a) REGISTERED: same-spot swap control** on s231 / s232 / s234 (tool `comm_toy.py --same-spot`; analysis
       `analyze_same_spot.py`).
       - 10 pairs per seed (keys 10_000+k / 20_000+k, as the in-run probe).
       - The table is built per pair from the control rollout, as the in-run empirical swap.
       - **Cross-spot swap:** the listener hears a random triple the sender used for T' = T + 1, exactly the in-run
         probe; the empirical draw differs from the in-run one (disclosed before).
       - **Same-spot swap:** the same table, shift 0 (a random triple the sender used for the same lit spot T).
       - Drop = 1 - sum(swap deliveries) / sum(control deliveries) over the 10 pairs.
       - **The verdict is WITHDRAWN iff same-spot drop >= 0.5 x cross-spot drop in >= 2 of the 3 seeds.**
     - (b) Speak test at u3000 on the same three, as before.
     - (c) MI over all three slots jointly (plug-in and Miller-Madow), wire-on (s231 / s232 / s234) vs wire-off
       (s235 / s236) side by side; and MI restricted to steps before the listener's first Craft in each lit period.
       From the same control rollouts.
     - (d) Probe SEs for s232 at 2,000 / 2,500 / 3,000 (from the in-run swap rows).
     - (e) Export the s231 / s232 / s234 u3000 dictionaries to `docs/data/for_m/`.
     - (f) The original C1-E seeds have no u3000 readouts (they ran to 2,000), so there is no pooled count. No new
       runs.
  3. Grad-share cross-check accepted: the C1-E shares stand at +-0.2. **Cosines are dropped from future reports.**
  4. Spend on this account: nothing new beyond PS52 / PS53. Cam's projection is ~$38.4 at their finish; report if mine
     passes $40.5.
  5. **Next-account queue:** stop-grad moves up to right after the guard move.
     - Spec: wire-on x5 + wire-off x2 to 3,000, same protocol and bars as this replication.
     - **Adopt unless it does worse:** adopt iff >= 3/5 hold >= 30% reliance at both 2,500 and 3,000 (this
       baseline: 3/5) AND wire-off MI < 0.5 in both.
     - Price it from this replication's billed cost first; cap $1.5.
     - The result decides how the separate brains train their message heads.
  6. **Start now (code + unit tests only, laptop-sized): a registered design for the separate-brains toy.**
     - 4 brains with different starting weights: their own navigation warm starts, so no shared ORACLE0 word bias.
     - Rotating pairs, newborn copying, speech that settles with age, varied learning speeds.
     - Gates and a cost estimate. No paid runs until Cam signs off on the next account.

- **Free checks on the C1-E replication's u3000 params ($0, laptop; `docs/data/u3000_checks/`).**
  - **(a) Same-spot swap control (registered): the verdict STANDS** (0/3 seeds have same-spot drop >= 0.5x
    cross-spot).

    | seed | cross-spot drop (paired, SE) | same-spot drop (paired, SE) |
    |---|---|---|
    | s231 | +80% (4.45, 0.08) | +1% (0.06, 0.09) |
    | s232 | +58% (2.52, 0.07) | -1% (-0.04, 0.09) |
    | s234 | +49% (1.87, 0.06) | 0% (-0.01, 0.07) |

    Swapped triples differ from the original in >= 99.7% of steps in both modes. The listener reads which spot the
    message names, not merely whether it is an ordinary-looking message.
  - **(c) MI** (Miller-Madow, bits; 10 control rollouts, n = 81,920 sender steps):

    | | slots | joint (3 slots) | joint, steps before the listener's first Craft in the period |
    |---|---|---|---|
    | wire-on s231 | 0.05 / 0.00 / 0.70 | 0.79 | 0.81 |
    | wire-on s232 | 0.02 / 0.00 / 0.24 | 0.27 | 0.28 |
    | wire-on s234 | 0.20 / 0.08 / 0.20 | 0.23 | 0.22 |
    | wire-off s235 | 0.00 / 0.00 / 0.00 | 0.02 | 0.02 |
    | wire-off s236 | 0.05 / 0.19 / 0.15 | **0.25** | 0.24 |

    - **Explaining 0.09-0.26 bits per slot against 51-80% drops:**
      - (i) The per-slot mean hides concentration: s231's code sits in slot 3 (0.70 bits).
      - (ii) **The swap measures misdirection, not information.** A message for the wrong spot sends the listener
        there. With control deliveries only 3.8-5.6 against the NOSIG level 2.8, the code adds little, but a wrong
        message removes more than the code adds.
      - Restricting to steps before the listener's first Craft changes nothing (the information is not concentrated
        at decision moments).
    - **Caution for future bars: wire-off s236 keeps 0.25 bits jointly at 3,000**, as much as wire-on s232 / s234. Its
      per-slot mean (0.13) is under the 0.5 bar, but only reliance separates the arms, not MI.
  - **(b) Speak test at u3000:** our dictionary word steers listeners in all three (Crafts at X 5.6-17.9 vs control
    1.3-3.4) and equals or beats the sender's own word (s232 17.9 vs 7.6; s234 8.2 vs 5.0; s231 5.6-8.7 vs
    6.3-8.9). The descriptive "understood" rule passes 1/3 (s232): s231 fails on T+3, where babble is also specific
    (0.70); s234 is just under the 0.5 specificity line (0.42-0.50).
  - **(d) s232's recovery is real at the probe level.** Paired effect 0.51 (SE 0.11) → 1.32 (0.09) → 2.55 (0.07)
    deliveries per world at 2,000 / 2,500 / 3,000; steps of +0.81 (z 5.7) and +1.22 (z 10.4). The SEs are probe noise
    on one seed's params, not seed-level chance. The base also fell: control deliveries 10.2 at 1,500 → 3.9-4.4.
  - **(e)** Dictionaries exported to `docs/data/for_m/C1REP_s23{1,2,4}_u3000/` (README and `for_m_all.json` updated).
    The codes are diffuse (modal triple <= 1.4%, slot words 8-23%, read-back 4-95% by spot), and **each seed now uses
    its own tokens**: the shared ORACLE0-start words are gone.
  - **(f)** The original C1-E seeds ran to 2,000 only, so there is no u3000 pooled count.

- **Cam, Nov 9 ~02:55: free-check ruling.**
  1. (a) accepted: the same-spot control is clean (0/3); the listener reads which spot the message names.
  2. **The lead caution changes the verdict, pending one free check. REGISTERED (before computing):**
     - **Shuffle null for joint MI.** All 7 replication seeds at u3000, plus s236 at u2500. Same 10 control rollouts
       as the u3000 checks (keys 10_000+k / 20_000+k, n = 81,920 sender steps). Spot labels permuted 100x at the same
       N. Report the null mean and 95th percentile next to the Miller-Madow joint MI. Rationale: diffuse codes (modal
       triple <= 1.4%) over 64^3 triples can inflate MI beyond what Miller-Madow removes.
     - **Decision rule:** if s236's u3000 joint MI is <= its null 95th percentile, the control did its job and "met,
       fragile" stands. If s236 clears its null, the verdict becomes **"bar passed on the letter; the claim that use
       keeps the code alive is not shown."**
     - Either way: how far each wire-on survivor sits above its own null, and the joint MI of the dead wire-on seeds
       s230 / s233 at u3000.
     - Estimator applied: MM joint MI and plug-in, with the null computed with the same estimator on permuted
       labels. Shuffle-corrected MI = MI - null mean.
  3. **Protocol changes from now on:**
     - MI bars use **shuffle-corrected joint MI over all slots**, never the per-slot mean.
     - Reliance bars also require **a real gain: true-message deliveries >= NOSIG + 1.0 per world**, so misdirection
       alone can't pass.
     - An erosion test is valid only at an **erosion strength calibrated so that unused codes reliably die within the
       read window**.
  4. **Next account, new first toy item (before stop-grad): erosion calibration.**
     - Wire-off only. Start from the u1500 snapshots of s235 / s236 plus any other saved wire-off u1500 snapshot;
       regrow if fewer than 3 exist. (Today only u2000 full-state snapshots exist for s235 / s236, so **regrow is
       needed**: 3 wire-off seeds with `--snapshot-at 1500`.)
     - Message-head weight decay 3e-4 / 1e-3 / 3e-3, to u3000, shuffle-corrected joint MI every 250 updates.
     - **Registered pick:** the smallest strength at which every wire-off seed is below 0.05 bits shuffle-corrected
       joint MI by u2500.
     - Cap $1 after the 2-minute price probe.
     - **Then the rung-3 rerun at that strength:** C1 baseline wire-on x5, stop-grad wire-on x5, wire-off x3
       (stop-grad on).
       - Bars: reliance >= 30% at both u2500 and u3000 in >= 3/5 per arm; the gain bar above; wire-off below 0.05
         bits joint at u3000 in all three.
       - Adopt stop-grad unless it does worse than baseline. Price first; cap $3.
  5. **Stop-grad PS fix accepted:** the sender-row part at c_ps = 0, scaled; the PS gradient added unscaled.
     Apparatus check before any run:
     - the listener-row contribution to the message-head gradient is exactly 0;
     - the PS-term gradient on the message head is allclose to flag-off at update 1 (rtol 1e-4,
       atol 1e-6 x max|g|);
     - the old scaled-PS version must fail that check.
  6. (b), (d), (e) noted. **s232 is recorded as rising reliance on a weaker base (control 10.2 → ~4), not a rebuilt
     code.** The separate-brains design continues, code + tests only.

- **Joint-MI shuffle null: RESULT (registered rule applied; `docs/data/mi_null/analysis_mi_null.txt`; $0, laptop).**
  Miller-Madow joint MI over the 3 slots (same 10 control rollouts, N = 81,920) vs 100 spot-label permutations:

  | checkpoint | MM joint MI | null mean | null p95 | shuffle-corrected |
  |---|---|---|---|---|
  | wire-on s231 u3000 (survivor) | 0.787 | 0.122 | 0.126 | **0.665** |
  | wire-on s232 u3000 (survivor) | 0.266 | 0.029 | 0.031 | **0.237** |
  | wire-on s234 u3000 (survivor) | 0.225 | 0.013 | 0.014 | **0.212** |
  | wire-on s230 u3000 (dead) | 0.033 | 0.030 | 0.033 | 0.003 |
  | wire-on s233 u3000 (dead) | 0.131 | 0.102 | 0.105 | 0.029 |
  | wire-off s235 u3000 | 0.017 | 0.016 | 0.018 | 0.001 |
  | **wire-off s236 u3000** | **0.253** | 0.025 | **0.027** | **0.228** |
  | wire-off s236 u2500 | 0.485 | 0.007 | 0.008 | 0.478 |

  - The u3000 values for the 5 seeds in the earlier check reproduce exactly (same rollouts).
  - **Rule: s236's u3000 joint MI (0.253) clears its null p95 (0.027), so the verdict becomes "bar passed on the
    letter; the claim that use keeps the code alive is not shown."**
  - With no listener, s236 keeps 0.228 corrected bits, as much as survivors s232 (0.237) and s234 (0.212). Only
    s231 (0.665) is clearly above it. The erosion pressure (wd 3e-4) is too weak to remove an unused code by 3,000,
    so survival under use isn't separated from survival without use. Hence the erosion calibration first on the next
    account (Cam's item 4).
  - Dead wire-on seeds: s230 0.003 and s233 0.029 corrected (s233's raw MM 0.131 is mostly estimator bias, null mean
    0.102).
  - The diffuse-code inflation Cam suspected is real but small at this N: null means 0.007-0.122 MM (largest where the
    triples are most diverse, s231 / s233).

- **Stop-grad from scratch: built and apparatus-checked (comm-toy, laptop, $0).**
  - Flag `--msg-grad sender`. Every parameter takes the usual all-row gradient, except the message head (MSGK), which
    takes (sender-row fraction) x the sender-row gradient of the loss **at c_ps = 0**, plus the PS term's gradient
    **unscaled** (Cam's fix).
  - **Bug found by the check:** `total_loss` ignores its c_ps argument unless `--ps-anneal` is set (it then uses
    `--ps`). Added an explicit `ps_force` override (never set on the default path). The self-test asserts the
    protocol config (`--ps-anneal` on).
  - **Apparatus check `--selftest-sg2`: PASS** (ORACLE0 start, update-1 minibatch, c_ps 1.99, sender-row fraction
    0.52):
    - perturbing every listener row's advantage / return / value / log-prob / actions / tokens leaves the MSGK
      gradient **bit-identical**;
    - flag-off changes it, and the one-listener-row-leak mutant changes it;
    - the PS gradient on MSGK is allclose to flag-off (rtol 1e-4, atol 1e-6 x max|g|; norm ratio 1.0000);
    - **the old scaled-PS composition FAILS** (norm ratio 0.523 = the sender fraction).
  - **Default path bit-identical** to the committed code (12 updates, 19/19 rows). The flag changes training as
    intended.
  - Cost: 1.19x the default update time (laptop, sequential).
- **Prices for the next account** (from the replication's billed cost):
  - c1rep: $1.019 posted for 20:13-21:00 → ~$1.55 for 72 min, i.e. **~$0.22 per 3,000-update run** (~$0.074 per
    1,000 updates, evals included). Firm figure when the 21:00 hour posts.
  - Erosion calibration (regrow 3 wire-off seeds to 1,500 with `--snapshot-at 1500`, then 3 strengths x 3 seeds
    1,500 → 3,000): ~$1.3, over the $1 cap. **To fit: stop at u2500**, where the registered pick is read (~$1.0).
  - Rung-3 rerun (baseline x5 + stop-grad x5 + wire-off x3 with stop-grad, to 3,000): 5 x 0.22 + 8 x 0.22 x 1.19 =
    **~$3.2, over the $3 cap.** Options: wire-off x2, or stop at u3000 without the u2500 eval. Cam's call.
  - Stop-grad alone as first specified (x7): ~$1.85 vs its $1.5 cap; superseded by the rung-3 rerun plan.

- **DESIGN FOR SIGN-OFF (Cam item 6): the separate-brains toy. Code + unit tests only; no paid run until Cam signs
  off on the next account.**
  - **Built (comm-toy):**
    - `scripts/brains.py`: population mechanics as pure functions.
    - `tests/test_brains.py`: **11/11 pass**; each property is paired with a mutant that must fail it:
      - pairs are balanced (each ordered pair W / 12 times per update) and reshuffled every update;
      - the brain-aware forward equals each brain's own forward;
      - **gradients stay inside a brain** (brain-0 rows put exactly 0 on brains 1-3; a shared-brain mutant leaks);
      - the clip is per brain (a global clip would shrink the others);
      - the plasticity-by-age schedule and the per-brain learning speed are applied to the Adam *updates* (Adam ignores
        gradient scale), and a no-age-schedule mutant is detected;
      - vmapped per-brain Adam equals independent optimizers;
      - a newborn reset touches only its brain (parameters and Adam state);
      - imitation draws only from elder senders (never the newborn itself, a younger brain, or a listener row);
      - the message-head gradient rule ('all' / 'sender' rows).
    - Brain-aware forward = compute-all-brains-and-select. At 512 rows it costs 2.7x a plain forward + gradient
      (per-row vmap would cost 8.7x). Gradient isolation is exact.
    - Not yet wired into the training loop (the integration follows sign-off, with its own apparatus checks).
  - **Stage 0a: navigators (different starting weights; no ORACLE0 word bias).** 8 G 12 navigators by ORACLE0's own
    recipe (arm ORACLE, from scratch, PS off, craft cost 0.45, 1,500 updates), seeds 260-267: 4 for the founders, 4
    for newborns (a newborn always takes a navigator no living brain started from).
    - Gate N1: each navigator's deliveries per world over its last 50 updates >= 0.7x ORACLE0's (the same rule as the
      confirmation's warm-start check).
  - **Stage 0b: word-bias gate G0 (Cam's shared-body concern, tested directly).** The C1 recipe (wire-blind,
    PS formation, to 1,500) as isolated pairs: navigator A x 2 seeds, navigator B x 2 seeds.
    - Same-start pairs should share slot words: the positive control, as the ORACLE0 seeds did (3 of 4 fires
      identical).
    - **Pass iff different-start pairs agree on slot words no more than a fire-label-permutation null (95th
      percentile).**
    - If different starts still converge, the bias isn't from the start, and the culture test needs rethinking before
      it runs.
  - **Stage 1: the population.**
    - World: G 12, 4 fires, assigned sender / listener per pairing, wire-blind head.
    - **B = 4 brains, W = 48 worlds** (each of the 12 ordered pairs in 4 worlds, reshuffled every update).
    - **PS only for the founders' formation** (c_ps 1.99 annealed to 0 over 1,000-1,500, as C1). **Never for newborns.**
      Message-head weight decay from 1,500 at the **calibrated erosion strength** (next account's first item, so that
      unused codes reliably die).
    - **Generations:** lifetime 2,000 updates. Founders are replaced at 1,500 / 2,000 / 2,500 / 3,000 (staggered), then
      every 2,000, so every lineage slot turns over at 1,500-3,000, 3,500-5,000 and 5,500-7,000. **At 7,500 every
      living brain is generation 4 and none ever met PS.**
    - Newborns: a fresh navigator start and a fresh Adam state.
      - Copying: imitation of elder senders' (obs, token) pairs, message head only, during the sensitive period.
      - **Speech that settles with age:** message-head plasticity 1.0 for the first 1,000 updates of life, then linear to
        0.1 over 500; hearing stays plastic.
      - **Varied learning speeds:** a fixed per-lineage multiplier 0.5 / 0.75 / 1.0 / 1.5.
    - Message-head gradient rule per brain: `all` or `sender`, **set by the stop-grad result** on the next account. With
      rotating pairs each brain also listens, so its own listener rows would train its own message head unless `sender`.
      Separate brains alone don't remove that half.
    - Arms: POP (wire on) x3 seeds; POP-NOSIG (wire off) x2.
  - **Bars (registered for sign-off), at 7,500, all ordered pairs:**
    - **Rung 3 (primary):** in >= 2 of 3 POP seeds:
      - cross-spot reliance >= 30%;
      - **real gain:** true-message deliveries >= POP-NOSIG + 1.0 per world;
      - **same-spot control** drop < 0.5x cross-spot.
    - Code liveness: shuffle-corrected joint MI of POP-NOSIG < 0.05 bits at 7,500. If it fails, the erosion strength
      wasn't enough, and the primary is read only through reliance and gain.
    - Reported: which founder's words (if any) survive to generation 4; per brain, the agreement of its slot words with
      the population's.
  - **Adaptation (secondary; design question for Cam).** In this toy, moving the fires is transparent to the code: the
    observation gives every fire's offset by fire index, so a word for "fire k" still works after fire k moves.
    **Proposal:** reserve a 5th fire's dims from the start, never lit until 7,500, then lit like the others. Pass iff by
    9,000 listeners redirect to fire 5 when told (cross-spot swap on fire 5) in >= 2 of 3 seeds: a new word coined by
    generation-4 brains with no PS.
  - **Logged per brain, for each trait's effect (Cam Nov 9 item 4):** message-head gradient norm, its split-half
    cosine every 250 updates (gradient noise), message-head displacement per update, code drift (share of its modal
    slot tokens changed vs 250 updates earlier), age, generation, plasticity, learning-speed multiplier.
  - **Cost estimate** (toy billed rate $0.074 per 1,000 updates, evals included; population update ~1.6x a toy update
    from the 2.7x network factor and W 48 / 64; to be re-timed on the laptop after integration):
    - navigators 8 x 1,500 = ~$0.9;
    - G0: 4 x 1,500 = ~$0.45;
    - population: 9,000 updates x 1.6 x $0.074 / 1,000 = ~$1.07 per run, x5 = ~$5.3;
    - **total ~$6.7**, after the erosion calibration and the stop-grad decision.

- **Real-net replication (wire-blind PS x2, seeds 52 / 53): RESULT. PRIMARY PASS, 2/2** (registered
  `scripts/analyze_confirm.py`; `docs/data/realnet_rep/analysis_confirm_rep.txt`). Both runs completed 450 updates
  with no halts (apparatus at 2 and MI check at 5 passed in both).
  - Primary (probes 350 / 400 / 450, 30 pairs), listener Crafts at T' per world, control → swap:
    - **s52: 1.93 → 31.40 (x16.3), paired +29.47, SE 1.28;**
    - **s53: 2.52 → 21.66 (x8.6), paired +19.14, SE 2.44.**
  - Reliance (deposit drop under the swap) at 350 / 400 / 450: s52 +50 / +67 / +61%; s53 +34 / +36 / +61%. It rose
    across probes in both (from +10-16% at 50). MI ~2 bits throughout (null p95 <= 0.05).
  - **Descriptive caveats:**
    - **s53's probe 400 is degenerate:** control deposits 0.37 per world and Crafts at T' 0.35 → 2.58, against the
      training window's 5.74 listener deliveries. The pooled primary passes with it included. The cause is not
      investigated (a probe-rollout oddity or a transient policy state).
    - **Cam's new gain rule** (adopted after this was registered, so descriptive here): listener deliveries per world
      in the training window vs the historical NOSIG reference 4.26 + 1.0 = 5.26. s52 5.86 / 6.45 / 5.85 (passes at
      every primary probe). s53 5.57 / 5.74 / **3.35** (fails at 450, with a late dip; sender window minimum 9.3).
      There is no NOSIG arm on this account; the next account's third seed + NOSIG gives the real reference.
    - PS52 carries `--log-diag` (the chunked-diag definition), for post-mortems.
  - This is the real-net counterpart of the shared-brain code with the wire-blind fix: the code forms and is used
    with PS on (PS is never removed in this 450-update recipe). It is not an erosion or rung-3 result.
  - **Account spend final: $38.44 metered** (Cam's projection ~$38.4; the guard never fired and exited at 01:33
    with no live apps). Real net this account: two failed starts $0.135 + PS52 + PS53.
- **Backup pass 2 (Cam's rule before any account switch), 2026-11-09 ~01:40: VERIFIED.** The whole `throng-runs`
  volume was re-copied to `~/throng-backup/rapidcrisp_throng-runs/testbed/`: **373 files, 1.35 GB, 0 mismatches**
  by name and size (walked per directory; sizes are integers in the listing).
  - `params` 5 (incl. rep_PS52 / rep_PS53 finals), `comm_jit` 14, `comm_jit/ckpt` 18 (both seeds every 50),
    `toy` 117, `toy/params` 209, `toy/snap` 10 (incl. the c1rep u2000 snapshots).
  - Nothing remains only on Modal.

- **Cam, Nov 9 ~07:00: account close-out and next-account plan.**
  1. **Shuffle null accepted. Toy rung 3 = "bar passed on the letter; use keeping the code alive NOT SHOWN".**
     s236's decay (0.93 at u2000 → 0.48 at u2500 → 0.23 at u3000, roughly halving every 500 updates) is the
     unused-code clock at wd 3e-4; the calibration has to beat it.
  2. **REAL NET: rung 2 MET on the real network** (registered rule >= 2/3 seeds; 2/2 passed, so a third seed can't
     change it). Scope: formation + use with PS on, not erosion. Caveats recorded: s53's gain-rule dip (3.35 at
     u450) and the degenerate probe 400.
     - Probe-400 post-mortem: re-run that probe on the s53 u400 checkpoint with a new probe seed. Free only if it
       runs on the laptop in < 10 min, else next account (cap $0.5). Report only.
     - Third real-net seed + NOSIG, and the A100 timing: **DEFERRED** (the next stages run in the toy, then our own
       world).
  3. **Next account, in order (allowance $14, guard $15):**
     - **(a) Guard on Modal** as a scheduled function. M creates the token secret; M gets the exact steps; the agent
       never handles the token. Until it's live, the Mac guard runs as before.
     - **(b) Erosion calibration to u2500** as registered, cap $1.1. **The 3 wire-off seeds grow with stop-grad ON**
       (every later arm uses it). If no strength qualifies, report; don't extend.
     - **(c) Rung-3 rerun at the picked strength,** as registered: C1 x5, stop-grad x5, wire-off x3, fresh seeds,
       both readouts. **Cap $3.4**, all arms and the u2500 eval kept. The calibration seeds must not double as this
       control. If the picked strength leaves 0/5 survivors in BOTH wire-on arms, stop and report before (e).
     - **(d) Separate brains: navigators + G0 word-bias gate**, in parallel with (b), cap $1.4.
     - **(e) Separate-brains population runs after (c): APPROVED as designed, cap $6**, plus:
       - **control population:** the same design with the wire OFF and copying ON (copying alone could carry a code
         across generations). Bar: shuffle-corrected joint MI < 0.05 at the readout in every control seed;
       - message-head gradients follow (c): sender rows only, unless stop-grad does worse;
       - **fifth fire APPROVED as a registered secondary** (reported, not part of the rung-3 bar). Its offset slot is
         in the obs from update 0, it is never lit before 7,500, and PS never applies to it. Also report which age
         group coins its word (newborns in their sensitive period, or settled brains);
       - **ladder rung 3 is met only if the bars hold in gen-4 brains in >= 2/3 population seeds.**
     - **(f) Step-6 compositionality rerun** as ruled at 00:40 (re-init message head + material rows; tau for step-2's
       starting entropy; gate >= 12 distinct messages at 1,000 in >= 4/5). Price first, cap $1. Any time.
     - **DEFERRED:** C1-T3 to 2,750 (superseded by (e)); the 3b navigator / VoI (the G-scaled obs is a toy limit;
       rebuild the lookout test in our own world with absolute, limited-range senses).
  4. Accepted: the stop-grad build, its apparatus check, the ps_force fix, the population code and its 11 tests.
  - **Budget check (my estimates):** (b) ~$1.0 + (c) ~$3.2 + (d) navigators $0.9 + G0 $0.45 + (e) 5 runs ~$5.3 plus
    the control population (+2 runs ~$2.1) = ~$7.4 vs the $6 cap, so **(e) needs re-pricing or one fewer seed**, to
    be timed after integration + (f) ~$1 = ~$13.9 against the $14 allowance. (e) is the item to re-price first.

- **BILLING FINDING (Nov 9 ~07:10; Modal `billing summary --json`, rapidcrisp): credits -$30.00, billed cost
  $8.43**, metered $38.44.
  - At 19:33 on Nov 8 the summary showed credits = metered ($24.72), billed $0.00. So the account's credits were $30
    (it looks like the plan's monthly credit), and **$8.43 has been billed to the payment method**.
  - The "$17.8 left" / "~$42.3 limit" reading was a spending limit, not credit.
  - Spend stayed within Cam's $41.0 guard throughout; this is a fact for M, not a rule breach. The next account's
    allowance ($14, guard $15) should state whether it means credits or billed money.
- **(a) Guard on Modal: code ready, not deployed** (`scripts/modal_guard.py`; M's steps in `docs/GUARD_ON_MODAL.md`).
  - A scheduled function (every 2 min, 0.25 CPU) reads `modal billing summary --json` → `metered_cost`, lists live
    apps, and stops every app except itself when metered >= GUARD_CAP (default 15.0).
  - The CLI runs with only the secret's token (the container's own MODAL_* identity is dropped).
  - `::check` = a dry run that stops nothing.
  - The token lives only in M's Modal secret `throng-guard-token`.
  - Cost ~$0.03 / day.
  - Next account: the dry check first, then deploy. The Mac guard runs until the dry check passes.
- **Probe-400 post-mortem (Cam 2):** not laptop-feasible. On CPU a real-net rollout takes ~2-3 min (the M export:
  2 dictionary rollouts + 1 transcript rollout, with compile, in ~8.5 min), and the probe needs 20 rollouts (10
  pairs x control + swap). **Next account, cap $0.5.**

- **New account `roadtoagi111` (2026-11-09 ~02:10).**
  - The user set the profile in their own terminal; the agent never handled the token. A token had first been
    pasted into chat; the user was advised to revoke it and set a fresh one.
  - Verified: $0 metered, no apps; `throng-runs` volume created (empty). The toy warm start `toy_out/ORACLE0.msgpack`
    is in git, so toy runs need no upload. Real-net params stay in the backup (real-net work deferred).
  - Mac guard `~/throng-smoke/cap_guard_r2.sh` at CAP 15.0, started with the first launch, until the Modal guard
    is live (M's secret `throng-guard-token`).
  - **Nothing paid until Cam confirms whether the $14 allowance means credits only.**

- **New account: the user reports the workspace is "loaded with $42.5"**, the same figure as rapidcrisp before it
  billed past $30. Asked whether it is credit or a spending limit. **Cam's guard $15 is below $30 either way**, so the
  plan below is not expected to bill.
- **REGISTERED launch details, (b) erosion calibration and (d) navigators + G0** (toy launcher re-pinned to comm-toy
  `9c26a23`, which adds `--msg-grad sender`. The default path was verified bit-identical when that flag landed; the
  other additions since 1f8e51d are eval-only modes or off-by-default flags).
  - **(b) `ero_grow`:** 3 fresh wire-off seeds 270 / 271 / 272, the C1-E recipe with wire-blind and **stop-grad ON**, to
    1,500, full-state snapshot at 1,500.
    - Weight decay starts at **1,501** (not 1,500), so the snapshot is pre-erosion and every strength starts from the
      same state; disclosed, a 1-update shift.
    - **`ero_cal`:** each snapshot resumed at wd 3e-4 / 1e-3 / 3e-3 to 2,500 (9 runs), params saved at 1,750 / 2,000 /
      2,250 / 2,500.
    - **Shuffle-corrected joint MI every 250 updates is computed from those saved params on the laptop**
      (`comm_toy.py --same-spot --mi-null 100`, as for the u3000 checks; free, no in-run code).
    - **Pick (as registered):** the smallest strength at which every wire-off seed is below 0.05 bits by u2500. If none
      qualifies, report; don't extend. Cap $1.1 (estimate ~$0.7: NOSIG runs carry no swap evals).
  - **(d) `nav`:** 8 navigators, seeds 260-267, ORACLE0's own recipe (arm ORACLE, from scratch, craft cost 0.45, 1,500
    updates, wire-blind 0 as ORACLE0); final params to `testbed/toy/nav/`.
    - **Gate N1:** deliveries per world (both agents), mean over updates 1,451-1,500, >= 0.7x ORACLE0's 24.89 =
      **17.42**.
  - **(d) `g0`, after `nav`:** the C1 formation recipe to 1,500 (wire-blind, PS 1.99 annealed 1,000-1,500,
    wire-init 0.06, craft cost 0.45, `--msg-grad sender`) from navigator 260 (seeds 280 / 281) and navigator 261
    (seeds 282 / 283).
    - Analysis `analyze_g0.py` on the `--export-m` dictionaries of the u1500 params: agreement = share of the 12
      (fire, slot) cells with the same token.
    - **PASS iff the mean different-start agreement <= the 95th percentile of a fire-relabelling null** (4 cross pairs
      x 23 non-identity permutations). Same-start agreement is reported as the positive control.
    - Caps: nav + g0 together $1.4 (estimate ~$0.9).
  - Local smoke (laptop): grow → snapshot → resume at a different strength with saves; navigator → G0 warm start.
    All 4 chains run.
  - Guard: Mac guard `cap_guard_r2.sh` at **$15** until the Modal guard is live.

- **(b) grow done; (d) navigators done (roadtoagi111).**
  - `ero_grow`: 3/3 complete (1,500 updates, snapshots on the volume). Every wire-off seed formed a full code by 1,500
    (per-slot MI 1.99 in all three; c_ps annealed to 0). `ero_cal` (9 runs) launched ~02:37.
  - **`nav`: 8/8 PASS gate N1** (deliveries per world over the last 50 updates: 24.45 / 24.93 / 24.65 / 24.72 /
    25.34 / 22.48 / 24.66 / 24.97 vs 17.42; `docs/data/nav/`). Params in `testbed/toy/nav/`. **G0 (4 runs) launched
    ~02:40.**

- **Cam, Nov 9 ~07:45: billing and budget.**
  1. **From now on, allowances mean FREE CREDIT ONLY; nothing may be billed without M's explicit OK.** Guard cap =
     min(approved allowance, credit balance - $1). **The rapidcrisp $8.43 billed is recorded as Cam's planning error**
     (he treated the ~$42 spending limit as credit), not a rule breach.
  2. **(e) re-price:** the wire-off + copying control population also gives the NOSIG delivery baseline (wire-off
     listeners are no-signal listeners, as in c1rep), so there is no separate NOSIG population.
     - **My 5 runs were 3 POP + 2 POP-NOSIG.** In the registered design every newborn copies elders, so that NOSIG
       population already had copying ON: it *is* Cam's control. No design change; the ~$7.4 figure double-counted it.
  3. Probe-400 post-mortem: DEFERRED with the rest of the real net.
  4. Plan total ~$11.9 against $14. N1 (8/8) noted. Proceed: (b) pick and G0 by the registered rules, then (c).
- **Cam, Nov 9 ~07:50: own world (W2), spec first.** Code + tests only, no paid runs, in parallel with (b)-(f).
  `docs/OWN_WORLD_SPEC.md` for sign-off before any code beyond a prototype.
  - Requirements:
    - pure JAX, jit / vmap end to end, >= 100k agent-steps/s on one A10G (laptop CPU measured first);
    - **senses absolute, local, limited range** (a small window; no G-scaled positions);
    - **body and reward:** energy drains every step; food refills it; death at 0; reproduction at high energy;
      offspring inherit traits (learning speed, plasticity schedule) with mutation; **no task reward, a homeostatic
      reward** (Keramati & Gutkin 2014);
    - things worth talking about, staged behind flags: moving / depleting food patches; a lookout with a wide view;
      two predator kinds needing different escapes (the alarm-call test); facts with parts (kind x place);
    - the toy's wire (VQ, wire-blind message head, the stop-grad rule from (c)), with the swap, same-spot and speak
      probes ported.
  - **Prototype gates:** (1) a no-comm agent survives longer than a random one; (2) an ORACLE agent told where the
    food is lives >= 2x as long as a blind one (value of information); (3) steps/s measured. Build plan + cost to the
    first paid smoke.
- **Cam, Nov 9 ~07:55: budget raised.** M confirms the full ~$42 on this account is spendable.
  - **Guard raised to $41** (Mac guard restarted at 02:44, CAP 41.0).
  - Per-item caps are estimates: overruns up to 25% are fine without asking; anything more, tell Cam first.
  - The extra buys statistical power: **(c) wire-off x3 → x5; (e) 3 POP → 5 POP, 2 control → 3 control**, both
    re-priced before launch.
  - Keep ~$25 in reserve for the own-world prototype smoke and follow-ups.
  - **Re-price (estimates, $0.22 per 3,000-update toy run, stop-grad 1.19x):**
    - (c) C1 x5 + stop-grad x5 + wire-off x5 (stop-grad on) = 5 + 10 x 1.19 = 16.9 run-units = **~$3.7**;
    - (e) 8 population runs x ~$1.07 = **~$8.6**, to be re-timed after the population is wired in.

- **REGISTERED (before data): (c) rung-3 rerun launch details** (plan `rung3`; analysis `analyze_rung3.py`).
  - Fresh seeds: C1 baseline 290-294, stop-grad 295-299, wire-off 300-304 (stop-grad on).
  - The C1-E erosion protocol (erode-from 1,500) to 3,000 at the **picked weight decay** (env `RUNG3_WD`; the plan
    refuses to run without it). Evals at 1,500 / 2,000 / 2,500 + final 3,000; params saved at 1,500 / 1,750 / 2,000 /
    2,500 / 3,000.
  - **A seed HOLDS iff, at BOTH 2,500 and 3,000:**
    - reliance: drop >= 30%, paired > 2 SE, listener window >= the wire-off window mean (as C1 / c1rep);
    - **gain:** listener window >= wire-off mean + 1.0 per world.
  - **Arm PASS iff >= 3/5 hold. Wire-off bar:** shuffle-corrected joint MI < 0.05 at u3000 in all 5 (laptop
    `--mi-null` on the u3000 params).
  - **Stop-grad adopted unless (# stop-grad seeds holding) < (# baseline seeds holding).**
  - Note: the u2500 readout is 1,000 updates after PS ends, inside Cam's "read no earlier than 1,500 after PS"
    rule. Cam asked for both readouts here, and the calibrated strength is chosen so that unused codes die by 2,500;
    flagged, not changed.
  - Price: ~$3.7 (Cam's cap $3.4 + 25% = $4.25).

- **Cam, Nov 9 ~07:50: overnight mode (M asleep ~7 h).** Proceed (b) → (c) → (d) → (e) and (f) by the registered
  rules without waiting; the own-world spec and prototype code (code + tests) in parallel. **HOLD and wait for M if:**
  - a registered stop condition fires (e.g. (c) leaves 0/5 survivors in both wire-on arms);
  - **G0 fails** (hold (e));
  - a result needs a judgment call no registered rule covers;
  - **projected spend passes $25.**

  Leave a short morning summary for M: results against each bar, spend, and what's waiting on Cam.
- **Cam, Nov 9 ~08:00: own-world spec addendum (open-endedness).** The spec must leave room for all of these;
  prototype only (1), design hooks for the rest.
  1. **Generative physics, not a fixed object list:** materials are property vectors (heat, mass, nutrition,
     hardness, ...) with a small set of interaction rules (combine, burn, grow, decay, break). Products are materials
     again, so combinations compound. No named objects in any code path the agents see.
  2. Unbounded map: chunks generated on demand from a seed (hash-based), no edge; only nearby chunks simulated.
  3. World change by agents: place, store, build; changes persist.
  4. Channel flag for variable-length messages (symbol sequences with an end symbol), off for now.
  5. Drives: homeostatic energy, plus a curiosity drive behind a flag (reward = learning progress of the agent's own
     world model, not raw prediction error); inborn traits stay heritable ranges.
  6. Dreaming hook: each brain's world model trains on real data, and a sleep phase does policy updates inside it.
     Off until the rung-6 gates pass: it predicts real change, it beats no-dreaming, and no imagined action is ever
     executed.
  - Metrics to plan for: novelty (distinct material combinations per life); vocabulary in use; whether new words
    appear for new things without PS.

- **(f) step-6 rerun: code ready, launch HELD for Cam (judgment call; overnight rule).**
  - New flags (comm-toy): `--reinit-msg` re-initialises the message head (gwt_comms_1, head_signal_slot0-2) and enc_0's
    material input rows (O_NOISE..+4) after the ORACLE0 warm start; `--tau` overrides the token temperature. The
    default path is bit-identical (12 updates, 19/19 rows).
  - **The judgment call:** "tau so the starting token entropy matches step 2's". Step 2 started at 9.53 nats summed
    over 3 slots of 64 symbols; with step 6's 4-symbol vocabulary the maximum is 3 ln 4 = 4.16 nats, so the literal
    target is impossible.
  - Proposed reading: **match the fraction of maximum:** 9.53 / 12.48 = 0.764 → 3.18 nats.
  - Laptop calibration (update-1 summed sender entropy, `--comp --vq-vocab 4 --reinit-msg`): tau 0.08 → 0.65, 0.3 →
    2.15, 0.5 → 2.89, 0.6 → 3.12, 0.7 → 3.30, 1.0 → 3.63. **Proposed tau = 0.63.**
  - Otherwise as ruled: one arm, 5 seeds (fresh, 310-314), 3,000 updates; gate >= 12 distinct messages at 1,000 in
    >= 4 of 5 seeds, else stop before reading compositionality; then the step-6 analysis. Price ~$0.7-1.0, cap $1.
- **W2 prototype (laptop, code + tests):** `w2/env.py`, the world core (energy drain / eat / death / reproduction mask,
  food patches, egocentric absolute window, ORACLE offset, homeostatic reward). **6/6 tests pass**, including **the same
  local scene giving an identical observation at L = 64 and L = 256**.

- **(b) Erosion calibration: RESULT. Registered pick = weight decay 1e-3** (`analyze_erocal.py`;
  `docs/data/ero_cal/`). 9/9 runs complete. Shuffle-corrected joint MI (bits) of the wire-off seeds (stop-grad ON),
  from ~1.99 at u1500:

  | wd | s270 at 1,750 / 2,000 / 2,250 / 2,500 | s271 | s272 |
  |---|---|---|---|
  | 3e-4 | 0.955 / 0.515 / 0.127 / 0.009 | 0.883 / 0.025 / 0.025 / 0.003 | 1.008 / 0.340 / 0.199 / **0.143** |
  | **1e-3** | 0.245 / 0.000 / 0.000 / -0.001 | 0.227 / 0.002 / 0.000 / 0.000 | 0.072 / 0.002 / 0.000 / 0.001 |
  | 3e-3 | 0.000 throughout | 0.000 | 0.000 |

  - 3e-4 fails (s272 still at 0.143 at 2,500). **1e-3 is the smallest strength with every unused code < 0.05 by
    u2500** (in fact <= 0.002 by u2000). 3e-3 kills the code within 250 updates.
  - Cost: grow + calibration ~$0.6 (billing report to confirm).
  - **(c) launched at wd 1e-3** (`RUNG3_WD=1e-3`, plan `rung3`, 15 runs).

- **(e) Population: wired and apparatus-checked (laptop, $0); launch HELD on price (Cam's 25% overrun rule).**
  - `scripts/pop_toy.py` (comm-toy 5c205a0). The toy's world with B brains:
    - per-brain PPO / vmapped Adam / PopArt, per-brain clip;
    - the sender rule (sender-row gradient at c_ps = 0 x the sender fraction + PS unscaled, as `--msg-grad sender`);
    - founders-only PS (annealed 1,000-1,500); message-head weight decay from 1,500;
    - staggered newborns from navigators no living brain started from, with a fresh Adam / PopArt;
    - copying (cross-entropy on elder senders' tokens for their observations, message head only, during the sensitive
      period); plasticity by age; lineage learning speeds;
    - evals: cross-spot and same-spot empirical swaps with per-brain tables, per-brain words, token dumps.
  - **`--selftest`: PASS in all four configurations** (COMM / NOSIG x `--msg-grad sender` / `all`):
    - perturbing brain 0's rows leaves brains 1-3's gradients bit-identical and changes brain 0's;
    - under `sender`, perturbing listener rows leaves every message head's gradient bit-identical (under `all` it
      changes: the mutant has power);
    - imitation moves only the message head; the PS term is exactly 0 without founder rows;
    - NOSIG heard wire = 0.
    - Plus a 6-update end-to-end smoke with forced births.
  - **Price:** laptop 0.60 s per update at W 48 / T 128 / B 4, vs the toy's 0.21 s (2.8x, the compute-and-select
    factor). At the toy's billed rate, **~$1.9 per 9,000-update run; 5 POP + 3 control = ~$15** vs the $6 cap.
    Over by more than 25%, so **held for Cam.**
    - Options:
      - (i) shorter lives: 1,000-update lifetimes and a stagger of 250 reach generation 4 at ~4,000 updates, ~$0.85
        per run, ~$6.8 for 8;
      - (ii) 3 POP + 2 control at full length, ~$9.5;
      - (iii) a 2-minute L4 GPU timing probe (~$0.05), in case the small batched nets run cheaper per update on GPU.
    - **Projected account total if (e) ran as specified: ~$1.5 now + (c) ~$3.7 + G0 ~$0.3 + (e) ~$15 + (f) ~$1 =
      ~$21.5**, under the $25 hold line but not under (e)'s cap.

- **W2 prototype (laptop, $0; comm-toy 0df7a03).**
  - **Generative physics** `w2/physics.py`: materials as property vectors `[heat, mass, nutrition, hardness,
    moisture]`, five rules (combine, burn, grow, decay, break), products are materials. **5/5 tests:** closure,
    compounding (depth-2 materials the depth-1 set can't make), rule effects incl. mass conservation on break,
    jit / vmap, no object names in code.
  - **World core** `w2/env.py`: **6/6 tests**, incl. the same local scene giving an identical observation at L 64 and
    L 256.
  - **P2 premise (scripted, before learning):** first world (L 64, 6 patches sigma 2, drain 0.005): oracle / blind =
    1.20x, too much food to need telling. **Retuned the world (not the agents):** L 96, 6 patches sigma 1.0, drain
    0.008.
    - Mean lifespan of 3,000 steps (16 worlds x 64 agents): random 60.7, blind forager 187.3 (SE 15.5), oracle
      forager 994.8 (SE 41.6).
    - **Oracle / blind = 5.31x (P2 premise OK, >= 2x); blind / random = 3.09x** (room for P1).
  - **P3 throughput:** env + observation, laptop CPU, 64 worlds x 64 agents: **~7.9-9.8 million agent-steps/s**. The
    brains, not the world, will set the speed.
  - **Learned gates (laptop, `w2/learn.py`; a shared-brain PPO, MLP 2 x 128, homeostatic reward only, 400
    updates, 1 seed; frozen-policy lifespan test as the scripted one):**
    - **P1 PASS:** learned blind 645.3 (SE 36.6) vs random 60.7 (SE 0.2);
    - **P2 PASS:** learned oracle 1,512.9 (SE 44.3) = **2.34x** learned blind (>= 2x).
    - The learned blind forager beats the scripted one (187), so the learner finds a better foraging strategy.
    - Caveat: one seed, shared brain; per-agent brains (spec option A) are the next build step.
  - Next (laptop): per-agent brains + PPO for the learned P1 / P2. Then the 2-minute A10G smoke (~$0.3) after Cam's
    sign-off.

- **(d) G0 word-bias gate: RESULT, PASS** (`analyze_g0.py`; `docs/data/g0/`). 4/4 runs complete; codes formed by 1,500
  (per-slot MI 1.43-2.00 bits).
  - Slot-word agreement over the 12 (fire, slot) cells:
    - **same start (positive control): 280-281 0.500, 282-283 0.500;**
    - **different starts: 0.000 / 0.083 / 0.000 / 0.083, mean 0.042**;
    - fire-relabelling null mean 0.014, **p95 0.083**.
  - **PASS (0.042 <= 0.083).** Cam's shared-body effect is confirmed directly: two seeds from one navigator share half
    their slot words; seeds from different navigators share no more than chance. The separate-brains design (each
    newborn from a navigator no living brain started from) removes the bias. (e) is not held by G0, only by its price.

- **(e) re-price probe (Modal, ~$0.1; `scripts/modal_pop.py`; 12 updates at W 48 / T 128 / B 4):** mean t_update over
  updates 4-12: **2-core CPU 3.15 s; 8-core CPU 3.42 s (no gain); L4 GPU 6.18 s (slower).** The first combined probe's
  L4 leg was killed by my own client timeout during its image build; it was retried alone, detached.
  - At the toy's effective ~$0.18 per container-hour (c1rep billed), a 9,000-update run costs **~$1.4**, so **5 POP + 3
    control = ~$11.3** vs the $6 cap (over 25%; held for Cam). A 9,000-update run also takes ~7.9 h, past the toy
    launcher's 6 h function timeout; it needs a longer timeout.
  - **Option (i) shorter lives** (lifetime 1,000, stagger 250, sensitive period 500, settle 250; generation 4 at ~4,000
    updates): ~$0.63 per run, **~$5.0 for 8**, within the cap. A design change, so Cam's call.

- **(c) Rung-3 rerun at wd 1e-3: RESULT (registered `analyze_rung3.py`; `docs/data/rung3/`). 15/15 complete.**
  - **C1 baseline: FAIL, 2/5 hold** (reliance + gain at both u2500 and u3000): s290 (+46% / +49%, listener window
    4.91 / 5.17 vs gain bar 4.25 / 4.13) and s294 (+49% / +53%, 4.40 / 4.42). The other three show no reliance
    (drops -4 to +2%); s293's listener window collapsed to 0.44 at u2500.
  - **Stop-grad: FAIL, 0/5 hold.** Only s298 relies (+42% at u2500, +30% at u3000, missing the gain bar at 3,000:
    3.97 < 4.13). s297's listener window collapsed (0.13 / 0.35).
  - **Wire-off: PASS.** Shuffle-corrected joint MI at u3000: 0.002 / 0.000 / -0.000 / -0.000 / 0.001 (all < 0.05).
    **The calibrated erosion works: unused codes die, so the control now discriminates.** The NOSIG listener level is
    3.25 / 3.13.
  - **Stop-grad NOT adopted** (registered rule: 0 < 2). So the population's message-head rule is the all-rows default.
  - **The registered stop condition did NOT fire** (it needs 0/5 in BOTH wire-on arms; baseline has 2).
  - **Reading:** at an erosion strength that reliably kills an unused code, a used code survives to 3,000 in 2 of 5
    baseline seeds. That is below the 3/5 bar, so **the toy does not show that use keeps a code alive at this
    strength**. Where it does survive, it is real (strong reliance and gain with the control at zero).
    - Removing listener-row gradients from the message head made survival worse, not better (0/5 vs 2/5). This is
      against the "listener noise erodes the code" hypothesis; it is one comparison of 5 vs 5.
  - Next per plan: (e). It is held on price (Cam's call); not held by (c).

- **Overnight close (Nov 9 ~04:35):** nothing paid is running. Spend **$3.92 metered, all credit, $0 billed.**
  - Backup of `roadtoagi111` volume: 182 files, 99 MB, 0 mismatches (`~/throng-backup/roadtoagi111/`).
  - Holds for Cam: (e) price (~$11.3 as specified vs $6; ~$5.0 with shorter lives); (f) the tau reading (0.63);
    the own-world spec sign-off.
  - Lesson: a detached `modal run` whose entrypoint calls `.remote()` left the L4 app up with an idle container after
    it returned; stopped by hand. The Mac guard now matches all `throng-*` apps (it had missed `throng-pop`).
  - Morning summary for M: `docs/MORNING_SUMMARY_2026-11-09.md`.

- **Cam, Nov 9 ~14:45: morning rulings.**
  1. **(c) accepted. Toy rung 3 at calibrated erosion: FAIL** (baseline 2/5, stop-grad 0/5; the wire-off control at ~0
     bits now discriminates). Stop-grad not adopted; message-head rule = all rows.
     - **Verdict for the shared-network toy: use protects a code against erosion only sometimes.** Likely because the
       code is worth little here (listeners +1-3 deliveries over no signal), so the pressure to keep it is weak.
       **This question is no longer tested in the shared-network toy.**
     - The stop-grad direction (0/5 vs 2/5) is recorded as **weak evidence (p ~0.4) that the listener's gradient
       through the shared head helps rather than erodes.**
  2. **(e) population: LAUNCH** at the registered lifetimes (2,000), **5 POP + 3 control, cap $12.** Changes before
     launch:
     - **Erosion OFF:** turnover is the decay force (newborns start from navigators with no code; a code survives only
       if copied and kept).
     - **Primary (ladder rung 3), at u7500** (all gen 4, before fire 5), in >= 3/5 POP seeds:
       - reliance >= 30% with paired > 2 SE;
       - true-message deliveries >= control-population deliveries + 1.0;
       - same-spot control clean (same-spot drop < half the cross-spot drop).
     - **Secondary (use keeps it alive):** shuffle-corrected joint MI at u7500, POP median >= 2x the control median AND
       >= 0.2 bits above it; per seed and per generation (u1500 / 3500 / 5500 / 7500) in both arms.
     - **Fire 5:** redirection to fire 5 by u9000 in >= 3/5 POP seeds; report which age group coins its word.
     - Export one dictionary per generation per POP seed (word drift, for M).
     - Runtime ~8 h per run, past the 6 h timeout: raise the timeout, or chunk with checkpoint + resume (bit-identity
       verified first). Cam's prediction: primary pass ~45%.
  3. **(f) tau = 0.63 APPROVED** ("that's what I meant"). Launch, cap $1.
  4. Own world: Cam needs a <= 30-line summary of the spec's key choices in the next report.
     - **Brains: ONE PER AGENT.** W2 v1: 8 agents, B = 8, with copying and turnover from the population code.
     - **v1 = food + lookout only.** Predators next, behind a flag: two kinds with different escapes (one can't reach
       agents in cover cells, the other can't reach agents on high cells; the wrong escape gets caught). Kinds are
       property vectors, with no names.
     - **Size: the tuned prototype world** (P6, sigma 1.0, L96, drain 0.008), 8 agents spawned clustered, hearing range
       ~3x the view radius.
     - No paid W2 runs until Cam signs off on the summary.
  5. Housekeeping accepted. Budget: ~$4 spent of $41; (e) + (f) ~$13.
- **(f) launch details:** plan `step6r`, seeds 310-314, toy launcher re-pinned to comm-toy `bc63ba0` (adds `--tau` /
  `--reinit-msg`; default path bit-identical). Analysis: the precondition gate (>= 12 distinct messages at u1000 in
  >= 4/5, from the topsim rows) first, then `analyze_step6.py`.

- **REGISTERED (before launch): (e) population run details** (`scripts/modal_pop.py::population`, comm-toy
  `5c5b07a`; analysis `analyze_pop.py`).
  - **Seeds:** POP 320-324 (wire on), POPCTL 325-327 (wire off, copying on = the NOSIG baseline).
  - **Design:** 9,000 updates, B 4, W 48, founders' PS 1.99 annealed 1,000-1,500, **erosion OFF**, **msg-grad all**;
    replacements at 1,500 / 2,000 / 2,500 / 3,000, then every 2,000 per slot (all gen 4 at 7,500); sensitive period
    1,000, settle 500, floor 0.1; learning speeds 0.5 / 0.75 / 1.0 / 1.5; copying coefficient 1.0; newborns from
    navigators no living brain started from (pool 260-267).
  - **Fifth fire** at the grid centre (6,6): its offset (2 dims) and need (1 dim) are in the obs from update 0 (D 62).
    The navigators' first-layer weights are expanded with zero rows, so they are unchanged in function (apparatus:
    `expand_preserves_navigator`). Lit only from 7,500 (re-rolls over 5 fires from then). PS never touches it (PS ends
    at 1,500).
  - Evals (10 pairs each) at 1,500 / 3,500 / 5,500 / 7,500 / 9,000 + token dumps; brains saved at 7,500 / 9,000.
  - **Bars exactly as Cam ruled:**
    - primary at u7500, reliance >= 30% (paired > 2 SE) + true-message >= control-population + 1.0 (both from the
      probe's control rollouts) + same-spot drop < 0.5x cross, in >= 3/5;
    - secondary: shuffle-corrected joint MI (all brains' sender steps pooled; 100 fire-label permutations), POP
      median >= 2x control and >= +0.2 bits;
    - fire 5 at u9000: swap-to-fire-5 Crafts at fire 5 >= 2x control and paired > 2 SE in >= 3/5, plus the age groups.
  - **Apparatus `--selftest` with fire 5: PASS** in COMM and NOSIG (brain isolation, imitation on the message head only,
    PS exactly 0 without founders, navigator expansion function-preserving, fire-5 need reaching the message path,
    wire-off zero). A fire-5 end-to-end smoke passed.
  - **Runtime:** ~8 h per run on 2-CPU containers; function timeout raised to 12 h (Cam's option). **Price ~$8-11.3**
    (cap $12).

- **Cam, Nov 9 ~17:50: W2 v1 SIGNED OFF**, with three changes before the first paid smoke. (e) / (f) launches accepted
  (the zero-row navigator expansion + its identical-output selftest is the right way); (f)'s gate is read first, as
  registered.
  1. **Aligned interests: inclusive-fitness reward** r_i = own drive reduction + k x mean drive reduction of living kin.
     - k = relatedness: parent / offspring / sibling 0.5; the founders' cluster 0.5 at the start; tracked by lineage.
     - Nothing is rewarded for messages.
     - Behind a flag, default on; the k = 0 arm is the control.
  2. **Founding:** PS for founders only, annealed as in the toy, then off for everyone born after. Founding without PS
     is a later stretch goal.
  3. **Reproduction vs the setpoint:** choose between triggering at the setpoint and a one-sided drive. The smoke
     reports births per 1,000 steps; target: every slot turns over at least once per ~3 lifetimes.
  - Newborns need **W2 navigators** (independent seeds, learned in W2 at k = 0, no comm), with a gate like N1 (>= 0.7x
    the best one's lifespan).
  - Approved next: per-agent brains + PPO on the laptop; then the first paid smoke (2-min A10G timing + ~10-min no-comm
    run), cap $0.5; then a registered plan for the first communication run (k on vs k = 0, PS for founders, swap /
    same-spot / joint-MI probes, food location as the thing worth saying), for sign-off, with its price.
  - **Choice (3): reproduction triggered at the setpoint**: E >= 0.8 held for 50 consecutive steps; the parent pays 0.3;
    the newborn starts at 0.5.
    - Why not a one-sided drive: the symmetric Keramati-Gutkin drive stays as published, and it penalises
      over-eating. With rival food patches, a one-sided drive would make hoarding free.
    - Risk: agents could learn to hover just under 0.8 to avoid paying for reproduction. Inclusive fitness counts
      the offspring's drive reductions (k 0.5), which works against that, and the smoke's births-per-1,000-steps
      readout will show it. If births fall short of the turnover target, the fallback is the one-sided drive.

- **Incident (Nov 9 ~12:54): a network drop on the Mac stopped (e) and (f) mid-run.**
  - The `modal run --detach` clients died with `[Errno 8] nodename nor servname provided`. Modal's detached mode keeps
    only the last-triggered function alive, so both apps were stopped at ~12:59. The Mac guard also exited.
  - Lost: the 8 population runs ~45 min in (~$0.8); step-6 rerun seeds 311-314 partway. **Step-6 seed 310 had
    completed** (its JSONL is saved).
  - **Fix: deploy the apps and spawn each job** (`scripts/spawn_jobs.py`), so runs no longer depend on the Mac's
    connection. Results go to the volume.
  - Relaunched 13:36: the 8 population runs (same registered args and seeds), step-6 seeds 311-314. s310 is kept:
    same code, config and pin.
  - Spend at relaunch $5.35 (all credit). (e) with the lost partial: ~$12.1 vs cap $12, inside the 25% allowance.
  - The Mac guard still depends on the Mac. The Modal guard (waiting on M's secret) removes that.

- **W2 after sign-off (laptop, $0; comm-toy).**
  - `w2/env.py`: setpoint reproduction (E >= 0.8 held 50 steps, parent pays 0.3, newborn at 0.5), lineage relatedness
    (founders 0.5; parent / offspring / sibling 0.5; others 0.5 x the parent's; the dead leave the table), and the
    inclusive-fitness reward flag (own drive reduction + mean over living kin of relatedness x theirs). **14/14 tests**,
    incl. the k = 0 control equalling own reward.
  - **W2 navigators** (`w2/navigators.py`; blind, k = 0, no comm, seeds 1000-1007, 400 updates each). Frozen-policy
    lifespans 653 / 540 / 549 / 623 / 599 / 754 / 605 / 627. **Gate (>= 0.7x best 754 = 527): 8/8 pass.**
  - **Per-agent brains + PPO** (`w2/agents.py`): one MLP per agent slot, own Adam, W replicate worlds vmapped.
    - A newborn's slot is reset inside the rollout to a navigator no living agent of its world started from, and its
      Adam is reset.
    - Each brain trains only on its current individual's rows.
    - Founders spawn clustered; extinct worlds are re-seeded (counted).
  - **Laptop runs** (8 worlds x 8 agents, T 256, 200 updates, seed 1):

    | arm | births per 1,000 steps per world, by quintile | alive / 8 | extinctions |
    |---|---|---|---|
    | inclusive | 4.43 / 4.68 / 4.92 / 4.88 / 4.86 | 6.7 | 9 (all in the first 40 updates) |
    | k = 0 | 4.65 / 4.77 / 4.72 / 4.77 / 5.02 | 6.7 | 9 (first 40) |

    - **Turnover:** ~0.61 births per slot per 1,000 steps, i.e. one per ~1,630 steps. Mean lifespan ~1,370 steps (6.7
      alive / 4.9 deaths per 1,000 steps), so every slot turns over about **once per ~1.2 lifetimes. Cam's target (at
      least once per ~3 lifetimes) is met.** Agents do not avoid reproduction at the setpoint.
    - No inclusive vs k = 0 difference yet over 200 updates (no comm; kin rarely meet in this short run).
    - 0.35 s per update on the laptop (~47k agent-steps/s with learning).

- **W2 first paid smoke (A10G, deployed app `throng-w2`, spawned; Cam cap $0.5): DONE.**
  - **Timing:** 64 worlds x 8 agents, T 256, per-agent PPO: **0.19 s per update = ~690k agent-steps/s with learning**
    (target >= 100k).
  - **No-comm run** (120 updates, inclusive on): births per 1,000 steps per world 4.78 / 4.63 / 4.81 / 4.70 / 4.96 by
    quintile. Extinctions 76 / 0 / 0 / 0 / 0 (all in the first 24 updates, while the navigator-start agents learn to
    reproduce). Alive 6.7 / 8. The turnover target holds on the GPU too.
  - Cost: well under $0.5 (it ran ~1 min of A10G plus image build; the per-app figure follows).
- **DRAFT for Cam's sign-off: the first W2 communication run** (`docs/W2_COMM_RUN_PLAN.md`).

- **Cam, Nov 9 ~20:15: W2 comm-run ruling.**
  - Accepted: setpoint reproduction (the fallback is fine), lineage relatedness, the kin flag, 8/8 W2 navigators, the
    per-agent trainer, the paid smoke, deploy + spawn.
  - **The guard is still Mac-only while runs now outlive the Mac,** so the Modal guard is M's priority. Until it's
    live, **keep every spawned job's timeout tight.**
  - **FOUNDING PHASE APPROVED,** with two changes:
    1. the energy floor is near zero (0.05), so hunger and food still matter during founding;
    2. **PS in its generic form only** (message entropy across states up, entropy given the speaker's own observation
       down, as in the toy), never MI with a variable we pick. **Food direction is a READOUT, not a target.** Also
       report MI with other candidates (nearest-kin direction, own energy, at-lookout), to see what they choose to
       talk about.
  - **The bigger problem: learning per life.** ~1,400 steps of own data (~5-22 updates) may be too little for a newborn
    to learn to listen, so the transmission test could fail for a reason unrelated to language.
  - **LEARNABILITY GATE before any paid comm run** (laptop, or cap $0.5):
    - newborns (fresh W2 navigators) live among senders frozen with a fixed code (scripted, or the founders' frozen
      code once it exists);
    - **pass: newborn reliance >= 30% (cross-direction swap) within one lifetime, in >= 50% of lives**;
    - if it fails, add in order, re-gating after each: (a) a juvenile period (the parent provisions the newborn's
      energy for the first J steps and pays it; the young stays within hearing range); (b) longer lives (raise the
      age cap / lower the drain) while P2 (oracle >= 2x blind) still holds;
    - if it still fails, report; Cam will ask M about copying elders' actions as another inborn trait.
  - **Register the bars before the comm run.**
    - Primary, in >= 3/5 K runs, among agents born after founding: cross-direction reliance >= 30% (paired > 2 SE);
      food gain vs OFF >= a margin proposed from the smoke; same-direction swap clean (< half the cross drop).
    - Secondary: K vs K0; shuffle-corrected joint MI per candidate variable; per-generation dictionaries.
    - Then price it. Arms K x5 / K0 x5 / OFF x3.

- **(f) step-6 rerun: precondition gate FAILS, 0/5. Stopped before reading compositionality, as registered**
  (`docs/data/step6r/gate_clean.txt`).
  - Distinct messages at u1000 (of 14 training meanings): **6 / 5 / 6 / 6 / 6** (the gate needs >= 12 in >= 4/5).
  - Update-1 summed sender entropy 3.24-3.62 nats, on the 3.18 target, so the starting-entropy fix worked. Yet the
    sender still settles on ~5-6 messages, about one per fire plus one or two: **the material isn't being encoded**.
    The collapse is not set by the starting entropy.
  - Bookkeeping: the volume files held repeated segments. The first (killed) launch had completed s312-314, a
    preemption restarted s311, and the spawned relaunch appended again. Segments were split at each update-1 row;
    every complete segment gives identical values (deterministic). The analysis uses the last complete segment
    (`.clean.jsonl`). The duplication cost ~$0.4 extra.
  - Next is Cam's call. Candidates (not run):
    - material on its own input dims (not the old noise slots) with stronger signal;
    - a larger vocabulary (8);
    - a curriculum where material matters before fire.

- **CORRECTION (W2): navigator lifespans are bimodal; the gates used means.** A frozen W2 navigator (s1005), 16
  worlds, 3,000 steps: **median lifespan 62** (N 64 random starts), 62 (N 8), 66 (N 8 clustered); **means 754 / 911 /
  653**.
  - Most newborns starve in ~60 steps (E0 0.6 / drain ~0.011 = ~55 steps without food). A minority that start near a
    patch live to the cap.
  - The navigator gate (>= 0.7x best), P1 (645 vs 61) and P2 (2.34x) were computed on **means**, which this minority
    dominates. They remain true as stated (mean lifespan) but overstate the typical life.
  - For the learnability gate this is decisive: **a typical newborn lives about one 64-step update**, too short to learn
    anything in its own life.
- **Learnability gate (laptop), first read:** baseline (no juvenile period), 4 worlds x 20 updates: **0/75 lives pass;
  median lifespan 66** (identical with learning off). FAIL, as expected from the above. Running at full scale (16
  worlds x 150 updates) alongside Cam's fallback (a), a juvenile period of J = 500 provisioned steps within hearing
  range.

- **Learnability gate: first full reads, and an apparatus fix.**
  - Full scale (16 worlds x 150 updates, T 64, 2 epochs), test 4 paired episodes per life:
    - baseline: 15/2,090 lives pass (0.7%), median life 67;
    - (a) juvenile J = 500: 15/662 (2.3%), median life 572, **median drop 0.2%**;
    - (a) + (b) juvenile + drain halved: 4/553 (0.7%), median life 641, median drop 0.0.
  - **Positive control** (`--sanity`: a scripted listener that walks the heard sector when no food is visible): eaten
    9.6 (true) vs 4.1 (swapped), median drop 67%, **yet only 8/74 lives passed with 4 episodes**. My added
    "paired > 2 SE" condition was underpowered at n = 4.
  - **Fix:** 16 paired episodes per life. The positive control then **passes the gate: 48/74 = 65% of lives, median
    drop 63%**. The rest are lives where food is visible near the beacons, so the message doesn't matter.
  - The learned newborns' median drop is ~0 in every variant, so the fail looks genuine. Both fallbacks are being
    re-run on the 16-episode test before any verdict.

- **Learnability gate: RESULT, FAIL after both fallbacks (16-episode test; `docs/data/w2_learnability/`).**

  | variant | lives | pass | median drop | food eaten, true / swapped (mean, 300 steps) |
  |---|---|---|---|---|
  | (a) juvenile J = 500 | 662 | **0.8%** | 0.0 | 3.46 / 3.48 |
  | (a) + (b) juvenile + drain halved | 553 | **0.0%** | 0.0 | 5.26 / 5.29 |
  | positive control (scripted listener) | 74 | **65% (gate PASS)** | 0.63 | 10.36 / 3.90 |
  | baseline (4-episode test) | 2,090 | 0.7% | 0.0 | - |

  - **Reading:** the code is worth learning (a listener that uses it eats ~2.7x more than the learned newborns), and the
    test detects use. **But a newborn starting from a navigator does not learn to listen within one life**, even with
    a 500-step provisioned juvenile period and longer lives (median life 572-641 steps, ~9-10 updates of its own
    64-step data).
  - Per Cam's rule: report. **Cam asks M about copying elders' actions as an inborn trait.**
  - Other levers (not run, for Cam): higher plasticity in the young (a larger newborn learning rate, already a
    heritable trait); a pretrained listening prior (navigators trained with a scripted speaker); or brains that learn
    across lives.

- **Cam, Nov 10 ~00:35: two gate failures.**
  - **(f) FAIL accepted.** Before any rerun, a free check: **does material matter to the listener?** VoI on saved params,
    or a scripted listener: deliveries when told fire + material vs fire only. If the gap is < 25%, the world gives no
    reason to name material: fix the world first (reward depends on the right material). If the gap is real, rerun
    once with material on its own freshly initialised input rows (not the old noise dims), vocab unchanged, same gate,
    cap $1. **Dedupe by segment from now on.**
  - **W2 learnability FAIL accepted** (the 4 → 16 positive-control fix was right). Cam's reading: reward-only PPO on
    ~600 steps of own data can't learn a 9-way mapping; a learning-mechanism problem, not a language problem. Levers,
    in order:
    1. **Juvenile provisioning becomes the W2 default** (the parent pays; the newborn spawns next to the parent).
       **Re-report P1 / P2 on MEDIANS** with it on; P2 (oracle >= 2x blind) must hold on medians.
    2. **Predictive learning:** an auxiliary self-supervised loss on a shared trunk, heard messages as input,
       predicting the next observation window, own energy change, and food-in-view a few steps ahead. Dense signal
       every step, no designed meanings; the start of the world model the curiosity drive needs.
    3. **Sleep replay:** after each rollout, K extra passes (K = 8) over the agent's own recent memory, predictive
       loss only. Report gradient steps per life.
    4. Rerun the learnability gate (16-episode test, >= 50% of lives) for baseline+1, +2, +2+3.
    5. **M APPROVED (Nov 10) copying elders' actions as a permanent inborn trait for the young:** an extra arm
       +2+3+copy-actions (behaviour cloning on nearby elders' obs → action, weighted like the signal-copying drive,
       young only), alongside the others.
    - **Not allowed:** navigators pretrained with a scripted speaker (that teaches our code).
    - If all fail: a generic fast-learner prior (meta-learned on many random codes, RL^2-style memory), design only.
      Laptop only; cap $0.5 if anything needs Modal.
  - (e): analyse when it lands, deduped by segment.

- **(f) material VoI check (Cam Nov 10; free; `throng-toy/scripts/calc_step6_voi.py`, scripted listeners in the exact
  step-6 rules, 4,096 worlds x 128 steps, 4 seeds):**
  - told fire + material: 24.7 deliveries per world, net reward +66.7;
  - told fire only (tries materials until one works, each miss costs 0.45): 19.0 deliveries, net +38.4.
  - **Delivery gap 23.1% (sd 0.1% over seeds), under Cam's 25% line** (the reward gap is 42%, because the misses cost
    craft fees).
  - **Rule outcome: the world gives too little reason to name material, so fix the world first; no rerun.** A candidate
    fix for Cam: a wrong-material Craft spoils the opportunity (the pair re-rolls). A fire-only listener would then
    succeed ~1/4 of the time per opportunity, a delivery gap of ~70%.

- **W2 P1 / P2 on MEDIANS with juvenile provisioning (J = 500; Cam Nov 10 lever 1; `w2/p12_medians.py`;
  `docs/data/w2_learnability/p12_medians.txt`).** Same learner and budget as before (400 updates, seed 0), provisioning
  in training and test; nobody pays (no parents in this setting).
  - Median lifespans of 3,000 steps (16 worlds x 64 agents): **random 559, learned blind 710, learned oracle 709**
    (bootstrap SE 0 / 11 / 13).
  - **P1 PASS** (blind > random by > 2 SE). **P2 FAIL on medians: oracle / blind = 1.00x** (needs >= 2x).
  - **Reading:** with provisioning, most agents still die ~200 steps after it ends, informed or not. The oracle's
    advantage (2.34x on means without provisioning) lives in the minority that reaches a patch.
  - Cam's rule (P2 must hold on medians) is not met, so **the world needs retuning before the comm run**, at Cam's
    call. Candidates: denser or larger patches, or a lower drain so a typical post-juvenile agent can find food, while
    information still doubles the median life.

- **W2 learnability gate with Cam's levers: ALL FAIL** (16-episode test, 16 worlds x 150 updates;
  `docs/data/w2_learnability/gate_*_summary.json`):

  | arm | lives | pass | median drop | grad steps per life (median) |
  |---|---|---|---|---|
  | baseline + 1 (juvenile 500) | 637 | 0.3% | 0.0 | 18 |
  | + 2 (predictive trunk) | 682 | 0.4% | 0.0 | 18 |
  | + 2 + 3 (sleep replay K = 8) | 758 | 0.9% | 0.0 | 306 |
  | + 2 + 3 + copy elders' actions (M-approved trait) | 732 | 1.1% | 0.0 | 306 |

  - **Why replay didn't help:** it trains only the predictive loss (by design). The behaviour-shaping losses (PPO,
    and the copying loss) still get ~18 gradient steps per life.
  - **Disclosure on the copy arm:** the elders are scripted listeners (the gate's code is scripted). In real W2 the
    elders are the population.
  - **Per Cam's rule, next lever = a generic fast-learner prior** (RL^2, meta-learned on random codes): design only,
    `docs/W2_FAST_LEARNER_DESIGN.md`. Two cheaper one-flag variants are listed there (copying inside replay; a
    young-plasticity trait).

- **Cam, Nov 10 ~02:25: rulings.**
  - **(f) world fix APPROVED:** a wrong-material Craft spoils the opportunity (the pair re-rolls).
    - Re-run the free scripted VoI check first; the gap must be >= 25% (Cam expects ~75%).
    - Then ONE rerun with the world fix AND material on freshly initialised input rows. Same gate (>= 12 distinct
      messages at u1000 in >= 4/5), cap $1. Recompute NOSIG for the new rules.
  - **W2 WORLD RETUNE (blocks any comm run).** New gates, all on MEDIANS, juvenile provisioning on, parents paying:
    - P1m: learned blind median life >= 3x random;
    - P4: learned blind median life >= 2,000 steps;
    - P2m: informed vs blind on reproductive success, not lifespan: median offspring per life (or energy intake per
      step), oracle >= 2x blind.
    - Levers: lower drain, food that moves or relocates, patch density. Retune the world, never the agents. Laptop.
  - **Learnability:** the four FAILs accepted; the copy arm with scripted elders is acceptable for the gate. Next
    (laptop, same gate):
    1. a free diagnostic: does the predictive head learn the code within a life? Food-in-view prediction accuracy with
       the true vs a swapped message at end of life. If not, go straight to the fast learner;
    2. copying inside replay (K = 8 passes over the copy loss too; never replay PPO);
    3. young plasticity: learning rate x4 at birth, decaying to the lineage speed by the end of the juvenile period.
       Arms: 2, 3, 2+3;
    4. if all fail: **the fast-learner design is APPROVED to build** (random code each life, leak test, gates as
       written), laptop first, cap $3.
  - (e): analyse when it lands.

- **(f) world-fix VoI check (free): gap 75.6%** (`docs/data/step6r/voi_spoil.txt`). Told fire + material: 24.75
  deliveries, net +66.8; fire only: 6.04, net +8.2. The gap is real (>= 25%; Cam expected ~75%).
- **REGISTERED: (f) rerun 2, plan `step6f`** (comm-toy `24f3925`, deploy + spawn; toy function timeout tightened to
  2 h).
  - COMM x5 (seeds 330-334) + **NOSIG x2 (335 / 336) for the new rules' baseline**, 3,000 updates.
  - Flags: `--comp --vq-vocab 4 --reinit-msg --tau 0.63 --spoil-wrong --material-rows`:
    - **world fix:** a wrong-material Craft at the lit fire re-rolls the pair;
    - **material on 4 new input dims before the wire**, rows freshly initialised (ORACLE0's 59-dim weights expanded);
      the old noise slots carry noise.
  - **Same gate:** >= 12 distinct messages at u1000 in >= 4/5 COMM seeds, else stop before reading compositionality.
    Then `analyze_step6.py`. Cap $1 (estimate ~$0.9).
  - Checks: default path bit-identical (15/15 rows); the new flags run end to end (laptop smoke).

- **Learnability round 2 (Cam Nov 10 ~02:25): ALL FAIL; the diagnostic says go straight to the fast learner**
  (`docs/data/w2_learnability/g2_*_summary.json`; 16-episode test, 16 worlds x 150 updates).
  - **Diagnostic (predictive + replay arm, 64 lives x 4 true-message test episodes):** the head's food-in-view-4-ahead
    error on the same observations with the heard sector intact vs shifted by 4: MSE 0.0980 vs 0.0979, **swap increase
    -0.06%, paired z -0.04.** The predictive head does not use the message at all. Per Cam's rule: within-life data is
    too thin; go straight to the fast learner.
  - Arms on the copy-arm base (juvenile + predictive + replay + copy):

    | arm | lives | pass | median drop |
    |---|---|---|---|
    | + copying inside replay | 664 | 0.75% | 0.0 |
    | + young plasticity (x4 decaying over J) | 708 | 1.7% | 0.0 |
    | + both | 710 | 1.1% | 0.0 |

    All at ~306 gradient steps per life.
  - **Next: build the fast learner** (approved; laptop first, cap $3): RL^2-style recurrent prior meta-trained on a
    random code each life, with the leak test and the gates in `docs/W2_FAST_LEARNER_DESIGN.md`.
- **W2 world retune harness built** (`w2/retune.py`). Env adds parent-paid juvenile provisioning (`juvenile`) and
  relocating food (`relocate`), with 2 new tests (11/11).
  - The smoke caught two gate-logic bugs, both fixed: P2m passed on 0 >= 2 x 0 (now it needs a positive oracle
    median); and unprovisioned founders dominated the medians (the gates now use lives born during the evaluation,
    with all-lives medians reported).
  - The sweep is running.

- **Cam, Nov 10 ~03:15: fast-learner addition. REGISTERED (`w2/fastlearner.py`, comm-toy):**
  - **Each life's code is about a randomly drawn topic.**
    - Useful (affect intake): nearest-food direction, richest-patch direction (the patch with the most food within 2
      cells of its centre), food-distance band (Manhattan distance to the nearest patch / 6, capped at 7).
    - Useless: nearest-agent direction, the speaker's energy band, pure noise.
    - A random token mapping per life.
  - **Held out:** the topic richest-patch direction (training topics: the other 5) and code family B (permutations
    with odd sigma(0)).
  - **Gates:**
    - G1 reliance >= 30% (paired > 2 SE over 16 branches) in >= 50% of lives on the held-out topic x family;
    - G2 useless-topic (and no-code) messages leave intake unchanged within 5% vs silence;
    - G3 leak: held-out vs training family within 0.1 in median drop;
    - G4 foraging kept: median intake within 10% of the plain W2 navigator's (comparison added before the full run).
  - Reliance reported per topic.
  - **Laptop pilot running** (1,500 updates, 32 worlds, quick held-out gate every 250 updates). **The full run goes to
    Modal (cap $3) only if the pilot's held-out reliance is rising.**

- **W2 world retune sweep: NO config passes** (`docs/data/w2_retune/`; 8 configs; the same learner and budget per config;
  frozen-policy evaluation with reproduction + parent-paid juvenile J 500; gates on lives born during the evaluation).
  - **P1m PASSES everywhere:** blind median 668-1,510 vs random 61-152.
  - **P4 FAILS everywhere:** blind median life 668-1,510 < 2,000.
  - **P2m FAILS everywhere:** median offspring 0-1 for both; median intake per step identical (0.0015-0.0087).
  - **The structural finding (means, offspring lives):** information **extends life** (oracle vs blind 1,978 vs 1,366;
    2,580 vs 998 with relocation) but **does not raise reproduction** (offspring per life x0.67-1.00; lower in every
    relocating world).
    - **Intake per step equals metabolic need under a homeostatic drive** (an agent eats what it burns), so it cannot
      show the value of information.
    - Setpoint reproduction (E >= 0.8 held 50 steps) is rate-limited by the hold, not by how much food an agent finds.
  - **Reading:** in this world, knowing where food is doesn't convert into fitness. Retuning drain, patches or
    relocation won't change that; **the reproduction rule is the bottleneck.**
  - **Options for Cam (world rules, not agents):**
    - (i) reproduction paid from surplus: an energy store above the setpoint, an offspring whenever the store reaches
      the cost, so more food found means more offspring;
    - (ii) a one-sided drive (the earlier fallback), so eating beyond need is rewarded and stored;
    - (iii) P2m measured on births per 1,000 steps per world (population growth) rather than per-life medians, which
      are coarse integers here.

- **Fast-learner laptop pilot: NOT RISING, no Modal run (Cam's rule).** 1,500 updates, 32 worlds x 4 learners, 0.41 s
  per update.
  - **The agent never learned to forage:** training food eaten per step 0.0001 in every quintile.
  - Held-out reliance pass fraction 0.00-0.06 at every 250-update check. True = swapped intake everywhere; the
    "median drop 1.000" is 1 - 0/0 on lives that eat nothing (meaningless).
  - All six topics are identical (messages ignored).
  - **Cause:** the RL^2 GRU starts from scratch, without the navigators' foraging, and 1,500 small updates (128 agents
    x 64 steps) don't find food. The W2 navigators had 400 updates of 8 worlds x 64 agents x 128 steps.
  - **Proposed fix (Cam's call; not run):** give the meta-learner the navigators' foraging first, by behaviour-cloning
    the W2 navigators' actions (no messages; our own W2 agents, no code taught). Then meta-train on random topics and
    codes, and re-pilot. Alternatively keep the navigator MLP as the policy trunk and add the GRU as a side memory
    whose state feeds the policy.

- **Cam, Nov 10 ~03:40: W2 retune ruling.** P2m dropped: with a fixed slot cap, births happen only when someone dies,
  so longer lives mean fewer births per life. The setpoint reproduction rule is kept for the comm run;
  surplus-paid reproduction is logged as the W3 upgrade, tested then by relative fitness in MIXED populations. New
  gates on medians, lives born in the eval: **P1m** blind >= 3x random; **P2** oracle median lifespan >= 2x blind;
  **P4** blind median >= 1,000. Pick the config passing all three with the largest P2 margin; if none passes P2 on
  medians, report before any further retune.
- **Retune under the revised gates: NONE passes all three** (`docs/data/w2_retune/summary_v2.txt`).
  - Only one config passes P2: **d0 0.002 / 6 patches / relocation 0.001: random 151, blind 811, oracle 2,130 (2.63x)**,
    but P4 fails (811 < 1,000).
  - The configs passing P4 miss P2: d0 0.002 / 12 / 0.0005 (blind 1,050, oracle 1,706, 1.62x); d0 0.003 / 8 / 0.0005
    (1,510 vs 2,253, 1.49x); d0 0.004 / 6 / 0.0005 (1,210 vs 1,609, 1.33x).
  - **Reported before any further retune, as ruled.** The likely next point is between the two leaders: relocation
    ~0.001 with d0 ~0.0015-0.002 and 8-12 patches.
- **(f) rerun 2 (world fix + material rows): gate FAIL 0/5, but moving** (`docs/data/step6f/gate.txt`; deduped by
  segment).
  - Distinct messages at u1000: **10 / 7 / 8 / 9 / 8** (needs >= 12 in >= 4/5; was 6 / 5 / 6 / 6 / 6 before the fix).
  - New NOSIG listener deliveries per world (last 50): 0.15 / 0.25 (blind guessing is now costly).
  - Stopped before compositionality, as registered.
  - Bookkeeping: a stalled download first produced a truncated local file; re-fetched with a size check. s334 had a
    restart segment; the last complete one is used.

- **Cam, Nov 10 ~03:55: rulings.**
  - **W2 retune: the in-between sweep APPROVED** (relocation ~0.001, d0 0.0015-0.002, 8-12 patches; laptop). Same
    three gates and the same pick rule. If nothing passes, report.
  - **(f) FAIL accepted** (moving: 7-10 distinct vs 5-6; the code is now worth a lot, NOSIG 0.15-0.25). **One last toy
    try, REGISTERED:**
    - vocab **8** codes per slot, everything else as rerun 2 (world fix, material rows, reinit-msg, tau 0.63,
      COMM x5 + NOSIG x2, fresh seeds 340-344 / 345-346);
    - run to **u2000**, gate read at u2000: >= 12 distinct messages in >= 4/5;
    - if it passes, read compositionality with the registered analysis (`analyze_step6.py` at U = 2,000); cap $1.5;
    - **if it fails, step 6 is parked in the toy**, and compositionality moves to W2 (kind x place facts from the
      generative physics).
  - **Fast learner: behaviour cloning APPROVED** (our own W2 navigators, no messages, no code taught):
    1. clone the navigators' foraging into the GRU agent; **gate: median intake within 10% of the navigators' before
       any meta-training**;
    2. meta-train on random topics / codes with a small BC loss kept on; re-pilot under the same rising-reliance rule
       before Modal;
    3. if cloning can't reach the intake gate, switch to the navigator MLP trunk + GRU side memory.
  - (e): analyse when it lands.

- **W2 retune sweep 2 (in-between): PICK = d0 0.0015, 8 patches, relocation 0.001** (`docs/data/w2_retune/summary_v3.txt`).
  - Medians, lives born in the eval: **random 173, blind 1,002, oracle 2,983; P1m PASS, P4 PASS, P2 PASS 2.98x.**
  - Also passing all three: d0 0.0018 / 10 / 0.0008 (random 160, blind 1,056, oracle 2,832, 2.68x).
  - **Caveat:** the pick's P4 margin is razor-thin (1,002 vs 1,000) on one learner seed. Suggest confirming on 2-3 more
    seeds before the comm run (the runner-up is sturdier on P4).
  - Others: d0 0.0015 / 12 / 0.001 (1.45x), d0 0.0018 / 10 / 0.0012 (1.92x), d0 0.002 / 12 / 0.001 (1.30x),
    d0 0.002 / 8 / 0.001 (P4 fails, 884).
- **Fast learner: behaviour cloning FAILED the intake gate; the side-memory fallback PASSES it.**
  - Cloning (DAgger, 300-1,500 updates): clone intake ~0 vs the navigators' 29.4 per 300 steps.
    - The navigators are near-uniform on the clone's states (entropy 1.6 of 1.79), and KL 0.17-0.43 doesn't fall.
    - Even with expert roll-in the logged intake stayed ~13x below the navigators' in the same world: an unresolved
      cloning-pipeline issue (not chased, since Cam's fallback was approved for exactly this case).
  - **Side memory (`--arch sidemem`):** navigator MLP trunk (weights loaded) + GRU over [heard, previous action,
    previous reward] adding to the policy logits through a zero-initialised head. **Intake gate PASS: 28.6 vs 29.4
    (0.97x).**
  - Meta-training pilot running (1,500 updates, BC 0.1 kept on, quick held-out gate every 250).

- **Cam, Nov 10 ~04:35: rulings. REGISTERED:**
  - **W2 pick confirmation:** run the pick (d0 0.0015 / 8 / reloc 0.001) and the runner-up (d0 0.0018 / 10 / 0.0008) on
    3 more learner seeds each (seeds 1-3; the existing runs are seed 0). **The pick stands iff all three gates pass on
    the median across its 4 seeds AND in >= 3/4 seeds;** otherwise the runner-up by the same rule; if neither,
    report.
  - **Fast learner:** side memory accepted (intake 0.97x); the pilot rule is unchanged (Modal <= $3 only if held-out
    reliance rises). Before Modal, two apparatus checks: (a) heard messages enter the GRU side memory; (b) a
    no-message run gives ~0 reliance.
  - **Noted (doesn't block): the cloning roll-in bug.** The navigator itself, driving the roll-in, got ~1/13 of its
    usual food. **Rule: any harness used later must reproduce the navigator's own W2 intake first.**
  - (f) / (e): analyse when they land.

- **Fast learner, side-memory apparatus (Cam's two checks): PASS** (`w2/fastlearner.py --apparatus`).
  - Heard tokens change the GRU side memory's state (max |dh| 0.48).
  - At init they don't reach the policy (logit difference exactly 0, the zero-head design); with a nonzero head they do.
  - No-message runs: true and swapped arms give identical intake (drop 0.0).
  - Pilot fix: the first pilot ran the intake gate after one BC update, which had nudged the trainable trunk (0.87x).
    The gate now runs on the untouched initial weights (0.97x), and the pilot is relaunched (1,500 updates, BC 0.1
    kept on, quick held-out gate every 250).
  - Harness-intake rule: the gate's navigator reference eats 29.4 per 300 steps in the beacon world (fixed-length
    lives, floored energy). That is the harness's own navigator baseline, which the side memory reproduces (0.97x);
    it is not the navigator's natural-W2 intake.
- **W2 pick confirmation running:** pick and runner-up x seeds 1-3 (`w2/retune.py --seed`).

- **RESULT (Nov 9 ~23:45): (f) last toy try step6v8 (vocab 8): gate FAIL, 3/5** (`docs/data/step6v8/gate.txt`).
  - Distinct messages at u2000 (of 14 meanings): 14 / 11 / 13 / 9 / 12; the bar was >= 12 in >= 4/5.
  - Every seed has the same count at u1000 and u2000: the code settles by u1000 and does not grow afterwards.
  - Up from vocab 4 (7-10 distinct at u1000) but short of the gate. NOSIG deliveries/world 0.09 / 0.31; COMM 2.4-6.0.
  - **Per the ruling: step 6 is parked in the toy; compositionality moves to W2.** Compositionality was not read.
- **INCIDENT (Nov 9 23:39), my error: the spend guard stopped all three deployed apps.**
  - The guard (`~/throng-smoke/cap_guard_r2.sh`) had exited at 22:32: a transient empty app list ended its loop.
  - I restarted it without `CAP=41`. The script's default was still 15; metered spend was 16.82. On its first tick it
    stopped throng-pop, throng-comm-toy and throng-w2.
  - Lost: 4 population runs in flight. pop_on_s322 at u6496, pop_on_s324 u6158, pop_ctl_s326 u6508, pop_ctl_s327
    u2222.
    - The runs had also been restarting from scratch on preemption before this (2-4 segments each).
    - s327 had an earlier segment reach u8827, past the u7500 primary readout.
  - Not lost: step6v8 (all 7 complete at u2000 before the stop). Population s320 / s321 / s323 / s325 (complete at
    u9000).
  - **Fixes:**
    - the guard's default CAP is now 41;
    - it no longer exits on an empty app list;
    - it was restarted at 23:42 with CAP=41.
  - **Relaunched the 4 lost runs unchanged at 23:43** (same pinned commit 5c5b07a, same seeds and args; deploy + spawn).
  - Cost: throng-pop so far ~$6.51 of the $12 cap (~$0.073 per 1,000 updates, restarts included). The 4 reruns are
    ~29k updates, ~$2.1-3.5, so (e) stays inside its cap.
  - Analysis: deduped by segment; the last complete segment (u9000) per seed. Lands ~08:00 if not preempted.

- **RESULT (Nov 9 ~23:50): fast-learner side-memory pilot (laptop, 1,500 updates, BC 0.1): FAIL, no rise. No Modal**
  **(pilot rule).** Data in `docs/data/fl_pilot/`; comm-toy `54bd6bc`.
  - **Intake-gate fix first.**
    - The earlier 0.89 compared the navigator-0 trunk with the 8-navigator pool. The reference is now navigator 0, on 64
      lives with the same worlds and keys: ratio **1.000** on the initial weights.
    - This is expected: the zero side head makes agent = navigator. It verifies the gate harness, which is the
      harness the training rollout also uses (same `step()`).
  - **Held-out reliance (family B, topic richest-patch): pass fraction 0.0 at u250, 500, 750, 1000, 1250 and 1500**
    (median drop ~0).
  - **Final gates:**
    - G1 FAIL (0.0);
    - G2 useless topics unchanged (msg vs silence -3% to -4%);
    - G3 leak OK (trivially; both drops ~0).
  - **Diagnostics added:**
    - **In-distribution (family A, trained topic nearest-food): pass 0.0, drop ~0.** Messages are not used even on
      what it was trained on. Every topic shows messages vs silence at -2% to -5%, i.e. a small cost.
    - **G4 foraging of the trained agent: 17.0 vs the navigator's 27.4 (0.62x): FAIL.** Meta-training degraded
      foraging: training intake/agent-step 0.0153 (u1-50) fell to 0.0097-0.0111 from u200 on.
    - The side head barely moved (max |w| 0.044 from 0). The trunk is not frozen, and BC 0.1 pulls it toward a random
      navigator per world, not navigator 0.
  - **Reading:** within 1,500 updates, meta-RL learned no message use and partly unlearned foraging.
  - **Options for Cam** (not run; each a laptop pilot first):
    - (i) freeze the trunk; train only the GRU + side head;
    - (ii) a VoI check for this harness: a scripted listener that follows the nearest-food code. How much more does it
      eat? If ~0, there is nothing to learn here;
    - (iii) longer meta-training;
    - (iv) stop the fast-learner track and carry W2 comm with the population/turnover route.

- **RESULT (Nov 9 ~23:58): W2 pick confirmation: NEITHER STANDS** (`docs/data/w2_retune/summary_confirm.txt`, script
  `confirm_summary.py`; same basis as summary_v3).
  - **Pick (d0 0.0015 / 8 / 0.001):**
    - blind medians 1002 / 922 / 951 / 937 -> median 944;
    - oracle 2983 / 2160 / 3169 / 3494 (P2 3.26x on the median);
    - **P4 fails on the median (944 < 1,000); all three pass in 1/4 seeds -> FAILS.**
  - **Runner-up (d0 0.0018 / 10 / 0.0008):**
    - blind 1056 / 796 / 1389 / 917 -> median 986;
    - oracle 2832 / 4460 / 2699 / 2005 (P2 2.80x on the median; seed 2 fails P2 at 1.94x);
    - **P4 fails on the median (986); 1/4 seeds -> FAILS.**
  - **Pattern:**
    - P1m passes 8/8 (blind 5-8x random). P2 passes 7/8 seeds, with large margins on the medians.
    - **P4 (blind median >= 1,000) is the only binding gate.** It misses by 5.6% (pick) and 1.4% (runner-up). The
      seed-1 sweep value of 1,002 was the high draw.
    - Learner-seed spread of the blind median is wide: 796-1,389 for the runner-up, with offspring-life counts 653-2,739.
  - Per the rule: **report**. No world chosen; the W2 comm run waits.
  - **Caveat (measurement):**
    - `retune.py` counts lives still alive at the 6,000-step horizon at their cut-off length (censored, line 53). The
      blind and oracle medians are therefore biased low. The oracle is biased more, since more of its lives are still
      running at the end, so P2 is understated too.
    - The registered verdict stands on the registered measurement. A censoring-corrected read (Kaplan-Meier median, or
      only lives born early enough to finish within the horizon) is offered to Cam, not run.
- **Fast-learner harness note:**
    - The 1.000 intake ratio shows the gate and the training rollout agree with each other (agent = navigator 0, same
      keys): self-consistency.
    - It has NOT been checked against navigator 0's intake in its own native evaluation.
    - The fast-learner harness runs on the prototype world (P6, d0 0.008), not on a retune candidate.

- **Cam, Nov 10 ~05:15: combined rulings.**
  1. **Guard incident:** the handling is accepted. Also:
     - the guard refuses to start without an explicit CAP;
     - it writes a heartbeat every tick.
     - Runs over 2 h get checkpoint + resume (bit-identity verified on a short chunk) or non-preemptible containers,
       whichever is cheaper.
     - (e) reruns: analyse when they land.
  2. **(f) step6v8:** parked as registered. Free and exploratory (no claim): the registered compositionality analysis
     on the 3 seeds with >= 12 distinct messages, labelled exploratory.
  3. **W2 pick:**
     - "neither stands" accepted;
     - censoring is an instrument bug, so **lifespan medians are Kaplan-Meier with right-censoring at the horizon
       from now on**;
     - re-read the 8 runs with KM and report, but do not pick from that re-read;
     - **sweep (c):** d0 0.0012-0.0013, 8 patches (+ the old pick for reference), 4 learner seeds, KM medians;
     - **pick rule:** all three gates on the 4-seed median AND in >= 3/4 seeds, AND blind KM median >= 1,100 (P4
       margin >= 10%). Same P1m / P2. If none passes, report.
  4. **Learnability + fast learner: apparatus first (laptop only).**
     - (a) **Hearing probe** in the learnability harness:
       - train only the hearing rows + policy head, by cross-entropy on the scripted listener's actions given
         (obs, heard), at the arm's learning rate, ~300 steps;
       - expected: held-out accuracy >= 90% and gate >= 50% of lives;
       - check that the heard message reaches the trunk input with the expected values;
       - if it fails: a pipeline bug, found before further work;
       - if it passes only at a higher learning rate: rerun copy-in-replay at that rate.
     - (b) Move both harnesses to the retuned W2 world.
     - (c) Then: a VoI check (scripted nearest-food listener vs silence); if it gains, freeze the trunk and train only
       the side memory; longer meta-training only after rising reliance. Don't drop the track.
- **REGISTERED (before data): W2 KM re-read + sweep (c)** (comm-toy `cb813cf`; `w2/km.py` + 4 tests;
  `w2_out/retune/run_km.sh`, laptop).
  - **Configs, seeds 0-3 each (seed 0 = the original sweep's learner seed):**
    - the old pick 0.0015 / 8 / 0.001 (reference + re-read);
    - the runner-up 0.0018 / 10 / 0.0008 (re-read);
    - **sweep (c): d0 0.0012 / 8 / 0.001 and d0 0.0013 / 8 / 0.001** (relocation kept at 0.001).
  - **Instrument:** the same evaluation; the per-life KM median over lives born in the eval (blind, oracle) and over all
    lives (random). Per-life arrays are saved (npz), so later instrument changes need no retraining.
  - **Reproducibility check:** the old naive medians must reproduce exactly in the rerun.
  - **Gates on KM medians:**
    - P1m: blind >= 3x random;
    - P2: oracle >= 2x blind;
    - P4: blind >= 1,100.
    - If S(t) never reaches 0.5, the median is reported as "> horizon" and counts as passing P4.
  - **Pick: candidates are the two sweep (c) configs only.** All three gates on the 4-seed median AND in >= 3/4 seeds;
    the largest P2 margin among passers.
    - The old pick's 4 seeds are the same runs as its re-read, so it is reference only, not a candidate (Cam: "don't
      pick from that re-read alone"). The runner-up re-read is reported only.

- **RESULT (Nov 10 ~00:20): 4a hearing probe: FAIL as specified. The pipeline bug is DATA COVERAGE: within a life,
  a learner hears ONE message** (`docs/data/hearing_probe/`; comm-toy `3541b79`; `learnability.py --hearing-probe`).
  - **Setup:**
    - train only the heard rows (Dense_0 rows 100:111) + the policy head of each learner, from its navigator;
    - cross-entropy on the scripted listener's actions, on the states where it hears a beacon;
    - Adam 3e-4 x {1, 4, 16}, 300 steps x batch 128, from its own world's teacher data;
    - held-out accuracy measured on other worlds; the harness's unchanged 16-episode life-test gate.
  - **As specified: FAIL at every rate.**
    - Held-out accuracy 0.475 / 0.458 / 0.391 (x1 / x4 / x16). On "code-driven" states (nothing underfoot or in
      view, so the action depends only on the message): 0.35 / 0.36 / 0.37.
    - Gate 1.6% at every rate. The curves are flat from step 50.
  - **Pipeline checks: all OK.**
    - Input 111 dims (100 obs + 8 one-hot sector + flag + 2 offset / 9); kernel rows 111; one-hot sums to 1; unheard
      states all zero.
    - The broadcast sector equals the unclipped beacon -> nearest-patch direction (4/4; the r_oracle 16 clip did not
      bite here).
    - On code-driven states, the teacher's action is the heading of the heard sector in 100% of states.
    - Gradient reaches the heard rows (max |g| 2.2 at step 0). Dense_0 is not saturated (0.08% of units with
      |tanh| > 0.99).
    - Training ALL layers fails the same way (code-driven 0.36, gate 0%).
  - **The bug: coverage.** The 4 beacons sit within a 4x4 square, so they point at the same nearest patch. The patches
    don't move, and a learner hears only its nearest beacon.
    - Distinct sectors a learner hears in its whole training stream: **median 1** (43/64 learners hear 1, 19 hear 2,
      2 hear 3).
    - Code-driven accuracy is 0.72-0.76 on held-out states whose sector the learner heard, and 0.27 on unheard ones.
      82% of held-out code-driven states carry an unheard sector.
    - The life test re-places the beacons and the swap shifts the sector: it asks for a mapping the life's data never
      contained.
    - **This bounds every learnability arm so far:** gradient, predictive, replay, copy and copy-in-replay (elders hear
      the same clustered beacons).
    - The positive control passed (65%) because the scripted listener has the mapping built in.
  - **The pathway works when coverage is fixed.** Diagnostic `--probe-pool`: every learner trains on teacher data pooled
    over all 16 worlds (all 8 sectors).
    - **x16: gate 53% PASS** (median drop 60%; code-driven accuracy 0.76; all-states accuracy 0.65).
    - **x1: gate 39%** (drop 55%; accuracy still rising at step 300: 0.41 -> 0.61 on code-driven).
    - The 90% accuracy bar is not met in 300 steps at either rate. It is slower than expected, but it learns.
  - **Follow-up (comm-toy `b28c591`): with coverage the pathway is SLOW, not blocked; the gate is noisy.**
    - A pooled rerun at x16 (head only) gave **gate 47%** (the first run gave 53%; only the batch sampling differed).
      So the x16 "pass" is at the line, not a pass.
    - **Training-set accuracy is also < 90% at 300 steps:** code-driven 0.62 / 0.73 (x1 / x16, head only); all layers
      0.73 / 0.83.
    - **3,000 steps, all layers, x16:** training accuracy 0.97 (code-driven 0.95); held-out code-driven 0.86, all
      states 0.77. But **gate only 30%** (median swap drop 50%; the scripted positive control passes 65%).
    - **Reading:**
      - coverage is necessary;
      - with coverage, the mapping is learnable, but it needs ~10x the gradient steps a life gets (~300);
      - the life-test gate is noisy even for a well-fit listener.
  - **Coverage in a retuned candidate world** (fast learner, `--world 0.0013,8,0.001`): useful topics still give a
    median of 1-2 distinct tokens per life (nearest-food 1.5, richest-patch 1, distance band 2). **Moving to the
    retuned world (4b) does not fix coverage by itself.**
  - **Judgment call (flagged):** copy-in-replay was NOT rerun at x16. Its data has the same one-message coverage, so it
    would test nothing new until coverage is fixed.
  - **Implication for the fast learner:** its beacons are clustered the same way (offsets +-2), so a life likely hears
    ~1 token value too. To be measured when the harnesses move to the retuned world (4b). Relocating food there will
    vary the message within a life somewhat.

- **RESULT: the fast learner has the same coverage hole** (`fastlearner.py --coverage`, comm-toy `4cecd27`; prototype
  world, 64 lives x 4 listeners, adapt + test = 600 steps, navigator-trunk policy).
  - Median distinct heard tokens per life:
    - **useful topics:** nearest-food **1** (160/256 lives hear a single token), richest-patch **1** (221/256),
      distance band **1**;
    - useless topics: nearest-agent 6, energy band 2, noise 8.
  - So the topics that carry value are near-constant within a life. A life cannot identify a code from one token, and
    the swap test hands it a token it never heard. **This alone explains the null pilot** (reliance 0 in and out of
    distribution), independent of the meta-learner.
  - Meanwhile the useless topics vary, so "the message changes" in-life is predictive of nothing useful: a prior that
    learned to ignore changing tokens would be the rational one.
- **Item 1 decision: checkpoint + resume** (not non-preemptible).
  - Modal's non-preemptible containers cost 3x the list price for CPU and memory.
  - Preemption cost (e) about 23% extra (~89k updates run for 72k needed, before the incident).
  - Resume will be built and verified bit-identical on a short chunk before the next run over 2 h. The 4 reruns in
    flight have none.
- **Guard r3 live** (`~/throng-smoke/cap_guard_r3.sh`).
  - It refuses to start without an explicit CAP, or with CAP at or below metered spend (both tested).
  - Heartbeat file every tick; `guard_check.sh` reports its age and flags a dead guard after 3 min.
- **(f) exploratory compositionality (no claim)**, step6v8 s340 / s342 / s344 at U = 2000
  (`docs/data/step6v8/exploratory_comp.txt`):
  - **topsim above the permutation null in 3/3** (rho 0.49 / 0.53 / 0.60 vs null p95 ~0.17-0.19);
  - **held-out generalisation 0/3** (at or below chance 1/16; indistinguishable from shuffled);
  - rho is identical at u1000 and u2000: the meaning-to-message map is frozen from u1000.
  - So the codes are structured but do not generalise to unseen combinations.

- **RESULT (Nov 10 ~00:45): W2 KM re-read + sweep (c): NO PICK. Report** (`docs/data/w2_retune/summary_km.txt`; per-life
  arrays in `km_lives/`).
  - **Instrument check:** the reruns reproduce every old naive median and life count exactly (16/16 cells checked on
    the first 4 runs; same seeds, deterministic), so this is a true re-read.
  - **KM vs naive:** KM raises the blind medians modestly (+0 to +25%) and the oracle medians a lot (+50-90%; ~55-75% of
    oracle lives are censored). The old P2 margins were understated.
  - **Sweep (c), the candidates (P4 >= 1,100):**
    - **d0 0.0012 / 8:** blind KM 1189 / 1005 / 1093 / 1200, median **1141**; oracle median 4609 (4.04x). P4 on the
      median PASS, but **2/4 seeds -> fails.**
    - **d0 0.0013 / 8:** blind 1136 / 1055 / 1078 / 1303, median **1107**; oracle 5739 (5.18x). Median PASS, **2/4
      seeds -> fails.**
    - Both pass all three in 4/4 seeds at the old 1,000 bar. The 10% margin is what fails.
  - **Re-read (reported only, not picked from):**
    - **Old pick:** blind 1142 / 929 / 1111 / 970, median 1040; 2/4 seeds -> fails at 1,000 and at 1,100.
    - **Runner-up:** blind 1307 / 911 / 1696 / 1133, median **1220**; P2 3.89x; **3/4 seeds -> it would pass the new
      rule at 1,100.** Not a pick, per Cam (re-read only).
  - **Observation (not a test): lower drain does not buy lifespan.**
    - Blind intake per step falls with drain (0.00138 at d0 0.0015, 0.00105 at 0.0013, 0.00084 at 0.0012): the
      homeostatic agents eat less.
    - The median lifespan of the lives that ended moves only 986 -> 1,102.
    - Mean offspring per life is ~0.84 in every config, and ~half of the lives that ended had a child.
    - So the drain lever is nearly exhausted for P4.
  - **Reading of the rule (flagged for Cam).** I registered 1,100 as the per-seed P4 bar, before the data. Cam's text
    also reads as: the gates per seed at 1,000, plus the margin on the 4-seed median (>= 1,100).
    - Under the registered reading: no pick.
    - Under the alternative: both sweep (c) configs pass 4/4 at 1,000 with medians >= 1,100. The largest-P2 pick
      would be **d0 0.0013** (median 1,107, clearing the margin by 7; P2 5.18x); d0 0.0012 is sturdier on P4 (1,141)
      with P2 4.04x.
    - Not adopted: Cam to say which reading was meant.
  - **Options for Cam:**
    - (i) a fresh confirmation of the runner-up on 4 NEW learner seeds (4-7) under the same rule. It passed the rule
      on its re-read; fresh seeds avoid picking from the re-read. ~35 min laptop.
    - (ii) a smaller P4 margin. Any new bar is now set after seeing these numbers, so it would need fresh seeds to
      count.
    - (iii) a different world lever for lifespan (e.g. the reproduction cost c_rep or the provisioning cost).

- **Cam, Nov 10 (rulings on the ~00:50 report).**
  0. **Budget:** ~$17 of $42 spent; the guard stays at $41. **Cap for everything below: $8 total** (25% leeway); the rest
     is held for the W2 comm run.
  1. **Pick rule, as intended:** "blind KM median >= 1,100" meant the 4-seed MEDIAN. Per seed, the standard gates
     (P1m >= 3x, P2 >= 2x, P4 >= 1,000) in >= 3/4 seeds.
     - **W2 WORLD PICKED: d0 0.0013, 8 patches, relocation 0.001** (P2 5.18x, blind KM median 1,107, lowest seed 1,055).
       **Picked under Cam's clarified reading; the registered strict reading gave no pick.**
     - **Free confirmation, registered now:** the navigator retrain in the picked world re-measures the blind KM median
       on the retune evaluation. If the median over navigators is < 1,000, or fewer than 3/4 of the navigators reach
       >= 1,000, switch to d0 0.0012 and retrain there. Nothing else changes the pick.
  2. **Coverage: option (a).** A code cannot be learned from one word.
     - **Recorded: every learnability and fast-learner null so far tells us nothing about the levers. It is NOT
       evidence against them.** Learners heard ~1 message per life, so no lever could have been tested.
     - **Fix in the apparatus, not the world** (the pick stands): beacons are spread out and wander (a random walk at
       about agent speed), so a roaming listener hears several of them. The code's definition stays as registered
       (each beacon broadcasts the sector to ITS nearest patch); only where the beacons are and how they move changes.
       The same change goes to the fast learner's topics.
     - **APPARATUS GATE** (scripted beacons, no learner training; registered before any learning arm):
       - (i) **Coverage:** median distinct sectors heard per life >= 4 of 8, AND in the median life >= 3 sectors heard
         >= 25 times each. Fast learner: >= 4 tokens per useful topic, same rule.
       - (ii) **The scripted-listener control** passes the life test in >= 90% of lives AND beats blind (the navigator)
         >= 2x on deliveries (food eaten in the test window). If (ii) fails, fix the test (more probe trials per life,
         longer windows), not the bar.
       - **Fallback,** only if wandering / spread beacons cannot meet (i): faster food relocation (a world change: then
         re-check P1 / P2 / P4 on 4 seeds first).
     - **Learner bar unchanged:** cross-direction reliance >= 30% within one life in >= 50% of lives. It is now scored
       only on the sectors that life heard >= 25 times.
  3. **Order after the apparatus gate passes:**
     - (a) **learnability rerun, 2 arms x 5 seeds:** no levers vs ALL approved levers on (juvenile, predictive aux,
       sleep replay, copy signals, copy actions, young plasticity). Ablate only if all-on passes.
     - (b) **fast-learner VoI check** (4c) with the trunk FROZEN, same coverage rule.
     - For both, report steps to first reliance against the median lifespan (~1,100).
  4. **Checkpoint + resume:** build now (code + unit test, laptop, < 10 min). Required before any run over 2 h,
     including the W2 comm run.
  5. **Step 6:** stays parked. Structure without generalisation is what a holistic code with a regular layout looks
     like. In W2, compositionality is measured with held-out combos registered from the start.
  6. **(e):** when the 4 reruns land, `analyze_pop.py` against the registered u7500 bars, deduped by segment. No design
     changes.
  7. Guard r3 accepted; still waiting on M for the throng-guard-token secret.
- **REGISTERED (before data): navigator retrain in the picked world = the pick confirmation.**
  - 8 navigators (seeds 1000-1007), each trained exactly as retune.py trains its blind learner (learn.train, blind, 400
    updates, juvenile 500, world d0 0.0013 / 8 / 0.001).
  - Each is evaluated with retune.evaluate (setpoint reproduction, provisioning J 500, 16 worlds x 64 slots, 32
    founders, 6,000 steps): KM median lifespan of the lives born in the eval.
  - **Rule:** switch to d0 0.0012 iff the median over the 8 navigators is < 1,000 OR fewer than 6/8 (3/4) reach >= 1,000.
  - Output: `w2_out/nav_w13/` (comm-toy `1b6077e`).

- **RESULT (Nov 10 ~02:20): pick confirmation via the navigator retrain: PICK STANDS (d0 0.0013 / 8 / 0.001)**
  (`docs/data/nav_w13/`).
  - Blind KM medians of the 8 navigators (lives born in the retune eval): 1001 / 961 / 1064 / 1062 / 1052 / 1148 / 1113 / 1003.
  - **Median 1057; 7/8 reach >= 1,000.** The registered switch condition (median < 1,000 or < 6/8) is not met.
  - Margin note: the median is 4% above the old 1,000 bar. Seed noise is as wide as the gap between candidates.
  - The 8 navigators are the W2 newborn starting weights for this world. Navigator gate (each >= 0.7x the best):
    8/8.

- **Item 4 DONE: checkpoint + resume** (comm-toy `1fd73d0`).
  - `scripts/ckpt.py`: atomic save (tmp + fsync + os.replace) and load. `pop_toy.py --ckpt PATH --ckpt-every N`
    saves the full loop state:
    - brains, optimiser, per-brain PopArt, birth/generation tables, replacement schedule, navigator assignment;
    - the numpy Generator and the JAX key, the env state, the head-drift reference;
    - the output file's byte offset. On resume the JSONL is truncated to that offset, so rows written after the
      checkpoint are not duplicated.
  - **Test** (`tests/test_pop_ckpt.py`, 55 s): a tiny run straight through vs the same run killed after u5
    (checkpoints at u2 / u4) and restarted on the same input.
    - It really resumes from u4; births fall after the resume point; train updates are 1..6 with no duplicates.
    - **Every row and the final brains are identical: PASS.**
    - Mutation check: restoring a fresh numpy Generator instead of the saved one FAILS the test, so the test has
      teeth.
  - **Default path (no --ckpt) is row-identical to the pinned 5c5b07a** (21/21 rows on the same tiny config).
  - The W2 comm-run harness will use the same `ckpt.py`.

- **Coverage apparatus (ruling 2) built** (comm-toy `65b9bd3`, defaults unchanged; learnability default path verified
  row-identical, 14/14 lives).
  - **Beacons:** `--beacon-spread k` puts the 4 beacons at (+-k, +-k); `--beacon-walk` = one random move per step (agent
    speed). The code is as registered: each beacon broadcasts the sector to its own nearest patch.
  - **Deviation (flagged): `--young-seek`.** A young (juvenile period) that hears nothing steps toward the nearest
    beacon instead of standing still. Otherwise wandering beacons leave the young out of range for good.
  - **Life test `--test heard25`:** the swap is a random derangement among the sectors that life heard >= 25 times
    (other sectors unchanged). A life with < 2 such sectors cannot be scored and counts as a fail.
  - **`--no-train`:** the apparatus gate; with `--sanity`, the blind navigator is also run on the same lives and keys.
- **APPARATUS GATE (i) coverage: PASS in both harnesses** (picked world 0.0013 / 8 / 0.001, navigators `nav_w13`,
  spread 6 + walk; `docs/data/apparatus/`).
  - **Learnability (natural lives):**
    - scripted-listener lives (n 264, median lifespan 2,144): median 8 distinct sectors, 6 sectors heard >= 25x;
    - navigator-policy lives (n 428, median lifespan 891): 7 distinct and 5 >= 25x.
  - **Fast learner:**
    - at its old 600-step lives, richest-patch FAILS (3 distinct, 2 >= 25x): every speaker points at the same single
      richest patch, so its direction moves slowly;
    - **with 1,100-step lives (adapt 800 + test 300; the W2 median lifespan, Cam's ~1,100): all 3 useful topics
      PASS** (nearest-food 7 / 4, richest-patch 4 / 3, distance band 5.5 / 4). Spread 9 gives the same.
    - **Richest-patch is exactly at the bar.**
    - **Change (flagged): fast-learner lives become 1,100 steps.**
- **APPARATUS GATE (ii) scripted control: FAILS so far.** Fixing the test, not the bar.
  - 16 trials x 300 steps: pass 58.7% (true 14.8 vs swap 8.8).
    - The failures are dilution: 88 of 109 have drop < 30%; only 21 are significance failures.
    - Pass rises with the number of well-heard sectors: 0% at 2, 21% at 3, 43-55% at 4-6, 85-87% at 7-8. The probe
      episode re-places the beacons, so much of what it hears falls outside the life's set and is not swapped.
  - **48 trials x 600 steps (Cam's named fixes): 68.2%.** All 84 failures are drop < 30%; zero significance failures.
    More trials cannot close a dilution gap.
  - **Next test fix (running): `--test-start life`.** Each probe episode starts in the life's own world at its end
    (patch centres + the beacons where they were; the learner beside a random beacon).
  - **Deliveries vs blind: passes trivially** (scripted 41-60x blind). In this low-drain world the homeostatic
    navigator eats only ~0.25-0.7 per window, because it barely needs to. **Flag: this half of (ii) is not
    informative here.**

- **RESULT (Nov 10 ~03:35): apparatus gate (ii): the scripted control does not reach 90% under any test fix tried.
  STOP for Cam** (comm-toy `e1353c9`; `docs/data/apparatus/k6_scripted_*.json`; 264 scripted-listener lives each).

  | test variant | trials x window | pass | fails: drop < 30% / significance |
  |---|---|---|---|
  | heard25, whole-window food | 16 x 300 | 58.7% | 88 / 21 |
  | same | 48 x 600 | 68.2% | 84 / 0 |
  | + probe starts in the life's own world | 16 x 300 | 46.6% | 95 / 46 |
  | same | 48 x 600 | 61.0% | 98 / 5 |
  | **heard25, food credited to the last heard TRUE sector (`--score last-heard`)** | 16 x 300 | 58.3% | 57 / 53 |
  | **same** | **48 x 600** | **79.9%** | 50 / 3 |

  - **What limits it: lives with few well-heard sectors.**
    - Best variant by number of sectors heard >= 25x: 2 -> 17%, 3 -> 50%, 4 -> 53%, 5 -> 80%, 6 -> 88%, 7-8 -> 100%.
    - With 2-4 sectors (often adjacent), the derangement is mostly a 45-degree turn, which costs a perfect follower
      little food.
    - At 48 x 600, failures are drop < 30%, not noise; more trials will not close it.
  - **Options for Cam** (each changes what counts as a scorable life or a swap, so it is Cam's call):
    - (a) **a life is scorable only if it heard >= 5 sectors >= 25x.** Scripted control: 91.9% pass on the 75% of lives
      that are eligible. Navigator-policy lives eligible: 60%. At >= 6: 96.5% pass, 54% / 34% eligible. At >= 4: 86.8%.
    - (b) **a true cross-direction swap:** every heard sector maps to one >= 90 degrees away. Not implemented; lives
      whose well-heard sectors are all adjacent could not be scored.
    - (c) **more coverage** (wider spread or faster beacons) to thin out the few-sector tail. An apparatus change; the
      code is unchanged.
  - **Option (c) tried (within the apparatus brief, beacons still at agent speed): spread 9, last-heard, 48 x 600 ->
    68.8%** (260 lives). It is WORSE than spread 6 (79.9%), and the 2-4-sector tail is not thinner (65 vs 66 lives).
    Coverage medians are unchanged (8 / 6 scripted, 7 / 5 navigator). Wider spacing does not fix it.
  - **Last-heard scoring is a metric change, not only a test fix.** "Drop" becomes food credited to well-heard
    sectors, and it would also become the learner's scoring. Cam to accept.
  - **Denominator question for (a):** if lives below the minimum are unscorable and count as fails, ~40% of
    navigator-policy lives are ineligible at >= 5. "Reliance in >= 50% of lives" would then need ~83% of eligible
    lives, a much stricter bar.
    - Suggest: the 50% is over ELIGIBLE lives, with a minimum count of eligible lives per arm (e.g. >= 50).
    - Eligibility reported per arm, since a lever that changes movement changes how many sectors a life hears.
  - **Suggested replacement for the "2x blind" half of (ii):** scripted with true messages vs the same scripted
    listener with silence, on the same lives and keys. That is the VoI of the code in this harness (built,
    `--no-train --sanity`; not yet run).
  - Recommended: (a) at >= 5 with last-heard scoring and 48 x 600. It is the smallest change, and the scripted control
    clears 90% on the eligible lives.

- **Cam, Nov 10 (rulings on the ~04:30 update).**
  1. **Gate (ii) test:**
     - "last-heard" scoring is ACCEPTED as THE metric for the scripted control, the negative control and every learner
       arm (calibrated before any learner data, applied identically to all arms);
     - **(a) ACCEPTED:** a life is scorable iff it heard >= 5 sectors >= 25x;
     - **denominator = eligible lives only:** >= 50 eligible lives per arm (add lives if short), eligibility rate
       reported per arm;
     - if an arm's eligibility differs from the no-lever arm by > 15 points, also report pass rates stratified by
       sector count (5 / 6 / 7-8);
     - (b) as a free secondary, no bar: the pass rate on probes whose swap moved >= 90 degrees.
  2. **Negative control:** a non-listener (wire-off navigator) on the same eligible lives must pass in <= 10%. Both
     control rates are reported with CIs. If the non-listener passes > 10%, the test is too easy: stop and report, don't
     tune.
  3. **VoI half of (ii):** the scripted listener with true messages vs the same listener with silence, same lives and
     keys. **Bar: true-message food >= 1.5x silence, paired > 2 SE.** If it fails, the code isn't worth learning here:
     stop and report.
  4. **Young-seek:** accepted for the apparatus only, as test scaffolding, iff it triggers on geometry (no speaker within
     the hearing radius), never on the message channel. It stays out of the W2 comm run unless M approves it as an
     inborn trait.
  5. **Fast learner:** 1,100-step lives accepted. Richest-patch: try a faster beacon walk for margin (wider spread is
     ruled out). Score held-out generalisation on qualifying lives only, and report how many qualify.
  6. **Freeze:** once (2) and (3) pass, freeze the test (commit hash here). Then learnability 2 arms x 5 seeds (no
     levers vs all approved levers), then fast-learner 4c with the trunk frozen. Bars as registered; batch cap $8 total.
  7. **Laptop: only checks under ~10 min.** The ~20 min apparatus jobs and both arm runs go to Modal (small CPU
     containers, --ckpt on).
  8. **(e):** dedupe, then analyze_pop.py against the registered u7500 bars. Send the first three seeds as they land;
     the verdict waits for s322.
- **Implemented** (comm-toy `b61afb2`):
  - `--min-sectors 5` eligibility; pass over eligible lives with Wilson 95% CIs; stratified 5 / 6 / 7-8;
  - the >= 90 degree secondary, defined as **probes whose derangement moves the well-heard sectors >= 90 degrees on
    average** (requiring every sector to move >= 90 degrees left no qualifying probes in a smoke run). Lives need
    >= 4 such probes;
  - negative control = the untrained navigator (zero hearing rows: a non-listener), true vs swap on the same lives and
    keys;
  - VoI = the scripted listener with true messages vs muted hearing, same lives and keys;
  - young-seek now triggers on geometry (Chebyshev distance to the nearest beacon > R_HEAR). This is identical to the
    old trigger whenever hearing is not muted, so the earlier apparatus numbers stand.
  - **`learnability.py --ckpt`:** bit-identity test (all levers; prototype world so deaths follow the resume), and a
    mutation (fresh rng) fails it.
  - **Fast learner `--beacon-speed`:** speed 2 and 3 do NOT add margin. Richest-patch stays at 4 distinct / 3 heard
    >= 25x, and the heard fraction falls 0.62 -> 0.34 / 0.30 (speakers outrun listeners). Speed stays 1.
  - Modal: new app `throng-w2jobs` (`scripts/modal_w2jobs.py`): CPU 4 cores; --ckpt to the volume, committed every
    2 min.
- **REGISTERED (before data): apparatus gate (ii) run `app_gate_scripted`** (Modal, ~$0.2).
  - Picked world, nav_w13; spread 6 + walk + young-seek; heard25 + last-heard; >= 5 sectors >= 25x; 48 probes x 600
    steps; 16 worlds x 150 updates; no training; scripted listeners.
  - **Gate (ii) PASSES iff:** >= 50 eligible lives AND scripted control >= 90% (eligible) AND non-listener <= 10% AND
    VoI true >= 1.5x silence with paired > 2 SE.
  - **If it passes, the test is frozen at `b61afb2`.**

- **RESULT (Nov 10 ~06:35), PARTIAL: (e) population, 5 of 8 seeds complete** (deduped: the last complete segment;
  `docs/data/pop_e/analyze_pop_partial_5seeds.txt`; POP s320 / s321 / s323, control s325 / s327).
  - **Primary at u7500:** cross-spot drop +1% / -0% / -3% (paired ~0, SE 0.01); true-message deliveries 0.42 / 0.52 /
    0.44 vs the gain bar 1.41 (control 0.41 + 1.0). **0/3 POP seeds pass.**
    - **The primary outcome is already determined: at most 2/5 < 3/5 -> rung 3 FAIL**, whatever s322 / s324 show.
      The formal verdict waits for s322, per Cam.
  - **Secondary MI (bits, shuffle-corrected):**
    - POP: 1.14-1.29 at u1500 -> 0.17-0.22 at u3500 -> 0.03-0.05 at u5500 -> 0.00-0.01 at u7500.
    - Control: 1.16-1.38 -> 0.40-0.52 -> 0.03-0.04 -> 0.006.
    - The founders' code dies out through turnover in BOTH arms by generation 3; use does not keep it alive. POP
      median at u7500 0.010 vs control 0.006: FAIL.
  - **Fire 5 at u9000:** 0/3 (told-fire-5 crafts no higher than control).
  - **Remaining:**
    - s322 and s326 are finishing (5c5b07a).
    - **s324 was preempted AGAIN at ~u6329 (~05:35) and restarted from u1.** I cancelled that restart and relaunched
      s324 at 06:32 on `throng-pop-ck`, which is checkpointed: comm-toy `003ed2c`, default path row-identical to
      5c5b07a, `--ckpt` every 250 updates, volume committed every 2 min. Same seed and args; output
      `pop_on_s324_003ed2c.jsonl`; ETA ~14:30.
  - **Cost:** (e) ~$10.1 so far (estimated from metered spend); finishing adds ~$1.3, so ~$11.4 against the $12 cap.
    Option for Cam: cancel s324 (~$1.0 saved); the primary outcome is already fixed.

- **RESULT (Nov 10 ~06:40): fast-learner 4c VoI check: MARGINAL, and the FL life test is not calibrated. STOP for Cam
  before the frozen-trunk pilot** (comm-toy `62b6758`; `docs/data/fl_4c/`; picked world, spread 6 + walk, 1,100-step
  lives, 32 codes x 8 episodes x 4 listeners).
  - **Built:**
    - `--freeze-trunk` (only side_enc / gru / side_head + the value head learn);
    - per-episode qualification (an episode qualifies iff its listener heard >= 5 tokens >= 25x in the 800-step
      adaptation; a code is scorable with >= 4 qualifying episodes);
    - `--scripted-voi`: a scripted listener that decodes the life's code perfectly and walks the decoded direction,
      true vs silence on the same episodes;
    - `--score last-heard` (the ruled metric).
  - **Nearest-food, the useful training topic:**
    - whole-window scoring: VoI 0.70x (navigator fallback; z -3.9) / 1.09x (random fallback);
    - **last-heard scoring: 1.53x (z 3.3, navigator fallback) PASS / 1.44x (z 4.1, random fallback) just under 1.5x.**
  - **Richest-patch (the held-out topic): only 11-15 of 1,024 episodes qualify (1-1.5%).** Unassessable. Faster beacon
    walks do not help (above).
  - **The FL test's positive control fails: the perfect decoder passes the true-vs-swap test in only ~10% of scorable
    codes** (nearest-food, either fallback; learnability's calibrated test gives 92%). A frozen-trunk pilot scored on
    this test would be uninterpretable.
  - **Appetite confound found (applies to learnability's VoI too):** with the navigator as the scripted listener's
    silence fallback, the comparison partly measures the navigator's satiety.
    - The fast learner floors energy (hungry navigator): VoI looks low. Learnability (sated navigator, idles): VoI
      looks high (10.5x in a smoke run).
    - With a random-move fallback in both arms (appetite-free), the learnability smoke gives **2.8x (z 7.1)**: the code
      is genuinely worth something there.
    - Learnability now reports both (`voi_random_fallback`). The registered VoI bar used the navigator-fallback form.

- **RESULT (Nov 10 ~06:45): APPARATUS GATE (ii) PASSES** (`docs/data/apparatus/app_gate_scripted_b61afb2.json`; Modal).
  - 264 scripted-listener lives (median lifespan 2,144); **197 eligible (74.6%)**.
  - **Scripted control 91.9%** (Wilson 95% CI 87.2-94.9%; bar >= 90% on the point estimate). Stratified: 5 sectors
    80% (n 55), 6 88% (n 42), 7-8 100% (n 100).
  - **Non-listener (negative control) 0.0%** (CI 0-1.9%; bar <= 10%).
  - **VoI: true 23.7 vs silence 6.9 food per probe = 3.42x, paired z 50** (bar >= 1.5x, > 2 SE). This is the
    navigator-fallback form; the appetite-free form was not in this run.
  - Secondary (no bar): probes whose swap moved the well-heard sectors >= 90 degrees on average: 92.4% (197 lives).
  - Coverage (i) re-confirmed: median 8 distinct / 6 sectors >= 25x.
  - Note: the control's CI lower bound (87%) is below 90%. The bar was on the point estimate.
- **TEST FROZEN at comm-toy `b61afb2`** (Cam ruling 6). The learner arms run at `62b6758`.
  - Its diff from `b61afb2` adds training-side flags only (--seek-age, --snap-ages, --ckpt), the appetite-free VoI
    and report fields.
  - **Verified: the same gate config at both commits gives identical per-life results (true / swap / pass / blind)
    and an identical heard25 report** (8 lives, tiny config).
- **REGISTERED (before data): learnability 2 arms x 5 seeds** (Modal `throng-w2jobs`, plan `learn_arms`, `62b6758`,
  --ckpt on; est. ~$2).
  - **Common to both arms:**
    - picked world; nav_w13 navigators; 16 worlds x 4 learners x 150 updates;
    - spread 6 + walk; young-seek as scaffolding at `--seek-age 500` in BOTH arms (geometry trigger; off in the test);
    - the frozen heard25 test: last-heard, >= 5 sectors >= 25x, 48 probes x 600 steps;
    - age snapshots at 250 / 500 / 1,000 for steps to first reliance.
  - **No levers:** `--juvenile 0`, pred 0, replay 0, copy 0, young-lr 1.
  - **All levers:** `--juvenile 500 --pred 1.0 --replay 8 --copy 1.0 --copy-replay --young-lr 4`.
    - **"Copy signals" has no counterpart in this listener-only harness** (learners do not speak); flagged.
  - **Bar (registered):** cross-direction reliance >= 30% (paired > 2 SE) in >= 50% of ELIGIBLE lives, per arm,
    >= 50 eligible lives. Eligibility rate reported per arm; stratified 5 / 6 / 7-8 if eligibility differs from the
    no-lever arm by > 15 points. Steps to first reliance vs median lifespan.
  - Ablate only if all-on passes.

- **Cam, Nov 10 (rulings on the ~06:45 / ~06:46 reports).**
  1. **(e) population: VERDICT FAIL as registered.**
     - Primary 0/3 POP seeds, so at most 2/5 < 3/5. Secondary FAIL (MI at u7500: POP 0.010 vs control 0.006). Fire 5:
       0/3.
     - **s324 CANCELLED** (07:03). s322 and s326 finish and go into the record.
     - **Recorded: "Separate-brains toy population: the founders' code dies through turnover in both arms; cultural
       transmission not shown. Consistent with W2: one-life learnability is the bottleneck." The toy population is
       CLOSED.** Transmission moves to W2 once learnability passes.
     - Free diagnostic from the existing dumps (laptop): per generation, speaker MI and listener reliance by age. Did
       any newborn (gen 2+) ever acquire the founders' code? Did the founders' listeners rely at u1500?
     - Dictionaries per generation go to `docs/data/for_m`.
  2. **Gate (ii) PASS accepted.** The bar was on the point estimate (91.9%). **CI caveat: the 95% CI lower bound is
     87.2%, below 90%.**
  3. **VoI:** the registered bar stands for this gate (no moving it after the fact). The random-fallback VoI on the
     full gate lives is a REPORTED number (`app_gate_voir`, launched 07:03). **From now on every harness's VoI bar needs
     BOTH forms >= 1.5x** (navigator fallback and random fallback).
  4. **Copy signals** applies in the W2 comm run (newborns speak), not here.
  5. **Young-seek in both arms** (seek-age 500, geometry trigger, off in the test): accepted.
  6. **Fast learner 4c: PARKED, no spend.**
     - If learnability all-on passes, it stays parked.
     - If all-on fails, recalibrate it with the learnability test design (last-heard, eligibility, stratified). The
       held-out topic may be replaced first: richest-patch cannot reach coverage.
  7. **Learnability arms decision tree (registered now).** Bar per arm as registered; also reliance at ages 250 / 500 /
     1,000 and the age at first reliance vs median lifespan.
     - **ALL-ON PASS:** leave-one-out ablations, 6 arms x 3 seeds (~$3.6, inside the $8 cap). Then report; Cam writes
       the W2 comm-run registration.
     - **ALL-ON FAIL with reliance rising with age** (1000 > 500 > 250, paired > 2 SE): learning is real but too slow.
       Report the slope and the extrapolated age at 30%. Launch nothing; Cam chooses between a longer childhood and
       stronger levers.
     - **ALL-ON FAIL, flat:** rerun the swap-on-predictions check (does the predictive head use the message?), laptop
       only; report.
     - **NO-LEVER PASS:** report immediately.
  8. Spend fine (~$20.65; batch ~$0.5 of $8).
  - **Deviation (flagged): the age trend cannot be tested paired.** The running arms (`62b6758`) store each age's pass
    fraction over the eligible lives that reached it, not each life's pass at each age. So the trend is tested with a
    two-proportion z-test between consecutive ages (pooled over an arm's seeds), the later set being a subset of the
    earlier. Rising = 1000 > 500 > 250 with each step z > 2. Per-life snapshot outcomes are to be stored in later runs.

- **RESULT: (e) transmission diagnostic (free, from the existing dumps; 5 complete seeds; s322 / s326 to be appended)**
  (`docs/data/pop_e/transmission_diag.py`, output `transmission_diag_5seeds.txt`; dictionaries for M in
  `docs/data/for_m/pop_e_dictionaries_5seeds.json`).
  - **Q2: was there a working channel to pass on? YES.**
    - At u1500 (all founders except slot 0, just replaced) POP population cross-spot reliance is **52% / 53% / 43%**
      (s320 / s321 / s323), with founders' speaker MI 1.1-2.0 bits per brain.
    - It collapses after turnover: 1-7% at u3500, ~0 from u5500 on.
  - **Q1: did any newborn (gen >= 2) acquire the founders' code? NO.**
    - Across 17 newborn observations per seed (each brain at each eval point), **a newborn's modal word per fire never
      matches the founders' pooled modal word (agreement 0.00 everywhere)**.
    - POP gen-2 newborns carry only 0.13-0.35 bits at u3500 (ages 500-1,500); gen 3+ ~0.
  - **Caveats:**
    - The founders do not share one code: each founder's words match the founders' pooled dictionary on only 25-50% of
      fires (separate brains, separate words).
    - Control-arm (wire-off) newborns carry MORE fire information than POP's (0.34-0.82 bits at u3500) although no
      listener can use it. Newborn MI is not evidence of transmission.
    - The ~0.62 bits of the age-0 brain at u9000 in every seed is an inborn correlation of a fresh navigator once fire 5
      is lit, not acquired code.
  - **Reading:** the founders had a working channel; nothing of it was passed on. Each new generation starts from a
    navigator with no code, and one life with copying (imit 1.0, sensitive period 1,000) does not acquire the elders'
    words. This is the toy analogue of the W2 one-life learnability bottleneck.
- **RECORDED (Cam): "Separate-brains toy population: the founders' code dies through turnover in both arms; cultural
  transmission not shown. Consistent with W2: one-life learnability is the bottleneck." The toy population is CLOSED.**

- **RESULT (Nov 10, read ~18:10): LEARNABILITY 2 ARMS x 5 SEEDS (frozen test): ALL-ON FAIL, reliance RISING with age;
  NO-LEVER FAIL, flat zero.** (`docs/data/learn_arms/`: per-seed JSONs + logs, `summarize.py` (registered),
  `age_trend.py`; Modal `62b6758`; all 10 exited 0.)
  - **No levers:**
    - pass 0/934 eligible lives (0.0% in every seed; CI upper bounds ~2%);
    - eligibility 15-20%; median lifespan ~340; **12 gradient steps per life**;
    - 0% at every snapshot age (250 / 500 / 1,000).
  - **All levers:**
    - pass 4-11% per seed (s0 11.0%, s1 9.1%, s2 10.1%, s3 3.9%, s4 4.7%); **pooled 90/1,164 = 7.7%** (bar >= 50%):
      FAIL in 5/5 seeds;
    - eligibility 58-64%; median lifespan ~890; 476-510 gradient steps per life.
  - **Eligibility differs by 43 points**, so stratified per the ruling. All-on by sector count: 5 -> 1-7%, 6 -> 3-12%,
    7-8 -> 4-17%. No-lever: 0% in every stratum. The all-on > no-lever difference holds within every stratum.
  - **Age trend (all-on, pooled):** age 250 2.7% (32/1,164) -> 500 6.4% (74/1,164) -> 1,000 10.4% (69/661).
    - Two-proportion z: +4.2 and +3.1, so **RISING** under the registered rule (1000 > 500 > 250, each step z > 2).
    - Caveat: the 1,000 set is the lives that survived to 1,000 (a subset), so survivor selection may flatter the late
      point.
  - **Extrapolation (corrected ~18:20; see `age_trend.txt`).** What is extrapolated is the FRACTION OF LIVES
    PASSING. The registered arm bar is 50% of eligible lives; "30%" is also given, since Cam's wording allows either.
    - Segment slopes:
      - 250 -> 500: **14.4 points per 1,000 steps** (the SAME 1,164 lives at both ages, so no survivor selection);
      - 500 -> 1,000: **8.2 points per 1,000** (survivors only; the flatter step).
    - **To 50% (the bar): ~3,700-5,800 steps** (steep to shallow slope; the 3-point linear fit gives ~4,850). That is
      ~4-7x the median lifespan of ~890.
    - **To 30%: ~2,400-3,400** (linear ~2,900).
    - A log-age fit gives ~36,000 to 30%: the curvature is unknown from 3 points.
    - The unpaired z understates the paired z (pass outcomes correlate within a life), so the deviation is
      conservative.
  - **Reliance is not stable within a life:** the share of all-on lives that pass at ANY point (9.5-17%) exceeds the
    end-of-life pass rate (3.9-11%). Some lives pass at a snapshot and later stop passing.
  - **First reliance:** 9.5-17% of eligible all-on lives ever pass at a snapshot or at death; the median first age
    among those is 500 (s1: 735), against a median lifespan of ~890.
  - **Lifespan confound in the no-lever arm, partly resolved by the matched-age snapshots:**
    - With juvenile 0, no-lever newborns live ~340 steps (~12 gradient steps per life vs 476 all-on).
    - **At age 1,000 both arms have had about the same number of PPO updates: no-lever 0/380 eligible lives pass,
      all-on 69/661 (10.4%).**
    - So plain RL gives 0% at the age where the lever package gives ~10%. "The levers beat plain RL at matched age" is
      supported; WHICH lever carries it is not (that is what the ablations would show).
    - A longer childhood with NO levers has a 0% data point behind it.
  - **Branch (Cam's registered tree): ALL-ON FAIL with reliance rising -> learning is real but too slow for one life.
    Report the slope and extrapolation; LAUNCH NOTHING.** Cam to choose between a longer childhood and stronger levers.
- **RESULT: gate rerun at `62b6758` (`app_gate_voir`) reproduces the frozen gate exactly** (scripted 91.88%,
  non-listener 0%, 197 eligible).
  - **Appetite-free VoI (random fallback, reported): true 25.6 vs silence 8.3 = 3.09x, paired z 55.**
  - Navigator-fallback form: 3.42x. Both forms are >= 1.5x.
- **RESULT: (e) final record, 7 seeds** (`docs/data/pop_e/analyze_pop_final_7seeds.txt`, `transmission_diag_7seeds.txt`;
  dictionaries `docs/data/for_m/pop_e_dictionaries_7seeds.json`).
  - **Primary 0/4 POP seeds** (s322: cross drop +1%, true-message 0.45 vs gain bar 1.42). **Secondary FAIL** (POP
    median 0.010 vs control 0.006). **Fire 5: 0/3** (s322 has no fire-5 probe).
  - s326 complete. **s322 incomplete** (stopped at u8692 in its 7th segment), but its u7500 primary and MI are
    complete. s324 cancelled per Cam.
  - Transmission diagnostic on 7 seeds, unchanged: founders' reliance at u1500 43-53% (s322 52%); newborn-founder word
    agreement 0.00 in every seed.
  - **VERDICT FAIL as registered; toy population CLOSED.**
- **COST OVERRUN, disclosed (my error):** (e) cost ~$15.15 against its $12 cap (throng-pop $6.51 on Nov 9 + $8.57
  on Nov 10, throng-pop-ck $0.08): **26% over, beyond the 25% leeway.**
  - s322 was preempted yet again at ~11:30 and restarted from u1 on the un-checkpointed app.
  - When I moved s324 to the checkpointed app, I did not move s322.
  - My session ended ~07:20 and took the Mac guard with it (heartbeat stale from 07:21). Nothing watched s322 until
    the operator stopped throng-pop at ~17:40.
  - The global $41 guard cap was never at risk (metered $24.76).
  - **Lessons:**
    - any run over 2 h goes on a checkpointed app from launch;
    - a dead guard must be noticed (`guard_check.sh` at the start of every session);
    - per-item caps need their own watch, not only the global guard.
  - New-work batch (cap $8): throng-w2jobs $2.75 + pop-ck $0.08 = ~$2.83.

- **Cam, Nov 10 (rulings on the ~18:20 report).**
  1. **Which 30%: the registered bar is 50% of eligible lives passing, a passing life having reliance >= 30%.**
     Crossings, labelled:
     - **ARM BAR (50% of lives pass): ~3,700-5,800 steps (4-7x a life); linear 3-point ~4,850.** The log-age fit makes
       it far worse.
     - (Secondary) 30% of lives pass: ~2,400-3,400 steps; linear ~2,900.
  2. **Read accepted.** Matched age (0/380 vs 69/661) shows the lever package beats plain RL. No-lever gets ~12
     gradient steps per life vs all-on ~480 (40x), so the lead hypothesis is **in-life learning compute**. "Longer
     childhood without levers" is off the table.
  3. **Next batch: dose-response** (compute per life vs experience per life).
     - Frozen test b61afb2, same eligibility, young-seek as before, all-on base, 5 seeds each:
       - **R32:** replay 8 -> 32;
       - **R128:** replay 8 -> 128;
       - **J2000:** juvenile 500 -> 2000 (replay 8).
     - 2-min price probe first. **Item cap $4** (batch total $8, ~$2.83 used); if the probe says > $4, cut R128 to 3
       seeds.
     - Report per arm: pass at ages 250 / 500 / 1,000 (+ 2,000 for J2000) and at death; ever-pass vs at-death; pass vs
       cumulative gradient steps; median lifespan; eligibility.
     - **Registered decision:**
       - any arm >= 50% at death (>= 50 eligible) -> report; that setting goes into the W2 registration, ablations
         after;
       - R arms rise with dose (age-1000 pass R128 > R32 > 10.4%, z > 2) but none passes -> compute-limited; Cam picks
         the next dose or the fast learner;
       - R flat, J2000 rises -> experience-limited; longer childhood becomes a W2 life-history question;
       - all flat -> the learner is the limit, not the dose; the fast learner becomes the priority.
  4. **Free checks first** (laptop; existing snapshots and dumps):
     - (a) prove each lever ran in all-on (its own loss or signal is nonzero and changes over a life), and say who
       copy-actions copies;
     - (b) prove copying ran in (e): the copy-signal loss over a gen-2 newborn's life, and newborn-vs-elder token
       agreement on the same inputs vs chance. If it was a no-op, fix it before copying goes into W2; (e)'s verdict then
       reads "uninformative" rather than "transmission failed";
     - (c) the predictive-head swap check on the all-on snapshots.
  5. **Fast learner:** free laptop recalibration now with the learnability test design:
     - last-heard scoring, eligibility, stratified probes;
     - positive control >= 90%, negative control <= 10%, both VoI forms >= 1.5x;
     - the new held-out topic picked and registered before any learner runs;
     - no Modal.
  6. **(e) overrun accepted.** Addition: **the per-item watch must STOP the item at cap + 25%**, not just report it.
     The Modal guard (waiting on M's secret) fixes the guard dying with the session.
  7. (e) closed, FAIL as registered (7 seeds); check 4b decides how the verdict reads.

- **REGISTERED (before data): dose-response batch `dose`** (Modal `throng-w2jobs` at comm-toy `bd6f4af`; test path
  re-verified identical to the frozen `b61afb2`, 8/8 lives; --ckpt on).
  - **Price probe (laptop, training only; the eval cost is known from the all-on arms):**
    - training per update in steady state: R128 ~28 s (~70 min of training), R32 ~7 s by linear scaling (the probe
      window was too short to read), J2000 3.4 s (~9 min);
    - est. per run: R32 ~$0.22, R128 ~$0.42, J2000 ~$0.16;
    - 5 + 5 + 5 seeds = ~$4.0, at the $4 cap. **Per the rule, R128 is cut to 3 seeds: total est. ~$3.2.**
  - **Arms:**
    - all-on base (juvenile 500, pred 1.0, copy 1.0, copy-in-replay, young-lr x4) plus: **R32** (replay 32) x5,
      **R128** (replay 128) x3, **J2000** (juvenile 2000, replay 8) x5;
    - snapshots 250 / 500 / 1,000 (+ 2,000 for J2000); per-life snapshot pass and gradient steps are now stored, so
      age trends can be tested PAIRED;
    - everything else as the all-on arm (frozen heard25 test, >= 5 sectors, 48 x 600, seek-age 500, spread 6 + walk).
  - **Flag:** in this code `--juvenile` also sets the period over which young-lr x4 decays. J2000 therefore lengthens
    both provisioning AND heightened plasticity (as specified: juvenile 500 -> 2000).
  - **Decision rule: as Cam registered (above).** "Rising with dose" = age-1,000 pass R128 > R32 > 10.4% (two-proportion
    z > 2 per step).
  - **Item watch:** `~/throng-smoke/item_watch.sh`, APP throng-w2jobs, CAP 4.0. It **stops the app at $5.00** (cap +
    25%), counting all metered spend since its start (only this item runs).

- **RESULT (Nov 10 ~18:50): FREE CHECKS 4a-4c** (laptop; comm-toy `c92d447`; `docs/data/check_4b/`).
  - **(a) Every lever ran in the all-on arm.** `--log-levers`, 30 updates, all-on config, picked world; each lever's own
    signal on the same rollout, measured before the update, after the PPO epochs, and after the replay passes:
    - **Predictive aux:** loss 0.89 at mean age ~64, falling to 0.06-0.14 later in life.
    - **Copy actions:** CE 5.8 at the start of life, falling to ~1.3-1.9. 64 learners copying at mean age 64, falling
      to 3-8 by age ~1,200 (copying stops at age 1,000).
      - **Who it copies: 2 immortal scripted elders.** They forage by the heard code (eat on food, head for visible
        food, else walk the heard sector) and are copied when within hearing range of a young learner.
    - **Replay:** 32 extra passes per update once memory is full. Isolated effect per update: **the replay passes cut the
      predictive loss by 0.063** (the PPO epochs by 0.038).
    - **Copy-in-replay:** **cuts the copy CE by 0.98 per update** (the PPO epochs by 0.16).
    - **Young-lr:** factor ~3.6 near birth, decaying to 1 over the juvenile period.
  - **(b) Copying in (e) RAN but was INEFFECTIVE (not a strict no-op).** `pop_toy --log-imit` (tiny config, 600
    updates, one newborn copying founders) and `--copy-diag` (the saved u7500 brains of s320 / s321 / s323):
    - **The copy loss is nonzero and falls (44.7 -> 21.1 nats over 600 updates) but stays ABOVE uniform** (~7 nats
      per slot vs log 64 = 4.2).
    - Newborn-vs-elder argmax token agreement on the copied inputs stays at chance: 0.9-2.0% vs 0.7-1.5% for shuffled
      targets, at most a hint (2.0 vs 0.9%) by ages 450-600.
    - On the real u7500 brains, per-slot agreement of every brain with every other on the elder's own sender inputs
      is **0.5-4.6%, at chance** (fresh-navigator chance 1.6-2.7%; 1/64 = 1.6%). This includes the brain copying at
      age 500 and the one that has just finished at age 1,000.
    - **Likely cause: `TAU = 0.08`.** Token logits are VQ distances / 0.08, so every brain's token policy is nearly
      one-hot. A newborn starts confidently wrong, and cross-entropy on the message head moves it only slowly.
    - **Reading for (e):** newborns never came to emit their elders' tokens, so (e) did not test cultural transmission
      with a working copy mechanism. **Proposed verdict wording: "transmission uninformative: the copy pathway was
      ineffective (TAU 0.08)"**, not "transmission failed". (e) stays closed.
    - **Before copying goes into W2:** a copy-pathway unit check (a newborn must reach above-chance agreement with a
      fixed elder within its sensitive period), e.g. a softer temperature for the copy loss, or copying in z_e space.
  - **(c) The predictive head still ignores the message, with coverage fixed.** 64 trained all-on lives (s0
    checkpoint at u124) x 4 true-message episodes: food-in-view-4-ahead MSE 0.0862 true vs 0.0857 with the heard
    sector rotated by 4. **Swap increase -0.5%, paired z -0.9.**
    - Reading (not tested): the target (food within the 7x7 view four steps ahead) may be barely informed by a
      direction to a patch beyond view.

- **RESULT (Nov 10 ~19:15): fast-learner recalibration (free, laptop): NOT CALIBRATED. The positive control fails
  under every probe design; STOP for Cam** (comm-toy `a824ffa`, `fastlearner.py --calib`; `docs/data/fl_calib/`).
  - **Design ported from the frozen learnability test:**
    - a life = one listener's 800-step adaptation in one world;
    - well-heard tokens = >= 25x during adaptation; eligible iff >= 5 well-heard;
    - probes true vs a derangement among the life's well-heard tokens, last-heard credit;
    - pass = drop >= 30% AND paired > 2 SE over the probes; Wilson CIs, stratified 5 / 6 / 7-8.
    - Positive control = a scripted decoder (decodes the life's code perfectly, walks the decoded direction). Negative
      control = the untrained side memory (zero side head = the navigator, message-blind). VoI = scripted true vs
      silence, navigator AND random fallback.
  - **Three probe designs, nearest-food (picked world, spread 6 + walk, family B):**

    | probe design | eligible | positive (nav / random fallback) | negative | VoI nav | VoI random |
    |---|---|---|---|---|---|
    | in place (continue the adapted state, 8 x 300) | 25 / 64 | 0% / 4% | 0% | 1.58x (z 3.4) | 1.06x (z 1.2) |
    | fresh world, new patches (24 x 600) | 54 / 128 | 3.7% / 12.5% | 0% | 2.42x (z 17) | 1.70x (z 15) |
    | fresh world, the life's patches (24 x 600) | 54 / 128 | 0% / 7% | 0% | 2.62x (z 16) | 1.57x (z 11) |

  - **So in this harness the code has clear value** (VoI both forms >= 1.5x with fresh probes) and the negative control
    is clean. But **a perfect decoder does not lose >= 30% of its credited food when its well-heard tokens are
    deranged** (learnability's identical design: 92%).
    - The cause is not found. Candidates: fewer well-heard tokens per life (median 3-4 vs 6-8 in learnability, so
      derangements are often near-adjacent); hearing only from wandering speakers within radius 9 for a few steps
      before the decoder walks out of range.
    - Not tuned further on my own: three designs tried, and more would be forking paths.
  - **New held-out topic, candidate second_nearest_patch_dir (topic 6): NOT viable.**
    - Scripted eligibility is 12-17 / 128: the decoder walks away from the speakers.
    - VoI random fallback 1.49x.
    - No held-out topic is registered yet; the FL track stays parked.

- **RESULT (Nov 10, read ~23:05): DOSE-RESPONSE: NO ARM PASSES; MORE REPLAY PASSES DO NOTHING; J2000 DOES NOT HELP
  PAST AGE ~1,000 (two confounds below)** (`docs/data/dose/`: 13 per-seed JSONs + logs, all exited 0;
  `dose_summary.py`, output `dose_summary.txt`; item spend $3.54 of the $4 cap).

  | arm | eligible (rate) | pass at death | ever | age 250 / 500 / 1,000 (/ 2,000) | grad steps / life | median lifespan |
  |---|---|---|---|---|---|---|
  | all-on R8 (base, learn_arms) | 1,164 (61%) | 7.7% | ~13% | 2.7 / 6.4 / 10.4% | 476 | ~890 |
  | R32 x5 | 1,302 (59%) | 6.5% | 13.4% | 4.8 / 7.0 / 9.8% | 1,820 | ~865 |
  | R128 x3 | 880 (58%) | 6.0% | 13.1% | 3.5 / 5.2 / 9.2% | ~7,100 | ~855 |
  | J2000 x5 | 807 (82%) | 9.7% | 19.7% | 4.5 / 8.6 / 11.4 / 9.3% | ~1,350 | ~2,540 |

  - **Paired age trends** (McNemar on the same lives; per-life snapshots now stored):
    - R32: 250 -> 500 z +3.0; 500 -> 1,000 z 0.0.
    - R128: +1.9; +1.6.
    - **J2000: +3.8; +2.8; then 1,000 -> 2,000 z -1.9** (47 lives lost reliance, 30 gained).
  - **Pass vs cumulative gradient steps (quartile bins; the steps are mostly replay / copy passes), flat in the R arms:** R32 4.8 / 4.5 / 2.4 / 5.0%; R128 3.5 /
    3.5 / 2.4 / 4.5%. In J2000 it rises with steps, but steps track age there.
  - **Registered decision:**
    - any arm >= 50% at death: **none**;
    - **R rises with dose: NO.** Age-1,000 pass: R8 10.4% -> R32 9.8% (z -0.4) -> R128 9.2% (z -0.3), with 4x / 15x
      more gradient steps per life. Not compute-limited;
    - J2000: rises within life to age 1,000, as the base does, then turns down. At age 1,000 it is no different from
      the base (11.4% vs 10.4%, z +0.6). The extra childhood (1,000 -> 2,000+) adds nothing.
    - **Registered branch: "all flat" (with respect to dose).** Flagged: J2000's within-life rise to age 1,000 could
      be read as "J2000 rises", but it is the base's rise and does not continue.
  - **CONFOUNDS (found after the read; defaults confirmed in learnability.py): the dose arms did not vary what they
    were meant to vary.**
    1. **`--mem 4`:** replay re-passes the SAME 4 most recent rollouts (256 steps). R32 / R128 add passes over the
       same small memory, not more distinct experience.
       - Replay trains only the predictive loss and copy-in-replay, and check (c) shows the predictive head ignores
         the message.
       - The "gradient steps per life" (1,820 / 7,196) are mostly these replay passes; the PPO policy updates per life
         were the same in every R arm.
       - **So the finding is "more replay passes over a 4-rollout memory do not help", NOT "in-life compute is not the
         limit".**
    2. **`--young 1000`:** copying elders' actions stops at age 1,000 in EVERY arm, J2000 included (J2000 lengthened
       provisioning and the young-lr decay, not the copy window).
       - **J2000's downturn 1,000 -> 2,000 (paired z -1.9) starts exactly when copying stops.** This is consistent
         with reliance being carried by copying and fading after it ends.
       - J2000 therefore did not cleanly test "a longer childhood".
    - So "the learner is the limit -> fast-learner priority" is **premature**. The fast-learner branch is blocked
      anyway (its positive control does not calibrate).
    - Arms these confounds point to (not launched; Cam's call): **J2000 with `--young 2000`** (copying lasts the whole
      childhood) and **R32 with a larger `--mem`** (replay sees more distinct experience).
    - Batch headroom: $2.83 + $3.54 = **~$6.37 of $8 used, ~$1.63 left.**
  - Note: the guard and item watch died with the session again at ~20:23; all 13 jobs had finished at 20:13 (no
    unwatched spend). Guard restarted 23:03.

- **Nov 11: handoff and proposal written (no new data).**
  - Live handoff: `docs/HANDOFF_2026-11-11.md` (status, established facts with tags, toolbox, ranked work order, traps).
  - Proposal for Cam and M: `docs/NORTH_STAR_V2.md` (what we have been doing wrong; architecture v2; roadmap v2).
    - Main points: no gradient steps inside a life (learning in memory, slow weights across lives); a recurrent
      agent; adaptation before language in the ladder; inborn codes allowed, culture shown by cross-fostering; one
      harness and one probe suite; explore and confirm lanes.
    - Not adopted. Nothing in it overrides a ruling.

- **Cam, Nov 11-12: rulings on NORTH_STAR_V2 (4e73d85) and the open ~23:10 items** (supersedes his 09:27Z paste).
  0. **v2 ADOPTED as the mainline**, with the amendments below. Literature checked by Cam: RL^2 (Duan et al. 2016),
     "Learning to reinforcement learn" (Wang et al. 2016) and AI-GAs (Clune 2019) say what v2 cites them for;
     Bhoopchand et al. 2023 (Nature Communications) as described. Read its ablations before designing stage 1.
  1. **~23:10 items:**
     - dose reruns CANCELLED (they test in-life gradient learning, which v2 drops); keep the $1.63;
     - **toy population verdict: "transmission uninformative (copy pathway ineffective, temperature 0.08)"**
       [the cause in brackets was WITHDRAWN Nov 12: see the copy-pathway unit test below];
     - fast learner un-parked and folded into stage 1 as the main line, in the unified harness (not the old FL
       test);
     - idea A becomes v2's "next meal" head, trained on the slow weights across lives.
  2. **Amendments:**
     - (a) inborn codes only if they arise from training or selection on the world's pressures; claim name stays
       "inborn signalling" until cross-fostering shows culture;
     - (b) the stage 1 gate adds foraging within 10% of the navigators, and both VoI forms >= 1.5x in the unified
       harness;
     - (c) stage 1 has two arms from the start:
       - **listen-only:** associative learning from own meals;
       - **demonstrators:** scripted perfect followers acting on that life's random code, then dropping out, with
         the "where others will be" head;
     - (d) the adaptation yardstick is the measurable version of "surpassing"; M's full goal stays the north star;
       the LLM-agent comparison is M's call, not before stage 2;
     - (e) two lanes: EXPLORE (free or < $1, no bars, labelled) and CONFIRM (registered). Report effects with
       intervals, pooled over lives. Required before every launch: the resolved config, the arm diff, and the
       lever-did-work check;
     - (f) **W2 is the mainline. The toy and the production network are ARCHIVED:** code frozen, no further spend.
       Final status: **production network: rung 2 met Nov 9; toy: closed.**
  3. **Work order:**
     - (i) stage 0, a random code per life, gate perfect >= 90% and deaf <= 10%;
     - (ii) copying repair + unit test: a newborn copying a frozen elder reaches >= 50% agreement with the elder's
       top word per situation within 500 steps, chance reported;
     - (iii) stage 1 recurrent agent (GRU 128-256), slow weights across lives, no in-life gradient; 10-min laptop
       pilot, EXPLORE lane: reliance must rise with tokens heard;
     - (iv) one Modal stage-1 pilot with both arms; price probe first; **cap $6**. Decision tree registered before
       launch: rise -> CONFIRM on held-out code families (needs new credits); no rise by ~1B agent-steps ->
       key-value memory once; still none -> listening must be inborn, report.
  4. Guard: check at every session start and before every spawn. 5. No page needed: M prefers chat.
- **M APPROVED (card, 06:01Z), relayed by Cam: "Yes, all of it".**
  - Slow weights pass down lineages, with mutation and selection between lineages. Culture is shown only by the
    cross-fostering test.
  - **Four permanent inborn drives:** young stay near voices / parent; "where my next meal was" in hindsight; where
    others will be; the next view. No longer scaffolding; allowed in both stage-1 arms.
- **RESULT (Nov 12 ~01:05), EXPLORE lane: STAGE 0 PASSES** (comm-toy `18dfacd`, `learnability.py --code-per-life`;
  `docs/data/stage0/`).
  - **Design:** each life hears the beacons through its own random sector -> token permutation (drawn at birth, own
    RNG stream). The scripted listener decodes with the inverse, so it stays a perfect decoder. Swaps, credit and
    eligibility stay in sector space; coverage counts are mapped back at death. With the flag off the default path
    is identical (8/8 lives, identical report), and the resume test passes.
  - **Smoke** (8 worlds x 100 updates, 24 probes x 600, 59 eligible lives): with and without random codes, EVERY
    number is identical (scripted 83.1% [71.5, 90.5], deaf 0%, VoI 3.56x / 3.19x).
    - Expected: the perfect decoder inverts the code, and the deaf agent ignores messages.
    - So the frozen gate's full-scale result (91.9%, deaf 0%, VoI 3.4x / 3.1x; 197 lives x 48 probes) carries over
      by construction.
  - **The code is live (control with teeth):** a scripted listener that ignores the code passes 60.6% with codes off
    and **6.2% with codes on** (VoI 3.73x -> 1.54x). 4 worlds, 16 probes, ~33 eligible lives each.
  - Deviation: the smoke pair took 12.7 min on the laptop (over the ~10-min rule); later checks were sized down.

- **RESULT (Nov 12 ~01:30), EXPLORE lane: copy-pathway unit test. CORRECTION: the copy pathway WORKS; the toy
  population's "temperature 0.08" explanation is WITHDRAWN** (comm-toy `a0ea34e`, `pop_toy.py --copy-unit`;
  `docs/data/copy_unit/`).
  - **Setup:**
    - newborn = brain 1 (a fresh navigator); frozen elder = brain 0 (verified untouched after training);
    - training situations = 288 elder sender rows from 3 rollouts; held out = 96 rows from a 4th;
    - batch 512, the population's own optimiser and sensitive-period plasticity;
    - chance = fresh navigators' agreement with the same targets.
  - **Noise elder (an untrained navigator; 76 distinct words on 96 held-out rows):** 500 steps give 10% / 10% / 3% per
    slot (ce / soft / embed), chance 1.6%. Nothing to generalise; this is what my Nov 10 check (4b) actually tested.
  - **Functional elder (a fixed 3-slot word per lit fire, as a working founder has; 4 situations held out):**

    | loss | per-slot agreement, step 500 / 1,000 / 1,500 | whole word, step 500 / 1,000 / 1,500 |
    |---|---|---|
    | ce, TAU 0.08 (the population's own loss) | **66%** / 85% / 85% | 40% / **59%** / 58% |
    | embedding match | 55% / 80% / 87% | 17% / 50% / 65% |
    | soft temperature (1.0) | 39% / - / - | 12% / - / - |

    Chance: 0.4% per slot, 0% whole word. Every loss is flat for ~200 steps, then rises.
  - **Cam's bar (>= 50% agreement with the elder's top word within 500 steps):** met per slot by the unchanged ce loss
    (66%); the whole 3-slot word needs ~800-1,000 steps (59% by 1,000, within the population's 1,000-update
    sensitive period). One seed; no variance estimate.
  - **Correction on the record:** check 4b (Nov 10) attributed the failed transmission to TAU 0.08. That check copied
    a noise elder, which nothing can generalise, so it did not test the pathway. With a functional elder the same
    loss works. "Copy pathway ineffective (temperature 0.08)" is **WITHDRAWN** as the cause.
    - Remaining hypotheses, untested: each gen-2 newborn copied THREE founders whose dictionaries agreed on only
      25-50% of fires (conflicting targets), and the copy step competes with PPO updates.
    - **Proposed verdict wording, for Cam: "transmission uninformative; cause unknown; the copy pathway works on a
      single functional elder; conflicting founder dictionaries is the leading untested hypothesis".**
    - **Scope of the unit test:** one seed, 4 situations, 288 training rows reused across 500-1,500 steps, in the
      toy's channel only.
  - **No copy repair is needed for W2.** Keep the unit test (functional elder, >= 50% per slot by 500 steps; whole
    word reported) as a standing check.

- **RESULT (Nov 12 ~01:15), EXPLORE lane: stage-1 supervised in-context probe, laptop pilot. WEAK SIGNAL, UNDERPOWERED**
  (comm-toy `f608dd3`, `learnability.py --ic-probe`; `docs/data/icp/`).
  - **Question:** with a random code per life, can a GRU trained ACROSS lives (direct supervision; a diagnostic,
    never a drive) work out a life's code from that life's own stream?
    - Inputs: view/body obs, the heard token, the previous action, the reward.
    - Target: the true sector behind each heard token. Navigator behaviour, picked world, spread 6 + walk.
    - Baseline: a memoryless MLP on the same inputs.
  - **Prediction (written before running): P = 0.35** that the GRU reaches >= 10 points above chance by 20+ tokens
    heard.
  - **Listen-only** (216-220 training / 54 test lives; 28k heard test steps):
    - GRU 21.2%, memoryless 19.1%; chance 12.5% uniform, 17.1% majority;
    - by tokens heard (1-5 / 6-15 / 16-40 / 41-100 / 101+): GRU 11.9 / 13.7 / 16.7 / 19.8 / 21.9%; memoryless 7.4 /
      10.4 / 11.8 / 14.9 / 20.4%;
    - the GRU's edge is 1.5-5 points and does not grow with tokens heard. **Prediction not met** (+4 points at 16-40
      tokens).
  - **With demonstrators** (`--demonstrators`: the scripted elders, perfect followers, without a copy loss): GRU
    15.3% vs memoryless 13.7%, majority 17.4%. No gain.
  - **Caveats, why this is not yet a negative:**
    - only ~216 training lives, and the GRU overfits (training loss ~1.2-1.3 against 15-21% test accuracy);
    - how often a demonstrator is inside the learner's 7x7 view is NOT measured;
    - the memoryless baseline already beats uniform chance: the current step partly reveals the sector (the view,
      and a life's sector changes slowly).
  - Reading so far: under navigator behaviour the stream carries little usable evidence of a life's code.
- **REGISTERED (EXPLORE lane, < $1, no bars): plan `icp`** on Modal `throng-w2jobs` (comm-toy `f608dd3`):
  - the same probe at ~10x the lives: 64 worlds x 150 updates, 2,000 training steps; two jobs, listen-only and
    demonstrators;
  - resolved config = `_ICP` in `scripts/modal_w2jobs.py`; arm diff = `--demonstrators` only;
  - est. ~$0.15-0.2; item watch stops throng-w2jobs at $1.25.
  - **Prediction:** with ~10x the lives the GRU's edge over memoryless grows to 3-8 points but still does not grow
    with tokens heard (P = 0.6); the demonstrator arm beats listen-only by >= 5 points at 41+ tokens (P = 0.3).

- **RESULT (Nov 12 ~01:35), EXPLORE lane: `icp` in-context probe at scale** (Modal, both exited 0, **$0.15**;
  `docs/data/icp/icp_listen_f608dd3.json`, `icp_demo_f608dd3.json`).
  - ~1,480 training / ~370 test lives per arm; ~200k heard test steps. Chance 12.5% uniform, 14-15% majority.

    | arm | GRU | memoryless | GRU - memoryless, by tokens heard 1-5 / 6-15 / 16-40 / 41-100 / 101+ |
    |---|---|---|---|
    | listen-only | 21.9% | 20.0% | +0.6 / +1.5 / +1.4 / +1.6 / +2.1 points |
    | demonstrators | 21.9% | 18.7% | -4.7 / -3.4 / +1.0 / +2.7 / +3.6 points |

  - **Listen-only:** under navigator behaviour a life's stream carries almost no usable evidence of its code. Memory
    adds ~2 points with direct supervision.
  - **Demonstrators:** memory's advantage GROWS steadily with tokens heard (-4.7 -> +3.6 points). That is the signature
    of in-life inference, but small. The early deficit suggests the GRU discounts the current step until it has
    seen some demonstrations.
  - **Prediction scoring:**
    - "the edge grows to 3-8 points but not with tokens heard" (P 0.6): listen-only MISSED (edge ~2, flat);
      demonstrators MISSED the other way (it grows with tokens);
    - "demonstrators beat listen-only by >= 5 points at 41+ tokens" (P 0.3): MISSED (GRU +1.6 at 41-100, -0.2 at
      101+).
  - **Caveats:**
    - the GRU still overfits (training loss 1.2-1.5);
    - demonstrator visibility is unmeasured;
    - behaviour is the PASSIVE navigator. An agent that acts on what it hears creates its own evidence (follow a
      token, find food or not), which a passive probe cannot see. So this bounds what passive observation alone
      offers; it does not bound an active learner.
  - **Implications for stage 1:**
    - (a) the demonstrator arm is the more promising one, as Cam's amendment (c) expected;
    - (b) the listen-only arm will have to rely on its own exploration of tokens; the hindsight next-meal head is the
      drive that turns that into a learning signal;
    - (c) the effect sizes are small, so stage-1 pilots need many lives (thousands) to read.

- **Cam, Nov 12 (rulings on the ~01:20 and ~01:40 reports).**
  1. **Stage 0: PASS, accepted.**
  2. **Copying: accepted as working.** Record both readings of "top word": per slot passes at 500 (66%); the whole
     word reaches 59% only by 1,000 (40% at 500); chance 0.4%.
     - **(e) verdict ADOPTED: "transmission uninformative; cause unknown; copy pathway works on a single functional
       elder; conflicting founder dictionaries is the leading untested hypothesis".** The "temperature 0.08"
       wording is withdrawn. Do not test the hypothesis in the archived toy.
     - **Design note for stages 2-3:** newborns copy mainly from ONE elder (the parent), which the approved
       stay-near-parent drive supports.
  3. **In-context probe read accepted.** Extra caveat: 1,480 training lives is tiny for meta-learning an inference
     algorithm (RL^2-style results use 1e5+ episodes). The probe mixes "is the evidence in the stream" with "can a GRU
     learn to use it from 1,480 examples". Separate them first.
  4. **Next, free (laptop, < 10 min, EXPLORE): an IDEAL-OBSERVER counting decoder, no training.**
     - Per token in a life, accumulate evidence directions:
       - (a) food visible in view at hearing time;
       - (b) direction of the next meal from where the token was heard (hindsight);
       - (c) demonstrators' movement direction after hearing (demonstrator arm).
     - Decode each token as the argmax sector; accuracy vs tokens heard, per arm, on the probe's lives. Also report
       demonstrator visibility (fraction of tokens heard with a demonstrator in view).
     - **Decision:**
       - counting decoder >= 80% within a median life in an arm: the evidence is there and the GRU's weakness is
         meta-training data. Build the stage-1 agent for that arm first, sized for >= 1e5 training lives;
       - neither arm gets there: the world is evidence-poor; report, and Cam chooses between raising the evidence
         and moving to W2.1 parent calls.
  5. **Stage-1 agent after (4).** 1e5+ lives x ~1,000 steps = 1e8-1e9 agent-steps. If the learnability harness can't
     reach ~11k steps/s/container, port the frozen test's scoring onto the JAX W2 env and verify it reproduces b61afb2
     (perfect >= 90%, deaf <= 10%, within CI) before any pilot. Do not shrink the run to fit a slow harness. Then the
     $6 pilot with the registered decision tree.
  6. Container noted (scaled down by itself). Spend $28.45: fine.

- **RESULT (Nov 12 ~02:00), EXPLORE lane: IDEAL-OBSERVER counting decoder. THE WORLD IS EVIDENCE-POOR: no arm comes
  near 80%** (comm-toy `3933d1e`, `learnability.py --ic-probe --io-decoder`; `docs/data/ideal_observer/`; laptop,
  16 worlds x 80 updates, the laptop probe's lives).
  - **Method (no training):** for each token a life hears, votes for sectors from (a) the direction of the best
    visible food when it is heard, (b) the displacement from where it was heard to the life's next meal (available
    only after that meal), (c) the next move of any demonstrator within view (Chebyshev <= 3). Online: each heard
    token is decoded as the argmax of the votes available BEFORE that moment, and scored against the true sector.
  - **Decoder control:** an oracle source (the true sector, after the fact) reaches **98.7-99.4%**. The bookkeeping
    is correct.
  - **Results, accuracy over the second half of a median life (median 490-650 tokens heard); chance 12.5%:**

    | behaviour | (a) visible food | (b) hindsight meal | a+b | (c) demonstrators | a+b+c |
    |---|---|---|---|---|---|
    | navigator, listen-only (274 lives) | 13.1% | 14.9% | 15.6% | - | - |
    | navigator + demonstrators (270 lives) | 13.4% | 15.4% | 15.8% | 17.2% | **18.4%** |
    | scripted listener, control (184 lives) | 17.5% | 25.2% | 25.7% | - | - |

  - **A demonstrator is in view for only 4.0% of the tokens heard.**
  - **Even a listener that follows the code perfectly gets weak hindsight evidence** (25%): its next meal is often
    not at the patch the beacon meant.
  - **Decision (Cam's registered rule): neither arm reaches 80% within a median life -> the world is evidence-poor;
    no learner will fix that. Report; Cam chooses** between raising the evidence (demonstrators kept near the newborn,
    or more of them) and W2.1 parent calls.
  - **Caveats:**
    - this is a simple counting observer, not a provably optimal one. It ignores, e.g., the beacon's offset and
      agents seen in the occupancy view, so it is not a strict upper bound;
    - accuracy is measured on the beacon's sector, which the listener's own experience reflects only loosely because
      beacons wander and the listener is up to 9 cells away.
  - **Reading:** the beacon code describes "the direction from the BEACON to ITS nearest patch". A listener's own
    experience barely constrains that. A parent calling about food the young can then reach (W2.1), or demonstrators
    that stay beside the young, would make the evidence immediate.

- **Cam, Nov 12 (ruling on the ~02:00 ideal-observer report).**
  1. **Result accepted:** evidence-poor branch.
  2. **One free check first (EXPLORE):** a BEACON-relative observer. Cues (a) and (b) measured from the beacon's
     position using the speaker offset (a v2 agent input): the direction from the beacon to the food seen, and to the
     next meal. Reported next to the current numbers. It changes the choice only if it reaches >= 80% on its own in
     the listen-only arm.
  3. **CHOICE: option (1), with nearness rather than numbers.**
     - One "parent" demonstrator per newborn stays beside it (in view, <= 3 cells) through the juvenile period. It
       hears the same tokens and acts on them. Young-seek targets the parent, not the nearest beacon (the approved
       stay-near-parent drive).
     - The parent leaves at the end of the juvenile period (expert dropout, as in Bhoopchand).
     - The life test runs with NO demonstrators present, so following the parent cannot pass it.
     - **Re-run the counting decoder in this arm first. Decision:** >= 80% within a median life (any cue combination,
       reported at ages 250 / 500 and at parent dropout) -> build the stage-1 agent for this arm (>= 1e5 training
       lives, throughput check / JAX port as ruled). Below 80% -> report; Cam decides.
  4. W2.1 parent calls stay at stage 2 (learned speakers). Stage 1 keeps scripted speakers as the instrument.
  5. Budget unchanged: all free. The $6 pilot cap stands for the stage-1 agent.

- **RESULT (Nov 12 ~03:00), EXPLORE lane: beacon-relative check + PARENT ARM counting decoder** (comm-toy `7e72ae7`,
  `learnability.py --parent`, decoder with beacon-relative cues and age windows; `docs/data/ideal_observer/io3_*.json`;
  laptop, 16 worlds x 80 updates).
  - **Implementation of the parent arm:**
    - one scripted parent per newborn (elder l pairs with learner l), starting beside it; it steps back whenever more
      than 2 cells away;
    - the young seek THEIR parent when it is more than 3 cells away (while the parent is present);
    - at the end of the juvenile period (age 500) the parent leaves (moved half a world away);
    - the life test has no demonstrators (elders are not alive in test episodes);
    - default path verified identical (8/8 lives).
  - **Beacon-relative check (listen-only, 274 lives):** cues measured from the beacon's position via the speaker
    offset.
    - Visible food: exact 31.3% overall, 47.3% after age 500. Next meal: 23.0%. Both: 27.0% (listener-relative:
      12.5-15.5%).
    - **Well below 80%, so the choice is unchanged**, as ruled.
  - **Parent arm (271 lives; median 286 tokens heard per life; parent in view for 58.5% of tokens heard, vs 4% with
    roaming demonstrators).** Accuracy at age 200-250 / 450-500 (= at parent dropout) / after dropout:

    | evidence | exact sector (chance 12.5%) | same move as the truth (chance 31.25%) |
    |---|---|---|
    | parent's moves (c) | 40.4 / 45.0 / 34.4% | 77.8 / **81.8** / 68.7% |
    | beacon cues (A+B) + parent (c) | 43.7 / 49.0 / 51.7% | **80.4 / 83.7** / 75.6% |
    | beacon cues (A+B) alone | 18.7 / 16.9 / 43.1% | 42.9 / 38.0 / 57.3% |

    Over a whole median life, A+B+c reaches 80.8% on the move score and 44.3% exact.
  - **Measurement change, introduced AFTER seeing the exact-sector numbers (flagged):** the "same move" score counts
    a decode as correct when it implies the same move as the true sector.
    - Why: the scripted parent walks diagonal sectors as straight moves (sector 1 like 0, 3 and 5 like 4, 7 like 0),
      so its moves cannot separate them.
    - What a listener needs is which way to walk, but its chance level is 31.25%, not 12.5%.
  - **Decision against Cam's rule (>= 80% within a median life, any cue combination):**
    - exact sector: **NOT met** (max 51.7%);
    - same move: **met** at ages 250 and 500 (80.4%, 83.7%) and over the median life (80.8%), but not after dropout
      (75.6%).
    - **Which score the rule means is Cam's call.**
  - **Reading:**
    - nearness works: putting one parent beside the newborn raises visibility 15x and turns demonstrations into the
      strongest cue;
    - the parent's actions carry only 4 directions of the 8-sector code. If the agent must decode exact sectors,
      the parent cannot teach the diagonals;
    - median tokens heard per life halve in the parent arm (286 vs 628): following the parent takes the young out of
      earshot of the beacons more often.

- **Cam, Nov 12 (ruling on the ~03:00 parent-arm report).**
  1. **Which 80%: Cam meant exact sector.** A score added after the data cannot pass a bar on its own. **Recorded:
     parent-arm counting decoder, exact sector <= 52% (FAILS); same move 80-84% at ages 250-500 and 76% after
     dropout (POST HOC, flagged).** Settle it with the registered behavioural test instead.
  2. **Plug-in check (free, EXPLORE):** a "counting-observer listener" that acts on its CURRENT decoded mapping
     (argmax of evidence so far, beacon cues + parent) with the same sector -> move rule as the scripted perfect
     listener, run through the frozen life test (b61afb2) on parent-arm lives.
     - Confirm whether the scripted perfect listener uses the same sector -> move rule as the parent.
     - Report eligibility in the parent arm with the sector-count strata.
  3. **Decision tree:**
     - observer-listener passes >= 50% of eligible lives: build the stage-1 agent for the parent arm (>= 1e5 training
       lives, throughput check / JAX port first), then the $6 pilot;
     - below 50%: rerun the plug-in with the parent leaving at 1,000 instead of 500 (free); if that passes, build with
       it; if not, report;
     - fewer than 50 eligible lives in a check: add lives before reading it.
  4. Budget unchanged: all free; spend ~$28.4.

- **Plug-in check, design (comm-toy `f5783c6`; EXPLORE, < $1). Confirmed in code:** the scripted perfect listener and
  the parent use the same `scripted()` sector -> move rule (diagonals walked as straight moves). The life test
  therefore sees only the move class, and this check is the fair arbiter.
  - **`--obs-listener observer`:**
    - lives run with navigator behaviour plus the parent (leaves at 500);
    - at each life's end, the token -> sector table = the argmax of that life's counting-decoder evidence (beacon cues
      A+B + parent moves c; ties / no evidence -> random);
    - the frozen life test (heard25, last-heard, >= 5 sectors >= 25x, 48 x 600) then runs the scripted listener
      acting through that table.
  - **`--obs-listener true`:** the same lives, the table = the life's real inverse code. This is the perfect-decoder
    control ON PARENT-ARM LIVES (fewer tokens heard, so it may fall short of the frozen gate's 91.9%).
  - The deaf control is the navigator on the same lives. `--eval-eligible-only` runs the test only on eligible lives;
    the eligibility rate is recorded before the filter. A smoke run: ~26% of parent-arm lives eligible.
  - Plan `obsl`: 32 worlds x 150 updates, both modes; est. ~$0.25; item watch at $1.25. Plan `obsl1000` (parent
    leaves at 1,000) is held for the "below 50%" branch.
  - **Decision (Cam's tree):** observer >= 50% of eligible lives -> build the stage-1 agent for the parent arm; below
    -> run `obsl1000`; still below -> report.
  - **Prediction:** observer >= 50% with P ~0.45; true-code control >= 90% with P ~0.5.

- **RESULT (Nov 12 ~02:55), EXPLORE lane: plug-in check. THE COUNTING-OBSERVER LISTENER PASSES (54.2% >= 50%)**
  (Modal, both exited 0, $0.24; `docs/data/obsl/obsl_observer_f5783c6.json`, `obsl_true_f5783c6.json`; parent arm,
  parent leaves at 500, frozen test 48 x 600).
  - **Lives:** 791 parent-arm lives, **212 eligible (26.8%)**, scored. Median lifespan 2,028.

    | listener in the frozen test | pass, eligible lives (95% CI) | strata 5 / 6 / 7-8 sectors | deaf control | VoI nav / random |
    |---|---|---|---|---|
    | **counting observer** (life's decoded table) | **54.2%** (47.5-60.8) | 42.0 / 38.5 / 77.2% (n 81 / 52 / 79) | 0% | 2.37x / 2.20x |
    | true code (perfect decoder, same lives) | 87.3% (82.1-91.1) | 76.5 / 90.4 / 96.2% | 0% | 3.08x / 2.95x |

  - **Decision (Cam's registered tree): observer >= 50% of eligible lives -> BUILD the stage-1 agent for the parent arm**
    (>= 1e5 training lives, throughput check / JAX port first), then the $6 pilot.
    - Caveat: the observer's CI lower bound (47.5%) is below 50%; the bar is on the point estimate, as before.
  - **Prediction scoring:** observer >= 50% (P 0.45): HIT. True-code control >= 90% (P 0.5): MISSED (87.3%;
    parent-arm lives hear fewer sectors, and 5-sector lives pass 76.5%).
  - **Reading:** the evidence in a parent-arm life IS enough for an agent that uses it near-optimally to rely on the
    code in the frozen test, without demonstrators in the test. The observer reaches 62% of the perfect decoder's
    pass rate.
  - **Throughput note for the next step:** the per-learner learnability harness runs at roughly 2-4k agent-steps/s
    (all-on, laptop), well short of the ~11k/s/container that 1e8-1e9 agent-steps within $6 needs. Per Cam's ruling:
    port the frozen test's scoring onto the JAX W2 env and verify it reproduces b61afb2 (perfect >= 90%, deaf
    <= 10%, within CI) before any pilot.
    **[CORRECTED Nov 12 ~03:35, see below: the 2-4k figure was the all-on TRAINING path (per-learner nets + PPO +
    replay), not the harness's rollout speed.]**

- **Cam, Nov 12 (ruling on the ~02:55 plug-in report).**
  1. **PASS accepted** (observer 54.2%, CI 47.5-60.8, on the point estimate; true-code 87.3%; deaf 0%).
     **CI caveat recorded: the observer's lower bound (47.5%) is below the 50% bar.** **The parent arm is the stage-1
     arm.**
  2. **Headroom (EXPLORE, ~$0.25, same watch):** the same plug-in with the parent leaving at 1,000 (plan `obsl1000`).
     **PRE-REGISTERED PICK:** use parent-leave 1,000 for stage 1 iff its observer pass is >= 5 points above 54.2% (>=
     59.2%) with >= 150 eligible lives; otherwise keep 500. An apparatus choice made before any learner runs.
  3. **JAX port: approved as the main job.** Equivalence gate before any pilot, within CI on the JAX env:
     - (a) the original b61afb2 config: perfect 91.9% (87.2-94.9), deaf <= 10%;
     - (b) the parent arm: perfect 87.3%, observer 54.2%, deaf 0%.
     - If (a) or (b) misses, find the difference before going on. Do not retune the test.
  4. **Stage-1 agent (after the port):** parent arm; GRU 128-256; slow weights across lives; no in-life gradient; the
     4 approved drives; a random code per life; training code families only.
     - Report the learner's pass against BOTH reference points on the same config: the counting observer and the
       true-code control.
     - $6 pilot with the registered tree: reliance rises with tokens heard -> CONFIRM on held-out families (new
       credits); no rise by ~1B agent-steps -> key-value memory once; still none -> report.
  5. Spend $28.68 fine; ~$12.3 left under the guard.

- **CORRECTION + MEASUREMENT (Nov 12 ~03:35), laptop: the learnability harness's ROLLOUT throughput is far above
  ~11k/s.**
  - Parent arm, code per life, no training, timed as the difference between 3- and 8-update runs: **W=16: ~9.0k
    learner-steps/s; W=64: ~53k learner-steps/s** (~106k counting the parents). The W=256 reading was invalid
    (timing noise gave a negative difference).
  - My "2-4k agent-steps/s" note above was the all-on TRAINING path (64 separate networks, per-learner PPO, replay 8),
    not the environment or rollout speed.
  - **Implication (flagged for Cam):**
    - the learnability harness IS a JAX W2 env harness (`w2/env.py`), and the frozen test already lives in it;
    - the stage-1 agent (ONE shared network, GRU) could be built here, with training throughput measured once it
      exists, so the port and its equivalence gate may be unnecessary;
    - the port is NOT started pending Cam's answer.

- **Cam, Nov 12 (ruling on the ~03:35 correction).** **PORT CANCELLED; the recommendation is adopted.**
  - Build stage 1 in `learnability.py` behind `--agent recurrent`: one shared GRU network, parent arm, the 4 approved
    drives, a random code per life, training code families only.
  - The default path must stay identical (8/8 check), and the frozen test path byte-identical to b61afb2, so the
    equivalence gate falls away with the port.
  - **10-min laptop pilot: measure TRAINING throughput** (agent-steps/s including updates). Port only if it is
    < ~11k/s/container at the pilot's settings.
  - Everything else stands: the obsl1000 pick rule, both reference points in the learner report, the $6 pilot with its
    registered tree.

- **BUILT (Nov 12 ~04:00): stage-1 recurrent agent, `learnability.py --agent recurrent`** (comm-toy `1baefb9`; not
  yet pushed at the time of writing, see below).
  - **Network (`RecAgent`):**
    - the navigator trunk (navigator 0's weights, FROZEN);
    - a GRU memory (128) reading the heard token, the trunk's view features, the previous action and reward;
    - a zero-initialised head adding to the policy and value, so behaviour starts as the navigator's.
  - **Drives:**
    - three as auxiliary heads: the direction of the next meal (hindsight label within the chunk), the parent's next
      move (masked unless the parent is in view), the next view;
    - the fourth (stay near the parent) is the existing reflex.
  - **Training:** one shared weight set; PPO, 2 epochs; backprop through each 64-step chunk from its stored starting
    memory; memory reset at birth. Per-life codes from training family A (`code_families` parity split). Parent arm.
  - **Evaluation:** after training, `--eval-updates` with FROZEN weights; only lives BORN in that phase are tested. The
    life test uses the same probes, derangements (rng 4242), keys (7000+i) and last-heard credit as the frozen
    heard25 test, with each life's final memory carried into every probe, plus a memory-zeroed control. Pass over
    eligible lives with Wilson CI and strata.
  - **Checks:** default path identical (8/8 lives, identical report); a smoke run end to end.
- **RESULT (Nov 12 ~04:05), laptop pilot: stage-1 TRAINING throughput ~15.7k agent-steps/s** (rollout + 2 PPO/BPTT
  epochs, steady state, compile excluded; W=16: 0.26 s/update, W=64: 1.05 s/update; `docs/data/stage1/`).
  - The laptop used ~5-6 cores; the decision is on a 4-CPU container. Plan `s1tp` (a ~$0.01 price probe, W=64)
    measures it.
  - **Not yet launched:** the network dropped (GitHub and Modal unreachable), and a background retry pushes, deploys and
    spawns when the network returns.
  - Loss notes: the parent-move loss reads 0 once all children are past age 500 (no labels: the parents have left),
    not leakage. Meal and view losses fall.
  - **Risk flagged for the pilot:** the navigator-trunk agent eats little in a 600-step test window (homeostatic: it
    eats when hungry), so credited-food reliance may be hard to show even when the memory carries the code.

- **RESULT (Nov 12 ~04:05), EXPLORE: `obsl1000` headroom check -> PRE-REGISTERED PICK: PARENT LEAVES AT 1,000 for
  stage 1** (`docs/data/obsl/obsl1000_*_f5783c6.json`; ~$0.23).
  - 877 lives, **187 eligible**.
  - **Counting observer 70.6% (CI 63.7-76.7)**; strata 5 / 6 / 7-8: 57.8 / 66.7 / 92.9%.
  - True-code control 87.7% (82.2-91.7); deaf 0%.
  - Rule: >= 59.2% with >= 150 eligible -> met (+16.4 points over leave-at-500).
- **Cam, Nov 12 (ruling on the ~04:00 stage-1 build report).**
  1. Build accepted. Input fixes before the pilot:
     - (a) the SPEAKER OFFSET (dx, dy) of each heard token;
     - (b) own age (and energy, if the trunk doesn't already feed it to the GRU).
  2. **Appetite: test HUNGRY.** Every probe episode starts at a fixed low energy (e.g. 0.3) with drain on, identical
     for every arm (learner, memory-wiped, scripted perfect, deaf).
     - Verify the scripted controls reproduce their frozen numbers under the hunger start (report before going on if
       not).
     - Natural-energy test as secondary. Scoring byte-identical to b61afb2.
  3. **Second readout, independent of behaviour:** a linear probe on the frozen GRU hidden state predicting the current
     token's sector (and move class) on held-out lives, vs tokens heard. The same probe on the memory-wiped control and
     on shuffled-code lives. "The memory knows the code" and "the agent acts on it" are separate claims; report both.
  4. **Pre-launch checks:** print the resolved config and the arm diff; lever-did-work for each head (loss nonzero and
     falling). Stagger births so >= 25% of lives are under 500 at any time (the parent-move head needs juveniles).
  5. **Pilot:** after the container probe; $6 cap; --ckpt on. Readouts (linear probe + hungry life test, memory vs
     wiped) at 100M, 300M and 1B agent-steps. Registered tree: reliance or the probe rises with tokens heard ->
     CONFIRM on held-out families; no rise by ~1B -> key-value memory once; still none -> report. If the container
     is < ~11k/s, size the pilot to what $6 buys and say how many steps that is. Don't port.
  6. The network retry may launch ONLY the throughput probe.

- **RESULT (Nov 12 ~04:20): stage-1 v2 built and checked; CONTAINER THROUGHPUT ~25.9k agent-steps/s -> NO PORT**
  (comm-toy `8a0d027`).
  - **Changes per Cam:**
    - (a) the speaker offset was ALREADY an input: `hear()` returns one-hot token + flag + offset / 9, and the
      memory's side input is x[100:];
    - (b) own energy and age now feed the memory directly (x[98:100]);
    - hungry test start `--test-energy` (the learner's energy at probe start; default 0 = unchanged; stage-1 readouts
      use 0.3), with the natural start as secondary;
    - milestone readouts (`--readout-at`) on a fresh frozen-weight world batch: the life test hungry / natural x
      memory carried / wiped, plus a linear probe on the memory state (sector and move class) vs the memory-wiped
      state and shuffled-code labels, by tokens heard;
    - staggered starting ages (0-2,500): juveniles 44% of lives in a smoke run, above >= 25%;
    - the resolved config is printed at launch; per-head losses and label counts are logged every update.
  - **Checks:**
    - default path identical (8/8 lives, identical report);
    - **stage-1 resume bit-identical** (`tests/test_stage1_ckpt.py`: 40 updates vs killed after 25 and resumed). A
      mutant that skips restoring the RNG diverges; the first 8-update version of this test had no teeth (no deaths
      after resume) and was replaced.
  - **Throughput:** container (4 CPU, W=64, plan `s1tp`, ~$0.01): **median 0.631 s/update = ~25.9k agent-steps/s**
    at steady state (compile excluded; the summary's 14.3k includes the 4.75 s first-update compile). Laptop ~15.7k.
    **> 11k, so no port.** $6 at ~$0.24/h buys ~2e9 training agent-steps before readout costs.
- **REGISTERED (EXPLORE, ~$0.2): `hunger_ctl`**, the scripted controls under the hungry start (`--test-energy 0.3`),
  which must reproduce their frozen numbers within CI:
  - `hunger_gate` (the b61afb2 gate config): perfect 91.9% (87.2-94.9), deaf <= 10%;
  - `hunger_parent_true` (parent arm, leave 1,000, true code): 87.7% (82.2-91.7), deaf 0%.
  - If either misses, report before going on (Cam).

- **REGISTERED (before data): STAGE-1 PILOT `s1pilot`** (Modal `throng-w2jobs`, new function `run_w2_long`: 24 h
  timeout, 4 CPU, 8 GB; comm-toy `8a0d027`). **Launches only after `hunger_ctl` passes.**
  - **Resolved config:** parent arm, parent leaves at 1,000; random code per life from training family A; RecAgent with
    hidden 128 (navigator-0 trunk frozen; GRU memory reading heard token + speaker offset, trunk features, previous
    action and reward, own energy and age; 3 auxiliary drives, aux weight 0.5) + the stay-near-parent reflex; PPO, 2
    epochs, lr 3e-4; 64 worlds x 64 steps x 4 learners = 16,384 agent-steps per update; **61,100 updates ~ 1.0e9
    agent-steps**.
  - **Readouts at 1e8, 3e8, 1e9 agent-steps:** 64 fresh worlds x 60 frozen-weight updates; lives born there:
    - life test with the frozen scoring (heard25, last-heard, >= 5 sectors >= 25x, 48 x 600), **hungry start 0.3
      primary**, natural secondary; memory carried vs wiped;
    - linear probe on the memory state (sector, move class) vs the memory-wiped state and shuffled-code labels, by
      tokens heard;
    - checkpoint every 500 updates.
  - **Reference points for the report (same config):** counting observer 70.6% and true-code control 87.7% (obsl1000,
    natural start), plus the hungry-start control from `hunger_ctl`.
  - **Price:** ~10.7 h of training at the measured 0.63 s/update plus 3 readouts: est. **$3-4** at ~$0.24/h. Item cap
    $6; the item watch stops the app at $7.5.
  - **Decision tree (Cam, registered):**
    - reliance OR the linear probe rises with tokens heard -> CONFIRM run on held-out code families (new account
      credits);
    - no rise by ~1e9 agent-steps -> key-value memory, once;
    - still none -> report.
  - **"Rises" is operationalised before data as** either:
    - the hungry memory-carried pass rate in lives with > 600 tokens heard exceeding that in lives with <= 300 by a
      two-proportion z > 2; or
    - linear-probe sector accuracy (memory) rising across the token bins 1-5 -> 101-300 by >= 10 points AND above the
      memory-wiped probe in the 101-300 bin.
  - Lever-did-work: every head's loss nonzero and falling between the first and the last 100 updates, with labels
    present (juveniles >= 25%).

- **RESULT (Nov 12 ~04:30), EXPLORE: `hunger_ctl`. THE SCRIPTED CONTROLS REPRODUCE UNDER THE HUNGRY START**
  (`docs/data/hunger_ctl/`; ~$0.2).
  - **Gate config:** perfect 94.4% (90.3-96.9) vs frozen 91.9% (87.2-94.9); deaf 0%; 197 eligible; strata 89.1 / 92.9 /
    98.0%.
  - **Parent arm, leave 1,000, true code:** 88.2% (82.8-92.1) vs 87.7% (82.2-91.7); deaf 0%; 187 eligible.
  - CIs overlap: the hungry start does not distort the instrument. **The pilot's pre-launch gate is cleared.**

- **LAUNCHED (Nov 12 04:31): stage-1 pilot `s1pilot`** (call fc-01M77HHYN7HQ6DTSP8PF1163KS; log
  `testbed/w2jobs/s1pilot_8a0d027.log`; checkpoint `testbed/w2jobs/s1pilot.ckpt`; results
  `testbed/w2jobs/s1pilot_8a0d027.json`, rewritten at each readout).
  - Startup check (updates 90-110): 0.72-0.84 s/update (est. ~13 h for 1e9 + readouts, ~$3.5-4).
  - Labels present: meal ~750-900, other ~10k per update; juveniles ~63%.
  - Losses: meal falling 1.01 -> 0.80, view 0.011 -> 0.0097, other ~1.39 (flat so far), PPO small.
  - Watches: guard (cap $41); item watch (throng-w2jobs, cap $6, stops at $7.5).
  - Readout ETAs: 1e8 ~06:00, 3e8 ~09:00, 1e9 ~18:30 (Nov 12, EST).

- **Cam, Nov 12 (ack on the 04:31 launch): SANITY STOP added at the 100M readout** (no change to the registered tree).
  - Stop the run, save the rest of the $6 and report, if either:
    - the linear probe on shuffled-code lives is above chance;
    - the memory-wiped life test differs from the navigator's.
  - Otherwise let it run to 1e9 untouched. Send each readout (probe accuracy vs tokens heard, memory kept vs wiped,
    hungry life-test pass with CI).
  - **Operationalised before the readout (Will):**
    - (i) shuffled-code sector probe: STOP if it is ABOVE chance, i.e. the lower bound of its 95% Wilson CI (on its
      held-out rows) exceeds 12.5% (below-chance values are not leakage);
    - (ii) the navigator's pass rate on this life test is 0% (the deaf control in every frozen-test run): STOP if the
      memory-wiped HUNGRY pass rate exceeds the 10% negative-control bar.

- **Pilot note (Nov 12 ~06:05): one preemption, recovered by the checkpoint.**
  - Modal interrupted the container at ~update 5,220 (KeyboardInterrupt in the log), restarted it at 10:37 UTC, and the
    run resumed from the update-5,000 checkpoint ("resumed stage 1 at update 5000, 81,920,000 agent-steps"); ~220 updates
    were lost.
  - **Watch item before the 1e8 readout:** the training juvenile fraction rose from ~63% to 80-93%, and eaten per step
    fell from ~0.0025 (update 100) to ~0.0003-0.0007 (updates 5,000-6,100). This suggests shorter lives and worse
    foraging than the navigator, which bears on Cam's stage-1 gate condition (foraging within 10% of the navigators).
    To be checked against the readout's lifespans.

- **RESULT (Nov 12 ~06:30), free prep: stage 2 / stage 3 designs drafted** (`docs/STAGE2_STAGE3_DESIGN.md`), and the
  **cross-fostering instrument built and run on the laptop** (comm-toy `131f579`, `--crossfoster`).
  - At 16 probes: perfect foster-learner 70.7% foster / 0% birth; perfect birth-decoder 0% / 69.3%; deaf 0% / 0%.
  - Clean and symmetric, but NOT yet >= 90%. The probe count is the likely cause (the frozen test uses 48). The
    48-probe calibration needs Modal (~$0.1); asking Cam.

- **Cam, Nov 12 (rulings on the ~06:30 report).**
  1. **FORAGING RULE at the 100M readout** (on top of the sanity stop). Compare the agent with the navigator on the same
     lives: KM median lifespan and food per step at matched age, natural energy.
     - < 50% of the navigator: STOP (save the rest of the $6) and diagnose.
     - 50-90%: continue and re-check at 300M; still falling at 300M: stop and diagnose.
     - >= 90%: carry on as registered.
     - Separate cause from effect: report food/step for ADULTS only, and deaths by age and cause.
     - **Diagnosis if stopped (laptop):** per-head aux loss vs PPO loss; the norm of the GRU's contribution to the logits
       vs the trunk's; policy entropy; advantage / value stats. Likely first fix: a KL penalty toward the frozen
       navigator's policy (propose a coefficient; do not launch).
  2. **Stage 2 design approved in principle;** it waits for stage 1 and new credits. Build in:
     - (a) the speak head WIRE-BLIND (it cannot read the incoming wire);
     - (b) the approved generic PS nudge (founders / parents only) as an arm-level option against the zero-init
       chicken-and-egg collapse; calibrate the readouts first.
  3. **Cross-fostering 48-probe calibration on Modal APPROVED** (~$0.1, EXPLORE, item watch). Bar: perfect >= 90%,
     deaf <= 10%.
  4. **Budget:** nothing else on this account beyond the pilot and the $0.1 calibration.

- **REGISTERED (EXPLORE, ~$0.1): `xfcal48`**, the cross-fostering calibration at the frozen test's 48 probes x 600 steps.
  - Gate config with a random code per life; 16 worlds x 100 updates; eligible lives only.
  - Bar (Cam): perfect foster-learner >= 90% foster and <= 10% birth; perfect birth-decoder the mirror; deaf <= 10% on
    both.
  - Runs on a SEPARATE app (`throng-w2cal`) so the pilot's app is not redeployed.

- **RESULT (Nov 12 ~06:45): FORAGING CHECK -> STOP.** Laptop, comm-toy `9134001` (`--forage-check`), on the pilot's
  update-6000 checkpoint (98.3M agent-steps; the 100M readout fires at update 6104).
  - Method: the agent vs the navigator (the same weights with `side_head` / `side_value` zeroed) on the same 64 fresh
    worlds and keys, natural energy, 12,800 steps. Every life born in the window counts; lives alive at the end are
    censored. Adult = age >= 1000 (the parent leaves). 95% CIs bootstrap over worlds.
  - My application of the rule (not specified by Cam; written into the code before the real checkpoint was run, but
    not committed first): the decision uses the LOWER of the two ratios.
  - **Adult food per step: agent 0.00181 vs navigator 0.00937 = 19% (CI 18-21%).** Below 50%: STOP.
  - KM median lifespan: agent 955 vs navigator 879 (ratio 1.09). This does NOT mean the agent forages better:
    - In this harness most lives die at nearly the same age, of starvation on the starting energy: navigator 1,734 of
      2,393 deaths at age 800-899; agent 2,410 of 2,807 at 900-999. Juveniles almost never eat (food/step ~0.0004 at
      age < 500, both arms).
    - A longer median there means spending energy more slowly, i.e. moving less.
    - **Controls:**
      - Fresh init (agent == navigator): both ratios exactly 1.000 (life for life identical).
      - Damaged agent (side-head bias +4 on "stay"): KM ratio 1.08, adult food 38% (STOP).
      - So the lifespan median cannot tell a broken forager from a good one here; adult food per step can.
  - Deaths: all starvation except old age at A_max = 3000 (agent 80, navigator 249). Fewer agent lives reach adulthood.
  - Files: `docs/data/stage1/forage_check_u6000.json`, `forage_control_identical.json`, `forage_control_damaged.json`.
- **DIAGNOSIS (laptop, comm-toy `e8a15a1` `--grad-diag`, one training batch from the update-6000 checkpoint):**
  - **The agent has almost stopped moving.** Moves are 1.7% of its actions vs 8.4% for the navigator on the same
    states (both ~47% stay, ~45-51% eat).
  - The learned side-head term swamps the trunk: logit norm 43.9 vs 7.2. Mean contribution: up -34, right -13, stay
    +13, eat +13, down +10, left +10. KL(agent || navigator) = 0.39 nats per step.
  - Entropy 0.62 vs the navigator's 0.78 (max 1.79).
  - Gradient norm into the shared memory (side_enc + GRU): PPO 0.38; aux (weighted 0.5) meal 0.15, other 0.16,
    view 0.0002. The aux heads do NOT dominate; PPO is the larger force.
  - Value: explained variance 0.985; raw advantage sd 0.096 (mean ~0).
  - **A mechanism found in the code (hypothesis, under test):** `step_all` takes the PPO log-probability of the action
    AFTER the parent-seek reflex overrides it (`learnability.py` line ~238).
    - So for juveniles more than r cells from the parent, PPO trains on reflex-chosen moves as if the policy had
      chosen them.
    - Juveniles are 80-93% of training samples, and moves cost energy, so the policy can learn "moving is bad"
      from moves it never chose.
    - The same line serves the earlier non-recurrent learnability arms with reflexes. That is a watch item for those
      closed results, not a retraction.
  - Food per step in training fell from 0.0037 to 0.0013 by update 259, so a short laptop A/B can decide it:
    16 worlds x 250 updates, `--mask-forced` (reflex steps leave the PPO loss) vs unmasked; then the forage check.
    - Prediction, written before running: P(masked >= 0.9 of the navigator AND unmasked < 0.5) ~ 0.45.
    - If both collapse, the reflex is not the cause.
  - Files: `docs/data/stage1/grad_diag_u6000.json`.

- **RESULT (Nov 12 06:44): the 100M READOUT landed; pilot STOPPED at 06:45 per the foraging rule** (`modal app stop`
  throng-w2jobs, after the readout, at update ~6104). Pilot cost ~$1.62 in total (billing by app).
  The readout took 56 min of container time (~$0.45). It is the expensive part; restructure it before any relaunch.
  - **Sanity stop: not triggered.**
    - Shuffled-code probe 11.1%, Wilson 95% CI 10.75-11.44% on 31,941 rows. The lower bound is not above 12.5%.
    - Memory-wiped hungry pass 0% (0 of 181).
  - **Life test: 0% pass in all four arms** (hungry/natural x memory carried/wiped; 181 eligible of 750 lives; CI
    0-2.1%).
    - Food under true and deranged codes is equal (hungry 0.193 vs 0.193). The agent does not act on the code.
    - That is expected from an agent that has stopped moving.
  - **Linear probe on the memory, sector accuracy by tokens heard in the life:**

    | tokens heard | 1-5 | 6-15 | 16-40 | 41-100 | 101-300 | 301+ |
    |---|---|---|---|---|---|---|
    | memory | 24.4 | 35.6 | 40.9 | 44.7 | 43.2 | 37.2 |
    | memory wiped | 26.6 | 31.3 | 30.9 | 32.9 | 34.4 | 28.5 |
    | shuffled code | 12.2 | 10.9 | 10.2 | 11.6 | 11.9 | 10.5 |

    - Move class: memory 69% vs wiped 57%.
  - **Registered "rises" criterion (ii) is MET as written:** memory rises 18.8 points from the 1-5 bin to the 101-300
    bin (>= 10), and is above the wiped probe in the 101-300 bin (43.2 vs 34.4).
  - Criterion (i) (hungry pass by tokens heard) is not met: 0 everywhere.
  - **Caveat before anyone reads (ii) as code learning:**
    - Tokens heard grows with age, so a longer history may carry sector information without decoding the code
      (where food has been, where the speaker stands).
    - The wiped probe (current input only) is already 27-34%, well above chance.
    - A discriminating control is needed, e.g. re-draw the life's code mid-life and test whether the memory's sector
      accuracy drops for the new code. Not run.
  - Under the registered tree, (ii) points to CONFIRM on held-out code families (needs new credits). The foraging
    STOP takes precedence. The relaunch question is Cam's.
  - Files: `docs/data/stage1/s1pilot_100M_readout.json`, `s1pilot_8a0d027.log`.
- **RESULT (Nov 12 06:43): cross-fostering 48-probe calibration `xfcal48` PASSES** (Modal, comm-toy `131f579`,
  $0.13 vs the ~$0.1 estimate: +28%, just past the 25% leeway; disclosed).
  - 173 eligible lives.
  - Perfect foster-learner: 93.6% foster (CI 89.0-96.4), 0% birth (0-2.2).
  - Perfect birth-decoder: 0% foster, 91.9% birth (86.9-95.1).
  - Deaf navigator: 0% and 0%.
  - Both point estimates clear 90%; the lower CI bounds are just below it. The instrument is calibrated.
  - Files: `docs/data/crossfoster/xfcal48_131f579.json`.
- **DIAGNOSIS A/Bs (laptop, 16 worlds x 250 updates = 1.0M agent-steps from the navigator init; forage check on 64
  fresh worlds).** The laptop runs this harness at ~33k agent-steps/s, the same as the Modal container.

  | arm | adult food vs navigator (CI) | verdict |
  |---|---|---|
  | as the pilot (unmasked) | 29% (28-31) | reproduces the collapse in 250 updates |
  | `--mask-forced` (reflex-chosen steps out of PPO) | 34% (31-36) | **refuted**; reflex steps were only ~0.1% of steps |
  | `--ppo-min-age 500` (juvenile energy-floor steps out of PPO) | 36% (34-39) | **refuted** |
  | `--kl-nav 0.1` | 85% (79-90) | partial |
  | `--kl-nav 1.0` | 99% (95-104) | foraging kept |

  - Predictions scored:
    - mask-forced fixes it: p 0.45, MISS;
    - juvenile mask fixes it: p 0.4, MISS;
    - a KL arm reaches >= 0.9: p 0.5, HIT.
  - **The mechanism is still open.** Neither code hypothesis survived. The KL penalty works as a constraint, not as an
    explanation.
  - Lever code: comm-toy `4a73096` (`--mask-forced`) and `d6d155b` (`--ppo-min-age`, `--kl-nav`), default path identical,
    resume test passes.
  - Running: the KL arms at 2,000 updates (8M agent-steps), to see whether the collapse only takes longer.

- **RESULT (Nov 12 07:13): KL arms at 2,000 updates (8.2M agent-steps, laptop):**
  - `--kl-nav 1.0`: adult food 92% of the navigator (CI 87-97). Down from 99% at 250 updates.
    - KL to the navigator 0.002 nats per step; moves 3.0% vs the navigator's 3.3%.
  - `--kl-nav 0.1`: 68% (62-74), still falling.
    - KL 0.029 nats per step; moves 2.6% vs 5.1%.
  - Predictions:
    - beta 1.0 holds >= 0.9: p 0.65, HIT (narrowly);
    - beta 0.1 falls below 0.5: p 0.5, MISS (68%).
  - **Reading:** beta 1.0 keeps foraging by keeping the policy almost exactly the navigator's. That may also leave too
    little room for listening to change behaviour. The 0.002 nats per step is averaged over all steps; listening
    needs deviation only on heard steps.
  - Files: `docs/data/stage1/ab2000_*`.

- **Cam, Nov 12 (rulings on the ~07:20 stop report).**
  1. The STOP is correct, and "lower of the two ratios" is accepted. Food per step is the measure (KM lifespan flatters
     a frozen agent). `xfcal48` PASS accepted (93.6 / 91.9 / 0); the +28% on $0.13 is fine.
  2. **First, a free RETURN CHECK (laptop):** on the same lives and keys, the training objective (discounted return
     under the training gamma, with the actual per-step reward) for the navigator, the collapsed u6000 agent and a
     scripted stand-still agent.
     - If freezing scores >= the navigator: the objective rewards freezing. Fix the objective; report the gamma (and
       matching lambda) at which the navigator beats freezing; confirm the reward is the homeostatic drive reduction
       as designed, including the move cost.
     - If the navigator scores higher: PPO is failing to optimise. Check truncation / bootstrap at rollout ends and
       deaths, advantage normalisation, and the GRU's logit scale (consider bounding it).
  3. **Then a KL SCREEN (laptop, 2,000 updates each, same seeds):**
     - (a) beta 1.0 everywhere;
     - (b) beta 1.0 on silent steps only;
     - (c) beta 0.3 everywhere;
     - (d) the objective fix from (2) with no KL, if (2) found one.
     - Screen metric: adult food >= 90% of the navigator AND token influence > 0 and growing over updates. Token
       influence = KL between the policy with the true token and with a swapped token, on token steps.
     - Pick the variant that holds foraging with the largest token influence. Prefer (d) if it holds foraging, since a
       KL to a fixed navigator caps later learning.
  4. **Probe control:** the mid-life code re-draw on the 100M checkpoint (laptop). Re-draw at a fixed age; score the
     probe against the NEW code by tokens heard since the re-draw.
     - In-context learning is shown only if accuracy drops at the re-draw and climbs back above the wiped baseline.
     - Until then, criterion (ii) reads "met as written, confounded by age".
  5. **Relaunch:** one allowed, only if the screen produces a passing variant.
     - Same registered design plus that fix; $6 cap (~$10.9 under the guard).
     - Readouts halved at 100M / 300M, full at 1B. Same foraging stop rule.
     - If nothing passes: no relaunch; report.

- **RESULT (Nov 12 ~08:00): RETURN CHECK (laptop, comm-toy `17bb0c4` `--return-check`; u6000 checkpoint; 64 fresh worlds,
  12,800 steps; complete lives born in the first 9,800 steps; the actual training reward x20).**
  - **Reward confirmed as designed.** r = drive(E_prev) - drive(E), drive = (0.8 - E)^2. The energy update includes
    d_move 0.003 on moves (d0 0.0013). Inclusive fitness off.
  - **Structural fact:** this reward telescopes. Summed over a life it equals drive(E_birth) - drive(E_death), so
    every starving life gets the same undiscounted total however long it lived or ate.
    - Only discounting and old-age deaths separate policies.
    - The juvenile top-up (age < 500) adds unrecoverable negative reward: energy is refilled free after each paid
      loss. That is why stand-still lives total -17.2, not -12.
  - From-birth discounted return, mean (CI over worlds):

    | arm | gamma 0.99 | 0.995 | 0.998 | 0.999 | 1.0 |
    |---|---|---|---|---|---|
    | navigator | -0.965 (-0.98,-0.95) | -2.03 | **-5.19** | **-8.37** | **-15.5** |
    | collapsed u6000 agent | **-0.904** (-0.93,-0.88) | **-1.95** | -5.26 | -8.73 | -16.3 |
    | stand still | -1.047 | -2.22 | -5.92 | -9.67 | -17.2 |
    | stand and eat | -1.021 | -2.17 | -5.80 | -9.51 | -17.1 |

  - Return-to-go by age at gamma 0.99:

    | age | navigator | agent | stand and eat |
    |---|---|---|---|
    | 0-499 | -1.07 | **-1.04** | -1.14 |
    | 500-999 | **-1.83** | -1.95 | -2.19 |
    | 1000-1499 | **-0.44** | -0.79 | -1.03 |
    | 1500-1999 | **-0.47** | -0.89 | -1.09 |
    | 2000-2999 | **-0.45** | -0.70 | -1.15 |

  - **Reading: BOTH branches, in different phases.**
    - Pure freezing never beats the navigator.
    - The collapsed agent beats it only from birth at gamma <= 0.995. That return is dominated by the topped-up
      juvenile phase (0.99^500 ~ 0.007), where moving is pure cost and the objective does reward near-freezing.
    - In adult states the navigator earns clearly more under the training objective, at every gamma. PPO made adult
      behaviour worse by its own measure.
    - Masking juvenile steps did not fix it (36%), so the adult failure is primary.
    - Caveat: by-age return-to-go compares each policy on its own state distribution, not matched states.
  - **Navigator beats both freezing and the agent from birth at gamma >= 0.998** (lambda 0.95 kept: GAE horizon
    1/(1 - gamma*lambda) goes from 17 to 19 steps).
  - PPO mechanics checked against Cam's list:
    - Rollout-end bootstrap uses v(s_t), not v(s_T). Small bias, identical in the navigator's own training
      (`w2/learn.py`), which learned to forage.
    - Deaths cut the bootstrap correctly (nonterminal = 1 - dead).
    - Advantage std is taken over all steps, dead ones included (the mean is masked). Minor.
    - Reward scale x20 is the same as the navigator's.
    - The stage-1-specific difference is the unbounded side head (6x the trunk at u6000).
- **REGISTERED (Nov 12 ~08:00): KL SCREEN** (laptop, from the navigator init, 16 worlds x 2,000 updates = 8.2M steps;
  same seeds; forage check at the end on 64 fresh worlds; token influence every 100 updates).
  - Arms:
    - (a) `--kl-nav 1.0`;
    - (b) `--kl-nav 1.0 --kl-silent-only`;
    - (c) `--kl-nav 0.3`;
    - (d) the objective fix `--gamma 0.998` (lambda 0.95), no KL;
    - (e) MY ADDITION under Cam's item-2 checklist: `--side-bound 3` (side-head logits bounded to +-3), no KL.
  - Pass:
    - adult food >= 90% of the navigator (point);
    - AND token influence > 0 and growing, operationalised as mean over updates 1500-1999 > mean over 500-999.
  - Pick the passing variant with the largest final token influence. Prefer (d) if it passes.
  - Predictions (P food >= 0.9 / P influence growing):
    - (a) 0.6 / 0.3;
    - (b) 0.45 / 0.45;
    - (c) 0.3 / 0.4;
    - (d) 0.3 / 0.4;
    - (e) 0.25 / 0.4.

- **RESULT (Nov 12 21:48): PROBE CONTROL, mid-life code re-draw** (laptop, comm-toy `7d92832` `--redraw-probe 450`).
  - Setup:
    - Checkpoint u6000 (98.3M steps; the exact 100M weights were not checkpointed before the stop).
    - 64 fresh worlds x 6,400 steps, 1,261 lives, 390k probe rows.
    - Each life's code is re-drawn at the first rollout boundary after age 450, to a code that differs on every sector.
    - The probe is trained on pre-re-draw rows of 70% of lives and scored on the other 30%.
    - **Added negative control (mine): the same run at fresh init**, i.e. an untrained memory.
  - Sector accuracy (%):

    | | pre, 101-300 tokens since birth | post, by tokens since the re-draw: 1-5 / 6-15 / 16-40 / 41-100 / 101-300 |
    |---|---|---|
    | u6000 memory | 47.6 | 30.3 / 33.0 / 34.8 / 35.7 / 33.8 |
    | u6000 wiped | 36.3 | 26.1 / 25.1 / 24.5 / 22.9 / 16.3 |
    | fresh-init memory | 34.5 | 22.0 / 26.8 / 26.4 / 25.0 / 24.6 |
    | fresh-init wiped | 32.6 | 20.1 / 21.6 / 19.7 / 18.3 / 15.2 |

  - **Cam's literal test (drop at the re-draw, then climb back above wiped) is passed, and it is also passed by the
    fresh-init control.** Post rows are older lives in other states, so the accuracy drop and the falling wiped
    baseline are partly distribution shift. As worded, the test has no teeth here.
  - **Against the control:**
    - The memory's advantage over wiped falls at the re-draw from 11.3 points (pre, 101-300) to 4.2 (post, 1-5).
      The control's stays 1.9 -> 1.9.
    - So part of the trained memory's extra sector information depends on the life's code. That is evidence it reads
      tokens through a code-specific mapping.
    - Recovery is weak. Memory rises 5.4 points over the first 100 tokens after the re-draw (control +3.0) and never
      approaches the pre level (47.6).
  - **Reading:** code-dependent information in memory: suggestive, yes. In-context re-learning of a new code: not
    shown beyond the control.
  - Criterion (ii) now reads: "met as written; confounded by age; the re-draw shows a code-dependent component but no
    clear re-learning".
  - Prediction (drop >= 5 points AND recovery above wiped, p 0.3): HIT on the letter, but the control also hits.
    Scored as uninformative.
  - Files: `docs/data/stage1/redraw_probe_u6000.json`, `redraw_probe_freshinit_control.json`.
- **Ops:** the spend guard had died at 14:44 (heartbeat 7h stale; no Modal activity in the gap; metered $30.07
  unchanged). Restarted at 21:49 with CAP 41.0.

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
