# @Will — Cam here. THRONG cold-restart work order (Sep 13, 2026)

**Context:** the project has been idle since late June. Last commit `7fbf7d9` (Jun 23) on
`feature/phase18-crafting`. There is **uncommitted July work in the tree** that nobody has
verified. Modal is on a **brand-new account** — the `thirtytwogeese` volume (last live run
reached ppo ~2617–2628, plus the corpus that produced our first significant ATE) is not
reachable from here. Newest local checkpoint is **`mnt/checkpoints/2244`** (Jun 17).

**The repo docs lag the last live run.** `THRONG.md` stops at ppo=2578; the run actually
continued to ~2628. What happened after 2578, from my session notes: breaking the barrier
fortress replaced it with a **pure-evasion attractor** — blues surviving entirely on
movement, `blue_caught` pinned at zero for 20+ updates — and codebook churn ran at
**100–157 dead-code resets per update**. An intervention raising `red_catch_radius` 1 → 2
was issued but **never verified**: the startup log still printed `catch_radius=1`, so the
predation spike at ppo=2617 may have been nothing more than a fresh-population restart
artifact. Treat every post-2578 number as unverified until a clean run reproduces it, and
fold the missing history back into `THRONG.md` as part of Task 1.

**Read first:** `THRONG.md` §0b, §0 (standing directives), §4 (ops);
`docs/STRATEGIC_ROADMAP.md` §4 and §12. Rules 11 and 12 still bind.

**Do not start a long GPU run until Tasks 1–4 are green.** Standing rule from the roadmap:
*no long training on un-asserted learning signals.*

---

## Task 0 — DEAD. The old account is suspended; the volume is gone.

M's `thirtytwogeese` account is suspended, so `throng-runs` is unreachable. Do not spend
further effort on it. M can file a support appeal with Modal in parallel, but nothing on the
critical path waits on it.

**What we actually lost — verified, not assumed.** I read the Orbax `_METADATA` on every
checkpoint in the folder. The newest local checkpoints (`mnt/checkpoints/2244`, Jun 17, and
`2214`) are **Phase 17.5 architecture, not Phase 18**:

| Param | Local ckpt 2244 | Phase 18.7 (what we were running) |
|-------|-----------------|-----------------------------------|
| `head_action` | `[256, 8]` | 12 actions |
| `codebook` | single `[64, 32]` | three slots, 12/8/12 |
| `emb_own` / `emb_env` | `[10, ·]` / `[10, ·]` | 22 / 15 |
| `gwt_comms_1` | `[2378, 256]` | `[2731, 256]` |
| `head_alarm` | present | present (vestigial) |

So the folder number 2244 is misleading: the PPO index kept counting across phases, but the
weights stored there predate Phase 18 entirely. **Every Phase 18 weight is gone** — the
12-action space, the 3-slot discrete VQ, the bypass amputation, the crafting ecology, the
receiver-necessity ecology. So is the corpus behind the Slot 2 ATE.

**What survives:** all of the code, in git, intact. The trunk carries ~1.15M steps of
evasion and survival skill. And the Phase 18 migration path is automatic and still in the
source — `clean_old_codebook` drops the old 32D codebook so the 12/8/12 slots initialise
fresh, `pad_gwt_comms_1` walks the observation graft 2378 → 2731, and `head_action` grafts
8 → 12. This is the exact path the project took the first time. Resuming 2244 into Phase
18.7 is supported by design, not a hack.

**Record this honestly in `THRONG.md`, and do not soften it.** The Slot 2 ATE
(+0.1295, CI [+0.0364, +0.2007]) was observed **once**, inside a two-update transient during
the fortress break, and the corpus that produced it no longer exists. Its status is now
*observed once, evidence destroyed, never replicated.* It is not a project milestone and
must not be cited as one until it is reproduced.

## Task 0.5 — Adversarial codebase audit, BEFORE any fix or any training

M wants a full skeptical pass over the codebase before we commit compute. I agree, and the
reason is not stylistic: this repo has a **documented history of silent-severance bugs** —
a learning signal disconnected from the thing it was meant to train, running undetected for
weeks. Three in recent memory: the VQ loss severed from the autodiff tape across all of
Phase 18; `NB_GAIN` initialised to 1.0 and never updated, a pure ghost metric; the red VQ
loss read at the wrong tuple index, feeding raw `z_e` into PPO. Each cost real compute and
one of them nearly cost us a false scientific conclusion. Audit for *that class*.

**This task is read-only.** Do not fix anything you find. Do not refactor opportunistically —
an unrequested cleanup mid-audit is precisely how the last severance bug got in. Produce a
report; I sign off; then you fix. The only code you may write is a throwaway assertion that
proves a finding.

**Be skeptical of the documentation, including everything I have written.** `THRONG.md`,
`docs/ARCHITECTURE.md` and this file are all secondary sources. Where prose and source
disagree, the source wins and the disagreement is a finding.

### A. Severance sweep — every loss must reach a parameter

For each loss term in the training step (PPO policy, value, entropy, logit penalty, VQ
commitment per slot, `fwd_env`, `carry_fwd`, `self_pred`, `proprio`, `confidence`, SRL
retention, and anything else you find), produce a row: the config coefficient, the tensor it
multiplies, the parameters it should update, and **whether a non-zero gradient actually
reaches them**. Prove it by computing `‖∂loss/∂param‖` on a CPU minibatch, not by reading
the code and reasoning about it. Any term whose gradient is structurally zero, detached by a
`stop_gradient`, or multiplied by a coefficient of 0 that nobody intended is a finding.
Generalise `tests/test_vq_gradient_flow.py` into this sweep — that is H2 finished properly.

### B. Ghost-metric sweep — every dashboard number must be live

For each value printed on the dashboard, establish whether it is computed from the current
rollout or is a stale initialised constant. `NB_GAIN` is the known case. Flag any metric
whose variance across updates is structurally zero, and any metric read from a buffer that
nothing writes during the rollout.

### C. Doc/code drift, with three known suspects

Roadmap item H5 already names these; confirm or refute each from source:
- `tom_logits` — computed, but is there any reward or loss attached in the JAX loop?
- Push (6) and Guard (7) — the masking decision rests on the claim that `grid_jax.py` gives
  them no mechanics. Verify that claim directly; the mask is only legitimate if it holds.
- **The blue carry path.** Docs claim a GRUCell pivot; my read is that blue still uses a soft
  EMA (`0.9·carry + 0.1·pooled`) and only red got a real `GRUCell`. If blue is on an EMA,
  every claim we have made about blue temporal memory and episodic-memory results rests on a
  different mechanism than the documentation says. **Settle this one first — it is the
  highest-stakes item in the audit.**

### D. Config reachability, both directions

Every key in `config.yaml` that nothing reads, and every hardcoded constant in code that
silently shadows config. Both directions have burned us: `modal_train.py` pinned
`env_channels` to a stale 12 and invisibly hid three resources from the agents' observations
for an entire phase, and `n_actions` defaulting to 8 on direct-YAML loads was a live trap.
Pay particular attention to magic slice indices over the observation and the 40D wire.

### E. Invariants

Confirm the PPO logit-mask invariant holds in all three places — rollout, imagination, and
the PPO backward pass — and that the obs-layout slices agree between `observations_jax.py`,
the network's `_obs_layout`, and the corpus writer.

### Deliverable

A written report at `docs/AUDIT_SEP2026.md`, findings ranked by severity, each with
`file:line`, what breaks, and a proposed fix. **Separate confirmed defects from
suspicions** and label which is which — a suspicion reported as a defect is its own kind of
bug. Include what you checked and found clean, so we know the coverage.

Send it to M, who will relay it to me. **Do not start Task 1 until I have signed off.** If
the audit turns up something that changes the restart plan — especially the carry path — say
so plainly and we will rethink before spending a single GPU-hour.

## Task 1 — Finish and land H1 (the severance-class fix)

The July tree replaces the positional network-output tuple with a
`NetworkOutputs` flax dataclass in `jax_sim/network_jax.py`. `main_jax.py` and `rl_jax.py`
are converted. **Two sites are not:**

1. `jax_sim/network_jax.py::monologue_forward`, lines ~443–447, still does
   `outs[5] / outs[6] / outs[7] / outs[8] / outs[1]`. `NetworkOutputs.__getitem__` raises
   `TypeError` by design, so this path is a guaranteed crash the moment
   `monologue_enabled` is ever set true. Convert to named attributes.
2. `tests/test_vq_loss_index.py` still builds raw tuples and passes `vq_loss_idx=` to
   `ppo_loss`, which no longer accepts it. **Our H2 regression test is currently broken.**
   Rewrite it against `NetworkOutputs`: construct blue (with `alarm_out`) and red
   (`alarm_out=None`) instances and assert `ppo_loss` reports `TRUE_VQ` for both — the
   dataclass now makes the red-index bug structurally impossible, so the test should pin
   *that* invariant, not the old integer indices. Keep the file name and the docstring
   history.

Then grep the whole repo (`jax_sim/`, `tools/`, `tests/`, `communication/`, `scripts/`) for
any remaining positional indexing of network outputs and convert it. Run the test suite on
CPU. Commit H1 as its own commit and mark H1 **DONE** in the roadmap §12 table.

## Task 2 — Verify §4.3 (imagination over the full action space)

The tree changes `imagination_n_actions` default 5 → 12 and adds an `imagined_action`
rollout field plus an `Actions (imag):` dashboard line. Verify end-to-end that
`imagine()` actually scores all 12 actions and that Strike/Craft/UseTool have defined
imagined values — not just that the constant changed. A short CPU smoke test that asserts
the imagined-value tensor has 12 columns and that the argmax can land outside 0–4 is enough.
This is roadmap execution item #2 and a hard prerequisite for the System-2 thesis.

## Task 3 — Land the per-slot decode instrumentation

`tools/decode_signals.py` now runs χ² per slot and gains a `--synthetic` validation mode
with known ground-truth rules. Run `python3 tools/decode_signals.py --synthetic` and confirm
the NPMI parser recovers the planted slot→noun / slot→verb / slot→adverb structure. If it
does, commit and close the remaining §4.0 instrumentation-debt item (per-slot NPMI in
`load_corpus`). Also review the July changes to `tools/causal_intervention.py` and state
plainly whether the live per-slot swap still matches the pass bar (`p < 0.05` **and**
`|Δ| > 0.05`).

## Task 4 — Break the Gate 2 / Gate 3 deadlock at the mechanism, not with a config knob

This is the real blocker and I want it fixed structurally.

**Diagnosis (code-grounded, `jax_sim/network_jax.py::dead_code_reset_codebook_params`):**
usage is `jnp.bincount(token_ids)` over **one rollout's alive agents**, and every code with
zero usage in that single window is overwritten with a random `z_e` sample. So when
predation kills a large fraction of the population, the surviving token pool shrinks, most
codes read as dead, and the codebook gets scrambled. That is why `codes_active` went
`35|51|46` → `7|11|14` the instant `blue_caught` hit 154 at ppo=2578 — and why the quiet
pure-evasion phase after it still burned 100–157 resets per update: a population that never
dies and never diversifies its contexts leaves most of the codebook unused, so the reset
fires just as hard from the other direction. Gates 2 and 3 are not
merely anti-correlated by bad luck — **the reset mechanism destroys the measurement
instrument precisely during the event we are trying to measure.** Escalating
`barrier_build_cost` a third time cannot touch this.

**Fix:** make death a code's cause of retirement only if it is *persistently* unused.
Maintain a per-slot usage EMA across PPO updates (persist it alongside the codebook so it
survives checkpointing) and reset a code only when its EMA usage stays at zero for W
consecutive updates (start W = 5). Additionally skip the reset entirely on any update where
the alive-agent token pool is below a floor (start: 25% of `max_pop`), since a small pool
cannot support a 64-code usage estimate. Log resets with the EMA value that triggered them
so we can see the mechanism working.

This should let Gate 2 and Gate 3 hold simultaneously for the first time since ppo=2515.

**Secondary (do not do it yet — tell me your read first):** the barrier fortress is an
escape hatch out of the ecology, and we have escalated build cost twice with only damped
oscillation to show for it. My inclination is a mechanism change instead — let red Strike
damage `barrier_hp_map` so walls are contested rather than merely expensive. That is an
ecological-parameter class change, so **Rule 12 applies: no ecology edits without my
sign-off.** Give me your assessment; do not implement.

## Task 5 — Restart on `agionetwo`: replay Phase 18, do not re-run the pathology

Only after Tasks 1–4 are green.

**Profile: `agionetwo`.** Confirmed by M — `onetwoagi` is a previous, dead Modal account;
do not use it. `modal token new --profile agionetwo` then `modal profile activate agionetwo`,
and check `modal profile list` before any volume operation.

Then `./scripts/migrate_modal.sh upload ~/throng_backup` with `mnt/checkpoints/2244` as the
checkpoints payload, and resume. Watch for these graft banners at startup and report them
verbatim — if any is missing, stop and tell M rather than training on a bad graft:

- `[JAX] Removed old 32D codebook from params to allow Phase 18 discrete slots to init.`
- the `pad_gwt_comms_1` expansion lines walking through to `obs_dim = 2731`
- `[JAX] Grafting padding to head_action: expanded from 8 to 12 actions`
- `[JAX] Restored params from step …`

Single `run_bg.py` process, `reset_red_vq_on_resume: false`, and the §4 startup checklist.

### The one change I am making to the replay — Rule 12 sign-off, logged

**Amputate the barrier mechanic.** Logit-mask `Build` (action 8) to `-1e9` exactly the way
Push and Guard are masked — rollout, imagination, and the PPO backward pass identically, or
we reproduce the NaN cascade.

Reasoning, and I want you to push back if you think it is wrong. Barriers came in at Phase
16.5 to force channel grounding through line-of-sight occlusion. By Phase 18.7 they had
become the dominant escape hatch out of the ecology: agents bulk-built walls, zeroed
predation entirely, and decoupled survival from communication. We then spent the whole of
the lost window fighting that with `barrier_build_cost` escalations — 0.06 → 0.10 → 0.15 —
and bought only a damped limit cycle. Breaking the fortress did not fix it either; it just
traded the fortress for a pure-evasion attractor with the same signature, `blue_caught`
pinned at zero and the codebook churning.

Crucially, **barriers are no longer carrying the grounding mechanism.** Phase 18.7's
receiver-necessity is recipe asymmetry — informed agents must transmit ingredients to blind
workers — not predator occlusion. And blues stay blind beyond their 5×5 patch regardless,
because `red_detection_radius: 0`. So removing Build costs us the 16.6 occlusion pressure we
had already superseded, and removes the single degree of freedom that lets the population
opt out of predation altogether.

Replaying Phase 18 verbatim would replay the pathology. This is the controlled version of
the same run.

### What makes this an experiment rather than a cleanup

With the fortress gone (this task) and the codebook no longer scrambled by mass death
(Task 4), **Gates 2 and 3 should hold simultaneously for the first time since ppo 2515** —
oscillating predation *and* `codes_active ≥ 40` per slot, sustained rather than transient.
That is the prediction. If the gates still refuse to co-open, my dead-code diagnosis is
wrong and I want to know early.

Then the real test: run the per-slot causal intervention in a *stable* ecology instead of a
two-update window. Slot 2 either replicates or it does not. If it does not, the June number
was a transient artifact and we write it up as one.

## The gate that matters

Everything above is in service of one thing. At ppo≈2576 we got the first statistically
significant causal result in this project's history:

> **Slot 2 ATE = +0.1295, 95% CI [+0.0364, +0.2007]** — tokens in slot 2 causally increase
> the flee rate of blind receivers. Slots 0 and 1 still include zero.

It was measured in a **two-update window** during the fortress break, while the codebooks
were briefly open at `35|51|46`, and the corpus behind it lives on a Modal volume we may no
longer be able to reach. **One measurement in a transient window is not a result yet.**

So the first science objective after restart is not a new phase. It is **replication**: get
the codebooks stable under active predation (Task 4), accumulate a clean post-restart
corpus, and re-run the per-slot causal test. If slot 2 replicates, receiver-necessity is
confirmed, roadmap item #4 closes, and we go to item #5 — productivity pressure and the
receiver-reset schedule — with topsim as the headline gate. If it does not replicate, we
say so plainly and treat the June number as a window artifact.

Report back before starting Task 5.

---

# CAM SIGN-OFF ON THE AUDIT (Sep 13, 2026)

Audit accepted. I independently verified Findings 1, 2 and 4 against source before signing:
no bare `import flax` in `network_jax.py`; `config.yaml:216,218` both `false` while
`imagination_gating_enabled: true` at `:221`; blue `new_carries = 0.9*carries + 0.1*pooled`
at `network_jax.py:355` against red's `carry_gru` at `:766`. The method was right — measuring
gradients rather than reasoning about them is what made Finding 2 a fact instead of an
opinion, and building the harness outside the repo rather than "just fixing" the import was
the correct call.

**Proceed.** Revised task order below. Three things first.

## 1. Sweep A is not finished, and it is the most valuable thing in the audit

I asked for a measured `‖∂loss/∂param‖` for **every** loss term. You measured the confidence
head and the VQ codebook and *read* the rest ("all nonzero in the live config and correctly
threaded"). Reading is what let the original severance bug survive for a phase.

Finish it: one table, every term — PPO policy, value, entropy, logit penalty, VQ commitment
per slot, `fwd_env`, `carry_fwd`, `self_pred`, `proprio`, `confidence`, SRL retention — each
with a measured gradient norm on a real CPU minibatch at **production shapes**
(`env_channels=15`, `own_state_dim=22`), for blue and red separately. Then land that table as
a permanent test. That is H2 finished properly, and it is the artifact that stops this bug
class recurring. It matters more than any single finding you reported.

## 2. Two interpretations you left open that I want closed

**Finding 2 — random or fossil?** You established the confidence head receives zero gradient.
You did not establish what is *in* it. If `head_confidence_*` in ckpt 2244 differs from a
fresh init, the head is a **fossil** — trained in the 9.1 era, then carried frozen through
every graft since, predicting carry-forward error for an architecture that no longer exists.
That is a different and worse story than "untrained," and it changes how every historical
`conf_gate_imagine_frac` and `imagination_agree` number should be read. Compare the
checkpoint's weights against a fresh init and tell me which it is.

**Finding 3 — the corruption predates 18.7, and that voids a published negative.** Your
formula diagnosis omits the neighbour-alarm block, which landed in Phase 17.5. So the
corruption was live during 17.5's exhaustive NPMI scan — the one recorded in THRONG.md as
*"Red_Dist ≈ 0, Resource ≈ 0, Puzzle = 0, Contested = 0"* and used to conclude the alarm
channel was cheap talk. Those nulls were measured against the wrong channels at shifted
cells. **A null measured on the wrong channel is not a null.** That result is void, not
negative, and THRONG.md must say so. Work out the offset error for the 17.5-era config and
state which of those scans survive and which do not.

## 3. What you filed as a footnote is a scorecard correction

`cross_attn_enabled: false` is not a docs/config mismatch. `NeighborCrossAttention` is the
receiver's mechanism for integrating neighbour signals, and receiver-side grounding is the
entire open problem of this project. We have spent six phases measuring receiver
insensitivity while the receiver ran the fallback per-neighbour path — and every document
says attention is on.

It also breaks a claim in `docs/STRATEGIC_ROADMAP.md` §11. GWT-2 is scored **Present** there,
justified as *"the GWT-masked 3-slot VQ wire is a near-textbook limited-capacity workspace;
cross-attention is the selective-attention mechanism"* — and called THRONG's genuine
structural edge over feedforward LLMs. The bottleneck half is real. The selective-attention
half is not running. Combined with Finding 2 (GWT-4's gate reading an untrained head), **two
of the four GWT indicators are scored higher than the code earns.** Correct the scorecard.

**Before I finalise this:** check `git log -p` on `phase9_canvas.cross_attn_enabled` and tell
me whether it has *ever* been `true` in a live run. If it has, this is a regression to
restore. If it never has, we are enabling it for the first time and it is a genuine
experimental variable, not a correction — which changes how I log it.

---

## Revised task order

**Task 1.0** — `import flax`. One line. Then run the full suite and report what passes.

**Task 1.1** — H1 completion as originally specified: `monologue_forward` named attributes,
`test_vq_loss_index.py` rewritten against `NetworkOutputs`.

**Task 1.2** — Fix the three broken tests, and fix the deeper problem you spotted: they
instantiate networks with zero constructor args, so they test dataclass defaults
(`env_channels=9`, `own_state_dim=10`) instead of production shapes. A test that cannot catch
a live-layout regression is a ghost test. Parameterise them from config.

**Task 1.3** — Finding 3. Replace the hand-rolled offsets with `make_obs_layout`. **This must
land before any corpus is collected** or we poison the new corpus identically.

**Task 1.4** — Finding 5. Mask red's Push/Guard at rollout sampling and in `r_log_probs`.

**Task 1.5** — Finding 8. `modal_train.py`'s 22-key shadow config, fixed **now**, not
opportunistically. We are about to restart on a new account and tune parameters — this is the
exact mechanism that hid three resources from the agents for a whole phase, and the next
tuning commit is not guaranteed the same luck the barrier commits had.

**Task 1.6** — Findings 6 and 9: delete the `NB_GAIN` plumbing outright rather than carrying
inert code that looks live, and fix the `red_hidden_dim` misread in the three tools.

**Finding 7 — my ruling:** leave `reward_starvation` and `reward_blue_red_proximity` dead and
delete the keys. I am not adding reward terms during a restart. Deleting a key that nothing
reads is not an ecology change and does not need a Rule 12 decision; adding a live reward term
would, and I am declining it.

Then Tasks 2, 3, 4 unchanged, then Task 5.

## Config corrections landing at the restart — Rule 12 sign-off, logged

1. `confidence_enabled: true` — the gate is already live and shaping behaviour every step.
   Running it on an untrained head is strictly worse than either training the head or
   switching the gate off, and §4.3 plus the System-2 line both require the gate to be real.
2. `cross_attn_enabled: true` — pending your git-history answer above.
3. Build (action 8) logit-masked, as specified in Task 5.
4. The `dead_code_reset` usage-EMA fix from Task 4.

**The honest caveat, and I want it in the decision log.** Four simultaneous changes violates
this project's one-variable-at-a-time rule. I am overriding it, with reasons: three of the
four are corrections to confirmed defects rather than experimental treatments, we are
re-deriving Phase 18 from a 17.5 checkpoint so fresh params are being grafted throughout
regardless, and the June baseline they would otherwise confound against no longer exists. The
only genuine experimental variable is the barrier amputation — plus cross-attention, if your
git check shows it has never run. Any result from this run is compared against the post-fix
baseline we are about to establish, never against the June numbers.

---

# CAM FINAL RULING (Sep 13, 2026) — order unblocked

Addendum accepted. Sweep A is what I wanted: `tests/test_severance_sweep.py` reading shapes
from config rather than hardcoding them, and the structural/effective split, are both better
than what I specified. The red aux-update BPTT asymmetry is a real find — blue's aux touches
heads only, red's scans the full trunk — and pinning it in a test is worth more than the
finding itself.

**I was wrong on Finding 3 and you corrected me.** I told M the Phase 17.5 NPMI nulls were
void. They are not: `alarm_correlation.py` reads grid state directly and never touches
`idx_offset`, so the alarm-is-cheap-talk verdict stands. The contaminated claim is the
adjacent one — the z_q-channel LRT, whose `adj_bg`/`adj_barrier` control covariates come from
the buggy corpus writer. Narrower and more precise than my framing. Corrected in the project
record.

Finding 7 deletion was within the ruling I gave. Fine.

## `cross_attn_enabled` — reversing my earlier position: leave it FALSE

It has never been `true` at any commit, so enabling it is a first-time architectural change to
the receiver pathway — the exact pathway whose insensitivity we are trying to measure.
Bundling it with the barrier amputation would give the replay **two** genuine experimental
variables at once and make an ATE replication uninterpretable. And the graft already handles
`nb_cross_attn` on any resume (`CHECKPOINT_GRAFT_TOP_KEYS`), so adding it later costs one
restart, while disentangling it later costs the whole run.

So: cross-attention stays off for the replay, and becomes **the first planned experiment after
ATE replication**, run as a proper A/B. Log it as a pending experiment, not a fix.

## Confidence head — enable training AND re-initialise the fossil

`confidence_enabled: true`, and at graft time **reset `head_confidence_1/2` to fresh init**
rather than carrying ckpt 2244's weights forward.

A fossil calibrated to a carry-forward target on a dead 8-action architecture is worse than a
clean head that starts training immediately — keeping it means spending unknown training time
un-learning a stale prior while the gate consults it. Fresh init plus live gradient is the only
defensible state, and it is a correction rather than a treatment: it makes the gate actually
epistemic, which is what every document already claims it is.

This needs a deliberate exception to the grafting rule that never resets existing parameters.
Make it explicit and loud in the startup banner so nobody later mistakes it for a bug.

## Restart config, final

1. `confidence_enabled: true` + fossil `head_confidence_*` re-initialised.
2. `cross_attn_enabled: **false**` — unchanged, now a logged pending experiment.
3. Build (action 8) logit-masked, all three sites.
4. `dead_code_reset` usage-EMA fix (Task 4).

That leaves the barrier amputation as the single genuine experimental variable. Better than
where I was yesterday.

## Go

Run 1.0 through 1.6 as written, then Tasks 2, 3, 4, then 5. Report after 1.6 with the suite
green. Two additions to the fix order:

- **1.3a** — once `make_obs_layout` replaces the hand-rolled offsets, add a test that pins
  `loc_env_start` against the layout for the live config, so this specific bug cannot silently
  return the next time `own_state_dim` grows. It has now grown three times and broken this
  formula every time.
- **1.7** — footnote the z_q-channel LRT claim in `THRONG.md` as covariate-contaminated
  pending re-run, and correct the Phase 15.4/15.5 entries to say blue carry = EMA. Leave the
  Phase 17.5 alarm verdict alone; it stands.

---

# EXPERIMENT QUEUE (post-replication) — Cam, Sep 13

Nothing here runs until slot-2 ATE replicates in a stable ecology. Recorded so the ordering
is fixed in advance rather than argued about later.

1. **Cross-attention A/B.** `cross_attn_enabled` true vs false, otherwise identical. First-time
   change to the receiver pathway; deliberately not bundled into the replay.
2. **Channel cost / opt-in broadcast.** Today the 40D wire is free, always-on, and unavoidable —
   no agent chooses to speak and none pays to. The Phase 15 contingency table lists sparse
   budgets (~30% broadcast limit) and it was never implemented. Make emission an action with a
   metabolic price, so silence is the default and speech is a decision. Productivity pressure
   (JAIR) points the same way: a channel nobody must choose is a channel nobody must mean.
3. **Population scale.** `max_pop` 200 / floor 150 is a narrow band for social structure to form
   in. Raising it is cheap to try and plausible post-grounding; it is worthless before, since
   scaling an ungrounded channel scales cheap talk (NeurIPS 2025: emergent protocols degrade as
   worlds grow, intention-sharing does not).
4. **Recurrent depth (DeepLoop-style looped trunk)** replacing the binary imagination gate with
   adaptive loop count driven by the confidence head. Parameters stay fixed, which matters for
   the Loihi actor. Brain change — last, and only on a working ecology.

---

# RULE 13 (new, standing) — Prove the work happened; never trust a success message

**Statement:** before believing any step did what it claims — a training update, a test, a
decode, a data write — prove it by *measuring an effect*. A success message, a green test, an
absent error, or a plausible wall-clock time are not evidence that work occurred. Every
degenerate case must fail loudly rather than pass quietly.

**Why this is a rule and not advice.** Every serious defect this project has shipped is one
failure class: *something reported success while doing nothing.* Nine instances on record —

1. VQ commitment loss severed from the autodiff tape for all of Phase 18; training "ran."
2. `NB_GAIN` initialised to 1.0, never written, printed on the dashboard for many phases.
3. Red's VQ loss read at blue's tuple index — PPO minimised raw `z_e` and reported a number.
4. `confidence_enabled: false` zeroing `conf_coef` while the epistemic gate consumed the head
   every step; every gate metric on record was produced by an untrained fossil.
5. `idx_offset` reading `contested`/`wood`/`stone` instead of `resource`/`blue_bg`/`barrier` —
   silent because the wrong indices were still *in bounds*.
6. `test_gradient_flow.py` constructing networks with zero args, testing dataclass defaults
   (`env_channels=9`) rather than production shapes — a test structurally unable to fail.
7. `n_actions` defaulting to 8, silently building an 8-action model against a 12-action
   checkpoint.
8. `cross_attn_enabled` documented as live on Modal, `false` at every commit in history.
9. **`n_minibatches = M // minibatch_size` with no floor** — at M=160, mb=512 this is **zero
   minibatches**, so the PPO loop body never executes and the log still prints
   `Blue PPO done in 0.1s`. Smoke tests "validating" PPO validated nothing.

Nine instances, one shape. Assume it is happening again.

**The checklist, applied before reporting any result:**

- **Iteration counts.** Assert the loop body ran at least once. `n_minibatches == 0` must raise,
  not skip. Same for empty batches, zero-length corpora, zero eligible records in a decode.
- **Coefficients.** A term multiplied by zero is dead. Print the effective coefficient next to
  the loss, and alert when a live consumer reads a head whose trainer is disabled.
- **Gradients.** Prove `‖∂loss/∂param‖ > 0` on real data at production shapes. Reading the code
  is not proof; `tests/test_severance_sweep.py` is the pattern.
- **Effects, not absence of errors.** Measure a parameter delta, a record count, a changed file
  size. "No exception" is not a result.
- **In-bounds is not correct.** An index arithmetic error that lands inside the right block is
  the hardest kind to see. Derive offsets from `make_obs_layout`, never by hand.
- **Build fixtures from production config.** Never zero-arg constructors, never hardcoded
  shapes. If a test cannot fail when the live layout changes, it is decoration.

**Fix now (from instance 9):** make `ppo_update` raise a clear error when `n_minibatches == 0`
rather than silently skipping, and clamp or report the minibatch size when `M < minibatch_size`.
Then re-run every smoke test that has "passed" PPO at small population — those results are void.

---

# `dead_streak` placement — verify before committing

Making the counter `float32` to get past `jax.grad` is the right immediate unblock, but a
non-learnable counter living inside the params pytree is a structural oddity of exactly the kind
that produces instance ten. Before committing, confirm and state plainly: does the optimizer
apply updates to `dead_streak`, and does the dead-code reset's write survive the optimizer step,
or can the two race? If it cannot be cleanly excluded from optimiser updates, move it to a
separate persisted state tree rather than leaving it in `params`. Either way, pin the answer with
a test — `tests/test_dead_code_usage_ema.py` is the right home and the 8 checks there are good
work.

---

# README rewrite (M's request)

Rewrite the repository README. Current file is ~21 KB and reads as an accreted lab notebook.
Target: **scientific, minimalist, roughly 150 lines.** Someone competent should understand the
bet, the mechanism and the current honest status in ninety seconds.

Structure: one-sentence statement of the thesis (meaning that is causally earned cannot be faked;
an LLM's symbols are grounded only in other symbols, a THRONG token is grounded in whether the
agent that hears it survives). Then the mechanism in a short paragraph plus one diagram —
partially blind prey, lethal predation, a 3-slot discrete VQ channel, receiver-necessity recipe
asymmetry. Then current status, stated honestly, including that the one significant causal result
is unreplicated. Then how to run it. Then a pointer to `docs/` for everything else.

Rules for it: no emoji, no badge wall, no phase-by-phase history (that is `THRONG.md`'s job), no
roadmap promises written as achievements, and **no claim that is not currently true of the code**
— `cross_attn_enabled` is false, blue carry is an EMA, the confidence gate is being repaired.
Prose over bullet soup. One diagram maximum, and only if it shows the actual signal path. Assume
the reader is a researcher deciding whether this is serious work.
