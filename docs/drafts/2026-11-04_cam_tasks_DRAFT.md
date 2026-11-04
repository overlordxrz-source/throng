# DRAFT -- Cam's two CPU tasks (requested 2026-11-04, to be reported after ppo 2763)

Status: written while the gate-off ablation runs (Modal `ap-R21DLvT8DELG11wxilbRam`, pin `bcaf1ce`).
Nothing here touches the running pin or launches anything.

## Task 1 -- September ecology defects, checked against the current pin (`jax_sim/` identical at `bcaf1ce` and HEAD)

- **flint/clay/vine grids are written -- FIXED.** Code: `jax_sim/main_jax.py:951-961` spawns each every step
  (`bernoulli(regen_rate*0.5) & material_zone_masks(gs)[m]`) and writes it into `grid` (previously zero-init in
  `GridState.__init__` with no producer, WILL_RESTART #10). Runtime: corpus 2707-2735 (103,812 sampled agent-steps)
  shows agents holding flint 8,051 / clay 8,143 / vine 8,138 times (wood 27,644, stone 24,549), impossible if those
  grids were never written. (The 1:3 ratio vs wood/stone is zone geometry + identical regen, not checked further.)
- **reward_red_catch is live -- FIXED.** Code: `main_jax.py:1160` `r_rew = r_rew + _reward_red_catch *
  r_caught_any` with `reward_red_catch: 3.0` in config (previously read and never applied, Rule 13 #11). Runtime:
  over 2707-2735 red's per-update reward mean tracks blue_caught at r = 0.95, slope ~6.5 reward per catch -- about 2x
  the 3.0 config value, consistent with more than one red being credited per caught blue (`r_caught_any` is per
  red); the multiplicity was not separated out.

## Task 2 -- communication experiment for the PASS branch (draft; launches only after 2763 is read)

**Question.** With navigation learnable (the pass branch), does a signal from an agent that knows the target change
where an agent that doesn't know it goes -- because of the signal, not because it follows the informed agent?

**World (one treatment relative to the passing run; resume from its final checkpoint).**
- `hearth_active_mode`: exactly one of the 4 hearths is *active*; deposits are accepted only there, any material (A0
  rule), so the only information needed is WHICH hearth. Active index T in {0..3} re-rolls uniformly among the other
  three on each completion, and at least every R = 256 steps, so the target keeps rotating (A0's rotating-need
  regime) and cannot be memorised as a location.
- **Senders** = informed agents (`can_see_recipe`): their existing 4 masked obs dims (today `hearth_need/4`) carry
  one-hot(T). **Listeners** = everyone else: zeros. Everyone keeps seeing all 4 hearth positions (`hearth_rel`).
- Shaping (beta = 24) points at the ACTIVE hearth for holders, senders and listeners alike. Listeners cannot see T,
  so the shaping gradient rewards them for using any predictor of T they have: the signal, or following a sender.
- **Needs Cam's ruling (would be a second change):** freeze the informed fraction at a fixed value (e.g. 25%).
  Today it is an inherited trait with 3% mutation, so it drifts, and drift in the sender:listener ratio confounds
  every metric below.
- Fingerprint: append `hearth_active_mode`, `hearth_active_rotate_steps` (and the informed fraction if frozen) ->
  designed tripwire/curriculum reset on resume.
- Corpus: append a `hearth_active` field to each blue record, so MI can be computed per sender record.

**Structural caveat, stated up front.** Signal tokens come from a straight-through Gumbel quantiser inside the
sender's own forward pass (`network_jax.py:81-98`); once broadcast they are plain observation data for listeners --
no gradient flows from a listener's outcome back into a sender's token choice, and nothing rewards a sender for being
understood. Tokens will carry T only to the extent the sender's own internal state encodes T and quantisation
preserves it. That is why the PRIMARY is the causal listener test, and why M1 = 0 is a finding about senders, not
a failure of the test.

**Measurements (to fix before launch):**
- *Apparatus, update 1:* listener target dims exactly 0 on every alive-step; sender one-hot == grid active index on
  100% of alive-steps. Enforced in code (halt), like the gate ablation's guard.
- *M1 -- signal->target MI (secondary).* I(token; T) per codebook slot (3 slots x 64 codes) over sender records,
  plug-in with Miller-Madow bias correction; ceiling 2 bits. Null: block permutation of T across rotation epochs
  (1,000 shuffles; epochs, not records, because tokens and T are autocorrelated within an epoch). Pass = above the
  99th percentile of the null in at least one slot.
- *M2 -- causal swap on listener movement (PRIMARY).* Offline evaluation from the final checkpoint, no training:
  n >= 10 paired rollouts with the same seed and initial state. In the swap arm, every sender's broadcast token is
  replaced by the token that same sender emits when its T input is replaced by T' = a fixed derangement of T
  (T' != T), so listeners receive a coherent message pointing at the wrong hearth. Listeners are restricted to
  those with no informed agent within radius 5 in the last 50 steps (the existing spatial-following control).
  Statistic: Delta = E[m.u toward T'] (swap) - E[m.u toward T'] (control), m.u on the same +/-1 scale as the
  red-flee control. **Primary bar: Delta >= 0.05 AND Delta > 2 x SE, SE across rollout pairs (never records).**
  The 0.05 is anchored to the strongest directional effect this project has measured (red flee at d<=1: -0.07). The
  mirror check: approach toward the true T should fall in the swap arm.
- *Controls:* channel-ablated evaluation (wire zeroed, `hearth_channel_ablation_enabled`) -> listener approach to T
  at its no-signal level; the same statistics for senders (who should not care about the swap).

**Interpretation, fixed now:**
- M2 passes -> a signal causally steers an uninformed agent toward the place it names: the channel has a job (Gate 2
  in its strong form).
- M1 passes, M2 fails -> information is in the tokens but listeners don't use it.
- M1 fails, M2 fails -> senders don't encode T in tokens; next step is the sender-incentive question (shared reward
  or a trainable channel) -- Cam's call, a separate design.
- M2 passes with M1 marginal -> trust M2 (causal beats correlational); report both.

**Work before any spend (CPU):** active-hearth mode in `resolve_hearth_deposits` + observation builder, the corpus
field, the fingerprint fields, the swap-eval script (sender re-run with T' injected, swapped tokens broadcast to
listeners only), unit tests (target dims zero for listeners, swap produces T'-consistent tokens, deposits accepted only
at the active hearth), and a smoke on Modal CPU rather than locally (the local smoke exhausted the Mac's RAM).

**Cost against ~$30 remaining after this window** (measured on this run: ~3.5 min/update on A100-80GB, ~9 min
startup; A100-80GB $2.50/h plus CPU/memory, ~$3.2/h observed so far, billing may lag):
- training, 40 updates: ~2.4 h, ~$8; or 60 updates: ~3.6 h, ~$12
- swap evaluation, ~10 paired rollouts, rollout-only (no PPO): ~0.5 h, ~$2
- Modal CPU smoke and preflight: ~$1
- **total ~$11-15**, leaving ~$15 of margin for one crash/restart. To be re-estimated from this window's final billed
  cost before launch.
