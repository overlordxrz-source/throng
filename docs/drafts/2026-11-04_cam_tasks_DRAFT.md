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

## Cam's rulings (2026-11-04) and the channel-learnability check

Rulings: (1) the comm experiment does not launch until the channel can learn; (2) freeze the informed fraction for the
test window; (3) causal swap stays primary; (4) red-catch slope ~6.5 vs 3.0 logged as OPEN, checked after 2763,
running pin untouched. Budget: ~$8 this window + ~$15 comm run leaves ~$15 for one fix-or-replicate run.

### (a) Does the token's log-prob enter the PPO loss? -- NO.
- The token is not sampled: `vector_quantize_signals` (`network_jax.py:64-100`) takes `token_ids = argmin(dist_sq)`,
  the deterministic nearest code. (The comment says "Gumbel Softmax", but no Gumbel noise is drawn; `y_soft` is a
  plain softmax used only as the straight-through surrogate.)
- `rl_jax.ppo_loss` builds the ratio from `log pi(action) + log pi(alarm)` only (`rl_jax.py:126-133`); no token
  log-prob exists anywhere.
- The token heads (`head_signal_slot0/1/2` on the `gwt_comms_1` branch, `network_jax.py:509-516`) are trained by
  exactly two losses: the VQ codebook/commitment loss (`vq_coef * loss_vq`), and the VQEL reconstruction loss, where
  `z_q` (straight-through) must reconstruct the sender's OWN egocentric spatial patch (`head_vqel_recon_1/2`,
  `network_jax.py:590-591`, `recon_coef` 1.0). Neither target contains the active hearth, which sits in the own-state
  block. **So under the current design the tokens have no pressure at all to carry T: a null would be guaranteed,
  as Cam said.**
- The listener side is already trainable: incoming neighbour signals enter through `emb_nb` into the policy trunk
  (`network_jax.py:319, 464`), which PPO trains.

### (b) Does any sender reward depend on what a listener does? -- NO.
Blue's reward (`main_jax.py:1112-1149`) is alive + energy + catch penalty + resource + move + puzzle + contested +
alarm penalty + futile-craft + **own** hearth deposit + **own** shaping + completion share. The completion share
goes only to agents who themselves deposited (`hearth_completion_share`, proportional to contribution). Nothing pays
an agent for another agent's arrival or deposit.

### Smallest fix (Cam's default, made concrete) -- DRAFT, lands after 2763 on a fresh pin
1. **Token as a policy action.** Per slot s in {0,1,2}: logits_s = -dist_sq_s / tau (tau a fixed config value, e.g.
   chosen so the initial token entropy is about half its maximum; `dist_sq` is unbounded, so tau=1 could be
   degenerate either way). Sample `tok_s ~ Categorical(logits_s)` in the rollout; broadcast `codebook_s[tok_s]` exactly
   as now. Store `token_log_prob = sum_s log softmax(logits_s)[tok_s]` and add it to the joint log-prob, the same way
   the alarm action is added (rollout `b_joint_log_probs_taken`, `ppo_loss` ratio). Small entropy bonus on the token
   distribution (`token_ent_coef`), like `alarm_ent_coef`. The VQ commitment loss stays; the VQEL recon coefficient
   is held as is (one change).
2. **Shared reward when a listener reaches the active hearth.** On each step a listener (`~can_see_recipe`) deposits
   at the active hearth, add `reward_listener_reach` (e.g. equal to `reward_hearth_deposit`) split equally among all
   living senders (informed agents). Global sharing is the smallest version; a locality-credited version (only
   senders who were among that listener's K signal-neighbours in the last L steps) needs a neighbour ring buffer and is
   deferred unless the global one fails.
3. **Fingerprint:** append `token_policy_enabled`, `token_tau`, `reward_listener_reach` (how actions are chosen and
   what is rewarded -> designed reset).
4. **Freeze the informed fraction (ruling 2):** `can_see_recipe_fraction_fixed` (e.g. 0.25), assigned at birth by a
   fixed Bernoulli instead of inheritance+mutation for the test window; appended to the fingerprint. Dashboard prints
   the live informed fraction every update, and the run halts if it leaves 0.25 +/- 0.03.

### Regression test (the proof the channel can now learn) + mutant
`tests/test_token_policy_gradient.py`, CPU, tiny shapes (not the full smoke -- that exhausted the Mac's RAM):
- Build a minimal batch: a sender emits a token at t; at t+1 a listener's only reward is `reward_listener_reach`,
  credited to the sender per item 2; all other rewards zero. Run the real `ppo_loss` (not a hand-built one) and take
  `jax.grad` with respect to the `head_signal_slot*` parameters.
- **Assert:** that gradient is nonzero (L2 norm > 1e-6) and its sign raises the log-prob of the rewarded token.
- **Mutant 1** (token log-prob dropped from the ratio, i.e. today's design): the same gradient is ~0 apart from the
  VQ term; the test is run against it and must FAIL. Hold the VQ/recon coefficients at 0 in the test so the only
  possible gradient source is the reward path.
- **Mutant 2** (shared reward zeroed): the gradient from the reward path vanishes; the test must FAIL.
- Plus an apparatus check in the loop: with token_policy on, the logged token entropy is > 0 and the stored token
  equals the broadcast code index on 100% of alive-steps (halt otherwise), mirroring the gate guard.

### Cost (unchanged envelope)
The fix itself is CPU work. It replaces nothing in the comm-run budget (~$15). If the channel-learnability fix needs
its own verification run before the comm run, that uses the remaining ~$15 fix-or-replicate slot.
