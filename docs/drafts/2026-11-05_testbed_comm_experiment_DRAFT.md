# DRAFT -- communication experiment in the learner testbed at 32x32 (Cam, 2026-11-05). Not launched.

**Question.** In a world where navigation is learnable (32x32 with C'), does a signal from an agent who knows
which hearth is active make an agent who doesn't know go there -- *because of the signal*, not because it follows
the informed agent?

This replaces the earlier full-world comm draft (`2026-11-04_cam_tasks_DRAFT.md`, Task 2). The measurement
design carries over; the world shrinks to the testbed, and the channel fix Cam approved is built in.

## World (testbed, extended)
- **32x32 torus, the four production hearths** (`generate_hearth_positions(32)`: (8,8), (8,24), (24,8), (24,24)).
  Exactly **one is active**; a deposit (Craft within radius 1 of the active hearth) pays 2.7 to the depositor,
  respawns the depositor, and **re-rolls the active hearth** uniformly among the other three. Also re-rolled every
  R = 256 steps with no deposit, so the target cannot be memorised as a place. A0 rule otherwise (any material).
- **Two agents per world, one shared network** (parameter sharing, as production): a **sender** (informed: its 4
  `hearth_need` dims carry one-hot of the active index; `can_see_recipe` = 1) and a **listener** (those dims 0,
  `can_see_recipe` = 0). Both always see all four `hearth_rel` vectors. The informed fraction is 50% by construction
  (frozen, satisfies the earlier ruling). 32 worlds = 64 agents per batch.
- **Wire:** each agent's 40-dim broadcast fills the other agent's first neighbour-signal slot (`nb_sigs`, the
  production `emb_nb` read path); remaining neighbour slots zero. Alarm head unchanged.
- **Comms subtree trains:** `freeze_comms=False` (the testbed currently passes True, the stage-0 freeze).

## Channel fix (Cam's default; code on a new branch, tested before any spend)
1. **Token as a policy action.** Per slot s: logits_s = -dist_sq_s / tau, `tok_s ~ Categorical(logits_s)` in the
   rollout, broadcast `codebook_s[tok_s]`. `token_log_prob = sum_s log softmax(logits_s)[tok_s]` joins the joint
   log-prob in the rollout and in `ppo_loss`'s ratio, the way the alarm log-prob does. Small token entropy bonus.
   tau fixed so that initial token entropy is about half its maximum (measured at init, registered).
   (This is also what wires `gumbel_tau`, one of the allowlisted pending keys.)
2. **Shared reward:** when the listener deposits at the active hearth, the sender in the same world gets
   `reward_listener_reach` = 2.7 (equal to a deposit).
3. **Tests first** (`tests/test_token_policy_gradient.py`, from the approved design): nonzero gradient to
   `head_signal_slot*` from the shared reward through the real `ppo_loss`, raising the rewarded token's log-prob;
   mutant 1 (token log-prob dropped = today's design) and mutant 2 (shared reward zeroed) must both fail.
   Apparatus guard in the loop: token entropy > 0; stored token == broadcast code index on 100% of steps (halt
   otherwise); listener target dims exactly 0, sender one-hot == active index on 100% of steps (halt otherwise).

## Arms (2 seeds each; estimator C' -- bootstrap; critic = PopArt if stage 3 fixes it, else the ladder's fix)
- **COMM:** channel fix on, wire live.
- **ABLATED:** identical, but the listener's received signal is zeroed (`hearth_channel_ablation_enabled`
  semantics) for the whole run -- what a listener achieves with no channel (it can still search the four hearths,
  or follow the sender).
- N = 150 updates (comm needs both navigation and a convention; 80 was enough for navigation alone at 32).

## Measurements (to be fixed in the registration before launch)
- **PRIMARY -- causal swap on listener movement (offline, frozen COMM final policy).** >= 10 paired eval rollouts,
  same seed and initial state. In the swap arm the listener receives the token its sender *would* emit with the
  active index replaced by T' (a fixed derangement, T' != T); the sender itself still acts on the true T, so
  "follow the sender" is unchanged between arms. Statistic: Delta = E[m.u toward hearth T'] (swap) - (control),
  listener moves only, same +/-1 scale as the red-flee control. **Bar: Delta >= 0.05 and > 2x SE across rollout
  pairs (never records).** Mirror check: approach toward the true T falls in the swap arm.
- **Secondary -- behaviour:** listener deposit rate, COMM vs ABLATED, last 10 updates, both seeds; COMM > ABLATED
  outside seed spread. (A behavioural difference cannot by itself separate "reads the signal" from "follows the
  sender better" -- that is what the swap is for.)
- **Secondary -- M1:** I(token; T) per slot over sender steps, Miller-Madow corrected, block-permutation null over
  rotation epochs (1,000 shuffles); pass = above the null's 99th percentile in at least one slot.
- **Interpretation (as in the earlier draft):** primary passes -> the channel steers an uninformed agent to the
  place it names (Gate 2, strong form). M1 passes, primary fails -> information is in the tokens but listeners
  don't use it. Both fail -> report; next is Cam's call.

## Cost (testbed measurements: ~18 s/update and ~$0.45 per 80-update run at 32 envs on A10/A10G)
- 64 agents per batch doubles the PPO minibatches: ~35 s/update -> 150 updates ~90 min -> **~$1.6 per run**.
- 4 training runs (COMM, ABLATED x 2 seeds): ~$6.4. Swap eval + M1 (rollout-only, from saved params): ~$0.5.
  Modal smoke of the new code (2 updates): ~$0.3. **Total ~$7-8** against the ~$12 expected to remain after
  stage 3, leaving ~$4-5 for one re-run.

## Work before any spend (CPU)
Branch from critic-fix (or master + the chosen critic once validated): token-policy sampling and log-prob in
`network_jax`/`ppo_loss`; the two-agent active-hearth world in the testbed (production hearth positions, re-roll
rule, wire, freeze_comms off, shared reward); the swap-eval mode; the gradient test + mutants; the apparatus
guards; a local 1-update smoke at 2 worlds (light on RAM, as stage 3's smoke was).
