# DRAFT: real-network confirmation of positive signalling (for M to decide; nothing launches)

Status: drafted 2026-11-07 after the CPU-toy PS replication passed P1/P2/P3 (docs/RESEARCH_PROTOCOL.md).
Budget: comm-confirmation sub-cap **$12** of the ~$42 on the new Modal account (profile `rapidcrisp`).

## Question

Does the toy result hold on the real network (AgentNetworkJax, 32x32, four hearths)? Specifically: with the
positive-signalling nudge on the sender, do listeners come to depend on the code and act on its meaning? And
with the nudge annealed away, does the code keep itself alive?

## Design (one change from the final comm run)

- Everything as the final comm run (`comm-testbed` `ceadaf8`): ORACLE0 warm start, tau 0.08, token entropy coef
  0.01, C' + PopArt, vf 0.5, craft cost 0.1, 32 worlds, T = 512, the same swap apparatus.
- **One change: + PS**, i.e. c_ps x sum over slots of MI(token; active hearth), the batch estimate on sender
  rows, maximised.
  - Computed inside `ppo_loss` from the same forward pass (`outs.z_e`), so no extra forward.
  - Opt-in flag on the comm-testbed branch; production untouched.
  - The same role self-test and mutants as the toy.
- **c_ps** by the toy's registered rule (0.25 x ||grad PPO|| / ||grad PS|| at ORACLE0, 4 minibatches). It is
  computed free on CPU before registration.
- **Craft cost stays 0.1.** That makes the final run's COMM0/1 and NOSIG0 (same settings, 120 updates) direct
  baselines, so PS is the only difference. The toy used 0.45; neither value closes blind search.
- **Primaries (as the toy, both swap modes logged):**
  - P1: reliance. Deposits drop >= 30% under the empirical swap, paired > 2 SE, and PS listeners >= the NOSIG
    level.
  - P2: meaning. Delta >= 0.05.
  - Redirection (crafts at the swapped hearth) is a co-reported secondary, because the movement score misread 2
    of 8 toy seeds that did redirect.
  - P3 (anneal arm): MI >= 1.0 bit and P1, after the nudge has been off for a quarter of the run.
- **Halts:**
  - Replace the token-entropy floor with an **MI floor**: sender MI, mean over slots, >= 0.5 bit by update 20.
    The floor is set from a free local CPU smoke before registering. PS lowers entropy by design, so the old
    floor would halt a working run.
  - Keep the sender-competence halt and the spend guard.

## The cost problem, and the fix I recommend first

The real testbed steps its world in a Python loop. That costs **~37 s per update on an A10, ~$1.31/GPU-hour, so
~$1.6 per 120-update run** (final run: $4.8 for three).

The toy says listener competence takes time:

- At toy update 1,000, PS listeners were at 2.3-6.9 deposits per world-update and still climbing.
- By 2,000 they reached 7.5-8.7.
- A real update carries 2x the toy's samples, but the world is 2.7x wider. So plausibly **>= 500 real updates**
  are needed, which is **~$7 per run**.
- At that price $12 buys under two runs: no anneal arm, no fresh control.
- Cutting to 300 updates fits but risks a false negative from stopping too early.

**Recommended step 0 (free): port the real testbed's rollout to a jitted `lax.scan`, as the toy does.**

- Exact-equivalence tests against the current Python world come first:
  - obs builder: same state -> identical obs
  - dynamics: same actions -> identical positions, deposits, re-rolls, rewards
  - swap apparatus: unchanged check
- Then one timing run (~$0.10) on the new account.
- The toy's jitted rollout runs ~0.2 s per update on a laptop CPU, so a 10x+ speed-up on GPU is plausible but
  unmeasured.
- If it is ~10x, the full design below costs ~$4-5. If the port fails its equivalence tests, it is not used.

| plan | runs (updates) | est. cost | fits $12? |
|---|---|---|---|
| **With the scan port (if ~10x)** | PS x3, PS-anneal x2, NOSIG x1 (500 each) | ~$4-5 + timing | yes, with margin |
| Without the port, A | PS x2 (300), reuse final-run NOSIG0 as the control | ~$8.5 + 30% margin | yes; anneal untested; may be too short |
| Without the port, B | PS x2 + PS-anneal x1 (500) | ~$21 | no (would need the reserve) |

The estimates include a 30% margin: the last estimate ($3.7) came in at $4.8.

## Before any launch (all free or near-free, none started)

1. Create the `throng-runs` volume on `rapidcrisp` and upload `comm_ORACLE0_run_b88fe7f_a2.msgpack`.
2. Record a fresh billing baseline and arm a guard at baseline + sub-cap.
3. PS in `ppo_loss`, with self-test/mutants and the CPU calibration of c_ps.
4. Local CPU smoke to set the MI-floor halt.
5. (Recommended) the scan port with equivalence tests, then a timing run.
6. Write the registration (analysis script committed before results), and get M's approval of the spend.
