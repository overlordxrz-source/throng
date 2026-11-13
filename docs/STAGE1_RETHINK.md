# Stage 1 rethink (draft for Cam; Nov 13, Will)

Stage 1 asked newborns to learn, within a life, to act on a code that names where food is. Every learner so far has
failed to ACT on the code, even when handed the answer. This note says what is known, gives three routes with costs,
lever checks and falsifiers, and makes a recommendation. Numbers come from `docs/RESEARCH_PROTOCOL.md` (Nov 12-13).

## What we know

1. **The world pays for listening.**
   - Against the best evolved forager (D-best), the scripted true-code follower eats **2.18x** as much as an adult.
     Its hungry life-test food is 22.2 vs 2.2, and it passes 62/62 vs 2/63.
   - Against the navigator: 3.0x adult food; survival past 1,300 2.3x.
2. **The objective sees it per state.** Under the training gamma 0.99, the follower's return-to-go beats the
   navigator's at ages 0-499, 500-999 and 1,500+. From birth it does not (-0.05), because the payoff lies beyond the
   ~100-step horizon.
3. **The memory holds heading.** A move-class probe on the GRU reads 69.8% with memory kept vs 57.0% wiped. The
   code re-draw shows a code-dependent part. A fixed KV table matches the counting observer on exact sector (1.11x)
   but not on heading (0.88x).
4. **PPO will not act, even on a perfect signal.**
   - Oracle arm: 1/87 pass.
   - KL screen (beta 0.3 silent-only, 0.1, 0.03): food ratio <= 1.05.
   - Hungry-adult life course (E): 1.005x.
   - Diagnosis: the policy almost never moves on heard steps. The few juvenile moves toward the signal get negative
     advantage (moving costs energy now; food comes later). PG outweighs the KL 2:1, so the KL is not the restraint.
5. **Evolution reshapes the policy but ignores the code.**
   - D (lifespan fitness): adult food 1.47x the navigator; lifespan saturated at the cap.
   - v2 (food fitness, body + head; 25M): adult food 3.05x the navigator, life-test food 5.1, but true vs deranged
     only 1.03x.
   - Selection moves the policy far (KL ~5 nats) where PPO barely moves it (KL <= 0.08).

**The common thread: partial following does not pay.** The food a token names is usually out of sight and many steps
away. A first step toward it earns nothing until the agent arrives, and moving costs energy at once.
- PPO's credit (GAE horizon ~17 steps) cannot connect the step to the meal.
- Selection sees following only as a small, noisy gain on top of strong code-free foraging. Its path from "ignore"
  to "follow" passes through policies that are no better.

## Three routes

**(b) Distance curriculum: make partial following pay.** Beacons announce food that is NEAR at first (within ~6-8
cells, a few GAE horizons away; just out of view), then farther as performance holds.
- Cost: world parameters plus a curriculum schedule. Laptop lever check; ~$0.3 Modal for an oracle arm.
- **Lever check (free):** at the nearest stage, the follower's per-state return under gamma 0.99 must beat the
  navigator's FROM BIRTH, and the follower / navigator food premium must be >= 1.5x.
- **Falsifier:** the oracle-input PPO arm at the nearest stage stays < 1.1x true/deranged at 25M. Then PPO cannot
  act even when the payoff is near, and the problem is the learner, not the distance.

**(a) End-to-end: unfreeze the trunk.** The frozen navigator trunk plus an additive side head may make it hard to
build a direction-specific move from a token.
- Cost: the same throughput as now; ~$3-5 per 1e9 steps; risk of forgetting how to forage early on (keep the KL to
  the navigator as a guard).
- **Lever check (free):** supervised. With the oracle sector as the label for "move toward it", an unfrozen network
  must reach >= 90% heading accuracy within 1M steps. This shows the architecture CAN represent following.
- **Falsifier:** the oracle-input PPO arm with an unfrozen trunk stays < 1.1x at 25M.

**(c) Evolution as the slow-weight learner, with energy-paid reproduction.** Lineages reproduce by spending gathered
energy, so fitness is offspring count paid in food: non-saturating, and it is the stage-2 rule.
- Cost: ~$0.6 per 50M at 32 lives per member; a population-level harness.
- **Lever check (free):** evolution on food from a noised head must recover the navigator's lifetime food. It failed
  at 16 lives per member (0.48x), so first rerun it at 64 lives per member.
- **Falsifier:** from D-best, 100M of evolution stays < 1.1x true/deranged.

## Recommendation and odds

**Do (b) first, with the oracle arm, then decide between (a) and (c).** It is the cheapest decisive test. It attacks
the one mechanism both learners share (partial following does not pay), and the oracle arm separates "can't act"
from "can't learn the code".

Odds (my estimates):
- (b) oracle arm passes at the nearest stage: **0.45**; the full curriculum then reaches today's distances: 0.2;
- (a) unfrozen trunk alone passes the oracle arm: 0.25;
- (c) energy-paid evolution reaches >= 1.5x within ~$1: 0.2.

Combining (b) with (c) (a curriculum under selection) is plausible later. The odds of the full stage-1 goal on this
account's remaining credit are low.

## Things we may be missing

- **The life test may penalise good foragers.** A strong code-free forager eats well under true and deranged codes
  alike, so its ratio stays near 1 even if it uses the code a little. The follower's 2.46x comes partly from being
  MISLED by deranged codes. A paired metric is worth adding beside the ratio: food gain over the same policy with
  silence.
- **The oracle input is current-step only** (zero on silent steps). If a re-run fails, test a held "last heard" input
  before blaming the learner.
- **The cheapest decisive test might be a smaller harness:** one agent, one beacon, food at a fixed distance in a
  known direction, no parents. If PPO cannot learn "token -> walk there" in that toy in 1M steps, no stage-1 variant
  will work. It costs nothing on the laptop.
- **Instruments are now in place:** the oracle-first rule; the sanity rule with permutation null and cluster CI;
  per-item watches; the evolve resume fix.
