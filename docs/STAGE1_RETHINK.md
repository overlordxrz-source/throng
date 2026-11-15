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

**Update after v2 at 50M (FAIL, 1.069x):** v2's best code-free forager eats MORE over a life than the perfect listener
(35.7 vs 22.7, the same 128 lives). Against it the adult-food premium is only ~1.2x. So in this world a competent
forager barely needs the code, which is Cam's "world is the bottleneck" branch. Route (b) addresses exactly this: it
makes the named food matter more than general foraging skill.

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

## Toy grid results (Nov 14, laptop, comm-toy `dd9299d`; Cam's spec plus addenda 5-6)

Cells read as reach-food rate (%) true / shuffled / silence within the 4d cap. `*` = learns (true >= 80% AND >= 2x
shuffled AND >= +30 points). 1M agent-steps per cell, seed 0. L3 = evolution (lambda does not apply to it, so the paired
lambda rows coincide, which also confirms determinism).

**Tier T1 (oracle input), the harness reward (move cost 0.003):**

| d | food | lambda | L1 KL0.0 | L1 KL0.3 | L2 | L3 |
|---|---|---|---|---|---|---|
| 3 | hidden | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 | 36/4/7 |
| 3 | hidden | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 | 36/4/7 |
| 3 | visible | 0.95 | 70/69/70 | 9/10/9 | 0/0/0 | 62/42/28 |
| 3 | visible | 0.99 | 70/70/72 | 9/10/10 | 0/0/0 | 62/42/28 |
| 6 | hidden | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 | 16/9/7 |
| 6 | hidden | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 | 16/9/7 |
| 6 | visible | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 | 18/18/18 |
| 6 | visible | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 | 18/18/18 |
| 12 | hidden | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 | 14/12/12 |
| 12 | hidden | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 | 14/12/12 |
| 12 | visible | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 | 15/12/12 |
| 12 | visible | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 | 15/12/12 |
| 24 | hidden | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 | 9/14/12 |
| 24 | hidden | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 | 9/14/12 |
| 24 | visible | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 | 11/12/6 |
| 24 | visible | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 | 11/12/6 |

**Audit, the same grid with the move cost set to 0** (not the registered grid; run because Cam's "neither learns at
d = 3 -> audit PPO" fired):

| d | food | lambda | L1 KL0.0 | L1 KL0.3 | L2 |
|---|---|---|---|---|---|
| 3 | hidden | 0.95 | 0/0/0 | 0/0/0 | 90/10/23* |
| 3 | hidden | 0.99 | 0/0/0 | 0/0/0 | 86/10/23* |
| 3 | visible | 0.95 | 71/70/70 | 10/11/10 | 97/36/68* |
| 3 | visible | 0.99 | 72/70/70 | 10/11/11 | 95/32/63* |
| 6 | hidden | 0.95 | 0/0/0 | 0/0/0 | 7/3/4 |
| 6 | hidden | 0.99 | 0/0/0 | 0/0/0 | 7/4/3 |
| 6 | visible | 0.95 | 0/0/0 | 0/0/0 | 13/6/9 |
| 6 | visible | 0.99 | 0/0/0 | 0/0/0 | 13/6/5 |
| 12 | hidden | 0.95 | 0/0/0 | 0/0/0 | 1/0/1 |
| 12 | hidden | 0.99 | 1/1/1 | 0/0/0 | 0/0/0 |
| 12 | visible | 0.95 | 0/0/0 | 0/0/0 | 0/0/1 |
| 12 | visible | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 |
| 24 | hidden | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 |
| 24 | hidden | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 |
| 24 | visible | 0.95 | 0/0/0 | 0/0/0 | 0/0/0 |
| 24 | visible | 0.99 | 0/0/0 | 0/0/0 | 0/0/0 |

**Extra seeds on the boundary cells** (true / shuffled, seeds 0 / 1 / 2):
- L2, move cost 0, d = 3 hidden: 90/10, 88/12, 89/13 (learns 3/3).
- L2, move cost 0, d = 6 hidden: 7/3, 15/4, 6/2 (0/3).
- L3, harness, d = 3 hidden: 36/4, 26/22, 15/14 (0/3; oracle use clear only in seed 0).
- L3, harness, d = 3 visible: 62/42, 58/41, 49/29 (0/3).

**Controls:** a scripted ray-walker reaches 100% with the true input in every cell tested (d = 3, 24; visible, hidden).
Shuffled input drops it to 0-3% wherever the food starts out of view.

**Tiers T2 / T3 did not run:** no learner passed T1 in any cell under the harness reward, so no cell is gated
through. L2' (the transformer memory, T3 only) is therefore not built.

### Reading against Cam's rules

1. **"Neither learns at d = 3 in T1 -> audit PPO first": fired, and the audit clears the code.** The same PPO code
   learns d = 3 (hidden 90%, visible 97%) once moving is free. With the harness's move cost and a sparse payoff,
   PPO from scratch freezes: stay / eat win the advantage. That is the stage-1 pathology reproduced at the smallest
   scale.
2. **"L2 learns where L1 doesn't -> the frozen-trunk architecture is a blocker": yes, under free movement.** L1 (frozen
   navigator trunk + side head) fails every hidden cell even at move cost 0, at KL 0 and 0.3. The navigator itself
   reaches visible food 3 cells away only 1.4% of the time.
3. **d\*:** under the harness reward there is none (nothing learns at d = 3). With free movement, L2's d\* = 3: it
   fails at d >= 6 within 1M steps, because random exploration rarely reaches 6+ cells inside the cap.
4. **lambda 0.99 vs 0.95 changes nothing** anywhere. Hidden vs visible matters only at d = 3 (visible food is in view
   from the start).
5. **Evolution (L3)** is the only learner that improves under the real move cost: d = 3 visible 1.4% -> 49-62%. It
   shows some oracle use at d = 3 hidden in one of three seeds (36/4). It is far from the bar and does nothing useful
   at d >= 6 within the budget.

**Implication for the world and learner (for Cam's ruling):** the move cost, together with a payoff out of random
reach, defeats PPO before distance or necessity even enter. Any stage-1 world has to make the first step toward
named food pay within reach. That means a very near start (d <= 3), with near-free movement or a denser payoff early,
then lengthening. And the learner should not be locked to the frozen navigator trunk.

### T3 (random code per life): FAIL for every memory (Nov 15, comm-toy `54c72b0`)

| memory (T3, seed 0, mixed curriculum, harness cost, 3M steps) | max d | at its d: true 1st->2nd half / within-life shuffled / wiped | d = 12: true / shuffled / wiped (2nd half) |
|---|---|---|---|
| ideal learner (scripted ceiling) | - | d 3: 83->100 / 34->35 / 38->41; d 2: 95->100 / 76->78 | 100 / 37 / 41 |
| L2 GRU | 4 | d 4: 37->37 / 37->38 / 37->38 | 0 / 0 / 0 |
| L1u (unfrozen navigator, its GRU) | 3 | d 3: 31->31 / 31->31 / 37->40 | 0 / 0 / 0 |
| L2t transformer (128-step window) | 5 | d 4: 0 / 0 / 0 | 0 / 0 / 0 |
| L2 GRU, life-level credit + 256-step BPTT (diagnostic, B 64) | 3 | d 3: 32->33 / 32->32 / 31->32 | 4 / 3 / 6 |
| L2e episode-level memory (64 slots) | 4 | d 2: 85->88 / 85->87 / 86->88; d 3: 60->64 / 60->63 / 60->64 | 0 / 0 / 0 |

- **No memory shows the in-context signature.** First half = second half = the controls everywhere. Every
  learner's curriculum stalls at d = 3-5.
- **The mechanism.** Where the code is unnecessary (d <= 2), code-free search already succeeds 85-88%, so there is
  nothing to learn. Where it becomes necessary (d >= 3), no learner bootstraps the find-then-recall loop; the ideal
  learner shows the loop is worth ~+60 points.
- **Not the cause:**
  - the credit setup: life-level credit with long backprop changes nothing;
  - a memory that cannot see far enough back: the episode-level memory holds every past episode's raw record and
    still does not learn to read it.
- What T1 / T2 showed works: following a fixed signal (oracle or fixed code). What fails is inferring a NEW code
  within a life from sparse successes, under PPO, in 3M steps.
