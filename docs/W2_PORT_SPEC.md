# W2 port spec: stage 1 on the toy's recipe (DRAFT for Cam; one page; pending T3 seeds 1-2)

Nov 15, Will. Toy evidence: `docs/RESEARCH_PROTOCOL.md`, Nov 14-15 "TOY" entries; `docs/STAGE1_RETHINK.md`.

## What the toy established

- **Following a fixed signal** under the harness move cost needs two things:
  - **minibatched PPO** (8 minibatches x 4 epochs; the old 2 full-batch epochs gave ~244 gradient steps per 1M
    samples);
  - **a mixed near-to-far curriculum** (each bout's distance drawn from 1 to a frontier that rises at >= 80% success).
    A frontier-only curriculum over-specialises.
- **Learning a NEW code within each life** (T3) needed three more pieces together:
  - an episode-level memory with a learned read;
  - the hindsight drive trained ONLY at the hearing moment;
  - a demonstrator on the first hearing.
  - GRU, transformer and life-credit GRU memories all failed. Without the demonstrator there is no signature.
- Seed 0 at 20M: second half of life 100% at d = 12 (within-life shuffled 3%, wiped 1%); the head goes from 16% on
  the 1st hearing to 98% on later hearings. Ablations (a) to (c) will say which pieces the port must keep.

## The W2 design

- **Learner.** The harness agent, trunk UNFROZEN (navigator-initialised), KL 0; minibatched PPO 8 x 4, entropy 0.01.
  The NaN-safe ratio (ratio-1 freeze), the in-run forage stop, milestone checkpoints.
- **The W2 "episode" = a hearing bout.** A record OPENS at each hearing (token, speaker offset). It CLOSES at the
  agent's next meal, with the displacement from the hearing place to the meal, or after T_max = 4 x the frontier
  distance with no meal.
- **Memory (M-approved inborn episodic memory, Nov 15).** 128 slots of these raw records per life; only the read is
  learned (key / value attention queried by the current input). A learned or evolved write replaces it later.
- **Hindsight drive (approved).** Trained ONLY on the hearing step: predict the sector of the next meal from memory;
  its softmax feeds the policy; weight 0.5. Never on later steps: there self-motion gives the answer away (toy lesson).
- **Demonstrator = the parent arm.** On a juvenile's FIRST hearing of a token in its life, its scripted parent walks
  to the named food; the young-seek reflex keeps the juvenile close. No demonstration on later hearings.
- **Curriculum mapped onto development.** The speaker names a personal food item placed at distance d from the
  juvenile, with d drawn from 1..frontier; the frontier rises when the cohort's last 500 bouts reach >= 80%; cap 24.
- **Necessity.** Named items are visible only when ADJACENT. A random code per life (the existing family split;
  held-out families for CONFIRM).

## Order, bars and checks (each step gated on the one before)

1. **NECESSITY CHECK** (free, laptop, the new W2 variant). Scripted follower vs the STRONGEST code-free policies
   (v2-best and the navigator) on the same lives. Pass = follower food >= 1.5x the best code-free policy.
2. **Oracle arm** (standing rule). The same harness with the oracle sector of the call as input. Pass = the hungry
   life test >= 50%, Wilson LB > 10%.
3. **Learner arm.**
   - Bar: second-half reach of named items >= 90% of a scripted ideal in-context learner, AND >= within-life shuffled
     + 30 points, AND >= 2x it, on 3/3 seeds.
   - The head's 1st vs 2nd+ hearing accuracy reported.
4. CONFIRM on held-out code families.

## Cost (replace with a 2-minute price probe first)

- The toy needed 20M steps; budget **1e8 agent-steps per run** in W2.
- At ~20k steps/s on a 4-CPU container (~$0.35/h): ~1.4 h, ~$0.5 per run, plus a readout ~$0.2.
- **Item caps** (each with its own watch at cap + 25%):
  - oracle arm $0.8;
  - learner $0.8 per seed (x 3);
  - CONFIRM $0.8.
- **Total ~$4.0.** Fits the ~$6.8 left under the $41 guard if Cam approves; laptop screens first.
