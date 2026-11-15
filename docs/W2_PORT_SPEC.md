# W2 port spec: stage 1 on the toy's recipe (DRAFT for Cam; one page; runs on new credits)

Nov 15, Will. Pending: T3 seeds 1-2 (seed 0 passes at 20M). Toy evidence: protocol, Nov 14-15 "TOY" entries.

## What the toy established

- Following a FIXED signal (T1 oracle, T2 fixed code) under the harness move cost needs two things:
  - **minibatched PPO**: 8 x 4 instead of 2 full-batch epochs, which gave only ~244 gradient steps per 1M samples;
  - **a mixed near-to-far curriculum**: each episode's distance drawn from 1 up to a frontier that rises at >= 80%
    success.
  Without the curriculum PPO freezes. A curriculum that trains only at its frontier over-specialises.
- Learning a NEW code within each life (T3) needed three more things together:
  - an **episode-level memory**: one raw record per past bout (token heard, where the meal was found relative to
    the hearing place, ate) with a learned read. A GRU, a 128-step transformer and a life-credit GRU all failed;
  - the **approved hindsight drive** "where my next meal was", trained ONLY at the moment of hearing. Trained at every
    step, it is solved from self-motion and teaches nothing;
  - a **demonstrator on the first hearing** of each token in a life.
  - Seed 0 at 20M: second half of life 100% (shuffled 3%, wiped 1%); head 16% -> 98% from the 1st to later hearings.
    Without the demonstrator there is no signature.

## The W2 design

- **Learner:** the harness agent with its trunk UNFROZEN, navigator-initialised. KL 0 (the toy's L1u passed T1 / T2 at
  KL 0). Minibatched PPO 8 x 4, entropy 0.01, the NaN-safe ratio, the in-run forage stop.
- **Memory:** an episode-level table per life, one record per HEARING BOUT:
  - the token heard and the speaker offset at the hearing;
  - the displacement from the hearing place to the NEXT meal (written when that meal happens);
  - whether a meal followed within N steps.
  This is the counting observer's evidence B, kept raw, with a learned key / value read over 64-128 slots.
- **Hindsight drive** (approved): at each hearing step only, predict the sector of the next meal from memory; its
  softmax feeds the policy. Weight 0.5.
- **Demonstrator = the parent arm:** on a juvenile's FIRST hearing of a token in its life, its scripted parent walks
  to the food the token names, and the young-seek reflex keeps the juvenile close. Later hearings: no demonstration.
- **Near-to-far curriculum mapped onto development:** the named food is a personal item placed at distance d from
  the juvenile when its speaker calls. d is drawn from 1..frontier; the frontier rises when the cohort's success over
  the last 500 bouts is >= 80%, cap 24. Adults forage the shared world as now.
- **Necessity:** named items are visible only when ADJACENT, so the code is needed. Premiums are checked against the
  strongest code-free policy (v2-best), never the navigator.
- **A random code per life** (the existing family split; held-out families for CONFIRM).

## Order and bars

1. **Oracle arm first** (standing rule): the same harness, the oracle sector of the call as input. Pass = the
   registered hungry life test >= 50%, Wilson LB > 10%.
2. **Learner arm:** the T3 bar transplanted to W2:
   - reach of named items in the second half of each life >= 90% of the scripted ideal in-context learner;
   - >= within-life shuffled + 30 points, and >= 2x it;
   - 3/3 seeds;
   - the in-context signature in the head reported (1st vs 2nd+ hearing).
3. Then the frozen life test on held-out code families (CONFIRM).

## Cost (to be replaced by a 2-minute price probe)

- The toy needed ~20M agent-steps to pass. W2 lives are longer and richer; budget ~1e8 steps per arm.
- At ~20k steps/s on a 4-CPU container at ~$0.35/h: ~1.4 h, ~$0.5 per arm-seed.
- Oracle + learner x 3 seeds: **~$3**, plus readouts ~$1. Laptop screens first (all free).
