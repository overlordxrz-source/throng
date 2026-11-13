# Stage 1, next learner: key-value memory (design note for Cam; no runs)

Nov 13 2026, Will. This is the tree's next step after `s1kl03`. It needs new credits. Architecture only: no meaning is
fed in, and the code is never shown to the agent.

**Why.** The world and the objective both pay for listening:
- a perfect listener survives past 1,300 2x as often and eats 3.2x the navigator's adult food;
- the follower's per-state return under gamma 0.99 is higher in most age bands.

Cam's ruling makes the learner the leading suspect: a GRU must infer a new code within one life under PPO. A key-value
(KV) memory gives the agent a slot per token to collect evidence in, so in-life inference becomes counting rather than
recurrent dynamics. The calibrated plug-in counting observer, which is exactly a fixed counting table, passed the life
test at **70.6%** (leave-1000). That is the target this design approximates with learned reading.

## What is stored

Each life has its own table with **8 rows, one per token** (tokens are discrete, so keys are exact). It is reset at
birth. Each row holds running sums of evidence vectors and a count.

**Write rule.** The write is fixed, not learned: it is how the slot fills in-life with no gradient.
- When a token is heard, the agent records the token, the speaker offset (dx, dy) and its own age. It is an "open
  episode" for up to H = 64 steps.
- Later evidence is attached to that token's row with no hand-made 8-sector binning. Each item is a raw 2-D unit vector
  or a one-hot of the agent's own action space:
  1. **food seen:** when food is in view, the unit vector from the SPEAKER's position to the best food cell;
  2. **next meal:** at the agent's next meal, the unit vector from the speaker's position at hearing time to the meal
     cell (a hindsight write when the meal happens);
  3. **parent's next move:** a one-hot over 5 moves, while the parent is within view.
- Each row stores [sum of food-seen vectors (2), sum of next-meal vectors (2), parent-move counts (5), number of writes
  per source (3)]. That is 12 numbers per row, 96 per table.

These are the evidence sources A, B and c of the counting observer (`ideal_observer` / `observer_decs`), kept as
continuous geometry so that the 8 sectors are not built in.

## Retrieval

- Key = the current token (one-hot); read = that token's row, normalised by its counts (12 numbers).
- On silent steps the read is zero, plus a "silent" flag.
- Optionally, the whole table (96 numbers) feeds the GRU, so the agent can also use rows of tokens it is not hearing.

## How it plugs in

The frozen navigator trunk is unchanged. The side input to `side_enc` gains the retrieved row (12) and the silent flag.
GRU (128), side head and aux heads stay as in `s1kl03`, with **KL 0.3 to the navigator kept**. Also kept: the Nov 13
NaN fix, the in-run forage stop and milestone checkpoints. Slow weights learn how to use the read; the table itself is
never trained.

## Lever-did-work unit test (before any RL; free, laptop)

- Streams: navigator lives in the harness with the s1pilot config and a random code per life. The table is built
  online by the write rule. A linear probe is trained on 70% of lives (its input is the token's row, its target the
  true sector) and scored on 30%, by tokens heard.
- **Count before building (already in hand):** on the same evidence, the ideal counting observer reaches only
  **26 -> 56% exact sector** (1-5 -> 301+ tokens heard) and **~80% heading-equivalent**
  (`docs/data/ideal_observer/io2_parent.json`, `io3_parent.json`).
- **So the proposed bar ">= 80% within a median life's tokens" is unreachable for the exact sector with this
  evidence.** Proposed bar instead, both by ~600 tokens heard (the median life):
  - the probe on the KV table reaches **>= 90% of the counting observer's accuracy** on the same streams, on both
    the exact sector and the heading-equivalent;
  - the probe on the plain GRU state (u6000) stays below it.
- **Negative control:** the same table with tokens permuted against the evidence (a shuffled code) must probe at
  chance.

## Throughput and cost

- The table update is O(8 x 12) per agent-step, and the read is one row. Expect about a 5-10% throughput loss, i.e.
  ~19-20k agent-steps/s with KL on a 4-CPU container. That is ~14 h and ~$4.8 to 1B at the measured ~$0.34/h.
- Readouts at 1e8 and 3e8 (32 worlds) plus 1e9 (64) add ~$0.6. **Total ~$5.4; ~$1.9 to the 3e8 conditional.**
- The laptop unit test costs nothing (~5 min).

## Risks

- Evidence quality is the ceiling: the counting observer shows exact-sector evidence is weak. The KV memory can only
  remove the inference bottleneck, not add evidence.
- Even with a perfect read, the policy must learn to act on it under KL 0.3. Token influence ~1e-4 was all PPO
  produced in 161M steps.
- Writes depend on the agent's own behaviour (where it walks and eats), which is navigator-like and moves little.
