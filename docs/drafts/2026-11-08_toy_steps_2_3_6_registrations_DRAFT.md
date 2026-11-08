# DRAFT registrations: build-order steps 2, 3 and 6 (CPU toy, local critic) — for Cam's approval

Per Cam's T4 ruling: these test how a code **forms**, so they use the local critic that passed rung 2 (the
replication's PS recipe). They do not wait for T5. Nothing is built or run until approved.

**Prices** come from measured Modal CPU runs (2 cores / 4 GiB, ~$0.005/min per container): ~$0.20 per 2,000-update
toy run, ~$0.33 per 3,000-update run. Toy/CPU allowance $15 total. Spent so far ~$4.3 (estimate: metered total
minus GPU jobs). T5 is capped at $3.

## Common base (all three)

The replication's PS arm: ORACLE0-toy warm start, `--wire-init-scale 0.06`, craft cost 0.45, c_ps 1.99 **on**
(no anneal; formation, not survival), the joint (local) credit, 2,000 updates, eval at 1,000 and 2,000 (MI vs
null + 10 empirical-swap pairs).

**Reliance bar (as rung 3):** swap drop >= 30%, paired > 2 SE, listener deliveries >= the run set's NOSIG level.
**Redirection:** listener crafts at the swapped fire, swap >= 2x control, paired > 2 SE.

---

## Step 2 — channel A/B: discrete symbols vs the VQ wire

- **Change (one variable):** the message is **3 symbols from a vocabulary of 8** (512 messages), each chosen as a
  categorical action from a per-slot linear head. It is broadcast as one-hot vectors (24 of the 40 wire dims; the
  rest zero), and the same wire-input init scale applies. No codebooks and no VQ loss. The PS term uses the same
  batch-MI formula over the 3 symbol slots.
- **Arms:** VQ (today's channel) x5, SYM x5, NOSIG x2 (the NOSIG level), all fresh seeds.
- **Primary:** SYM is **adopted** if, at 2,000, its reliance passes in >= 4 of 5 seeds **and** its mean deposit drop
  is >= VQ's mean drop - 0.10 (a non-inferiority margin). Redirection is co-reported with the same 4 of 5 bar.
  - Otherwise VQ stays.
  - If both fail their own bars, the A/B is uninformative.
- **Secondary:** MI, how fast the code forms (first update with MI > 1 bit), listener deliveries, message entropy.
- **Tests before running:** symbol-head gradient tests (the PS and token terms reach only the symbol heads); an
  apparatus check that the listener receives the sender's one-hot; a mutant that broadcasts a constant must fail
  the apparatus check.
- **Price:** 12 runs x ~$0.20 = **~$2.4**.

## Step 3 — no assigned sender (knowledge from proximity)

- **Change:**
  - Both agents are identical. Each sees the active fire's one-hot only while within Chebyshev radius 2 of it
    (the production 5 x 5 view); otherwise those dims are zero.
  - Both always speak and listen.
  - The shared reward is symmetric: either agent's delivery also pays the partner 2.7.
  - PS applies to whichever rows are informed at that step.
- **Swap test:** the partner's broadcast is replaced by the informed agent's own token for T', on steps where the
  speaker is informed and the hearer is not (from the paired control rollout).
- **Arms:** PROX x5 (no assigned sender), ASSIGNED x3 (today's sender/listener, the concurrent reference), NOSIG
  x2, all fresh seeds.
- **Primary:** redirection under swap in >= 4 of 5 PROX seeds **and** the PROX mean redirection ratio >= 0.5 x the
  ASSIGNED mean ("on a par"). Reliance is co-reported.
- **Secondary:** the fraction of steps on which an agent is informed; how often uninformed agents speak; whether
  the code is shared (both agents use the same token for the same fire: cross-agent decoding accuracy).
- **Tests:** an apparatus check that the need dims are nonzero only within radius 2 of the active fire; symmetric
  reward accounting; a mutant that leaves the need dims always on must fail the apparatus check.
- **Price:** 10 runs x ~$0.20 = **~$2.0**.

## Step 6 — compositionality pilot (after step 2 picks the channel)

- **Change:**
  - The target is a **pair (fire, material)**: 4 fires x 4 materials = 16 meanings.
  - The listener's Craft becomes 4 Craft-material actions; a delivery needs the active fire **and** the target
    material.
  - The sender sees both one-hots.
  - The channel is step 2's winner. If that is symbols at 3 x 8, the slot vocabulary is cut to 4, so the 16
    meanings outnumber one slot's codes.
- **Held out:** 2 of the 16 pairs never occur in training (chosen before running, one per fire family).
- **Arms:** COMP x5, plus a shuffled-meaning control computed in analysis (no extra run); 3,000 updates.
- **Gates (both):**
  1. **Topographic similarity:** the Spearman correlation between meaning distance (Hamming over (fire, material))
     and message distance (Hamming over symbols) at 3,000 is above the 95th percentile of a 1,000-permutation
     null, in >= 4 of 5 seeds.
  2. **Held-out generalisation:** in a frozen test where only the 2 held-out pairs are active, listener deliveries
     with the correct material at the correct fire are above chance (1/16 per attempt, binomial p < 0.01) in >= 3
     of 5 seeds.
- **Tests:** the deposit rule needs both conditions; held-out pairs never appear in training (asserted every
  rollout); a mutant that lets a held-out pair through must trip the assertion.
- **Price:** 5 runs x ~$0.33 = **~$1.7**.

---

**Total if all three are approved: ~$6.1** (+ T5 <= $3), against ~$10.7 left in the toy allowance.
