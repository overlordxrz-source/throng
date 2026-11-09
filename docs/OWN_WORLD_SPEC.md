# W2: our own world. Spec for Cam's sign-off

Draft, Nov 9. Cam asked for this spec before any code beyond a prototype. Code + tests only; no paid runs until
sign-off.

**Principle (unchanged):** we design the world and its pressures, never the meanings. No reward ever names a
signal, a meaning or a task. Probes (swap, same-spot, speak) are measurements on frozen test episodes.

## 1. Engine

- Pure JAX. The whole step `(state, actions) -> (state', obs)` is jitted, vmapped over worlds and scanned over time,
  so a rollout is one XLA program (as the toy's `lax.scan` rollout).
- **Throughput target: >= 100k agent-steps/s on one A10G.** Measured first on the laptop CPU, then by a 2-minute
  GPU smoke.
- State is a pytree of fixed-shape arrays (max agents per world `N_max`, an `alive` mask). There are no Python
  lists, so births and deaths are mask flips, never reshapes.

## 2. World

- 2-D torus of `L x L` cells, default 64 (configurable; senses never depend on `L`).
- **Food:** a per-cell amount in [0, 1], grown in **patches** (Gaussian blobs of radius ~3), regrowing
  logistically.
  - Stage flags: `food_move` (patch centres drift), `food_deplete` (no regrowth while grazed below a threshold).

## 3. Body

- **Energy** E in [0, 1]. It drains every step: basal `d0` plus `d_move` per move. *Eat* takes up to `bite` of the
  food on the agent's cell. **Death at E = 0** (the `alive` mask flips; the slot frees).
- **Age** counts steps since birth. **Lifespan cap** `A_max` (old age).
- **Reproduction** is automatic when E >= `E_rep` and age >= `A_min`: the parent pays `c_rep`, and the offspring
  appears on an adjacent free cell with E = `c_rep` x efficiency.
- **Traits (inherited with mutation):**
  - learning-speed multiplier (log-normal mutation, sigma 0.1);
  - message-head plasticity schedule: sensitive period and floor (as the separate-brains toy).
  - Traits are heritable; **weights are not** (see section 6).

## 4. Reward: homeostatic, no task reward

Drive reduction (Keramati & Gutkin, eLife 2014): drive `D(E) = |E* - E|^n` (setpoint `E* = 0.8`, `n = 2`), reward
`r_t = D(E_t) - D(E_{t+1})`. Death ends the agent's episode (no reward after it), so staying alive and near the
setpoint is the only thing rewarded. There is no reward for messages, for reproduction as such, or for any task.

## 5. Senses: absolute, local, limited range

- **Egocentric window** of `(2r+1)^2` cells, r = 3 (7 x 7). Channels: food amount, agent present, and later stages'
  channels (lookout, predator kinds). Offsets are in cells; there are **no positions and no division by L**, so a
  navigator trained at L = 64 sees exactly the same thing at L = 256.
- **Interoception:** own E (absolute), own age / `A_max`.
- **Lookout (stage flag):** while standing on a lookout cell, the window widens to r = 9 for food (and predators).
  That is the "wide view" Cam asked for; information comes from where you stand.
- **Hearing:** the messages of up to K = 2 nearest *speaking* agents within hearing radius R_h = 6. Each gives its 3
  VQ codebook vectors + its offset clipped to the radius, one step delayed (as the toy's wire).

## 6. Brains: the open decision

The language goal needs separate brains (culture can't happen in one shared brain), and newborns must start from
their own weights, not their parent's. Weights passed down would be inheritance, not culture.

- **Option A (recommended): one brain per agent slot, one big world per run.** `N_max` brains (e.g. 64), stacked;
  a forward pass is vmapped over agents with their own params (~40k floats each, so 64 brains = 2.6M floats).
  - Each brain learns from its own agent's experience only. PPO batches are per agent over long rollouts.
  - A newborn takes a free slot with a fresh navigator-start brain (from a pool of independently trained
    navigators, so no shared word bias) + inherited traits.
  - **Risk:** little data per brain means slow, noisy learning. Gate P1 below tests that the body can be learned
    this way at all.
- **Option B: lineage brains** (B ~ 8-16 brains, each shared by a family). Cheaper and more data per brain, but
  offspring inherit weights inside a lineage, which confounds the culture test. Only for a fallback prototype.
- Batching worlds (W copies) cannot share per-agent brains, because births happen at different times in different
  worlds. Option A is therefore W = 1 world with many agents (or a few independent worlds, each with its own brains).

## 7. Wire (as the toy)

The 3-slot VQ token policy (each token an action, tau as the toy), the **wire-blind message head**, and the
message-head gradient rule from (c) (sender rows only if stop-grad is adopted). In W2 every agent both speaks and
hears; "sender rows" become "rows whose message was heard by someone". The rule is restated in the code with its
own apparatus test.

**Probes ported:**
- **cross-spot / same-spot swap:** a heard message replaced by one the speaker used in a different / the same
  situation (situation = the nearest food patch in the speaker's window, or the predator kind once stage 3 is on);
- **speak:** a dictionary from frozen rollouts, injected as an outside speaker;
- **joint MI with a shuffle null.**

## 8. Things worth talking about, staged behind flags (one at a time, each with its own registered test)

1. `food_move` / `food_deplete`: patches out of a listener's view that a speaker can see.
2. `lookout`: the wide view from one spot (the 3b question rebuilt with absolute senses).
3. `predators`: two kinds needing different escapes (e.g. an aerial one, escaped by staying still under cover; a
   ground one, escaped by fleeing uphill / away). The vervet alarm-call test: do calls differ by kind, and does a
   played call trigger the matching escape?
4. `parts`: facts with parts (kind x place, e.g. which food x which direction) for compositionality.

## 9. Prototype gates (laptop, $0)

- **P1:** a learned no-comm agent survives longer than a random one (mean lifespan, 4 seeds, > 2 SE).
- **P2:** value of information: an ORACLE agent told the offset to the nearest food patch (clipped, absolute)
  lives **>= 2x** as long as a blind one under the same learner and budget.
- **P3:** steps/s measured: env only, and env + policy forward, at N = 64 agents, laptop CPU.
- Before learning, a **scripted check of P2's premise:** a greedy-to-visible-food agent vs a random walker, and an
  oracle-greedy agent vs a blind-greedy one. If the scripted oracle isn't >= 2x, the world, not the learner, is
  retuned first.

## 10. Build plan

1. `w2/env.py`: state, step, senses, energy, food, reproduction mask; unit tests (energy conservation per step,
   death at 0, birth only above `E_rep`, the window identical at L = 64 and L = 256 for the same local scene,
   torus wrap).
2. Scripted gate check (P2 premise) + throughput (P3, env only).
3. `w2/brains.py`: per-agent stacked brains (from the toy's `brains.py`, compute-and-select replaced by a vmap
   over agents), the wire, per-agent PPO; tests (gradient isolation, newborn reset, trait inheritance + mutation).
4. P1, P2 with learning (laptop).
5. Probes ported + their apparatus tests.
6. **First paid smoke:** 2-minute A10G timing (P3 on GPU) + one short no-comm run, ~$0.3. Then registered runs
   per stage.

**Cost to the first paid smoke: ~$0.3** (A10G ~$1.10/h; a 2-minute timing + a ~10-minute run). Everything before it
is laptop work.

## 11. Decisions for Cam

- Option A (per-agent brains, one world) vs B.
- The predator escapes in stage 3.
- `N_max` and `L` for the first stage (proposal: N_max = 64, L = 64).
