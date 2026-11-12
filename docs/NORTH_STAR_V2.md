# THRONG North Star v2: what to change, the new architecture, the new roadmap

Nov 11 2026. Written by Will (the coding agent) for Cam, M, the operator and any agent that continues this work.
It is a proposal. Nothing here overrides a ruling by Cam or M until they adopt it.
It builds on `docs/THRONG_North_Star_Roadmap.md` (Nov 8) and does not repeat it. Evidence is in
`docs/RESEARCH_PROTOCOL.md`; the live work order is in `docs/HANDOFF_2026-11-11.md`.

## 0. The goal, and the honest odds

**Goal:** intelligence that emerges in silicon from agents' own needs, with no human text, and that eventually
surpasses the best language models.

**Odds, stated once.** As literally worded, at budgets of tens to hundreds of dollars, the chance is close to
zero on any near horizon. Language models distil the output of billions of human lives. Nothing here shortcuts
that. The project is still worth doing for three results that are within reach and that nobody gets from text:

- agents that learn a new signal from a handful of experiences, inside one life;
- a code that is passed between generations and changes as the world changes;
- agents that adapt to a world with new rules faster, per unit of experience and energy, than a language-model
  agent placed in the same world.

The third is the only form of "surpassing" that can be measured soon. Section 4 makes it the yardstick.

## 1. What we have been doing wrong

Each item gives the evidence and the change. Items 1 to 5 are about the science; 6 to 10 about method.

**1. We use the wrong learner for a lifetime.**
- A newborn learns by PPO gradient steps at learning rate 3e-4. A life is ~900 steps, about 14 policy updates.
  PPO needs millions of steps. Biology does not learn a word by slow weight change either.
- Evidence: plain learning gives 0 of 934 lives; all helpers give 7.7%; a supervised probe with perfect data
  needed ~3,000 steps; more replay changed nothing.
- **Change:** no gradient steps inside a life. In-life learning happens in memory (recurrent state, fast
  binding). Slow weights are shaped across many lives. The parked "fast learner" was this idea; it was shelved
  for harness problems, not for evidence against it. It should be the main line, not the fallback.

**2. The W2 agent has no memory.**
- It is a feedforward network of two layers of 128 units. It cannot hold "that sound came before that food".
- The Nov 8 roadmap itself says keep the GRU memory. W2 dropped it for speed.
- **Change:** a recurrent core is the default agent from now on.

**3. The ladder puts language before the ability to learn.**
- The Nov 8 ladder goes signal, stable code, composition, culture, planning, and only then "solves a new problem
  within one lifetime". Memory and varied worlds were scheduled after rung 3.
- Evidence since then says the first blocker is fast learning within a life. Culture cannot exist without it.
- **Change:** adaptation first. Fast in-life learning becomes rung 1 of the new ladder.

**4. We demanded culture before instinct.**
- Almost all animal signalling is inborn. Learned signalling came much later and only in a few species.
- Our one clear success was effectively inborn: the shared-brain toy (5 of 5 seeds). Blank separate brains that
  must learn the code in one life have failed every time.
- **Change:** allow an inborn code as a legitimate early rung. Claim "culture" only when a test shows it: raise a
  newborn of one lineage among speakers of another and see whose code it uses (cross-fostering).

**5. The world gives them three bits to talk about, and little reason to say it.**
- "Which of 8 directions is food" is the whole vocabulary. Intelligence cannot exceed what the niche demands.
- Food calls help competitors. In W2 the speaker side has never been learned; the speakers are scripted.
- **Change:** start with signals where speaker and listener gain together (parent to young, alarm calls, tasks
  that need two agents), and make the world's demands grow with the agents' ability (section 3, world ladder).

**6. Three harnesses, three instruments.**
- Learnability, fast learner and toy population each had their own test. Each needed calibrating. One never
  calibrated; one had a copy step that never worked.
- **Change:** one world, one agent interface, one probe suite. A new experiment is a configuration, not a script.

**7. We register exploration as if it were confirmation.**
- Bars were registered for exploratory sweeps, then needed rulings on wording ("1,100 per seed or on the
  median?", "which 30%?"). The world was retuned until three thresholds passed, which is tuning to the gate.
- **Change:** two lanes. EXPLORE: free or under $1, no bars, labelled exploratory, fast. CONFIRM: registered
  design, bar and script, used only for a claim we intend to stand behind.

**8. Pass or fail on 3 of 5 seeds throws information away.**
- **Change:** report the effect with its interval, pooled over lives, plus the per-seed spread. Keep a bar only
  where a decision depends on it.

**9. Defaults silently decided experiments.**
- Replay memory of 4 rollouts, copying that stops at age 1,000 and a token temperature of 0.08 each undermined a
  paid result.
- **Change:** before launch, print the resolved configuration and the exact difference between arms, and run
  the "each lever did measurable work" check. Both exist now; they move from after the run to before it.

**10. The loop is slow and the infrastructure leaks.**
- Each judgment call waits hours for a relayed ruling. The spend guard has died with the session five times.
- **Change:** Cam registers a decision tree with each batch (this worked on Nov 10), Will proceeds within it and
  a spend cap. Deploy the guard on Modal as soon as M creates the secret.

**Also worth stating:** the production network has been untouched for months and the toy is closed. W2 is the
project now. Say so, and archive the rest, so effort is not split three ways.

## 2. Architecture v2

### Three timescales, kept separate

| Timescale | What changes | How | Inherited? |
|---|---|---|---|
| Lineage (slow) | Network weights, plasticity settings, drives | Gradient across many lives, plus selection between lineages | Yes, with mutation |
| Life (fast) | Recurrent state and memory contents | Experience only. No gradient steps | No |
| Culture | Signals, habits, things built in the world | Hearing and watching others; the world itself | Only through the world |

The old rule "offspring never inherit weights" is replaced by "what must not be inherited is tested, not
assumed". Newborns already start from trained navigators, so weights are inherited today in all but name.

### The agent

- **Inputs:** the 7x7 view (food, agents), own energy and age, the tokens heard with the speaker's offset, the
  previous action and its reward.
- **Core:** a GRU of 128 to 256 units. An optional small key-value memory, adopted only if an ablation shows gain.
- **Outputs:** action (6), speak (a token or silence), value.
- **Self-supervised heads** (inborn drives, each needs M's approval):
  - where my next meal turns out to be, labelled in hindsight from my own path;
  - where the other agents will be, which teaches attention to others;
  - the next view.
- **Channel:** keep the 3-slot VQ wire. The Nov 8 A/B kept it over plain symbols (protocol line 2562). Fix
  copying instead: copy in embedding space or with a soft temperature, and keep a unit test for it.

### The population

- 4 to 8 lineages, each one set of slow weights shared by all its members.
- Individuals differ only in memory. Thousands of agents then cost one batched forward pass.
- Lineages that reproduce more are copied over those that reproduce less, with mutation of their settings.
- Risk to check first: shared starts biased words in an earlier control (G0). Per-life random codes and the
  cross-fostering test measure that bias instead of assuming it away.

### The world ladder

Each level is entered only when agents have saturated the one before.

| Level | Adds | Gives agents to say |
|---|---|---|
| W2.0 (now) | Food patches that relocate | Where food is |
| W2.1 | Parents provisioning young nearby | "Come here": same interest on both sides |
| W2.2 | Two predator kinds needing different escapes | Which threat: alarm calls |
| W2.3 | Materials with generated properties; two-agent tasks | What, where, and combinations |
| W2.4 | Things that persist: caches, marks, structures | Knowledge outliving a life |
| W2.5 | A different rule set every life; worlds written by a generator | New words for new things |

A language model may write worlds and tasks at W2.5. It shapes challenges and never meanings, as the Nov 8
roadmap already allows.

### One probe suite

Every claim is read with the same instruments, each calibrated with a perfect agent (must pass at least 90%)
and a deaf agent (at most 10%) before any learner is scored:

- swap test with last-heard credit and eligibility (frozen at comm-toy `b61afb2`);
- value of the code: true messages against silence, in both forms;
- cross-fostering, for culture;
- held-out combinations, for structure;
- adaptation curve: reward against experience in a world with new rules.

## 3. Roadmap v2

One gate per stage. Costs are estimates to be replaced by a two-minute price probe before each launch.

| Stage | Goal | Gate | Estimate |
|---|---|---|---|
| 0 | One harness; per-life random code in the calibrated test; guard on Modal | Perfect agent >= 90%, deaf <= 10% with random codes | Free |
| 1 | **Learn to listen within a life**, with memory and no in-life gradient | >= 50% of eligible lives rely at death, on code families never trained on | $5-20 |
| 2 | Learned speakers where both sides gain (parent calls) | Reliance and food gain against a wire-off arm | $10-30 |
| 3 | Culture | Cross-fostered young adopt the foster code; code persists 3+ generations | $20-40 |
| 4 | More to say (predators, materials) | Vocabulary grows with the world; held-out combinations understood | $30-80 |
| 5 | Adapting to new rules within a life | Beats a fresh learner; first comparison with a language-model agent | $50-150 |
| 6 | Open-ended growth: generated worlds, arms races, persistent things | New abilities keep appearing without new design | Open |
| 7 | Words used for thinking; talking with them; low-power hardware | As in the Nov 8 roadmap | Open |

**Stage 1 in detail, since it is the bottleneck.**

- Each life gets a random code (a random mapping from the 8 directions to tokens), so nothing about the code
  can sit in the weights. The agent must work it out from experience, in memory.
- Speakers stay scripted at this stage. That is an instrument, not a claim.
- Train the slow weights across many thousands of lives; score with the frozen test.
- **First probe (free):** a perfect decoder that is told the code must pass at least 90% with random codes.
- **Kill criterion:** no rise in reliance with tokens heard after about a billion agent-steps. Then add the
  key-value memory. If that also fails, listening has to be inborn, and stage 3's claim narrows.
- **Why this should work:** learning a small mapping in memory across episodes is the textbook case for
  meta-learned recurrent agents. Verify these references before relying on them; they are from memory:
  - RL-squared (Duan et al. 2016) and "Learning to reinforcement learn" (Wang et al. 2016);
  - "Learning few-shot imitation as cultural transmission" (Bhoopchand et al. 2023), which got real-time cultural
    transmission from memory, experts who drop out, an attention loss on other agents and randomised worlds;
  - "AI-generating algorithms" (Clune 2019), the general case for learning the learner and generating the worlds.

## 4. The yardstick for "surpassing"

- **Test:** worlds from the W2 family with rules the agent has never met.
- **Measure:** reward against experience within one life, and cost per life.
- **Opponents, in order:** a random agent, a scripted agent, a fresh gradient learner, then a language-model
  agent given the same world as text.
- **Claim ladder:** beat each opponent on a fixed set of worlds, then widen the set. "Surpassing" is the set of
  worlds on which ours wins, and the roadmap is making that set larger.
- The language-model comparison costs API money and is M's decision. It means little before stage 2.

## 5. What M and Cam need to decide

- Adopt, amend or reject the reordered ladder (adaptation first, instinct allowed, culture by cross-fostering).
- Replace "never inherit weights" with the tested rule in section 2.
- Approve the self-supervised heads as inborn drives.
- Declare W2 the mainline and archive the toy and the production network.
- Budget: stages 1 to 3 need roughly $35 to $90 in total. About $12.7 of headroom remains under the current guard.
