# THRONG North Star Roadmap

Nov 8, 2026 · @Dimitar

Will: this is the plan from here to the north star, and the order of work for the next month. M has confirmed a few hundred dollars of Modal credit over the coming months, so plan each stage against its cap rather than the current balance. The standing rules near the end apply to every step.

## The north star

THRONG bets that intelligence built on symbols earned through consequences can eventually do things text-trained models can't. Creatures earn signals by using them to survive and cooperate. The signals then grow into language, culture and reasoning, and finally run on brain-like hardware.

"Surpassing LLMs" needs a concrete meaning, or it can't be tested. Here it means three things: more reliable causal and planning reasoning in a world the agent acts in, faster learning of new tasks through communication, and continuous operation at a tiny fraction of the energy. This is a long-horizon research bet, not a product plan. Every rung below has a test that could prove it wrong.

Evidence, from the first rung to the last:

- A signal that causally changes what a listener does.
- A code that survives once the training nudges are removed.
- Compositional reuse: new combinations of signals understood without training on them.
- Knowledge passed between generations, so novices learn faster in a world with culture.
- Language used for internal planning that beats agents without it.
- The same policy running on neuromorphic hardware with matching behavior.

## Where THRONG stands

THRONG now has a learner that works and a first functional communication channel, though only in a small lab world. Until November, the main simulation had never actually tested its central thesis. Silent training bugs blocked learning before communication could even start.

| Area | Status (Nov 8) | Evidence |
| --- | --- | --- |
| Training signal | Fixed in the testbed | Imagination gate overrode \~77% of actions; with it off, clip fraction fell 0.384 → 0.06 |
| Credit assignment | Root cause found, fix validated in the testbed | Truncated 512-step returns with no bootstrap; bootstrapped scoring gave 4–8× more deliveries |
| Value estimate | PopArt fix works in the testbed | EV +0.42 at plateau vs −1.2 for the earlier fix |
| Navigation | Learned at 32×32, transfers frozen to 128×128 | 37× and 20× the frozen-control delivery rate at 128 |
| Communication, real network | Two nulls | Signals collapsed toward a constant wherever a listener received them |
| Communication, CPU toy | Working channel, replicated on fresh seeds as registered | Reliance 5/5 seeds (deliveries drop 67–91% under swap), meaning 4/5, redirection 5/5. Persistence without the nudge turned out to be inertia |
| Production world | Untouched | Still runs the old estimator and critic |

What is still open: whether listener use keeps the code alive against a force that erodes it, and whether the channel holds on the real network.

## The milestone ladder

The path to the north star is eight rungs, and each one must pass its test before the next is worth funding. Rung 1 is done, and rung 2 is in progress.

&#91;embedded content: milestone ladder · 8 rungs, each with a pass test\]

Rungs 4 to 7 follow the field's evidence on what makes emergent language compositional and lets it scale. Rung 8 can start early as a free simulator test, but real hardware waits for a stable language.

## Where the earlier threads fit

Every earlier idea stays on the plan. But most were built while training was broken, so each one comes back in order and has to earn its result again.

| Thread | State now | Where it returns |
| --- | --- | --- |
| Imagination ("dreaming") | Switched off. Its world model predicted "nothing changes", so imagined actions were noise, and overriding 77% of actions corrupted training | Rung 6. Retrain the world model until it predicts real change, then bring imagination back as a planning aid, with a test that it beats no imagination. It never again overrides actions without an off-policy correction |
| Memory (GRU carry, 20-slot episodic memory) | Still in the real network | Re-verify the earlier lag-10 memory result on the fixed learner before relying on it |
| Global workspace (3-slot VQ wire, GWT router) | The wire carries the signals in the communication tests. The GWT router is untested on the fixed learner | Rungs 2–4 make the wire carry meaning. The workspace itself belongs inside each brain (see the architecture review), and rung 6 adds it with state-dependent attention |
| Culling the weakest | Not meaningful yet: all agents share one network, so removing one changes nothing that learns | After rung 3, move to a population of 4–8 separate networks with population-based training, where top performers are copied over the bottom ones. Newborn replacements double as listener resets, which favor compositional codes, and they set up the culture test |
| Talking to them in English | The earlier Rosetta Stone alignment ran on tokens from the broken-training period, and a small vocabulary aligns plausibly to almost anything | The translation track below |

**Translation track.** Every signal is about something in the world, so translation can be grounded rather than guessed. None of this teaches them English. We learn their language the way a field linguist would, every step runs on test episodes with their brains frozen, and they never learn from it.

1. **Dictionary.** Record what each signal predicts in the world, such as which fire or which material. This comes after rung 2 passes on the real network.
2. **Speak to them.** On a test episode, a person picks a fire, the system inserts the agents' own signal for it, and the listener goes there. Gate: success on a par with a real sender.
3. **Read them.** A live viewer shows each message as English, using the dictionary.
4. **English in, English out.** An LLM on our side maps typed English onto the grounded dictionary and back, so the agents only ever receive their own signals. Once codes are compositional (rung 4), this becomes grammar induction. Gate: English instructions held out of training are understood above chance.

## Architecture review

The main limit on their language is what the world gives them to say, not the channel or the brain. So the world should get richer while the channel and the brain get simpler. In the toy, signals filled their whole channel (1.96 of 2 bits) as soon as there were four fires worth telling apart.

**The rule: we design the world and its pressures, never the meanings.** Nothing outside the agents tells them what a signal means or rewards a particular meaning. The swap test and the translation track are measurements on test episodes, so English never enters their world. Positive signalling is the one nudge aimed at signals: it rewards being informative without saying what anything means, and rung 3 removes it.

| Area | Now | Change | Test before adopting |
| --- | --- | --- | --- |
| Who knows what | Assigned by a flag: the sender sees the lit fire in the toy, and 50% of agents see the recipe in production | Knowledge comes from where an agent is. Anyone who comes within view of the lit fire sees it, every agent can both speak and listen, and events appear, move and run out. Fixed landmarks can stay on the compass | Toy, registered: the same task with no assigned sender. Pass: redirection under swap on a par with the assigned-sender toy |
| What is worth saying | One fact at a time: which fire is lit | Facts with parts (which material × which fire, which threat × which direction) and things out of sight. Two predator kinds that need different escapes would give natural alarm calls, as vervet monkeys have (Seyfarth, Cheney & Marler, Science 1980) | The compositionality pilot (step 6) |
| Channel | 3 slots × 64-code VQ wire (about 262,000 possible messages), dead-code resets, and at least four silent bugs since June | Shrink it: a message of 2–3 symbols from a small vocabulary (8–16), each chosen as an action, sized so meanings outnumber single symbols. A large vocabulary lets agents name each meaning whole; a small one pushes them to combine symbols (Kottur et al., EMNLP 2017). Symbols are also closer to words and natural for spiking hardware | Toy A/B, registered: symbols versus the VQ wire on the same task and seeds. Switch if symbols match or beat it on reliance and redirection |
| Brain | 11 output heads (including theory of mind, alarm, writing and two culture heads) plus imagination and the GWT router. 44 of 206 settings are never read | A lean core: encoder, GRU memory, then policy, value and message heads. Keep the GRU, since it lets an agent talk about something it saw earlier. Every other module returns only when an ablation shows it helps | One ablation per module, one at a time |
| Workspace and language | The VQ wire doubles as the global workspace | Keep them separate. The workspace is a broadcast inside one brain, and language passes between brains. They meet at rung 6, if agents start reusing their language to think | The rung 6 gate |
| Population | One brain shared by up to 200 agents, so no one has anything to teach | 4–8 separate brains after rung 3, as above | The rung 5 culture test |

Nothing here touches the registered real-network confirmation. The toy tests are free and run alongside it, one change at a time: the channel A/B first, then the world with no assigned sender. Production takes a change only after it passes in the toy and then the testbed.

## When intelligence could emerge

Intelligence won't switch on at a date. It shows up as abilities we can test, one rung at a time. The first one that counts as intelligence in a strong sense is solving a problem the agent has never met, within its own lifetime. That is roughly 6–12 months away if most gates pass. The timings below are guesses at this month's pace, not forecasts.

| Sign | Test | Rung | Rough guess |
| --- | --- | --- | --- |
| Signals change what a listener does | Swap test on the real network | 2 | \~2 weeks |
| The code survives without the nudge | Erosion test, then the anneal on the real network | 3 | Weeks after rung 2 |
| New combinations of signals are understood | Held-out pairs above chance | 4 | 1–3 months |
| Knowledge passes between generations | Novices learn faster in a world with elders | 5 | 3–6 months |
| Language is used to plan | Agents that can talk to themselves beat agents that can't | 6 | 6–12 months |
| A never-seen problem is solved within one lifetime | New task types solved in a few tries, faster than a fresh learner | 7 | 6–12 months |
| Better than LLMs at something real | The test M picks under "Decisions M still owns" | 8 and beyond | Years. This is an open research problem |

**What gives them room to evolve.** Every change here shapes the world or its pressures, never the meanings.

| Change | Why | When |
| --- | --- | --- |
| A different world every life: recipes, seasons, layouts and threats change between episodes | Agents learn to adapt only when one fixed plan stops working. DeepMind's AdA reached human-timescale adaptation from a vast, diverse task space, an attention-based memory and a curriculum at the edge of its ability ([Bauer et al., ICML 2023](https://proceedings.mlr.press/v202/bauer23a.html)) | Start small after rung 3 |
| Arms races: predators and prey both learning on the fixed learner | In hide-and-seek, competition alone produced up to six distinct strategies, including shelter building, with no reward for using objects ([OpenAI, 2019](https://openai.com/index/emergent-tool-use)) | With the size curriculum |
| A world that writes new tasks: an LLM designs tasks that are learnable and interesting at the agents' level | OMNI-EPIC had a foundation model write new tasks and rewards as code, tuned to the agent's progress ([Faldor et al., 2024](https://www.arxiv.org/abs/2405.15568)). The LLM shapes challenges, never meanings | Rung 7 |
| Real evolution: separate brains, offspring inherit and mutate brain size and learning traits, and every neuron costs energy | In an evolved foraging world, brains grew over generations whenever size was free. An energy cost gave smaller brains with no loss of performance ([Heesom-Green et al., 2025](https://arxiv.org/html/2511.20018v1)) | After rung 3, with the move to separate brains |
| Messages cost a little energy, and everyone can speak and listen | A cost on message length, with listeners that act early, gives short words for common meanings, as in human languages ([Galke & Raviv, 2024](https://arxiv.org/pdf/2403.14427)) | With the channel A/B |

**Bigger brains: not yet, and not by default.** Bigger brains can unlock new abilities: going from the usual 2–5 layers to as many as 1024 gave 2–50× gains in goal-reaching RL, and new kinds of behavior ([Wang et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e74ee34cc0f2d0780f34ee77d8fba25b-Abstract-Conference.html)). But oversized networks have little need to converge on a shared, structured language, and periodic re-learning is what pushes structure ([Galke & Raviv, 2024](https://arxiv.org/pdf/2403.14427)). Larger populations alone did not give robust, general protocols either ([Chaabouni et al., ICLR 2022](https://www.iclr.cc/virtual/2022/poster/6459)).

So brains grow when the world outgrows them, each step gated on bigger beating smaller, and evolution with an energy cost picks the size. The one upgrade worth planning now is memory. An attention-based memory, as in AdA, is what lifetime learning needs, and it arrives with the different-world-every-life step.

Compute is the real constraint. AdA and hide-and-seek ran at big-lab scale, and THRONG has a few hundred dollars. So the route is small, fast worlds where each new ability shows up cheaply, and the budget buys tests rather than scale.

## Next 30 days: directives

Work in this order. A step starts only when the one before it passes, report at each gate, and keep everything that can run on CPU on CPU.

1. **Toy replication: done, passed ($0, `6428c84`).** Results:
   - Reliance passed in 5 of 5 seeds, meaning in 4 of 5, and redirection in 5 of 5.
   - Persistence with the nudge off passed as registered, but it measured inertia: the code persisted just as well with the wire off. Self-maintenance is still open.
2. **Erosion test (CPU toy, $0, runs alongside step 3).**
   - Add a force that erodes unused codes. Calibrate its strength on separate seeds so a wire-off code falls below 0.5 bits within 500 updates.
   - Then run the registered comparison: anneal the nudge with the wire on versus off.
   - Pass: with the wire on, MI stays ≥1 bit and reliance holds in ≥2 of 3 seeds, while the wire-off code decays.
3. **Jitted rollout for the real testbed ($0, plus \~$0.10 timing).** Port the Python-loop rollout (\~37 s per update) to a jitted scan. Prove exact equivalence against the old path on fixed seeds, then do one timing run.
4. **Real-network confirmation (cap $15).**
   - Setup: the production network at 32×32, PopArt critic, vf 0.5, and warm start from ORACLE0.
   - Craft cost stays at 0.1, so the last run's COMM and NOSIG arms are direct baselines and PS is the only change.
   - Primary: redirection at the swapped fire, with movement reported alongside.
   - Arms: PS ×3 plus a fresh NOSIG control if the speed-up allows, otherwise PS ×2. Halt at update 2 on an MI floor. *[Superseded by Cam's ruling, Nov 8: update 2 is an apparatus check only (PS gradient reaches the message head, messages not constant, listener wire input nonzero); the MI floor is set from a CPU smoke at a registered check update K <= 25% of the run. See docs/RESEARCH_PROTOCOL.md. The updated roadmap file had not arrived when this copy was saved.]*
5. **Replication (cap $20, only on a pass).** Rerun the winning arm with 3 fresh seeds. Nothing is called a result before this passes.
6. **Compositionality pilot (CPU toy, $0).**
   - Make the target a pair, for example fire × material, so meanings outnumber one slot's codes. Hold 2 pairs out of training.
   - Gates: topographic similarity above a shuffled baseline, and listeners succeeding on held-out pairs above chance. Use whichever channel wins the A/B in the architecture review.
7. **Lava port test (CPU, $0, anytime).** Convert the saved 32×32 navigation actor to a Sigma-Delta network with lava-dl and run it in Lava's CPU simulator. Gate: delivery rate within 20% of the JAX policy. Record the toolchain versions.
8. **Land the learner fixes in production (cap $25).**
   - Put bootstrapped λ = 0.95 scoring and PopArt behind config flags, with gradient tests and mutants, and get the config audit passing.
   - Then run the size curriculum 32 → 64 → 128, gating each stage on deliveries ≥2× the frozen control.

## Compute and budget

M has confirmed a few hundred dollars over the coming months, with \~$42 available now. Treat each cap as a ceiling with an automatic spend guard, and replace these estimates with measured costs as runs come in. Measured so far: \~$0.45 per 80-update testbed run, and \~$1.60 per 120-update real-network communication run.

| Stage | Cap (USD) | Starts when |
| --- | --- | --- |
| Real-network communication confirmation | 15 | The toy replication passes |
| Replication of that result | 20 | The confirmation passes |
| Production learner fixes + size curriculum | 25 | After the communication result |
| Rungs 3–4 on the real network: self-sustaining and compositional code | \~60 | The toy pilots pass |
| Rung 5: writing and culture | \~80 | Rung 4 passes |
| Reserve | \~50 | Crashes, reruns, surprises |

That totals about $250. Rungs 6–8 get budgets once rung 5 shows which design scales.

## Prior research to build on

Most of what THRONG hit this month is documented in the literature, and the remedies that worked came from there. These are the papers that shape the next rungs.

| Paper | What it gives THRONG |
| --- | --- |
| [Eccles et al., NeurIPS 2019: Biases for Emergent Communication in Multi-agent RL](https://proceedings.neurips.cc/paper/2019/hash/fe5e7cb609bdbe6d62449d61849c38b0-Abstract.html) | Positive-signalling and positive-listening biases for the joint exploration problem. Source of the toy's working remedy |
| [Bailey, Child & Weyde, JAIR 2026: Emergence of Compositional Languages](https://www.jair.org/index.php/jair/article/view/17302) | Productivity pressure is required for compositionality, and listener resets help. Topographic similarity and systematic generalization are the standard tests |
| [Lowe et al., 2019: On the Pitfalls of Measuring Emergent Communication](https://arxiv.org/abs/1903.05168v1) | Why signal information alone doesn't prove a listener uses it, and why the swap test stays primary |
| [Pardo et al., 2017: Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378v2) | Bootstrap at rollout cuts instead of treating them as the end of the world. That was production's estimator bug |
| [van Hasselt et al., NeurIPS 2016: PopArt](https://arxiv.org/pdf/1602.07714) | Rescales value targets while preserving outputs. The critic fix that worked |
| [Stewart et al., 2025: RL Robot Control on Loihi 2](https://arxiv.org/html/2512.03911) | A PPO actor converted to a spiking network with Lava: \~20× better energy-delay, with a real accuracy cost |
| [Kagan et al., Neuron 2022: DishBrain](https://www.eurekalert.org/news-releases/967286) | \~800,000 living neurons learned Pong in closed loop. The proof of concept for wetware |

## Neuromorphic and wetware path

Neuromorphic deployment is realistic once the language is stable. Wetware is a long-range experiment, not a deployment target. Either way, a free start is available now.

**Neuromorphic (Intel Loihi 2).** The proven route is converting a small ReLU actor into a spiking Sigma-Delta network with Lava, then deploying it. [Stewart et al.](https://arxiv.org/html/2512.03911) did this with a 12×64×64×6 PPO actor and got \~20× better energy-delay than a GPU. But position error rose from 0.06 m to 0.40 m, so the policy must be trained or distilled with conversion in mind. The navigation actor is small enough to try this in Lava's CPU simulator now.

The hardware itself is gated. Loihi chips are available only through the [Intel Neuromorphic Research Community](https://intel-ncl.atlassian.net/wiki/spaces/INRC/pages/1810432001), mostly via a remote research cloud. Free hardware loans go to academic members who have already built and tested a model in Lava. That page was last updated in 2022, so the program's current terms are an open question.

**Wetware (living neurons).** Two commercial routes exist today:

- [Cortical Labs CL1](https://www.techradar.com/pro/what-a-deal-you-can-now-rent-human-brain-cells-on-a-silicon-chip-for-7-days-for-less-than-a-nintendo-switch-2): about $300 a week for remote access, or $35,000 to buy, as of June 2025.
- [FinalSpark Neuroplatform](https://finalspark.com/neuroplatform): remote organoids through a Python API. Pricing is by request, and a 30 TB recording dataset is free.

The realistic first wetware experiment is closed-loop learning on one simple THRONG task, in the spirit of DishBrain's Pong. Running a THRONG policy on living neurons is not.

## Risks and honest caveats

The biggest risk is mistaking a toy result for progress on the north star. Each risk below has a guard in the plan.

- **Toy results may not transfer.** The toy reproduced the real network's failure, which is encouraging but not proof. Guard: nothing is claimed until the real-network confirmation and its replication pass.
- **Nudges can make a language designed rather than emergent.** Positive signalling forces signals to carry information, though not which signal means what. Guard: the anneal test checks whether the code survives with the nudge off.
- **"Surpassing LLMs" is decades-scale and contested.** LLM labs are pursuing grounding too, through robotics and world models. A small-world agent will not rival LLM breadth. The defensible near-term claim is narrower: grounded reliability inside its own world, at far lower energy.
- **Silent bugs remain the dominant failure mode.** Five configured-but-unwired settings and three training-signal bugs turned up this month alone. Guard: config audit and gradient tests before every launch, and pre-registration before every result.
- **One person plus AI agents, with no outside review.** Guard: once a result replicates, release the code and write a short workshop paper, so others can check it.

## Standing rules for every step

These rules exist because silent bugs and post-hoc readings cost this project months. They apply to every stage above.

- Register the design, bars and analysis script, and commit them, before any registered run. Disclose every peek and every change made after seeing data.
- Run the config audit and the gradient tests with their mutants before every launch.
- Prove each new path on CPU first, at the smallest size that can show the effect.
- Give every Modal launch a spend guard set to its stage cap, and an auto-halt on an early apparatus check.
- Report at gates only: the apparatus check, then the registered result. A failed gate stops the stage until it's reviewed.
- Leave production untouched until a fix has passed in the testbed.

## Decisions M still owns

Will: don't assume any of these. Each one is M's call, and none of them blocks the next month's work.

- [ ] The testable meaning of "surpassing LLMs": grounded reliability in its own world at low energy, or a harder benchmark.
- [ ] Whether to open-source the code and write up the result once communication replicates.
- [ ] Whether to apply to Intel's neuromorphic community after the Lava port test.
- [ ] Whether to hold wetware until a stable language exists (recommended), or fund a one-week Cortical Labs trial earlier.
