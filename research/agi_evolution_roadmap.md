# The Path to Grounded AGI: Complexity, Not Editing

## Why We Don't "Let LLMs Edit Their Code" to Achieve AGI

A common intuition is that if an AI can edit its own weights or code in an infinite recursion loop, it will rapidly become a superintelligence. But intelligence requires a **grounding vector**—a direction to evolve toward. 

If you let an ungrounded LLM edit its own code without an external physical metric for success, it will optimize for what it already knows: generating plausible text that looks right to human evaluators, or collapsing into an infinite loop of degenerate mathematical efficiency. It won't become a rocket-building superintelligence because it has no physical concept of "rocket," "gravity," or "building."

**Human intelligence didn't evolve because we decided we wanted to be smart.** It evolved because the physical world was adversarial, complex, and unyielding. We developed a larger brain volume and language because those mutations allowed us to coordinate to hunt mammoths, predict seasons, and survive harsh winters. The intelligence was a *pragmatic hack* to survive.

## "Don't design the intelligence, design the pressure."

To surpass LLMs, we don't try to match their architecture or edit their weights. We scale the **survival pressure** and the **complexity of the environment**.

### How do we know what environments to build?
Do we need a million different environments for each use case? No. We need **one** environment that is deeply combinatorial and governed by consistent physical laws.

Evolution only gave us one environment: Earth. But Earth has:
1. **Scarcity** (energy/food is hard to get).
2. **Physics** (objects have mass, momentum, permanence).
3. **Other Agents** (predators, prey, cooperators, competitors).
4. **Combinatorial Mechanics** (wood + stone + vine = axe).

Instead of building a "math environment" and a "language environment," we build a survival environment where abstract reasoning becomes necessary. When agents realize that moving `3` blocks to build a wall is mathematically safer than moving `2` blocks, they will develop an internal concept of `3 > 2`.

## The Interface: How Will Grounded Agents Talk to Us?

If you ask an LLM "What is 1+1?", it answers "2" because it has read that sequence in billions of human documents.

If you ask a THRONG agent "What is 1+1?" through our Rosetta Stone interface:
1. The English text "1+1" is translated into the agent's internal geometric vector for `[Gather Resource] + [Gather Resource]`.
2. The agent runs a forward pass through its world model (imagination) and predicts the state: `[Possess 2 Resources]`.
3. The resulting state is translated back through the Rosetta Stone into English: "2".

The agent isn't reciting a fact; it is **simulating physics**. When an agent explains "why x = y," it will be pointing to the causal mechanics of its universe, not to a Wikipedia article.

## When Do We Move to Neuromorphic Chips?

We request access to neuromorphic hardware (like Intel Loihi 2 or SpiNNaker) when we hit the **Synchronous Matrix Math Wall**.

Right now, THRONG runs in JAX on Blackwell B200 GPUs. GPUs are incredibly fast, but they simulate time in discrete, synchronous "ticks," and they are energy-blind. 

Real intelligence evolved in **continuous time** under strict **energy constraints**. Neuromorphic chips use Spiking Neural Networks (SNNs), meaning neurons only fire (and consume energy) when they have something to say. 

We will transition to neuromorphic hardware when:
1. We need agents to learn continuously online without "episodes" or discrete rollout steps.
2. We want the metabolic cost of thinking to be defined by actual electricity consumption on the chip, rather than a simulated software penalty (`MetabolicTax`).

## The Roadmap

1. **Phase 16 (Current):** Prove that agents are using communication causally to survive (The Causal Intervention Test).
2. **Phase 17:** The Rosetta Stone. Translate their 64-token alien language to English.
3. **Phase 18:** Combinatorial Tool Use. Introduce mechanics where agents must combine objects (e.g., Block A + Block B = Wall).
4. **Phase 19:** Continuous Time SNNs (Neuromorphic). Move the agents off GPUs and onto energy-constrained spiking hardware.
