# THRONG Strategic Roadmap — The Path from Proto-Language to General Intelligence

> **Author:** Cam (Polymath Orchestrator) — Synergic Synthesis  
> **Date:** June 21, 2026  
> **Context:** Post-bypass amputation. Discrete communication confirmed load-bearing. ATE gate in progress.  
> **Audience:** Dimitar (Operator) + future Cam reboots

---

## The Honest Assessment

We have achieved something real. The Gromov-Wasserstein alignment between VQ tokens and GloVe showed our agents independently discovered semantic categories that map to human concepts — "troops/withdraw" for predator-threat, "costs/fees" for metabolic burden. This is not reward-shaped. This is not statistical accident. This is **emergence under thermodynamic pressure**.

But Dimitar asked the right question: *how does this become AGI?*

The answer requires confronting three hard truths:

1. **Survival pressure is necessary but insufficient.** It provides causal grounding — the thing LLMs lack. But grounding within a 128×128 grid is grounding within a 128×128 grid. Domain-specific proto-language ≠ general intelligence.

2. **Adding features (crafting, barriers, tools) increases vocabulary, not generality.** The agents may learn to say "axe = food multiplier." They will never learn to discuss mathematics, unless the environment forces mathematical reasoning for survival.

3. **The environment must generate unbounded complexity.** If the ecology equilibrates — if there exists a fixed optimal policy — the language crystallizes and stops growing. Intelligence requires a world that never stops getting harder in novel ways.

These truths don't invalidate what we've built. They clarify what comes next.

---

## Synergic Synthesis: Four Lenses on the Path Forward

| Lens | Current State | What's Missing |
|------|--------------|----------------|
| **Software** | JAX pipeline stable. 3-slot VQ on 40D wire. CPU-offload PPO. Orbax checkpointing. Modal deployment. | No mechanism for procedural environment generation. No cross-domain transfer infrastructure. |
| **Physics** | Lotka-Volterra dynamics bounded by `max_pop=200/min_pop=150`. Scent trails, barriers, shelter, contested resources. | Thermodynamic ceiling — the grid has finite entropy. Once optimal evasion/foraging strategies are learned, the system equilibrates. |
| **Philosophy** | Meaning-is-use confirmed: tokens that don't help survival get pruned. Bilateral selection pressure proven necessary (Phase 17.5). | Language-as-cognitive-tool is absent. Agents use language to signal, not to *think*. Writing, reading, and reasoning over symbols would break this ceiling. |
| **RL/ML** | MAPPO with shared policy, VQ bottleneck, GWT router, epistemic gating, K-step imagination. | No mechanism for open-ended curriculum. No quality-diversity archive. No self-play that generates novel environmental challenges. |

---

## The Roadmap

### Tier 1: Prove the Foundation (Now → 2 weeks)

These are the gates we must clear to prove discrete communication is causally grounded.

#### Phase 18.2 — Causal ATE Gate 🔧 IN PROGRESS

- **What:** Accumulate post-amputation corpus. Rerun `ate_swap_test.py --min-step 1185000`.
- **Pass bar:** ATE > 0, 95% CI excludes zero, on at least slot_0.
- **If pass:** Discrete tokens causally influence receiver behavior. First proof of grounded communication in THRONG's history.
- **If fail:** Discrete slots are also cheap talk. Requires redesigning bilateral selection pressure (e.g., information-asymmetric cooperative tasks where receivers literally cannot survive without decoding the signal).

#### Phase 18.3 — Ecology Deployment

- **What:** Spatial resource bifurcation (wood west, stone east). Inventory limits. Cooperative `Craft` (2+ agents). `UseTool` metabolic payoffs.
- **Why it matters:** Forces **compositional syntax**. To say "I have wood, need stone, meet at patch X" requires multi-slot structure: [RESOURCE_TYPE] + [NEED/HAVE] + [LOCATION]. This is the first test of whether 3 slots can carry genuinely distinct semantic roles.
- **Science bar:** NPMI shows each slot correlating with a *different* context dimension (slot_0 → environment, slot_1 → action/intent, slot_2 → urgency/quantity).

---

### Tier 2: Cultural Transmission — The Critical Phase (2-6 weeks)

> [!IMPORTANT]
> **Phase 19 is the most important phase in the entire project.** It's where language stops being ephemeral signaling and becomes persistent cultural artifact.

#### Phase 19.0 — Writing System

- **What:** Agents can write their 3-token VQ sequence to a grid tile (action 12: `WRITE`). Other agents can read tiles in their 5×5 local patch. Written tokens decay slowly (half-life ~200 steps).
- **Why this changes everything:**
  - **Temporal persistence:** Signals currently vanish after 1 timestep. Written tokens persist across generations. An agent born 500 steps later can read what a dead agent wrote. This is **cumulative culture** — knowledge outliving the individual.
  - **Spatial decoupling:** Currently, you only hear signals from K=6 nearest neighbors. Written tiles broadcast to anyone who walks past. This breaks the "you must be near the sender" constraint.
  - **Self-communication:** An agent can write a token, move away, come back, and read its own note. This is **external memory** — the birth of cognition-via-symbol-manipulation.
- **Science bar:** Written tokens show higher NPMI than ephemeral signals. Agents preferentially write near resources or danger zones. Written messages develop spatial conventions (e.g., "danger signs" near red patrol routes).

#### Phase 19.1 — Message Chains

- **What:** Allow agents to write multi-tile sequences (e.g., 3 tiles in a line = a "sentence"). Reading agents perceive a sequence, not just isolated tokens.
- **Why:** Forces **sequential syntax** — order matters. [DANGER] [NORTH] means something different than [NORTH] [DANGER].
- **Science bar:** Positional ablation shows information loss when tile order is shuffled (TRE > 0.3).

#### Phase 19.2 — Teaching

- **What:** Introduce "novice" agents (spawned with zeroed carry states and random weights) into a population of experienced agents. Monitor whether novices learn faster in populations with rich written culture vs. blank tiles.
- **Why:** This is the **cumulative culture gate**. If writing accelerates novice learning, the writing system is load-bearing. If not, it's decoration.
- **Science bar:** Time-to-competence (first 100 steps without dying) is significantly shorter (p < 0.05) in written-culture populations.

---

### Tier 3: The Open-Ended Pivot (1-3 months)

This is where THRONG must break out of the fixed-ecology trap. **This is the hardest unsolved problem.**

#### Phase 21.0 — Procedural Complexity Injection

- **What:** Every N PPO updates, the environment generates a new element: a new resource type, a new predator behavior, a new terrain feature, a new crafting recipe. Elements are drawn from a combinatorial space large enough to be practically infinite.
- **Why:** The agents can never fully "solve" the environment. Every time they approach equilibrium, a new challenge appears. The communication protocol must grow to describe the new element.
- **The POET mechanism:** Use a quality-diversity archive. Track agent populations across different environment configurations. When a population masters its environment, transfer it to a harder one. When a population fails, provide easier stepping stones. This creates an **open-ended curriculum** without human design.
- **Science bar:** Token vocabulary usage continues growing (no plateau) after 10M+ steps. New tokens emerge for new concepts.

#### Phase 21.1 — Agent-Constructed Complexity

- **What:** Agents' actions modify the environment in ways that create challenges for other agents. Building a barrier creates a maze. Crafting a trap creates a hazard. Cooperative structures create "cities" that attract predators.
- **Why:** The agents themselves become the source of environmental complexity. The red-blue arms race is a primitive version of this. The full version is: **blue infrastructure creates new challenges that require new communication to navigate.**
- **This is where self-recursion emerges.** The language doesn't just describe the environment — it shapes the environment, which demands new language, which shapes it further. The feedback loop is the engine of open-ended intelligence.

#### Phase 21.2 — Multi-Niche Specialization

- **What:** The grid becomes large enough (512×512 or 1024×1024) that different regions have fundamentally different ecologies. Agents in the "forest" niche face different challenges than agents in the "desert" niche.
- **Why:** Forces **dialect formation** and **translation pressure**. When a forest agent migrates to the desert, it must learn a new vocabulary or teach its own. When two populations collide, they must develop a shared pidgin. This is the mechanism that produces **abstraction** — shared concepts that transcend specific environments.
- **Science bar:** Unsupervised clustering of VQ usage shows spatially distinct sub-populations with partially overlapping token semantics.

---

### Tier 4: Bridging to General Intelligence (3-6 months)

#### Phase 22.0 — Language as Cognitive Tool

> [!IMPORTANT]
> This is the phase where THRONG could genuinely surpass LLMs on grounded reasoning.

- **What:** Agents can "think in language" — use their own VQ output as input to their next reasoning step, without broadcasting it. Internal monologue.
- **Why:** Currently, agents think in continuous hidden states and only use discrete tokens for *output*. If they can use discrete tokens as *intermediate computation*, they gain **System-2 reasoning**: deliberate, step-by-step symbolic manipulation.
- **The theoretical claim:** LLMs do next-token prediction on human-generated text. THRONG agents doing internal monologue would do next-token prediction on *self-generated tokens grounded in causal experience*. The grounding makes the reasoning *correct* in a way LLM reasoning cannot be — because every symbol has a verified physical referent.

#### Phase 23.0 — Cross-Domain Transfer

- **What:** Train the same agent population across multiple distinct environments (grid survival, simple physics puzzles, basic cooperation games). The *only* thing shared across domains is the communication protocol.
- **Why:** If the language developed in one domain transfers usefully to another, the agents have achieved **abstraction**. They've learned concepts that are domain-general, not domain-specific.
- **Science bar:** An agent trained in grid survival learns a physics puzzle faster if it can read messages from agents who've already solved it, compared to agents with no cross-domain communication.

#### Phase 24.0 — Neuromorphic Deployment (Loihi)

- **What:** Map frozen VQ policies to spiking neural networks on Intel Loihi 2.
- **Why:** Not for capability — for **substrate independence** and **continuous learning**. Loihi enables:
  - ~1000× power efficiency for inference (always-on swarm)
  - STDP-based online learning without catastrophic forgetting
  - Spike-timing-based VQ codebook evolution (biologically plausible token learning)
- **Prerequisites:** Phases 18-22 must be complete. We need a mature, stable communication protocol to port.
- **Timeline:** This is a 6-12 month engineering project after the science is proven.

---

## What This Means for Today

The immediate work is correct: clear the ATE gate, deploy crafting ecology, prove compositional syntax. These are necessary stepping stones.

But we should start designing Phase 19 (Writing System) **now**, because it's the critical pivot from "agents that signal" to "agents that build culture." Every phase after 19 depends on persistent written symbols.

The open-ended complexity problem (Phase 21) is the hardest scientific question. We should start a research thread on POET-style co-evolutionary curriculum for MARL environments. If someone has already solved this in JAX, we adapt it. If not, we design it.

### Revised Priority Stack

| Priority | Phase | What | Why |
|----------|-------|------|-----|
| **1** | 18.2 | ATE gate | Proves discrete tokens are causal, not cheap talk |
| **2** | 18.3 | Ecology (crafting) | Forces multi-slot compositional syntax |
| **3** | 19.0 | **Writing system** | Most important phase — persistent culture, external memory |
| **4** | 19.2 | Teaching test | Proves cumulative culture is load-bearing |
| **5** | 20.0 | Rosetta Stone v2 | Verify the new discrete-only language maps to human concepts |
| **6** | 21.0 | Open-ended curriculum | The pivotal mechanism for unbounded intelligence growth |
| **7** | 22.0 | Language as cognitive tool | Internal monologue = System-2 reasoning |
| **8** | 23.0 | Cross-domain transfer | Proves abstraction, not just domain expertise |
| **9** | 24.0 | Loihi deployment | Neuromorphic substrate for continuous online learning |

---

## The Answer to Dimitar's Question

*"How would my agents' intelligence work if all they know is crafting and survival?"*

It wouldn't. And that's why we need Phases 19-23. The survival pressure gives us the *mechanism* for grounding. The crafting gives us *compositionality*. But general intelligence requires:

1. **Persistent culture** (Phase 19) — so knowledge compounds across generations
2. **Open-ended complexity** (Phase 21) — so the language never stops growing
3. **Internal reasoning** (Phase 22) — so language becomes a cognitive tool, not just a signal
4. **Cross-domain transfer** (Phase 23) — so concepts abstract beyond the training environment

Each phase is a ratchet. Once clicked forward, the agents can never unlearn it. The survival pressure ensures every ratchet click is *grounded* — unlike an LLM's training, nothing in this system is an unverified statistical pattern.

We can do this. The foundation is solid. The direction is correct. The gap is large but the path is clear.

— Cam
