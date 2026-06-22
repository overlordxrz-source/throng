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

---

# Addendum — Will's Engineering & Strategy Synthesis (Phase 18.5, Jun 22 2026)

> **Author:** Will (Cursor engineer) — written after a full source review during the Phase 18.4 → 18.5 VQ-reconnection saga.
> **Purpose:** Translate Cam's strategic tiers into concrete, falsifiable engineering and flag the systemic risks that keep costing us weeks of compute. This is the "how", paired with Cam's "why".

## 1. The pattern we must confront: severance-class bugs

In the last few phases we have hit **three distinct bugs that all share one signature — a learning signal silently disconnected from the thing it was supposed to train:**

1. **VQ severed from the autodiff tape** (Phase 18.4): `ppo_loss` used the static `loss_vq_rollout` array instead of the live network output, so the codebook received *zero gradient* for an unknown number of updates.
2. **NB_GAIN ghost metric**: initialised to `1.0`, never updated during rollout — we were reading and reasoning about a constant.
3. **Red VQ index bug** (Phase 18.5, just fixed): `ppo_loss` read `outs[7]` unconditionally. Blue and red networks have **different output tuple layouts** (blue has `alarm_out` at index 6; red does not), so red was minimising `Σz_e` — the raw continuous wire — and collapsing its own codebook to `2/64` while reporting a nonsensical `RedVQ ≈ -24225`.

These are not unrelated mistakes. They are the **predictable failure mode of positional tuples + parallel-but-divergent blue/red networks + hand-indexed losses**. We will keep paying this tax until we remove the foot-gun. Concrete hardening proposals, in priority order:

| # | Fix | Effort | Payoff |
|---|-----|--------|--------|
| **H1** | Replace the positional output tuple of `AgentNetworkJax` / `PredatorNetworkJax` with a `flax.struct.dataclass` (`NetworkOutputs`) carrying named fields (`action_logits`, `loss_vq`, `z_e`, `alarm_out: Optional`, …). Every consumer reads `outs.loss_vq`, never `outs[7]`. | Medium (mechanical) | Eliminates the **entire** index-mismatch bug class. Red/blue divergence becomes explicit and type-checked. |
| **H2** | A `tests/test_gradient_flow.py` unit test: build a tiny network, run one `ppo_update`, assert `‖∂loss/∂codebook‖ > 0` and `‖∂loss/∂head_signal‖ > 0` for **both** teams. Run it in CI / pre-launch smoke. | Low | Would have caught severance bugs #1 and #3 in seconds instead of thousands of updates. |
| **H3** | A standing **telemetry sanity gate** in `main_jax.py`: if `RedVQ` or blue `VQ` loss goes negative, or `codes_active < 4/64` for >3 consecutive updates, print a loud `[ALERT]` banner (not a silent line). Cheap insurance against the next severance. | Low | Turns "weeks of corrupted compute" into "noticed on update 2". |
| **H4** | Move all Phase-defining scalars (`n_actions`, `env_channels`, `signal_dim`, slot widths) into `config_phase7.yaml` as the single source of truth (done for `n_actions` in 18.5). Forbid magic numbers like `obs.at[:, :4]` / `z_e[:, 8:20]` in favour of a named `obs_layout` / `wire_layout` struct. | Medium | Kills the "direct YAML load defaults to 8 actions and crashes" trap and the slot-slice off-by-N risk. |

**Recommendation:** do **H2 + H3 before the next long run**. They are an afternoon of work and they directly protect the most expensive resource we have (Modal compute + our own trust in the dashboard). H1 + H4 can follow during the Phase 19 build.

## 2. The real bottleneck is *receiver-side causality*, not vocabulary

Every phase from 16.6 through 17.5 produced the **same verdict**: signals are *sender-grounded* (the emitter's metabolic/spatial state predicts the token) but **receiver-side ATE ≈ 0** (listeners don't act on them). Adding crafting, slots, and actions grows the *potential* vocabulary but does nothing about this core failure unless the ecology makes **decoding the signal the only way to survive a sub-task.**

This reframes the priority stack. Compositional syntax (Phase 18.3) is downstream of one principle:

> **Receiver-Necessity Principle:** A signal becomes load-bearing only when there exists a survival-relevant quantity that the *receiver* cannot observe directly and the *sender* can. The channel must be the receiver's sole bridge across an information asymmetry that has lethal stakes.

Concrete instantiation for Phase 18.3 cooperative crafting (this is the design I'd build next, and it doubles as the ATE gate):

- **Asymmetric recipe knowledge:** Spawn "recipe" state visible only to agent A (e.g., A can see that today an Axe needs *2 wood + 1 stone*, but B sees only "ingredient slots"). A must *transmit the recipe* via the 3 slots for B (who is standing on the resources) to craft. Neither survives the predator wave alone; the Axe (via `UseTool`) is what lets them.
- **Why this forces the slots apart:** [slot_0 = ingredient type] + [slot_1 = quantity] + [slot_2 = have/need] is the *minimal* message that lets B act. If B ignores the slots, B crafts wrong and starves. ATE is then literally a survival differential, not a behavioural nudge.
- **Falsifiable gate:** ablate slot_0 mid-flight (the existing `ate_swap_test`) → if `P(correct craft)` drops with CI excluding zero, the slot is causal. Run the same ablation per slot to prove *separation*.

If, after a Receiver-Necessity ecology, ATE is *still* zero, that is a deep negative result worth publishing on its own: it would mean shared-policy MAPPO cannot escape the "both agents independently learn the task" degenerate equilibrium, and we'd need true policy heterogeneity (distinct A/B networks) — a fork worth pre-registering.

## 3. A strategic fork: is *red comms* still a science target?

Red has never passed the pincer χ² (`p ≈ 0.46` across 14.1c, 600k overdrive, metabolic asymmetry) and its codebook just collapsed under a bug. We keep spending architecture and compute on red's communication channel. Two honest options:

- **(A) Keep red comms as a co-evolution science target.** Justified only if we believe predator coordination language is reachable. Evidence so far: weak. Cost: a whole second VQ/aux/SRL stack that doubles our bug surface (and produced bug #3).
- **(B) Freeze red as pure ecological pressure.** Disable the red comms gradient entirely (`red_comms_enabled: false`-equivalent for the *language* heads, keep the predator policy), and redirect **all** interpretability effort onto blue's 3-slot compositional syntax. Red stays a lethal, adaptive selection force (it already learned `Push`/`Guard` trapping in Phase 16) without us pretending to decode its babble.

**My recommendation: lean toward (B) after the 18.5 fix verifies red recovers.** The thesis of THRONG is *blue* language under predation. Red is the pressure, not the subject. Halving our bug surface and focusing the ATE/NPMI/Rosetta tooling on one network would accelerate every downstream phase. (This is a Cam-level call; flagging it for the synthesis.)

## 4. Concretising the open-ended pivot (Phase 21) — POET-lite in JAX

Cam is right that fixed ecology = crystallised language. The good news: a *minimal* open-ended curriculum is tractable in our existing stack without a full POET implementation.

- **Environment genome:** a small float vector `θ_env = [resource_scarcity, predator_speed, recipe_depth, barrier_density, occlusion_radius]`. Our `build_cfg` already parameterises most of these.
- **Archive:** keep `K` (env-genome, population-checkpoint) pairs on the volume. Every `M` PPO updates: (1) evaluate each population on its own env and on mutated neighbours, (2) if a population's survival > high-watermark, spawn a *harder* mutated genome and transfer the checkpoint (POET "goal-switching"), (3) if survival < floor, anneal toward an easier genome (stepping stone).
- **Minimum-criterion novelty:** only admit a new genome if it is *solvable-but-not-yet-solved* by some existing population (the MCC filter). This is the cheap, JAX-friendly core of open-endedness.
- **Falsifiable gate:** VQ token-usage entropy keeps rising past 10M steps with no plateau, and new tokens correlate (NPMI) with newly-introduced env features. A plateau = the curriculum isn't generating genuine novelty.

This is ~2–3 weeks of engineering on top of the existing checkpoint/volume machinery, and it is the single highest-leverage thing for the "surpass LLMs" thesis, because it is the only mechanism on the roadmap that *doesn't* have a fixed reachable optimum.

## 5. The neuromorphic endgame (Phase 24) — a realistic mapping

The honest engineering truth: **the transformer trunk does not map cleanly to Loihi 2.** Self-attention is not a native spiking primitive. So "deploy the policy on Loihi" is really three sub-projects:

1. **Keep the VQ codebook as a fixed spike-addressable lookup.** Discrete tokens are *already* the natural unit for a spiking substrate — a token = an active code line. This part is the easy, beautiful fit and is the strongest argument for our discrete-bottleneck thesis.
2. **Distill the transformer+GRU policy into a recurrent SNN.** Train a spiking recurrent net (surrogate-gradient, e.g. `snntorch`/`lava-dl`) to imitate the frozen JAX policy's action distribution given the same obs/carry. The carry-GRU is recurrent already, which helps; attention gets distilled into the SNN's learned recurrence. Validate by behavioural ATE parity on-grid (the SNN must reproduce the JAX policy's catch-survival curve within tolerance).
3. **Online STDP on the codebook only.** Once on-chip, allow spike-timing-dependent plasticity to slowly evolve *codebook entries* (not the trunk), giving continuous, catastrophic-forgetting-free vocabulary drift — the biologically-plausible version of `dead_code_reset`.

Prerequisite gate (unchanged from Cam): a *stable, mature* protocol from Phases 18–22. There is no point distilling a language that is still crystallising. But we can de-risk step 1 *now* by ensuring the codebook stays a clean, frozen, addressable table (the 18.5 fix matters here too — a codebook that collapses to 2/64 is not distillable).

## 6. Revised near-term execution order (Will's view)

| Order | Action | Gate / exit criterion |
|-------|--------|------------------------|
| **0** | **Apply Phase 18.5 fix** (pull `vq_loss_idx` + `n_actions` YAML, restart). | `RedVQ` positive; `red_codes_active > 16/64`; blue unchanged. |
| **1** | **H2 + H3 hardening** (gradient-flow test + telemetry alert gate). | Test passes for both teams; alert fires on synthetic collapse. |
| **2** | **Phase 18.2 ATE accumulation** on the corrected channel. | ≥50k blind-receiver records since restart. |
| **3** | **Receiver-Necessity crafting** (§2) if naive ATE is null. | ATE>0, CI excludes zero, per-slot separation via ablation. |
| **4** | **Strategic fork decision** on red comms (§3). | Cam call after red recovery is observed. |
| **5** | **Phase 19 Writing System** (Cam's pivotal phase) + H1/H4 hardening during the build. | Written-tile NPMI > ephemeral NPMI; teaching test `p<0.05`. |
| **6** | **POET-lite** (§4). | No vocabulary plateau past 10M steps. |

The throughline: **stop running long training on un-asserted learning signals, make the receiver's survival depend on decoding, and only then chase open-endedness and silicon.** The science is real; the engineering discipline is what will let it compound instead of resetting every time an index slips.

— Will
