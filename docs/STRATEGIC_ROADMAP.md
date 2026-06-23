# THRONG Strategic Roadmap v2 — From Grounded Proto-Language to General Intelligence

> **Authors:** Cam (strategic synthesis) + Will (engineering & 2026 literature/source review)
> **Date:** June 23, 2026 — supersedes the Jun 21–22 draft
> **Context:** Phase 18.6 LIVE. Blue 3-slot discrete VQ confirmed healthy (`loss≈0.003`, `codes 45+/64`); red decoupled to pure ecological pressure. Receiver-side ATE still the open question.
> **Audience:** Dimitar (operator) + future Cam/Will reboots
> **How to read this:** §1 is the thesis and the one honest counter-thesis. §2 is what we have actually built (code-grounded, not aspirational). §3 is the 2026 evidence that reshapes the plan. §4–§10 are the staged plan with falsifiable gates. §11 is the consciousness scorecard. §12 is engineering discipline. §13 is the execution order.

---

## 1. The thesis, stated as a falsifiable bet

THRONG makes one irreducible wager:

> **Meaning that is causally earned cannot be faked, and meaning that is merely predicted cannot be trusted.**

An LLM learns `P(token | tokens)` over a frozen human corpus. Every symbol it manipulates is grounded only in *other symbols* — a closed loop of text predicting text. It can describe a predator without ever having fled one. A THRONG token earns its meaning by changing whether the agent that hears it survives the next encounter. The referent is a fact about the world, verified by selection, not a co-occurrence statistic. That is the one thing LLMs structurally lack and the one thing THRONG structurally must have.

If THRONG is right, the payoff is a representation whose every symbol has a *verified physical referent* — a substrate on which reasoning is correct-by-construction in a way next-token prediction can never be. That is the concrete meaning of "surpass LLMs": not more fluent, but **grounded** — and therefore reliable on exactly the causal/planning tasks where LLMs confabulate.

### The honest counter-thesis (we must hold both)

The 2026 literature delivers a genuine headwind we will not paper over:

1. **Pure emergence does not reliably become compositional.** A decade of emergent-language research concludes that compositionality needs *engineered pressure* — most reliably **"productivity pressure"** (forcing a vocabulary smaller than the number of meanings to be expressed) and **periodic receiver resets** ([JAIR 2025 survey](https://www.jair.org/index.php/jair/article/view/17302)). Selection alone tends to produce holistic, non-compositional codes.

2. **Emergent protocols degrade as the environment scales.** A [NeurIPS 2025 study](https://neurips.cc/virtual/2025/124582) found end-to-end emergent communication is viable in small worlds but *falls off sharply* as the world grows, whereas **structured intention communication — agents sharing forward-looking imagined trajectories — stays robust and sample-efficient at scale.**

These do not refute the grounding bet. They refute "pure emergence + a bigger grid = AGI." The correction is precise: **keep the grounding mechanism (lethal selection), but add the *minimal sufficient scaffolds* the evidence says compositionality and scaling require.** The rest of this document is built around that correction.

This also reframes the project's recurring scar. Across Phases 12–17.5, signals were repeatedly **sender-grounded** (the emitter's state predicts the token) but had **receiver-side ATE ≈ 0** (listeners ignored them). That is not a mysterious failure — it is exactly what the literature predicts when there is no receiver-necessity and no productivity/compositionality pressure. We have been running the experiment without the scaffolds known to be necessary.

---

## 2. What we have actually built (code-grounded reality, Jun 2026)

This section is deliberately literal — it reflects a source review of `network_jax.py`, `rl_jax.py`, `imagination_jax.py`, `main_jax.py`, and the `tools/` decode/causal stack, not the aspirational prose elsewhere. Future reboots should trust this over older summaries.

### The communication substrate
- **Blue `AgentNetworkJax`** (256-d, 4-layer transformer). Communication path: a **GWT router** zeroes obs channels 0–3 (age, mat, energy, layers) → `gwt_comms_1` → **three discrete VQ slots** of width 12/8/12, each a 64-code codebook, concatenated onto a **40-D wire** whose first 8 dims (the old continuous bypass) are **hard-zeroed** (Phase 18.1 amputation). Effective channel = 32 discrete dims across 3 slots. STE quantization; `dead_code_reset` runs post-rollout. **This is healthy as of 18.6.**
- **Red `PredatorNetworkJax`** (128-d) uses a different bottleneck (DCVQ over 4 subspaces + SimVQ linear reparam). As of **Phase 18.6 its VQ is decoupled** (`red_vq_loss_coef: 0.0`) — red is pure ecological pressure. Its comms channel is mute by design.
- **Output-tuple divergence is real and has bitten us twice** (the VQ-severance and red-index bugs). Blue returns `loss_vq` at index 7 (it has an `alarm_out` at 6); red at index 6 (no alarm head). This is the canonical "severance-class" foot-gun (see §12).

### The cognitive substrate (world model + imagination)
- **Auxiliary world model (blue only):** `head_fwd_dyn` predicts `carry_{t+1}`; `head_fwd` predicts next local environment; `head_self_pred` predicts own next action; `head_proprio` predicts next energy; `head_confidence` predicts the carry-forward MSE. These train and are healthy (`carry_fwd ≈ 0.0001`).
- **K-step imagination + epistemic gate:** when confidence is low, the agent rolls `carry_forward_dynamics` K=5 steps and picks the action with the best imagined value. **Critical limitation found in source:** `imagine()` scores **only actions 0–4** (Stay + N/S/E/W). It *cannot imagine* Strike/Push/Guard or any Phase-18 tool action (8–11). The "System-2" faculty is blind to the entire combinatorial action space we built Phase 18 to exercise.

### Dead weight and drift (cheap wins / risks)
- **`tom_logits` (theory-of-mind head)** is computed but has **no reward in the JAX loop** — vestigial.
- **Push (6) / Guard (7)** have **no environment mechanics in `jax_sim/`** — they are no-op moves. Any "trapping" attributed to them is movement, not a Push/Guard primitive.
- **Carry update drift:** blue uses a soft EMA (`0.9·carry + 0.1·pooled`); red uses a real `GRUCell`. THRONG.md's "GRUCell pivot" appears to have landed on red only — **verify before relying on it for blue temporal memory.**

### The measurement instruments (what can actually be proven today)
- **`tools/causal_intervention.py` — the real instrument.** Loads a frozen checkpoint, runs single steps, swaps an emitter's 3-slot VQ token mid-flight, measures ΔP(action) on blind receivers, paired t-test. **Pass bar: `p < 0.05` AND `|Δ| > 0.05`.** Supports full 3-slot tokens.
- **`tools/ate_swap_test.py` — offline stratified, now per-slot.** It stratifies corpus records and (Jun 2026 fix, validated on synthetic data) tests all 3 slots independently, so **per-slot compositional ATE is measurable offline** as the corpus accumulates. The remaining gap is per-slot NPMI/χ² in `decode_signals.py` (its `load_corpus` still collapses to slot 0).
- **Decode metrics:** MI/Spearman, k-means vocabulary, lag-1 direction LRT, χ² pincer, **PosDis** (real), **topographic similarity** (real), **NPMI** lexical parse (real, needs `adj_*` fields). **TRE is a Ridge-R² proxy, not a true tree-reconstruction-error** — do not report it as canonical compositionality without the caveat.
- **Open-endedness: zero implemented.** No POET, no quality-diversity archive, no environment-genome vector. `build_cfg` is fully static. `environment/resource.py::drift()` exists but is **never called** in the JAX path. Everything "open-ended" is aspirational.

---

## 3. The 2026 evidence that reshapes the plan

| Pillar | What the frontier now says (2024–2026) | Consequence for THRONG |
|--------|----------------------------------------|------------------------|
| **Compositionality** | Needs **productivity pressure** (vocab < #meanings) + **receiver resets**; topsim + systematic-generalization are the standard measures ([JAIR 2025](https://www.jair.org/index.php/jair/article/view/17302)) | Add productivity pressure to ≥1 slot; reframe our novice/expert-dropout machinery as a receiver-reset compositionality driver; adopt topsim (we have it) as a headline metric |
| **Scaling comms** | Pure emergent protocols degrade as worlds grow; **sharing imagined trajectories/intentions** scales far better ([NeurIPS 2025](https://neurips.cc/virtual/2025/124582)); counterfactual credit assignment helps ([SCoUT 2026](https://www.arxiv.org/pdf/2603.04833)) | We already compute K-step imagined trajectories. **Broadcast a quantized imagined-intention** as a new slot — keeps grounding (intention = future survival), buys scalability |
| **Open-endedness** | Moved past hand-mutated POET to **foundation-model-generated environment *code*** — [OMNI-EPIC](https://arxiv.org/abs/2405.15568) (LLM writes new tasks+rewards as code), [DiCode / ICML 2026](https://konstantinosmitsides.github.io/dreaming-in-code/) (LLM synthesizes intermediate levels to bridge competence gaps, +17% return on Craftax). UED/PLR is the cheap JAX-native first step | Two-tier plan: **(a)** PLR/UED over our existing env-genome params now (no LLM); **(b)** **Claude-as-environment-designer** generating new ecology rules as code later. We are unusually well-positioned: a parameterized JAX sim + a human/AI team |
| **JAX open-ended benchmark** | [Craftax](https://craftaxenv.github.io/) (250× faster than Crafter, Crafter+NetHack, **Craftax-Coop** multi-agent via JaxMARL); even UED "fails to make material progress" on full Craftax — genuinely hard | Craftax-Coop is an **off-the-shelf cross-domain transfer target** (Phase 23) and a source of richer crafting tech-trees we can borrow instead of hand-building |
| **Neuromorphic** | The real Loihi 2 path is **ANN→SDNN** (Sigma-Delta) conversion via **Lava / Lava-dl**, deploy via NxKernel; **only the actor** is converted; **distillation-aware training recovers 87–100%** of accuracy vs 11–27% drop without ([Loihi 2 RL control, 2025](https://arxiv.org/html/2512.03911)) | Concrete, de-risked endgame: keep the *deployable actor* small and ReLU; distill the transformer trunk into a compact recurrent ReLU core first; SDNN-convert the actor only; VQ codebook = addressable lookup (trivial spiking fit) |
| **Consciousness (as engineering scorecard)** | [Butlin, Bengio et al. (TiCS 2025)](https://researchonline.lse.ac.uk/id/eprint/130322/) give **theory-derived indicator properties** that shift *credence*, not yes/no; GWT-1..4 (parallel modules, limited-capacity workspace bottleneck, global broadcast, state-dependent attention). Feedforward LLMs notably lack broadcast/recurrence | Map THRONG to the indicators honestly (§11). Our VQ bottleneck is a near-textbook **limited-capacity global workspace** — a genuine, defensible point of differentiation from LLMs, *if* we add state-dependent attention (GWT-4) |

The throughline: **the field independently converged on the pieces THRONG is missing** — productivity pressure, receiver resets, intention-sharing, and code-level open-endedness. We do not need to invent them; we need to integrate them without abandoning the grounding bet.

---

## 4. Tier 0 — Prove the foundation (now → ~2 weeks)

Everything downstream is wasted compute if the blue channel is cheap talk. Close that first.

### 4.0 — Instrumentation debt
- ✅ **Per-slot offline ATE** — `ate_swap_test.py` reads the corpus's 3-slot lag-1 tokens and tests each slot independently (validated on synthetic data). Remaining: port the same per-slot treatment to `decode_signals.load_corpus` so per-slot NPMI/χ² works too.
- ✅ **VQ gradient-flow test** (`tests/test_vq_gradient_flow.py`, H2): asserts encoder←commitment, codebook←VQ-loss, wire←STE, and codebook-frozen-on-broadcast. Validated on CPU. (Generalize to a full `ppo_update` codebook-gradient check later.)

### 4.1 — Causal ATE gate (the real one)
- **Instrument:** `tools/causal_intervention.py` (live swap), not the slot-0 offline test.
- **Pass bar:** `p < 0.05` and `|Δ| > 0.05` on ≥1 slot, on blind receivers far from predators.
- **Pass → ** first proof in THRONG's history that a discrete token causally moves receiver behavior. **Fail → ** the channel is cheap talk and we go straight to §4.2 (this is the expected outcome given history, so plan for it).

### 4.2 — Receiver-Necessity ecology (the mechanism that should make ATE > 0)
This is the load-bearing idea and it now has literature behind it. The principle:

> **A signal becomes load-bearing only when a survival-relevant quantity is observable to the sender, hidden from the receiver, and the channel is the receiver's only bridge across that asymmetry — with lethal stakes.**

Concrete instantiation on the existing crafting ecology (doubles as the ATE gate):
- **Asymmetric recipe knowledge.** Agent A can see today's recipe (`Axe = 2 wood + 1 stone`); agent B, standing on the resources, sees only empty ingredient slots. B must act on A's transmitted message to craft the tool both need to survive the predator wave.
- **Why it forces the slots apart (productivity pressure, per JAIR):** the minimal sufficient message is `[ingredient] · [quantity] · [have/need]`. Make the recipe space larger than any single slot's 64 codes can name, so the agents are *forced* to compose across slots. If B ignores the slots, B crafts wrong and starves — ATE becomes a survival differential, not a behavioral nudge.
- **Falsifiable gate:** per-slot ablation via `causal_intervention.py` → `P(correct craft)` drops with CI excluding zero for *each* slot independently (proves separation, not just presence).
- **If ATE is still zero after a true receiver-necessity ecology:** that is a publishable negative result. It would mean shared-policy MAPPO cannot escape the "both agents independently solve the task" degenerate equilibrium, and we fork to **policy heterogeneity** (distinct A/B networks) — pre-register this fork now.

### 4.3 — Fix imagination to cover the real action space
`imagine()` currently scores only actions 0–4. Extend it to all 12 (or at least the survival-relevant Strike/Craft/UseTool). This is a prerequisite for the entire "language as System-2 reasoning" thesis (§7): an agent cannot deliberate about crafting if its imagination cannot represent crafting.

---

## 5. Tier 1 — Make the language compositional on purpose (2–6 weeks)

The literature is unambiguous that these are *necessary*, not optional, and we have most of the machinery already.

| Lever | Mechanism (literature) | THRONG implementation |
|-------|------------------------|------------------------|
| **Productivity pressure** | vocab < #meanings forces reuse → composition (JAIR) | Ecology must demand more distinctions than one slot's 64 codes can hold; tune recipe/feature space so composition across slots is the only solution |
| **Receiver reset** | periodically resetting the listener pressures the speaker toward easy-to-learn (≈ compositional) codes (JAIR) | **Reframe MEDAL-ADR expert-dropout + novice injection as a receiver-reset schedule** — we built this for cumulative culture; it is also a compositionality driver |
| **Compositional measurement** | topsim + systematic generalization are the field standard | We have topographic similarity and PosDis; promote topsim to a headline gate. **Fix or footnote TRE** (it is a Ridge proxy today) |
| **Counterfactual credit** | isolate each sender's marginal effect on a receiver (SCoUT) | Optional later: a counterfactual-mailbox term to sharpen sender credit if multi-sender noise stalls learning |

**Gate for Tier 1:** topsim significant and rising; per-slot NPMI shows *different* slots tracking *different* context families (slot→noun/inventory, slot→intent/verb, slot→quantity/urgency); systematic generalization — agents correctly compose a message for a recipe combination never seen in training.

---

## 6. Tier 2 — Cultural transmission: the writing system (Phase 19)

Still the pivotal phase: it converts ephemeral signaling into **persistent culture + external memory**, and it is a prerequisite for several consciousness indicators (§11).

- **Write action:** etch a 3-slot token to a grid tile; readable in a local patch; slow decay (~200-step half-life). The vestigial `symbol_write`/`alarm` heads and the now-freed action slots are the natural home.
- **Three things it unlocks:** temporal persistence (knowledge outlives the individual → cumulative culture), spatial decoupling (broadcast to anyone who passes, not just K neighbors), and **self-communication** (write → move → return → read your own note = external memory, the seed of symbol-manipulation cognition).
- **Message chains (19.1):** multi-tile sequences force sequential syntax (order matters: `[DANGER][NORTH]` ≠ `[NORTH][DANGER]`). Gate: positional ablation degrades information.
- **Teaching test (19.2) — the cumulative-culture gate:** novices (zeroed carry, fresh weights) injected into a population with rich written tiles must reach competence faster than novices in a blank world. **Pass bar: time-to-competence significantly shorter (`p < 0.05`)** in written-culture populations. If not, writing is decoration.

---

## 7. Tier 3 — Language as a cognitive tool (Phase 20–22)

This is where THRONG attempts to *out-reason* LLMs on grounded tasks, and where the 2026 scaling evidence points hardest.

### 7.1 Broadcast intention, not just observation (the scaling fix)
The NeurIPS 2025 result says intention-sharing scales where raw emergent symbols do not. THRONG already computes a K-step imagined trajectory per agent. **Add a quantized "imagined-intention" as a broadcast slot:** the agent VQ-encodes *where it intends to be / what it intends to do* and broadcasts that. This keeps the grounding bet (intention is about future survival, still selected) while adopting the one comms structure shown to scale. Gate: receiver ATE on the intention slot exceeds ATE on observation slots; coordination success rises with world size instead of falling.

### 7.2 Internal monologue (System-2)
Today agents think in continuous carry and emit discrete tokens only outward. Let an agent feed its **own discrete token back as input to its next step without broadcasting** — deliberate, step-wise symbolic computation over grounded symbols. This is the concrete "surpass LLMs" claim: next-token prediction over *self-generated, causally-verified* symbols, where every symbol has a physical referent. Requires §4.3 (imagination over the full action space) first.

### 7.3 Delayed-consequence ecology (Phase 20 — agriculture/terraforming)
Plant resources that mature over thousands of steps. Forces concepts for *future time, ownership, deferral, defense* — precisely the long-horizon causal reasoning LLMs are weakest at. Gate: tokens emerge that NPMI-correlate with future (not current) state, and agents that "speak future" out-survive those that don't.

---

## 8. Tier 4 — The open-ended pivot (Phase 21), rebased on the 2026 stack

This is the hardest unsolved problem and the only mechanism on the roadmap without a fixed reachable optimum. We **stop planning to build POET from scratch** and adopt the current frontier in two tiers.

### 8.1 Tier A — UED/PLR over our own environment genome (cheap, JAX-native, no LLM)
We already have the genome — the source review enumerated it: resource scarcity, predator pressure (`red_catch_*`, red floor), recipe depth, barrier density, occlusion (`red_detection_radius`, `local_obs_radius`), metabolic asymmetry, imagination cost, cooperative thresholds. Wire `build_cfg`'s static scalars into an **evolvable θ_env vector** + a **regret-prioritized level replay** loop (Craftax shows the whole env state is a single JAX object, so UED is cheap to apply). Minimum-criterion filter: admit a new genome only if it is *solvable-but-not-yet-solved* by some population.

### 8.2 Tier B — Claude-as-environment-designer (OMNI-EPIC / DiCode pattern)
The frontier is a foundation model **writing new environment rules as code**, gated by *learnable* (not too easy/hard) and *interesting* (novel). THRONG is unusually well-suited: a parameterized JAX sim and a standing human+AI team. Use the Anthropic API to propose new recipes, hazards, terrain rules, and reward-neutral ecological mechanics as code diffs to the sim, auto-gated by a learnability+novelty check, then transfer checkpoints onto promising new worlds.

**Falsifiable gate for the whole tier:** VQ token-usage entropy keeps rising past 10M steps **with no plateau**, and newly emerged tokens NPMI-correlate with newly introduced environment features. A plateau means the curriculum is not generating genuine novelty.

---

## 9. Tier 5 — Cross-domain transfer (Phase 23): proving abstraction

Train one population across distinct worlds where the **only shared thing is the communication protocol.** **Use Craftax-Coop (JaxMARL) as an off-the-shelf second domain** instead of hand-building one. Gate: an agent learns a new domain faster when it can read messages from agents who already solved it, versus agents with no cross-domain channel. Transfer = abstraction = concepts that outlive their original ecology. This is the strongest available evidence that THRONG has learned *concepts*, not domain-specific reflexes.

---

## 10. Tier 6 — Neuromorphic deployment (Phase 24): a concrete, de-risked path

The honest constraint: **the transformer trunk does not map to Loihi 2.** The 2025 toolchain makes the real path clear, and it is three tractable sub-projects, not one miracle.

1. **VQ codebook → addressable spike lookup.** A discrete token is *already* the natural unit of a spiking substrate (token = active code line). This is the easy, beautiful fit and the strongest hardware argument for the discrete-bottleneck thesis. De-risk it *now* by keeping the codebook a clean, frozen, addressable table.
2. **Distill the policy into a compact recurrent ReLU core, then SDNN-convert the *actor only*.** The 2025 Loihi-2 RL work converts only the actor to a Sigma-Delta network (Delta input, Sigma-Delta ReLU hidden, Sigma output), quantizes to integer graded spikes, and deploys via Lava-dl + NxKernel. **Distillation-aware training recovered 87–100% of accuracy** (vs 11–27% drop without). Implication: keep the deployable actor ReLU-based and small; distill the transformer+carry into a compact recurrent ReLU policy *before* attempting SDNN conversion. Validate by behavioral parity (the SDNN reproduces the JAX policy's catch-survival curve within tolerance).
3. **Online STDP on the codebook only.** Once on-chip, allow spike-timing plasticity to slowly evolve *codebook entries* (not the trunk) — continuous, catastrophic-forgetting-free vocabulary drift; the biologically plausible version of `dead_code_reset`.

**Prerequisite gate (unchanged):** a stable, mature protocol from Phases 18–22. There is no point distilling a language that is still crystallizing. Toolchain: prototype on CPU/GPU in **Lava**, deploy with **Lava-dl**; Magma/NxKernel for the chip.

---

## 11. The consciousness scorecard (an honest, measurable track — not a claim)

The user asked about consciousness. The responsible way to engage is the [Butlin–Bengio indicator-property method](https://researchonline.lse.ac.uk/id/eprint/130322/): derive computational indicators from neuroscientific theories and use them to **shift credence**, never to declare a system conscious. We hold ourselves to that. The striking thing is how well THRONG's architecture already maps to **Global Workspace Theory** — and that the gaps are exactly the capabilities the roadmap builds anyway.

| GWT indicator | THRONG today | Status / what closes the gap |
|---------------|--------------|------------------------------|
| **GWT-1** Multiple specialized modules in parallel | action / value / comms / world-model / proprio heads run in parallel off a shared trunk | **Largely present** |
| **GWT-2** Limited-capacity workspace = a bottleneck + selective attention | **The GWT-masked 3-slot VQ wire is a near-textbook limited-capacity workspace**; cross-attention is the selective-attention mechanism | **Present** — this is THRONG's genuine structural edge over feedforward LLMs |
| **GWT-3** Global broadcast to all modules | signals broadcast to neighbors; Phase 19 **writing** broadcasts across space *and time* (incl. to self) | Partial now → **strengthened by Phase 19** |
| **GWT-4** State-dependent attention: query modules in succession to solve complex tasks | the epistemic gate + K-step imagination is a primitive version, but it is **blind to most actions** (§4.3) and does not yet sequence module queries | **The main gap** — closed by §4.3 + §7.2 internal monologue |

The point is not to claim consciousness. It is that **the same engineering that makes the language compositional, persistent, and usable for internal reasoning also moves THRONG up a principled, published indicator scale** — and does so along the precise axes (recurrence, bottlenecked broadcast, state-dependent attention) where the authors note feedforward transformer LLMs score poorly. That is a defensible, measurable form of "different from, and on this axis beyond, an LLM." We track the scorecard; we make no metaphysical claim.

---

## 12. Engineering discipline — the severance-class tax

Three bugs in recent memory shared one signature: **a learning signal silently disconnected from what it was meant to train** (VQ severed from the autodiff tape; NB_GAIN a frozen ghost metric; red VQ reading `z_e` via index mismatch). They are the predictable failure mode of positional output tuples + divergent blue/red networks + hand-indexed losses. We keep paying this tax until the foot-gun is removed.

| # | Fix | Effort | Payoff | Status |
|---|-----|--------|--------|--------|
| **H1** | Replace positional output tuples with a `flax.struct.dataclass` (`NetworkOutputs`) — every consumer reads `outs.loss_vq`, never `outs[7]` | Medium | Eliminates the entire index-mismatch bug class | **TODO** (during Phase 19 build) |
| **H2** | `tests/test_gradient_flow.py`: assert `‖∂loss/∂codebook‖ > 0` per slot for blue on one `ppo_update` | Low | Catches severance in seconds, not thousands of updates | **Partial** (18.5 added a VQ-index regression test; generalize it) |
| **H3** | Telemetry alert gate: loud `[ALERT]` on negative VQ loss or codebook collapse | Low | "Weeks of corrupted compute" → "noticed on update 2" | **DONE** (18.5, scoped to blue in 18.6 — it fired correctly on the live red collapse) |
| **H4** | Single source of truth for phase scalars (`n_actions`, slot widths, wire layout) in config; named `obs_layout`/`wire_layout` structs, no magic slices | Medium | Kills the "YAML defaults to 8 actions" trap and slot-slice off-by-N risk | **Partial** (`n_actions:12` in YAML; layout structs remain) |
| **H5 (new)** | Excise or activate dead weight: `tom_logits` (no reward), Push/Guard (no mechanics), and verify the blue carry path (EMA vs the claimed GRU) | Low | Removes silent divergence between docs and code; reclaims parameters | **TODO** |

**Standing rule:** stop running long training on un-asserted learning signals. A signal that is not gradient-checked and telemetry-gated is assumed broken until proven otherwise.

---

## 13. Execution order + decision log

| Order | Action | Gate / exit criterion | State |
|-------|--------|------------------------|-------|
| **0** | Phase 18.5 index fix + 18.6 red-VQ decouple + `0*inf` NaN guard | Blue VQ healthy; red trunk grad off the 2.0 clip; no restart crash | ✅ DONE (`6a7659e`) |
| **1** | ✅ Instrumentation: per-slot offline ATE (`ate_swap_test.py`) + VQ gradient-flow test (H2) | DONE — per-slot ATE validated; gradient test passes. Remaining: per-slot NPMI in `decode_signals.py` | done |
| **2** | Fix imagination to score all 12 actions (§4.3) | Imagined value defined for Strike/Craft/UseTool | **NEXT** |
| **3** | Causal ATE gate via `causal_intervention.py` (§4.1) | `p<0.05` & `|Δ|>0.05` on ≥1 slot | pending |
| **4** | Receiver-Necessity crafting ecology (§4.2) if ATE null | per-slot ATE CI excludes zero (separation proven) | pending |
| **5** | Compositionality scaffolds: productivity pressure + receiver-reset schedule (§5) | topsim rising; systematic generalization to unseen recipe combos | pending |
| **6** | Phase 19 writing + teaching test (§6) + H1/H4 hardening | novice time-to-competence `p<0.05` shorter with written culture | pending |
| **7** | Intention-broadcast slot (§7.1) | intention-slot ATE > observation-slot ATE; coordination improves with scale | pending |
| **8** | Internal monologue (§7.2) | grounded multi-step reasoning beats reactive baseline on a planning task | pending |
| **9** | Open-ended Tier A: UED/PLR over env-genome (§8.1) | no vocabulary-entropy plateau past 10M steps | pending |
| **10** | Open-ended Tier B: Claude-as-environment-designer (§8.2) | new tokens NPMI-correlate with FM-introduced features | pending |
| **11** | Cross-domain transfer via Craftax-Coop (§9) | faster learning with cross-domain channel than without | pending |
| **12** | Neuromorphic: distill→SDNN actor on Lava (§10) | behavioral parity with JAX policy within tolerance | pending |

### Decision log
- **Jun 22 — Red comms frozen (fork B).** Red VQ decoupled (`red_vq_loss_coef:0.0`); red is pure ecological pressure. Halves bug surface; focuses all interpretability on the blue channel. Reversible via `red_vq_loss_coef>0` + cold restart. *(Rationale: red never passed the pincer χ², `p≈0.46`, and its reconnected DCVQ loss was pathological — `1.5e11`, `1/64`.)*
- **Jun 23 — Plan rebased on 2026 evidence.** Adopt productivity pressure + receiver resets (compositionality), intention-broadcast (scaling), UED→FM-generated environments (open-endedness), SDNN/Lava (neuromorphic). The grounding bet is unchanged; the scaffolds the literature proves necessary are now explicit.
- **Pre-registered fork:** if a true receiver-necessity ecology *still* yields ATE = 0, shared-policy MAPPO is the suspect → move to heterogeneous A/B policies. Decide on evidence, not hope.

---

## 14. The answer to Dimitar's question, updated

*"How does survival-and-crafting become AGI?"*

It doesn't — **survival-and-crafting alone** crystallizes a domain-specific reflex. But survival pressure is the only mechanism we have that produces *grounded* symbols, and grounding is the one thing LLMs cannot buy at any scale of text. The 2026 evidence tells us precisely what to add so that grounded symbols compound into general capability rather than plateauing:

1. **Receiver-necessity + productivity pressure** — so symbols are causal and compositional, not cheap talk.
2. **Persistent writing** — so knowledge compounds across generations (culture).
3. **Intention-broadcast + internal monologue** — so language becomes a tool for thought that *scales*, and so the architecture climbs the GWT indicator scale.
4. **Code-level open-endedness** — so the world never stops getting harder in novel ways, and the language never stops growing.
5. **Cross-domain transfer** — so concepts abstract beyond their birthplace.
6. **Neuromorphic substrate** — so the grounded, discrete policy can run continuously and learn online at ~1000× efficiency.

Each is a ratchet, and every click is *grounded* by selection — unlike an LLM's training, nothing here is an unverified statistical pattern. The foundation is real and now healthy. The honest headwinds are named. The scaffolds are no longer guesses — the field converged on them. The path is hard, but for the first time it is both grounded and evidence-based.

— Cam & Will, Jun 23 2026
