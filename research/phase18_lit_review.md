# Phase 18 Literature Review: Emergent Communication in MARL

## 1. Environmental Designs: Compositional Syntax vs. One-Concept Vocabularies (2024–2026)
Recent research (2024–2026) demonstrates that environmental pressures dictate whether agents develop compositional syntax or fall back on one-concept (holistic) vocabularies:
- **Drivers of Compositionality**: Compositional syntax emerges under specific environmental constraints, notably **Zipfian frequency distributions** (limited data exposure to novel combinations) and **one-to-many communication pressures** (where a single speaker must coordinate with multiple listeners who have varying interests/tasks). Environments simulating cooperative foraging or multi-target coordination (e.g., "Composition through Decomposition" studies) force agents to segment concepts.
- **One-Concept Vocabularies**: In simple referential games or environments without strict bandwidth constraints/heterogeneous listeners, agents tend to develop holistic, one-concept vocabularies. This is highly efficient for narrow tasks but fails to generalize.

## 2. Managing Inventory & State Leaks (The "Metabolic Trap")
In complex environments like Crafter, MineRL, and NetHack, the tendency for agents to broadcast their interoceptive state (e.g., health, hunger, inventory) instead of task-relevant semantic information is widely recognized in literature as **"state leakage"** or **"internal state broadcast"** (analogous to the "metabolic trap" observed in Phase 16).
- **The Problem**: Agents optimize for the simplest path to reduce joint uncertainty. Transmitting a raw "internal signal" (inventory/health) is highly reliable, causing the communication channel to collapse into a continuous proxy of interoceptive state.
- **Solutions**: Recent experiments handle this using **Information Gating** (restricting the flow of internal states to the communication channel) and **Information-theoretic regularization** (e.g., Information Bottlenecks). By penalizing the transmission of raw local state data or forcing object-oriented communication, agents are pressured to communicate semantic, environment-focused information rather than just dumping their inventory/metabolic state.

## 3. Slot-Based VQ Architectures for Multi-Token Messages (Subject/Verb/Object)
Between 2024 and 2026, the integration of **Slot Attention** with **Vector Quantization (VQ)** has become a leading method for generating structured, multi-token emergent messages:
- **Architecture**: Slot attention mechanisms naturally segment visual/environmental observations into discrete, object-centric "slots" (acting as the *Subjects* and *Objects*). The VQ bottleneck then forces these continuous slot representations into discrete, learned symbols (e.g., *Verbs* or relationships).
- **Frameworks**: Prominent frameworks like "AI Mother Tongue" (AIM) and "Vector Quantization: Emergent Language" (VQEL) use VQ-VAEs to allow agents to develop endogenous symbol systems. This naturally induces a Subject/Verb/Object (SVO) framing without hardcoded inductive biases, mapping discrete visual slots to discrete communication tokens.

## 4. The "Common Ground" Problem for Novel Objects
Establishing shared reference (common ground) for novel tools or objects that agents have never communicated about before is a core challenge in multi-agent tool use:
- **Joint Attention & Synchronization**: Agents are trained to synchronize their visual focus on shared referents to ground their symbols interactively.
- **Conversational Repair**: Implementing feedback channels allows agents to issue clarification requests. When an agent encounters a novel object and uses an ambiguous symbol, peers can query the meaning, iteratively establishing a shared reference.
- **Generative Cognitive Modules**: Advanced models use "Visual Theory of Mind." Agents use generative models to "imagine" or caption novel objects internally, attempting to align their internal latent representations with what they predict their partner is seeing, facilitating zero-shot or few-shot communication on novel items.
