# Path to AGI Synthesis: Autocurricula, Neuro-Symbolic, and Grounded MARL

## The Core Thesis
Most scaling-law arguments for Artificial General Intelligence (AGI) rely on feeding increasingly massive datasets of human text to Large Language Models (LLMs). This produces excellent statistical pattern matchers, but they remain intrinsically ungrounded. They do not experience causality, time, or survival pressure; they merely predict the tokens humans use to describe those concepts.

THRONG's path to AGI flips this paradigm. We are building bottom-up, causally grounded intelligence where concepts are forged by thermodynamic survival.

## Stage 1: Grounded Perception & Channel Grounding
Intelligence requires a model of causality. In THRONG, the GRU carry state acts as an internal forward model of the world. Agents pay a real metabolic cost to run this model (the Epistemic Gate). 
The immediate hurdle is **Channel Grounding** (Phase 16.6). Agents must be forced to externalize their causal models into discrete tokens (language) because direct perception is physically occluded. Once causal intervention proves ATE > 0.05, the language is grounded.

## Stage 2: Unsupervised Translation (The Rosetta Stone)
Once an alien, grounded language exists, it must be mapped to human concepts. Phase 17 uses `ott-jax` and Gromov-Wasserstein geometric alignment to map the MARL VQ latent sequences to a continuous GloVe 50d space. This provides a bidirectional interface where a frozen LLM can decode grounded world-states into English, creating an AI that answers questions based on physical simulation rather than text prediction.

## Stage 3: The Neuro-Symbolic Bridge
Grounded perception alone does not yield abstract arithmetic. To achieve general reasoning, THRONG's grounded representations must interface with a discrete symbolic reasoning module. 
The neural "body" (THRONG) provides causal honesty and spatial intuition. The "symbolic layer" handles abstract computation, counterfactuals, and axiomatic logic. This hybrid architecture prevents hallucination while enabling high-level abstraction.

## Stage 4: Recursive Complexity via Autocurricula
We do not manually design a million environments. We rely on autocurricula.
1. **Adversarial Co-evolution:** Predator vs Prey arms races naturally scale cognitive demand.
2. **Environment Building:** As agents gain the ability to place objects (Phase 16.5 Build action) and alter topology, they create novel spatial puzzles for each other.
3. **UED (Unsupervised Environment Design):** An adversarial generator creates terrain and challenges optimized for the edge of the agents' current capabilities.

## Stage 5: Neuromorphic Scale (The Hive-Mind)
AGI cannot remain a frozen software snapshot updating via synchronous PPO rollouts.
Phase 19 targets migration to **Spiking Neural Networks (SNNs)** on neuromorphic hardware (e.g., Intel Loihi 2). This provides continuous-time, asynchronous online learning where the metabolic cost of thought is constrained by actual electrical spikes. At this scale, the swarm operates as a decentralized, continuously adapting Hive-Mind.
