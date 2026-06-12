# Phase 17 Subagent Research Findings

*Compiled: June 2026*
*Context: Preparing for Unsupervised Semantic Translation (The Rosetta Stone) and Hive-Mind Aggregation.*

## 1. Verifying Vocabulary Stability (Semantic Crystallization)
*Reported by the Semantic Stability Researcher*

When a MARL system recovers from a vocabulary collapse (via a deliberate "burn-off"), researchers use several longitudinal tests to verify that the new token semantics won't drift:
- **Mutual Intelligibility:** Agents are evaluated against frozen historical checkpoints. If an agent at Epoch $N$ can communicate with Epoch $N-100$, semantics have crystallized.
- **Representational Similarity Analysis (RSA):** Tracks topological alignment between internal meaning space and discrete message space over time.
- **Diachronic Codebook Tracking:** Probes are attached to codebook vectors. Crystallization is confirmed when probe accuracy for predicting environmental concepts plateaus near 100%.
- **Conclusion:** The "Forget-and-Relearn" paradigm (our burn-off) forces the new communication layer to map mature perceptual concepts cleanly onto discrete tokens, resulting in a much more stable proto-language than the initial attempt.

## 2. Causal Validation of Tokens (The ATE Standard)
*Reported by the MARL Causality Researcher*

- **The Standard:** General literature rarely agrees on an absolute threshold, but the specific rigor of THRONG requires an Average Treatment Effect (ATE) > 0.05 and $p < 0.05$ to claim causal grounding over mere correlation.
- **NPMI vs. Causality:** Literature highlights that Normalized Pointwise Mutual Information (NPMI) frequently produces high correlation scores for tokens that are functionally ignored by the receiver's policy. This validates our Phase 16 findings where Token 3 (Strike) showed high NPMI but an ATE of 0.0000. Causal intervention is mandatory.

## 3. Untranslatable Tokens in Gromov-Wasserstein Alignment
*Reported by the Optimal Transport Researcher*

- **The Phenomenon:** When an emergent token occupies a relative position with no topological equivalent in the human language space (GloVe), the GW optimization places it at a "geometric compromise"—often an interstitial void or manifold gap.
- **Detection:** Untranslatable tokens exhibit a **diffuse coupling (high entropy)** in the transport plan matrix, spreading probability mass across multiple disparate regions of GloVe space.
- **Scientific Value:** These tokens define the boundaries of the Platonic Representation Hypothesis, proving that agents optimizing distinct RL tasks develop fundamentally alien ontologies. GW entropy provides a quantifiable boundary for AI interpretability.

## 4. Hive-Mind Aggregation (State Pooling)
*Reported by the Swarm Aggregation Researcher*

- **Mechanism:** Multi-Head Attention Pooling (MHAP) is the SOTA for aggregating variable-length hidden GRU states from a population. It captures both local temporal dependencies and global context while remaining permutation invariant.
- **Mean-Field Theory:** Aggregating statistical features (means/variances) reduces complex N-agent states to a 2-agent problem, vastly reducing query bandwidth.
- **Failure Modes:** Aggregation breaks down when agents possess divergent epistemic states (e.g., multimodal beliefs). Simple mean-pooling blends these into nonsensical averages (collaborative oscillation). Future Phase 18 architectures must use consistency-based plan selection or EpiGNN-style distinct memory traces.

## 5. Neuromorphic Convergence (Continuous Online Learning)
*Reported by the Neuromorphic Researcher*

- **SNNs & MARL:** Platforms like Intel Loihi 2 and SpiNNaker 2 process asynchronous, event-driven spikes, removing high-overhead global synchronization for multi-agent swarms.
- **VQ to STDP Mapping:** There is a robust mathematical mapping between Vector Quantization and Spike-Timing-Dependent Plasticity (STDP). In SNNs with lateral inhibition, STDP performs equivalent clustering to VQ, allowing hardware to dynamically learn codebooks.
- **Catastrophic Forgetting:** STDP's inherent sparsity and local updates protect previously established memory traces. Neuromorphic hardware provides biological plasticity-stability mechanisms, enabling true "always-on" environmental adaptation at extreme energy efficiency.
