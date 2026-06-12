# Phase 17 Subagent Literature Review: Grounding, Occlusion, and Neuro-Symbolic Paths

The following is a synthesis of five independent literature reviews conducted by the research subagents, focusing on the theoretical underpinnings for Phase 16.6 and Phase 17/18.

## 1. Information Occlusion and Channel Necessity
*   **Mechanisms:** Standard occlusion mechanisms in Dec-POMDPs include Ray-Casting/Line-of-Sight blocking, Fog of War, and Asymmetric Referential Games.
*   **Impact:** Occlusion increases sample complexity and makes symbol grounding harder, forcing temporal/social grounding rather than simple visual grounding.
*   **The Crucial Caveat:** **Occlusion alone is insufficient to force genuine semantic content.** Agents under pure occlusion often converge on degenerate "cheap talk." To force true semantics, literature dictates adding **Information Bottlenecks (e.g., VQ-VIB)** and **Reconstruction/Metabolic Constraints**. 
*   **THRONG Status:** We are uniquely positioned to succeed here. THRONG already implements a strict 64-code VQ Bottleneck and a Thermodynamic Epistemic Gate. Adding Phase 16.6 Barrier Occlusion completes the exact triad of constraints required by the literature.

## 2. Cross-Attention Receiver Collapse
*   **The Lazy Agent Problem:** Receivers ignore the communication channel (attention weights collapse to zero) when they can achieve high rewards relying purely on local policies.
*   **Redundant Perception:** There is strong evidence that **redundant local perception causes cross-attention weights to collapse** even when the channel contains valid semantic content. The optimization landscape favors the shorter, more stable gradient paths of local processing over the complex, delayed dependencies of cross-agent communication.
*   **Mitigation:** The primary mitigation is **Task Design (Breaking Local Sufficiency)**—structuring the environment so local perception is explicitly insufficient. This perfectly validates Cam's proposed Phase 16.6 Barrier Occlusion.

## 3. Causal Language Grounding Verification
*   **Frozen Counterfactual Intervention:** This is the literature gold standard. Fixing the receiver policy and intervening on the channel definitively proves whether a token is causally parsed. (This confirms our current methodology in `causal_intervention.py` is correct).
*   **NPMI vs Causal Divergence:** The literature explicitly warns of the exact phenomenon we saw with Token 44. NPMI indicates high correlation, but causal intervention shows an ATE of zero. This happens when the receiver relies on shared environmental biases (direct vision) while the sender accurately describes the object. Causal intervention serves as the "ground truth" to debunk these spurious NPMI correlations.
*   **ATE Thresholds:** There is no universal threshold (like p < 0.05). The ATE must safely exceed a minimum sensitivity threshold defined relative to the baseline task variance. Our target of `ATE > 0.05` on probabilities is structurally sound.

## 4. Neuro-Symbolic AGI Architecture (Phase 18+)
To move from grounded perception to abstract reasoning (e.g., arithmetic, logic), the literature from 2022-2026 focuses on explicit hybrid architectures:
*   **The Split Workload:** Deep neural networks (like THRONG's GRU) handle high-dimensional, noisy continuous perception, while symbolic engines handle long-horizon planning and causal inference.
*   **Integration Points:** 
    *   *Probabilistic Logic Shields:* Intercepting neural action distributions and projecting them into a safe discrete subset.
    *   *Neuro-Symbolic Action Masking (NSAM):* Symbolic rules compute a binary mask applied to continuous logits before the softmax.
    *   *Differentiable Logic:* Relaxing Boolean predicates into continuous variables (e.g., 0.85 probability that `is_holding` is true).

## 5. Recursive Environment Complexity
*   **Autocurricula:** Tool-use and environment-modification (e.g., OpenAI Hide-and-Seek, DeepMind XLand) create "ecological inheritance." Agents program their own increasingly difficult training data by modifying the environment.
*   **Metrics for Scaling:** Because internal rewards are often zero-sum, cognitive scaling must be measured by **zero-shot generalization and transfer learning sample efficiency** on diverse, out-of-distribution downstream tasks, or by tracking the emergence of novel complexity phases. 
*   **THRONG Status:** The Phase 16.5 `Build` action lays the foundation for exactly this type of autocurriculum. If blues build barriers to block reds, they create dynamic navigational puzzles for themselves.
