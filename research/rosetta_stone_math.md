# Phase 17: The Rosetta Stone - Unsupervised Semantic Translation & Topological Alignment

This research report surveys the current state-of-the-art (2023–2026) in unsupervised manifold alignment and geometric isometry, specifically targeting the translation of a low-dimensional discrete MARL semantic space (a 64-token VQ codebook) into a continuous, high-dimensional LLM embedding space.

## 1. Gromov-Wasserstein & Optimal Transport for Disparate Spaces

The Gromov-Wasserstein (GW) distance is uniquely suited for aligning spaces with unequal dimensionality and incomparable topologies because it compares *intra-domain* geometric structures (e.g., pairwise distance matrices) rather than *inter-domain* point coordinates. 

**Theoretical Framing:**
If $\mathcal{X}$ is the discrete MARL token sequence space and $\mathcal{Y}$ is the continuous LLM embedding space, we define intra-domain cost matrices $C_\mathcal{X}$ (e.g., transition probabilities or co-occurrence frequencies of VQ integers) and $C_\mathcal{Y}$ (cosine distances between LLM token embeddings). 

The GW alignment seeks a coupling matrix $\pi \in \mathbb{R}^{|\mathcal{X}| \times |\mathcal{Y}|}$ minimizing:
$$ \mathcal{L}_{GW}(\pi) = \sum_{i,j,k,l} \left| C_\mathcal{X}(i, j) - C_\mathcal{Y}(k, l) \right|^2 \pi_{i,k} \pi_{j,l} $$

**Implementation Blueprint (JAX):**
Scaling to massive LLM vocabularies is computationally prohibitive with standard GW (which scales at $O(N^3)$). Recent literature (Vedula et al., 2024; Beier et al., 2025) and standard practices utilize **Low-Rank Gromov-Wasserstein (LR-GW)** or **Entropic GW**.
*   **Tooling:** Use the `ott-jax` library (Google's JAX-based optimal transport toolbox) which natively supports auto-vectorization, JIT compilation, and provides `ott.solvers.quadratic.gromov_wasserstein`.
*   **Approach:** Factorize the coupling $\pi = U V^T$ to reduce time complexity to $O(N \cdot r^2)$, where $r$ is the rank bottleneck. This makes aligning a 64-token VQ system to a 50,000+ LLM vocabulary highly tractable.

## 2. Unsupervised Word Translation for Artificial Languages

Classic unsupervised mapping (e.g., MUSE) heavily relies on the assumption of isomorphic structural distributions (i.e., human languages share latent geometric shapes). Alien MARL codebooks will violate strictly structural isomorphism due to varying syntactic constraints and the Minimum Description Length optimization inherent to RL communication protocols.

**Modern Adaptations:**
*   **Continuous Relaxations:** Since MARL tokens are discrete ($x \in \{1...64\}$), directly computing gradients for translation into continuous space is impossible. We represent the MARL tokens as continuous probability distributions over the codebook via **Gumbel-Softmax** or a **Straight-Through Estimator (STE)** during training.
*   **Unbalanced Optimal Transport:** Since there won't be a 1:1 mapping (many LLM tokens map to a single MARL intent), we relax the marginal constraints of the GW coupling (Beier et al., 2025). This allows partial mapping, where the 64 discrete concepts act as "centroids" or "anchors" spanning broader sub-manifolds in the LLM embedding space.

## 3. Geometric Isometry via Cycle-Consistency & Minimum Description Length

To enforce geometric isometry between the disparate sets, we rely on a combination of Cycle-Consistency and the Minimum Description Length (MDL) principle.

### A. Differentiable Discrete Cycle-Consistency
We define two mappings:
*   $F_\theta : \text{MARL}_{discrete} \to \text{LLM}_{continuous}$
*   $G_\phi : \text{LLM}_{continuous} \to \text{MARL}_{discrete}$

The standard cycle consistency loss $\mathcal{L}_{cyc}$ needs modification for the non-differentiable $G_\phi$ step.
**Forward Cycle (Discrete $\to$ Continuous $\to$ Discrete):**
$$ \mathcal{L}_{fwd} = \mathbb{E}_{x \sim \mathcal{X}} \left[ -\log P_{MARL} (x | G_\phi(F_\theta(x))) \right] $$
*(Implemented via a cross-entropy loss against the original 64-token sequence, using Gumbel-Softmax on the output of $G_\phi$).*

**Backward Cycle (Continuous $\to$ Discrete $\to$ Continuous):**
$$ \mathcal{L}_{bwd} = \mathbb{E}_{y \sim \mathcal{Y}} \left[ \| y - F_\theta(G_\phi(y)) \|_2^2 \right] $$
*(We use VQ-STE here: the forward pass assigns the nearest VQ integer, but the backward pass copies gradients directly from $F_\theta$ to $y$).*

### B. Minimum Description Length (MDL) Regularization
In unsupervised alignment contexts (often utilized in grammar induction and shape-matching), MDL prevents "hallucinated" mappings by minimizing the complexity of the alignment rules. 
In the neural context, we formulate this as an information-theoretic bottleneck on the joint embedding geometry. We enforce that the description length of the LLM interpretation $y$ given the MARL source $x$ is minimized. This acts as an "Isometry Loss" constraint:
$$ \mathcal{L}_{iso} = \left\| \text{cos\_sim}(F_\theta(x_1), F_\theta(x_2)) - \text{kernel}_{\mathcal{X}}(x_1, x_2) \right\|_1 $$
where $\text{kernel}_\mathcal{X}$ is a diffusion kernel or normalized co-occurrence frequency between the two discrete tokens in the MARL environment.

## Summary Architecture for Implementation

1.  **Extract Co-occurrences:** Pre-compute the $64 \times 64$ distance matrix $C_\mathcal{X}$ based on MARL agent trajectories. 
2.  **LLM Cost Matrix:** Pre-compute $C_\mathcal{Y}$ for a targeted subset (e.g., top 10,000 words) of LLM word embeddings.
3.  **GW Prior:** Solve for the optimal coupling $\pi$ using `ott-jax`'s Low-Rank Gromov-Wasserstein solver.
4.  **Neural Mapping:** Train MLPs $F_\theta$ and $G_\phi$ using:
    $$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{fwd} + \lambda_2 \mathcal{L}_{bwd} + \lambda_3 \mathcal{L}_{iso} + \lambda_4 \mathcal{L}_{dist}(\pi) $$
    where $\mathcal{L}_{dist}$ guides the MLPs to respect the prior coupling $\pi$ found by OTT.
