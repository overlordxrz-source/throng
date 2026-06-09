# Phase 17: NPMI Lexical Parser Spec

*Status: Drafted (Pre-Implementation)*  
*Purpose: Classify emergent VQ tokens into syntactic categories (nouns, verbs, adverbs) without human supervision to track the emergence of combinatorial grammar.*

## Core Design
The methodology relies on computing Normalized Pointwise Mutual Information (NPMI) between individual codebook tokens ($z_i$) and orthogonal, non-overlapping axes of the environment/agent state.

### 1. Noun Classification (Entity-Bound Correlates)
**Axis: $V_{noun}$**
- **Target Feature:** Co-occurrence of tokens with persistent static entities.
- **Metric:** $V_{noun}(z_i) = \text{NPMI}(z_i, \text{Adjacent}(Entity))$
- **Implementation:** Check adjacency against grid layers (`blue_bg_map`, `red_map`, barrier, resource). A high score indicates the token functions as a noun or entity referent.

### 2. Verb Classification (Action-Bound Correlates)
**Axis: $V_{verb}$**
- **Target Feature:** Co-occurrence of tokens with the emitter's own discrete actions.
- **Metric:** $V_{verb}(z_i) = \text{NPMI}(z_i, \text{Action}(a_j))$
- **Implementation:** Computed using the probability distribution over the 9 action logits (`Strike`, `Push`, `Build`, `Guard`, etc.) at the timestep of emission ($t$ or $t+1$). A high score indicates an imperative or action verb.

### 3. Adverbial Classification (Spatial/Directional Correlates)
**Axis: $V_{adverb}$**
- **Target Feature:** Co-occurrence of tokens with cardinal geometry and proximity.
- **Metric:** $V_{adverb}(z_i) = \text{NPMI}(z_i, \text{blue\_bear\_bin}) + \text{NPMI}(z_i, \text{blue\_dist\_bin})$
- **Implementation:** Using cardinal quadrant bins (N/S/E/W) and proximity brackets. Tokens scoring high here (but low on Noun/Verb) represent pure spatial markers (e.g., "southward").

---

## Token Role Assignment & The "Composite" Metric

Tokens are categorized by thresholding their vectors across the three axes:
*   `noun`: High $V_{noun}$, low others
*   `verb`: High $V_{verb}$, low others
*   `adverb`: High $V_{adverb}$, low others
*   `composite`: High on **2 or more** axes simultaneously (e.g., high $V_{noun}$ + high $V_{verb}$ = "barrier build")
*   `unclassified`: Below threshold on all axes

### The Phase 16 Grammar Metric
The aggregate count of **`composite`** tokens is our primary metric for combinatorial grammar. If the frequency of `composite` proto-sentences rises significantly following the Phase 16.5 environmental enrichment (grounding), it provides formal evidence that the communication channel has scaled from singular referents to compressed multi-concept utterances.
