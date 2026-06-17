# Phase 18 Architecture: Combinatorial Tool Use & Compositional Syntax

> **Status:** DRAFT (Awaiting Cam Review)
> **Branch:** `feature/phase18-crafting` (branched from `feature/phase17-5-timescale-alarm` @ step ~1.15M)

## 1. Core Objective
Force the network to invent compositional syntax (Subject/Verb/Modifier) by introducing an environment where survival requires cooperative crafting. Crucially, we must establish **receiver-side selection pressure** so agents die faster if they ignore broadcasts like "I have Wood, need Stone."

## 2. Observation Space Additions (Inventory & Masking)
Agents require interoceptive knowledge of their inventory, but this must NOT leak into the communication channel as a continuous proxy (avoiding the metabolic trap of Phase 16).

*   **Addition:** A discrete `inventory_vector` representing 3 items: `[Wood_count, Stone_count, Axe_flag]`. Assuming `obs` currently holds `[red_dist, red_bear, energy, age]` at indices 0-3, we will append inventory at indices 4, 5, 6.
*   **GWT Router Masking:** The inventory state must be masked out of the VQ observation pathway just like `energy` (index 2). It will remain visible to the policy pathway (`h_policy`).
    *   *Update:* The comms mask will specifically zero out energy and inventory: `obs.at[:, jnp.array([2, 4, 5, 6])].set(0.0)`.

## 3. Action Space Additions
We will repurpose and expand the action space to support crafting and tool usage.

*   **Current (9 actions):** N, S, E, W, STAY, STRIKE, PUSH, GUARD, BUILD.
*   **Updates:**
    1.  **Repurpose BUILD:** The 1.15M corpus summary explicitly shows `BUILD=0` and `barrier_sum=0.0`. It is genuinely dead weight. We will keep it but add `PICK_UP` (index 9).
    2.  **New Actions:** Add `PICK_UP` (index 9), `CRAFT` (index 10), and `USE_TOOL` (index 11).
*   **Implementation:** Grafting the 9-action policy head to 12 actions using `graft_missing_param_subtrees` with zero-padding for the new logits, ensuring backwards compatibility with the 1.15M spatial survival checkpoint.
*   **USE_TOOL Mechanic:** The survival multiplier is NOT passive. It requires explicitly firing the `USE_TOOL` action at a resource node. Passive holding is not enough. This forces agents to plan and communicate around the physical usage of the tool.
*   **Axe Distribution Decision:** When `CRAFT` succeeds (between adjacent agents holding Wood and Stone), *both* agents independently gain `inventory_axe`. This symmetric incentive prevents a race to be the sole beneficiary and sets up a clean payoff-dominant equilibrium for coordination. Causality checks should expect two axes spawned per successful CRAFT.

## 4. Communication Architecture: Slot-Based Message Heads
This is the central architectural leap. We are moving from a single 32D VQ token to a multi-slot structure to force compositional syntax.

*   **Design:** Three parallel message heads: `Slot_0` (Subject), `Slot_1` (Verb), `Slot_2` (Modifier), *plus* a continuous spatial channel.
*   **Dimensionality (The Wire Budget):** We will widen the total wire from 32D to 40D. 
    *   **8D Continuous Channel:** Dedicated to preserving the proven evasion geometry (Lag-1 Direction LRT, PosDis > 0.95). Evasion grounding survives alongside the new crafting layer.
    *   **32D Discrete Slots:** Split as `12D / 8D / 12D` across the three new VQ slots.
*   **Codebook Design:** 
    *   **Shared Noun Codebook:** `Slot_0` (Subject) and `Slot_2` (Modifier) share a codebook of 64 discrete tokens (`vocab=64`). This prevents object referents from having to be learned twice, allowing agents to generalize a token for "Wood" as both a Subject and a Modifier.
    *   **Separate Verb Codebook:** `Slot_1` (Verb) gets an independent, smaller codebook of 16 tokens (`vocab=16`). Verbs map to intents (e.g., Have, Need, Give) which is a much smaller space. Keeping this codebook separate structurally prevents semantic bleed between objects and intents.
*   **Masking Architecture:** All three VQ heads (and the continuous channel) share a single GWT-masked observation embedding `h_comms`. The outputs are then concatenated on the wire before being broadcast.

## 5. Decode Gate (The Phase 18 Win Condition)
NPMI is a Statistical Shadow. To clear Phase 18, we require a Tier-3 causal intervention.

*   **New Gate:** 
    1.  `NPMI(slot_0, inventory_item) > 0.3` AND `NPMI(slot_1, action_intent) > 0.3`
    2.  **AND** `causal_intervention.py ATE(slot_0 swap → partner pickup-target shift)` is significant at `p < 0.05`.
    3.  **AND** `causal_intervention.py ATE(slot_1 swap Have/Need → partner approach-vs-ignore shift)` is significant at `p < 0.05`. Both gated slots need a causal leg.
*   **Tooling Updates:** `tools/decode_signals.py` must parse the 3-slot concatenated signal. We must run causal freeze tests swapping tokens mid-flight to prove they shift receiving partners' behavior. NPMI passing without ATE passing is a failure.

## 6. Checkpoint Grafting Plan
We branch from the live `feature/phase17-5-timescale-alarm` run (checkpoint ~1.15M).

*   **Preservation:** Spatial survival skills (evasion, foraging) must be perfectly preserved.
*   **Grafting:**
    *   Use the existing `alarm_head` parallel output architecture as a template for the new slot heads.
    *   Call `graft_missing_param_subtrees` to initialize the 3 new VQ slot heads with near-zero weights.
    *   Zero-pad the action head to accommodate `CRAFT` and `USE_TOOL`.

## 7. Survival Multiplier & CtD Bootstrap (No Reward Shaping)
Per strict directives, there are **NO** crafting rewards. 

*   **Mechanism:** Crafted tools (e.g., Axe) provide a survival multiplier (e.g., 2x energy extraction from resource nodes, or reduced metabolic decay).
*   **Constraint (Capacity=1 Enforcement):** An Axe requires Wood + Stone. A single agent can only carry one primitive at a time.
    *   **Concrete Collision Rule:** If an agent holding Wood executes `PICK_UP` on a Stone tile, the action fails (no-op) and the inventory is unchanged. This explicit collision rule in `grid_jax.py` mathematically guarantees that solo crafting is impossible and coordination is required.
*   **CtD-Style Bootstrap:** To prevent mass extinction (the risk-dominant equilibrium where agents starve before discovering coordination), we will ramp the metabolic pressure.
    *   **Schedule:** Over the first 100k steps (1.15M to 1.25M), the `energy_decay` will linearly scale from the current survivable floor (`0.001`) up to the lethal baseline (`0.005`). This 5x jump relative to `resource_gain` creates a lethal cliff without imposing an arbitrary 50x jump that would guarantee extinction.
    *   This gives the population time to learn basic coordination before solo foraging becomes a guaranteed death sentence.
*   **Selection Pressure:** Once the lethal baseline is reached, agents without tools will starve. Agents who coordinate via the communication channel to assemble tools will survive. Listening to a "Need Stone" broadcast becomes a matter of life and death.
