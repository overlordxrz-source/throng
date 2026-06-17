# Phase 18 Crafting Design: Forcing Compositional Communication

## 1. Minimum Crafting Depth Requiring Coordination
**Hypothesis Validated**: Depth-2 with complementary resource distribution and mutual exclusion (e.g., carrying capacity constraints) is the *mathematical minimum* to require coordination.

**Game Theory & MARL Context**: 
In a Depth-1 environment, agents simply optimize individual pathfinding. In Depth-2 (where `A + B = C`), if an agent can gather and hold both A and B, the problem collapses into a sequential solo task (a single-agent routing problem). 

To force *MARL coordination*, we must impose **asymmetric state**. Coordination games (like complementary resource pooling or spatial Stag Hunts) dictate that coordination strictly requires individual incapacity. By constraining the carrying capacity to 1 (Agent A holds Wood and cannot hold Stone; Agent B holds Stone and cannot hold Wood), the agents are forced into a cooperative assembly protocol. Neither can complete the recipe alone.

## 2. Resource Scarcity and the VQ Bottleneck
**Information Theory of VQ**: 
The VQ channel acts as an Information Bottleneck (IB), selectively quantizing the continuous state space into discrete latent codes based on *utility variance* and *entropy*.

**The Scarcity Interaction**: 
If Wood is abundant (low surprisal/entropy) and Stone is scarce, the VQ bottleneck will drop Wood from the codebook entirely. Communicating "I have Wood" has near-zero marginal utility. The communication channel will collapse to a 1-bit signal prioritizing the scarce resource ("I have Stone" / "Where is Stone?").

**Compositional Forcing**: 
To force *compositional* communication (requiring multi-token sequences like `[Need] [Stone]`), the environment must enforce **symmetric scarcity**. Both Wood and Stone must be scarce enough that their spatial distribution possesses high entropy. When both are scarce, a single discrete token cannot efficiently cover the joint probability distribution, forcing the VQ bottleneck to allocate separate, compositional codes for different resource types and states.

## 3. Receiver-Side Selection Pressure (Metabolic & Mortality Tweaks)
**The Phase 17.5 Lesson**: The Alarm channel failed because it lacked a receiver payoff differential (ATE = 0). The sender paid a cost, but the receiver didn't benefit. For Phase 18, receivers *must* die faster if they ignore broadcasts.

**The Mechanic**:
Instead of shaping arbitrary rewards (which leads to RL lazy-stay collapse), we ground the protocol in pure thermodynamics:
1. **Lethal Metabolic Baseline**: Set the base `energy_decay` to be highly punishing. An agent foraging bare-handed cannot reliably outpace the thermodynamic tax of survival and reproduction. Without the crafted tool, their expected lifespan is severely truncated (high solo mortality).
2. **The Survival Multiplier**: The crafted tool (e.g., `Axe`) provides a massive metabolic advantage—either doubling the energy extracted from food nodes (`resource_gain * 2`) or drastically reducing the baseline `energy_decay`.
3. **The Coordination Lock**: Because of the Capacity=1 constraint, Agent A (with Wood) must broadcast "Need Stone". If Agent B (with Stone) ignores this signal, both remain bare-handed and inevitably starve due to the lethal baseline. If Agent B decodes the signal and navigates to Agent A to craft the tool, both gain the survival multiplier.

**Conclusion**: This creates a non-linear fitness cliff. The crafted tool is not an extrinsic reward bonus; it is a thermodynamic requirement. Agents that decode signals live; deaf agents starve.
