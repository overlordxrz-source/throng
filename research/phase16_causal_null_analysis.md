# Phase 16: Causal Null Analysis & Information Redundancy

## Overview
During Phase 16, a strictly scoped Frozen Counterfactual Causal Test was conducted to determine if the emergent communication tokens (specifically Token 44: "Predator/Danger") were causally load-bearing for individual agent survival behaviors (fleeing). 

Despite Token 44 demonstrating a robust correlational NPMI score (404/601 emissions coinciding with predators within 5 units, 67%), the causal interventions definitively established an Average Treatment Effect (ATE) of effectively zero. 

## Methodology & Interventions

The causal test utilized `tools/causal_intervention.py`. The simulation is deterministically rewound and hydrated from a frozen checkpoint. The target token (Token 44) is injected mid-flight, and the shift in action probability (Delta P) for relevant behaviors (e.g., Strike, Flee) is measured against a baseline token.

We conducted multiple interventions:
1. **Flee Full Context:** Tested Token 44 against arbitrary receivers. ATE = 0.0000.
2. **Flee Distance-Filtered:** Scoped specifically for isolated events to ensure the receiver was far from the predator. ATE = 0.0000.
3. **Flee Blind Receiver (`--receiver-dist-min 10`):** Strict filtering to guarantee the receiver had no direct line of sight to the predator within a 10-unit radius. ATE = 0.0003 ($p=0.217$).

## Diagnosis: Information Redundancy

The null ATE results prove that the cross-attention weights for the communication channel have collapsed or are actively ignored during fleeing scenarios. Token 44 is merely a correlational proxy—a side-effect of local perception. 

**Root Cause:**
Agents have direct perception of predators through their local observation patches (`loc_env` red channel) and global Chebyshev distance checks. Because direct exteroception provides all necessary information for evasion, there is zero evolutionary pressure to parse the noisy, low-bandwidth VQ communication channel. The channel acts as a broadcaster with no active listeners.

## Next Steps
To forge genuine semantic grounding, the communication channel must become an exclusive, non-redundant source of survival-critical information. 

This requires **Phase 16.6 Barrier Occlusion**: deliberately blinding agents to predators that are physically occluded by environmental barriers, forcing reliance on neighbor signals to anticipate hidden threats.
