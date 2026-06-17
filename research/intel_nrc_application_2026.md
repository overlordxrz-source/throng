# Proposal for the Intel Neuromorphic Research Community (INRC)

> **[ON HOLD] Pending ATE > 0 validation of the discrete channel per Cam's instructions.**

**Project Title**: Scaling Causally Grounded Emergent Communication via Asynchronous Spiking Populations  
**Principal Investigator**: [Your Name / Lab]  
**Target Hardware**: Intel Loihi 2  

## 1. Executive Summary

Current large language models (LLMs) excel at statistical pattern matching but lack causal grounding; they manipulate symbols without understanding their physical or thermodynamic consequences. Project THRONG aims to solve the symbol grounding problem from the bottom up by evolving language in a massive, multi-agent reinforcement learning (MARL) ecosystem subject to physical survival pressures. 

In Phase 17.5 of THRONG, we successfully demonstrated the emergence of a "Timescale Grammar"—the separation of continuous spatial geometry (pointing) and discrete, metabolically constrained signals (alerting). However, to scale this ecosystem to populations of millions of agents capable of sustaining multi-generational culture, we require hardware that maps naturally to asynchronous, continuous-time interactions. We propose porting the THRONG architecture to Intel's Loihi 2 neuromorphic processor to leverage its inherent asynchronous scaling and spike-timing-dependent plasticity (STDP).

## 2. The Theoretical Gap

The fundamental barrier to evolving complex language in standard GPU-accelerated MARL (e.g., JAX on TPUs) is synchronous clocking. In nature, language evolves because communication incurs a physical energy cost and happens asynchronously in continuous time. 

In our JAX-based simulations, simulating continuous-time metabolic costs requires complex workarounds (e.g., explicitly gating Vector Quantization layers and deducting floating-point energy pools). On Loihi 2, **spikes inherently cost energy**. By mapping our discrete "alarm" channels to physical spike emissions, the metabolic penalty for communication is enforced natively by the hardware architecture.

## 3. Proposed Implementation

We propose adapting our existing JAX/Flax agent architecture into an event-based spiking neural network (SNN) utilizing the Lava software framework.

### Phase A: Spiking Representation of Timescale Grammar
- **Continuous Channel**: Represented by graded potential rate coding over a localized neural population.
- **Discrete Channel (The Alarm Bit)**: Represented by sparse, high-threshold spike emissions. The Loihi 2 architecture's energy-per-spike metrics will serve as the literal metabolic fitness function for the reinforcement learning loop.

### Phase B: Continuous Online Learning
Currently, our JAX agents undergo massive periodic PPO updates, causing catastrophic forgetting of localized sub-dialects. We will implement spike-timing-dependent plasticity (STDP) or a reward-modulated STDP equivalent directly on Loihi 2. This will allow the agents to adapt to novel ecological niches and negotiate local pidgins in real-time without global gradient synchronization.

## 4. Expected Outcomes

1. **Massive Scale**: Escaping the synchronous GPU bottleneck to simulate >1,000,000 interacting agents simultaneously.
2. **True Thermodynamic Grounding**: Demonstrating, for the first time, an emergent language where the fundamental building blocks of grammar (verbs vs. nouns) arise natively from the hardware's energy constraints.
3. **Open-Source Contribution**: Releasing a Lavabased MARL environment specifically tuned for ecological simulation and emergent communication research.

## 5. Justification for Loihi 2
Loihi 2's enhanced programmable neuron models and continuous-time asynchronous execution are strictly necessary for this research. Standard Von Neumann architectures simulate energy constraints mathematically; Loihi 2 enforces them physically, providing the precise evolutionary pressure required for true symbol grounding.
