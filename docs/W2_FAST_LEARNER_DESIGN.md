# W2: a generic fast-learner prior. Design only (Cam Nov 10 ~00:35, the lever after the gate failures)

**Why.** The newborn learnability gate failed in every arm: juvenile; +predictive; +replay; +copying elders' actions
(0.3-1.1% of lives vs >= 50%). The positive control passes (65%). Behaviour-shaping losses get ~18 gradient steps per
life even with replay, which trains only the predictive head. Gradient learning within one ~570-step life is too slow
for a 9-way mapping.

**Idea (RL^2 / meta-RL; Duan et al. 2016, Wang et al. 2016).** Give newborns a recurrent policy whose *activations*
learn within a life. The weights are meta-learned across many lives, on many **random** codes, so the prior is "learn
whatever code the elders use, fast", never our code.

## Meta-training (offline, before any W2 comm run)

- **Inner episode = one simulated life:** the W2 world plus frozen scripted speakers using a **freshly randomised
  code every life**: a random bijection from food sectors (or other world variables) to tokens, or random
  noise-codes as a null.
- **Agent:** a GRU core (128) over [obs, heard tokens, previous action, previous reward]; policy and value heads; the
  predictive head kept (dense signal).
- **Outer loop:** PPO / A2C across many lives. The hidden state carries over across steps within a life (never
  across lives). The objective is the homeostatic return over the whole life.
- **Anti-leak rule (Cam: "never teach our code"):** codes are drawn uniformly at random per life, so no particular
  mapping is learnable across lives. Apparatus test: the meta-trained newborn's reliance on a **held-out random code
  family** must equal its reliance on the training family (no memorised mapping).

## Use in W2

- Newborns start from the meta-trained weights (an inborn prior, like navigation). The W2 navigators become the
  meta-learner's initialisation, so foraging is kept.
- Gate: the same learnability test (16 episodes, >= 50% of lives), with a code the meta-learner never saw.

## Gates before any W2 use

1. **Meta-train sanity:** on a held-out random code, reliance >= 30% within ~300 steps of life, in >= 50% of lives.
2. **No-code control:** with uninformative messages, reliance ~0 (the prior doesn't hallucinate a code).
3. **Leak test:** held-out code family = training family (within SE).
4. Foraging kept: median lifespan without messages >= 0.9x the navigators'.

## Cost (estimate)

- Meta-training ~2-5M agent-steps of lives on CPU or one A10G: ~1-3 h of A10G (~$1-3), or laptop overnight.
- Laptop prototype first.

## Cheaper variants to try before meta-learning (each is one flag, for Cam to choose)

- **Copying inside replay:** behaviour cloning on stored elder (obs, action) pairs during the K replay passes. That
  gives the copying loss ~300 steps per life instead of ~18. It is still elder-copying, the trait M approved.
- **A young-plasticity trait:** a higher learning rate for the young (already a heritable trait in the spec).
