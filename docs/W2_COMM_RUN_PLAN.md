# W2: first communication run. Plan for Cam's sign-off (draft, Nov 9)

**The question:** in our own world, with separate brains and aligned interests (kin), does a code about food location
form, get used, and pass to newborns?

## A design problem to decide first: lives are short in learning time

- A W2 life is ~1,400 steps, which is ~5 PPO updates at T 256 (~22 at T 64).
- In the toy a code needed ~1,000 PS-updates to form. Founders would die long before a code forms, and newborns start
  from navigators with no code.
- **Proposal: a founding phase.**
  - For the first F = 1,000 updates (T 64), founders don't age or die (energy floored at 0.05; the drive still
    punishes low energy) and don't reproduce. PS acts on them (annealed over updates 500-1,000).
  - Then the life cycle switches on (death, setpoint reproduction, turnover) and PS is off for everyone.
  - The founders then age normally. Their code survives only if newborns copy it and keep using it.
  - This mirrors the toy's founders-only formation, adapted to mortal bodies. Founding without PS stays the stretch
    goal.
- **Copying:** newborns imitate elder speakers' (situation, message) pairs during a sensitive period (as `pop_toy.py`,
  message head only).
- **Within-life learning:** T 64 and 2 PPO epochs per update, so a life spans ~20-40 gradient steps.

## The wire (as the toy, adapted)

- Every agent speaks 3 VQ token slots every step (wire-blind message head; message-head rule = all rows).
- Every agent hears the 2 nearest speakers within radius 9 (their codebook vectors + their clipped offset), one step
  delayed.
- **What's worth saying = food location.** PS target (founders only) = MI(token; the direction of the richest food in
  the speaker's own view: 8 sectors + "none").
- **The lookout** (3 x 3 cells, one per world) widens a speaker's view to 19 x 19, so a speaker there knows food a
  listener can't see.

## Arms (64 replicate worlds per run, 8 agents each)

| arm | kin reward | wire | seeds |
|---|---|---|---|
| K (aligned) | inclusive, k = relatedness | on | 5 |
| K0 (no kin) | own drive only | on | 5 |
| OFF (no signal) | inclusive | off (copying on) | 3 |

## Readouts at the end of the life phase (registered at sign-off)

- **Reliance:** cross-direction swap (a heard message replaced by one the same speaker used for another direction):
  listeners' food intake drops >= 30% with paired > 2 SE. Same-direction swap clean (< half the cross drop).
- **Gain:** K listeners' energy intake per step >= OFF + a margin (set at sign-off).
- **Code alive:** shuffle-corrected joint MI(token; food direction) >= 0.2 bits above OFF.
- **Aligned interests:** K vs K0 on reliance and MI. The prediction is K > K0; a K0 code would mean kin selection
  isn't needed here.
- **Transmission:** reliance among agents born after founding ended (no PS, no founders alive).

## Price

- From the smoke: 0.19 s per update at T 256, W 64. At T 64 with the wire, PS and 2 epochs, ~0.1-0.15 s per update.
- 1,000 founding + 3,000 life updates = ~7-10 min per run on an A10G = ~$0.15-0.2 per run, plus image builds.
- **13 runs = ~$2.5-3.**

## Code needed (laptop, before any paid run)

- The wire in `w2/agents.py` (token heads, hearing, the PS term with founders-only weights, imitation).
- The founding-phase flags.
- Ported swap / same-spot / MI probes.
- Apparatus tests (gradient isolation, PS 0 for non-founders, hearing radius, wire-off zero, same-spot control).
