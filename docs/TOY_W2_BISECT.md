# Toy → W2 bisection plan (Cam Nov 16 ~08:30Z stop-loss)

Status: **plan only, no runs.** It applies only if both last single-tweak screens fail: the lookahead screen
(seed 4) and the K-screen. Written by Will, Nov 16 ~03:25 EST, from the code (comm-toy `w2/toy_grid.py`,
`w2/learner_w2_kv.py` / `_la.py`, `w2/named_item.py`, `w2/env.py`, `w2/ep_agent_dev.py`) and the record
(`docs/RESEARCH_PROTOCOL.md`).

## The two ends

- **Start: toy screen 2 (5/5 take-offs, 5/5 bar passes).** `w2/toy_grid.py --tier T3 --learner L2e --hidden
  --curriculum --curriculum-mix --hindsight 0.5 --hind-at-start --shared-tok --demonstrator --demo-hearings 1
  --episodes-per-life 64 --epochs 4 --minibatches 8`, 20M steps, seeds 8-12. Data:
  `docs/data/toy_grid/T3_round2/screen2_sharedtok/`.
  - In that run the read formed fast. The frontier reached 3 at ~0.2M steps, 6 at ~2M, 12 at 5-6M and 24 at
    ~10M (seeds 8, 10, 12).
- **End: the registered W2 learner.** `learner_w2_kv.py --learner ep --slots 96 --shared-tok --hindsight 0.5
  --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32`, 5e7-1e8
  steps. It FAILED: seeds 1 and 2 were at frontier 3 at 5e7, with the read at chance.

## Already the same in both (not bisection steps)

- 7x7 view, and the item is visible only when adjacent (the toy's `--hidden`).
- An item is worth 0.1 energy (toy 0.5 x 0.2; W2 item size 0.1).
- Item placement: `round(d cos, d sin)` along sector x 45°. T_max = 4d. Mixed curriculum 1..frontier, cap 24.
  The bar is at d = 12.
- A code per life (a permutation of 8 tokens).
- Memory record = [token one-hot, displacement / 24, ate], written at each bout's end, read by key / value attention
  with the shared token embedding.
- Hindsight head, weight 0.5, labelled ONLY at the hearing step and ONLY on bouts the agent ended by eating, within
  the training sequence. The head's softmax feeds the policy.
- The demonstrator appears on a token's first hearing in a life.
- PPO: lr 3e-4, clip 0.2, value 0.5, entropy 0.01, epochs 4, minibatches 8, ~8192 steps per update.
- Basal drain 0.0013, move cost 0.003, reward 20 x drive reduction with drive = (0.8 - E)^2.

## Every difference

Read cost = my guess at how much the difference alone slows or blocks the read forming (low / med / high).
"Switch" = the cheapest way to make the TOY like W2 in that respect.

| # | difference | toy screen 2 | W2 registered | switch in the toy (cost to build) | read cost |
|---|---|---|---|---|---|
| 1 | **Hearings per step and per life** | Every episode is a bout. ~1 hearing per 4-25 steps. 64 hearings per life (~8 per token). No deaths within a life. | Bouts open only when hungry, between stretches of foraging. ~1.2-1.7% of steps are hearings (11-17k per 1e6 steps). Lives of ~300-3000 steps, so ~5-45 hearings per life. A death wipes the memory. | `--episodes-per-life` (exists): 64 → 16 → 8. New `--gap G`: G non-hearing steps before each episode (cheap). New `--die-ends-life`: carry E across episodes; E <= 0 ends the life (cheap-moderate). | **high** |
| 2 | **What one item is worth** | E is reset to 0.3 every episode, so one item is worth 20 x (0.5² - 0.4²) = **1.8**. | A bout opens at E < 0.7, so an item is worth about 20 x (0.15² - 0.05²) = **0.4**, and generic food can do the same job. | New `--e-start` (trivial): 0.3 → 0.65. | **high** |
| 3 | **Curriculum gate** | OLD gate: last 500 episodes >= 0.8, checked every update. It can advance on a lucky window. | HONEST gate: two consecutive non-overlapping 2000-bout windows of hungry bouts, each >= 0.8. | Import `w2/gate.py` HonestGate; score episodes with d <= frontier (trivial). | **med-high** (measured cause (1) in W2: no memory-only bouts below d = 3 for 31.6M steps) |
| 4 | **Value bootstrapping at bout ends** | Every episode end is terminal for GAE, so finding the item is a clean, short credit problem. | A bout end is not terminal. The life continues (γ 0.99, λ 0.95); only death is terminal. | `--life-credit` (exists: terminal only at life end). | **med** |
| 5 | **Network** | `EpMem`: an MLP on the input + memory read → head → policy. No GRU, no navigator prior. | `EpAgent`: a pretrained navigator trunk whose logits form the base policy, + GRU 128 + memory read + head, through a ZERO-initialised side head added to the navigator's logits. The read starts with no influence on action. | New `--learner ep`: run `w2/ep_agent_dev.EpAgentDev` on the toy's input, padding the 8 oracle columns with zeros (moderate). | **med** (A1: this network's read reached d = 12 when the head had plenty of labels, so it can read) |
| 6 | **Generic food** | None: the item is the only food. | 8 regrowing food patches (relocating). Hunger can be met without the item; bouts close "sated". | New `--food-patches P` in the toy: a sparse regrowing food map (moderate; the toy has no food grid). Partial cheap proxy: #2 + #1's gap. | **med** (overlaps 1 and 2) |
| 7 | **Own-position input** | The beacon offset, every step: `torus(start - pos) / 24`, unclipped. | The hearing-place offset, during a bout only: `clip(torus(start - pos) / 9, ±1)`. It saturates beyond 9 cells. | New `--hear-clip 9` (trivial). | **low-med** (at d >= 10 the agent's own offset saturates; the record still uses / 24) |
| 8 | **Training-sequence length** | T = 32. | T = 64. | `--T 64` (exists). | low (the own-meal label needs the meal inside the sequence; T = 64 truncates less) |
| 9 | **Demonstrator** | Walks straight to the item and stops on it; can leave the young behind. | Waits (never more than 2 cells ahead) and steps back toward a young that is more than 2 cells away. | Copy the two rules from `named_item.step_toward` (trivial). | low (W2's parent is the stronger scaffold) |
| 10 | **World and other agents** | 64 x 64 torus, one agent, a beacon mark at the start cell in the agent window. | 96 x 96, 4 agents per world (all visible in the agent window), no beacon. | `L = 96`; other agents are moderate to add (the toy is single-agent). | low |
| 11 | **Memory size** | 64 slots. | 96 slots. | `EpMem(S=96)` (trivial). | low |
| 12 | **Hearing frame** | Every episode starts at the beacon cell, so all records share one frame. | Each hearing place is wherever the agent is; records are relative to it. | Start each episode at the previous episode's end cell (cheap). | low |
| 13 | **Steps** | 20M (the read formed by ~6M). | 5e7 (early stop) to 1e8. | — | low (W2 had more steps) |

## Ranking (biggest expected read cost first)

1. **#1 hearings per step / per life (with deaths)**: the memory needs repeated tokens within a life. W2 gives
   ~5-45 hearings per life against the toy's 64, plus memory-wiping deaths while the young is still bad at
   foraging.
2. **#2 what one item is worth**: ~4.5x less reward per item in W2, and the agent has an alternative.
3. **#3 the gate**: it held the W2 learner at d <= 3, where an item at d <= 2 can be in view and no bout needs
   memory.
4. **#4 terminal bout ends**: the toy's credit problem is short and clean.
5. **#5 the network**: the zero-initialised side head on top of a strong navigator prior.
6. **#6 generic food**: mostly a mechanism behind #1 and #2.
7. **#7 hearing-offset clip**, then #8-#13 (low).

## Bisection order (biggest first, halving where it is cheap)

Each step is a short laptop screen on the toy (`w2/toy_grid.py` plus the new flags): 1 seed, at most 1e7 steps
(the toy formed its read by ~6M), seeds kept from screen 2's range.

Registered criterion, **READ FORMS**: head accuracy on 2nd+ hearings at d = 3 >= 0.5 at the budget, with the
toy's own evaluation (`--eval-modes 0,3,4`, so shuffled and wiped come for free). If a step cuts the
curriculum's speed (#3), the budget doubles to 2e7 once, and that is recorded.

- **Step 0: positive control.** Toy screen 2 as it was (seed 8), to 1e7. It must form the read; if it doesn't,
  stop: the toy result does not reproduce.
- **Step 1: the top block, all at once: #1 + #2 + #3 + #4.** (`--episodes-per-life 16 --gap G --die-ends-life
  --e-start 0.65`, the honest gate, `--life-credit`.) This is the half with the biggest expected effect.
  - Read still forms → none of the top four blocks it on its own terms. Go to step 2.
  - Read fails → halve: **1a = #1 + #2** (the life and incentive structure) vs **1b = #3 + #4** (gate and
    credit). Then halve the failing pair once more (one difference each). That is at most 3 more screens.
- **Step 2: #5 the W2 network** (on top of whatever passed). Read fails → the zero-initialised side head and the
  navigator prior are the suspects: one more screen with the side head initialised small but non-zero, or with
  the head's read fed straight to the policy, to tell them apart.
- **Step 3: #6 generic food** (patches) on top.
- **Step 4: the low block together: #7 + #8 + #9 + #10 + #11 + #12.** Halve only if it kills the read.
- **The end of the chain IS W2 except for its own engine** (multi-agent env, foraging dynamics). If every toy
  step forms the read, the last screen is a W2 run with step 1's settings undone one at a time, cheapest first,
  starting with the honest gate (the one that's trivial to switch in W2 too).

Expected length: 5-8 screens. The build per step is trivial to moderate (new flags, each with a flag-off
bit-identity check against screen 2's curve, as for every flag so far). Each step's prediction is written before
it runs.

## What this plan does not do

- No run until both screens have failed and Cam gives the go.
- It does not retune any ecological parameter to make a gate open (Part 4). The steps make the toy MORE like W2;
  they never make W2 easier.
- The differences are as found in the code on Nov 16. A difference I missed is likeliest in the W2 env's own
  dynamics (food regrowth, other agents' eating), which the toy has no counterpart for. Step 3 and the final W2
  step cover them as blocks.
