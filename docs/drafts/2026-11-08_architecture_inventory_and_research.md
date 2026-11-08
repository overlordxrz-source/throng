# THRONG architecture: inventory from the code, and research on the best target (DRAFT)

Requested by Cam (Nov 8). $0; no code changed; no running job touched.

Part A is read from the code: `master` `4353fae` for production, `comm-jit` `326c8cb` for the real testbed,
`comm-toy` `69ad45e` for the toy. Each claim names the file it comes from. "Not verified" means I did not trace it
fully. Part B cites only sources I opened, with what each says. The last section recommends a target and an order,
with a test for each step.

---

## Part A — what exists today

### Quick answer: the critics in credit fixes A and B

**They saw only the sender's own observation, not global state.**
- `V_shared(s)` (fix A) and the per-slot `q_i(s, m_i)` (fix B) are dense heads on the sender's own GRU state `h`
  (`comm-toy` `scripts/comm_toy.py`, `ToyNet`).
- `h` comes from the sender's local obs: position, hearth compass, fire one-hot, 4 noise dims, the heard wire.
- Neither critic saw the listener's position or state, or the world, so this was **not** a MAPPO-style centralized
  critic. MAPPO puts global state in the critic during training, and that is the textbook remedy for the noisy
  credit we measured (Part B1). It has not been tried.

### A1. World

| | Production (`master`) | Real testbed (`comm-jit`) | Toy (`comm-toy`) |
|---|---|---|---|
| Size | 128 x 128 torus (`config.yaml` `grid_size`) | 32 x 32 torus | 12 x 12 torus |
| Populations | Blues (prey) capped at 200 (`population_size`); reds (predators) 250 (`red_population_size`) | 32 independent worlds x 2 blues (one sender, one listener) | 64 worlds x 2 |
| Red vs blue | Reds catch blues within Chebyshev 1 with p = 0.8 (`red_catch_radius`, `red_catch_prob`, `grid_jax.apply_catches`). Reds starve after 400 steps without a catch. Reds are blind beyond their 5 x 5 patch (`red_detection_radius: 0`) and rely on neighbour signals. | No reds | No reds |
| Big Green | A blue subtype (`is_big_green`). A cooperative red catch pays 8; a solo attempt is penalised (−1) or, under the CtD ramp, pays 2. Offspring flip type with p = 0.03 (`main_jax.py` ~453-472). | — | — |
| Crafting, materials | 12 actions: N/S/E/W/Stay/Strike/Push/Guard/Build/PickUp/Craft/UseTool (`n_actions`). Five materials (wood, stone, flint, clay, vine) spawn in zones (`grid_jax.material_zone_masks`). | Craft is used only to deposit at the active hearth | Same |
| Hearths | 4 hearths (`generate_hearth_positions`), deposits via `resolve_hearth_deposits`, radius 3, CtD competence ramp (`ctd_ramp.py`) | The 4 production hearths, one active; Craft within radius 1 deposits 2.7; the listener's deposit also pays the sender 2.7; non-deposit Craft costs 0.1 | Same rules; cost 0.45 |
| Barriers, shelters, puzzle | Build makes barriers that decay (`decay_barrier_grid`); 5 shelter spots; lock-key puzzle that always runs (`puzzle_*`; config audit: pending wiring) | — | — |
| Reproduction, death | Clone when energy >= 0.8 (cost 0.4), with min-pop cloning (`population_jax.apply_auto_reproduce`). Death by starvation (energy decay 0.001), age (`max_age` 1000) or capture. | Depositor respawns; no death or birth | Same |

### A2. Observations

| | Production | Real testbed | Toy |
|---|---|---|---|
| View | 5 x 5 local patch (`local_obs_radius` 2) x 15 env channels; local symbol and culture cells | Production layout, mostly zero | No patch |
| Compass | `hearth_rel` (4 hearths x 2) inside `own_state` (29 dims) | Same formula | Same formula, 8 dims |
| Others | k = 6 nearest neighbours' 40-dim signals; episodic memory slots (`memory_slots`) | Partner's last broadcast in neighbour slot 1; sender's need dims = one-hot of the active fire | 59-dim obs: pos, compass, informed flag, fire one-hot (sender only), 4 noise dims, 40-dim wire |
| Team differences | Reds: 5 x 5 only, plus neighbour signals; `observations_jax._mask_loc_env_red_channel` masks a red channel | — | — |

### A3. Brains

| | Production blue | Production red | Real testbed | Toy |
|---|---|---|---|---|
| Core | **Transformer**, d = 256, 4 heads, 4 of max 6 layers, tokens from own / neighbour / symbol / env / signal / memory / culture embeddings (`network_jax.AgentNetworkJax`) | Transformer d = 128 (`red_hidden_dim`), 4 layers, + neighbour cross-attention (`PredatorNetworkJax`) | Production blue | MLP 64 x 2 + GRU 64 |
| Memory | **Not a GRU:** `new_carries = 0.9 * carries + 0.1 * pooled` (an EMA of the pooled transformer output), plus episodic slots | **A real GRU** (`carry_gru`) | Production blue | GRU |
| Heads, trained | action (12), value, alarm (2), 3 VQ signal slots (64 codes each; via `loss_vq`, and as policy actions on `comm-testbed` onwards), forward models (`head_fwd_*`, `head_fwd_dyn_*`), self-prediction, confidence, proprio (all `auxiliary_update`), VQEL reconstruction (`vqel_monologue_update`) | action, value, DCVQ signal (4 subspaces; red VQ loss 0), proprio, retention (`red_auxiliary_update`) | action, value, alarm, token policy, VQ, + PS (opt-in) | action, value, comms pathway + 3 codebooks; + V_shared / q heads (credit fixes) |
| Heads with no loss found ("dead" as learners) | ToM (`head_tom`), symbol write, culture fast / slow. Their outputs are written to the grids, but no gradient reaches them (no loss in `rl_jax.py`; listed only as frozen in the VQEL mask). *Not verified exhaustively.* | ToM, symbol, culture | Present but unused | — |
| Comms pathway | `gwt_comms_1` (from obs, with metabolic dims zeroed) -> per-slot heads -> VQ (`network_jax.py` ~508-530) | `gwt_comms_1` -> DCVQ | Same as blue | Same shape |
| Weights | **One network shared by all blues** | **One shared by all reds** | Shared by sender and listener | Shared |
| Imagination | Wired (`imagination_jax.py`); config gate `imagination_gating_enabled: true` under a Phase 11.3 block; switched off for the November runs (roadmap) | Off | — | — |

### A4. Communication

- **Production blue:** a 40-dim wire, made of 8 continuous dims (zeroed: "amputated") plus three VQ slots (12 / 8 / 12
  dims, 64 codes each). Each agent hears its 6 nearest neighbours. A 1-bit alarm channel, plus symbols and culture
  written into grid cells, which anyone on the cell can read.
- **Red:** its own DCVQ signal, heard by neighbour reds through cross-attention. The red VQ is decoupled from the
  gradient (`red_vq_loss_coef: 0`).
- **Real testbed and toy:** one partner, one-step-delayed broadcast of the sampled tokens' codebook vectors. Tokens
  are policy actions (tau 0.08).

### A5. Elder and culture features

| Feature | What it does (code) | On? | Does what an elder learned reach a newborn? |
|---|---|---|---|
| Mind-meld | Agents within radius 1 blend carries; older -> younger at rate 0.1 per step (`population_jax.apply_mind_meld`) | `mind_meld_enabled: true` | **Partly, short-term only:** an elder's carry (recent state) leaks into a younger neighbour's carry. Weights are shared anyway, so no learned skill is transferred. |
| MEDAL-ADR "expert dropout" | Zeroes the carries of the **oldest** 10% of agents (`main_jax.apply_medal_adr_carry_reset`) | `medal_adr_enabled: true`, p 0.10 | No; it erases elders' state. |
| Distill | Every 10,000 steps, clone the top 5% of carries with noise | `distill_enabled: false` | No (off) |
| Culture grids | `culture_fast` / `culture_slow` heads write into decaying grid layers (0.90 / 0.995) that everyone reads | Written every step | **Readable, but the writer heads are untrained** (no loss), so the content is a random projection of the writer's state, not learned writing. |
| Symbols | `head_symbol` writes into a grid layer | Written every step | Same: untrained writer |
| Reproduction | Clones the parent's position and type; carry not traced | On | The brain is shared, so a newborn already has "the elders' brain"; only its carry is fresh. |

Summary: before T3 nothing an individual elder *learned* was handed on, because there was no individual learning to
hand on. T3 (newborn imitation into a re-initialised head) is the first real transmission.

### A6. Training

| | Production blue | Production red | Real testbed | Toy |
|---|---|---|---|---|
| Algorithm | PPO, 1 epoch, minibatch 512, lr 3e-4, clip 0.2, gamma 0.999, ent 0.02, vf 0.1 (`ppo_value_coef`); per-group gradient clip 2.0 | Same `ppo_update`, separate optimizer | PPO + C' (GAE 0.95 + bootstrap) + PopArt, vf 0.5 | Same as testbed, plus credit options |
| Estimator / critic | **Production still runs truncated MC returns with no bootstrap** (lambda discarded) and the raw-units value clip; the fixes are validated only in the testbed (roadmap) | Same | Fixed | Fixed |
| Critic inputs | **Local:** the value head reads the agent's own transformer state; no global state anywhere | Local | Local | Local (including credit fixes A and B) |
| Other losses | VQ commitment (0.1), auxiliary forward / self-pred / confidence / proprio, VQEL monologue | Retention + proprio aux | VQ, token entropy 0.01, PS | Same |

---

## Part B — research (sources opened)

### B1. Training: MAPPO with a centralized critic

- **Source:** Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*
  (https://ar5iv.labs.arxiv.org/html/2103.01955).
- **What it says:**
  - Critic-input options: CL (concatenated local obs), EP (environment state), **AS (environment state + the agent's
    own local obs)**, FP (AS with duplicates removed). FP was best on several maps.
  - Five practical rules: value normalisation (never hurt, often helped; we already use PopArt); agent-specific +
    global critic input; at most 10-15 epochs with few minibatches; clip <= 0.2; large batches.
- **Fit:** high, and directly targeted at our measured problem. The token's credit is noise-dominated (cos ~0.001
  with the restoring direction). A critic for the shared reward that sees the listener's state and the fire removes
  the variance the sender can't see. Execution stays decentralised.
- **Cost:** low (toy first, $0). **Risk:** low; a train-time-only change.
- Note: our "1 epoch, 64 minibatches" is the opposite of MAPPO's advice. That is a later speed and quality lever,
  needing its own validation.

### B2. Messages: targeted attention, discrete symbols, when to talk

- **TarMAC** (Das et al., ICML 2019; https://arxiv.org/abs/1810.11187, PDF read):
  - Each message = (signature key, value). The receiver's query attends over all senders' keys; messages are
    *continuous*.
  - Multi-round communication; GRU policies with shared parameters; a **centralized critic** over all agents' hidden
    states and actions.
  - Fit: the *targeting* (receiver-side attention over senders) is the right shape for N > 2 agents, but our
    discrete, action-like tokens are closer to words and to spiking hardware. Take TarMAC's attention as the
    *receiver*, not its continuous messages.
- **IC3Net** (Singh et al., ICLR 2019; https://arxiv.org/abs/1812.09755): a gate lets each agent learn *when* to
  communicate; individualised rewards; works in cooperative to competitive settings and scales with agent count.
  Fit: a "speak / stay silent" action is cheap to add and gives a natural cost-of-talking pressure (the roadmap's
  energy-cost idea).
- **Kottur et al., EMNLP 2017** (https://arxiv.org/abs/1706.08502): the agents' languages are effective but not
  compositional unless communication is restricted. That supports the roadmap's small-vocabulary symbol channel
  (2-3 symbols from 8-16) over 3 x 64 VQ codes.

### B3. Memory: GRU vs attention memory

- **AdA** (Bauer et al., ICML 2023; https://arxiv.org/abs/2301.07608, abstract only): a large attention-based
  memory policy, an automated curriculum at the edge of ability, and scaling laws in network size, memory length
  and task diversity.
- **Fit:** attention memory is what lifetime adaptation needs at scale, but it is expensive. Our measured
  bottleneck is the PPO update (74% of wall-clock). Production's blue "memory" is an EMA, not a GRU (A3), which is
  worth fixing first.
- **Recommendation:** a GRU core now (the toy and red already have one); attention memory only when rung 7 needs
  it, gated on "bigger beats smaller".

### B4. Population: separate brains, PBT, iterated learning

- **Seeded Iterated Learning** (Lu et al., ICML 2020; https://arxiv.org/abs/2003.12694): a student is periodically
  refined by imitating data from a teacher. Each cycle the teacher is a copy of the student, fine-tuned on the task.
  It counters language drift and improves task completion.
- **Neural iterated learning** (Ren et al., ICLR 2020; https://arxiv.org/abs/2002.01365): iterated learning
  amplifies compositionality, and compositional languages are learned faster.
- **Fit:** high. T3 is exactly this (newborn imitation, approved as permanent). Separate brains (4-8) with
  population-based selection are needed for any culture test, because one shared brain has nothing to teach.
- **Cost:** moderate (N networks); toy first.

### B5. Worlds: our JAX world vs JaxMARL, Craftax, XLand-MiniGrid

- **JaxMARL** (https://arxiv.org/abs/2311.10090): JAX multi-agent environments + baselines; ~14x faster wall-clock,
  up to 12,500x with vectorised runs. The abstract doesn't list communication tasks.
- **Craftax** (https://arxiv.org/abs/2402.16801): JAX Crafter rewrite, 250x faster; 1B steps of PPO in under an
  hour on one GPU; needs exploration, memory, planning. Multi-agent not stated.
- **XLand-MiniGrid** (https://arxiv.org/abs/2312.12044): JAX meta-RL grid worlds, millions of unique tasks, millions
  of steps per second. Multi-agent not stated.
- **Fit:** keep our own JAX world. The emergent-language pressures (hearths, informed agents, predators) are
  THRONG's own, and our jitted rollout is now exact-equivalent and in place.
- **Borrow:** JaxMARL's MAPPO baseline code as a reference implementation for B1; Craftax and XLand-style
  procedural task diversity for the "different world every life" step.
- **Risk of switching wholesale:** high. We would lose the validated apparatus.

### B6. Predator–prey arms races

- **OpenAI hide-and-seek** (Baker et al. 2019; https://arxiv.org/abs/1909.07528): a simple competitive objective
  produced 6 strategy phases (e.g. box shelters, ramp use) as a self-supervised autocurriculum, without
  hand-designed rewards.
- **Fit:** our red / blue world already has the structure, but it ran at big-lab scale.
- **Recommendation:** only after the size curriculum, with the fixed learner on *both* teams. A test: the number of
  distinct strategy shifts over training against a frozen-opponent control.

### B7. Neuromorphic limits (Lava)

- **lava-dl** (https://lava-nc.org/dl.html):
  - SLAYER direct training (with recurrent support), Bootstrap rate-coded ANN-to-SNN, and NetX inference from hdf5.
  - Stated limits: direct training is costly; ANN-SNN conversion gives high latency; Lava processes can't be trained
    in place.
  - The page doesn't list which layers convert; Recurrent appears only in a SLAYER example.
- **Fit:** a small dense / GRU actor is the realistic target. A transformer with attention is not.
- **Recommendation:** the lean GRU core (B3) also keeps the neuromorphic path open.

---

## Recommended target architecture

| Part | Target | Why | Instead of (today) |
|---|---|---|---|
| Brain | Lean core: encoder -> **GRU** -> policy, value and message heads; other modules return only after an ablation shows they help | Credit, speed and Lava conversion all favour it; the toy GRU core reproduces the real failure and its fix | Blue transformer + EMA carry + ~12 heads, several with no loss |
| Messages | **Discrete symbols as actions:** 2-3 symbols from 8-16, plus an optional "stay silent" action (IC3Net) | Compositionality (Kottur); word-like; spike-friendly | 3 x 64 VQ codes (~262k messages), continuous dims amputated |
| Receiving | TarMAC-style attention over heard messages when N > 2 | Picks whom to listen to | Fixed k = 6 nearest |
| Training | PPO + C' + PopArt + a **MAPPO centralized critic** (agent-specific global state: own obs + partner / listener state + world facts), train time only | Targets the measured noise-dominated message credit | Local critic only |
| Population | 4-8 separate brains, population-based selection, **newborn imitation (SIL)** as part of the agents' nature | Culture needs individual knowledge; iterated learning favours structure | One shared brain; untrained culture writers |
| World | Our JAX world, jitted; procedural variety later | Keeps the validated apparatus | — |
| Hardware | Lean GRU actor -> Lava SDNN (CPU sim first) | Feasible conversion path | — |

**Order, each step gated by its test (toy first, $0, unless noted):**

1. **MAPPO critic for message credit (toy).** The fix-A stream with a critic that sees sender + listener state + the
   fire. **Test:** the erosion protocol on Cam's reliance bar (wire-on reliance in >= 2 of 3; wire-off still erodes).
2. **Channel A/B (toy):** symbols (2-3 of 8-16) vs the VQ wire, same task and seeds. **Test:** switch if symbols
   match or beat it on reliance and redirection (roadmap).
3. **No assigned sender (toy):** knowledge from proximity; everyone can speak and listen. **Test:** redirection
   under swap on a par with the assigned-sender toy (roadmap).
4. **Lean core on the real testbed:** GRU core vs today's transformer. **Test:** deposit and transfer parity at
   32 x 32 (step-0-style equivalence is not possible; use registered learning-curve bars), plus speed. ~$2-5.
5. **Separate brains + newborn imitation (testbed):** builds on T3's result. **Test:** a culture test where novices
   learn faster with elders present than in an elder-free world (rung 5).
6. **Compositionality pilot (toy):** fire x material targets, 2 held-out pairs. **Test:** topographic similarity
   above a shuffled baseline, and held-out pairs above chance (roadmap step 6).
7. **Production port:** the learner fixes behind flags, then the chosen brain, channel and critic. **Test:** the
   size curriculum 32 -> 64 -> 128, each stage >= 2x the frozen control.
8. **Lava:** convert the lean actor; run it in Lava's CPU simulator. **Test:** delivery rate within 20% of JAX.

*Open items I did not verify:* every use of the ToM, symbol and culture heads outside `rl_jax.py`; red comms
details beyond the module structure; whether production's imagination gate is on in the latest config profile.
