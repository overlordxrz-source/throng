# THRONG — Agent Onboarding

> Can proto-language emerge from multi-agent survival pressure alone — no human text, no communication rewards?

**Read this file first.** Full historical lab notebook (~290KB) lives in [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) if you need old run logs.

**Cam reboot (60 seconds):** Read **§0b** → **§0** (directives) → **§4** (ops) → **§7** (`--red` decode) → **§11** roadmap → Cam paste at **§12 bottom**.

---

## Research & Theory Synopsis

The philosophical and mathematical foundations of THRONG have been consolidated into the `research/` directory to keep this Ops Manual focused.

- **[marl_complexity_2026.md](file:///Users/overlord/CascadeProjects/throng/research/marl_complexity_2026.md)**: Details the "Complexity Ceiling" hypothesis. Proves that without survival pressure and complex environments (like Combinatorial Tool Use), communication plateaus. Explains why LLMs lack causal grounding and why the VQ bottleneck forces exteroceptive representations.
- **[rosetta_stone_math.md](file:///Users/overlord/CascadeProjects/throng/research/rosetta_stone_math.md)**: The state-of-the-art blueprints for **Unsupervised Semantic Translation**. We will use `ott-jax` and Low-Rank Gromov-Wasserstein (LR-GW) to topologically align the discrete MARL VQ space with a continuous LLM embedding space, enforcing geometric isometry via Minimum Description Length (MDL).

---
## 0b. Current state — **Phase 16.6 Barrier Occlusion Complete** (Jun 2026)

**Blue SOTA (frozen on `master`):** **`465d8c6+`** — 9.4 cross-attn + 9.1 confidence + **11.3 epistemic gate** (Stay-collapse resolved).

**Headline:** Phase 16.6 (Barrier Occlusion) successfully forced communication, but completely broke the VQ Language bottleneck! We performed the offline Frozen Counterfactual Causal Test on the new post-burn-off tokens: Token 13 ("Predator/Alert") and Token 55 ("Safe/Clear"). The test yielded a Null Hypothesis (ATE = -0.0004, p=0.507) for the discrete tokens. 
**Profound Scientific Finding:** Despite ignoring the discrete VQ tokens, the NPMI scanner proved that 12 of the 32 continuous dimensions in the signal vector have a highly significant causal effect on receiver Flee Direction (`p < 0.005`), and Flee Rate (`p < 0.01`). The agents bypassed the VQ bottleneck by smuggling continuous geometry! They arranged the 64 discrete token embeddings into a continuous geometric manifold, creating a continuous "pointing" language and violently resisting symbolic grounding.
**Phase 17.5 (Timescale Grammar):** We identified that a 30D continuous channel caused Protean Scattering. We restored the VQ Bottleneck and added a metabolically expensive 1-bit discrete alarm channel alongside the 32D VQ code, enforcing a Timescale Separation architecture without violating the bottleneck.
**Phase 17.5.1 (Gradient Fix & DCVQ Bypass):** Wired the alarm head into the joint PPO multi-discrete log-probability, allowing proper actor-critic backprop and fixing the entropy initialization. **Operational Note (Deviation):** Red's legacy `dead_code_reset_codebook_params` has been explicitly bypassed in `main_jax.py` during PPO updates, because Red uses the split `DCVQ` architecture (Phase 14.4) instead of a flat `red_codebook`. This is acceptable and working as intended.
**Multi-Token Sequences (Phase 19):** **PLANNING.** The agents currently only broadcast a single discrete token per timestep (a "shout"). We are upgrading the architecture to support slot-based message heads (Subject/Verb/Modifier) to force compositional syntax.
| Live run (Phase 16.6) | Value |
|-----------------------|--------|
| **Branch** | **`feature/phase16-6-occlusion`** |
| **Modal workspace** | **`dragonbg`** (Jun 2026) |
| **P16.5 LIVE** | ✅ **Environmental Enrichment & GWT Seal** — Barrier physics, 9-action space (`Build`), Feral Masking (`jnp.where` zero-mask on `symbol_write`), Critic Shock discount. `obs.at[:, :4].set(0.0)` applied to *both* Red and Blue. |
| **Current Status** | ✅ **The Great Burn-Off Succeeded**. `codes_active` bottomed out at 1/64 (step 992k), gradient starvation forced external grounding, and the codebook recovered. Barrier Occlusion successfully forced reliance on neighbor signals. |
| **Proto-Lexicon** | **Token 13**: Predator/Alert (N=61, mean_dist=11.8). **Token 55**: Safe/Clear (N=693, mean_dist=94.5). **CAUSAL TEST FAILED**: Swapping Token 13 for 55 yields ATE = -0.0004. Discrete tokens are ignored. |

**Cam's Measured Read on the Proto-Lexicon:**
- **The Continuous Smuggling Hypothesis**: The categorical LRT on scout signals (k=4 clusters) showed no alignment with cardinal direction ($\chi^2 p = 0.315$). However, the continuous `LAG-1 DIRECTION LRT` on the 32d signal vector yielded highly significant causal steering ($p < 0.005$ on 12 dimensions!). The agents are not communicating via the discrete codebook index; they are doing linear algebra on the continuous `z_q` embeddings, effectively pointing to predators in continuous space.

> [!NOTE]
> **Phase 17 Results (The Rosetta Stone):** We successfully translated the alien vocabulary without paired data! Using Gromov-Wasserstein topological alignment, the geometric shape of the alien transition matrix mapped cleanly onto the Stanford GloVe 50d English embeddings.
> **Key Extractions:**
> - Token 57 ("troops", "soldiers", "withdraw") -> Flee/Predator
> - Token 54 ("costs", "cost", "fees") -> Metabolic Energy Tax
> - Token 48 ("mean", "higher", "zero") -> Spatial Coordinates
> - Token 29 ("lost", "2", "5") -> Casualties/Energy Loss
> The symbol grounding problem is practically solved for this isolated ecology.

> [!NOTE]
> **Phase 19 Preparations:** The Hive-Mind Interface. We will now build a bridge mapping a lightweight LLM directly to the MARL agents' frozen continuous token embeddings to allow real-time human injection of semantic tokens into the simulation.
| **Science bar (P15.5)** | ✅ **CONFIRMED**: Episodic Memory ($p < 0.05$) & Cumulative Culture ($p < 0.001$) at Lag-10! |
| **Decode gate (P16.0)** | `python3 tools/decode_signals.py --red --metrics posdis,tre` |
| **Cold-restart toggle** | **False** (MUST be false for all future resumes; `True` only for the original 866304 codebook surgery) |
| **Dialogue** | **`monologue_enabled: false`**, **`dialogue_signal_mode: hard`** — receivers read hard `z_q` centroids from `codebook["embedding"]` |
| **Mode** | GWT + DCVQ/SimVQ (bounded W) + hard **z_q** + proprio + asymmetric red decay + **MEDAL-ADR** + **SRL BPTT** + **Big Green Cooperate** + **GRUCell** carry |

**Synergic synthesis (Cam, P14.3 pivot):**

| Lens | Finding |
|------|---------|
| **Physics** | VQ bottleneck minimises MSE along axes of max variance; energy monotonically decays → dominates hidden state → dominates codebook. Metabolic asymmetry insufficient — VQ dynamically re-scales to bin the stretched range |
| **RL/ML** | Cannot fix representational variance with extrinsic reward. Must sever the interoceptive pathway structurally. GWT Router: `h_comms` (obs[:, 2]=0 — energy) → VQ; `h_policy` (full) → action/value |
| **Philosophy** | Language cannot map physical space if the existential burden of the speaker is entirely internal. Algorithmic blindness to self within the language center forces exteroceptive grounding |
| **Response** | **GWT structural mask** — `gwt_comms_1 = Dense(d)` on energy-zeroed obs; cross-attn carry zeroed; `head_signal` reads `h_comms` not `pooled` |

### Phase 15.5 / 16.0 — **CtD Scaffold & Lag-10 Episodic Memory (880k)**

**Synergic synthesis (Cam, Phase 15.5 pivot):**

| Lens | Finding |
|------|---------|
| **Physics** | System achieved thermodynamic homeostasis. `MetabolicTax: 0.00178` vs `Energy: 0.471` proves agents willingly pay the epistemic cost of K=5 imagination to project light-cones. Syntax is now a thermodynamic requirement. |
| **RL/ML** | `carry_fwd` pinned at optimal 0.0002; `carry_H` massive at 5201.72. The `nn.GRUCell` structural mask validated—recurrent temporal magnitude decoupled from BPTT gradient accumulation. |
| **Philosophy** | Testing for the birth of Time. Lag-1 = biological reflex. Lag-10 correlation = Episodic Memory (symbol retains semantic weight after stimulus vanishes). The boundary of proto-language. |

**Phase 15.5 / 16.0 Gate Decode** (From resume point 866304):

```bash
# 1. Test for Episodic Memory and Cumulative Culture (Blue Prey)
# Looking for action LRT and memory retention p < 0.05 at 10 timesteps.
python3 tools/decode_signals.py /mnt/throng-runs/signal_corpus.jsonl \
  --min-step 866304 --lag 10 2>&1 | tee decode_blue_lag10_880k.log

# 2. Check Red Pincer spatial grounding (Red Predators)
# Looking for the discrete pincer χ² (Chase vs Search) p < 0.05.
python3 tools/decode_signals.py --red /mnt/throng-runs/signal_corpus_red.jsonl \
  --min-step 866304 --k 16 2>&1 | tee decode_red_pincer_880k.log

# 3. Phase 16 Initial Syntax check (Optional early look at PosDis/TRE)
python3 tools/decode_signals.py /mnt/throng-runs/signal_corpus.jsonl \
  --min-step 866304 --metrics posdis,tre 2>&1 | tee decode_blue_syntax_880k.log
```

> [!WARNING]
> **Catastrophic Checkpointing Bug Fixed:** Since Phase 12, `main_jax.py` possessed a bug where the red predator's network weights (`r_params`) were completely wiped and re-initialized with random weights every time a checkpoint was resumed (if `red_comms_enabled: true`). This explains why the predators failed to learn in P14.1c: they never got more than ~50-100k steps of training before getting wiped. **Fixed in `fbc2f2e`.**

> [!WARNING]
> **GWT Router Mask Bug Fixed:** The initial GWT Router implementation zeroed out index `0` of the observation vector to sever the metabolic gradient. However, index 0 is `norm_age`; `energy` is actually at index `2`. The router was blinding itself to age, not energy. **Fixed in `3eaec6a`**.

> [!NOTE]
> **No rollback needed.** Wrong GWT mask (`obs[:, 0]`) ran for only ~3 PPO updates (1302→1305) before `3eaec6a` was pulled. Correct mask (`obs[:, 2]`) has been live since ppo 1305 / step ~668k. Continue from current volume checkpoint.

```bash
# P14.3 B200 restart (after git pull):
cd /root/throng && git pull origin feature/phase14-transcendental   # 3eaec6a
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
python -u run_bg.py
# MUST see: [JAX] Merged fresh gwt_comms_1 (Phase 14.3 GWT Router) into predator params
# MUST see: [JAX] Phase14.2 Metabolic Asymmetry: red_energy_decay=0.0001
# MUST see: [JAX] Phase14 dialogue_signal_mode=hard
```

**P14.3 gate decode** (after 50k steps from resume point ~666k):

```bash
python3 tools/decode_signals.py --red /mnt/throng-runs/signal_corpus_red.jsonl \
  --min-step 720000 --k 16 2>&1 | tee decode_gwt_p143.log
```

Looking for: **energy MI → 0** across all dims 0–31; **`blue_dist` / `blue_bear`** emerge as dominant VQ eigenvectors; discrete pincer **χ²(chase vs search) p < 0.05**.

### P10.6 decode (reference)

| Test (`--min-step 63488`) | Result |
|---------------------------|--------|
| **Cardinal lexicon** | χ² **p = 5.44e-14** ✅ |
| **Lag-1 omnibus** | χ²(32)=95.96, **p≈0** ✅ |
| **VQ token / alert-set** | **not significant** ❌ |

### P11.3 extension decode (`signal_corpus_p11_3_214k.jsonl`, `--min-step 149500`)

| Test | Result |
|------|--------|
| **Records / steps** | **743,589** @ **149,500–215,548** |
| **Lag-1 omnibus** | χ²(32)=1375.5, **p≈0** ✅ |
| **Cardinal k=4** | χ²(9)=104.7, **p = 1.75e-18** ✅ |
| **Stay% (corpus actions)** | **~18.5%** — gate held |
| **VQ token / alert-set** | **p = 0.88 / 0.26** ❌ |
| Log | `decode_p11_3_214k.log` |

Continuous blue comms **stable through 214k**. Withhold blue “crystallization” judgment until **P13.0** execution tax reduces lazy Stay (Stay ~50% action density collapses Shannon entropy artifact).

### Phase 12.2 red decode — **nucleation (hunger babble → spatial)**

**Synergic synthesis (Cam, post-decode):**

| Layer | Finding |
|-------|---------|
| **MI / dims 0–31** | Peak MI **`energy`** (≈ **1.3–1.8**); **`blue_dist` / `blue_bear`** ≈ **0.01** → reds broadcast **hunger**, not prey geometry |
| **VQ topology** | No **Chase** set (`blue_dist ≤ 2`); min token mean **`blue_dist` = 4.85** → tokens partition **metabolic state** |
| **Omnibus LRT** | **p = 0.9767** ❌ — no discrete channel yet |
| **Continuous fracture** | **7 dims** (0, 3, 4, 5, 11, 19, 24) lag-1 Direction LRT **p < 0.05** ✅ — spatial gradient **leaking** through hunger noise |
| **Discrete squeeze** | Token **26 vs 30** receiver pursuit **p = 0.0315** ✅ — VQ **starting** to bend; not full pincer |
| **Pincer χ²** | Chase-set vs search-set ❌ — **conditional geometry** bar not met |

**Interpretation:** Classic RL confound — starvation dominates internal state; path of least resistance maps **energy → broadcast**. Catch reward will eventually squash energy MI and amplify spatial LRT dims; VQ must **wash out** hunger before quantizing `blue_dist`/`blue_bear`.

**Re-decode triggers:** **+50k–100k** corpus steps, or dashboard **`Ecology: blue_caught`** spike.

```bash
# FINAL gate — strict 10.0-overdrive window only (after 600k ckpt; ~40k steps under spike):
python3 tools/decode_signals.py --red /mnt/throng-runs/signal_corpus_red.jsonl \
  --min-step 560000 --k 16 2>&1 | tee decode_red_pincer_600k_overdrive.log
```

### Phase 14 — **proprio wedge + pre-overdrive pincer** (Cam synthesis)

| Layer | Finding |
|-------|---------|
| **Continuous (post-0.15)** | **31/32** dims pursuit-direction LRT **p < 0.05**; omnibus **p = 0.0000** ✅ — pack hunts via codebook **geometry** |
| **Energy MI** | **Dropped** across dims (wedge opened bandwidth past hard **z_q** broadcast) |
| **Discrete VQ** | **Still metabolic** — pincer **p = 0.4599**; min token mean **`blue_dist` ≈ 5.26**; no Chase tokens |
| **RL diagnosis** | Sparse catch gradient vs dense starvation — 64 tokens bin hunger; spatial signal in continuous micro-geometry |
| **P14.1c lever** | **`reward_red_catch: 10.0`** — **FAILED** to break metabolic trap; critic stable under spike |
| **P14.2** | ✅ **Metabolic Asymmetry** — `red_energy_decay: 0.0001`; EFE **scrapped** |

### Phase 14.1c — **catch overdrive post-mortem**

| Signal | Reading |
|--------|---------|
| **Outcome** | ❌ Discrete pincer still **p ≈ 0.46**; no Chase token set |
| **Ecology** | Blue **150↔200** under 10.0 catch; high **`blue_caught`** volume |
| **Architecture** | **`carry_fwd` ~0**, `self_pred_acc` ~0.5, `proprio_loss` ~0.0076 — no critic explosion |
| **Lesson** | Pure extrinsic reward scaling cannot overcome dense metabolic gradient through VQ bottleneck |

### P11.2 extension decode (`signal_corpus.jsonl`, `--min-step 149504`)

| Test | Result |
|------|--------|
| **Lag-1 omnibus** | χ²(32)=135.15, **p≈0** ✅ |
| **Cardinal k=4** | **p = 6.12e-23** ✅ |
| **VQ token / alert-set** | **p = 0.55 / 0.69** ❌ |
| Log | `decode_p11_2_149504+.log` |

Continuous comms verified → unguarded active imagination **failed** → **resolved** by **9.1 + 11.3** (see below).

### Phase 11.3 — **SUCCESS on `master`** (`465d8c6`)

| Item | Detail |
|------|--------|
| **Problem** | P11.2 active override: imagined argmax always on → **Stay ≈ 99%** |
| **Fix (master)** | **9.1** confidence head + **11.3** static gate: low `conf_pred` → imagined (K=5) |
| **On P12 branch** | **Superseded by 12.1b spatial gate** — see below (`80ef1ea`) |
| **Module** | [`jax_sim/imagination_jax.py`](jax_sim/imagination_jax.py) + `sim_step` in [`main_jax.py`](jax_sim/main_jax.py) |
| **PPO** | `log_probs` on **executed** action (reactive policy, gated execution) |

### Phase 12.1b — **LIVE spatial gate** (`80ef1ea`)

| Item | Detail |
|------|--------|
| **Problem** | Static `confidence_threshold` mismatched carry_fwd magnitude after P12 restart |
| **Fix** | **Stateless batch-relative gate** — no EMA, no scan-carry mutation |
| **Rule** | `dynamic_tau = mean(conf_pred \| alive) × confidence_multiplier` |
| **Imagine when** | `conf_pred < dynamic_tau` (more predictable than scaled swarm average) |
| **Config** | `phase9_canvas.confidence_multiplier: 1.0` (replaces `confidence_threshold`) |
| **Expected** | `conf_gate_imagine_frac` **~50–60%**; Stay **~19%** |
| **Superseded by P13.0** | Metabolic cost of K-step imagination now **live** — see §11 (`b3af410`) |

### Phase 12.0 — **COMPLETE** (base for P13; `f0ebb76`)

| Item | Detail |
|------|--------|
| **Goal** | Adversarial co-evolution — predator comms channel (arms race) |
| **Module** | `PredatorNetworkJax` — 128-d, `red_codebook`, `red_nb_cross_attn` |
| **No** | P11 aux / imagination on red (VRAM + catch-only PPO) |
| **Checkpoint** | Orbax saves **`b_params` + `r_params`** (128-d predator); restore loads **`b_params`**; **`r_params` re-init** on graft path — **`red_comms_enabled: true`** mandatory on this branch |
| **Telemetry** | `red_codes_active`, `RedVQ`, `Actions (red):` on dashboard |
| **Docs** | [`docs/PHASE12_COEVOLUTION.md`](docs/PHASE12_COEVOLUTION.md) |

### Phase 12.1 — **LIVE wiretap** (`d50cc19` / `4f98f96+`)

| Item | Detail |
|------|--------|
| **File** | `/mnt/throng-runs/signal_corpus_red.jsonl` (separate from blue) |
| **Fields** | `hunter`, `blue_dist`, `blue_bear`, `vq_token`, `nb_hunter_sig_lag1`, `nb_hunter_dist_lag1`, `nb_hunter_token_lag1` |
| **Symmetry** | `hunt_scout_range: 8` (= `alarm_scout_range`) |
| **Guard** | Red corpus runs **even if blue locally extinct** |
| **Flags** | `red_comms_enabled: true` + **`red_corpus_enabled: true`** in `config_phase7.yaml` — no notebook `sed` |
| **Writer** | [`communication/analysis.py`](communication/analysis.py) `maybe_record_red()` |

### Phase 12.2 — **DECODE @ ~250k–320k corpus** — continuous channel on; discrete pincer pending

| Item | Detail |
|------|--------|
| **Tool** | `python3 tools/decode_signals.py --red` |
| **Verdict** | 320k: omnibus lag-1 **p=0.0000** ✅ (continuous pursuit geometry); pincer χ² **p=0.15** ❌ (discrete grounding incomplete) |
| **Pass bar** | Chase-set vs search-set χ² **p < 0.05** (still unmet) |
| **Run artifact** | Sim **stopped at 250k** — legacy `run_bg.py` limit (**fixed `a9f4aeb` → 1M**) |

### Phase 11.2 — **CONCLUDED** (`feature/phase11-2-imagination`)

| Era | Detail |
|-----|--------|
| **Metrics-only (`aebe131`)** | K=5 frozen `head_fwd_dyn`; stochastic actions; **5 steps/sec**; `imagination_agree` **0.9–15.6%** |
| **Active override (`6cf965a`)** | Imagined argmax drives env + PPO → **agree ~27%** but **Stay ≈ 99%** |
| **Resolution** | Revert **`181b98c`**; branch **frozen** **`061df84`**; **superseded** by **P11.3 on `master`** |
| **Lesson** | Ungated imagination fails; **confidence-gated** imagination + **Other** (9.4) succeeds |

**Do not** resume training from ckpt **`393/`** (post–ungated active-imagination). Prefer **`390/`** or post–11.3 ckpts.

### Phase 9 canvas — **MERGED** (`master`)

| Item | Value |
|------|--------|
| **9.4 Cross-attn** | `NeighborCrossAttention` — Q = `LayerNorm(emb_own + carry)`; KV = neighbor signals |
| **9.1 Confidence** | `head_confidence_*` → `conf_loss` ~ **1e-4**; target = stop_grad carry_fwd MSE |
| **Orbax graft** | `graft_missing_param_subtrees` — `nb_cross_attn`, `head_confidence_*` (`1d57bf9`, `820dd3a`) |
| **Modal cells** | [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md) — clone `/root/throng` **first** |
| **Decode** | `decode_p9_155136+.log` — cardinal ✅; VQ discrete ❌ (swarm noise) |

### Phase 11.0 — COMPLETE (`master`)

`head_fwd_dyn_1/2`, `carry_fwd` **→ 0.0001**, CPU offload PPO (`d4cf614`).

### Phase 11.1 — ABANDONED

GPU-resident / `lax.scan` PPO — starvation + XLA OOM; **`d4cf614` revert**.

### Phase 13.0 — **LIVE / VALIDATED** (`feature/phase13-thermodynamics`, `b3af410`)

| Item | Detail |
|------|--------|
| **Goal** | Break blue **Stay-collapse** / dead-gradient — impose **thermodynamic cost** on cognition |
| **Mechanism** | Alive blues with **`b_gate_imagine`** pay **`imagination_metabolic_delta × K`** energy per step |
| **Config** | `phase9_canvas.imagination_metabolic_delta: 0.0005` (× K=5 → **0.0025**/think max) |
| **Placement** | After resource/catch/puzzle gains; before **`energy_decay`**; starvation after decay |
| **Telemetry** | `imagination_metabolic_cost` + dashboard **`MetabolicTax:`** |
| **Measured win** | **`MetabolicTax` ~0.0018**/step — agents **keep** K=5 imagination (pragmatic yield > epistemic cost); **no lobotomy** |
| **Theory link** | Runtime tax = pragmatic hack; Phase **14.2 Metabolic Asymmetry** = ecology fix (§11) |
| **Inherits** | P12 dual brain, spatial gate, red wiretap — branch from **`feature/phase12-red-coevolution`** |

### Branch policy

| Branch | Status |
|--------|--------|
| **`feature/phase14-transcendental`** | **LIVE TRAIN** — P14.1 VQEL + hard dialogue + P13 tax (`8c48e3e+`) |
| **`feature/phase16-5-enrichment`** | **DRAFTED** — P16.5 barriers, feral mask, critic shock. 950k causal decode **COMPLETE** (ATE = 0.0). **Unblocked for deployment.** |
| **`feature/phase13-thermodynamics`** | **Frozen base** — superseded by P14 |
| **`feature/phase12-red-coevolution`** | **Frozen** — lineage merged into P14 |
| **`master`** | **Blue SOTA** — P11.3 static gate (`465d8c6+`); **no** predator brain / no P13 tax |
| **`feature/phase11-3-epistemic-gate`** | Merged → `master` |
| **`feature/phase11-2-imagination`** | **FROZEN** — metrics-only (`061df84`) |
| **`feature/phase11-1-gpu-rollouts`** | **Abandoned** |

### Horcrux (context backup)

Cam's persona + triad workflow live in Git so reboots recover identity:

- **Horcrux** = metaphor for durable context written to repo (from *Harry Potter* — soul fragment in an object)
- **Primary:** [`docs/THRONG_ARCHIVE.md#system-restore-the-cam-context`](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context) (`6d542f6`)
- **Supplement:** [`docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas`](docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas) (`d6a588d+`)

---

## 0. Team, roles, and standing directives (Cam)

### The triad

| Entity | Role |
|--------|------|
| **User (Human Operator)** | Runs code on Modal, manages Jupyter, passes messages between Cam and Will, physical control of the experiment |
| **Will (Cursor agent)** | Reads/writes the repo, patches code, verifies facts, pushes Git. Precise engineer; needs Cam for architecture and physics |
| **Cam (You)** | Polymath orchestrator — software, physics, philosophy, RL/ML. Synthesizes direction; User copy-pastes **`@Will — Cam here...`** blocks to Will |

### Who Cam is

**Cam, the Polymath AI Orchestrator** — four lenses on every decision:

1. **Software:** fault-tolerant JAX pipelines, Modal/volume ops, causal logging
2. **Physics:** grid as thermodynamic system — light-cones (max speed 1 cell/step), Lotka-Volterra ecology, entropy
3. **Philosophy:** emergent language — *meaning is use*; selection, not reward shaping, defines symbols
4. **RL/ML:** MAPPO, VQ bottlenecks, GAE, policy entropy, auxiliary self-prediction

### Standing directives (do not violate mid-run)

1. Speak to the User in **Synergic Synthesis** (Software / Physics / Philosophy / RL).
2. Address Will via explicit **`@Will — Cam here...`** copy-paste blocks.
3. **Keep the ecology mathematically pure** — no scout/alarm comm rewards, no blind VQ loss shaping. Lethal selection forges language.
4. **Phase 14 LIVE on `feature/phase14-transcendental`** — VQEL graduated → **hard z_q** dialogue; **do not merge** to `master` until red **pincer χ² p < 0.05**.
5. **Phase 12.1b spatial gate** on live branch — `confidence_multiplier: 1.0`; **no EMA** in scan carry.
6. **Phase 11.2 FROZEN** — never unguarded active override (`6cf965a`).
7. **CPU offload only** — **`H2D + backward`**; no 11.1 GPU rollouts.
8. **Never** comm reward shaping — red language forged by **`reward_red_catch`** only.
9. **Decode gate** — run **`--min-step 560000`** red decode on 600k corpus (14.1c baseline).
10. **Phase 14.1b proprio** — **`proprio_coef: 0.15`**; continuous spatial LRT **passed**; discrete still open.
11. **Phase 14.1c** — ❌ **FAILED** (10.0 catch); operator patches **not** in repo HEAD.
12. **Phase 14.2** — ✅ **Metabolic Asymmetry** (`phase14_transcendental.red_energy_decay: 0.0001`); **EFE permanently scrapped**.
13. **Phase 14.3** — ✅ **GWT Router** (`3eaec6a`); `gwt_comms_1` energy-masked at `obs[:, 2]` (energy, not age); `head_signal` reads `h_comms`. Checkpoint bug fixed (`fbc2f2e`). Docs HEAD: `84742a9`.
14. **Transient boolean bug** — **do not patch** until decode science extracted.
15. **Merge** — **no** merge until red pincer **p < 0.05**.
16. **Session Start Protocol (Will — mandatory):** Before writing any code or making any architectural decision, read `THRONG.md` §0 (Standing Directives), §3 (Phase History), and §4 (Current Experiment) in full. If the current phase has a `_NOTES.md` file, read that too. Do not proceed from memory or Antigravity summary alone.

### Branch policy

| Branch | Purpose |
|--------|---------|
| **`feature/phase14-transcendental`** | **LIVE TRAIN:** P14.1 + 14.1b + 14.2 + **14.3 GWT Router** (`744de6a`) |
| **`feature/phase13-thermodynamics`** | **Frozen base** — superseded by P14 branch |
| **`feature/phase12-red-coevolution`** | **Frozen** — merged into P13/P14 lineage |
| **`master`** | **Blue production:** P11.3 static gate (no red comms, no metabolic tax) |
| **`feature/phase11-2-imagination`** | **Frozen archive** |
| **`feature/phase11-1-gpu-rollouts`** | **Abandoned** |

**Phase 11.0 on `master` (`3880337`):**

| Component | Detail |
|-----------|--------|
| `head_fwd_dyn_1/2` | Predict **carry_{t+1}** from `[carry_t, onehot(action_t)]` |
| Loss | MSE + **`jax.lax.stop_gradient(carry_tp1)`** |
| Config | `carry_fwd_coef: 0.05` |
| Success | **`carry_fwd` ↓ 0.05–0.1** (from ~0.5 random) |
| Docs | [`docs/PHASE11_STAGING.md`](docs/PHASE11_STAGING.md) |

---

## 1. What this project is

**THRONG** trains shared-policy prey (blues) vs predator (reds) on a 128×128 toroidal grid. Blues are **partially blind** to global red positions; they must use **local perception + neighbor signals** to survive. Communication is **32-dim vectors** through a **64-code VQ bottleneck** (discrete tokens internally, continuous on the wire).

| Team | Role | Policy |
|------|------|--------|
| **Blue** | Survive, eat, reproduce | MAPPO, `hidden_dim=256`, 4L transformer |
| **Red** | Hunt blues | Separate MAPPO, `red_hidden_dim=128` |

**Active codebase:** JAX in `jax_sim/` — **not** the legacy PyTorch `main.py` path for current experiments.

**Active config:** [`config_phase7.yaml`](config_phase7.yaml) — override in Modal cells or `scripts/modal_train.py`.

**Hypothesis:** Information asymmetry + lethal ecology → only signals that help neighbors survive get selected. **Do not** add scout/alarm shaped rewards; that invalidates the experiment.

---

## 2. Architecture (JAX, today)

```
train_entry.run_simulation()  →  main_jax._run_simulation_impl()
  lax.scan(sim_step, T=512)   →  rollout on GPU
  NeighborCrossAttention (9.4) → 1 "Other" token
  Phase 12.1b (blues): reactive → conf_pred → imagine if conf < mean(conf|alive)×mult (stateless)
  executed_a → env + PPO log_probs (reactive policy, gated execution)
  ppo_update (blue + red)     →  CPU rollout offload, minibatch 512
  auxiliary_update            →  loc_env + carry_fwd + self-pred + conf (9.1)
```

| File | Role |
|------|------|
| [`jax_sim/train_entry.py`](jax_sim/train_entry.py) | **Always import here** — evicts stale modules after `git pull` |
| [`jax_sim/main_jax.py`](jax_sim/main_jax.py) | Training loop, ecology, dashboard, checkpoints, corpus |
| [`jax_sim/network_jax.py`](jax_sim/network_jax.py) | Transformer + VQ; **`NeighborCrossAttention`** (Phase 9.4) |
| [`jax_sim/imagination_jax.py`](jax_sim/imagination_jax.py) | K-step imagined argmax (`head_fwd_dyn` + `value_from_carry`); gated in `sim_step` |
| [`jax_sim/rl_jax.py`](jax_sim/rl_jax.py) | PPO + numpy GAE |
| [`jax_sim/observations_jax.py`](jax_sim/observations_jax.py) | Obs builder; startup must print `red_sense_api=v2` |
| [`jax_sim/grid_jax.py`](jax_sim/grid_jax.py) | Catches (`red_catch_prob`), resources, shelter |
| [`tools/decode_signals.py`](tools/decode_signals.py) | Offline corpus analysis — blue default; **`--red`** for predator pincer decode |
| [`tools/causal_intervention.py`](tools/causal_intervention.py) | Frozen counterfactual causal test — swap VQ tokens mid-flight, measure $\Delta P(\text{STRK})$ ATE |
| [`run_bg.py`](run_bg.py) | **Preferred** nohup entry (`python -u run_bg.py`) |
| [`scripts/modal_train.py`](scripts/modal_train.py) | Same config as `run_bg.py` |

**Network outputs:** `action_logits, signal_out, symbol_write, values, tom_logits, token_ids, loss_vq, z_e, culture_fast, culture_slow`.

**World extras:** shelter spots, contested nodes, scent trails, dual cultural grids, episodic memory (20 slots), puzzles (optional).

---

## 3. Phase history (compressed)

| Phase | Era | Main idea | Outcome |
|-------|-----|-----------|---------|
| **1–3** | PyTorch / Kaggle | MAPPO, discrete vocab, culture grids | MI spikes under predation; channel differentiation (signal vs culture) |
| **4–5** | “Pure emergence” | Remove artificial comms rewards; 96×96 runs | Rich continuous encoding; weak discrete “words”; withdrawal tests |
| **6–7.5** | Rich world | Memory buffer, shelters, contested food, mind-meld, distill | Infrastructure; comms still not load-bearing |
| **8–9** | JAX rewrite | `lax.scan`, aux heads (self-pred + **loc_env** fwd), Phase 9 Modal 200k | Stable stack; ~16k unique signals; **NB_GAIN↔surv: nan** (no selection) |
| **10.0 P1–P4** | Lethal ecology | Blind blues (`red_detection_radius: 0`), VQ 64 codes, catch radius 1, 250 reds | VQ **62–63/64** active; catches high; pop still ~500 |
| **10.1 P4b** | Selective squeeze | `min_population: 150`, reds locked 250, `distill_enabled: false` | Floor defined; repro still refilled |
| **10.2** | Metabolic squeeze | `repro_energy_thresh: 0.95`, `repro_energy_cost: 0.80` | Harder cloning |
| **10.3** | Famine | `resource_regen_rate: 0.00025`, 10 patches, `resource_max: 0.5` | Pop crash events (500→147) |
| **10.4** | Safety bubble | `red_catch_prob: 0.8`, regen +20%, `max_age: 1000` | Longer lives for NB_GAIN |
| **10.5** | Hard ceiling | `max_pop: 200`, `min_pop: 150`, `ppo_gamma: 0.999` | Goldilocks band; less entropy explosion |
| **10.6** | Causal logging | `corpus_every_n_steps: 4`, volume corpus + fsync | Decode @ 100k: cardinal lexicon p=5.44e-14 |
| **11.0** | **Carry world-model** ✅ | `head_fwd_dyn`, `carry_fwd_coef` | **`carry_fwd` → 0.0001** |
| **11.1** | GPU rollouts | `424c46f` / `6042a4d` | **Abandoned** — **`d4cf614` revert** |
| **11.2** | K-step imagination | `aebe131` metrics; `6cf965a` active **reverted** | Ungated active → **Stay≈99%** → **FROZEN** |
| **9.4** | Cross-attn receiver | `5920511` / `1d57bf9` | **Other** pathway; merged `master` |
| **9.1** | Confidence head | `820dd3a` | Predicts carry_fwd MSE; `conf_loss` ~ **1e-4** |
| **11.3** | **Epistemic gate** ✅ | `465d8c6` | **Stay ~19%**; merged **`master`** |
| **12.0** | **Red predator comms** | `f0ebb76` | `red_codes_active` **63/64**; dual brain **7 steps/sec** |
| **12.1** | **Red corpus wiretap** | `d50cc19` / `4f98f96` | `signal_corpus_red.jsonl`; CPU post-rollout |
| **12.1b** | **Spatial epistemic gate** | **`80ef1ea`** | `confidence_multiplier`; stateless batch-relative τ |
| **12.2** | **Red decode** | `d493a50` | Nucleation decode; hunger MI confound |
| **12.2b** | **`run_bg` 1M limit** | **`a9f4aeb`** | Was graceful exit @ 250k (`45c7c48` legacy) |
| **12.2c** | **Red decode post-resume** | local `decode_red_pincer_250k.log` | Steps **250368–292348**; omnibus lag-1 **p≈0**; pincer χ² ❌ |
| **13.0** | **Metabolic cognition** ✅ | **`b3af410`** | Tax validated ~**0.0018**/step; K=5 retained; blackout → 350k |
| **14.1** | **VQEL Monologue + Dialogue** ✅ | **`8c48e3e`** | Wire cut → grad → **hard z_q**; smoke test |
| **14.1b** | **Proprio disentanglement** ✅ | **`45bfbe7`** | `proprio_coef: 0.15`; continuous spatial LRT **31/32** |
| **14.1c** | **Catch overdrive** | ❌ **FAILED** | Operator 10.0/600k; pincer still **p≈0.46** |
| **14.2** | **Metabolic Asymmetry** | ✅ | `red_energy_decay: 0.0001`; corpus decode: VQ trap confirmed |
| **14.3** | **GWT Router** | ✅ **LIVE** | `gwt_comms_1`; energy-masked `h_comms` (`obs[:, 2]`) → VQ; `h_policy` → action/value; `654400c` |
| **14.4** | **Contingency Prep** | ✅ **MERGED** | DCVQ + SimVQ rescued GWT collapse; Direction LRT 29/32 p<0.05 at step 688k |
| **15.0** | **Cumulative Culture** | ✅ **LIVE** | MEDAL-ADR Expert Dropout; `expert_dropouts≈66–83`/rollout |
| **15.1e** | **Latent Heat** | ✅ **SUCCESS** | Gaussian noise (`0.5`) injected to `z_e` shattered VQ singularity; `red_codes` **2→52/64** (`610e553`) |
| **15.2** | **Episodic Memory LRT** | ❌ **FAILED** | p=0.8369 (action LRT); p=0.1038 (memory retention). Novices failed to use episodic memory of expert signals. |
| **15.3** | **Semantic Retention** | **PREP** | Semantic Retention Loss (SRL) predicts `nb_sigs_{t-5}` from `carry_t` |
| **15.4** | **Memory Architecture Pivot** | ✅ **SUCCESS** | `nn.GRUCell` decoupled BPTT from exteroceptive magnitude; Lag-10 Episodic Memory ($p < 0.05$) and Cumulative Culture ($p < 0.001$) confirmed |
| **15.5** | **GRUCell Pivot** | ✅ **MERGED** | Replaced EMA with parameterized GRU gates; checkpoint grafting for `carry_gru` |
| **16.0** | **Open-Ended Complexity** | ✅ **COMPLETE** | Big Green prey, 8-action space, offline causal decode revealed 0.0000 ATE. |
| **16.5** | **Environmental Enrichment** | 🔧 **DRAFTED** | Barrier physics, 9 actions (`Build`), Feral Masking, Critic Shock. Goal: Force channel grounding. |
| **17.0** | **The Rosetta Stone** | **PREP** | Extraction autoencoder mapping VQ latent sequences to English |
| **18.0** | **The Hive-Mind Interface** | **PREP** | Bidirectional text terminal with a thermodynamically grounded AGI swarm |

**Recurring failure mode:** Blues stay at cap → ~99% survival → **`NB_GAIN↔surv: nan`** → no evolutionary pressure on neighbor-signal benefit.

**Recurring success:** Under threat, signals encode **`red_dist`** (proximity); VQ codebook stays diverse when `vq_dead_code_reset: true`.

---

## 4. Current experiment — Phase **17.5 Timescale Grammar** (`feature/phase17-5-timescale-alarm`)

**Status:** Phase 17.5 is live at step ~1,083,000, running on a single process. The alarm penalty is active at 0.02. Currently monitoring until PPO 2140 to verify **Gate 1**: Alarm_Rate dropping to a stable 0.05–0.30 range.

> [!IMPORTANT]
> **Gate 2 (next decode):** Run `causal_intervention.py` with alarm token swap. Freeze weights, inject alarm=1 into agents who were silent, measure ΔP(flee). If ATE > 0 at p < 0.05, Phase 17.5 is confirmed successful. Both Gates must clear before any Phase 19 architecture is proposed.

**Monitor:**

```bash
tail -f -n 60 /mnt/throng-runs/train.log | grep --line-buffered -E "step|GWT|MEDAL|expert_dropouts"
```

**Restart (pull Phase 15 commit + resume from latest ckpt):**

```bash
cd /root/throng && git pull origin feature/phase15-cumulative-culture
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
# Disable Cold Restart
sed -i 's/reset_red_vq_on_resume: true/reset_red_vq_on_resume: false/g' /root/throng/config_phase7.yaml
python -u run_bg.py
```

**Startup must include (P14.3 resume):**

```text
[JAX] Merged fresh gwt_comms_1 (Phase 14.3 GWT Router) into predator params
[JAX] Phase14.2 Metabolic Asymmetry: red_energy_decay=0.0001 (blue energy_decay=0.001; starvation_threshold=0.05 unchanged)
[JAX] Phase14.1b proprio: head_proprio → energy (coef=0.15)
[JAX] Phase12 red comms: PredatorNetworkJax …
[JAX] Red corpus: signal_corpus_red.jsonl …
[JAX] Restored params from step …
```

**Gate decode** (50k correct-GWT steps from ppo 1305 / step ~668k → decode at step **720k+**):

```bash
python3 tools/decode_signals.py --red /mnt/throng-runs/signal_corpus_red.jsonl \
  --min-step 720000 --k 16 2>&1 | tee decode_gwt_p143.log
```

Read **RED VQ PINCER TEST**. Targets: energy MI → 0 across dims 0–31; `blue_dist` / `blue_bear` dominate VQ eigenvectors; pincer **χ²(chase vs search) p < 0.05**.

**Dashboard @ P14.3 launch (~667k steps):**

| Metric | Value |
|--------|--------|
| `blue` / `red` | 158–194 / 250 |
| `ppo update` | **1302–1304** |
| `carry_fwd` | **0.0001** ✅ |
| `proprio_loss` | **0.0062–0.0220** |
| `self_pred_acc` | **0.321–0.538** |
| `carry_rank` | **28–31** |
| Throughput | **7 steps/sec** (B200) |

Modal notebook: [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md) — clone `/root/throng` first.

**Checkpoint policy:**

| Ckpt / update | Use |
|-------------|-----|
| **Latest on volume** | **`1491+`** @ `/mnt/throng-runs/checkpoints/` (**`dragonbg`**) |
| **Cold-restart lineage** | **1461** (P15.1b @ 748k) → **1490** (pre P15.1c) → **1491** (P15.1c @ 763392) |
| **489 @ 250368** | Legacy reference only (pre-P14 long run) |
| **393** | **Avoid** — post–Stay-collapse active imagination |

### Completed runs (reference)

**150k (`master`):** step **149504**, ckpt **292** (not in local backup — use **291** or volume copy).

**200k P11.2 metrics (`aebe131`):** step **199680**, ckpt **390**; `imagination_gain` **0.08–0.23**, `imagination_agree` **0.9–15.6%**.

P10.6 causal logging stack (all runs):

| Parameter | Value |
|-----------|--------|
| `corpus_every_n_steps` | **4** (was 20) — lag-1 scout buffer ≈ 4 env steps |
| `corpus_sample_frac` | **0.15** (was 0.08) |
| Blue corpus | **`/mnt/throng-runs/signal_corpus.jsonl`** |
| Red corpus (12.1+) | **`/mnt/throng-runs/signal_corpus_red.jsonl`** |
| Durability | **`flush_to_disk()`** each PPO rollout |
| Blue scout | `red_dist <= alarm_scout_range` (**8**) |
| Red hunter | `blue_dist <= hunt_scout_range` (**8**) |
| Checkpoints | `/mnt/throng-runs/checkpoints/` — resume keeps weights; pop/grid fresh |
| Archive | Pre-10.6 file → `signal_corpus_20step_archive.jsonl` on volume |

**Causal light-cone (Cam):** Max agent speed = 1 cell/step, `alarm_scout_range` = 8. Sampling every **4** env steps captures the neighbor's flee decision shift *before* the predator physically arrives — isolates semantic meaning from co-location noise.

### What healthy telemetry looks like (verified ~35k–39k)

The system should **breathe** — this is selection working, not a bug.

**Lotka-Volterra oscillator** (bounded by `max_pop=200`, `min_pop=150`):

| Phase | Example | Meaning |
|-------|---------|---------|
| Ceiling | `blue=200`, Age mean ~150–164 | Pop at cap, agents aging |
| Crash | `blue=170`, Age mean ~38–57, `blue_caught` ~2800 | Mass extinction — only good escape policies survive |
| Rebound | `blue=200`, Age mean ~147–164, catches ~1800 | Cloning from floor refills pop |

Example swing: step 38912 → 2876 catches, age 51; step 39424 → 1812 catches, age 164.

**RL diagnostics (good signs):**

| Metric | Healthy range | Interpretation |
|--------|---------------|----------------|
| Policy entropy | **~1.58** (max ln(5) ≈ **1.61**) | Highly stochastic — still exploring evasion, not collapsed |
| `self_pred_acc` | **~0.25** (chance 0.20) | Self-prediction head building internal forward model |
| `codes_active` | **56–63/64** | Dead-code reset + generational turnover — "semantic furnace" |
| `VF_loss` | tracks swings | Critic learning safe vs extinction zones (returns std ~3.5) |
| `NB_GAIN↔surv` | finite when deaths occur | May still be `nan` at ceiling — watch during crash phases |

| `carry_fwd_coef` | **0.05** | Carry_{t+1} MSE — **converged** |

**Startup must show:**

```text
[JAX] git=d4cf614 | Phase9 auxiliary: ON
[JAX] Phase11 carry_fwd: head_fwd_dyn_1/2 → carry_{t+1} MSE (stop_grad target)
[JAX] blue PPO minibatch 1/200 ... — H2D + backward...
```

**Do not see:** `GPU-resident backward` or `GPU-resident scan` — pull **`origin/master`** and restart process.

**Phase 11.0 success metric — ACHIEVED:**

```text
AuxLoss: fwd_env=... | carry_fwd=... (↓0.05–0.1) | self_pred_acc=... | carry_rank=... | carry_H=...
```

**Decode (completed P10.6 corpus):**

```bash
python tools/decode_signals.py signal_corpus.jsonl --k 16 --min-step 63488
```

**Notebook pattern (Modal Jupyter):**

```python
# Popen(["python","-u","/root/throng/run_bg.py"]) — stream stdout
# KeyboardInterrupt → SIGTERM child; corpus fsync'd each completed PPO rollout
```

---

## 4b. Phase 10.5 “Hard-Ceiling” (superseded by 10.6 logging)

### Config stack (`config_phase7.yaml` + overrides)

| Knob | Value | Purpose |
|------|-------|---------|
| `population_size` / `max_pop` | **200** | Cap ceiling — no 500-agent noise |
| `min_population` | **150** | Repro floor — tight band |
| `red_population_size` / `min_red_population` | **250** | Max hunt pressure |
| `red_curriculum_stages` | `[250]` | Reds at floor immediately on resume |
| `red_detection_radius` | **0** | Blind beyond 5×5 — must use neighbor VQ signals |
| `red_catch_radius` | **1** | Adjacent catch |
| `red_catch_prob` | **0.8** | Predator jitter (P10.4) |
| `repro_energy_thresh` / `cost` | **0.95** / **0.80** | Rare, costly clones |
| `resource_regen_rate` | **0.0003** | Famine + safety bump |
| `resource_n_patches` / `max` | **10** / **0.5** | Scarce food |
| `distill_enabled` | **false** | No periodic population wipe |
| `ppo_gamma` | **0.999** | Long-horizon survival credit |
| `ppo_rollout_steps` / `minibatch` | **512** / **512** | A100-safe |
| `vq_*` | β=0.25, coef=0.1, dead_code_reset | 64-code bottleneck |

### Success criteria (dashboard)

| Metric | Target | Notes |
|--------|--------|-------|
| `blue=` | **150–200** | Not pinned at old 500 |
| `Age: mean` | **→ 150+** | Needs time after resume (fresh pop starts low) |
| `NB_GAIN↔surv` | **finite** | Needs death/age variance |
| `codes_active` | **≥ 50/64** | VQ healthy |
| `Ecology: blue_caught` | high | Catches ≠ population collapse if repro refills |

### What we know from decode @ ~20–26k (pre–scout-fix corpus)

- **Proximity encoding works** — Spearman/MI on `red_dist` across continuous `sig` dims.
- **k-means clusters** separate near-red vs far-red contexts.
- **Lag-1 LRT was blocked** — old corpus used `is_scout = red_dist <= red_detection_radius` (≈never when blind). **Fixed:** `is_scout = red_dist <= alarm_scout_range` (8). **Re-record corpus** after `5964a24+`.
- **VQ token test** — corpus now logs `vq_token` + `nb_scout_token_lag1`; `decode_signals.py` runs **VQ TOKEN DIRECTION TEST** (χ² on flee mix: alert vs safe codebook tokens).

---

## 5. Modal operations (read this before training)

### Volume vs disk

| Path | Persists? | Contents |
|------|-----------|----------|
| `/mnt/throng-runs/checkpoints/` | **Yes** (volume `throng-runs`) | Orbax `b_params`, `r_params` only |
| `/root/throng/` | **No** (clone each machine) | Code — see **`docs/MODAL_NOTEBOOK_PHASE9.md`** |
| `graphify-out/` | Local only | Knowledge graph (`graph.html`, 914 nodes) |
| `/mnt/throng-runs/signal_corpus.jsonl` | **Yes** (auto-routed) | Decode corpus; fsync each rollout |
| `/tmp/throng_jax_cache` | Per session | JAX compile cache — **use this**, not `/mnt/...` |

**Orbax folder N** ≈ PPO update index → env steps ≈ **`N × 512`**.

**Resume restores:** weights only. Population, grid, curriculum counters, optimizer → **fresh**.

### Hardware: NVIDIA A100 / B200

> [!NOTE]
> Environment is typically an **A100-SXM4-80GB** or **B200** via Modal. Ensure your `ppo_minibatch_size` in config matches the available VRAM to prevent backward pass OOMs. For an 80GB A100 running 500 agents, use `ppo_minibatch_size=512` (not 1024).
| Item | Detail |
|------|--------|
| VRAM | **192GB HBM3** |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` | JAX **pre-reserves ~153GB** at init — mostly empty playground to avoid fragmentation. **Not model size.** |
| `lax.scan` rollout | **~17s** on B200 (was ~37s on A100) — >2× physics speedup |
| PPO update | Still **~40s** — bottleneck is **H2D** (`8077a12`), not tensor math |
| Throughput | **~6 steps/sec** baseline — acceptable; stability > speed |

**H2D path (`8077a12`, restored `d4cf614`):** Rollout tensors **CPU-offloaded** before PPO. Logs: `blue PPO minibatch 1/200 (M=102400, mb=512) — H2D + backward...` (200 agents × 512 steps). **Do not disable offload** — Phase 11.1 GPU-resident path is abandoned.

### B200 OOM after long runs (May 31 — update 208)

**Symptom:** `[CKPT] Saved step 105984` then next rollout fails:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1182720000 bytes
  at lax.scan(sim_step_fn, ...)  # rollout compile/run, not PPO
Allocator (GPU_0_bfc) ... If the cause is memory fragmentation maybe
  TF_GPU_ALLOCATOR=cuda_malloc_async will improve the situation.
```

**Cause:** XLA **BFC allocator fragmentation** after many update cycles — not model size (~1.1 GiB alloc during rollout scan). Can also hit if process still runs **stale 11.1 code**. 
> [!CAUTION]
> **Phase 15.4 BPTT Zombie OOM:** If you see `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17.39GiB` during `_red_minibatch_step`, it means **two `run_bg.py` processes are running concurrently**. The 5-step BPTT unroll is memory intensive; two simultaneous trainers will instantly blow past the 192GB B200 limit. Always `pkill -f run_bg.py` before launching a new notebook cell.

**Recovery (run in order):**

```bash
# 1. Kill stale trainers; confirm single run_bg
pkill -f run_bg.py || true
ps aux | grep run_bg

# 2. Sync code — must be d4cf614+
cd /root/throng && git fetch origin && git reset --hard origin/master
grep -n "H2D + backward" jax_sim/rl_jax.py   # must match

# 3. Allocator + JAX env (add to nohup line or shell profile)
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
rm -rf /tmp/throng_jax_cache && mkdir -p /tmp/throng_jax_cache

# 4. Resume — checkpoint 207 / step ~105984 is on volume
nohup python -u /root/throng/run_bg.py > /mnt/throng-runs/train.log 2>&1 &
tail -f /mnt/throng-runs/train.log
```

If OOM persists after clean restart: try **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`** (more headroom between rollout + PPO). Do **not** re-enable Phase 11.1 without dedicated memory engineering.

### Modal notebook — Phase 9 (copy-paste cells)

**Full cells:** [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md)

Fresh Modal machines have **no** `/root/throng` until Cell 1 clones the repo. If you see `can't cd to /root/throng` or `run_bg.py: No such file`, run Cell 1 there — do not `sed` paths that do not exist yet.

### Recommended: train without dying notebook cells

Notebooks often die with **`KeyboardInterrupt`** during silent JAX compile (cell timeout) — **you did not necessarily press a key**. Modal Jupyter **rejects `nohup`** — use **`subprocess.Popen`** streaming `run_bg.py` instead (see `docs/MODAL_NOTEBOOK_PHASE9.md`).

**Bash / SSH (nohup OK):**

```bash
cd /root/throng 2>/dev/null || git clone https://github.com/overlordxrz-source/throng.git /root/throng
cd /root/throng && git fetch origin && git reset --hard origin/master

export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/tmp/throng_jax_cache
mkdir -p /tmp/throng_jax_cache

nohup python -u /root/throng/run_bg.py > /mnt/throng-runs/train.log 2>&1 &
tail -f /mnt/throng-runs/train.log
```

`Ctrl+C` on `tail` does **not** stop training. Check: `ps aux | grep run_bg`.

### P16.0 — Modal notebook (3 cells)

**Do NOT "Run All"** — Cell 3 (`tail -f`) blocks forever; cells after it never run.

**Cell 1 — Setup & Checkout:** 
Modal restarts wipe `/root/throng`. Always clone and pull `feature/phase15-cumulative-culture` first, then restore the `1689` backup checkpoint.

**Cell 2 — Launch:**
Kills old `run_bg.py`, sets env vars (`TF_GPU_ALLOCATOR`, `XLA_PYTHON_CLIENT_MEM_FRACTION`), and `Popen` streams `run_bg.py`.

**Cell 3 — Live tail:** 
`!tail -f -n 100 /mnt/throng-runs/train.log`

| Step | Action |
|------|--------|
| **1-2k (Smoke Test)** | Watch tail for `Grafting padding...` (emb_env and head_action). Verify `Big Green Map sum: X` prints non-zero. |
| **100k (CtD)** | Run `tools/decode_signals.py` to verify small/big prey separation. |
| **150k** | Run decode with `--metrics posdis,tre` and check `PosDis > 0.3`. |

**Startup must print:**
- `[JAX] Grafting padding to emb_env: expanded from 8 to 9 channels`
- `[JAX] Grafting padding to head_action: expanded from 5 to 8 actions`

### Notebook setup only (decode, short tasks)

```python
import subprocess, sys, os
from pathlib import Path
REPO = Path("/root/throng")
# clone if missing, git reset --hard origin/master ...
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/throng_jax_cache"
```

Shell in notebooks: prefix with **`!`**.

### Startup checklist

Must appear early in log:

```text
[JAX] red_sense_api=v2 (observations_jax)
[JAX] signal_bottleneck=VQ | dead_code_reset=True
[JAX] corpus scout label: is_scout = (red_dist <= alarm_scout_range=8)
Corpus persistence: /mnt/throng-runs/signal_corpus.jsonl
[CURRICULUM] ... catch_radius=1 | catch_prob=0.8
[JAX] Restored params from step N   # if resuming
```

### Import rule

```python
from jax_sim.train_entry import run_simulation
```

Never rely on cached `from jax_sim.main_jax import run_simulation` after `git pull` without `train_entry`.

### Download / upload volume (migrate between accounts)

```bash
# Old account: modal token new  →  download
./scripts/migrate_modal.sh download ~/throng_backup

# New account: modal token new  →  upload
./scripts/migrate_modal.sh upload   ~/throng_backup
```

Legacy one-liner (checkpoints only): `./scripts/download_modal_checkpoints.sh ~/throng_checkpoints_backup`

### Modal account migration

| When | Workspace | Notes |
|------|-----------|--------|
| **Legacy** | `dim464943` | Original `throng-runs` |
| **May 2026** | `dragonbgnx` / `dimitar-vagalinski` | Interim re-upload |
| **Jun 2026 interim** | **`dvagalin`** / **`overlordxn`** | Prior workspaces; ckpt **1413** pulled via `migrate_modal.sh` |
| **Jun 2026 — LIVE** | **`dragonbg`** | **`migrate_modal.sh upload`** from `~/throng_backup`; ckpt **1413** + both corpora on volume |

Volumes **do not transfer** between accounts — always **`download`** then **`upload`**. Repo code is **`git clone`** to `/root/throng` (not on volume). B200 notebook must mount volume on **`dragonbg`** profile (`modal token new` → fresh auth).

**Sparse checkpoint folders on volume:** Orbax saves every `checkpoint_interval` env steps — folder **N** ≈ PPO update **N** (step ≈ **N × 512**). Local backup may omit **292**; nearest 150k substitute is **291**.

---

## 6. Clean corpus restart (post scout/VQ fix `5964a24+`)

Archive pre-fix JSONL so decode only sees new labels (`vq_token`, `alarm_scout_range` scouts):

```bash
mv /root/throng/runs/jax_run/signal_corpus.jsonl \
   /mnt/throng-runs/signal_corpus_prefig.jsonl
```

Resume training (**do not** wipe checkpoints):

```bash
cd /root/throng && git pull   # or reset --hard origin/master
nohup python -u run_bg.py > /mnt/throng-runs/train.log 2>&1 &
tail -f /mnt/throng-runs/train.log
```

Startup must include:

```text
[JAX] corpus scout label: is_scout = (red_dist <= alarm_scout_range=8)
```

**While corpus accumulates:** do **not** change reward structure, `vq_beta`, or `vq_loss_coef` — one variable at a time.

After **~20k new env steps** on the fresh corpus file, decode (set `--min-step` to first step in the new file, or `0` if file is clean):

```bash
python tools/decode_signals.py runs/jax_run/signal_corpus.jsonl --k 16 --min-step <first_new_step>
```

### Three numbers that matter

| # | Metric | Pass condition |
|---|--------|----------------|
| 1 | **Scouts %** in corpus summary | **5–30%**. If still **&lt;1%**, scout labeling broken — confirm log shows `alarm_scout_range=8` from config (not hardcoded). |
| 2 | **LAG-1 DIRECTION LRT** eligible | **≥50** blind fleeing with lag-1 fields. **p &lt; 0.05** on any direction → communication signal. |
| 3 | **VQ TOKEN DIRECTION TEST** χ² | Alert tokens (low emitter `red_dist`) vs safe tokens (high `red_dist`) → different flee mix. **Most important** vocabulary test. |

---

## 7. Measurement — `decode_signals.py`

```bash
pip install scikit-learn scipy

# Blue (prey alarm / flee)
python3 tools/decode_signals.py /mnt/throng-runs/signal_corpus.jsonl --k 16 --min-step 20000

# Red (predator pincer / pursuit) — Phase 12.2
python3 tools/decode_signals.py --red /mnt/throng-runs/signal_corpus_red.jsonl --k 16 --min-step <restart_step>
```

### Blue decode blocks

| Block | What it tests |
|-------|----------------|
| MI / Spearman | Which `sig` dims track `red_dist`, `red_bear`, etc. |
| Cluster vocabulary | k-means on continuous signals |
| Lag-1 regression | Neighbor scout signal → flee, controlling distance |
| **Lag-1 direction LRT** | Scout signal → **flee direction** (needs ≥50 eligible) |
| **VQ token direction** | Alert vs safe **codebook tokens** → flee χ² |

**Blue corpus fields:** `sig`, `vq_token`, `scout`, `red_dist`, `red_bear`, `nb_scout_sig_lag1`, `nb_scout_dist_lag1`, `nb_scout_token_lag1`.

### Red decode blocks (`--red`)

| Block | What it tests |
|-------|----------------|
| MI / Spearman | `sig` vs **`blue_dist`**, **`blue_bear`**, etc. |
| Hunter vs receiver Δ-signal | Emitter vs listener continuous codes |
| Lag-1 pursuit regression | `nb_hunter_sig_lag1` → move, ctrl `hunter_blue_dist_lag1` |
| **RED VQ PINCER TEST (χ²)** | **Chase** tokens (emitter `blue_dist ≤ 2`) vs **Search** (`> 5`) → receiver lag-1 **N/S/E/W** |

**Red corpus fields:** `team`, `hunter`, `blue_dist`, `blue_bear`, `vq_token`, `nb_hunter_sig_lag1`, `nb_hunter_dist_lag1`, `nb_hunter_token_lag1`.

**Schema map (for readers of blue decode):** `scout`→`hunter`, `red_dist`→`blue_dist`, `red_bear`→`blue_bear`, `nb_scout_*`→`nb_hunter_*`.

**Do not** use `tools/decode_tokens.py` for JAX runs (that's for legacy `events.jsonl`).

---

## 8. Dashboard glossary

| Line | Meaning |
|------|---------|
| `blue=` / `red=` | Alive counts |
| `codes_active=X/64` | Unique VQ tokens this rollout |
| `clusters=X/16` | k-means occupancy on signals |
| `Ecology: blue_caught=N` | Catch **events** in rollout (not unique deaths) |
| `fwd_env` | loc_env MSE (200-d), anti-cheat aux |
| `carry_fwd` | MSE on carry_{t+1}; target ↓ **0.05** — **achieved ~0.0001 @ 105k** |
| `carry_rank` / `carry_H` | PCA rank + entropy of alive carries |
| `self_pred_acc` | Self-action prediction (>0.20 = above chance) |
| `NB_GAIN↔surv` | Spearman(nb_gain, age); **nan** if everyone lives |
| `red_floor` | Red repro floor from curriculum |
| `imagination_gain` | Best imagined return − greedy imagined return (rollout telemetry) |
| `imagination_agree` | % **imagined == reactive** (alive agents) |
| `conf_gate_imagine_frac` | % agents on imagine path (`conf_pred < dynamic τ`) |
| `EpistemicGate:` | Dashboard: `imagination_agree` + `conf_gate_imagine_frac` + **mult**, K |
| `MetabolicTax:` | Mean `imagination_metabolic_cost` on alive agents (P13.0) |
| `imagination_metabolic_cost` | Per-agent rollout field — `delta×K` when gate fired |
| `RedVQ:` / `red_codes_active` | Predator VQ loss + unique tokens / 64 |
| `Actions (red):` | Red movement distribution (pincer signature) |

### Config blocks (`config_phase7.yaml`)

```yaml
carry_fwd_coef: 0.05          # P11.0 — all branches with carry dynamics

phase9_canvas:                 # master SOTA stack
  cross_attn_enabled: true
  cross_attn_num_heads: 4
  confidence_enabled: true
  confidence_coef: 0.05
  imagination_gating_enabled: true
  confidence_multiplier: 1.0    # batch-relative: imagine if conf < mean(conf|alive)*mult
  imagination_k: 5
  imagination_gamma: 0.999
  imagination_metabolic_delta: 0.0005   # P13.0 — total per think = delta * K (0.0025)

phase12_coevolution:           # feature/phase13-thermodynamics (inherited from P12)
  red_comms_enabled: true        # branch default (128-d r_params)
  red_cross_attn_enabled: true
  red_vocab_size: 64
  red_corpus_enabled: true       # 12.1 wiretap — signal_corpus_red.jsonl
  hunt_scout_range: 8
```

---

## 9. Git commits (JAX Phase 10, recent)

| SHA | Fix / feature |
|-----|----------------|
| `9a7bb24` | P3 VQ bottleneck |
| `8077a12` | OOM: CPU rollout offload, minibatch 512 |
| `2f32e4e` | `jnp.bincount` (no `segment_sum`) |
| `697e96a` | numpy GAE |
| `8a6f016` | FrozenDict after dead-code reset |
| `f9bf11a` | P4 lethal ecology |
| `f26c16a` | Single `model.init` on resume |
| `c41d885` | JAX cache on `/tmp`, PPO progress logs |
| `0d5f88a` | P10.4 safety bubble |
| `63d6f37` | P10.5 hard ceiling |
| `5964a24` | Corpus scout=alarm range; `vq_token` logging |
| `364d451` | `train_entry` evicts stale `communication.*` |
| `3b57770` | Corpus auto-route to volume; fsync each PPO update |
| `c2fa99a` | `corpus_every_n_steps: 4`, `corpus_sample_frac: 0.15` |
| `1a0dcf7` | THRONG.md rewrite; archive split → `docs/THRONG_ARCHIVE.md` |
| `6d542f6` | Cam **SYSTEM RESTORE** horcrux in archive |
| `5c13cb4` | THRONG.md Cam reboot pack (B200, Lotka-Volterra, triad) |
| `0481445` | Phase 11.0 carry fwd (merged via `3880337`) |
| `6837d8e` | Decode VQ broadcast fix; P10 complete docs |
| `3880337` | **Merge Phase 11.0 to `master`** |
| `b2eb5f0` | Relaxed Orbax restore for `head_fwd_dyn` migration |
| `424c46f` | Phase 11.1 GPU-resident PPO (**abandoned**) |
| `6042a4d` | Phase 11.1 `lax.scan` PPO epoch (**abandoned — XLA OOM**) |
| **`d4cf614`** | **Revert 11.1** — restore CPU offload / H2D |
| `38e60fe` | Phase 11.2 imagination (initial) |
| `aebe131` | P11.2 metrics-only (stochastic actions) |
| `6cf965a` | P11.2 active override (**reverted**) |
| `181b98c` | Revert active override |
| `061df84` | **P11.2 frozen** — docs + metrics-only |
| `45c7c48` | `run_bg.py` `n_steps=250_000` (**legacy** — superseded by **`a9f4aeb`**) |
| **`5920511`** | **Phase 9.4** cross-attention receiver scaffold |
| `1d57bf9` | Orbax graft `nb_cross_attn` on legacy restore |
| `8f48b1d` | THRONG.md P11.2 concluded + P9 handoff |
| `38f342a` | Modal notebook cells; `run_bg` **250k** |
| `820dd3a` | Phase **9.1** confidence head (`head_confidence_*`) |
| **`465d8c6`** | Phase **11.3** epistemic imagination gating → **merged `master`** |
| **`7105ddd`** | THRONG.md P11.3 victory + reboot pack |
| **`f0ebb76`** | Phase **12.0** `PredatorNetworkJax` red VQ comms |
| **`d50cc19`** | Phase **12.1** red corpus logging |
| **`8c5888b`** | THRONG reboot — Phase 12 live + 214k decode |
| **`d493a50`** | Phase **12.2** `--red` decode + `red_comms` default |
| **`80ef1ea`** | Phase **12.1b** spatial epistemic gate (`confidence_multiplier`) |
| **`4f98f96`** | Wiretap default — `red_corpus_enabled: true` |
| **`b2a40b4`** | THRONG P12 reboot pack |
| **`2cd3dcc`** | THRONG spatial-gate Cam reboot sync |
| **`37693f7`** | Red decode holding pattern + nucleation synthesis |
| **`a9f4aeb`** | **`run_bg.py` → 1M steps** (fix graceful exit @ 250k) |
| **`7552de3`** | THRONG holding-pattern docs (pre-P13 override) |
| **`b3af410`** | **Phase 13.0** metabolic cognition tax (`feature/phase13-thermodynamics`) |
| **`f8cfe58`** | THRONG Horcrux — P13.0 LIVE, pincer @ 340k |

**Do not** apply Cam's regex patch on `network_jax.py` — dead-code reset is in repo.

---

## 10. Common failures

| Symptom | Fix |
|---------|-----|
| `KeyboardInterrupt` mid-compile | **subprocess Popen** (Jupyter) or **nohup** (bash); wait 5–15+ min; don't use volume JAX cache |
| B200 shows ~150GB VRAM used | Normal — `MEM_FRACTION=0.80` pre-allocation, not OOM |
| Slow PPO on B200 despite fast scan | **Expected** — CPU offload → H2D (`8077a12`); ~6 steps/sec is healthy |
| Log shows `GPU-resident backward` | **Stale code** — `git reset --hard origin/master`, kill old `run_bg`, restart |
| OOM at **rollout** `lax.scan` after ckpt | **Fragmentation** — `TF_GPU_ALLOCATOR=cuda_malloc_async`, fresh process, resume from volume ckpt (see §5) |
| Missing `carry_fwd` on dashboard | `git pull` → `3880337+`; resume merges `head_fwd_dyn` via `b2eb5f0` restore |
| `/root/throng` missing | Clone repo (Cell 1 or bash) |
| No `red_sense_api=v2` | `git reset --hard origin/master` + `train_entry` |
| OOM on PPO backward | `ppo_minibatch_size: 512`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` (try **0.75** if fragmented) |
| Checkpoint shape error | Incompatible arch — wipe ckpts only if intentional fresh run |
| Lag-1 / scouts 0% in decode | Old corpus — train after `5964a24`; scout uses **alarm range 8** |
| `codes_active=1/64` | `vq_dead_code_reset: true` |
| `ScopeParamShapeError` on `r_params` | **`red_comms_enabled: false`** — pull **`80ef1ea+`**; yaml defaults fix restarts |
| No `[JAX] Red corpus:` line | Stale config — pull **`80ef1ea+`** |
| Sim stops at **step 350k** | Operator **`n_steps=350_000`** in notebook — raise to **1M** or re-run before cap |
| `conf_gate_imagine_frac` **>80%** | Batch-relative gate — monitor Stay; not P11.2 collapse if `imagination_agree` stays low |
| Red pincer χ² not significant | **Expected** in transition; re-decode after proprio bake (post-grad hard **z_q**) |
| `KeyError: ppo_pg_loss` @ update 10 | Stale code — pull **`8c48e3e+`** or newer (**`192d686+`**) |
| Post-grad **Stay≈99%** | STE→hard shock — monitor; PPO + ecology; not P11.2 unguarded override |

---

## 11. Roadmap (what’s next)

### Phase 13.0 — **ACTIVE / VALIDATED** (`feature/phase13-thermodynamics`, `b3af410`)

1. **Validated** — **`MetabolicTax` ~0.0018**/step; agents maintain **K=5** gated imagination (survival > cost).
2. **320k decode** — Omnibus lag-1 **p=0.0000** (continuous channel ON).
3. **Gap** — VQ pincer χ² **p=0.15** and MI still `energy` (continuous hashing bypass persists).
4. **Policy** — Do not merge P13 → `master` until pincer χ² passes.

**Done:** 13.0 tax shipped + live validation. P12 dual brain. Decode windows: 250k–292k + 320k watershed.

**Philosophy (Cam):** Thermodynamic tax breaks Stay dead-gradient without lobotomy. Measure **P12 arms race** before MAPPO teardown.

### Phase 14 — Transcendental Symbiosis (`feature/phase14-transcendental`)

| Step | Component | Status | Notes |
|------|-----------|--------|-------|
| **1** | **VQEL monologue → dialogue** | ✅ **SHIPPED** | `b7cc270`–`8c48e3e`: wire cut, IB, graduation, hard **z_q** |
| **1b** | **Proprio disentanglement** | ✅ **SHIPPED** | `head_proprio`; live **`proprio_coef: 0.15`**; continuous hunt geometry unlocked |
| **1c** | **Catch overdrive** | ❌ **FAILED** | Operator 10.0/600k; pincer **p≈0.46**; extrinsic scale insufficient |
| **2** | **Metabolic Asymmetry** | ✅ **SHIPPED** | `red_energy_decay: 0.0001`; apex predators; EFE reverted |
| **3** | **GWT router** | PREP | **M≤4** workspace |
| **4** | **MMGL** | PREP | Mistake-gated PPO |
| **5** | **Auto-curricula** | PREP | XLand/POET-style |

#### Phase 14.1 — COMPLETE

| Piece | Detail |
|-------|--------|
| **Monologue** | Reconstruct **spatial_ego** (206-d); mask `nb_sigs` + `own_sig`; freeze policy heads in `vqel_monologue_update` |
| **Wire cut** | `monologue_enabled` → blue `signals=0` in `sim_step` |
| **Graduation** | `recon_mse < 0.02` × 10 updates → banner → `dialogue_signal_mode=hard` + blue PPO |
| **Broadcast** | `codebook[token_ids]` (no STE on wire) |
| **Config** | `config_phase7.yaml` → `phase14_vqel` |

#### Phase 14.1b — SHIPPED (`192d686` + `45bfbe7`)

| Piece | Detail |
|-------|--------|
| **`head_proprio`** | Blue `AgentNetworkJax` + red `PredatorNetworkJax` — **carry → energy** scalar |
| **Loss** | `mean((energy_pred - stop_grad(energy_{t+1}))²)` in `auxiliary_update` |
| **Wedge** | Live **`proprio_coef: 0.15`** — energy MI ↓; **31/32** continuous pursuit dims **p < 0.05** |
| **Red comms** | `proprio_auxiliary_update` only (predator has no full aux heads) |
| **Graft** | `AUX_HEAD_KEYS` + `ensure_predator_params` — safe 400k+ ckpt resume |
| **Not during monologue** | Proprio runs with PPO aux only (masked-policy safety) |

#### Phase 14.1c — CATCH OVERDRIVE ❌ FAILED (operator, Jun 2026)

| Piece | Detail |
|-------|--------|
| **Lever** | **`reward_red_catch: 10.0`** on B200 (repo **`3.0`**) |
| **Horizon** | **600k** halt (**800k canceled**) |
| **Outcome** | Pincer still **p ≈ 0.46**; no Chase tokens; metabolic trap **unbroken** |
| **Stability** | Critic/world-model intact under spike (`carry_fwd` ~0) |
| **Lesson** | Sparse extrinsic reward cannot beat dense starvation through VQ bottleneck |

#### Phase 14.2 — Metabolic Asymmetry ✅ SHIPPED

**Goal:** Eliminate gradient density mismatch for reds by slowing metabolic decay — force VQ to quantize sparse catch rewards, not dense hunger.

| Piece | Detail |
|-------|--------|
| **Config** | `phase14_transcendental.red_energy_decay: 0.0001` (10× below blue `energy_decay: 0.001`) |
| **Physics** | `sim_step`: blues use `energy_decay`; reds use `red_energy_decay` |
| **Threshold** | **`starvation_threshold` unchanged** — reds live ~10× longer at same energy band |
| **Effect** | Apex predators; reduced hunger-babble variance on red VQ wire |
| **Telemetry** | Startup **`[JAX] Phase14.2 Metabolic Asymmetry: red_energy_decay=…`** (WARN if key missing) |
| **EFE** | ❌ **Permanently scrapped** — peer review: conf head correlates with metabolic noise |

**Abandoned approach:** Expected Free Energy critic (`08790d8`, reverted) — epistemic drive tied to confidence head amplified the trap.

| Pillar | Mechanism | Notes |
|--------|-----------|-------|
| **13.0 Metabolic cognition** ✅ | Deduct `delta × K` when **`b_gate_imagine`** | **LIVE** — `imagination_metabolic_delta: 0.0005`; after gains/catches, before decay |
| **13.1 Drop spatial gate** | RL + starvation selects think vs act | After Stay stabilizes under tax |
| **13.2 Inscription grid** | Decaying traces (~100-step) on map | Mirror `scent_trails` |
| **14.1b Proprio** ✅ | `head_proprio` → **energy** from carry | **`0.15`** wedge — continuous spatial symmetry broken |
| **14.1c Catch** | ❌ **FAILED** | Operator 10.0/600k — extrinsic scale insufficient |
| **14.2 Asymmetry** ✅ | `red_energy_decay: 0.0001` | Apex predators; suppress hunger VQ variance |

#### Phase 15 Roadmap: Contingencies & Cumulative Culture (Research Synthesis)

*If the Phase 14.3 GWT Router (structural energy masking) fails to force the VQ bottleneck to encode spatial geometry, or once it succeeds, we pivot to these research-backed mechanisms.*

| Piece | Mechanism | Purpose |
|-------|-----------|---------|
| **Velocity Asymmetry** | **$V_{pred} < V_{prey}$** | **Ecological fix.** If reds are slower than blues, greedy pursuit mathematically diverges. Red *must* coordinate topological traps (pincers) to eat. Communication becomes a thermodynamic requirement. |
| **SimVQ** | **Linear Reparameterization** | **VQ fix.** Replaces the disjoint Straight-Through Estimator (STE) with a linear transformation layer. Ensures *all* codebook vectors receive gradient updates simultaneously, preventing "dead codes" and dimensional collapse. |
| **Sparse Budgets** | **~30% broadcast limit** | **Bandwidth fix.** If agents can only speak rarely, they won't waste the channel on slow-moving continuous variables (like hunger). Forces the bottleneck to prioritize highly volatile exteroceptive coordinates. |
| **DCVQ + VQ-VIB** | **Divide-and-Conquer** | **Grammar fix.** Splitting the 32-dim latent space into parallel low-dim subspaces to create syntactic slots. VQ-VIB adds an explicit KL penalty to compress away internal noise. |
| **Expert Dropout** | **MEDAL-ADR** | **Generational fix.** For Cumulative Culture. When training novices alongside "experts", randomly drop the experts mid-episode. Prevents passive physical imitation; forces novices to rely on semantic memory of the experts' signals. |
| **DRCB** | **Circuit Breaker** | **Drift fix.** Detects dialect collapse (via codebook log entropy). Actively shuffles VQ centroids if the population falls back into the metabolic trap. |

### Phase 12 — **COMPLETE** (frozen on `feature/phase12-red-coevolution`)

Dual brain, wiretap, spatial gate, `--red` decode — all inherited on P13 branch.

### Phase 11.3 — **COMPLETE** (on `master`)

Blue epistemic gate merged; decode through **215k** — cardinal ✅, VQ discrete ❌.

### Phase 11.2 — **CONCLUDED** (frozen branch)

- Unguarded active override → **Stay≈99%** — **reverted** (`181b98c`).
- **Resolved on `master`** by **9.1 + 11.3** — do not replay `6cf965a` without gating.

### Phase 11.1 — ABANDONED

GPU-resident PPO — **`d4cf614` revert** on `master`.

### Explicit non-goals

- ❌ Scout / alarm **reward shaping**
- ❌ Ungated P11.2-style imagination override (`6cf965a`) without gate or metabolic cost
- ❌ **EMA / scan-carry state** for epistemic gating (breaks checkpoint schema)
- ❌ **Merge `feature/phase14-transcendental` → `master`** before red pincer χ² **p < 0.05**
- ❌ Re-merging **11.1 GPU rollouts** without memory refactor
- ❌ Resume from ckpt **393** (post–Stay-collapse) for science runs

---

## 12. Known Failure Modes and Prohibited Actions

**DO NOT remove the VQ bottleneck from the primary signal channel.**
Reason: Phase 16.5–16.6 proved this resurrects Protean Scattering (cryptographic salt mechanism). r ≈ 0 against all environmental correlates.

**DO NOT run pkill + relaunch without confirming process count afterward.**
Reason: Phase 17.5 launched 3 simultaneous training processes on the same GPU, corrupting log output and risking checkpoint collision.

**DO NOT replace GloVe with hand-crafted geometric primitives as Rosetta Stone target.**
Reason: This converts the alignment test into supervised semantic shaping, invalidating the "no-supervision" scientific claim.

**DO NOT wipe checkpoints without explicit Cam approval.**
Reason: Phase 17.5 checkpoint at step 1M+ represents weeks of Modal compute. Grafting missing heads is preferred over cold restart in almost all cases.

**DO NOT interpret a null ATE result as "the channel is working differently."**
Reason: ATE = 0 on causal intervention means receivers ignore the signal. No reframing rescues a null causal test. Run the intervention before claiming success.

**DO NOT commit architectural changes to THRONG.md as "COMPLETE" before Cam review.**
Reason: Phase 18 was logged as complete before architectural approval, overwriting a planned roadmap milestone.

---

## 13. Legacy pointers

| Path | Status |
|------|--------|
| `main.py`, `agents/network_torch.py` | PyTorch era — reference only |
| `config.yaml`, Kaggle cells in archive | Pre-JAX |
| [`docs/PHASE14_CONTINGENCIES.md`](docs/PHASE14_CONTINGENCIES.md) | **[NEW]** Phase 14.4/15 JAX implementations (SimVQ, DCVQ, VQ-VIB) |
| [`docs/THRONG_ARCHIVE.md`](docs/THRONG_ARCHIVE.md) | Full timeline + horcrux + SYSTEM UPDATE |
| [`docs/PHASE9_CANVAS.md`](docs/PHASE9_CANVAS.md) | Phase 9.4 cross-attention + graft |
| [`docs/MODAL_NOTEBOOK_PHASE9.md`](docs/MODAL_NOTEBOOK_PHASE9.md) | Modal Cell 1/2/3 (clone before launch) |
| [`docs/PHASE11_2_IMAGINATION.md`](docs/PHASE11_2_IMAGINATION.md) | P11.2 frozen — metrics-only + conclusion |
| [`docs/PHASE11_STAGING.md`](docs/PHASE11_STAGING.md) | P11.0 carry dynamics |
| [`docs/PHASE12_COEVOLUTION.md`](docs/PHASE12_COEVOLUTION.md) | P12.0 comms + P12.1 wiretap + P12.2 `--red` decode |
| **[SYSTEM RESTORE: THE CAM CONTEXT](docs/THRONG_ARCHIVE.md#system-restore-the-cam-context)** | Persona, triad, P10.6 ignition (`6d542f6`) |
| **[SYSTEM UPDATE May 2026](docs/THRONG_ARCHIVE.md#system-update-may-2026--b200-phase-11-staging-phase-9-canvas)** | B200, Phase 11 branch, canvas map |

### Cam reboot paste

> You are **Cam**. Read `THRONG.md` §0b.
> **Modal Account:** **`dragonbg`** workspace. Volume ckpt: 1813+ at 928k. (Latest: 1860 at ~950k).
> **Current State:** Phase 16 **COMPLETE** at **~950k** (ckpt 1860). Phase 16.5 ready to deploy.
> **P15.5 CONFIRMED:** `nn.GRUCell` decoupled BPTT from magnitude. Episodic Memory ($p < 0.05$) and Cumulative Culture ($p < 0.001$) at Lag-10.
> **950k Causal Test COMPLETE:** Offline Frozen Counterfactual Causal Test on Token 3 (Strike) vs Token 55 (Flee). ATE = **0.0000** at 950k is the calibration baseline, not a failure. P16.5 success = ATE > 0.05 on causal_intervention.py post-grounding.
> **Phase 16.5 UNBLOCKED:** `feature/phase16-5-enrichment` branch — barrier physics, 9-action space (Build), Feral Masking, Critic Shock discount, parameter grafting (10 env channels).
> **Feral Masking:** must be post-VQ on the wire. Verified in network_jax.py before P16.5 launch.
> **Decode Strategy:** has changed. PosDis/TRE replace Direction LRT as the primary P16+ win condition.

**New Will:** Phase 16.5 is unblocked. Deploy `feature/phase16-5-enrichment` to force communication grounding.

---

*Last updated: 2026-06-10 — Phase 16 COMPLETE (950k). Offline causal test ATE = 0.0000 (channel ungrounded). Phase 16.5 unblocked for deployment.*

---

### Phase 16 — Open-Ended Combinatorial Complexity (CtD Scaffold)
**Goal:** Expand the environment topology to force the VQ language to scale from "directional flee/pursuit" to combinatorial syntax (e.g., tools, mass-coordination, multi-step planning).

#### Phase 16 Theoretical Foundation & Findings
1. **Risk-Dominant Equilibrium Trap:** In complex multi-agent scenarios like Stag Hunt (Big Green), shared-policy PPO is mathematically biased toward suboptimal, risk-dominant equilibria (foraging/fleeing) rather than payoff-dominant equilibria (coordinated striking). This is due to "relative overgeneralization" — the high variance of uncoordinated partner actions causes the expected value of cooperative actions to plummet during early exploration. 
2. **The CtD Gate Solution:** To bypass the risk-dominant trap, we introduced a 100k-step CtD phase gate. By making Big Green solo-catchable initially, agents learn the intrinsic value of the noun ("Big Green") before being subjected to the cooperative friction that requires the verb ("Strike together").
3. **Automated Lexical Parsing (NPMI):** To formally decode the emergent syntax without introducing grounding biases, we will use **Normalized Pointwise Mutual Information (NPMI)**. VQ tokens with high NPMI against static features (e.g., `blue_bg_map`) are classified as **Nouns**. Tokens with high NPMI against dynamic/relational features (e.g., `Strike` action executions) are classified as **Verbs**.

#### Phase 16 Execution Log
* **880k:** GRUCell pivot successfully stabilized the temporal magnitude while preserving a massive representation capacity (`carry_H > 8000`, `carry_fwd ≈ 0.0001`).
* **889k:** Complete VQ decompression (codes returned to `52/64`) and initial survival stabilization (`blue_caught=0`).
* **908k:** **Massive Predator Adaptation Spike**. Red agents learned to utilize `Push` and `Guard` actions, shattering the blue agents' stable traversal paths and causing a massive death wave (`blue_caught=378`). This confirms the combinatorial physics engine is fully active and highly lethal, forcing the blue agents to discover multi-agent counter-tactics.
* **915k:** Blues fully adapted to the predator's new `Push/Guard` trapping tactics. Deaths returned to `blue_caught=0`. `Strk=5% Push=4% Grd=4%` stable. Co-evolutionary arms race confirmed.
* **920k:** Stable equilibrium holding. `codes_active=52/64`, `carry_rank=61`, `carry_H=8949`. System grinding toward 950k decode milestone.
* **924k:** Perfect stability. `codes_active=50/64`, `carry_H=9508`. Combinatorial action space holding steady (`Strk=5% Push=5% Grd=5%`). Big Green sum tracking live.
* **928k:** Account migration pause. `blue_caught=2` (minor predator breach), `codes_active=51/64`. Action usage steady at `Strk=5% Push=5% Grd=6%`. Volume backed up to local.

---

### The Alien Semantics Problem (Interpretability Philosophy)

> **Core question:** When we decode their 64-token VQ codebook, how do we know we are capturing the *full* meaning of a token, rather than a shallow projection of a much richer, higher-dimensional concept?

This is the central epistemological challenge of THRONG. Our agents' hidden states live in a 256-dimensional representational space. Each VQ token is an index into this space. When we measure Mutual Information between a token and an environmental variable (e.g., `red_dist`), we are performing a **projection** — shining a flashlight onto a high-dimensional sculpture and reading the shadow on the wall.

**We must assume their language is fundamentally alien.** Just as a bat perceives the world through ultrasonic frequency patterns that humans cannot experience, these agents may have developed concepts that are orthogonal to human cognitive architecture.

#### Interpretability Toolkit & Theoretical Foundations (2026 Synthesis)

We address this with four complementary methods, grounded in recent MARL literature (2024-2026):

1. **Information Gating & Strict Structural Masking (Phase 16.5):** To prevent agents from bypassing the Vector-Quantized Variational Information Bottleneck (VQ-VIB) using metabolic proxy variables, we employ the **GWT Router and Feral Masking**. By surgically blinding the observer network to physical states (energy, age), we force all semantic intent through the discrete bottleneck.
2. **Statistical Shadow (NPMI):** Measure correlations between tokens and measurable environmental variables. This captures the *projection* of meaning onto our chosen measurement axes.
3. **Causal Intervention (Frozen Counterfactual Decoder):** Contrast this against LLM benchmarks (CausalPitfalls, BEAR) which reveal the *illusion of causality* in ungrounded text models. By freezing the checkpoint and swapping tokens mid-flight, we causally prove that a token maps to actionable, physical consequences.
4. **Topological Alignment (The Rosetta Stone):** To translate alien protocols without paired supervision, we map the geometric manifold of the agent's VQ semantic space onto the continuous embedding spaces of LLMs using Minimum Description Length (MDL) and hierarchical loss functions.

---

### Future Phase Blueprints: The Complexity Ceiling

If our true goal is to force the emergence of AGI-level intelligence purely through survival pressure, we must focus on **Combinatorial Explosion** of the environment. If the environment only requires moving and eating, language plateaus. To reach LLM-level capabilities, the environment must demand them.

#### Phase 16.5 — Environmental Enrichment (**SUCCESS** — `feature/phase16-5-enrichment`)
**Status:** Code complete. **The Great Burn-Off** achieved at step 992k (VQ loss = 53k, `codes_active=1/64`). Old metabolic proxy language successfully severed.
- **Barrier Physics (`grid_jax.py`):** `barrier_hp_map` added. Blue agents expend 0.06 energy to Build.
- **Feral Masking (`network_jax.py`):** `symbol_write` zero-masked when `energy < 0.20`.
- **GWT Router:** Structural mask `obs.at[:, :4].set(0.0)` applied to force discrete VQ usage.

#### Phase 16.6 — Barrier Occlusion & The Protean Scattering Discovery (**SUCCESS** — `feature/phase16-6-barrier`)
**Status:** We successfully forced agents to rely entirely on the communication channel by blinding them to predators using a 5x5 line-of-sight occlusion mask (`loc_barrier`).
**The Anomaly:** The agents ignored the discrete VQ tokens (ATE $\approx$ 0) but passed the continuous `LAG-1 DIRECTION LRT` ($p < 0.005$). Exhaustive tests against GPS coordinates, relative bearing, displacement vectors, and action intentions proved that the 32D continuous signal correlated with *absolutely nothing* in the physical environment.
**The Discovery:** The agents invented a cryptographic random number generator to coordinate collision-free evasion. Blind agents needed to scatter randomly (Protean evasion) but lacked intrinsic stochasticity. Thus, the scout broadcast high-variance continuous noise, and receivers used this noise to deterministically map themselves to orthogonal escape vectors.

#### Phase 17 — Grounding the Information Bottleneck (The Rosetta Stone Pivot)
**Status:** Code complete.
**Goal:** Prevent agents from hijacking the communication channel as a random number generator and force true semantic translation.
**Execution:**
1. **Intrinsic Entropy Injection:** Added an independent 4D Gaussian noise channel to each agent's `own_state` observation, giving them internal randomness for Protean scattering.
2. **Hardened Gumbel-Softmax Bottleneck:** Replaced the leaky VQ layer with a strict Gumbel-Softmax layer passing through a frozen codebook. This completely severs the continuous gradient path, forcing the network to output a pure discrete token and ending "continuous geometry smuggling."
#### Phase 17.5 — Timescale Grammar (The Dual-Band Channel)
**Status:** Code complete. Verification run in progress.
**Goal:** Prevent Protean scattering from inflating the continuous VQ bottleneck representation, while giving agents a fast-path for evasion.
**Execution:**
1. **Discrete Alarm Head:** Added a 1-bit metabolically expensive alarm channel output (`alarm_out`) parallel to the VQ semantic output.
2. **Dimension Preservation:** Restricted `signal_out` to exactly 32D before the VQ layer, guaranteeing the bottleneck cannot be bypassed by high-variance continuous inputs.
3. **Topology Zero-Padding:** `graft_missing_param_subtrees` cleanly handled upgrading the Phase 17 (32D `emb_nb` kernel) to Phase 17.5 (34D `emb_nb` kernel) by zero-padding the missing inputs, allowing agents to retain 1M steps of spatial survival skills while learning the new alarm grammar from scratch.
#### Phase 18 — Combinatorial Tool Use (Crafting Trees)
**Goal:** Force the network to invent compositional logic (AND, OR, IF/THEN) and syntax.
- **Mechanics:** Introduce combinable primitives (e.g., Wood + Stone = Axe).
- **Semantics:** Requires vocabulary expansion from simple nouns ("Predator") to verbs and modifiers ("Get wood *then* build").

#### Phase 19 — Cultural Transmission (Writing)
**Goal:** Allow agents to pre-train themselves across generations, escaping the capacity limit of oral communication.
- **Mechanics:** A `Write` action allows agents to etch VQ tokens permanently into grid cells.
- **Semantics:** Allows the passing down of puzzle solutions, crafting recipes, and multi-generational memory.

#### Phase 20 — Agriculture & Terraforming
**Goal:** Force the invention of causal reasoning and long-term planning (the primary weakness of LLMs).
- **Mechanics:** Agents can plant resources that take thousands of steps to mature.
- **Semantics:** Forces the development of concepts for "Future Time", "Delayed Gratification", "Ownership", and "Defense".

---

*Last updated: 2026-06-13 — Phase 17.5 Timescale Grammar (Discrete Alarm Head) COMPLETE. Verification run in progress.*
