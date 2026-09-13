# THRONG Adversarial Codebase Audit — Task 0.5 (Sep 2026)

**Scope:** read-only, per the work order (`docs/WILL_RESTART_SEP2026.md`, Task 0.5). No fixes applied. The
only code written was throwaway verification scripts in a scratch directory outside the repo, used to
empirically prove or disprove gradient-flow and index-arithmetic claims per Cam's instruction not to
"reason about it" from the source alone. Nothing in `jax_sim/`, `tools/`, `tests/`, or `config.yaml` was
modified.

**Method:** five sweeps (A–E) as specified. Sweep A (severance) and Sweep E (invariants) were done directly,
including running the existing regression tests and a purpose-built harness (see Appendix) to get real
gradient norms on CPU, since `jax_sim/network_jax.py` currently cannot even be imported (Finding 1) — the
harness works around that without touching the repo file. Sweeps B (ghost metrics), C (tom_logits/Push-Guard),
and D (config reachability) were delegated to three parallel research passes and their results verified
against source before inclusion here.

**Bottom line for Cam:** the carry-path question is settled — blue is EMA, not GRUCell (Finding 4). Two
findings are more urgent than that, though: the tree currently cannot run at all (Finding 1), and the
"epistemic gate" that the Phase 18.7 SOTA claim leans on is gating on an untrained head (Finding 2). See
"What this changes about the restart plan" at the end.

---

## Findings, ranked by severity

### Finding 1 — `jax_sim/network_jax.py` fails to import at all — CONFIRMED DEFECT (blocking)

**`jax_sim/network_jax.py:44`**
```python
@flax.struct.dataclass
class NetworkOutputs:
```
The module imports `from flax import linen as nn`, `from flax.core import freeze, unfreeze`, and
`from flax.core.frozen_dict import FrozenDict` (`network_jax.py:18-20`) — it never does `import flax`. The
bare name `flax` is therefore undefined at the point the decorator runs, and the module raises
`NameError: name 'flax' is not defined` at **import time**, before any function is even called.

This is not in `HEAD` — `git diff HEAD -- jax_sim/network_jax.py` shows `NetworkOutputs` is exactly the
uncommitted July change the work order already flagged ("There is uncommitted July work in the tree that
nobody has verified... The July tree replaces the positional network-output tuple with a `NetworkOutputs`
flax dataclass"). Because `jax_sim/main_jax.py` imports from `network_jax`, **the entire training pipeline
is currently unable to start.** Verified directly:
```
$ python3 -c "import jax_sim.network_jax"
NameError: name 'flax' is not defined. Did you mean: 'float'?
$ python3 -c "import jax_sim.main_jax"
... (same NameError, via the network_jax import)
```
`jax_sim/rl_jax.py` imports fine standalone — the breakage is isolated to `network_jax.py`, but it's fatal
because everything else depends on it.

**What breaks:** nothing in `jax_sim` can run, be tested, or be verified — including Task 1's own two named
fix sites (`monologue_forward`'s positional indexing, `test_vq_loss_index.py`) and every other finding in
this report that touches runtime behavior. This is a zeroth prerequisite, more fundamental than anything
Task 1 currently describes.

**Proposed fix:** add `import flax` alongside the existing `from flax import linen as nn` at the top of
`jax_sim/network_jax.py`. One line. (Not applied — Task 0.5 is read-only.)

---

### Finding 2 — The "epistemic gate" is gating on an untrained head — CONFIRMED DEFECT (empirically proven)

`config.yaml:218` — `confidence_enabled: false` (comment: *"9.1 — enable after git pull post P9.4 run"*).
`jax_sim/main_jax.py:1019-1020`:
```python
_conf_enabled = bool(_p9.get("confidence_enabled", False))
_conf_coef = float(_p9.get("confidence_coef", 0.05)) if _conf_enabled else 0.0
```
So with the current config, `conf_coef` is forced to `0.0` regardless of `confidence_coef: 0.05` sitting
right below it in the same file (`config.yaml:219`) — that key is currently unreachable.

Meanwhile `config.yaml:221` — `imagination_gating_enabled: true` — **is** on, so `_imagine_fn` is not `None`
(`main_jax.py:260-266`), and every rollout step unconditionally computes and *uses* the confidence
prediction to make a real behavioral choice (`main_jax.py:393-407`):
```python
b_conf_pred = model.apply(..., method=model.predict_carry_fwd_confidence)   # uses head_confidence_1/2
...
dynamic_tau = mean_conf * _conf_multiplier
b_gate_imagine = b_conf_pred < dynamic_tau
b_actions = jnp.where(b_gate_imagine, b_a_imagined, b_actions_reactive)
```
`head_confidence_1`/`head_confidence_2` are unconditionally instantiated in `AgentNetworkJax.setup()`
(`network_jax.py:225-226`) — the head always exists and always produces an output — but with
`confidence_enabled: false` it never receives gradient, so it sits at whatever it was initialized to (or
loaded from checkpoint) forever.

**Proven empirically** (not just read from the code, per Cam's requirement), using the real
`auxiliary_update`/`_aux_minibatch_step` functions from `jax_sim/rl_jax.py` against a real
`AgentNetworkJax` instance on CPU:

| Config passed | `head_confidence_1` kernel Δ after one aux update | `head_confidence_2` kernel Δ |
|---|---|---|
| `conf_coef=0.0` (**current live value**, `confidence_enabled: false`) | `0.0000000000` | `0.0000000000` |
| `conf_coef=0.05` (config's own `confidence_coef`, if `confidence_enabled` were `true`) | `5.1097211838` | `0.3190548122` |

`conf_loss` itself is computed and non-zero (`0.4534` in the test batch) — the loss is live, but its
gradient contribution is multiplied by exactly zero, so the head never moves. The mechanism is structurally
sound (proven by the second row); the *current config value* is what zeroes it. This is precisely the bug
class Cam named: "a coefficient of 0 that nobody intended," except here the consumer (the gate) is fully
live and shapes real rollout behavior every step.

**What breaks:** THRONG.md's headline — *"Blue SOTA... 9.4 cross-attn + 9.1 confidence + 11.3 epistemic
gate"* — is not true of the current config: `cross_attn_enabled: false` too (see note below), and the
confidence head backing "9.1" is untrained. The gate is real (it does switch behavior) but it is not
*epistemic* right now — `b_gate_imagine` is effectively splitting the population by an arbitrary, frozen
function of carry+action, not by calibrated forecast-uncertainty. Every "Stay ~19%" / `conf_gate_imagine_frac`
/ `imagination_agree` number on record for Phase 18.7 configs should be read as "gated by an untrained head,"
not "gated by confidence." (Sweep B independently confirmed `conf_gate_imagine_frac`/`imagination_agree` are
live-computed each update — live computation, meaningless signal.)

**Related, lower severity, same root cause:** `cross_attn_enabled: false` (`config.yaml:216`) means the 9.4
`NeighborCrossAttention` path is also not compiled in the current config (`network_jax.py:322-326` falls to
the `else` branch, per-neighbor tokens instead of attention) — so the "9.4 cross-attn" half of the same
headline claim is also not what's currently running. This one isn't a severed-gradient bug (there's no
half-trained head sitting unused), just a docs/config mismatch worth fixing in the same pass.

**Proposed fix:** either set `confidence_enabled: true` (and verify the graft note in the comment — "grafts
`head_confidence_*`" — still applies cleanly to a checkpoint that already has that head), or, if the gate is
being deliberately run in this degraded mode, stop calling it "epistemic" in dashboard output and docs until
it's trained. Not applied — read-only.

---

### Finding 3 — Corpus NPMI fields (`local_resource`, `adj_bg`, `adj_barrier`) read the wrong observation columns — CONFIRMED DEFECT

**`jax_sim/main_jax.py:2368-2376`:**
```python
env_channels = int(config.get("env_channels", 10))
idx_offset = 6 + (config["neighbor_k"] * config["signal_dim"]) + (25 * config["symbol_dim"])
idx_resource = idx_offset + (12 * env_channels) + 3   # 12th cell (center), channel 3 = resource
adj_cells = [7, 11, 12, 13, 17]
idx_adj_bg = [idx_offset + (c * env_channels) + 8 for c in adj_cells]
idx_adj_barrier = [idx_offset + (c * env_channels) + 9 for c in adj_cells]
```
This hand-rolls the offset to `loc_env` in the flat observation vector instead of using the canonical
`make_obs_layout()` (`jax_sim/obs_layout.py`), which the same file already uses correctly elsewhere
(`main_jax.py:1003-1011`). The hand-rolled formula assumes `own_state_dim = 6` (a pre-Phase-18.7 value) and
**omits the neighbor-alarm block** (`neighbor_k * 2`) entirely — both were added by later phases.

Computed against the current live config (`own_state_dim=22, neighbor_k=6, signal_dim=40, symbol_dim=16,
env_channels=15`):

| | value |
|---|---|
| Correct `loc_env_start` (via `make_obs_layout`) | **674** |
| This code's `idx_offset` | **646** |
| Error | **−28 columns** (16 from `own_state_dim`, 12 from the missing alarm block) |

Concretely, walked through by hand and cross-checked against `observations_jax.py`'s channel order
(`pres_blue, pres_red, wall, resource, shelter, contested, scent, puzzle, blue_bg, barrier, wood, stone,
flint, clay, vine`):

- `local_resource` (intended: **resource** channel, center cell) actually reads the **contested** channel at
  a non-center cell.
- `adj_bg` (intended: **blue_bg** channel at 5 named adjacent cells) actually reads the **wood** channel, each
  shifted by roughly one grid cell from the intended position.
- `adj_barrier` (intended: **barrier** channel, same 5 cells) actually reads the **stone** channel, same
  shift.

These are not out-of-bounds crashes — the shifted indices still land inside the (correct) `loc_env` block by
coincidence of its size, which is exactly why this has run silently. `loc_res = b_obs_all[t, alive_idx,
idx_resource]` (`main_jax.py:2427`) and `adj_bg`/`adj_barrier` (`main_jax.py:2460-2461`) feed directly into
`corpus_writer.maybe_record(local_resource=..., adj_bg=..., adj_barrier=..., ...)` (`main_jax.py:2488,
2496-2497`), i.e. straight into `signal_corpus.jsonl`.

**Downstream impact, confirmed:** `tools/decode_signals.py:1852` uses exactly these three fields —
`noun_features = [adj_bg_v, adj_barrier_v, adj_red_v, resource]` — as candidate semantic referents for the
per-slot NPMI decode (the analysis Task 3 is about to instrument further). Any published or future NPMI
result claiming a VQ slot correlates (or fails to correlate) with "background" or "barrier" proximity, or
with local resource, has been testing against the wrong channel at a shifted cell. Sweep D independently
derived the same 674-vs-646 arithmetic and flagged the same three fields; this is now confirmed from two
independent passes.

**What breaks:** this doesn't affect training or reward — it's an offline-analysis-only corruption — but it
directly poisons the corpus fields Task 3 is planning to build on, and it will silently corrupt the new
post-restart corpus (Task 5) exactly the same way unless fixed first.

**Proposed fix:** replace the hand-rolled `idx_offset`/`idx_resource`/`idx_adj_bg`/`idx_adj_barrier` block
with the already-available `_layout` (`ObsLayout` from `make_obs_layout`, already constructed at
`main_jax.py:1003`) — e.g. `_layout.loc_env_start + cell*env_channels + channel`. Not applied — read-only.

---

### Finding 4 — Blue's carry path is EMA, not GRUCell — CONFIRMED DOC/CODE DRIFT (Cam's priority item, settled)

**`jax_sim/network_jax.py:355`** (inside `AgentNetworkJax.__call__`, the blue network):
```python
new_carries = 0.9 * carries + 0.1 * pooled  # soft update
```
Grepping the full `AgentNetworkJax` class body (`network_jax.py:144`–`543`) for `carry_gru`/`GRUCell` returns
**zero hits**. The only `nn.GRUCell` in the file is `self.carry_gru = nn.GRUCell(features=d, name="carry_gru")`
at `network_jax.py:646`, inside `PredatorNetworkJax` (red), used at `network_jax.py:766`:
`new_carries, _ = self.carry_gru(carries, pooled)`. The checkpoint-grafting code that merges a fresh
`carry_gru` into params on load (`network_jax.py:889-891`, print: *"Merged fresh carry_gru (Phase 15.5
GRUCell upgrade) into predator params"*) is likewise red-only.

**Cam's read was correct: blue never got the GRUCell.** THRONG.md describes it as a project-wide change:
- Phase 15.4 (`THRONG.md:508`): *"`nn.GRUCell` decoupled BPTT from exteroceptive magnitude; Lag-10 Episodic
  Memory ($p<0.05$) and Cumulative Culture ($p<0.001$) confirmed."*
- Phase 15.5 (`THRONG.md:509`): *"GRUCell Pivot — Replaced EMA with parameterized GRU gates; checkpoint
  grafting for `carry_gru`."*

Both entries attribute a project-wide architecture change (and the episodic-memory result) to a mechanism
that only exists on the **red** predator network. Blue — the team the episodic-memory / cumulative-culture
result was actually about (receivers of other blues' signals) — has always run the plain EMA.

**Important nuance:** `docs/ARCHITECTURE.md` already has this right and already flags the exact discrepancy —
it is not a blind spot in the more authoritative doc:
> `docs/ARCHITECTURE.md:58` — *"Blue uses a soft EMA... (Note: `THRONG.md` references a 'GRUCell pivot' —
> that landed on the red network's `carry_gru`, not blue. Verify before relying on blue having gated
> recurrent memory.)"*
> `docs/ARCHITECTURE.md:170` (table) — *"Blue carry | EMA, not GRU | verify before relying on gated memory
> for blue."*

So the drift is specifically: THRONG.md (the primary onboarding doc, read first per its own §0b) was never
corrected after `docs/ARCHITECTURE.md` identified the discrepancy, and the Phase 15.4 result's mechanism
attribution has never been resolved one way or the other.

**What this settles, plainly:** the Lag-10 episodic-memory and cumulative-culture results credited to "the
GRUCell pivot" were produced by blue running a plain EMA the entire time. Whatever mechanism explains those
results, it is not gated recurrence — it's the same soft update blue has always had. This doesn't invalidate
the *results* (the $p$-values are what they are), but it invalidates the *causal story* currently written
into THRONG.md about why they happened. If red's real GRUCell is later assumed to be "the same architecture
blue also benefits from" for any future design decision, that assumption is false today.

**Proposed fix (documentation only, not code):** correct THRONG.md Phase 15.4/15.5 entries to state blue
carry = EMA (matching ARCHITECTURE.md), and either re-attribute the episodic-memory mechanism honestly or
mark it as an open question. Not applied — read-only, and this is a documentation, not a code, change.

---

### Finding 5 — Red's Push(6)/Guard(7) logit-mask is missing at rollout sampling — CONFIRMED DEFECT

Blue is masked at all three sites the invariant requires:
- Rollout: `main_jax.py:379-382` — `b_action_logits.at[:,6].set(-1e9)`, `.at[:,7].set(-1e9)`.
- Imagination: `imagination_jax.py:85-90` — `scores.at[6:8,:].set(-1e9)` when `_n_imag > 7`.
- PPO backward pass: `rl_jax.py:109-114` — `action_logits.at[...,6:8].set(-1e9)` when `shape[-1] > 7`.

Red is masked in the backward pass only. `rl_jax.py`'s masking is team-blind (no `team` argument gates it),
and `ppo_update` is called for both blue (`main_jax.py:1718`) and red (`main_jax.py:1836`), with
`n_actions: 12` applying to both populations (`config.yaml:44`, read identically at `main_jax.py:1043` and
`:1082`) — so the `shape[-1] > 7` condition is true for red too, and red's logits **are** masked during the
gradient step. But at rollout, `r_actions = jax.vmap(jax.random.categorical)(r_action_keys, r_action_logits)`
(`main_jax.py:388`) samples from the raw, unmasked 12-way distribution, and `r_log_probs` at
`main_jax.py:790-791` is computed from that same unmasked distribution. There is no red imagination path at
all, so that third site simply doesn't apply to red (not itself a defect).

**Effect:** red can and will sample Push/Guard during rollout with real (if small) probability. Those are
harmless to the environment (Finding confirmed clean, below — Push/Guard are true no-ops in `grid_jax.py`
for both teams), but the PPO backward pass then re-masks those same two logits before recomputing
`new_log_probs` for the ratio — so for any (agent, timestep) where red actually chose Push or Guard, the
probability ratio computation is comparing an unmasked `old_log_probs` against a masked `new_log_probs`,
and every other action's log-prob in that softmax is shifted by the renormalization (removing two live
probability masses from the denominator changes all the others). This is a silent bias in red's PPO updates,
not a crash — the documented NaN-cascade failure mode (`old=-1e9, new=finite → ratio=inf`) runs in the
opposite direction from this one (`old=finite, new=-1e9 → ratio≈0`), so it's less violent, but it's the same
invariant, half-applied.

This also means THRONG.md's claim (`THRONG.md:1407`, *"Logit-mask `-1e9` on actions 6 (Push) and 7 (Guard) in
rollout, imagination, and PPO loss... `Push=0%, Guard=0%` confirmed ✅"*) is true for blue only. Red's own
dashboard line (`main_jax.py:2016-2026`) tracks red's raw sampled action frequencies and should show nonzero
Push/Guard unless the policy happens to assign them near-zero probability on its own — nothing in the code
guarantees that.

**Proposed fix:** apply the same `.at[:,6:8].set(-1e9)` mask to `r_action_logits` at `main_jax.py:388`
(before sampling) and to whatever produces `r_log_probs` at `:790-791`. Not applied — read-only.

---

### Finding 6 — `NB_GAIN` is not a decayed/rare signal, it is structurally incapable of varying — CONFIRMED (already known as deprecated, but the framing is wrong and it's still live)

`pop.nb_gain` is initialized to `jnp.ones(max_pop)` (`population_jax.py:46`) and the only other write,
`new_nb_gain = jnp.where(activate_mask, 1.0, pop.nb_gain)` (`population_jax.py:306`), resets newly-spawned
slots back to `1.0` and leaves the rest unchanged — which, inductively, was always `1.0`. **No code path
anywhere can make any element of this array anything but `1.0`.** The `nb_gain` kwarg the network methods
accept is never actually passed at any call site in `main_jax.py`, and red's `__call__` explicitly
`del nb_gain`s it — the whole thing is vestigial plumbing, not merely a metric that "usually collapses."

THRONG.md (`THRONG.md:39`) says *"NB_GAIN Deprecated: Variance fully collapsed to `[1.]`. Officially ghosted"*
— which reads as "we know, and we removed it." It wasn't removed: `main_jax.py` still computes a
`spearmanr` call and prints a `[DEBUG] NB_GAIN variance collapsed!` line plus `NB_GAIN↔surv: nan` on the
dashboard every single interval (`main_jax.py:2095-2108, 2194`). Low severity (wasted compute, not wrong
data), but worth actually deleting rather than continuing to carry as inert code that looks live.

---

### Finding 7 — Two reward-shaping config keys have no effect — CONFIRMED (config reachability)

`config.yaml:140` (`reward_starvation: -0.5`) and `config.yaml:165` (`reward_blue_red_proximity: -0.01`) are
never read anywhere in the reward composition (`main_jax.py:736-753`). These read as intentional levers on a
reward function that doesn't have the corresponding terms — worth Cam's attention given Rule 12 (no ecology
edits without sign-off): either the terms were removed from the reward function without removing the config
keys, or they were never wired up. Distinguishing which is a question for Cam, not something resolvable from
source alone — flagged as a **suspicion about intent**, not a confirmed defect, since "the key does nothing"
is confirmed but "was that a mistake" is not.

---

### Finding 8 — `scripts/modal_train.py` maintains a ~22-key shadow copy of `config.yaml` — SYSTEMIC RISK, not currently live

`scripts/modal_train.py:32-61` re-asserts ~22 individual config values on top of the loaded `config.yaml`
(population sizes, PPO hyperparameters, red ecology params, `n_actions`, `env_channels`, etc.). All 22
currently match `config.yaml` exactly — verified line-by-line by the config-reachability pass — so there is
no live drift today. But this is the exact mechanism that produced the historical `env_channels=12` bug
(THRONG.md §0b headline): someone tunes a value in `config.yaml`, runs via `modal_train.py`, and the tuned
value is silently overwritten back to whatever is hardcoded here. The two most recent commits
(`barrier_build_cost` escalations) happened to be safe only because `barrier_build_cost` isn't in this
override list. The next tuning commit to any of the 22 listed keys is not guaranteed the same luck.

**Proposed fix:** have `modal_train.py` read `config.yaml` directly and override only genuinely
Modal-specific settings (e.g. `checkpoint_dir`), rather than re-asserting training hyperparameters. Not
applied — read-only, and this is a design change beyond Task 0.5's scope; flagging for Cam's prioritization.

---

### Finding 9 — `red_hidden_dim` misread in three offline analysis tools — CONFIRMED, dormant

`tools/alarm_correlation.py:91,122,124`, `tools/alarm_lag_analysis.py:108,139,142`, and
`tools/causal_intervention.py:114` all read `config.get("phase14_transcendental", {}).get("hidden_dim", 128)`.
`red_hidden_dim` is actually a **top-level** config key (`config.yaml:110`), and `phase14_transcendental` has
no `hidden_dim` field — so all three always fall through to the hardcoded default `128`, which happens to
equal `config.yaml`'s `red_hidden_dim: 128` today. The correct pattern is used in production
(`main_jax.py:994`) and in `tests/test_checkpoint_compat.py:89`. If `red_hidden_dim` is ever changed, these
three tools will keep building a 128-dim red network and fail (or silently misload) against a
differently-sized checkpoint. Not touched by this audit's other findings; flagged for whoever next resizes
red.

---

### Finding 10 — Three tests are currently broken, independent of Finding 1

Verified by running each against the import-workaround harness (Appendix):

- **`tests/test_gradient_flow.py::test_blue_codebook_gradient_flow`** — hardcodes
  `carries = jnp.zeros((2, 128))` for `AgentNetworkJax()`, whose default `hidden_dim` is **256**. Crashes on
  a shape mismatch inside `__call__` before ever reaching the assertion. (The red test in the same file,
  `test_red_vq_decoupled`, is fine — `PredatorNetworkJax` defaults to `hidden_dim=128`.) Once the shape is
  corrected (`jnp.zeros((2, model.hidden_dim))`), the underlying VQ-codebook gradient claim holds: measured
  codebook gradient norms of `2.42 / 2.07 / 2.09` for slots 0/1/2 — healthy, non-zero.
- **`tests/test_checkpoint_compat.py`** — `ImportError: cannot import name 'default_config' from
  'jax_sim.main_jax'`. The actual name is `DEFAULT_CONFIG` (`main_jax.py:95`). This test cannot even be
  collected.
- **`tests/test_vq_loss_index.py`** — already named in the work order (Task 1, item 2): calls
  `ppo_loss(..., vq_loss_idx=...)`, a keyword argument that no longer exists on `ppo_loss` after the
  `NetworkOutputs` migration. Confirmed still broken; no new information beyond what Task 1 already says.

Also worth noting for coverage: `tests/test_gradient_flow.py` instantiates both networks with **zero
constructor args** (`AgentNetworkJax()`, `PredatorNetworkJax()`), which silently tests the dataclass
*defaults* (`env_channels=9`, `own_state_dim=10` — a pre-Phase-18.7 shape) rather than the shape actually
used in production (`env_channels=15`, `own_state_dim=22`). This test currently cannot catch a regression in
the live observation layout even after its shape bug is fixed.

**`tests/test_vq_gradient_flow.py` passes cleanly** once the import workaround is applied — all four
gradient-wiring assertions (commitment→encoder, VQ-loss→codebook, STE wire→encoder, frozen-broadcast) hold.
This is the strongest piece of good news in this audit: the specific H2 severance class (VQ loss disconnected
from the tape) that bit the project twice before is currently healthy.

---

## Checked and found clean (coverage)

- **VQ gradient flow (the H2 severance class)** — `test_vq_gradient_flow.py` passes; independently
  reproduced codebook gradients (`2.42/2.07/2.09` for slots 0/1/2) via a real forward+backward pass.
- **`tom_logits`** — no loss or reward is attached anywhere in `rl_jax.py` or `main_jax.py`; only ever read
  into an unused local variable. `tom_reward_coef` is a dead config key (present only in the inert
  `DEFAULT_CONFIG` fallback, never read from the live-loaded YAML). `docs/ARCHITECTURE.md` and
  `docs/STRATEGIC_ROADMAP.md` already correctly call this vestigial — no drift to fix there. (A stale
  narrative claim about Push/Guard "trapping tactics" survives uncorrected at `THRONG.md:1311-1312` — a
  doc/doc inconsistency, not doc/code, noted for completeness.)
- **Push(6)/Guard(7) environment mechanics** — confirmed by direct read of `grid_jax.py`: zero dispatch code
  for either action index anywhere in the environment step, population update, or observation builder. They
  are true no-ops for both teams. Docs already say this correctly.
- **Obs-layout internal consistency** — `own_state_dim=22`'s field list in `observations_jax.py`'s `else`
  branch (13 stacked scalars + 5-wide `current_recipe` + 4-wide intrinsic entropy) sums to exactly 22;
  `env_channels=15`'s branch (2 presence + 8 single-channel terrain/social + 5 resource channels) sums to
  exactly 15. Both match `config.yaml`. `make_obs_layout()` is the single canonical source of these
  boundaries and is used consistently by `network_jax.py`'s `_obs_layout()`, `main_jax.py`'s live model
  construction, and `test_checkpoint_compat.py`. (The one place that does *not* use it — the hand-rolled
  `idx_offset` in the corpus-writing section — is Finding 3.)
- **`env_channels` four-way agreement** — `config.yaml` (15), `scripts/modal_train.py` (15, explicit
  override matches), `main_jax.py`'s primary model-construction call sites (default 15), and
  `observations_jax.py` (reads 15, correct branch) all agree today. (Two unrelated call sites deep in
  `main_jax.py` still carry a stale `default=10` fallback that is never hit because the config key is always
  present — latent hygiene issue, not a live bug, noted under Finding 3's fix.)
- **`n_actions=12` reachability** — the historical "defaults to 8" trap object still exists in most call
  sites' fallback values and in one now-dead config key (`phase16_combinatorial_syntax.num_actions: 8`), but
  is currently inert because `config.yaml:44` sets `n_actions: 12` at top level and every production call
  site reads it.
- **Aux-loss coefficients other than confidence** — `fwd_coef`, `carry_fwd_coef`, `self_pred_coef`,
  `proprio_coef` (blue and red), `retention_coef` are all nonzero in the live config (or nonzero function
  defaults where the key is absent) and correctly threaded through to `auxiliary_update`/
  `red_auxiliary_update`. `red_vq_loss_coef: 0.0` is intentional and documented (Phase 18.6 decoupling) — not
  a severance bug, a deliberate diagnostic-only mode.
- **Dashboard metrics** — the large majority (rewards, values, entropy, VF loss, clip fraction, VQ loss,
  codes_active, ecology counters, crafting stats, epistemic-gate fractions, metabolic tax, brain-vote
  history) are genuinely recomputed from the current rollout each update. Only `NB_GAIN↔surv` (Finding 6) is
  a true ghost; `ret_mean` is a harmless always-zero orphan (`main_jax.py:2052`, key never written by
  `ppo_update`'s metrics dict).

---

## Suspicions (not confirmed — flagged for judgment, not fact)

- **`reward_starvation` / `reward_blue_red_proximity` being dead** (Finding 7) — confirmed dead, but whether
  that's a bug or an intentional removal is not resolvable from source.
- **`RedVQ` loss magnitude** — confirmed live-computed each update, but whether its printed value shows real
  update-to-update variance versus saturating near a numerical ceiling (given it can run `≈1e6–1e11`
  decoupled) was not verified at runtime — would need a real multi-update log sample, not static reading.
- **`sp_acc` gap for red** — `red_auxiliary_update` never computes a self-prediction accuracy the way blue's
  `auxiliary_update` does; not currently printed, so not an active ghost, but a latent gap if a red
  `self_pred_acc` display is ever added expecting that key to exist.

---

## What this changes about the restart plan

1. **Nothing in `jax_sim/` can be run, tested, or verified until Finding 1 is fixed.** This has to happen
   before Task 1 is attempted, not as part of it — Task 1's two named sites (`monologue_forward`,
   `test_vq_loss_index.py`) can't even be reached with the current import error.
2. **The Phase 18.7 "epistemic gate" claim needs a decision, not just a docs edit** (Finding 2): either turn
   `confidence_enabled: true` on before trusting any Stay/imagination-rate number from a Phase 18.7 run, or
   explicitly reclassify the gate as non-epistemic in current docs until it is.
3. **The carry-path question is answered** (Finding 4, as requested first): blue is EMA. The Phase 15.4
   episodic-memory / cumulative-culture results were produced by that EMA, not by a GRUCell. This doesn't
   invalidate the results, but the causal story attached to them in THRONG.md is wrong and should be fixed
   before it's cited again, and before assuming blue would get "the same benefit" from any future GRUCell
   discussion.
4. **Fix Finding 3 before the next corpus-collection run** (Task 5) — otherwise the new post-restart corpus
   carries the same three corrupted NPMI fields forward, and Task 3's planned per-slot NPMI instrumentation
   would be built on top of already-wrong data for background/barrier/resource correlates.
5. Findings 5, 6, 7, 8, 9, 10 are real but lower-stakes — none of them block Tasks 1–4, but 5 (red PPO bias)
   and 8 (the modal_train.py shadow-config pattern) are worth fixing opportunistically given they're exactly
   the bug class this audit was commissioned to find.

---

## Addendum (post-sign-off) — Sweep A completed, Findings 2 & 3 resolved, `cross_attn_enabled` history

Cam signed off on the initial audit and asked for three things before starting the fix order: finish
Sweep A properly (every loss term, measured, blue and red, at production shapes, landed as a permanent
test), determine whether Finding 2's untrained confidence head is actually a *fossil* (trained once, then
frozen), and determine whether Finding 3's corruption reaches the Phase 17.5 "alarm is cheap talk" scan.
Separately, Cam asked for `git log -p` on `phase9_canvas.cross_attn_enabled` before finalizing the GWT
scorecard question. All four are resolved below. **No fixes were applied in this pass either** — this is
still pre-1.0 investigative work; the numbered fix order (1.0–1.6, then Tasks 2–4, then 5) has not been
started.

### Sweep A, completed — every loss term, measured, blue and red, at production shapes

Landed permanently at [`tests/test_severance_sweep.py`](../tests/test_severance_sweep.py), which generalizes
`test_vq_gradient_flow.py` per the work order. Critically, it loads shapes from `config.yaml` at import time
(via `DEFAULT_CONFIG` + `_normalize_config`) rather than hardcoding numbers or using dataclass defaults —
the exact anti-pattern Cam flagged in the "ghost test" complaint about `test_gradient_flow.py`/
`test_checkpoint_compat.py`. All 10 checks currently pass, verified via the Finding-1 import-workaround
harness (Appendix); they will run directly once the one-line `import flax` fix lands.

Two assertion classes, kept deliberately separate: **structural** (does a raw, unweighted gradient reach
the term's designated parameters at all — this must always hold, independent of config, and failing it is
a severance bug like Phase 18.4's) and **effective** (does the coefficient-weighted, optimizer-applied delta
match what `config.yaml`'s own coefficient currently says it should be — this reads the live config so it
self-updates rather than pinning today's numbers).

| Team | Term | Coefficient (live config) | Target | Raw grad norm | Effective Δ | Verdict |
|---|---|---|---|---|---|---|
| blue | PPO policy (pg) | 1.0 (intrinsic) | `head_action` + trunk | 0.0437 | 0.124 | live |
| blue | PPO value (vf) | `ppo_value_coef`=0.1 | `head_value` | 1.474 | 0.147 | live |
| blue | PPO entropy (spatial+alarm) | ent=0.02, alarm=0.0 | `head_action`, `head_alarm` | 0.888 | 2.571 | live (alarm_ent_coef=0 is intentional — alarm is documented vestigial) |
| blue | PPO logit L2 penalty | 0.01 (baked in) | `head_action` | 0.0106 | 0.0106 | live |
| blue | VQ commitment, slot 0/1/2 | `vq_loss_coef`=0.1 | `codebook_0/1/2` | 0.664 / 0.720 / 0.660 | 0.066 / 0.072 / 0.066 | live |
| blue | VQ commitment → signal heads | `vq_loss_coef`=0.1 | `head_signal_slot0/1/2` | 0.191 / 0.212 / 0.202 | 0.019 / 0.021 / 0.020 | live |
| blue | fwd_env (forward dynamics) | `fwd_coef`=0.05 (config-absent, function default) | `head_fwd_1/2` | — | 4.77 / 5.91 (optimizer-step Δ) | live |
| blue | carry_fwd (latent dynamics) | `carry_fwd_coef`=0.05 | `head_fwd_dyn_1/2` | — | 4.82 / 4.95 | live |
| blue | self_pred | `self_pred_coef`=0.1 (config-absent, function default) | `head_self_pred` | — | 0.555 | live |
| blue | proprio | `proprio_coef`=0.15 | `head_proprio` | — | 0.160 | live |
| blue | **confidence** | `confidence_enabled`=false → coef forced 0.0 | `head_confidence_1/2` | — | **0.0 / 0.0** | **effectively dead — Finding 2, now proven at production scale too** (contrast: 5.15 if `conf_coef=0.05`, the value already sitting in `confidence_coef`) |
| blue aux trunk leak check | — | — | everything outside the 8 aux heads | — | **0.0** | clean — `auxiliary_heads` takes `carry_t` as data, no trunk gradient, as designed |
| red | PPO policy (pg) | 1.0 | `head_action` + trunk | 0.0469 | 0.142 | live |
| red | PPO value (vf) | `ppo_value_coef`=0.1 (same key, both teams) | `head_value` | 0.714 | 0.0714 | live |
| red | PPO entropy (spatial; no alarm head) | `ppo_entropy_coef`=0.02 | `head_action` | 0.159 | 0.460 | live |
| red | PPO logit L2 penalty | 0.01 (baked in) | `head_action` | 0.0046 | 0.0046 | live |
| red | VQ (dcvq) | `red_vq_loss_coef`=0.0 | `dcvq` | **1.040 (raw, mechanism intact)** | **0.0 (effective)** | intentional Phase 18.6 decoupling, confirmed working exactly as documented — not a severance bug |
| red | proprio | `proprio_coef`=0.15 (same key as blue) | `head_proprio` | — | 0.114 | live |
| red | retention (SRL) | `retention_coef`=0.1 | `head_retention` | — | 1.760 | live |
| red aux trunk | — | — | everything else | — | **9.995** | **not a leak — intentional.** Unlike blue, `main_jax.py`'s `_red_aux_apply_fn` scans the *full* red network forward over `lag` obs steps to build `final_carry` before reading the aux heads off it (real BPTT — the documented "SRL BPTT," Phase 15.3/15.4). Gradient legitimately reaches the whole red trunk here; this is a genuine, previously-undocumented-in-the-audit architectural asymmetry between blue's (heads-only) and red's (full-BPTT) aux update, now pinned by the permanent test. |

**Net result: no new severance bugs found in the full sweep.** Every term reaches its intended parameters
structurally. The one term that's effectively dead (blue confidence) was already Finding 2, and it's a
config-coefficient issue, not a broken gradient path — proven identically at both toy and production scale
now. Red's `vq_coef=0.0` is confirmed exactly as intentional as documented. The blue/red aux-update asymmetry
(heads-only vs. full-BPTT) is real, intentional per the docs, and now has permanent test coverage either way.

### Finding 2, resolved: it's a fossil, not merely "untrained"

Restored `head_confidence_1`/`head_confidence_2` directly from `mnt/checkpoints/2244` (local, real Orbax
data — Task 0's backup effort turned out to already have what was needed) and compared against a freshly
initialized `AgentNetworkJax` at matching shape (`hidden_dim=256`, `n_actions=8`, matching this checkpoint's
Phase 17.5 architecture):

| | fresh init (seed 0) | ckpt 2244 |
|---|---|---|
| `head_confidence_1` bias | **exactly zero** (Flax `nn.Dense` default `bias_init=zeros`) | norm 1.119, mean 0.0127, std 0.0326 — **not zero** |
| `head_confidence_2` bias | `[0.]` | `[-0.0173]` — **not zero** |
| `head_confidence_1` kernel | norm 31.96 (≈ lecun_normal theoretical std 0.0615) | norm 33.87, std 0.0651 — statistically consistent with *an* independent random-scale draw, not diagnostic either way |

The kernel alone wouldn't settle it (a lightly-trained kernel and a fresh one can look similar in
norm/scale). The bias is the tell: Flax's `nn.Dense` default is `bias_init=zeros`, confirmed directly from a
fresh `model.init()` above, and nothing in the checkpoint-grafting code ever resets an *existing* parameter
to a new value — `pad_auxiliary_heads` (`network_jax.py:1256-1289`, the function that touches
`head_confidence_1` when `n_actions` changes) only zero-*pads* new rows/columns on a shape change; it never
touches the bias or overwrites existing weights. A non-zero bias could only have arrived through real
gradient descent, which requires `conf_coef` to have been non-zero at some point in this checkpoint's
training lineage.

**Cam's hypothesis is confirmed: this is a fossil.** `head_confidence_1/2` was trained during some earlier
session (plausibly the actual Phase 9.1 validation, when `confidence_enabled` was presumably briefly true),
then carried frozen through every subsequent resume and graft — including through Phase 18's `n_actions`
8→12 expansion, which only pads, never resets — all while `confidence_enabled: false` sat in every config
since. Every historical "epistemic gate" number on record (Stay ~19%, `conf_gate_imagine_frac`, etc.) was
produced by a head trained once, for a carry-forward-error target on an architecture (`n_actions=8`,
pre-Phase-18) that no longer exists, frozen ever since — not merely a random, never-touched head. This is
the worse story Cam anticipated: the gate wasn't just guessing randomly, it's been consulting a stale oracle
calibrated to a network that has since changed underneath it.

### Finding 3, resolved: the alarm-channel verdict survives; a different, adjacent claim does not

Delegated git archaeology traced the `idx_offset` bug's origin precisely:

- `d2b106a` (2026-06-10, Phase 17): `idx_offset = 6 + ...` introduced. **Correct at the time** — `own_state`
  really was 6-dim then.
- `c1a76d1` (2026-06-12, Phase 17): `own_state` grows 6→10 dims (4D intrinsic entropy added). `idx_offset`
  not updated — **bug introduced here**, off by 4.
- `d9905c4` (2026-06-13, **Phase 17.5**, "Alarm dimensions fixed, smoke test passing"): adds the `nb_alarms`
  block into the flat observation. `idx_offset` still not updated. At this commit: correct offset = 614,
  code's offset = 598 — **off by 16, already wrong at the moment Phase 17.5 shipped**, on the very commit
  titled "alarm dimensions fixed." (This is the same bug Finding 3 caught in the current tree, just smaller
  then — 16 columns short instead of today's 28, because `own_state_dim` has grown further since.)

But the specific Phase 17.5 "alarm channel is cheap talk" verdict (`THRONG.md`, commit `b4f1430`: *"Red_Dist
≈ 0, Resource ≈ 0, Puzzle = 0, Contested = 0"*) was produced by a **different script**,
`tools/alarm_correlation.py`, which reads `grid_state.resources`/`puzzle_grid`/`contested_res` directly at
the agent's grid position — it never touches the flat observation vector or `idx_offset` at all. That
verdict is genuine and does **not** need retraction.

What the buggy `idx_offset` *does* feed is the separate `local_resource`/`adj_bg`/`adj_barrier` corpus
fields (main_jax.py's corpus writer, Finding 3's original subject), which `tools/decode_signals.py` uses as
control covariates in its NPMI/LRT lexical analysis of the VQ token channel — the code behind the *adjacent*
headline in the same THRONG.md section: *"The z_q VQ channel IS load-bearing... LAG-1 DIRECTION LRT
p<0.005."* That claim's control covariates have been wrong since Phase 17.5 shipped. **This is a narrower,
more precise correction than originally framed: the alarm-is-cheap-talk verdict stands; the z_q-channel
LRT's covariate corruption is the one that needs a footnote (or a re-run) once Finding 3 is fixed.**

### `phase9_canvas.cross_attn_enabled`: never `true` in git history — this is a new experimental variable

Exhaustive `git log --all -p` / pickaxe search across every commit, every branch, both `config.yaml` and its
pre-rename `config_phase7.yaml`: **the key has been `false` at every single commit since its introduction**
(`5920511`, 2026-05-31, the same commit that added `NeighborCrossAttention` to `network_jax.py`). There is no
`false→true→false` toggle anywhere.

THRONG.md's own history claims otherwise — `cea2ffb`/`7105ddd`/`38f342a` (2026-05-31) declare
*"cross_attn_enabled: true on Modal (Cell 1 patches yaml)"* — but checking `config_phase7.yaml` at those
exact commits shows `false` in every case. The "true" was a runtime monkey-patch applied inside a Modal
notebook cell, never committed, leaving no reproducible artifact. A later commit (`8c5888b453`) even
mis-transcribes the checked-in default as `true` in a reference table — doc drift compounding doc drift.
`red_cross_attn_enabled` (Phase 12, predator-side) is a different, unrelated flag that has always been
`true` — several raw grep hits are for that key, not this one; don't conflate them.

**Answer to Cam's question: enabling `cross_attn_enabled` now is a new experimental variable, not a
regression to restore.** The one session where it may have run "live" did so via an unreproducible,
uncommitted patch. `docs/STRATEGIC_ROADMAP.md` §11's GWT-2 "Present" score, which leans on cross-attention as
blue's validated selective-attention mechanism, rests on that same unverified claim — combined with Finding
2's fossil confidence head undermining GWT-4, that's two of the four GWT indicators resting on weaker ground
than the scorecard currently shows.

---

## Appendix — verification method for Finding 1/2/10

`jax_sim/network_jax.py` cannot be imported (Finding 1), which would have blocked verifying anything else by
running real code. To still satisfy Cam's "prove it on a CPU minibatch, don't just reason about it"
requirement for the other findings, a throwaway harness (outside the repo, in the session scratch directory)
loaded `network_jax.py`'s source via `importlib` and pre-bound the name `flax` in its execution namespace
before running the module body — equivalent to what the one-line fix would do, without writing that fix into
the repo. This harness was used only to run the existing test suite and the Finding 2 gradient check; it was
not used to change, fix, or paper over anything in the report's findings. No repo file was modified.
