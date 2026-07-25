# Phase 18.8 — Receiver-Necessity Ecology

> **Status:** implemented, **all knobs default to 18.7 behaviour**. Nothing changes
> until `config.yaml → phase18_8_receiver_necessity` is edited. Enabling any of
> these is an ecology change and needs Cam sign-off (Rule 12).
>
> **Prerequisite:** the Phase 18.2 ATE gate must be re-run first with the fixed
> `tools/causal_intervention.py` (see §4). Do not change the ecology and the
> measurement in the same step — you will not be able to attribute the result.

---

## 1. What 18.7 actually enforced

18.7 is described as the "Receiver-Necessity Ecology". Read against the source,
it enforces **cooperation** but not **communication**:

| Claim | Reality in code |
|---|---|
| Solo crafting is impossible | **True.** Inventory capacity is 1 item total (`main_jax.py` `currently_empty`) and every generated recipe needs 3–4 units, so ≥3 agents are required. |
| The receiver needs the message | **False.** Three independent leaks, below. |

**Leak 1 — the bounty pays for proximity, not contribution.** `craft_success = is_craft & recipe_satisfied` is true for *every* member of the adjacent craft group, and the full `+3.0` is paid to all of them. An agent holding nothing, standing beside three crafters and pressing Craft, is paid identically to the agents whose materials were consumed. Against a `−0.20` futile penalty, "hold anything, loiter, spam Craft" is profitable at roughly a 7% hit rate — no listening required.

**Leak 2 — the recipe is memorisable.** `current_recipe` is global and rotates every 1000 steps. `ppo_rollout_steps` is 512, so a recipe survives ~2 PPO updates. That is ample for the policy to encode "the currently live recipe" directly in its weights from reward feedback alone. The `can_see_recipe` bit is then decorative: a blind agent recovers the recipe from experience, not from a neighbour.

**Leak 3 — sighted agents are not senders.** Nothing stops a recipe-sighted agent from doing the whole job itself: it sees the recipe *and* can pick up materials *and* can craft. There is no information asymmetry by construction — only an information *inequality* that any agent can route around by acting alone-ish.

This is receiver-necessity by *incentive*. The referential-game literature is consistent that only receiver-necessity by *construction* produces positive listening.

---

## 2. What 18.8 changes

All in `jax_sim/main_jax.py`, all gated.

| Knob | Default | Effect |
|---|---|---|
| `strict_asymmetry` | `false` | Recipe-sighted agents cannot execute PickUp. Sighted = pure sender, blind = pure actor. A blind agent now has **no non-channel route** to knowing which material to fetch. Closes leak 3. |
| `contributor_only_reward` | `false` | The `+3.0` is paid only to agents whose material was consumed. Closes leak 1. |
| `consume_required_only` | `false` | Only destroy materials the live recipe requires. 18.7 destroyed everything the crafter held, which also made "hold anything" equivalent to "hold the right thing". |
| `exclude_self_from_craft_group` | `false` | A lone agent cannot form a craft group with itself. Belt-and-braces today; load-bearing if a 1-unit recipe is ever generated. |
| `recipe_rotation_steps` | `1000` | Set below `ppo_rollout_steps` (512) so the live recipe cannot be carried in weights. Closes leak 2. |
| `sender_credit_frac` | `0.0` | See §3 — the open design question. |
| `sender_credit_radius` | `8` | Earshot for the above. |

New telemetry on the dashboard:

```
Crafting: success=N | contrib=M | free-rider=N-M (x%) | futile=F | rate=r%
```

**`free-rider%` is the diagnostic that matters.** If it stays high with
`contributor_only_reward: false`, the ecology is paying for proximity. If it
collapses when you enable the flag, the mechanic is biting.

---

## 3. The open problem: sender credit

Blue is a **shared policy**. The same weights are both sender and receiver, but
PPO credits each *agent* with its own reward. Under `strict_asymmetry` a
recipe-sighted agent can never craft, therefore never earns the bounty,
therefore receives **no gradient whatsoever** toward emitting an informative
token. The predicted outcome is not a language — it is silence.

This is the credit-assignment hole that produces cheap talk, and it is the real
reason eight phases of sender-side surgery never moved the receiver number. It
has three possible resolutions and they are genuinely different bets:

1. **`sender_credit_frac > 0`** (implemented, off). Pay a sighted agent a
   fraction of the bounty when a contributor succeeds within earshot. It is
   outcome credit — paid only on a real craft, never on the act of signalling —
   so it is arguably compatible with Standing Directive 3 ("never comm reward
   shaping"). But it is an ecological parameter and the boundary is a judgement
   call, not a fact. **Cam's call.**
2. **Gradient flow (DIAL).** Let the receiver's loss backpropagate through the
   message into the sender. This is the intervention with the strongest
   mechanism-level evidence in the literature and it dissolves the credit
   problem instead of paying around it. It does not touch rewards at all, so it
   sidesteps Directive 3 entirely — but it couples the two networks during
   training. Not implemented.
3. **Population selection.** Let differential survival do the work. This is the
   pure-emergence answer and it is what THRONG has been betting on. It is also
   the one the telemetry says is not firing: `NB_GAIN↔surv` has been `nan` for
   most of the project because blues sit at the population cap with ~99%
   survival, so there is no selection pressure on signal benefit at all. This
   route requires fixing the cap/mortality first, not just waiting longer.

**Recommendation:** stage 2 before 1. If DIAL produces non-zero CIC, the ecology
stays mathematically pure and Directive 3 is never tested. If it does not, the
sender-credit knob is the fallback and Cam decides the coefficient.

---

## 4. Staging (do not enable everything at once)

The 2763 checkpoint has 1.4M steps of behaviour tuned to the 18.7 ecology.
Enabling `strict_asymmetry` and a 64-step recipe simultaneously changes both the
action space's effective semantics and the task distribution in one shot.

**Back up the checkpoint before step 0.** (`/mnt/throng-runs/ckpt_backup_2763`.)

| Step | Change | Gate before proceeding |
|---|---|---|
| **0** | Nothing. Re-run the ATE gate with the fixed instrument. | A believable ATE number, positive or null, at n≥1500. |
| **1** | `consume_required_only: true`, `exclude_self_from_craft_group: true` | Pure bug-fixes. Craft rate should barely move. If it moves a lot, something else was depending on the old behaviour. |
| **2** | `contributor_only_reward: true` | `free-rider%` collapses. Expect a transient drop in craft rate and a Craft-action frequency dip as the loiter policy stops paying. Let it recover ~50k steps. |
| **3** | `recipe_rotation_steps: 256` (then 64) | Craft rate will drop hard. This is the point at which memorisation stops working. Watch that it recovers *at all* — if it does not, the task became unlearnable rather than communication-dependent, and you should back off to 256. |
| **4** | `strict_asymmetry: true` | The real experiment. Re-run the ATE gate. |

**Falsification.** If, at step 4 with adequate power, per-slot CIC is still
indistinguishable from zero *and* craft rate is non-zero, then agents are
solving the task without the channel and the necessity is still not binding —
find the remaining leak before adding more machinery. If craft rate is ~zero,
the ecology is too hard and the result is uninformative; back off step 3.

---

## 5. Files

| File | Change |
|---|---|
| `jax_sim/main_jax.py` | Config block `_p188`; pickup gating; craft-group self-exclusion; `consume_required_only`; `contributed`; contributor-only bounty; sender credit; recipe rotation period; `craft_contributed` telemetry |
| `config.yaml` | `phase18_8_receiver_necessity` block, all defaults = 18.7 |
| `tools/causal_intervention.py` | Slot-slicing + audibility fixes (see THRONG.md §0b) |
| `scripts/train_rosetta.py` | Untrained-params bug + Phase-18 token-format bug |
