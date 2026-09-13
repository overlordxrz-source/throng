# THRONG

Meaning that is causally earned cannot be faked. A large language model's
symbols are grounded only in other symbols — text produced by humans who were
themselves grounded, at one remove the model never touches. A THRONG token is
grounded in whether the agent that hears it survives. That is the entire bet:
build a world where communication has to pay rent in survival, and see whether
anything recognizable as language, culture, or planning is selected for.

## Mechanism

THRONG is a multi-agent simulation on a toroidal grid. Two populations —
**blue** (prey) and **red** (predator) — share one policy per team, trained
with MAPPO. Blues are partially blind: they see a local patch and a handful of
nearest neighbours, nothing more. Predation is lethal and unshaped — there is
no reward for communicating, cooperating, or exploring, only for staying
alive.

Blues speak through a single channel: three discrete VQ codebooks (an 8-bit,
an 8-bit, and a 12-bit slot on the wire), broadcast to nearby agents every
step. Nothing about the channel is free-form; a token is one of a small,
fixed vocabulary, chosen by the same policy that chooses movement. The
grounding pressure comes from **receiver necessity**: agents must craft using
a shared, rotating recipe that only half the population can see. An informed
agent has no reward for telling a blind one what to gather — but a blind
agent that never receives usable information cannot craft, and a population
that cannot craft loses whatever craft confers. If a token carries no
information a receiver can act on, there is nothing selecting for it to
persist.

```
  blind agent -------- 5x5 local patch --------\
                                                 \
  informed agent -- 3-slot VQ signal (12/8/12) -- shared transformer -- action
                                                 /
  neighbours (k=6) ------- signals + culture ---/
```

Two slower channels run alongside the per-step signal: a fast-decaying
"culture" grid (danger traces, ~10-step memory) and a slow one (~200-step
landmarks), both written and read by the same agents. Whatever any blue
learns is shared instantly across the whole team, because they share one set
of weights — the "cultural transmission" channel is the gradient itself.

## Current status

The infrastructure is live: a 12-action space (movement, Strike, Push, Guard,
Build, PickUp, Craft, UseTool — Push and Guard are logit-masked everywhere,
they have no environment mechanics), the 3-slot discrete channel, and the
receiver-necessity crafting ecology are all implemented and exercised
end-to-end. `docs/AUDIT_SEP2026.md` and `docs/WILL_RESTART_SEP2026.md` record
an adversarial pass over the codebase and the fixes that came out of it —
worth reading before trusting any specific number in this repository, current
or historical.

Three things worth stating plainly rather than letting the code imply
otherwise: cross-attention over neighbour signals (`cross_attn_enabled`) is
off — blue currently integrates neighbour signals through a simpler
per-neighbour path, not attention, and turning attention on is logged as the
next planned experiment, not a shipped feature. Blue's recurrent state update
is a fixed exponential moving average, not a learned gate (red predators do
use a gated GRU cell; blue does not, yet). And the epistemic gate that
decides whether an agent acts reactively or deliberates over a short mental
rollout is being actively repaired — its confidence head was found to have
been running untrained (and, worse, carrying stale weights forward from a
now-obsolete architecture) for an unknown span of training.

The one significant causal result on record — that tokens in one VQ slot
causally raise a blind receiver's flee rate — was originally measured in a
two-update window selected because it looked anomalous, which inflates
effect size by construction; it was never a clean measurement. An
independent offline re-test against a later, stable ~100k-step span of the
same recovered corpus (`tools/ate_swap_test.py`) is the trustworthy number:
the effect **replicates and is real, but below the pre-registered
significance bar** (`|Δ| > 0.05`) — smaller than the original figure implied.
The other two slots remain statistically indistinguishable from no effect in
both measurements. The channel is weaker than earlier numbers suggested. A
live replication under a corrected codebook-reset mechanism has not yet run.
Nothing downstream of that replication (cross-attention, channel
cost, population scale, recurrent depth) is scheduled to run before it does.

## Running it

The default branch is **`master`**; it is the only branch a fresh clone or
training box needs. Confirm you're on it (`git branch --show-current`) and
that `git log -1 --oneline` doesn't look stale before trusting any run —
this repository has previously had `master` sit hundreds of commits behind
active development, which is exactly the kind of thing to check for rather
than assume.

```bash
git clone https://github.com/overlordxrz-source/throng.git && cd throng
pip install jax[cuda12] flax optax orbax-checkpoint pyyaml
python -c "
import yaml
from jax_sim.train_entry import run_simulation
cfg = yaml.safe_load(open('config.yaml'))
run_simulation(cfg, seed=42, n_steps=1_000_000)
"
```

`config.yaml` is the single source of hyperparameters; nothing else should
override it silently (this used to not be true — see the audit). For a
low-memory smoke run, shrink `population_size`, `red_population_size`, and
`grid_size` before launching, and make sure `ppo_minibatch_size` stays at or
below the resulting rollout size — `ppo_update` now raises rather than
silently skipping the gradient step if it doesn't.

Offline analysis tools live in `tools/`: `decode_signals.py` for corpus-level
NPMI/χ²/LRT decode, `ate_swap_test.py` for the offline stratified causal test
against a recovered corpus (no GPU required), `causal_intervention.py` for a
live frozen-checkpoint token-swap intervention (requires a GPU and a real
checkpoint). Regression tests for the training step itself live in `tests/`
and run on CPU in seconds — `test_severance_sweep.py` is the one to read
first, since it measures every loss term's gradient rather than assuming it.

## Where everything else lives

`THRONG.md` is the full research log, ops manual, and phase-by-phase history
— read it before making an architectural decision. `docs/ARCHITECTURE.md` is
the code-grounded reference for what the network and observation layout
actually do, and is authoritative over prose anywhere else when the two
disagree. `docs/STRATEGIC_ROADMAP.md` lays out the longer-term bet and the
open counter-arguments to it. `docs/AUDIT_SEP2026.md` and
`docs/WILL_RESTART_SEP2026.md` record the most recent adversarial review and
the restart plan that followed it.
