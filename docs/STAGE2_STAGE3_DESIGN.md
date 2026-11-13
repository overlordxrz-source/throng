# Stage 2 and stage 3 designs (draft for Cam; no spend)

Nov 12 2026, Will. Two designs Cam asked for while the stage-1 pilot runs. Neither is registered or launched.
Both reuse the stage-1 harness (`w2/learnability.py`) and its calibrated instruments.

## Stage 2: W2.1 parent calls with learned speakers

**Question.** When a parent can call to its own young and gains when the young eats, does a code arise that the young
relies on? Nothing names what a call means.

**World.** The picked W2 world (drain 0.0013, 8 patches, relocation 0.001), with no beacons.
- Each newborn has one parent beside it through the juvenile period. The parent leaves at age 1,000, as in stage 1.
- The young hears only its own parent: the token, with the parent's offset. This keeps copying to a single elder (Cam's
  design note) and makes the call the only channel.

**Agents.** One lineage: one set of slow weights shared by parents and young; the role comes from age.
- The stage-1 `RecAgent` (frozen navigator trunk, GRU memory, the four approved drives) gains a speak head: a token
  (8-way, or the 3-slot channel) chosen every step as an action.
- The speak head starts at zero logits, so calls start random and carry no meaning.

**Pressure.** The existing inclusive-fitness reward with k = 0.5: a parent's reward includes half its young's drive
reduction. A parent therefore gains when its young eats.
- No reward ever names a call or a meaning.
- Positive signalling stays off. If the arms show no channel at all, the fallback is founders-only positive
  signalling with an anneal, as approved for the toy, and that needs a ruling first.

**Copying.** A young may copy its parent's actions and calls, from its parent only. This lever is off in the
first run and enters only if reliance appears without it.

**Arms.** 5 seeds each:
- **wire on:** the young hears its parent;
- **wire off:** the young hears zeros, identical otherwise.

**Readouts,** at fixed agent-step milestones, as in stage 1:
1. **Reliance with CI (primary).**
   - The frozen life test, adapted: the young is probed with its frozen parent beside it, both carrying their
     end-of-life memories.
   - True calls vs calls deranged among the tokens the young heard ≥ 25 times; last-heard credit; eligibility ≥ 5
     tokens heard ≥ 25 times.
   - Pass = drop ≥ 30% with paired > 2 SE; pass rate over eligible lives with Wilson CI.
2. **Food gain to the young:** juvenile intake per step, wire on vs wire off, paired by seed, with CI.
3. **Speaker MI:** shuffle-corrected MI between a parent's calls and candidate variables:
   - direction to the nearest food from the parent;
   - direction to the nearest food from the young;
   - distance band;
   - the young's energy.
   Reported per variable, never gated on one chosen in advance.

**Calibration before any learner** (the stage-0 standard):
- Positive control: a scripted parent that calls the direction to the food it sees, using that life's random code,
  with a perfect scripted listener. Must reach ≥ 90%.
- Deaf control: ≤ 10%.
- Value of the call, both forms ≥ 1.5×.

**Bars (to register with Cam).**
- Wire-on reliance ≥ 50% of eligible lives.
- Food gain over wire-off with a CI excluding 0.
- Wire-off reliance ≤ 10%.

**Cost.** One stage-2 run is about one stage-1 run (~$3–4 per 1e9 agent-steps), so ~$15–20 for 2 arms × 5 seeds at
1e9 each. That is above what is left; it would need new credits.

**Risks.**
- The usual emergent-communication collapse: calls stay uninformative because no one listens yet and no one
  speaks usefully yet.
- The parent's memory must hold what it saw for the call to be useful.
- Stage 1 tells us whether listening can work at all. Stage 2 should wait for that result.

## Stage 3: the cross-fostering instrument

**Question.** When a newborn of lineage A is raised by a parent of lineage B, whose code does it use?
- Following B's code means the code is learned in the life (culture).
- Following A's means it is carried in the lineage's weights (inborn signalling).

**Instrument.** For each tested life we know two codes:
- **birth code** (lineage A);
- **foster code** (lineage B, the one it was raised hearing).

Probe episodes play the same scripted speaker twice, once encoding with the foster code and once with the birth
code. Same keys, same worlds; last-heard credit; eligibility as in the frozen test.
- **Uses foster code:** credited food under foster-coded probes exceeds birth-coded probes by ≥ 30%, with paired > 2 SE.
- **Uses birth code:** the reverse.
- The two are scored separately, so "neither" (deaf, or no code) shows up as failing both.

**Calibration with scripted lineages first** (Cam's condition). Each listener is scored by its foster score and
its birth score:

| Listener | Foster score must be | Birth score must be |
|---|---|---|
| perfect foster-learner (decodes with the foster code) | ≥ 90% | ≤ 10% |
| perfect birth-decoder (decodes with the birth code) | ≤ 10% | ≥ 90% |
| deaf (navigator) | ≤ 10% | ≤ 10% |

**For learned lineages (stage 3 proper):** a lineage's code is not a known permutation, so it is measured as that
lineage's dictionary: its modal call per situation in frozen test episodes. Probes then play back each lineage's
dictionary through a scripted speaker. Defining "situation" for learned calls depends on what stage 2's MI readout finds.

## Cross-fostering calibration, first run (laptop, Nov 12; comm-toy `131f579`, `learnability.py --crossfoster`)

75 eligible lives, gate config with a random code per life, **16 probes x 600 steps** (laptop limit):

| listener | uses foster code | uses birth code |
|---|---|---|
| perfect foster-learner | 70.7% (59.6-79.8) | 0.0% (0-4.9) |
| perfect birth-decoder | 0.0% (0-4.9) | 69.3% (58.2-78.6) |
| deaf navigator | 0.0% | 0.0% |

- **The instrument is clean and symmetric:** no listener is ever credited with the wrong code, and the deaf
  navigator passes nothing.
- **Not yet calibrated:** the perfect listeners reach ~70%, not 90%. The likely cause is the probe count: the frozen
  test uses 48 probes, and the same perfect listener scored 83% at 24 probes and 92% at 48, because the paired > 2 SE
  criterion needs probes.
- **Next:** the 48-probe calibration (~25 min laptop, so Modal ~/bin/zsh.1), when Cam allows.

## Design note: fitness must be paid in food (Cam, Nov 13)

The listening-premium check (protocol, Nov 13 ~09:40) found that the best evolved stage-1 forager (D-best) outlives
the scripted true-code follower: 0.82 vs 0.54 survive past 1,300. Yet the follower eats 2.18x as much as an adult. So
a pure survival-time fitness favours a code-free longevity strategy over listening.

**Rule for stage 2:** when reproduction arrives, it must be paid from gathered energy (an energy threshold to breed,
or an energy cost per offspring). Then food intake, not just time alive, drives fitness. Re-run the return check by
age band under the stage-2 reward before deciding on any extra term (the alive bonus stays parked).
