# Morning summary for M (Nov 9, overnight run)

Account `roadtoagi111`. Everything is recorded in `docs/RESEARCH_PROTOCOL.md`, with data in `docs/data/`.

## Spend

- **$3.92 metered, all covered by credits, $0 billed** (Modal billing summary, ~04:30). Nothing paid is running.
- The guard is at $41; the overnight hold line is $25.
- The Mac guard is still the guard: the Modal guard waits for M's secret `throng-guard-token`
  (`docs/GUARD_ON_MODAL.md`).

## Results against each bar

| Item | Bar | Result |
|---|---|---|
| (b) erosion calibration | smallest wd with every unused code < 0.05 bits by u2500 | **wd 1e-3** (3e-4 fails: one seed still 0.14 bits) |
| (c) rung-3 rerun at wd 1e-3 | reliance + gain at u2500 and u3000 in >= 3/5 per arm; wire-off < 0.05 bits; stop-grad adopted unless worse | **baseline FAIL 2/5; stop-grad FAIL 0/5; wire-off PASS** (all <= 0.002 bits, so the control now discriminates); **stop-grad not adopted**. The stop condition (0/5 in both arms) did not fire. |
| (d) navigators | each >= 17.4 deliveries per world | **8/8 pass** (22.5-25.3) |
| (d) G0 word-bias gate | different starts share words <= chance | **PASS:** 0.04 vs null 0.08; same start 0.50 (positive control) |
| (e) population | registered design | **built, apparatus PASS; not launched (price)** |
| (f) step-6 rerun | registered design | **code ready; not launched (judgment call)** |
| W2 prototype | P1 learned > random; P2 oracle >= 2x blind; P3 steps/s | **P1 PASS** (645 vs 61 steps); **P2 PASS** (2.34x); **P3** ~8M agent-steps/s (world only, laptop) |

## What it means

- With erosion strong enough to reliably kill an unused code, a *used* code survived in 2 of 5 baseline seeds, strongly
  where it did. That is below the 3/5 bar: **the toy does not show that use keeps the code alive.**
- Taking the listener's gradient off the message head made it worse (0/5), against the idea that listener noise erodes
  the code.
- The word-bias gate shows separate starting brains give separate words, so the population test is set up correctly.

## Waiting on Cam

1. **(e) price.** Re-priced on Modal: a population update takes 3.15 s on 2 CPUs (8 CPUs no faster; an L4 GPU slower,
   at 6.2 s). That puts 5 POP + 3 control at **~$11.3, against the $6 cap**, and each 9,000-update run at ~8 h. The
   option that fits: **shorter lives** (lifetime 1,000; generation 4 at ~4,000 updates), **~$5.0 for 8 runs**. That is a
   design change, so it's Cam's call.
2. **(f) tau.** Step 2's starting entropy (9.5 nats) is impossible with 4 symbols (maximum 4.16). The proposed
   reading is the same fraction of maximum (0.76), giving **tau = 0.63** (calibrated). OK to launch?
3. **The own-world spec** (`docs/OWN_WORLD_SPEC.md`): sign-off, plus three decisions: per-agent brains vs lineage
   brains, the predator escapes, and the first N / L.
