# DRAFT — step 3b: the lookout world (for Cam; priced; not built or run)

Per Cam (Nov 8): knowledge comes from the **geography**, not from an assigned role. Whoever stands at a lookout sees
which fire is lit; nobody else does. Lookout distance and lit duration are set so that walking there yourself loses
to telling a nearer partner.

## The free calculation, and why the current toy torus can't do it

Agents move 1 cell per step (4-neighbour); delivery needs Chebyshev <= 1 of the fire. So the steps from p to fire h
are max(|dx| - 1, 0) + max(|dy| - 1, 0) on the torus.

**The symmetric layout fails.** With the four fires on the generator's lattice (G/4, 3G/4), the cell farthest from
all fires is equidistant from every fire, and it is also the best place to *wait*. A partner is then no nearer on
average: a uniformly placed partner is nearer than the lookout only 40-45% of the time at G = 12 / 16 / 20 / 24, and
the mean partner distance equals the lookout's (4.2 vs 4 at G = 12; 10.1 vs 10 at G = 24). **A lookout needs
asymmetric geography.**

**Proposed layout:** G = 24 torus; the four fires **clustered** at (3,3), (3,9), (9,3), (9,9) (today's G = 12
spacing); the lookout is the 3 x 3 area centred on (18,18).
- Walking from the lookout to the lit fire: **16 steps** to every fire (computed).
- A partner waiting at the cluster centre (6,6): **4 steps** to every fire, so a told partner arrives in **5**
  (1 step of wire delay).
- **Lit duration R = 10 steps** (the fire re-rolls after 10 steps without a delivery, or on a delivery): walking from
  the lookout always loses (16 > 10), and a told partner always makes it (5 < 10). Uninformed blind search from the
  centre reaches one fire per ~4 steps, so it finds the lit one within R only part of the time.
- Respawn after a delivery: uniform over cells not near a fire, as now. Both agents are identical, speak and listen,
  and the shared reward is symmetric (as step 3).

## Value-of-information check (before any comm arm)

- **LOOKOUT-ORACLE** (every agent always sees the lit fire) vs **NOSIG** (wire off; knowledge only at the lookout),
  same world, 2 seeds each, 2,000 updates, PS off (no messages matter).
- **Qualifies** if ORACLE deliveries >= 1.5x NOSIG. If not: retune the distances / R (in the world), rerun, and
  never change the agents.

## Arms (once the world qualifies)

LOOKOUT x5 (comm), NOSIG x2, ASSIGNED x3 (today's sender / listener roles, placed in the same world), with the
replication's PS recipe and C1's **wire-blind message head** (C1-T3 confirmed the coupling).
- **Pass:** redirection under the swap (every currently informed speaker's broadcast replaced; uninformed hearers'
  Crafts at T') in >= 4 of 5 LOOKOUT seeds, at >= 0.5x the ASSIGNED mean.
- **Reported, not gated:** the share of agent-steps spent at the lookout; whether a lookout / runner division of
  labour emerges (per-agent position histograms).

## Code needed (free)

`--hearths` (custom fire positions), `--lookout x,y,r` (informed iff within the lookout area; the prox swap / MI
machinery is reused), and `--reroll 10`. Apparatus: need dims nonzero only inside the lookout area; mutant
(lookout everywhere) caught.

## Price

VOI check 4 runs (~$0.8) + arms 10 runs (~$2.0) = **~$2.8**. Remaining toy allowance after C1 (~$2.0), step 2 (~$2.4)
and step 6 (~$1.7) is ~$2.3, so **step 3b waits for M's top-up** (Cam's rule). The VOI check alone (~$0.8) would fit
if Cam wants it first.
