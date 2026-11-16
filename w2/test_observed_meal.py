"""Unit tests for w2/observed_meal.py (Cam Nov 16 ~07:30Z check (a)). Run: python w2/test_observed_meal.py"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import observed_meal as OM  # noqa: E402
from w2 import named_item as NI  # noqa: E402
from w2.learner_w2_kv import own_meal_mask  # noqa: E402

L = 64; FAIL = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAIL.append(name)


# 1. the observed displacement's sector equals the named sector for every placement (all sectors, d 1..24, random starts)
key = jax.random.PRNGKey(0); bad = 0; n = 0
for d in range(1, 25):
    key, k = jax.random.split(key); start = jax.random.randint(k, (512, 2), 0, L)
    sec, _, ipos = NI.place(jax.random.fold_in(k, 1), start, lambda kk: jnp.full((512,), float(d)), L)
    pos = (ipos + jax.random.randint(jax.random.fold_in(k, 2), (512, 2), -1, 2)) % L          # the young adjacent to the item
    disp = OM.own_frame_disp(pos, start, ipos, L)
    bad += int((OM.sector_of(np.asarray(disp)) != np.asarray(sec)).sum()); n += 512
    bad += int((np.asarray(disp) != np.asarray(NI.torus(ipos - start, L))).sum())
check(f"own-frame displacement = item - start and its sector = the named sector ({n} placements)", bad == 0)

# 2. the STRICT event
z = jnp.zeros((1, 2), jnp.int32); T = jnp.array([True]); Fa = jnp.array([False])
it = jnp.array([[10, 10]]); young_adj = jnp.array([[11, 11]]); young_far = jnp.array([[12, 10]])
check("event: parent on the item, item adjacent -> True", bool(OM.event(T, T, it, it, young_adj, L)[0]))
check("event: parent on the item but 2 cells away (in the 7x7 view, item not visible) -> False", not bool(OM.event(T, T, it, it, young_far, L)[0]))
check("event: parent adjacent but NOT on the item -> False", not bool(OM.event(T, T, jnp.array([[10, 11]]), it, young_adj, L)[0]))
check("event: parent not shown (dem_on False) -> False", not bool(OM.event(Fa, T, it, it, young_adj, L)[0]))
check("event: no bout -> False", not bool(OM.event(T, Fa, it, it, young_adj, L)[0]))
check("event across the torus edge", bool(OM.event(T, T, jnp.array([[0, 0]]), jnp.array([[0, 0]]), jnp.array([[L - 1, L - 1]]), L)[0]))

# 3. keep the first event; own meal wins at the record
f, dd = OM.keep_first(Fa, z, T, jnp.array([[3, 4]])); f, dd = OM.keep_first(f, dd, T, jnp.array([[5, 6]]))
check("keep_first keeps the first event's displacement", bool(f[0]) and dd.tolist() == [[3, 4]])
disp, ate, uo = OM.record(T, T, jnp.array([[1, 2]]), jnp.array([[3, 4]]))
check("record: own meal wins over an observed meal (own displacement, ate, not observed)", disp.tolist() == [[1, 2]] and bool(ate[0]) and not bool(uo[0]))
disp, ate, uo = OM.record(Fa, T, jnp.array([[1, 2]]), jnp.array([[3, 4]]))
check("record: observed meal without own meal -> observed displacement, bit 1, observed", disp.tolist() == [[3, 4]] and bool(ate[0]) and bool(uo[0]))
disp, ate, uo = OM.record(Fa, Fa, jnp.array([[1, 2]]), jnp.array([[3, 4]]))
check("record: neither -> own displacement, bit 0 (unchanged behaviour)", disp.tolist() == [[1, 2]] and not bool(ate[0]) and not bool(uo[0]))

# 4. label masks on synthetic sequences (B=5, T=8); hearing at t=1, close at t=5
B, Tn = 5, 8; hear = np.zeros((B, Tn), bool); hear[:, 1] = True; closed = np.zeros((B, Tn), bool); ate = np.zeros((B, Tn), bool)
dead = np.zeros((B, Tn), bool); oc = np.zeros((B, Tn), bool); ot = np.full((B, Tn), -1)
closed[0, 5] = True; oc[0, 5] = True; ot[0, 5] = 6                       # 0: observed close -> observed label 6
closed[1, 5] = True; ate[1, 5] = True; oc[1, 5] = True; ot[1, 5] = 6     # 1: own meal at the close (observed flag too) -> own label
closed[2, 5] = True                                                      # 2: T_max close, no observed meal -> unlabelled
dead[3, 5] = True; oc[3, 5] = True                                       # 3: death (obs flag) -> unlabelled
closed[4, 5] = True; oc[4, 5] = True; ot[4, 5] = 2; hear[4, 6] = True    # 4: a new bout heard after the close, never closed
mk, mo, tg = OM.label_mask(hear, closed, ate, dead, oc, ot)
check("label: observed close labels the hearing step, as observed, target carried back", bool(mk[0, 1]) and bool(mo[0, 1]) and tg[0, 1] == 6)
check("label: own meal wins (labelled, not observed)", bool(mk[1, 1]) and not bool(mo[1, 1]) and tg[1, 1] == -1)
check("label: T_max close without observed meal -> unlabelled", not mk[2].any())
check("label: death -> unlabelled", not mk[3].any())
check("label: the next bout's hearing with no end in the sequence -> unlabelled; the first bout labelled", bool(mk[4, 1]) and not bool(mk[4, 6]))
check("label: never two labels per hearing (observed only where labelled)", not (mo & ~mk).any())
rng = np.random.default_rng(0); hear = rng.random((64, 64)) < 0.1; closed = rng.random((64, 64)) < 0.1; ate = closed & (rng.random((64, 64)) < 0.5)
dead = rng.random((64, 64)) < 0.01; zero = np.zeros((64, 64), bool)
mk0, mo0, _ = OM.label_mask(hear, closed, ate, dead, zero)
check("label: with no observed closes it equals own_meal_mask exactly (random 64x64)", np.array_equal(mk0, own_meal_mask(hear, closed, ate, dead)) and not mo0.any())

print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
