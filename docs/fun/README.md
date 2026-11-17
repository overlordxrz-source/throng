# Fun pack (Cam, Nov 17 ~14:20Z), eval only

All three items are recorded in `docs/RESEARCH_PROTOCOL.md` (search "FUN PACK"). F2 and F3 predictions were
registered before any run.

## F1: watch a life
- `f1_seed7_3e7_first_then_alone.gif` / `.mp4`: learner seed 7 at 3e7, world 0, agent 0.
  - Word #3 (sector 1) is heard twice. The FIRST time (d = 8) the parent (orange) leads, and the young (blue)
    follows and eats. While no record of word #3 exists, the young attends to another word's record.
  - LATER in the same life (d = 6) it hears word #3 again, alone. It attends to its own record of word #3 ("went
    (6, 6), ate"), the head arrow (red) points to sector 1, and it walks there and eats.
- `f1_seed7_1e8_dead_diagonal.gif` / `.mp4`: seed 7 at 1e8, a bout in diagonal sector 3 (d = 7).
  - The memory record and the head both say sector 3 with full confidence, and the young sits until the bout
    times out.
- Still frames: `f1_seed7_3e7_frames.png`, `f1_seed7_1e8_frames.png`. Renderer: `f1_render.py`; dumps from
  `traj_fun.py --dump`.

## F2: false memory (causal)
`f2_result.json`, `f2.py`. Seed 7 at 3e7, d = 6. Each token's SECOND hearing in a life: its one record, from the
parent-led first bout (solved 87-88%), is rotated before the hearing.

| edit | n | head -> true | head -> implanted | heading -> true | heading -> implanted | reach |
|---|---|---|---|---|---|---|
| none (0 deg) | 907 | .90 | (same) | .61 | (same) | .71 |
| 90 deg | 1310 | .02 | **.92** | .01 | **.61** | **.00** |
| 180 deg | 1187 | .01 | **.91** | .01 | **.62** | **.00** |

The young believes its memory: it goes where the record says, as often as it goes to the true place when the record
is true.

## F3: odometer probe
`f3b_*_probe.json` (the first pass, `f3_*`, had a torus-wrap target and an unstable ridge; it is kept as evidence and
not used). Linear probe for the distance travelled past the clip (9 cells), held out by world.

| features | oracle (b') 1e8, R^2 > 9 cells | seed 7 3e7, R^2 > 9 cells | positive control: R^2 within 9 cells |
|---|---|---|---|
| trained GRU state | -.14 | -.16 | .28 / .15 |
| random-init GRU, same inputs | -.05 | .10 | .97 / .97 |
| raw input | -.20 | .00 | 1.00 / 1.00 |
| raw input without energy / age | -.29 | -.03 | 1.00 / 1.00 |
| energy + age only | .03 | -.02 | -.02 / .01 |

No net carries a linearly readable step counter past the clip. The trained GRU state does not even carry the
in-range offset linearly (.15-.28): the trained policy reads the offset from the input directly.
