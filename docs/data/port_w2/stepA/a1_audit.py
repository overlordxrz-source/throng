"""A1 (Cam Nov 17 ~13:00Z): sector-asymmetry audit of the W2 named-item world, from the code, tested numerically."""
import sys, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy"); sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy/w2")
from w2 import named_item as NI, env, relabel
from w2.learnability import sector as lsector
import traj_analysis as TA
L = 96; out = {}; fails = []
# 1. placement on every ray, d = 1..24, with the code's own float32 JAX path (NI.place with a forced sector)
def place_sec(sec, d, start):
    ang = jnp.float32(sec) * (jnp.pi / 4)
    return np.asarray((start + jnp.stack([jnp.round(d * jnp.cos(ang)), jnp.round(d * jnp.sin(ang))], -1).astype(jnp.int32)) % L)
# confirm the helper matches NI.place: draw many items and compare
key = jax.random.PRNGKey(5); starts = jax.random.randint(key, (4000, 2), 0, L)
sec, d, ipos = NI.place(jax.random.PRNGKey(6), starts, lambda k: jax.random.randint(k, (4000,), 1, 25).astype(jnp.float32), L)
mine = np.stack([place_sec(int(s_), jnp.float32(d_), st[None])[0] for s_, d_, st in zip(np.asarray(sec), np.asarray(d), np.asarray(starts))])
out["helper_matches_NI_place"] = bool((mine == np.asarray(ipos)).all())
cells = {}
for k in range(8):
    for dd in range(1, 25):
        o = place_sec(k, jnp.float32(dd), jnp.zeros((1, 2), jnp.int32))[0]; o = (o + L // 2) % L - L // 2; cells[(k, dd)] = o.tolist()
sym = []
for dd in range(1, 25):
    car = [cells[(k, dd)] for k in (0, 2, 4, 6)]; dia = [cells[(k, dd)] for k in (1, 3, 5, 7)]
    ok_c = car == [[dd, 0], [0, dd], [-dd, 0], [0, -dd]]
    m = int(round(dd / np.sqrt(2))); ok_d = dia == [[m, m], [-m, m], [-m, -m], [m, -m]]
    if not (ok_c and ok_d): fails.append(f"placement asymmetric at d={dd}: car {car} dia {dia}")
    sym.append(ok_c and ok_d)
out["placement_symmetric_all_d"] = all(sym)
out["cells_d3_12_21"] = {f"sec{k}": [cells[(k, dd)] for dd in (3, 12, 21)] for k in range(8)}
# exact-half rounding cases (d / sqrt2 within 1e-3 of x.5): none means no half-to-even issue
out["half_cases"] = [dd for dd in range(1, 25) if abs((dd / np.sqrt(2)) % 1 - 0.5) < 1e-3]
# 2. torus wrap at the item cell: starts at every edge/corner, every sector, d = 3/12/21
wr = []
for st in [(0, 0), (95, 0), (0, 95), (95, 95), (90, 3), (3, 90), (48, 48)]:
    for k in range(8):
        for dd in (3, 12, 21):
            ip = place_sec(k, jnp.float32(dd), jnp.array([st], jnp.int32))[0]
            wr.append(NI.torus(np.array(ip) - np.array(st), L).tolist() == cells[(k, dd)])
out["torus_wrap_ok"] = all(wr)
if not all(wr): fails.append("torus wrap")
# 3. MOVES: displacements and the "toward" sets of each sector (traj_analysis.into); stay and eat are one action each
out["MOVES"] = np.asarray(env.MOVES).tolist(); out["EAT"] = int(env.EAT)
out["toward_moves_by_sector"] = {k: [m for m in (1, 2, 3, 4) if TA.into(m, k)] for k in range(8)}
# 4. visibility (NI.observe's mark): agent at each of the 8 neighbours and on the item, every sector, item mark at the
#    right window cell, and never beyond Chebyshev 1
cfg = env.Cfg(L=L, N=1, patch_sigma=1.0, reproduce=False, A_max=3000)
s = env.reset(cfg, jax.random.PRNGKey(0)); s = s._replace(food=jnp.zeros((L, L)))
vis_ok = []
for k in range(8):
    for dd in (3, 12, 21):
        st = jnp.array([[40, 40]], jnp.int32); ip = jnp.asarray(place_sec(k, jnp.float32(dd), st))
        item = dict(NI.new_item(1), pos=ip, start=st, act=jnp.ones(1, bool))
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                pos = (ip - jnp.array([[dx, dy]])) % L; o = np.asarray(NI.observe(cfg, s._replace(pos=pos), item))[0, :49].reshape(7, 7)
                want = max(abs(dx), abs(dy)) <= 1; got = o[dx + 3, dy + 3] > 0.5
                vis_ok.append(want == got and o.sum() == (1.0 if want else 0.0))
out["visibility_ok"] = all(vis_ok)
if not all(vis_ok): fails.append("visibility")
# 5. eat: the rule is item act & pos == item pos & action == EAT, evaluated BEFORE env.step; EAT moves 0 -> symmetric.
#    Check env.step keeps the position on EAT and stay.
for act in (0, 5):
    s2, _, _ = env.step(cfg, s._replace(pos=jnp.array([[10, 10]])), jnp.array([act]), jax.random.PRNGKey(1))
    if np.asarray(s2.pos).tolist() != [[10, 10]]: fails.append(f"action {act} moves")
# 6. sector binning of head labels / records: learnability.sector, relabel.sector_of, traj_analysis.sector_of on every
#    placed offset, and on the record round trip (disp / 24 -> *24 -> round)
bins = []
for (k, dd), o in cells.items():
    o = np.array(o); rt = np.round((o / 24.0).astype(np.float32) * 24.0)
    bins.append(int(lsector(jnp.array(o))) == k and int(relabel.sector_of(rt)) == k and TA.sector_of(o) == k)
out["sector_binning_ok"] = all(bins)
if not all(bins): fails.append("sector binning")
# sector binning of ALL integer offsets in a 49x49 box: count cells per sector (a binning asymmetry would show here)
cnt = np.zeros(8, int)
for x in range(-24, 25):
    for y in range(-24, 25):
        if x or y: cnt[int(lsector(jnp.array([x, y])))] += 1
out["binning_cells_per_sector_49x49"] = cnt.tolist()
cnt2 = np.zeros(8, int)
for x in range(-24, 25):
    for y in range(-24, 25):
        if x or y: cnt2[int(relabel.sector_of(np.array([x, y])))] += 1
out["relabel_cells_per_sector_49x49"] = cnt2.tolist()
# 7. the scripted parent (step_toward): path from the start to each sector's item at d = 12; length and action counts
par = {}
for k in range(8):
    pos = jnp.array([[40, 40]], jnp.int32); tgt = jnp.asarray(place_sec(k, jnp.float32(12), pos)); acts = []
    for _ in range(40):
        pos, a_ = NI.step_toward(pos, tgt, L); acts.append(int(a_[0]))
        if acts[-1] == 0: break
    par[k] = dict(n_moves=len(acts) - 1, first8=acts[:8])
out["parent_paths_d12"] = par
pl = [par[k]["n_moves"] for k in (1, 3, 5, 7)]
if len(set(pl)) > 1: fails.append(f"parent path lengths differ across diagonals {pl}")
# 8. sector draw: uniform over 0..7 (NI.place uses randint(0, 8))
sc = np.bincount(np.asarray(sec), minlength=8); out["sector_draw_counts_4000"] = sc.tolist()
out["FAILS"] = fails
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
