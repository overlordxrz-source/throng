"""Cam Nov 17 ~07:45Z G2b: offset-clip blindness. named_item.follower's rule in numpy (cross-checked against the JAX
original first), with the offset the agent actually sees (clip(off / 9, +-1) * 9) or unclipped, +- noise, tie rule."""
import sys, json, types
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy")
import numpy as np
import jax.numpy as jnp
from w2 import named_item as NI
L = 96; EAT = 5; MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])

def torus(o):
    return (o + L // 2) % L - L // 2

_a = jnp.arange(8) * (jnp.pi / 4); U32 = np.asarray(jnp.stack([jnp.cos(_a), jnp.sin(_a)], -1), np.float32)

def follow(pos, ipos, start, sec, clip, tie_rng=None):
    """numpy twin of NI.follower (no world food). pos/ipos/start (B,2), sec (B,)."""
    on = np.all(pos == ipos, -1); adj = torus(ipos - pos); seen = np.abs(adj).max(-1) <= 1
    tw = np.where(np.abs(adj[:, 0]) >= np.abs(adj[:, 1]), np.where(adj[:, 0] < 0, 1, 2), np.where(adj[:, 1] > 0, 3, 4))
    u = U32[sec]; off = torus(pos - start).astype(np.float32)          # float32 with the JAX u: exact ties break as in the original
    if clip:
        off = (np.clip(off / np.float32(9.0), -1, 1) * np.float32(9.0)).astype(np.float32)
    cand = off[:, None, :] + MOVES[1:5][None].astype(np.float32)
    score = (cand * u[:, None]).sum(-1) - np.float32(2.0) * np.abs(cand[..., 0] * u[:, None, 1] - cand[..., 1] * u[:, None, 0])
    if tie_rng is None:
        mv = np.argmax(score, 1) + 1
    else:
        best = score >= score.max(1, keepdims=True) - 1e-4
        mv = np.array([tie_rng.choice(np.flatnonzero(b)) for b in best]) + 1
    return np.where(on, EAT, np.where(seen, tw, mv))

# cross-check (unclipped, argmax ties) against the JAX original on random in-bout states
rng = np.random.default_rng(0); B = 4000
start = rng.integers(0, L, (B, 2)); sec = rng.integers(0, 8, B); d = rng.integers(1, 25, B)
ang = sec * np.pi / 4; ipos = (start + np.stack([np.round(d * np.cos(ang)), np.round(d * np.sin(ang))], -1).astype(int)) % L
pos = (start + rng.integers(-26, 27, (B, 2))) % L; pos[:200] = ipos[:200]; pos[200:600] = (ipos[200:600] + rng.integers(-1, 2, (400, 2))) % L
item = dict(NI.new_item(B), sec=jnp.array(sec, jnp.int32), d=jnp.array(d, jnp.float32), pos=jnp.array(ipos, jnp.int32), start=jnp.array(start, jnp.int32), act=jnp.ones(B, bool))
ref = np.asarray(NI.follower(types.SimpleNamespace(pos=jnp.array(pos, jnp.int32), food=jnp.zeros((L, L))), item, L, jnp.zeros(B, jnp.int32)))
mine = follow(pos, ipos, start, sec, clip=False)
assert (ref == mine).all(), f"numpy follower disagrees on {(ref != mine).sum()} of {B}"
print(f"cross-check vs NI.follower: {B}/{B} identical")

def run(sec_k, d, clip, noise, tie, n=200, seed=1):
    r = np.random.default_rng(seed + 1000 * sec_k + d); a_ = sec_k * np.pi / 4
    start = np.full((n, 2), 40); ipos = (start + np.array([round(float(np.round(d * np.cos(a_)))), round(float(np.round(d * np.sin(a_))))])) % L
    pos = start.copy(); sec = np.full(n, sec_k); done = np.zeros(n, bool); ate = np.zeros(n, bool); used = np.zeros(n, int)
    for t in range(1, 4 * d + 1):
        act = follow(pos, ipos, start, sec, clip, r if tie else None)
        if noise:
            rnd = r.random(n) < noise; act = np.where(rnd, r.integers(0, 6, n), act)
        e = ~done & np.all(pos == ipos, -1) & (act == EAT)
        ate |= e; used = np.where(~done, t, used); done |= e
        pos = np.where(done[:, None], pos, (pos + MOVES[act]) % L)
    return float(ate.mean()), float(used[ate].mean()) if ate.any() else None

arms = {"a_offset_only": (True, 0.0, False), "b_offset_only_noise10": (True, 0.1, False), "c_unclipped": (False, 0.0, False),
        "c2_unclipped_noise10": (False, 0.1, False), "d_offset_only_randtie": (True, 0.0, True)}
out = {}
for nm, (clip, noise, tie) in arms.items():
    print("==", nm)
    for d in (3, 9, 13, 17, 21):
        res = [run(k, d, clip, noise, tie) for k in range(8)]
        out[f"{nm}_d{d}"] = dict(reach=[round(x[0], 3) for x in res], steps=[None if x[1] is None else round(x[1], 1) for x in res])
        print(f"  d={d:2d} reach " + " ".join(f"{x[0]:.2f}" for x in res) + f"   card {np.mean([res[k][0] for k in (0,2,4,6)]):.2f} diag {np.mean([res[k][0] for k in (1,3,5,7)]):.2f}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
