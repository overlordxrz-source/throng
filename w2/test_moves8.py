"""Cam Nov 17 ~13:00Z L2 (8-WAY MOVEMENT) checks. (1) env.step with Cfg.moves8: actions 0..5 move exactly as 4-way;
actions 6..9 are the diagonal steps (+1,+1), (-1,+1), (-1,-1), (+1,-1) and cost exactly what a cardinal step costs.
(2) A scripted SECTOR + OFFSET-ONLY follower (no counter: it steers by the CLIPPED /9 hearing input, NI.hear, and eats
when it stands on the visible item) with the 8-way table reaches 8/8 sectors at every d = 3..24 within the 4d timeout.
(3) Negative control: the same follower restricted to 4-way moves fails diagonals at long d (G2b's blindness).
(4) Mutant: an 8-way table whose sector-7 diagonal is wrong must fail (the test can fail)."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
from w2 import env, named_item as NI
L = 96; EAT = 5
M8 = np.asarray(env.MOVES8); _a = jnp.arange(8) * (jnp.pi / 4); U32 = np.asarray(jnp.stack([jnp.cos(_a), jnp.sin(_a)], -1), np.float32)


def test_env():
    cfg8 = env.Cfg(L=L, N=10, patch_sigma=1.0, reproduce=False, A_max=3000, moves8=True); cfg4 = cfg8._replace(moves8=False)
    s = env.reset(cfg8, jax.random.PRNGKey(0)); s = s._replace(food=jnp.zeros((L, L)), cap=jnp.zeros((L, L)), pos=jnp.full((10, 2), 50, jnp.int32), energy=jnp.full(10, 0.6))
    acts = jnp.arange(10)
    s8, _, _ = env.step(cfg8, s, acts, jax.random.PRNGKey(1)); s4, _, _ = env.step(cfg4, s, jnp.minimum(acts, 5), jax.random.PRNGKey(1))
    d8 = np.asarray(s8.pos) - 50; e8 = 0.6 - np.asarray(s8.energy)
    assert (d8[:6] == np.asarray(s4.pos)[:6] - 50).all(), "actions 0..5 differ from 4-way"
    assert d8[6:].tolist() == [[1, 1], [-1, 1], [-1, -1], [1, -1]], d8[6:].tolist()
    assert np.allclose(e8[6:], e8[2]) and np.allclose(e8[1:5], e8[2]) and e8[2] > e8[0] + 1e-6, e8
    for k, a_ in zip((1, 3, 5, 7), range(6, 10)):                    # each diagonal action points exactly at its sector
        assert np.allclose(M8[a_] / np.sqrt(2), U32[k], atol=1e-6)
    return f"env: actions 0..5 = 4-way; 6..9 = {d8[6:].tolist()}; energy cost per step cardinal {e8[2]:.4f} = diagonal {e8[6]:.4f} (stay {e8[0]:.4f})"


def follower(hr, pos, ipos, sec, moves):
    """sector + offset-only: candidates = the move set; the offset is decoded from the clipped input only (off = -hr*9)."""
    on = np.all(pos == ipos, -1); adj = (ipos - pos + L // 2) % L - L // 2; seen = np.abs(adj).max(-1) <= 1
    mv = np.asarray(moves[1:5] if len(moves) == 6 else np.concatenate([moves[1:5], moves[6:10]]))
    ids = np.array([1, 2, 3, 4] if len(moves) == 6 else [1, 2, 3, 4, 6, 7, 8, 9])
    step_to = np.array([ids[np.flatnonzero((mv == a_).all(1))[0]] if (mv == a_).all(1).any() else -1 for a_ in adj])
    u = U32[sec]; off = (-hr[:, 9:11] * np.float32(9.0)).astype(np.float32)
    cand = off[:, None, :] + mv[None].astype(np.float32)
    score = (cand * u[:, None]).sum(-1) - np.float32(2.0) * np.abs(cand[..., 0] * u[:, None, 1] - cand[..., 1] * u[:, None, 0])
    tw4 = np.where(np.abs(adj[:, 0]) >= np.abs(adj[:, 1]), np.where(adj[:, 0] < 0, 1, 2), np.where(adj[:, 1] > 0, 3, 4))
    near = np.where(step_to >= 0, step_to, tw4)                       # step onto the visible item (diagonally if allowed)
    return np.where(on, EAT, np.where(seen, near, ids[np.argmax(score, 1)]))


def reach_table(moves):
    ds = np.arange(3, 25); S, D = np.meshgrid(np.arange(8), ds, indexing="ij"); S = S.ravel(); D = D.ravel(); B = len(S)
    st = np.full((B, 2), 40); ang = S * np.pi / 4
    ipos = (st + np.stack([np.round(D * np.cos(ang)), np.round(D * np.sin(ang))], -1).astype(int)) % L
    code = jnp.tile(jnp.arange(8)[None], (B, 1))
    item = dict(NI.new_item(B), sec=jnp.array(S, jnp.int32), start=jnp.array(st, jnp.int32), act=jnp.ones(B, bool))
    pos = st.copy(); done = np.zeros(B, bool); used = np.zeros(B, int)
    for t in range(1, 4 * 24 + 1):
        hr = np.asarray(NI.hear(types.SimpleNamespace(pos=jnp.array(pos, jnp.int32)), item, code, L))
        a = follower(hr, pos, ipos, S, moves)
        live = ~done & (t <= 4 * D); used += live
        done |= live & np.all(pos == ipos, -1) & (a == EAT)
        pos = np.where(done[:, None] | (t > 4 * D)[:, None], pos, (pos + np.asarray(moves)[a]) % L)
    return done.reshape(8, len(ds)), ds, used.reshape(8, len(ds))


if __name__ == "__main__":
    print(test_env())
    R8, ds, used = reach_table(M8)
    print("8-WAY, clipped /9 offset: 8/8 at every d = 3..24:", bool(R8.all()), " reached per sector:", R8.sum(1).tolist(), "of", len(ds))
    print("  actions used at d = 3 / 12 / 21 (cardinal sec 0 | diagonal sec 7):", [int(used[0, list(ds).index(d)]) for d in (3, 12, 21)], "|", [int(used[7, list(ds).index(d)]) for d in (3, 12, 21)])
    assert R8.all()
    R4, _, _ = reach_table(np.asarray(env.MOVES))
    fail4 = [(int(k), int(ds[j])) for k, j in zip(*np.nonzero(~R4))]
    print("NEGATIVE CONTROL 4-way: failing (sector, d):", fail4[:12], "... n =", len(fail4))
    assert not R4.all() and R4[[0, 2, 4, 6]].all()
    MUT = M8.copy(); MUT[9] = [1, 1]                                   # sector 7's diagonal points the wrong way
    RM, _, _ = reach_table(MUT)
    print("MUTANT (action 9 = (+1,+1)) caught:", not RM.all(), " failing sectors:", sorted(set(int(k) for k in np.nonzero(~RM)[0])))
    assert not RM.all()
    print("ALL PASS")
