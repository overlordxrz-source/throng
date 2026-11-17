"""Cam Nov 17 ~07:55Z L4 checks. (1) hear_wide vs NI.hear on random states: token / flag identical; offset = exact
torus offset / 24 (NI.hear's clipped value x 9 / 24 within 9 cells). (2) G2b follower (a) (named_item.follower's rule,
offset-only, no hidden counter) fed the WIDE input must reach 8/8 sectors at every d = 3..24; fed the ORIGINAL input it
must fail diagonals at long d (the defect); a MUTANT wide input (clip at 9 cells) must also fail (the test can fail)."""
import sys, types
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
from w2 import named_item as NI, offset_wide as OW
L = 96; EAT = 5; MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])
_a = jnp.arange(8) * (jnp.pi / 4); U32 = np.asarray(jnp.stack([jnp.cos(_a), jnp.sin(_a)], -1), np.float32)


def test_unit():
    r = np.random.default_rng(0); B = 5000
    st = r.integers(0, L, (B, 2)); pos = (st + r.integers(-48, 48, (B, 2))) % L; code = jnp.array(np.stack([r.permutation(8) for _ in range(B)]))
    item = dict(NI.new_item(B), sec=jnp.array(r.integers(0, 8, B), jnp.int32), start=jnp.array(st, jnp.int32), act=jnp.array(r.random(B) < 0.8))
    s = types.SimpleNamespace(pos=jnp.array(pos, jnp.int32))
    h0 = np.asarray(NI.hear(s, item, code, L)); h1 = np.asarray(OW.hear_wide(s, item, code, L)); act = np.asarray(item["act"])
    assert (h0[:, :9] == h1[:, :9]).all()
    exact = ((st - pos + L // 2) % L - L // 2) * act[:, None]
    assert np.allclose(h1[:, 9:11] * 24, exact, atol=1e-4)
    near = (np.abs(exact) <= 9).all(1)
    assert np.allclose(h1[near, 9:11] * 24, h0[near, 9:11] * 9, atol=1e-4)
    assert (np.abs(h1[:, 9:11]) <= 2.0 + 1e-6).all() and (~near).sum() > 1000
    return f"unit: {B} states; token/flag identical; offset exact/24 ({(~near).sum()} beyond 9 cells)"


def follow_from_input(hr, pos, ipos, sec, scale):
    """follower (a): the offset it steers by is decoded from the input alone: off = -hr[:, 9:11] * scale."""
    on = np.all(pos == ipos, -1); adj = (ipos - pos + L // 2) % L - L // 2; seen = np.abs(adj).max(-1) <= 1
    tw = np.where(np.abs(adj[:, 0]) >= np.abs(adj[:, 1]), np.where(adj[:, 0] < 0, 1, 2), np.where(adj[:, 1] > 0, 3, 4))
    u = U32[sec]; off = (-hr[:, 9:11] * np.float32(scale)).astype(np.float32)
    cand = off[:, None, :] + MOVES[1:5][None].astype(np.float32)
    score = (cand * u[:, None]).sum(-1) - np.float32(2.0) * np.abs(cand[..., 0] * u[:, None, 1] - cand[..., 1] * u[:, None, 0])
    return np.where(on, EAT, np.where(seen, tw, np.argmax(score, 1) + 1))


def reach_table(hear_fn, scale):
    """All 8 sectors x d = 3..24 in one batch (deterministic follower: one bout per cell)."""
    ds = np.arange(3, 25); S, D = np.meshgrid(np.arange(8), ds, indexing="ij"); S = S.ravel(); D = D.ravel(); B = len(S)
    st = np.full((B, 2), 40); ang = S * np.pi / 4
    ipos = (st + np.stack([np.round(D * np.cos(ang)), np.round(D * np.sin(ang))], -1).astype(int)) % L
    code = jnp.tile(jnp.arange(8)[None], (B, 1))
    item = dict(NI.new_item(B), sec=jnp.array(S, jnp.int32), start=jnp.array(st, jnp.int32), act=jnp.ones(B, bool))
    pos = st.copy(); done = np.zeros(B, bool)
    for t in range(1, 4 * 24 + 1):
        hr = np.asarray(hear_fn(types.SimpleNamespace(pos=jnp.array(pos, jnp.int32)), item, code, L))
        a = follow_from_input(hr, pos, ipos, S, scale)
        done |= (t <= 4 * D) & np.all(pos == ipos, -1) & (a == EAT)
        pos = np.where(done[:, None] | (t > 4 * D)[:, None], pos, (pos + MOVES[a]) % L)
    return done.reshape(8, len(ds)), ds


def mutant(s, item, code, L):
    h = OW.hear_wide(s, item, code, L)
    return h.at[:, 9:11].set(jnp.clip(h[:, 9:11], -9 / 24, 9 / 24))


if __name__ == "__main__":
    print(test_unit())
    W, ds = reach_table(OW.hear_wide, 24.0)
    print("WIDE: 8/8 at every d = 3..24:", bool(W.all()), " per-sector reached d:", W.sum(1).tolist(), "of", len(ds))
    assert W.all()
    O, _ = reach_table(NI.hear, 9.0)
    print("ORIGINAL (clip 9): failing (sector, d):", [(int(k), int(ds[j])) for k, j in zip(*np.nonzero(~O))][:40])
    assert not O.all() and O[[0, 2, 4, 6]].all()
    M, _ = reach_table(mutant, 24.0)
    print("MUTANT (wide but clipped at 9 cells) caught:", not M.all())
    assert not M.all()
    print("ALL PASS")
