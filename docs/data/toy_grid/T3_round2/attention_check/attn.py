import sys, numpy as np, flax.serialization, jax, jax.numpy as jnp
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy")
npz, params = sys.argv[1], sys.argv[2]
D = np.load(npz); X, H = D["x"], D["h"]
raw = flax.serialization.msgpack_restore(open(params, "rb").read())["params"]
P = lambda n: (np.asarray(raw[n]["kernel"]), np.asarray(raw[n]["bias"]))  # noqa: E731
(f1k, f1b), (qk, qb), (kk, kb) = P("f1"), P("q"), P("k")
Dx = X.shape[1]; tok = X[:, Dx - 18: Dx - 10].argmax(1); heard = X[:, Dx - 18: Dx - 10].sum(1) > 0
mem = H[:, :64 * 11].reshape(-1, 64, 11); ptr = H[:, -1].astype(int) % 64
filled = (np.abs(mem[:, :, -3:]).sum(-1) != 0) | (mem[:, :, :8].sum(-1) > 0)
z = np.tanh(X @ f1k + f1b); q = z @ qk + qb; k = mem @ kk + kb
att = np.einsum("bsd,bd->bs", k, q) / np.sqrt(32.0); att = np.where(filled, att, -1e9)
w = np.exp(att - att.max(1, keepdims=True)) * filled; w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
mtok = mem[:, :, :8].argmax(-1); same = filled & (mtok == tok[:, None]) & (mem[:, :, :8].sum(-1) > 0)
nf = filled.sum(1); ns = same.sum(1); last = (ptr - 1) % 64
sel = heard & (ns >= 1) & (nf >= 2)              # 2nd+ hearings with >= 2 records
ms = (w * same).sum(1)[sel]; cs = (ns / nf)[sel]
mr = w[np.arange(len(w)), last][sel]; cr = (1.0 / nf)[sel]
print("n=%d  same-token mass %.3f vs chance %.3f -> %.2fx   most-recent mass %.3f vs chance %.3f -> %.2fx   (per-step mean ratio same %.2fx)" % (
    sel.sum(), ms.mean(), cs.mean(), ms.mean() / cs.mean(), mr.mean(), cr.mean(), mr.mean() / cr.mean(), np.mean(ms / cs)))
rng = np.random.default_rng(0); otok = (tok + rng.integers(1, 8, len(tok))) % 8      # negative control: a random OTHER token
oth = filled & (mtok == otok[:, None]); no = oth.sum(1); sel2 = heard & (no >= 1) & (nf >= 2)
print("   control (random other token): mass %.3f vs chance %.3f -> %.2fx" % ((w * oth).sum(1)[sel2].mean(), (no / nf)[sel2].mean(), (w * oth).sum(1)[sel2].mean() / (no / nf)[sel2].mean()))
