"""FUN F3 odometer probe (registered). Pairs (h2_t = the GRU state after step t's input = the dumped 'gru' at t + 1,
displacement since the hearing at t), in-bout steps t and t + 1 of the same bout and life. Subsets: (axis, sign) with
|disp_axis| > 9; target = |disp_axis|. Ridge probe per subset, trained on worlds 0-5, tested on worlds 6-7; held-out
R^2, n-weighted mean. Features: trained net's h2; the same inputs replayed through a RANDOM-INIT net of the same
architecture (PRNGKey 123; GRU state only: the memory never enters the GRU); the raw input x_t.
usage: f3.py npz kind(rec|ep) out.json"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy")
import jax, jax.numpy as jnp
from w2.learnability import RecAgent
from w2.ep_agent_dev import EpAgentDev
f, kind, outp = sys.argv[1:4]
z = dict(np.load(f)); Wn, T, N = z["hear"].shape; L = 96
G = z["gru"].astype(np.float32); X = z["x"].astype(np.float32)
if kind == "rec":
    net = RecAgent(128); Hd = 128
else:
    net = EpAgentDev(S=96, use_hind=True, shared_tok=True); Hd = 128 + 96 * 11 + 1
prm = net.init(jax.random.PRNGKey(123), jnp.zeros((Hd,)), jnp.zeros((X.shape[-1],)))
def replay(xs, dd):
    def body(h, xa):
        xx, d = xa; h2, _, _, _ = net.apply(prm, h, xx); return jnp.where(d, 0.0, h2), h2[:128]
    _, hs = jax.lax.scan(body, jnp.zeros(Hd), (xs, dd)); return hs
rp = jax.jit(replay)
RND = np.zeros_like(G)
for w in range(Wn):
    for n in range(N):
        RND[w, :, n] = np.asarray(rp(jnp.asarray(X[w, :, n]), jnp.asarray(z["dead"][w, :, n])))   # RND[t] = state AFTER input t
disp = ((z["pos"].astype(int) - z["start"].astype(int) + L // 2) % L) - L // 2
inb = z["inb"].astype(bool); tt = z["t"].astype(int); dead = z["dead"].astype(bool)
ok = np.zeros((Wn, T, N), bool)
ok[:, :-1] = inb[:, :-1] & inb[:, 1:] & (tt[:, 1:] == tt[:, :-1] + 1) & ~dead[:, :-1]
XA = X.copy(); XA[..., 98:100] = 0.0                                   # POST HOC (disclosed): raw input without energy / age
feats = {"trained": np.concatenate([G[:, 1:], np.zeros_like(G[:, :1])], 1),   # h2_t = dumped gru at t + 1
         "random_init": RND, "raw_input": X, "raw_no_energy_age": XA, "energy_age_only": X[..., 98:100]}
def r2(F, mask, y):
    wtr = np.zeros(Wn, bool); wtr[:6] = True
    tr = mask & wtr[:, None, None]; te = mask & ~wtr[:, None, None]
    if tr.sum() < 50 or te.sum() < 20: return None, int(te.sum())
    A, b = F[tr], y[tr]; mu, sd = A.mean(0), np.maximum(A.std(0), 0.05); A = (A - mu) / sd; B = (F[te] - mu) / sd
    Ab = np.c_[A, np.ones(len(A))]; best = None; idx = np.arange(len(A)) % 5          # ridge alpha by 5-fold CV on the TRAIN worlds only
    for al in (1.0, 10.0, 100.0, 1000.0):
        e = 0.0
        for k in range(5):
            m_ = idx != k; w_ = np.linalg.solve(Ab[m_].T @ Ab[m_] + al * np.eye(Ab.shape[1]), Ab[m_].T @ b[m_]); e += ((Ab[~m_] @ w_ - b[~m_]) ** 2).sum()
        best = (e, al) if best is None or e < best[0] else best
    w_ = np.linalg.solve(Ab.T @ Ab + best[1] * np.eye(Ab.shape[1]), Ab.T @ b); p = np.c_[B, np.ones(len(B))] @ w_; yt = y[te]
    return float(1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), int(te.sum())
out = dict(file=f, kind=kind, subsets={}, summary={})
for nm, F in feats.items():
    tot = 0.0; ntot = 0
    for ax in (0, 1):
        for sg in (1, -1):
            m = ok & (sg * disp[..., ax] > 9) & (np.abs(disp[..., ax]) <= 30); y = np.abs(disp[..., ax]).astype(float)   # <= 30: no torus wrap
            v, nte = r2(F, m, y); out["subsets"][f"{nm}_ax{ax}_{'+' if sg > 0 else '-'}"] = dict(r2=v, n_test=nte, n_all=int(m.sum()), y_range=[int(y[m].min()), int(y[m].max())] if m.any() else None)
            if v is not None: tot += v * nte; ntot += nte
    # a within-range reference: the same probe on |disp| <= 9 steps (where the input itself is unclipped)
    pt = 0.0; pn = 0
    for ax in (0, 1):
        for sg in (1, -1):
            m = ok & (sg * disp[..., ax] >= 1) & (sg * disp[..., ax] <= 9); y = np.abs(disp[..., ax]).astype(float)   # POSITIVE CONTROL range: unclipped
            v, nte = r2(F, m, y)
            if v is not None: pt += v * nte; pn += nte
    out["summary"][nm] = dict(r2_beyond9=round(tot / max(ntot, 1), 3), n_test=ntot, r2_within9=round(pt / max(pn, 1), 3))
print(json.dumps(out["summary"])); json.dump(out, open(outp, "w"), indent=1)
