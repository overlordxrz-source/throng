"""Cam Nov 18 ~16:15Z (M's card: "Yes, for stage 1 only"): the D4 TIE ON THE SIDE PATH (--d4-side).

The side path (everything that carries the sector input into the policy logits: side_enc on [x[100:], energy/age, trunk
view features] -> GRU -> side_head) is made D4-equivariant by EXACT group averaging:
    side_tied(x)[a] = mean_g side(g . x)[pi_g(a)]
run as 8 parallel streams (one per g, each with its own GRU hidden state, fed g-transformed inputs from life start), so
the recurrence is equivariant too. The trunk (Dense_0/1 -> lg_nav, v_nav) is unchanged and sees the untransformed view.
Which move goes with which sector is still learned; the tie only forbids a sector-PRIVATE side-path mapping within an
orbit (cardinals; diagonals).

g acts on: the two 7x7 view windows (grid permutation), the in-bout offset (signed 2x2 permutation), the previous action
(move permutation; stay / eat fixed), the oracle sector one-hot (sector permutation). Unchanged: energy, age, heard token
(per-life random code), bout flag, reward. Every transform is a signed index permutation: g . x = x[P_g] * S_g.

Conventions: positions are (x, y); sector k has direction (cos 45k, sin 45k); the view window index is i * 7 + j with
offset (i - 3, j - 3) (env.window). MOVES = [stay, -x, +x, +y, -y, eat] (+ 8-way: (1,1), (-1,1), (-1,-1), (1,-1)).
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

_R = np.array([[0, -1], [1, 0]]); _F = np.array([[1, 0], [0, -1]])
MATS = [np.linalg.matrix_power(_R, k) for k in range(4)] + [np.linalg.matrix_power(_R, k) @ _F for k in range(4)]   # MATS[0] = identity
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]                                    # sector k
MOVEV = [None, (-1, 0), (1, 0), (0, 1), (0, -1), None, (1, 1), (-1, 1), (-1, -1), (1, -1)]                       # action -> vector


def sector_perm(M):
    return np.array([DIRS.index(tuple(M @ np.array(d))) for d in DIRS])


def action_perm(M, n_act):
    out = []
    for a in range(n_act):
        v = MOVEV[a]
        out.append(a if v is None else MOVEV.index(tuple(M @ np.array(v))))
    return np.array(out)


def grid_src(M, r=3):
    """new window cell k (offset v_k) takes the old cell at M^-1 v_k."""
    Mi = np.round(np.linalg.inv(M)).astype(int); n = 2 * r + 1; src = []
    for i in range(n):
        for j in range(n):
            u = Mi @ np.array([i - r, j - r]); src.append((u[0] + r) * n + (u[1] + r))
    return np.array(src)


def tables(n_act=6, perm_override=None):
    """P (8, Dx), S (8, Dx), PI (8, n_act) [action perms], SIG (8, 8) [sector perms]. perm_override(g, P, S, PI, SIG):
    test hook to build a MUTANT."""
    Dx = 126 + (4 if n_act == 10 else 0); P = np.tile(np.arange(Dx), (8, 1)); S = np.ones((8, Dx)); PI = []; SIG = []
    pos = lambda a: 111 + a if a < 6 else 126 + (a - 6)  # noqa: E731   prev-action one-hot slots
    for g, M in enumerate(MATS):
        src = grid_src(M); P[g, 0:49] = src; P[g, 49:98] = 49 + src
        for i in range(2):                                                  # offset x[109:111]: new_i = sum_j M[i, j] old_j
            j = int(np.flatnonzero(M[i])[0]); P[g, 109 + i] = 109 + j; S[g, 109 + i] = M[i, j]
        pi = action_perm(M, n_act); pinv = np.argsort(pi)
        for b in range(n_act):                                              # one-hot(pi(a)): new slot b takes old slot pi^-1(b)
            P[g, pos(b)] = pos(pinv[b])
        sg = sector_perm(M); sinv = np.argsort(sg)
        for m in range(8):
            P[g, 118 + m] = 118 + sinv[m]
        PI.append(pi); SIG.append(sg)
    PI, SIG = np.array(PI), np.array(SIG)
    if perm_override is not None:
        P, S, PI, SIG = perm_override(P, S, PI, SIG)
    return P, S.astype(np.float32), PI, SIG


def transform(x, P, S):
    """(Dx,) -> (8, Dx): g . x for every g."""
    return x[..., P] * S


def untie(sg, PI):
    """(8, n_act) stream outputs -> (n_act,): tied[a] = mean_g sg[g, pi_g(a)]."""
    return sg[jnp.arange(PI.shape[0])[:, None], PI].mean(0)


class RecAgentD4(nn.Module):
    """RecAgent (w2/learnability.py) with the side path D4-tied by 8-stream group averaging. Same parameter names and
    shapes as RecAgent (a params file loads either way). Hidden state: 8 * hd (one GRU state per g; stream 0 = identity)."""
    hd: int = 128
    n_act: int = 6
    mutant: bool = False        # test only: a wrong permutation for one g (the unit test must catch it)

    @nn.compact
    def __call__(self, h, x):
        def mut(P, S, PI, SIG):
            PI = PI.copy(); PI[3, [1, 2]] = PI[3, [2, 1]]; return P, S, PI, SIG
        P, S, PI, _ = tables(self.n_act, mut if self.mutant else None)
        xg = transform(x, jnp.asarray(P), jnp.asarray(S))                   # (8, Dx)
        d0, d1 = nn.Dense(128, name="Dense_0"), nn.Dense(128, name="Dense_1")
        tg = nn.tanh(d1(nn.tanh(d0(xg[..., :100]))))                         # trunk features of g . view, (8, 128)
        t = nn.tanh(d1(nn.tanh(d0(x[..., :100]))))                           # the untransformed trunk (lg_nav, v_nav)
        lg_nav = nn.Dense(self.n_act, name="Dense_2")(t); v_nav = nn.Dense(1, name="Dense_3")(t)[..., 0]
        side = jnp.concatenate([xg[..., 100:], xg[..., 98:100]], -1)
        z = nn.tanh(nn.Dense(64, name="side_enc")(jnp.concatenate([side, tg], -1)))
        hg2, y = nn.GRUCell(self.hd, name="gru")(h.reshape(8, self.hd), z)
        zi = nn.initializers.zeros
        sh = nn.Dense(self.n_act, name="side_head", kernel_init=zi, bias_init=zi)(y)          # (8, n_act), stream g in g's frame
        lg = lg_nav + untie(sh, jnp.asarray(PI))
        v = v_nav + nn.Dense(1, name="side_value", kernel_init=zi, bias_init=zi)(y)[..., 0].mean(0)   # invariant: mean over streams
        meal = nn.Dense(9, name="aux_meal")(y[0]); other = nn.Dense(5, name="aux_other")(y[0])     # unused by port_w2's loss
        view = nn.Dense(49, name="aux_view")(jnp.concatenate([t, y[0]], -1))
        return hg2.reshape(-1), lg, v, (meal, other, view)
