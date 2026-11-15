"""The W2 named-item world, shared by necessity_w2.py and port_w2.py (Cam Nov 15 ~03:55).

One personal NAMED food item per agent, announced through its life code (token = code[sector], flag 1, offset to the
bout start / 9), at distance d from the bout start, VISIBLE ONLY WHEN ADJACENT, worth `size` energy, closed when eaten,
after 4d steps, or at death.

HUNGER-GATED CALLS (gate on): a bout (hearing + item) OPENS only when there is no open bout and E < H = E_set - size,
so every item meal is paid by the setpoint reward. Between bouts the hearing input is all zero and there is no item.
Gate off: a new bout opens at every close (the original necessity / oracle #1 design).

PAID INTAKE = energy gained below the setpoint: per step max(0, min(E_pre + gain, E_set) - min(E_pre, E_set)).
"""
import jax
import jax.numpy as jnp

from w2 import env
from w2.learnability import RecAgent

MOVES = env.MOVES; EAT = env.EAT; R = 3
OFF = jnp.stack(jnp.meshgrid(jnp.arange(-R, R + 1), jnp.arange(-R, R + 1), indexing="ij"), -1).reshape(-1, 2)


def torus(o, L):
    return (o + L // 2) % L - L // 2


def place(key, start, dfn, L):
    """Sector uniform 0..7 (from k1); distance dfn(k2) -> (N,) float (the original key use, so gate-off reproduces)."""
    k1, k2 = jax.random.split(key); N = start.shape[0]; sec = jax.random.randint(k1, (N,), 0, 8); d = dfn(k2)
    ang = sec * (jnp.pi / 4)
    ipos = (start + jnp.stack([jnp.round(d * jnp.cos(ang)), jnp.round(d * jnp.sin(ang))], -1).astype(jnp.int32)) % L
    return sec, d, ipos


def new_item(N):
    z = jnp.zeros(N)
    return dict(sec=jnp.zeros(N, jnp.int32), d=jnp.ones(N), pos=jnp.zeros((N, 2), jnp.int32), start=jnp.zeros((N, 2), jnp.int32),
                t=z, act=jnp.zeros(N, bool), es=z)


def open_bouts(item, s, key, dfn, L, want):
    """Open a bout where `want` (bool, N) and none is open."""
    sec, d, ipos = place(key, s.pos, dfn, L); o = want & ~item["act"]
    return dict(sec=jnp.where(o, sec, item["sec"]), d=jnp.where(o, d, item["d"]), pos=jnp.where(o[:, None], ipos, item["pos"]),
                start=jnp.where(o[:, None], s.pos, item["start"]), t=jnp.where(o, 0.0, item["t"]), act=item["act"] | o,
                es=jnp.where(o, s.energy, item["es"])), o


def close_and_open(item, on_item, dead, s3, key, dfn, L, gated, H):
    """One step of bout bookkeeping. on_item: ate the item this step; dead: died this step; s3: the state after the item
    energy and respawn. Returns the new item, closed (bool) and opened (bool)."""
    t2 = jnp.where(item["act"], item["t"] + 1.0, 0.0)
    closed = item["act"] & (on_item | (t2 >= 4 * item["d"]) | dead)
    item = dict(item, t=t2, act=item["act"] & ~closed)
    want = (s3.energy < H) if gated else jnp.ones(s3.energy.shape, bool)
    item, opened = open_bouts(item, s3, key, dfn, L, want)
    return item, closed, opened


def observe(cfg, s, item):
    o = env.observe(cfg, s); cell = (s.pos[:, None, :] + OFF[None]) % cfg.L
    vis = jnp.all(cell == item["pos"][:, None, :], -1) & (jnp.abs(OFF).max(-1) <= 1)[None] & item["act"][:, None]
    return o.at[:, :49].set(jnp.maximum(o[:, :49], vis.astype(jnp.float32)))


def hear(s, item, code, L):
    N = s.pos.shape[0]; tok = code[jnp.arange(N), item["sec"]]
    hr = jnp.concatenate([jax.nn.one_hot(tok, 8), jnp.ones((N, 1)), jnp.clip(torus(item["start"] - s.pos, L) / 9.0, -1, 1)], 1)
    return hr * item["act"][:, None]


def oracle(item, on):
    return jax.nn.one_hot(item["sec"], 8) * item["act"][:, None] * (1.0 if on else 0.0)


def follower(s, item, L, fallback):
    """Scripted true-code follower: during a bout, eat on the item, step to it when adjacent, eat world food it stands
    on, else walk the named sector's ray from the bout start. Outside a bout: `fallback` (v2-best's action)."""
    on_item = jnp.all(s.pos == item["pos"], -1); here = s.food[s.pos[:, 0], s.pos[:, 1]] >= 0.2
    adj = torus(item["pos"] - s.pos, L); seen = jnp.abs(adj).max(-1) <= 1
    tw = lambda dd: jnp.where(jnp.abs(dd[:, 0]) >= jnp.abs(dd[:, 1]), jnp.where(dd[:, 0] < 0, 1, 2), jnp.where(dd[:, 1] > 0, 3, 4))  # noqa: E731
    ang = item["sec"] * (jnp.pi / 4); u = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1); off = torus(s.pos - item["start"], L).astype(jnp.float32)
    cand = off[:, None, :] + MOVES[1:5][None].astype(jnp.float32)
    score = (cand * u[:, None]).sum(-1) - 2.0 * jnp.abs(cand[..., 0] * u[:, None, 1] - cand[..., 1] * u[:, None, 0])
    scripted = jnp.where(on_item, EAT, jnp.where(seen, tw(adj), jnp.where(here, EAT, jnp.argmax(score, 1) + 1)))
    return jnp.where(item["act"], scripted, fallback)


def paid(E_pre, gain, E_set):
    return jnp.maximum(0.0, jnp.minimum(E_pre + gain, E_set) - jnp.minimum(E_pre, E_set))


def build_v2best(nav, evo_pop0, Dx=126):
    """v2-best as in the harness: RecAgent(128), init seed 0, navigator trunk, the evolved top elite."""
    rnet = RecAgent(128); rp = rnet.init(jax.random.PRNGKey(0), jnp.zeros((128,)), jnp.zeros((Dx,)))
    pp = {k: dict(v) for k, v in rp["params"].items()}
    for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
        pp[k] = dict(nav["params"][k])
    pp.update(jax.tree_util.tree_map(jnp.asarray, evo_pop0)); return rnet, {"params": pp}
