"""W2 fast-learner prior (Cam Nov 10 ~02:25: APPROVED to build; docs/W2_FAST_LEARNER_DESIGN.md). Laptop first, cap $3.

RL^2-style: a GRU agent over [obs (100), heard (11: one-hot token + flag + speaker offset), previous action one-hot (6),
previous reward]. Its WEIGHTS are meta-trained across many lives; within a life only its hidden state adapts.
World = the learnability gate's beacon world (4 stationary immortal speakers per world, 4 learners). Each life the
speakers use a FRESH RANDOM CODE: token = sigma(sector), sigma a random permutation of the 8 food sectors, so no
particular mapping can be learned across lives -- only "pick up whatever code is in use".
Leak test: codes are split into two families by sigma(0)'s parity; meta-training uses family A only; the gate is run on
family B (held out). Gates (design doc):
  G1  within-life reliance: a life adapts for --adapt steps, then the state is branched into a true-code continuation
      and a cross-direction-swap continuation (300 steps each, same keys); food eaten drop >= 30% (paired > 2 SE over
      16 branches) in >= 50% of lives, held-out code family B
  G2  no-code control: a meta-learner trained with uninformative tokens shows reliance ~0
  G3  leak: reliance on family B ~= family A (within SE)
  G4  foraging kept: median life without any messages >= 0.9x the W2 navigators'
Lives here have a fixed length (--life steps, juvenile provisioning throughout) so that every life is a full
adaptation episode; the hidden state resets at each life start.
usage: python w2/fastlearner.py --updates 300 [--null-code] [--eval-only CKPT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402

NB, NL, R_HEAR, H = 4, 4, 9, 128
PERMS_A = PERMS_B = None


class Agent(nn.Module):
    @nn.compact
    def __call__(self, h, x):
        z = nn.tanh(nn.Dense(H, name="enc")(x))
        h, y = nn.GRUCell(H, name="gru")(h, z)
        return h, nn.Dense(6, name="pi")(y), nn.Dense(1, name="v")(y)[..., 0]


class SideMem(nn.Module):
    """Cam's fallback (Nov 10 ~03:55): the W2 navigator MLP is the policy trunk (its weights loaded; obs only), and a GRU
    side memory over [heard (11), previous action (6), previous reward (1)] adds to the policy logits through a
    ZERO-initialised head -- at init the agent IS the navigator; meta-training can only add message use."""
    @nn.compact
    def __call__(self, h, x):
        obs, side = x[..., :100], x[..., 100:]
        t = nn.tanh(nn.Dense(128, name="Dense_0")(obs)); t = nn.tanh(nn.Dense(128, name="Dense_1")(t))
        lg_nav = nn.Dense(6, name="Dense_2")(t); v = nn.Dense(1, name="Dense_3")(t)[..., 0]
        z = nn.tanh(nn.Dense(64, name="side_enc")(side))
        h, y = nn.GRUCell(H, name="gru")(h, z)
        lg = lg_nav + nn.Dense(6, name="side_head", kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(y)
        return h, lg, v


def sector(off):
    ang = jnp.arctan2(off[..., 1].astype(jnp.float32), off[..., 0].astype(jnp.float32))
    return jnp.mod(jnp.round(ang / (2 * jnp.pi / 8)).astype(jnp.int32), 8)


TOPICS = ["nearest_food_dir", "richest_patch_dir", "food_distance_band", "nearest_agent_dir", "speaker_energy_band", "noise",
          "second_nearest_patch_dir"]       # 6: Nov 10 candidate held-out topic (richest-patch cannot reach coverage)
USEFUL, USELESS = (0, 1, 2), (3, 4, 5)
HELD_TOPIC = 1                                   # held out of meta-training (Cam Nov 10 ~03:15)
TRAIN_TOPICS = (0, 2, 3, 4, 5)


def code_families(seed=0):
    """all 8! permutations is large; sample 4,000 random ones and split by parity of sigma(0)"""
    r = np.random.default_rng(seed)
    P = np.stack([r.permutation(8) for _ in range(4000)])
    return P[P[:, 0] % 2 == 0], P[P[:, 0] % 2 == 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300); ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--world", default="proto", help='"proto" or "d0,P,relocate" (Cam Nov 10 4b)')
    ap.add_argument("--beacon-spread", type=int, default=2, help="speakers at (+-k, +-k) around the cluster centre (Cam Nov 10 coverage)")
    ap.add_argument("--beacon-walk", action="store_true", help="speakers random-walk at agent speed")
    ap.add_argument("--beacon-speed", type=int, default=1, help="with --beacon-walk: unit random moves per step (Cam Nov 10: faster walk for richest-patch margin)")
    ap.add_argument("--freeze-trunk", action="store_true", help="Cam Nov 10 (4c): only the side memory (side_enc, gru, side_head) and the value head learn")
    ap.add_argument("--min-tokens", type=int, default=5, help="a listener-life qualifies iff it heard >= this many tokens >= --min-heard x during adaptation")
    ap.add_argument("--min-heard", type=int, default=25)
    ap.add_argument("--score", default="window", choices=["window", "last-heard"], help="last-heard (Cam Nov 10 metric): only food eaten while the last TRUE token heard is one the life heard >= --min-heard x in adaptation counts")
    ap.add_argument("--voi-fallback", default="policy", choices=["policy", "random"], help="scripted listener's move when it hears nothing and sees no food: the agent's own policy (navigator) or a random move")
    ap.add_argument("--scripted-voi", action="store_true", help="4c VoI: a scripted listener that decodes the life's code and follows the direction, true vs silence (direction topics)")
    ap.add_argument("--calib", default="", help="Nov 10: recalibrate the FL life test with the learnability design on these topic ids (e.g. 0,6); exit")
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--probe-window", type=int, default=300)
    ap.add_argument("--probe-fresh", action="store_true", help="each probe is a FRESH world with the same code (listeners start near the speakers, as in the learnability test), carrying the life's adapted hidden state")
    ap.add_argument("--coverage", action="store_true", help="diagnostic: distinct heard tokens per life (adapt + test), per topic")
    ap.add_argument("--T", type=int, default=64); ap.add_argument("--life", type=int, default=640)
    ap.add_argument("--adapt", type=int, default=300); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--null-code", action="store_true", help="G2 control: tokens carry no information")
    ap.add_argument("--eval-lives", type=int, default=64); ap.add_argument("--branches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="w2_out/fastlearner")
    ap.add_argument("--eval-only", default=None)
    ap.add_argument("--pilot-every", type=int, default=0, help="quick held-out gate every N updates (pilot trend)")
    ap.add_argument("--navs", default="w2_out/nav/*.msgpack", help="W2 navigators to clone (Cam Nov 10 ~03:55)")
    ap.add_argument("--bc-pretrain", type=int, default=0, help="phase 1: N updates of behaviour cloning (no messages)")
    ap.add_argument("--continue-on-fail", action="store_true")
    ap.add_argument("--arch", default="gru", choices=["gru", "sidemem"])
    ap.add_argument("--apparatus", action="store_true", help="Cam Nov 10 ~04:35: messages enter the side memory; no-message reliance ~0")
    ap.add_argument("--bc", type=float, default=0.0, help="phase 2: BC loss weight kept on during meta-training")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from w2.worlds import world_kw
    cfg = env.Cfg(L=96, N=NB + NL, patch_sigma=1.0, reproduce=False, A_max=100_000, **world_kw(a.world))
    W, T = a.worlds, a.T; net = SideMem() if a.arch == "sidemem" else Agent()
    D = 100 + 11 + 6 + 1
    params = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, H)), jnp.zeros((1, D)))
    FA, FB = code_families()
    import glob as _g
    nav_net = Net(); nt = nav_net.init(jax.random.PRNGKey(0), jnp.zeros((1, 100)))
    NAVP = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[flax.serialization.from_bytes(nt, open(f, "rb").read())
                                                            for f in sorted(_g.glob(a.navs))])
    KN = jax.tree_util.tree_leaves(NAVP)[0].shape[0]
    if a.arch == "sidemem":                         # load a navigator into the trunk
        pp = {k: dict(v) for k, v in params["params"].items()}
        nav0 = flax.serialization.from_bytes(nt, open(sorted(_g.glob(a.navs))[0], "rb").read())
        for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
            pp[k] = dict(nav0["params"][k])
        params = {"params": pp}
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)); ost = opt.init(params)

    def topic_values(s, key):
        """(6, NB): each speaker's value (0..7) for every topic"""
        bp = s.pos[:NB]
        d = (s.centres[None] - bp[:, None] + cfg.L // 2) % cfg.L - cfg.L // 2         # (NB, P, 2)
        dist = jnp.abs(d).sum(-1); near = jnp.argmin(dist, 1)
        mass = jnp.stack([env.window(cfg, s.food, s.centres[i][None], 2).sum() for i in range(cfg.P)])
        rich = jnp.argmax(mass)
        v0 = sector(d[jnp.arange(NB), near])
        v1 = sector(d[:, rich])
        v2 = jnp.clip(dist[jnp.arange(NB), near] // 6, 0, 7).astype(jnp.int32)
        dl = (s.pos[NB:][None] - bp[:, None] + cfg.L // 2) % cfg.L - cfg.L // 2          # (NB, NL, 2)
        v3 = sector(dl[jnp.arange(NB), jnp.argmin(jnp.abs(dl).sum(-1), 1)])
        v4 = jnp.clip((s.energy[:NB] * 8).astype(jnp.int32), 0, 7)
        v5 = jax.random.randint(key, (NB,), 0, 8)
        sec2 = jnp.argsort(dist, 1)[:, 1]                                                # second-nearest patch
        v6 = sector(d[jnp.arange(NB), sec2])
        return jnp.stack([v0, v1, v2, v3, v4, v5, v6])

    def new_world(key, perm):
        s = env.reset(cfg, key)
        c = jax.random.randint(jax.random.fold_in(key, 1), (2,), 0, cfg.L)
        k_ = a.beacon_spread
        bpos = (c[None] + jnp.array([[-k_, -k_], [-k_, k_], [k_, -k_], [k_, k_]])) % cfg.L
        lpos = (c[None] + jax.random.randint(jax.random.fold_in(key, 2), (NL, 2), -3, 4)) % cfg.L
        return s._replace(pos=jnp.concatenate([bpos, lpos], 0)), perm

    def hear(s, perm, shift, key, null, topic):
        sec = topic_values(s, key)[topic]
        tok = perm[jnp.mod(sec + shift, 8)]
        if null:
            tok = jax.random.randint(key, (NB,), 0, 8)
        lp, bp = s.pos[NB:], s.pos[:NB]
        d = (bp[None] - lp[:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
        cheb = jnp.abs(d).max(-1); j = jnp.argmin(cheb, 1); dj = d[jnp.arange(NL), j]
        ok = cheb[jnp.arange(NL), j] <= R_HEAR
        return jnp.concatenate([jax.nn.one_hot(tok[j], 8), jnp.ones((NL, 1)), dj / R_HEAR], 1) * ok[:, None]

    def scripted_act(s, hd, perm, fallback):
        """eats on food, heads for visible food, else walks the direction the heard token decodes to (through the
        life's own code: the scripted listener knows the mapping), else the fallback"""
        pos = s.pos[NB:]; here = s.food[pos[:, 0], pos[:, 1]]
        wv = env.window(cfg, s.food, pos, cfg.r).reshape(NL, -1); best = jnp.argmax(wv, 1); seen = wv.max(1) > 0.2
        o_ = jnp.arange(-cfg.r, cfg.r + 1)
        tw = lambda dx, dy: jnp.where(jnp.abs(dx) >= jnp.abs(dy), jnp.where(dx < 0, 1, 2), jnp.where(dy > 0, 3, 4))  # noqa: E731
        vis = tw(o_[best // (2 * cfg.r + 1)], o_[best % (2 * cfg.r + 1)])
        inv = jnp.argsort(perm); sec = inv[jnp.argmax(hd[:, :8], 1)]; ang = sec * (2 * jnp.pi / 8)
        hdg = tw(jnp.round(jnp.cos(ang) * 3), jnp.round(jnp.sin(ang) * 3))
        return jnp.where(here >= 0.2, env.EAT, jnp.where(seen, vis, jnp.where(hd[:, 8] > 0.5, hdg, fallback)))

    def step(p, s, perm, h, pa, pr, key, shift, null, topic, no_msg=False, expert=None, scripted=False, remap=None, rfb=None):
        k1, k2, k3 = jax.random.split(key, 3)
        hd = hear(s, perm, shift, k3, null, topic)
        if remap is not None:                      # Nov 10 calibration: per-listener token derangement (identity = true)
            tk_ = remap[jnp.arange(NL), jnp.argmax(hd[:, :8], 1)]
            hd = hd.at[:, :8].set(jax.nn.one_hot(tk_, 8) * hd[:, 8:9])
        if no_msg:
            hd = jnp.zeros_like(hd)
        x = jnp.concatenate([env.observe(cfg, s)[NB:], hd, jax.nn.one_hot(pa, 6), pr[:, None]], 1)
        h2, lg, v = net.apply(p, h, x)
        act = jax.random.categorical(k1, lg)
        if scripted:
            use_r = (a.voi_fallback == "random") if rfb is None else rfb
            fb = jax.random.randint(jax.random.fold_in(k1, 5), (NL,), 1, 5) if use_r else act
            act = scripted_act(s, hd, perm, fb)
        if expert is not None:                     # DAgger roll-in: the navigator drives, the clone learns on its states
            nlg, _ = nav_net.apply(expert, x[:, :100]); act = jax.random.categorical(jax.random.fold_in(k1, 3), nlg)
        lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act[:, None], 1)[:, 0]
        act_b = jax.random.randint(jax.random.fold_in(key, 77), (NB,), 1, 5) if a.beacon_walk else jnp.zeros(NB, jnp.int32)
        s2, r, info = env.step(cfg, s, jnp.concatenate([act_b, act]), k2)
        if a.beacon_walk and a.beacon_speed > 1:   # extra unit moves for the speakers only
            dirs = jnp.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
            extra = dirs[jax.random.randint(jax.random.fold_in(key, 78), (a.beacon_speed - 1, NB), 0, 4)].sum(0)
            s2 = s2._replace(pos=s2.pos.at[:NB].set((s2.pos[:NB] + extra) % cfg.L))
        E = s2.energy.at[:NB].set(1.0)
        E = E.at[NB:].set(jnp.maximum(E[NB:], 0.02))                                   # lives have a fixed length (no death)
        s2 = s2._replace(energy=E, alive=s2.alive.at[:].set(True))
        rr = r[NB:] * 20.0
        return s2, h2, act, rr, (x, act, lp, v, rr, info["take"][NB:])

    def roll(p, s, perm, topic, h, pa, pr, key, navp, use_expert):
        def body(c, k):
            s, h, pa, pr = c
            s2, h2, act, rr, rec = step(p, s, perm, h, pa, pr, k, 0, a.null_code, topic, expert=navp if use_expert else None)
            return (s2, h2, act, rr), rec + (h,)
        (s, h, pa, pr), R = jax.lax.scan(body, (s, h, pa, pr), jax.random.split(key, T))
        return s, h, pa, pr, R
    roll_w = jax.jit(jax.vmap(roll, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, None)), static_argnums=(9,))

    def loss(p, x, h0, act, lp0, adv, ret, navp, w_rl, w_bc):
        """recompute the GRU over the chunk from its stored start state (truncated BPTT over T); + BC (DAgger) toward the
        world's navigator's action distribution on the agent's own states (obs only, no messages)"""
        def body(h, xt):
            h2, lg, v = net.apply(p, h, xt)
            return h2, (lg, v)
        _, (lg, v) = jax.lax.scan(body, h0, x)                                           # x (T, NL, D)
        ls = jax.nn.log_softmax(lg); lp = jnp.take_along_axis(ls, act[..., None], -1)[..., 0]
        ratio = jnp.exp(lp - lp0)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        L = w_rl * (pg + 0.5 * (v - ret) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)).mean()
        nl, _ = nav_net.apply(navp, x[..., :100])
        L = L + w_bc * (-(jax.nn.softmax(nl) * ls).sum(-1)).mean()
        return L

    def _bc_parts(p, x, h0, navp):
        def body(h, xt):
            h2, lg, v = net.apply(p, h, xt)
            return h2, lg
        _, lg = jax.lax.scan(body, h0, x)
        nl, _ = nav_net.apply(navp, x[..., :100])
        return jax.nn.log_softmax(lg), nl

    @jax.jit
    def bc_ce(p, x, h0, navp):
        return jax.vmap(lambda x_, h_, n_: (lambda ls, nl: (-(jax.nn.softmax(nl) * ls).sum(-1)).mean())(*_bc_parts(p, x_, h_, n_)))(x, h0, navp).mean()

    @jax.jit
    def bc_agree(p, x, h0, navp):
        return jax.vmap(lambda x_, h_, n_: (lambda ls, nl: (ls.argmax(-1) == nl.argmax(-1)).mean())(*_bc_parts(p, x_, h_, n_)))(x, h0, navp).mean()

    @jax.jit
    def upd(p, ost, x, h0, act, lp0, adv, ret, navp, w_rl, w_bc):
        g = jax.grad(lambda q: jax.vmap(loss, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None, None))(q, x, h0, act, lp0, adv, ret, navp,
                                                                                             w_rl, w_bc).mean())(p)
        if a.freeze_trunk:
            g = {"params": {k: (jax.tree_util.tree_map(jnp.zeros_like, v) if k in ("Dense_0", "Dense_1", "Dense_2") else v)
                            for k, v in g["params"].items()}}
        u, ost = opt.update(g, ost, p); return optax.apply_updates(p, u), ost

    rng = np.random.default_rng(a.seed)

    def fresh(W_, fam):
        perms = jnp.asarray(fam[rng.integers(0, len(fam), W_)])
        tops = jnp.asarray(np.asarray(TRAIN_TOPICS)[rng.integers(0, len(TRAIN_TOPICS), W_)])   # a topic per life
        ks = jax.random.split(jax.random.PRNGKey(int(rng.integers(1 << 30))), W_)
        S, Pm = jax.vmap(new_world)(ks, perms)
        return S, Pm, tops, jnp.zeros((W_, NL, H)), jnp.zeros((W_, NL), jnp.int32), jnp.zeros((W_, NL))

    # ---- gates
    def life_test(p, perm, key, shift_mode, adapt, topic, no_msg=False):
        s, _ = new_world(key, perm)
        h = jnp.zeros((NL, H)); pa = jnp.zeros(NL, jnp.int32); pr = jnp.zeros(NL)
        def body(c, k):
            s, h, pa, pr = c
            s2, h2, act, rr, _ = step(p, s, perm, h, pa, pr, k, 0, a.null_code, topic, no_msg)
            return (s2, h2, act, rr), None
        (s, h, pa, pr), _ = jax.lax.scan(body, (s, h, pa, pr), jax.random.split(jax.random.fold_in(key, 1), adapt))
        sh = jax.random.randint(jax.random.fold_in(key, 9), (), 1, 8) * shift_mode
        def body2(c, k):
            s, h, pa, pr, eaten = c
            s2, h2, act, rr, rec = step(p, s, perm, h, pa, pr, k, sh, a.null_code, topic, no_msg)
            return (s2, h2, act, rr, eaten + rec[5].sum()), None
        (_, _, _, _, eaten), _ = jax.lax.scan(body2, (s, h, pa, pr, 0.0), jax.random.split(jax.random.fold_in(key, 2), 300))
        return eaten
    lt = jax.jit(jax.vmap(jax.vmap(life_test, in_axes=(None, None, 0, None, None, None, None)), in_axes=(None, 0, 0, None, None, None, None)),
                 static_argnums=(3, 4, 6))

    def life_test_l(p, perm, key, shift_mode, adapt, topic, no_msg=False, scripted=False):
        """per-listener version (Cam Nov 10): food eaten per listener in the 300-step test, and each listener's TRUE heard
        token counts during adaptation (qualification)"""
        s, _ = new_world(key, perm)
        h = jnp.zeros((NL, H)); pa = jnp.zeros(NL, jnp.int32); pr = jnp.zeros(NL)
        def body(c, k):
            s, h, pa, pr, cnt = c
            ht = hear(s, perm, 0, jax.random.fold_in(k, 3), a.null_code, topic)        # what the life truly hears
            cnt = cnt + ht[:, :8] * (ht[:, 8:9] > 0.5)
            s2, h2, act, rr, _ = step(p, s, perm, h, pa, pr, k, 0, a.null_code, topic, no_msg, scripted=scripted)
            return (s2, h2, act, rr, cnt), None
        (s, h, pa, pr, cnt), _ = jax.lax.scan(body, (s, h, pa, pr, jnp.zeros((NL, 8))), jax.random.split(jax.random.fold_in(key, 1), adapt))
        sh = jax.random.randint(jax.random.fold_in(key, 9), (), 1, 8) * shift_mode
        well = cnt >= a.min_heard                                                     # (NL, 8) well-heard tokens
        def body2(c, k):
            s, h, pa, pr, eaten, last = c
            ht = hear(s, perm, 0, jax.random.fold_in(k, 3), a.null_code, topic)
            last = jnp.where(ht[:, 8] > 0.5, jnp.argmax(ht[:, :8], 1), last)
            s2, h2, act, rr, rec = step(p, s, perm, h, pa, pr, k, sh, a.null_code, topic, no_msg, scripted=scripted)
            cr = jnp.where(last >= 0, well[jnp.arange(NL), jnp.maximum(last, 0)], False).astype(jnp.float32) if a.score == "last-heard" else 1.0
            return (s2, h2, act, rr, eaten + rec[5] * cr, last), None
        (_, _, _, _, eaten, _), _ = jax.lax.scan(body2, (s, h, pa, pr, jnp.zeros(NL), jnp.full(NL, -1, jnp.int32)),
                                                 jax.random.split(jax.random.fold_in(key, 2), 300))
        return eaten, cnt
    ltl = jax.jit(jax.vmap(jax.vmap(life_test_l, in_axes=(None, None, 0, None, None, None, None, None)), in_axes=(None, 0, 0, None, None, None, None, None)),
                  static_argnums=(3, 4, 6, 7))

    def gate_q(fam, n, tag, topic, cur=None, scripted=False):
        """qualifying-lives scoring: a listener-life qualifies iff it heard >= min-tokens tokens >= min-heard x during
        adaptation; pass = drop >= 30% AND paired > 2 SE over the branches; reported over qualifying lives"""
        p = params if cur is None else cur
        perms = jnp.asarray(fam[np.random.default_rng(99).integers(0, len(fam), n)])
        keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(5000 + i), a.branches))(jnp.arange(n))
        tru, cnt = (np.asarray(x) for x in ltl(p, perms, keys, 0, a.adapt, topic, False, scripted))   # (n, B, NL), (n, B, NL, 8)
        swp, _ = (np.asarray(x) for x in ltl(p, perms, keys, 1, a.adapt, topic, False, scripted))
        nom, _ = (np.asarray(x) for x in ltl(p, perms, keys, 0, a.adapt, topic, True, scripted))
        qe = (cnt >= a.min_heard).sum(-1) >= a.min_tokens                       # (n, B, NL): qualifying episodes
        oks, drops = [], []
        for i in range(n):
            t_, s_ = tru[i][qe[i]], swp[i][qe[i]]
            if len(t_) >= 4:
                dd = t_ - s_; dr = 1 - s_.sum() / max(t_.sum(), 1e-9)
                oks.append(bool(dr >= 0.30 and dd.mean() > 2 * dd.std(ddof=1) / np.sqrt(len(dd)))); drops.append(float(dr))
        tq, nq = tru[qe], nom[qe]; dv = tq - nq
        out = dict(family=tag, topic=TOPICS[topic], scripted=scripted, codes=n, episodes=int(qe.size), qualifying_episodes=int(qe.sum()),
                   qualifying_frac=float(qe.mean()), scorable_codes=len(oks), pass_frac_scorable=float(np.mean(oks)) if oks else None,
                   median_drop=float(np.median(drops)) if drops else None,
                   eaten_true=float(tq.mean()) if qe.any() else None, eaten_silence=float(nq.mean()) if qe.any() else None)
        if qe.sum() > 1:
            out["voi_ratio"] = float(tq.mean() / max(nq.mean(), 1e-9)); out["voi_paired_z"] = float(dv.mean() / max(dv.std(ddof=1) / np.sqrt(len(dv)), 1e-12))
            out["voi_ok"] = bool(out["voi_ratio"] >= 1.5 and out["voi_paired_z"] > 2)
        return out

    if a.scripted_voi:
        res = {TOPICS[t]: gate_q(FB, a.eval_lives, "B", t, scripted=True) for t in (0, 1)}
        print("SCRIPTED_VOI", json.dumps(res)); os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/scripted_voi.json", "w"), indent=1); return

    def gate(fam, n, tag, topic, quick=False, cur=None):
        p = params if cur is None else cur
        nb = 4 if quick else a.branches
        perms = jnp.asarray(fam[np.random.default_rng(99).integers(0, len(fam), n)])
        keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(5000 + i), nb))(jnp.arange(n))
        tru = np.asarray(lt(p, perms, keys, 0, a.adapt, topic, False)); swp = np.asarray(lt(p, perms, keys, 1, a.adapt, topic, False))
        out = dict(family=tag, topic=TOPICS[topic], lives=n, eaten_true=float(tru.mean()), eaten_swap=float(swp.mean()))
        d = tru - swp; drop = 1 - swp.sum(1) / np.maximum(tru.sum(1), 1e-9)
        se = d.std(1, ddof=1) / np.sqrt(d.shape[1]); ok = (drop >= 0.30) & (d.mean(1) > 2 * se)
        out.update(pass_frac=float(ok.mean()), median_drop=float(np.median(drop)))
        if not quick:
            nom = np.asarray(lt(p, perms, keys, 0, a.adapt, topic, True))          # no messages at all
            out.update(eaten_no_msg=float(nom.mean()), msg_vs_silence=float(tru.mean() / max(nom.mean(), 1e-9) - 1))
        return out
    def nav_life(navp, perm, key):
        s, _ = new_world(key, perm)
        def body(c, k):
            s, eaten = c
            k1, k2 = jax.random.split(k)
            lg, _ = nav_net.apply(navp, env.observe(cfg, s)[NB:]); act = jax.random.categorical(k1, lg)
            s2, _, info = env.step(cfg, s, jnp.concatenate([jnp.zeros(NB, jnp.int32), act]), k2)
            E = s2.energy.at[:NB].set(1.0).at[NB:].set(jnp.maximum(s2.energy[NB:], 0.02))
            s2 = s2._replace(energy=E, alive=s2.alive.at[:].set(True))
            return (s2, eaten + jnp.where(jnp.arange(1) >= 0, info["take"][NB:].sum(), 0.0)[0]), None
        (s, _), _ = jax.lax.scan(body, (s, 0.0), jax.random.split(jax.random.fold_in(key, 1), a.adapt))
        (s, eaten), _ = jax.lax.scan(body, (s, 0.0), jax.random.split(jax.random.fold_in(key, 2), 300))
        return eaten

    def intake_gate(p, n=64):
        """the agent (no messages) vs its reference navigator, same worlds and keys. For the side memory the reference is
        the navigator its trunk was loaded from (navigator 0); for a clone, the pool (one navigator per life)"""
        perms = jnp.asarray(FA[np.random.default_rng(7).integers(0, len(FA), n)])
        keys = jax.random.split(jax.random.PRNGKey(4242), n)
        idx = np.zeros(n, int) if a.arch == "sidemem" else np.arange(n) % KN
        navi = jax.tree_util.tree_map(lambda x: x[idx], NAVP)
        nv = np.asarray(jax.jit(jax.vmap(nav_life))(navi, perms, keys))
        fl = np.asarray(lt(p, perms, keys[:, None], 0, a.adapt, 0, True))[:, 0]           # no messages
        return dict(nav_median=float(np.median(nv)), fl_median=float(np.median(fl)),
                    ratio=float(np.median(fl) / max(np.median(nv), 1e-9)), **{"pass": bool(np.median(fl) >= 0.9 * np.median(nv))})


    if a.calib:
        """Cam Nov 10 (5): the learnability test design on the fast-learner harness. A LIFE = one listener's adaptation
        (--adapt steps, one world); then --probes branches of 300 steps from the SAME adapted state, true vs a
        derangement among that listener's well-heard tokens (>= --min-heard x during adaptation), food credited to the
        last heard true token if well-heard. Eligible iff >= --min-tokens well-heard tokens. Pass = drop >= 30% AND
        paired > 2 SE over the probes. Controls: positive = scripted decoder (>= 90%), negative = the untrained side
        memory, i.e. the navigator (message-blind) (<= 10%); VoI = scripted true vs silence, navigator AND random
        fallback (both >= 1.5x, paired > 2 SE)."""
        def adapt_fn(p, perm, key, topic, scripted, rfb):
            s, _ = new_world(key, perm)
            h = jnp.zeros((NL, H)); pa = jnp.zeros(NL, jnp.int32); pr = jnp.zeros(NL)
            def body(c, k):
                s, h, pa, pr, cnt = c
                ht = hear(s, perm, 0, jax.random.fold_in(k, 3), a.null_code, topic)
                cnt = cnt + ht[:, :8] * (ht[:, 8:9] > 0.5)
                s2, h2, act, rr, _ = step(p, s, perm, h, pa, pr, k, 0, a.null_code, topic, False, scripted=scripted, rfb=rfb)
                return (s2, h2, act, rr, cnt), None
            (s, h, pa, pr, cnt), _ = jax.lax.scan(body, (s, h, pa, pr, jnp.zeros((NL, 8))), jax.random.split(jax.random.fold_in(key, 1), a.adapt))
            return (s, h, pa, pr), cnt
        def probe_fn(p, perm, st, key, topic, remap, well, no_msg, scripted, rfb):
            s, h, pa, pr = st
            if a.probe_fresh:                          # a fresh episode with the same code and the life's food patches (as the
                s0 = s                                 # learnability test keeps the life's centres); the life's memory (h) carried in
                s, _ = new_world(jax.random.fold_in(key, 31), perm)
                s = s._replace(centres=s0.centres, cap=env.capacity(cfg, s0.centres), food=env.capacity(cfg, s0.centres))
            def body(c, k):
                s, h, pa, pr, eaten, last = c
                ht = hear(s, perm, 0, jax.random.fold_in(k, 3), a.null_code, topic)
                last = jnp.where(ht[:, 8] > 0.5, jnp.argmax(ht[:, :8], 1), last)
                s2, h2, act, rr, rec = step(p, s, perm, h, pa, pr, k, 0, a.null_code, topic, no_msg, scripted=scripted, remap=remap, rfb=rfb)
                cr = jnp.where(last >= 0, well[jnp.arange(NL), jnp.maximum(last, 0)], False).astype(jnp.float32)
                return (s2, h2, act, rr, eaten + rec[5] * cr, last), None
            (_, _, _, _, eaten, _), _ = jax.lax.scan(body, (s, h, pa, pr, jnp.zeros(NL), jnp.full(NL, -1, jnp.int32)), jax.random.split(key, a.probe_window))
            return eaten
        AD = jax.jit(jax.vmap(adapt_fn, in_axes=(None, 0, 0, None, None, None)), static_argnums=(3, 4, 5))
        PR = jax.jit(jax.vmap(jax.vmap(probe_fn, in_axes=(None, None, None, 0, None, 0, None, None, None, None)),
                              in_axes=(None, 0, 0, 0, None, 0, 0, None, None, None)), static_argnums=(4, 7, 8, 9))
        def wilson(k, n, z=1.96):
            if n == 0: return [None, None]
            ph = k / n; c = (ph + z * z / (2 * n)) / (1 + z * z / n); hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            return [float(c - hw), float(c + hw)]
        def life_pass(t, s_):
            d = t - s_; se = d.std(ddof=1) / np.sqrt(len(d))
            return bool((1 - s_.sum() / max(t.sum(), 1e-9)) >= 0.30 and d.mean() > 2 * se)
        n = a.eval_lives; B = a.probes; p_nav = params          # untouched side memory = the navigator (zero side head)
        perms = jnp.asarray(FB[np.random.default_rng(99).integers(0, len(FB), n)])
        wkeys = jax.random.split(jax.random.PRNGKey(8100), n)
        pkeys = jax.vmap(lambda k: jax.random.split(jax.random.fold_in(k, 77), B))(wkeys)
        out = {}
        for t in [int(x) for x in a.calib.split(",")]:
            R = {}
            for cond, pp, scr, rfb in (("scripted", p_nav, True, False), ("scripted_rfb", p_nav, True, True), ("nonlistener", p_nav, False, False)):
                st, cnt = AD(pp, perms, wkeys, t, scr, rfb); cnt = np.asarray(cnt)                   # (n, NL, 8)
                well = cnt >= a.min_heard; elig = well.sum(-1) >= a.min_tokens                          # (n, NL)
                rq = np.random.default_rng(4242); ident = np.tile(np.arange(8), (n, B, NL, 1)); der = ident.copy()
                for i in range(n):
                    for l in range(NL):
                        S_ = np.where(well[i, l])[0]
                        if len(S_) < 2: continue
                        for b in range(B):
                            while True:
                                q = rq.permutation(S_)
                                if np.all(q != S_): break
                            der[i, b, l, S_] = q
                W_ = jnp.asarray(well)
                tru = np.asarray(PR(pp, perms, st, pkeys, t, jnp.asarray(ident), W_, False, scr, rfb))     # (n, B, NL)
                swp = np.asarray(PR(pp, perms, st, pkeys, t, jnp.asarray(der), W_, False, scr, rfb))
                sil = np.asarray(PR(pp, perms, st, pkeys, t, jnp.asarray(ident), W_, True, scr, rfb)) if scr else None
                ok = np.array([[life_pass(tru[i, :, l], swp[i, :, l]) for l in range(NL)] for i in range(n)])
                ne = int(elig.sum()); k = int(ok[elig].sum()); gm = well.sum(-1)
                r = dict(lives=n * NL, eligible=ne, eligibility_rate=float(elig.mean()), pass_frac=float(k / ne) if ne else None, ci95=wilson(k, ne),
                         stratified={lab: dict(n=int((elig & m).sum()), pass_frac=float(ok[elig & m].mean()) if (elig & m).any() else None)
                                     for lab, m in (("5", gm == 5), ("6", gm == 6), ("7-8", gm >= 7))},
                         median_distinct=float(np.median((cnt > 0).sum(-1))), median_well=float(np.median(gm)))
                if sil is not None and ne > 1:
                    tm, sm = tru.mean(1)[elig], sil.mean(1)[elig]; dv = tm - sm
                    r["voi"] = dict(eaten_true=float(tm.mean()), eaten_silence=float(sm.mean()), ratio=float(tm.mean() / max(sm.mean(), 1e-9)),
                                    paired_z=float(dv.mean() / max(dv.std(ddof=1) / np.sqrt(len(dv)), 1e-12)))
                R[cond] = r
            R["gate"] = dict(positive_ok=bool(R["scripted"]["pass_frac"] is not None and R["scripted"]["pass_frac"] >= 0.90),
                             negative_ok=bool(R["nonlistener"]["pass_frac"] is not None and R["nonlistener"]["pass_frac"] <= 0.10),
                             voi_nav_ok=bool(R["scripted"].get("voi", {}).get("ratio", 0) >= 1.5 and R["scripted"].get("voi", {}).get("paired_z", 0) > 2),
                             voi_rand_ok=bool(R["scripted_rfb"].get("voi", {}).get("ratio", 0) >= 1.5 and R["scripted_rfb"].get("voi", {}).get("paired_z", 0) > 2),
                             enough_eligible=bool(R["scripted"]["eligible"] >= 50))
            R["gate"]["PASS"] = all(R["gate"].values())
            out[TOPICS[t]] = R
            print("CALIB", TOPICS[t], json.dumps(R), flush=True)
        os.makedirs(a.out, exist_ok=True); json.dump(out, open(f"{a.out}/calib.json", "w"), indent=1); return

    if a.coverage:
        def cov_life(p, perm, key, topic):
            s, _ = new_world(key, perm)
            h = jnp.zeros((NL, H)); pa = jnp.zeros(NL, jnp.int32); pr = jnp.zeros(NL)
            def body(c, k):
                s, h, pa, pr = c
                hd = hear(s, perm, 0, jax.random.fold_in(k, 7), a.null_code, topic)
                s2, h2, act, rr, _ = step(p, s, perm, h, pa, pr, k, 0, a.null_code, topic, False)
                return (s2, h2, act, rr), (hd[:, :8].argmax(1), hd[:, 8] > 0.5)
            _, (tok, okh) = jax.lax.scan(body, (s, h, pa, pr), jax.random.split(jax.random.fold_in(key, 1), a.adapt + 300))
            return tok, okh
        f = jax.jit(jax.vmap(cov_life, in_axes=(None, 0, 0, None)), static_argnums=(3,))
        n = 64; perms = jnp.asarray(FA[np.random.default_rng(5).integers(0, len(FA), n)]); keys = jax.random.split(jax.random.PRNGKey(31), n)
        out = {}
        for t in range(len(TOPICS)):
            tok, okh = (np.asarray(x) for x in f(params, perms, keys, t))            # (n, steps, NL)
            d = [len(np.unique(tok[i, :, l][okh[i, :, l]])) for i in range(n) for l in range(NL)]
            g25 = [int((np.bincount(tok[i, :, l][okh[i, :, l]], minlength=8) >= 25).sum()) for i in range(n) for l in range(NL)]
            out[TOPICS[t]] = dict(distinct_tokens_median=float(np.median(d)), tokens_ge25_median=float(np.median(g25)),
                                  hist=np.bincount(d, minlength=9).tolist(), heard_frac=float(okh.mean()),
                                  gate_i=bool(np.median(d) >= 4 and np.median(g25) >= 3))
        res = dict(world=a.world, steps=a.adapt + 300, beacon_spread=a.beacon_spread, beacon_walk=a.beacon_walk, beacon_speed=a.beacon_speed, per_topic=out,
                   gate_i_useful=all(out[TOPICS[t]]["gate_i"] for t in (0, 1, 2)))
        print("COVERAGE", json.dumps(res)); os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/coverage.json", "w"), indent=1); return

    if a.apparatus:
        p0 = params if a.eval_only is None else flax.serialization.from_bytes(params, open(a.eval_only, "rb").read())
        r_ = np.random.default_rng(0); x = jnp.asarray(r_.normal(size=(16, D)), jnp.float32); h = jnp.zeros((16, H))
        xa = x.at[:, 100:108].set(jax.nn.one_hot(jnp.zeros(16, jnp.int32), 8)); xb = x.at[:, 100:108].set(jax.nn.one_hot(jnp.full(16, 5), 8))
        ha, la, _ = net.apply(p0, h, xa); hb, lb, _ = net.apply(p0, h, xb)
        res = dict(heard_changes_gru_state=float(jnp.abs(ha - hb).max()) > 0, gru_state_diff=float(jnp.abs(ha - hb).max()),
                   logits_diff_at_this_params=float(jnp.abs(la - lb).max()))
        if a.arch == "sidemem":   # a nonzero side head makes the GRU difference reach the policy
            q = jax.tree_util.tree_map(lambda v: v, p0); q = {"params": dict(q["params"])}
            q["params"]["side_head"] = {"kernel": jnp.ones_like(p0["params"]["side_head"]["kernel"]) * 0.1, "bias": p0["params"]["side_head"]["bias"]}
            _, la2, _ = net.apply(q, h, xa); _, lb2, _ = net.apply(q, h, xb)
            res["heard_reaches_policy_with_nonzero_head"] = float(jnp.abs(la2 - lb2).max()) > 0
        n = 16; perms = jnp.asarray(FB[np.random.default_rng(3).integers(0, len(FB), n)])
        keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(6000 + i), 4))(jnp.arange(n))
        t0 = np.asarray(lt(p0, perms, keys, 0, a.adapt, HELD_TOPIC, True)); t1 = np.asarray(lt(p0, perms, keys, 1, a.adapt, HELD_TOPIC, True))
        res["no_message_true_vs_swap_identical"] = bool(np.allclose(t0, t1)); res["no_message_drop"] = float(1 - t1.sum() / max(t0.sum(), 1e-9))
        res["PASS"] = bool(res["heard_changes_gru_state"] and res["no_message_true_vs_swap_identical"]
                           and res.get("heard_reaches_policy_with_nonzero_head", True))
        print("APPARATUS", json.dumps(res)); return

    if a.eval_only:
        params = flax.serialization.from_bytes(params, open(a.eval_only, "rb").read())
    else:
        S, Pm, Tp, Hh, PA, PR = fresh(W, FA); t_life = 0; key = jax.random.PRNGKey(a.seed + 1); log = open(f"{a.out}/train.jsonl", "w")
        nav_w = jax.tree_util.tree_map(lambda x: x[rng.integers(0, KN, W)], NAVP)
        if a.bc_pretrain == 0:              # side memory: the intake gate on the untouched initial weights (agent = navigator)
            ig = intake_gate(params); print("intake gate", json.dumps(ig), flush=True)
            log.write(json.dumps(dict(update=0, intake_gate=ig)) + "\n"); log.flush()
            if not ig["pass"] and not a.continue_on_fail:
                print("STOP: intake gate failed"); return
        for it in range(a.bc_pretrain + a.updates):
            phase1 = it < a.bc_pretrain
            t0 = time.time()
            key, k = jax.random.split(key)
            roll_in = phase1 and it < int(0.67 * a.bc_pretrain)                         # expert roll-in, then the clone drives
            S, Hh, PA, PR, (x, act, lp, v, r, take, h0s) = roll_w(params, S, Pm, Tp, Hh, PA, PR, jax.random.split(k, W), nav_w, roll_in)
            v_, r_ = np.asarray(v), np.asarray(r)                                      # (W, T, NL)
            adv = np.zeros_like(r_); last = 0.0
            for t in reversed(range(T)):
                nv = v_[:, t + 1] if t + 1 < T else v_[:, t]
                last = r_[:, t] + 0.99 * nv - v_[:, t] + 0.99 * 0.95 * last; adv[:, t] = last
            ret = adv + v_; A = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(2):
                params, ost = upd(params, ost, x, h0s[:, 0], act, lp, jnp.asarray(A, jnp.float32), jnp.asarray(ret, jnp.float32),
                                  nav_w, 0.0 if phase1 else 1.0, 1.0 if phase1 else a.bc)
            t_life += T
            if t_life >= a.life:                                                        # every life ends: new code, new world
                S, Pm, Tp, Hh, PA, PR = fresh(W, FA); t_life = 0
                nav_w = jax.tree_util.tree_map(lambda x: x[rng.integers(0, KN, W)], NAVP)
            if a.bc_pretrain and it + 1 == a.bc_pretrain:                               # the intake gate before meta-training
                ig = intake_gate(params)
                log.write(json.dumps(dict(update=it + 1, intake_gate=ig)) + "\n"); log.flush(); print("intake gate", json.dumps(ig), flush=True)
                if not ig["pass"] and not a.continue_on_fail:
                    open(f"{a.out}/params_bc.msgpack", "wb").write(flax.serialization.to_bytes(params)); print("STOP: intake gate failed"); return
                open(f"{a.out}/params_bc.msgpack", "wb").write(flax.serialization.to_bytes(params))
            if it < a.bc_pretrain:
                if (it + 1) % 50 == 0:
                    ce = float(bc_ce(params, x, h0s[:, 0], nav_w)); agree = float(bc_agree(params, x, h0s[:, 0], nav_w))
                    nl_, _ = nav_net.apply(jax.tree_util.tree_map(lambda q: q[0], nav_w), x[0, ..., :100]); pn = jax.nn.softmax(nl_)
                    Hn = float((-(pn * jnp.log(pn + 1e-9)).sum(-1)).mean())
                    print(f"bc {it + 1}: CE {ce:.3f} (navigator entropy {Hn:.3f} -> KL ~{ce - Hn:.3f}) argmax agreement {agree:.3f} | eaten/step {float(np.asarray(take).mean()):.5f}", flush=True)
                continue
            if a.pilot_every and (it + 1 - a.bc_pretrain) % a.pilot_every == 0:
                g = gate(FB, 16, "B", HELD_TOPIC, quick=True)
                log.write(json.dumps(dict(update=it + 1, pilot_heldout=g)) + "\n"); log.flush()
                print("pilot", it + 1, json.dumps(g), flush=True)
            log.write(json.dumps(dict(update=it + 1, eaten_per_step=float(np.asarray(take).mean()), t=round(time.time() - t0, 2))) + "\n"); log.flush()
        open(f"{a.out}/params.msgpack", "wb").write(flax.serialization.to_bytes(params))

    per_topic = {TOPICS[t]: gate(FB, a.eval_lives, "B", t) for t in range(6)}
    gA = gate(FA, a.eval_lives, "A", HELD_TOPIC)
    held = per_topic[TOPICS[HELD_TOPIC]]
    useless_ok = all(abs(per_topic[TOPICS[t]]["msg_vs_silence"]) <= 0.05 for t in USELESS)
    res = dict(null_code=a.null_code, held_out=dict(topic=TOPICS[HELD_TOPIC], family="B"), per_topic=per_topic, leak_check_A=gA,
               G1_pass=held["pass_frac"] >= 0.5, G2_useless_unchanged=bool(useless_ok),
               G3_leak_ok=abs(gA["median_drop"] - held["median_drop"]) < 0.1)
    # diagnostics (Nov 10): in-distribution reliance (a training family + a useful training topic) and G4 foraging of the
    # TRAINED agent (no messages) vs its reference navigator(s), same worlds and keys
    res["in_dist_A_nearest_food"] = gate(FA, a.eval_lives, "A", TOPICS.index("nearest_food_dir"))
    ig = intake_gate(params); res["G4_intake_trained"] = ig; res["G4_pass"] = ig["pass"]
    print(json.dumps(res)); json.dump(res, open(f"{a.out}/gates{'_null' if a.null_code else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
