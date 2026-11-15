"""Cam Nov 14: the TOY GRID (laptop only). One agent; one beacon at its start emitting a token that names the sector of
one food item at distance d. No parent, no top-up, no reflex. Same observation format as the W2 harness (7x7 food
window, 7x7 agent window, energy, age; then hear = token one-hot + flag + beacon offset, previous action, previous
reward; plus the oracle sector in tier T1), same reward (x20 drive reduction) and move cost, so results transfer.

Grid cell = (d, food visible|hidden, GAE lambda) x tier x learner:
  T1 oracle sector input | T2 fixed code across lives | T3 random code per life (episodes grouped into lives, memory)
  L1 harness learner: frozen navigator trunk + GRU side head (KL to the navigator 0.3 or 0) | L2 plain MLP (T1/T2) / GRU (T3)
Readout: reach-food rate within the 4d cap with the true input, a shuffled (always-wrong) input and silence.
BAR: true >= 80% AND >= 2x shuffled AND >= +30 points over shuffled.

Deviations (disclosed): hearing is global (the token is heard everywhere; the beacon offset is scaled by 24), so
T1/T2 need no memory; PPO is a re-implementation of the harness's loss and settings (clip 0.2, value 0.5, entropy
0.01, adam 3e-4 with clip-norm 1.0, 2 full-batch epochs, w2/gae.py), not the harness's own code path."""
import argparse, glob, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import optax
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2.learnability import RecAgent  # noqa: E402
from w2.learn import Net  # noqa: E402
from w2.gae import gae  # noqa: E402

L, R = 64, 3                      # torus size, view radius
D0, DMOVE, EFF, BITE, E_SET, E_START, A_MAX = 0.0013, 0.003, 0.5, 0.2, 0.8, 0.3, 3000
MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], jnp.int32); EAT = 5
START = jnp.array([L // 2, L // 2], jnp.int32)
OFF = jnp.stack(jnp.meshgrid(jnp.arange(-R, R + 1), jnp.arange(-R, R + 1), indexing="ij"), -1).reshape(-1, 2)   # (49, 2), window order


def drive(E):
    return (E_SET - E) ** 2


def food_cell(sec, d):
    ang = sec * (jnp.pi / 4)
    return (START + jnp.stack([jnp.round(d * jnp.cos(ang)), jnp.round(d * jnp.sin(ang))], -1).astype(jnp.int32)) % L


def torus(o):
    return (o + L // 2) % L - L // 2


class MLP(nn.Module):
    @nn.compact
    def __call__(self, h, x):
        z = nn.tanh(nn.Dense(64)(x)); z = nn.tanh(nn.Dense(64)(z))
        return h, nn.Dense(6)(z), nn.Dense(1)(z)[..., 0], None


class GRUNet(nn.Module):
    hd: int = 64
    @nn.compact
    def __call__(self, h, x):
        z = nn.tanh(nn.Dense(64)(x)); h, y = nn.GRUCell(self.hd)(h, z)
        return h, nn.Dense(6)(y), nn.Dense(1)(y)[..., 0], None


class TfNet(nn.Module):
    """Cam addendum 6, L2': a small causal transformer memory over a sliding window of the life (blank init). The state
    holds two cached buffers (inputs and layer-1 outputs, each with a 'filled' flag), so each step attends over the window
    in O(W) per layer (exact causal attention with a KV cache; the window slides)."""
    W: int = 128; dm: int = 40; nh: int = 4          # dm 40 ~ the GRU's parameter count

    def mha(self, q, kv, mask, name):
        dh = self.dm // self.nh
        Q = nn.Dense(self.dm, name=f"{name}_q")(q).reshape(self.nh, dh); K = nn.Dense(self.dm, name=f"{name}_k")(kv).reshape(-1, self.nh, dh)
        V = nn.Dense(self.dm, name=f"{name}_v")(kv).reshape(-1, self.nh, dh)
        att = jnp.einsum("hd,whd->hw", Q, K) / jnp.sqrt(dh); att = jnp.where(mask[None] > 0, att, -1e9)
        return nn.Dense(self.dm, name=f"{name}_o")(jnp.einsum("hw,whd->hd", jax.nn.softmax(att, -1), V).reshape(-1))

    @nn.compact
    def __call__(self, h, x):
        W, dm = self.W, self.dm; buf = h.reshape(2, W, dm + 1)
        pos = self.param("pos", nn.initializers.normal(0.02), (W, dm))
        e = nn.Dense(dm, name="emb")(x)
        B0 = jnp.concatenate([buf[0][1:], jnp.concatenate([e, jnp.ones(1)])[None]], 0)
        y = e + self.mha(nn.LayerNorm(name="ln1")(e + pos[-1]), nn.LayerNorm(name="ln1k")(B0[:, :dm] + pos), B0[:, -1], "a1")
        y = y + nn.Dense(dm, name="m1b")(nn.gelu(nn.Dense(2 * dm, name="m1a")(nn.LayerNorm(name="ln2")(y))))
        B1 = jnp.concatenate([buf[1][1:], jnp.concatenate([y, jnp.ones(1)])[None]], 0)
        z = y + self.mha(nn.LayerNorm(name="ln3")(y + pos[-1]), nn.LayerNorm(name="ln3k")(B1[:, :dm] + pos), B1[:, -1], "a2")
        z = z + nn.Dense(dm, name="m2b")(nn.gelu(nn.Dense(2 * dm, name="m2a")(nn.LayerNorm(name="ln4")(z))))
        return jnp.stack([B0, B1]).reshape(-1), nn.Dense(6, name="pi")(z), nn.Dense(1, name="v")(z)[0], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=12); ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tier", choices=["T1", "T2", "T3"], default="T1"); ap.add_argument("--learner", choices=["L1", "L1u", "L2", "L2t", "L3", "scripted", "ideal"], default="L1")
    ap.add_argument("--kl", type=float, default=0.3); ap.add_argument("--steps", type=float, default=1e6)
    ap.add_argument("--B", type=int, default=256); ap.add_argument("--T", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes-per-life", type=int, default=8, help="T3: episodes sharing one random code (memory carried within a life)")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--minibatches", type=int, default=1, help="harness default: 2 full-batch epochs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--evo-pop", type=int, default=32); ap.add_argument("--evo-elite", type=int, default=8); ap.add_argument("--evo-eps", type=int, default=64)
    ap.add_argument("--evo-sigma", type=float, default=0.3); ap.add_argument("--evo-sigma-body", type=float, default=0.02)
    ap.add_argument("--curriculum", action="store_true", help="round 2: d starts at 1; +1 when true-input reach over the last 500 episodes >= 80%%; cap --d-max")
    ap.add_argument("--curriculum-mix", action="store_true", help="EXPLORATORY (Will): with --curriculum, each episode's d ~ uniform{1..current d} (stops forgetting nearer d)")
    ap.add_argument("--d-max", type=int, default=24); ap.add_argument("--test-d", type=int, default=12, help="round 2: after training, reach at this d (true vs shuffled)")
    ap.add_argument("--save-params", default="", help="save the final weights (flax bytes)"); ap.add_argument("--skip-final-eval", action="store_true")
    ap.add_argument("--eval-params", default="", help="EVAL ONLY: load these weights and evaluate --eval-modes at --test-d")
    ap.add_argument("--eval-modes", default="0,3,4", help="0 true, 1 always-wrong sector, 2 silence, 3 within-life shuffled code (T3), 4 memory wiped each episode")
    ap.add_argument("--ckpt", default="", help="checkpoint path: resume if present; save every --ckpt-every updates")
    ap.add_argument("--ckpt-every", type=int, default=20); ap.add_argument("--max-seconds", type=float, default=0, help="stop (exit 3) after this long; rerun to resume")
    ap.add_argument("--move-cost", type=float, default=DMOVE, help="audit only: the per-move energy cost (harness 0.003)")
    ap.add_argument("--eval-episodes", type=int, default=512); ap.add_argument("--curve-every", type=int, default=10)
    ap.add_argument("--navs", default="w2_out/nav_w13/*.msgpack"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, cap, B, T = a.d, 4 * a.d, a.B, a.T
    rng = np.random.default_rng(a.seed); FIXED = jnp.asarray(rng.permutation(8))     # T2: one code for all lives
    Dx = 100 + 11 + 6 + 1 + (8 if a.tier == "T1" else 0)
    # ---- learner
    if a.learner in ("L1", "L1u", "L3"):            # L3 = evolution of L1's side_enc + GRU + side head (v2's machinery); trunk frozen
        net = RecAgent(128); Hd = 128
        p = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((Hd,)), jnp.zeros((Dx,)))
        nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob(a.navs))[0], "rb").read())["params"]
        pp = {k: dict(v) for k, v in p["params"].items()}
        for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
            pp[k] = dict(nav[k])
        p = {"params": pp}; FROZEN = () if a.learner == "L1u" else ("Dense_0", "Dense_1", "Dense_2", "Dense_3")   # L1u: trunk unfrozen
    elif a.learner == "L2t":                      # addendum 6: transformer memory (T3 only)
        net = TfNet(); Hd = 2 * net.W * (net.dm + 1)
        p = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((Hd,)), jnp.zeros((Dx,))); FROZEN = ()
    elif a.learner == "L2":
        net = (GRUNet() if a.tier == "T3" else MLP()); Hd = 64
        p = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((Hd,)), jnp.zeros((Dx,))); FROZEN = ()
    else:
        net, Hd, p, FROZEN = None, (17 if a.learner == "ideal" else 1), None, ()  # ideal: map (+1; 0 = unknown), per-token search pointer, current target (+1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)); ost = opt.init(p) if p is not None else None

    def code_of(life_code):
        return life_code                           # (B, 8): sector -> token
    def obs(pos, fpos, alive_f, E, t, sec, code, mode):
        """mode 0 true, 1 shuffled (always-wrong sector), 2 silence"""
        cell = (pos[:, None, :] + OFF[None]) % L
        isf = jnp.all(cell == fpos[:, None, :], -1) & alive_f[:, None]
        if a.hidden:
            isf = isf & (jnp.abs(OFF).max(-1) <= 1)[None]
        fw = isf.astype(jnp.float32)
        aw = (jnp.all(cell == pos[:, None, :], -1) | jnp.all(cell == START[None, None, :], -1)).astype(jnp.float32)   # self + beacon
        tok = code[jnp.arange(pos.shape[0]), sec]
        heard = jnp.where(mode == 2, 0.0, 1.0)[:, None]
        hear = jnp.concatenate([jax.nn.one_hot(tok, 8) * heard, heard, torus(START[None] - pos) / 24.0 * heard], 1)
        base = jnp.concatenate([fw, aw, E[:, None], (t / A_MAX)[:, None]], 1)
        return base, hear, sec

    def build_x(base, hear, osec, pa, pr, mode):
        parts = [base, hear, jax.nn.one_hot(pa, 6), pr[:, None]]
        if a.tier == "T1":
            parts.append(jax.nn.one_hot(osec, 8) * jnp.where(mode == 2, 0.0, 1.0)[:, None])
        return jnp.concatenate(parts, 1)

    def scripted_act(pos, fpos, sec, base):
        """positive control: walk along the named sector's ray; head for food once seen; eat on it"""
        on = jnp.all(pos == fpos, -1)
        fw = base[:, :49]; seen = fw.max(1) > 0.5; fo = OFF[jnp.argmax(fw, 1)]
        ang = sec * (jnp.pi / 4); u = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1); o = torus(pos - START).astype(jnp.float32)
        cand = o[:, None, :] + MOVES[1:5][None].astype(jnp.float32)
        score = (cand * u[:, None]).sum(-1) - 2.0 * jnp.abs(cand[..., 0] * u[:, None, 1] - cand[..., 1] * u[:, None, 0])
        ray = jnp.argmax(score, 1) + 1
        tw = jnp.where(jnp.abs(fo[:, 0]) >= jnp.abs(fo[:, 1]), jnp.where(fo[:, 0] < 0, 1, 2), jnp.where(fo[:, 1] > 0, 3, 4))
        return jnp.where(on, EAT, jnp.where(seen, tw, ray))

    def ideal_act(st, h, code, osec, base):
        """Cam's (a) ideal in-context learner. A known token -> walk to its sector point. An unknown token -> search the
        sector points at this episode's distance (it is told d, never the direction: a generous ceiling), RESUMING where
        this token's last search stopped and SKIPPING sectors already mapped to other tokens (the code is a permutation).
        Head for food once it is in view; eat on it. An empty known target is forgotten."""
        n = h.shape[0]; ar = jnp.arange(n); tok = code[ar, osec]
        known = h[:, :8] - 1.0; ptr = h[:, 8:16].astype(jnp.int32); cur = h[:, 16].astype(jnp.int32) - 1
        ks = known[ar, tok].astype(jnp.int32); isk = ks >= 0
        mapped = jnp.zeros((n, 8), bool).at[ar[:, None], jnp.maximum(known, 0).astype(jnp.int32)].set(known >= 0)   # sectors already assigned
        def next_free(s0):                                         # first unmapped sector at or after s0 (cyclic)
            cand = (s0[:, None] + jnp.arange(8)[None]) % 8; free = ~jnp.take_along_axis(mapped, cand, 1)
            return jnp.take_along_axis(cand, jnp.argmax(free, 1)[:, None], 1)[:, 0]
        s_now = jnp.where(cur < 0, next_free(ptr[ar, tok]), cur)
        tgt = jnp.where(isk, ks, s_now)
        delta = torus(food_cell(tgt, st["dd"]) - st["pos"]); at = jnp.all(delta == 0, -1)
        fw = base[:, :49]; seen = fw.max(1) > 0.5; fo = OFF[jnp.argmax(fw, 1)]; on = jnp.all(st["pos"] == st["fpos"], -1)
        mv = lambda dd_: jnp.where(jnp.abs(dd_[:, 0]) >= jnp.abs(dd_[:, 1]), jnp.where(dd_[:, 0] < 0, 1, 2), jnp.where(dd_[:, 1] > 0, 3, 4))  # noqa: E731
        act = jnp.where(on, EAT, jnp.where(seen, mv(fo), jnp.where(at, 0, mv(delta))))
        miss = at & ~seen & ~on
        known2 = known.at[ar, tok].set(jnp.where(miss & isk, -1.0, known[ar, tok]))
        cur2 = jnp.where(isk, cur, jnp.where(miss, next_free((tgt + 1) % 8), s_now))
        ptr2 = ptr.at[ar, tok].set(jnp.where(isk, ptr[ar, tok], cur2))                     # remember where this token's search is
        return act, jnp.concatenate([known2 + 1.0, ptr2.astype(jnp.float32), (cur2 + 1)[:, None].astype(jnp.float32)], 1)

    def env_reset(key, n, dval, mix=False):
        sec = jax.random.randint(key, (n,), 0, 8)
        dd = jnp.broadcast_to(jnp.asarray(dval, jnp.float32), (n,))
        if mix:                                                   # exploratory: d ~ uniform{1..dval} per episode
            dd = jnp.floor(jax.random.uniform(jax.random.fold_in(key, 9), (n,)) * dd) + 1.0
        return dict(pos=jnp.tile(START[None], (n, 1)), fpos=food_cell(sec, dd), alive_f=jnp.ones(n, bool), E=jnp.full(n, E_START), t=jnp.zeros(n),
                    sec=sec, pa=jnp.zeros(n, jnp.int32), pr=jnp.zeros(n), dd=dd)

    def env_step(st, act):
        mv = MOVES[act]; pos = (st["pos"] + mv) % L; moved = (jnp.abs(mv).sum(-1) > 0).astype(jnp.float32)
        on = jnp.all(st["pos"] == st["fpos"], -1) & st["alive_f"]
        ate = (act == EAT) & on
        E1 = jnp.clip(st["E"] + EFF * BITE * ate - (D0 + a.move_cost * moved), 0.0, 1.0)
        r = 20.0 * (drive(st["E"]) - drive(E1)); t = st["t"] + 1
        done = ate | (E1 <= 0.0) | (t >= 4 * st["dd"])                    # the cap is 4 x this episode's distance
        return dict(st, pos=jnp.where(ate[:, None], st["pos"], pos), E=E1, t=t, pa=act, pr=r, alive_f=st["alive_f"] & ~ate), r, done, ate

    # ---- one rollout of T steps over B envs (stochastic policy); resets on done; carries memory within an episode
    #      (T3: within a life of --episodes-per-life episodes)
    def policy(pp_, h, x):
        return jax.vmap(lambda hh, xx: net.apply(pp_, hh, xx))(h, x)

    def rollout(pp_, carry, key, mode, dval, mix=False):
        def body(c, k):
            st, h, code, ep_in_life = c
            k1, k2, k3, k4 = jax.random.split(k, 4)
            m = jnp.full(B, mode)
            osec = jnp.where(mode == 1, (st["sec"] + jax.random.randint(k3, (B,), 1, 8)) % 8, st["sec"])
            base, hear, _ = obs(st["pos"], st["fpos"], st["alive_f"], st["E"], st["t"], osec, code, m)
            x = build_x(base, hear, osec, st["pa"], st["pr"], m)
            if net is None and a.learner == "ideal":
                act, h2 = ideal_act(st, h, code, osec, base); lp = jnp.zeros(B); v = jnp.zeros(B)
            elif net is None:
                act = scripted_act(st["pos"], st["fpos"], osec, base); lp = jnp.zeros(B); v = jnp.zeros(B); h2 = h
            else:
                h2, lg, v, _ = policy(pp_, h, x); act = jax.random.categorical(k1, lg)
                lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act[:, None], 1)[:, 0]
            st2, r, done, ate = env_step(st, act)
            if net is None and a.learner == "ideal":                # record token -> the sector where food was found; reset the search
                ar_ = jnp.arange(B); tk_ = code[ar_, osec]; kn = h2[:, :8]
                kn = kn.at[ar_, tk_].set(jnp.where(ate, st["sec"].astype(jnp.float32) + 1.0, kn[ar_, tk_]))
                h2 = jnp.concatenate([kn, h2[:, 8:16], jnp.where(done, 0.0, h2[:, 16])[:, None]], 1)      # a new episode re-reads its token's pointer
                if os.environ.get("IDEAL_DEBUG"):
                    jax.debug.print("done={d} ate={e} tok={t} sec={s} known={k} t={tt} epi={ep}", d=done[0], e=ate[0], t=tk_[0], s=st["sec"][0], k=kn[0], tt=st["t"][0], ep=ep_in_life[0], ordered=True)
            # reset finished episodes; T3: a new life (new code, memory wiped) every episodes_per_life episodes
            ep2 = ep_in_life + done.astype(jnp.int32); newlife = done & (ep2 >= (a.episodes_per_life if a.tier == "T3" else 1))
            fresh = env_reset(k2, B, dval, mix=mix)
            st3 = jax.tree_util.tree_map(lambda u, w: jnp.where(done.reshape((-1,) + (1,) * (u.ndim - 1)), w, u), st2, fresh)
            if a.tier == "T3":
                nc = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(k4, B))
                redraw = done if mode == 3 else newlife                 # mode 3: within-life shuffled code (a new code every episode)
                wipe = done if mode == 4 else newlife                   # mode 4: memory wiped at every episode start
                code2 = jnp.where(redraw[:, None], nc, code); h3 = jnp.where(wipe[:, None], 0.0, h2); ep3 = jnp.where(newlife, 0, ep2)
            else:
                code2 = code; h3 = jnp.where(done[:, None], 0.0, h2); ep3 = ep2 * 0
            reset = (newlife if a.tier == "T3" else done).astype(jnp.float32)      # when the rollout wiped the memory
            return (st3, h3, code2, ep3), (x, act, lp, v, r, done.astype(jnp.float32), ate, h, reset, ep_in_life)
        return jax.lax.scan(body, carry, jax.random.split(key, T))
    roll = jax.jit(rollout, static_argnums=(3, 5))

    def seq_loss(pp_, x, h0, act, lp0, adv, ret, first):
        def body(h, xa):
            xx, f = xa; h = jnp.where(f > 0, 0.0, h)              # same resets as the rollout (episode / life starts)
            h2, lg, v, _ = net.apply(pp_, h, xx); return h2, (lg, v)
        _, (lg, v) = jax.lax.scan(body, h0, (x, first))
        ls = jax.nn.log_softmax(lg); lp = jnp.take_along_axis(ls, act[:, None], 1)[:, 0]
        ratio = jnp.exp(jnp.clip(lp - lp0, -20.0, 20.0)); pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        L_ = (pg + 0.5 * (v - ret) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)).mean()
        if a.learner == "L1" and a.kl > 0:                     # KL to the frozen navigator (trunk alone)
            P_ = pp_["params"]; tn = jnp.tanh(x[:, :100] @ P_["Dense_0"]["kernel"] + P_["Dense_0"]["bias"])
            lsn = jax.nn.log_softmax(jnp.tanh(tn @ P_["Dense_1"]["kernel"] + P_["Dense_1"]["bias"]) @ P_["Dense_2"]["kernel"] + P_["Dense_2"]["bias"])
            L_ = L_ + a.kl * (jnp.exp(ls) * (ls - lsn)).sum(-1).mean()
        return L_

    @jax.jit
    def upd(pp_, os_, x, h0, act, lp0, adv, ret, first):
        def tot(q):
            return jax.vmap(seq_loss, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(q, x, h0, act, lp0, adv, ret, first).mean()
        g = jax.grad(tot)(pp_)
        g = {"params": {k_: (jax.tree_util.tree_map(jnp.zeros_like, v_) if k_ in FROZEN else v_) for k_, v_ in g["params"].items()}}
        u, os_ = opt.update(g, os_, pp_); return optax.apply_updates(pp_, u), os_

    LATE, EARLY = {}, {}

    def evaluate(pp_, mode, seed, dval=None):
        """reach-food rate over --eval-episodes complete episodes (stochastic policy) at distance dval (default --d)"""
        dval = float(a.d if dval is None else dval)
        key = jax.random.PRNGKey(seed); st = env_reset(key, B, dval); h = jnp.zeros((B, Hd))
        code = (jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(key, B)) if a.tier == "T3" else jnp.tile(FIXED[None], (B, 1)))
        c = (st, h, code, jnp.zeros(B, jnp.int32)); n_ep = 0; n_ok = 0; late_ep = 0; late_ok = 0; early_ep = 0; early_ok = 0
        while n_ep < a.eval_episodes:
            key, k = jax.random.split(key); c, R_ = roll(pp_, c, k, mode, jnp.float32(dval))
            dn, ate, epi = np.asarray(R_[5]), np.asarray(R_[6]), np.asarray(R_[9]); n_ep += int(dn.sum()); n_ok += int((ate & (dn > 0)).sum())
            lt = (dn > 0) & (epi >= a.episodes_per_life // 2); late_ep += int(lt.sum()); late_ok += int((ate & lt).sum())
            et = (dn > 0) & (epi < a.episodes_per_life // 2); early_ep += int(et.sum()); early_ok += int((ate & et).sum())
        LATE[(mode, dval)] = late_ok / max(late_ep, 1); EARLY[(mode, dval)] = early_ok / max(early_ep, 1)   # T3: halves of each life
        return n_ok / max(n_ep, 1)

    out = dict(cfg=vars(a), cap=cap, curve=[])
    if a.eval_params or a.learner == "ideal":                       # EVAL ONLY (Cam T3 controls): per mode, overall / first / second half
        if a.eval_params:
            p = flax.serialization.from_bytes(p, open(a.eval_params, "rb").read())
        res = {}
        for mo in [int(z) for z in a.eval_modes.split(",")]:
            ov = evaluate(p, mo, 779, a.test_d); res[str(mo)] = dict(overall=ov, first_half=EARLY.get((mo, float(a.test_d))), second_half=LATE.get((mo, float(a.test_d))))
            print("EVALMODE", mo, json.dumps(res[str(mo)]), flush=True)
        out["eval_modes"] = res; json.dump(out, open(a.out, "w"), indent=1); return
    if a.learner == "L3":
        GR = ("side_enc", "gru", "side_head"); P_, E_, er = a.evo_pop, a.evo_eps, np.random.default_rng(a.seed + 7)
        SIGG = {g: (a.evo_sigma if g == "side_head" else a.evo_sigma_body) for g in GR}
        base = {g: p["params"][g] for g in GR}
        def mut(m):
            return {g: jax.tree_util.tree_map(lambda w: w + jnp.asarray(er.normal(0, SIGG[g], w.shape), jnp.float32), m[g]) for g in GR}
        pop = [base] + [mut(base) for _ in range(P_ - 1)]
        def member_eps(mp, key, dval):
            """one member's E_ episodes in parallel (shared keys across members = common random numbers); reach rate"""
            pp_ = {"params": {**p["params"], **mp}}
            st = env_reset(key, E_, float(dval)); h = jnp.zeros((E_, Hd)); code = jnp.tile(FIXED[None], (E_, 1)); done = jnp.zeros(E_, bool); ok = jnp.zeros(E_, bool)
            def body(c, k):
                st, h, done, ok = c
                base_, hear, _ = obs(st["pos"], st["fpos"], st["alive_f"], st["E"], st["t"], st["sec"], code, jnp.zeros(E_, jnp.int32))
                x = build_x(base_, hear, st["sec"], st["pa"], st["pr"], jnp.zeros(E_, jnp.int32))
                h2, lg, _, _ = jax.vmap(lambda hh, xx: net.apply(pp_, hh, xx))(h, x); act = jax.random.categorical(k, lg)
                st2, r, dn, ate = env_step(st, act)
                keep = done[:, None]
                st3 = jax.tree_util.tree_map(lambda u, w: jnp.where(done.reshape((-1,) + (1,) * (u.ndim - 1)), u, w), st, st2)
                return (st3, jnp.where(keep, h, h2), done | dn, ok | (ate & ~done)), (~done).sum()
            (_, _, _, ok), live = jax.lax.scan(body, (st, h, done, ok), jax.random.split(jax.random.fold_in(key, 1), 4 * dval))
            return ok.mean(), live.sum()
        score = jax.jit(jax.vmap(member_eps, in_axes=(0, None, None)), static_argnums=(2,))
        d_cur = 1 if a.curriculum else a.d
        steps, gen, t0 = 0, 0, time.time()
        out["eval_init"] = {m_: evaluate(p, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))}
        while steps < a.steps:
            stk = jax.tree_util.tree_map(lambda *ls: jnp.stack(ls), *pop)
            fit, live = score(stk, jax.random.PRNGKey(10_000 + gen), d_cur); fit = np.asarray(fit); steps += int(np.asarray(live).sum())
            order = np.argsort(-fit); elites = [pop[i] for i in order[:a.evo_elite]]
            out["curve"].append(dict(gen=gen, steps=steps, d=d_cur, best=float(fit.max()), elite_mean=float(fit[order[:a.evo_elite]].mean()), mean=float(fit.mean())))
            if a.curriculum and fit[order[:a.evo_elite]].mean() >= 0.8 and d_cur < a.d_max:   # the 8 elites' 512 episodes
                d_cur += 1
            pop = elites + [mut(elites[er.integers(a.evo_elite)]) for _ in range(P_ - a.evo_elite)]; gen += 1
        p = {"params": {**p["params"], **elites[0]}}
        out["eval"] = {m_: evaluate(p, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))}
        out["train_seconds"] = round(time.time() - t0, 1); out["generations"] = gen
    elif net is None:                                             # the scripted positive control: evaluation only
        out["eval"] = {m_: evaluate(None, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))}
    else:
        d_cur = 1 if a.curriculum else a.d; recent = []
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")); import ckpt as _ck  # noqa: E401
        key = jax.random.PRNGKey(a.seed + 1); st = env_reset(key, B, float(d_cur))
        code0 = (jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(key, B)) if a.tier == "T3" else jnp.tile(FIXED[None], (B, 1)))
        c = (st, jnp.zeros((B, Hd)), code0, jnp.zeros(B, jnp.int32)); steps, it, t0 = 0, 0, time.time()
        Ck = _ck.load(a.ckpt) if a.ckpt else None
        if Ck is not None:                                          # resume
            tj = lambda t_: jax.tree_util.tree_map(jnp.asarray, t_)  # noqa: E731
            p, ost, c, key = tj(Ck["p"]), tj(Ck["ost"]), tj(Ck["c"]), jnp.asarray(Ck["key"])
            steps, it, d_cur, recent, out["curve"], out["eval_init"] = Ck["steps"], Ck["it"], Ck["d_cur"], Ck["recent"], Ck["curve"], Ck["eval_init"]
            rng.bit_generator.state = Ck["rng"]; print(f"resumed at {steps:,} steps, d {d_cur}", flush=True)
        else:
            out["eval_init"] = ({m_: evaluate(p, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))} if a.tier != "T3" else None)
        t_start = time.time()
        while steps < a.steps:
            if a.ckpt and it % a.ckpt_every == 0 and it > 0:
                _ck.save(a.ckpt, dict(p=p, ost=ost, c=c, key=key, steps=steps, it=it, d_cur=d_cur, recent=recent, curve=out["curve"],
                                      eval_init=out["eval_init"], rng=rng.bit_generator.state))
                if a.max_seconds and time.time() - t_start > a.max_seconds:
                    print(f"CHUNK_STOP at {steps:,} steps", flush=True); sys.exit(3)
            key, k = jax.random.split(key); c, R_ = roll(p, c, k, 0, jnp.float32(d_cur), a.curriculum_mix)
            x, act, lp, v, r, dn, ate, hin, rs = (np.asarray(z) for z in R_[:9])    # (T, B, ...)
            adv = gae(np.swapaxes(r, 0, 1), np.swapaxes(v, 0, 1), np.swapaxes(dn, 0, 1), a.gamma, a.lam)            # (B, T)
            ret = adv + np.swapaxes(v, 0, 1); A_ = (adv - adv.mean()) / (adv.std() + 1e-8)
            first = np.concatenate([np.zeros((1, B)), rs[:-1]], 0)                  # wipe memory exactly where the rollout did
            sw = lambda z: jnp.asarray(np.swapaxes(z, 0, 1))  # noqa: E731
            Bx = (sw(x), jnp.asarray(hin[0]), sw(act), sw(lp), jnp.asarray(A_.astype(np.float32)), jnp.asarray(ret.astype(np.float32)), sw(first))
            for _e in range(a.epochs):                               # minibatches over whole sequences (envs)
                perm = rng.permutation(B)
                for mb in np.array_split(perm, a.minibatches):
                    p, ost = upd(p, ost, *(z[mb] for z in Bx))
            steps += B * T; it += 1
            if a.curriculum:                                                        # episodes that ended this rollout, at the current d
                recent.extend((ate[dn > 0]).astype(int).tolist()); recent = recent[-500:]
                if len(recent) >= 500 and np.mean(recent) >= 0.8 and d_cur < a.d_max:
                    d_cur += 1; recent = []
            if it % a.curve_every == 0:
                actf = np.bincount(act.reshape(-1), minlength=6) / act.size; at = np.swapaxes(act, 0, 1)
                out["curve"].append(dict(steps=steps, d=d_cur, reach_train=float(ate.sum() / max(dn.sum(), 1)), sec=round(time.time() - t0, 1),
                                         act_freq=[round(float(q), 3) for q in actf], adv_by_act=[round(float(adv[at == k].mean()), 3) if (at == k).any() else None for k in range(6)],
                                         ret_mean=round(float(ret.mean()), 3), r_eat=round(float(r[ate].mean()), 3) if ate.any() else None))
        out["train_seconds"] = round(time.time() - t0, 1)
        if a.save_params:
            open(a.save_params, "wb").write(flax.serialization.to_bytes(p))
        if a.skip_final_eval:
            out["eval"] = {"true": 0, "shuffled": 0, "silence": 0}; out["d_max_reached"] = int(max([c_.get("d", 1) for c_ in out["curve"]] + [1]))
            json.dump(out, open(a.out, "w"), indent=1); print("TRAINED", json.dumps(dict(d_max=out["d_max_reached"], sec=out["train_seconds"])), flush=True); return
        out["eval"] = (None if a.tier == "T3" else {m_: evaluate(p, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))})
    if net is not None:
        out["d_max_reached"] = int(max([c_.get("d", a.d) for c_ in out["curve"]] + [a.d if not a.curriculum else 1]))
        out["test"] = {m_: evaluate(p, i, 778, a.test_d) for i, m_ in enumerate(("true", "shuffled") if a.tier != "T3" else ("true", "shuffled", "silence"))}
        if out["eval"] is None:                                     # T3: whole-life evaluation done once, at the test d
            out["eval"] = out["test"]
        if a.tier == "T3":
            out["test_late_life"] = {m_: LATE.get((i, float(a.test_d))) for i, m_ in enumerate(("true", "shuffled"))}
        tt = out["test"]; out["test_pass"] = bool(tt["true"] >= 0.8 and tt["true"] >= 2 * tt["shuffled"] and tt["true"] - tt["shuffled"] >= 0.30)
    e = out["eval"]
    out["learns"] = bool(e["true"] >= 0.8 and e["true"] >= 2 * e["shuffled"] and e["true"] - e["shuffled"] >= 0.30)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print("TOY", json.dumps(dict(d=d, hidden=a.hidden, d_max=out.get("d_max_reached"), test=out.get("test"), test_pass=out.get("test_pass"), lam=a.lam, tier=a.tier, learner=a.learner, kl=a.kl, eval=e, learns=out["learns"],
                                  init=out.get("eval_init"), sec=out.get("train_seconds"))), flush=True)


if __name__ == "__main__":
    main()
