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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True); ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tier", choices=["T1", "T2", "T3"], default="T1"); ap.add_argument("--learner", choices=["L1", "L2", "scripted"], default="L1")
    ap.add_argument("--kl", type=float, default=0.3); ap.add_argument("--steps", type=float, default=1e6)
    ap.add_argument("--B", type=int, default=256); ap.add_argument("--T", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes-per-life", type=int, default=8, help="T3: episodes sharing one random code (memory carried within a life)")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--minibatches", type=int, default=1, help="harness default: 2 full-batch epochs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--move-cost", type=float, default=DMOVE, help="audit only: the per-move energy cost (harness 0.003)")
    ap.add_argument("--eval-episodes", type=int, default=512); ap.add_argument("--curve-every", type=int, default=10)
    ap.add_argument("--navs", default="w2_out/nav_w13/*.msgpack"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, cap, B, T = a.d, 4 * a.d, a.B, a.T
    rng = np.random.default_rng(a.seed); FIXED = jnp.asarray(rng.permutation(8))     # T2: one code for all lives
    Dx = 100 + 11 + 6 + 1 + (8 if a.tier == "T1" else 0)
    # ---- learner
    if a.learner == "L1":
        net = RecAgent(128); Hd = 128
        p = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((Hd,)), jnp.zeros((Dx,)))
        nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob(a.navs))[0], "rb").read())["params"]
        pp = {k: dict(v) for k, v in p["params"].items()}
        for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
            pp[k] = dict(nav[k])
        p = {"params": pp}; FROZEN = ("Dense_0", "Dense_1", "Dense_2", "Dense_3")
    elif a.learner == "L2":
        net = (GRUNet() if a.tier == "T3" else MLP()); Hd = 64
        p = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((Hd,)), jnp.zeros((Dx,))); FROZEN = ()
    else:
        net, Hd, p, FROZEN = None, 1, None, ()
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

    def env_reset(key, n, code_draw):
        sec = jax.random.randint(key, (n,), 0, 8)
        return dict(pos=jnp.tile(START[None], (n, 1)), fpos=food_cell(sec, d), alive_f=jnp.ones(n, bool), E=jnp.full(n, E_START), t=jnp.zeros(n),
                    sec=sec, pa=jnp.zeros(n, jnp.int32), pr=jnp.zeros(n))

    def env_step(st, act):
        mv = MOVES[act]; pos = (st["pos"] + mv) % L; moved = (jnp.abs(mv).sum(-1) > 0).astype(jnp.float32)
        on = jnp.all(st["pos"] == st["fpos"], -1) & st["alive_f"]
        ate = (act == EAT) & on
        E1 = jnp.clip(st["E"] + EFF * BITE * ate - (D0 + a.move_cost * moved), 0.0, 1.0)
        r = 20.0 * (drive(st["E"]) - drive(E1)); t = st["t"] + 1
        done = ate | (E1 <= 0.0) | (t >= cap)
        return dict(st, pos=jnp.where(ate[:, None], st["pos"], pos), E=E1, t=t, pa=act, pr=r, alive_f=st["alive_f"] & ~ate), r, done, ate

    # ---- one rollout of T steps over B envs (stochastic policy); resets on done; carries memory within an episode
    #      (T3: within a life of --episodes-per-life episodes)
    def policy(pp_, h, x):
        return jax.vmap(lambda hh, xx: net.apply(pp_, hh, xx))(h, x)

    def rollout(pp_, carry, key, mode):
        def body(c, k):
            st, h, code, ep_in_life = c
            k1, k2, k3, k4 = jax.random.split(k, 4)
            m = jnp.full(B, mode)
            osec = jnp.where(mode == 1, (st["sec"] + jax.random.randint(k3, (B,), 1, 8)) % 8, st["sec"])
            base, hear, _ = obs(st["pos"], st["fpos"], st["alive_f"], st["E"], st["t"], osec, code, m)
            x = build_x(base, hear, osec, st["pa"], st["pr"], m)
            if net is None:
                act = scripted_act(st["pos"], st["fpos"], osec, base); lp = jnp.zeros(B); v = jnp.zeros(B); h2 = h
            else:
                h2, lg, v, _ = policy(pp_, h, x); act = jax.random.categorical(k1, lg)
                lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act[:, None], 1)[:, 0]
            st2, r, done, ate = env_step(st, act)
            # reset finished episodes; T3: a new life (new code, memory wiped) every episodes_per_life episodes
            ep2 = ep_in_life + done.astype(jnp.int32); newlife = done & (ep2 >= (a.episodes_per_life if a.tier == "T3" else 1))
            fresh = env_reset(k2, B, None)
            st3 = jax.tree_util.tree_map(lambda u, w: jnp.where(done.reshape((-1,) + (1,) * (u.ndim - 1)), w, u), st2, fresh)
            if a.tier == "T3":
                nc = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(k4, B))
                code2 = jnp.where(newlife[:, None], nc, code); h3 = jnp.where(newlife[:, None], 0.0, h2); ep3 = jnp.where(newlife, 0, ep2)
            else:
                code2 = code; h3 = jnp.where(done[:, None], 0.0, h2); ep3 = ep2 * 0
            reset = (newlife if a.tier == "T3" else done).astype(jnp.float32)      # when the rollout wiped the memory
            return (st3, h3, code2, ep3), (x, act, lp, v, r, done.astype(jnp.float32), ate, h, reset)
        return jax.lax.scan(body, carry, jax.random.split(key, T))
    roll = jax.jit(rollout, static_argnums=(3,))

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

    def evaluate(pp_, mode, seed):
        """reach-food rate over --eval-episodes complete episodes (stochastic policy)"""
        key = jax.random.PRNGKey(seed); st = env_reset(key, B, None); h = jnp.zeros((B, Hd))
        code = (jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(key, B)) if a.tier == "T3" else jnp.tile(FIXED[None], (B, 1)))
        c = (st, h, code, jnp.zeros(B, jnp.int32)); n_ep = 0; n_ok = 0
        while n_ep < a.eval_episodes:
            key, k = jax.random.split(key); c, R_ = roll(pp_, c, k, mode)
            dn, ate = np.asarray(R_[5]), np.asarray(R_[6]); n_ep += int(dn.sum()); n_ok += int((ate & (dn > 0)).sum())
        return n_ok / max(n_ep, 1)

    out = dict(cfg=vars(a), cap=cap, curve=[])
    if net is None:                                             # the scripted positive control: evaluation only
        out["eval"] = {m_: evaluate(None, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))}
    else:
        key = jax.random.PRNGKey(a.seed + 1); st = env_reset(key, B, None)
        code0 = (jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(key, B)) if a.tier == "T3" else jnp.tile(FIXED[None], (B, 1)))
        c = (st, jnp.zeros((B, Hd)), code0, jnp.zeros(B, jnp.int32)); steps, it, t0 = 0, 0, time.time()
        out["eval_init"] = {m_: evaluate(p, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))}
        while steps < a.steps:
            key, k = jax.random.split(key); c, R_ = roll(p, c, k, 0)
            x, act, lp, v, r, dn, ate, hin, rs = (np.asarray(z) for z in R_)        # (T, B, ...)
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
            if it % a.curve_every == 0:
                actf = np.bincount(act.reshape(-1), minlength=6) / act.size; at = np.swapaxes(act, 0, 1)
                out["curve"].append(dict(steps=steps, reach_train=float(ate.sum() / max(dn.sum(), 1)), sec=round(time.time() - t0, 1),
                                         act_freq=[round(float(q), 3) for q in actf], adv_by_act=[round(float(adv[at == k].mean()), 3) if (at == k).any() else None for k in range(6)],
                                         ret_mean=round(float(ret.mean()), 3), r_eat=round(float(r[ate].mean()), 3) if ate.any() else None))
        out["eval"] = {m_: evaluate(p, i, 777) for i, m_ in enumerate(("true", "shuffled", "silence"))}
        out["train_seconds"] = round(time.time() - t0, 1)
    e = out["eval"]
    out["learns"] = bool(e["true"] >= 0.8 and e["true"] >= 2 * e["shuffled"] and e["true"] - e["shuffled"] >= 0.30)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print("TOY", json.dumps(dict(d=d, hidden=a.hidden, lam=a.lam, tier=a.tier, learner=a.learner, kl=a.kl, eval=e, learns=out["learns"],
                                  init=out.get("eval_init"), sec=out.get("train_seconds"))), flush=True)


if __name__ == "__main__":
    main()
