"""Cam Nov 15: the W2 PORT trainer (oracle arm first). The necessity-check variant of the picked W2 world: every agent
always has one personal named item at distance d from its bout start, announced through its life code, visible only
when adjacent, +0.1 energy, replaced when eaten or after 4d steps. Distance: the registered MIXED curriculum (d ~
uniform{1..frontier}; the frontier +1 when item reach over the last 500 bouts >= 80%; cap 24).

Learner: the harness agent (RecAgent: navigator trunk UNFROZEN, GRU, side head), KL 0; minibatched PPO (8 x 4),
entropy 0.01, gamma 0.99, lambda 0.95, the clipped log-ratio and the NaN guard. --oracle adds the named item's sector
(one-hot) as input (the ORACLE ARM). Reward = x20 drive reduction including the item's energy. Lives respawn on death
(memory wiped, new code). Adult lives only (no parent / juvenile phase): a disclosed simplification for the oracle arm.
Checkpointed chunks (--ckpt --max-seconds); --save-params for the evaluator (necessity_w2.py --policy-params)."""
import argparse, glob, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import flax.serialization
import optax
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402
from w2.learnability import RecAgent  # noqa: E402
from w2.worlds import world_kw  # noqa: E402
from w2.gae import gae  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ckpt as _ck  # noqa: E402

MOVES = env.MOVES; EAT = env.EAT; R = 3
OFF = jnp.stack(jnp.meshgrid(jnp.arange(-R, R + 1), jnp.arange(-R, R + 1), indexing="ij"), -1).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", default="w2_out/nav_w13/*.msgpack"); ap.add_argument("--world", default="0.0013,8,0.001")
    ap.add_argument("--W", type=int, default=16); ap.add_argument("--N", type=int, default=4); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--steps", type=float, default=3e7); ap.add_argument("--oracle", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--minibatches", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-max", type=int, default=24); ap.add_argument("--curve-every", type=int, default=20)
    ap.add_argument("--ckpt", default=""); ap.add_argument("--max-seconds", type=float, default=0); ap.add_argument("--save-params", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = env.Cfg(L=96, N=a.N, patch_sigma=1.0, reproduce=False, A_max=3000, **world_kw(a.world)); L, W, N, T = cfg.L, a.W, a.N, a.T
    nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob(a.navs))[0], "rb").read())
    rnet = RecAgent(128); Dx = 100 + 11 + 6 + 1 + 8                 # the harness layout (oracle slot always present; zero when off)
    rp = rnet.init(jax.random.PRNGKey(a.seed), jnp.zeros((128,)), jnp.zeros((Dx,)))
    pp = {k: dict(v) for k, v in rp["params"].items()}
    for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
        pp[k] = dict(nav["params"][k])
    p = {"params": pp}                                                # trunk UNFROZEN (nothing zeroed in the gradient)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)); ost = opt.init(p)

    def torus(o):
        return (o + L // 2) % L - L // 2

    def drive(E):
        return jnp.abs(cfg.E_set - E) ** cfg.drive_n

    def place(key, start, dfront):
        k1, k2 = jax.random.split(key); sec = jax.random.randint(k1, (N,), 0, 8)
        d = jnp.floor(jax.random.uniform(k2, (N,)) * dfront) + 1.0      # mixed curriculum: uniform{1..frontier}
        ang = sec * (jnp.pi / 4); ipos = (start + jnp.stack([jnp.round(d * jnp.cos(ang)), jnp.round(d * jnp.sin(ang))], -1).astype(jnp.int32)) % L
        return sec, d, ipos

    def obs_x(s, item, code, pa, pr):
        o = env.observe(cfg, s); cell = (s.pos[:, None, :] + OFF[None]) % L
        vis = jnp.all(cell == item["pos"][:, None, :], -1) & (jnp.abs(OFF).max(-1) <= 1)[None]
        o = o.at[:, :49].set(jnp.maximum(o[:, :49], vis.astype(jnp.float32)))
        tok = code[jnp.arange(N), item["sec"]]
        hr = jnp.concatenate([jax.nn.one_hot(tok, 8), jnp.ones((N, 1)), jnp.clip(torus(item["start"] - s.pos) / 9.0, -1, 1)], 1)
        orc = jax.nn.one_hot(item["sec"], 8) * (1.0 if a.oracle else 0.0)
        return jnp.concatenate([o, hr, jax.nn.one_hot(pa, 6), pr[:, None], orc], 1)

    def world_reset(key, dfront):
        s = env.reset(cfg, key); sec, d, ipos = place(jax.random.fold_in(key, 1), s.pos, dfront)
        code = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(jax.random.fold_in(key, 3), N))
        return (s, dict(sec=sec, d=d, pos=ipos, start=s.pos, t=jnp.zeros(N)), jnp.zeros((N, 128)), jnp.zeros(N, jnp.int32), jnp.zeros(N), code)

    def rollout_world(pp_, c, key, dfront):
        def body(c, k):
            s, item, h, pa, pr, code = c
            k1, k2, k3, k4 = jax.random.split(k, 4); x = obs_x(s, item, code, pa, pr)
            h2, lg, v, _ = jax.vmap(lambda hh, xx: rnet.apply(pp_, hh, xx))(h, x); act = jax.random.categorical(k1, lg)
            lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act[:, None], 1)[:, 0]
            on_item = jnp.all(s.pos == item["pos"], -1) & (act == EAT)
            E0 = s.energy; s2, _, info = env.step(cfg, s, act, k2)
            E1 = jnp.clip(s2.energy + 0.1 * on_item * s2.alive, 0.0, 1.0); s2 = s2._replace(energy=E1)
            r = 20.0 * (drive(E0) - drive(E1))                              # x20 drive reduction, item energy included
            gain = cfg.eff * info["take"] + 0.1 * on_item
            t2 = item["t"] + 1; dead = ~s2.alive; renew = on_item | (t2 >= 4 * item["d"]) | dead
            nsec, nd, npos = place(k3, s2.pos, dfront)
            item2 = dict(sec=jnp.where(renew, nsec, item["sec"]), d=jnp.where(renew, nd, item["d"]), pos=jnp.where(renew[:, None], npos, item["pos"]),
                         start=jnp.where(renew[:, None], s2.pos, item["start"]), t=jnp.where(renew, 0.0, t2))
            s3 = s2._replace(energy=jnp.where(dead, cfg.E0, s2.energy), age=jnp.where(dead, 0, s2.age), alive=jnp.ones(N, bool))
            ncode = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(k4, N))
            code2 = jnp.where(dead[:, None], ncode, code); h3 = jnp.where(dead[:, None], 0.0, h2)
            return (s3, item2, h3, act, r, code2), (x, act, lp, v, r, dead.astype(jnp.float32), h, on_item, renew & ~dead, gain)
        return jax.lax.scan(body, c, jax.random.split(key, T))
    roll = jax.jit(jax.vmap(rollout_world, in_axes=(None, 0, 0, None)))

    def seq_loss(pp_, bt):
        def body(h, xa):
            xx, f = xa; h = jnp.where(f > 0, 0.0, h)                         # the same memory wipes as the rollout (deaths)
            h2, lg, v, _ = rnet.apply(pp_, h, xx); return h2, (lg, v)
        _, (lg, v) = jax.lax.scan(body, bt["h0"], (bt["x"], bt["first"]))
        ls = jax.nn.log_softmax(lg); lp = jnp.take_along_axis(ls, bt["act"][:, None], 1)[:, 0]
        ratio = jnp.exp(jnp.clip(lp - bt["lp"], -20.0, 20.0)); adv = bt["adv"]
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        return (pg + 0.5 * (v - bt["ret"]) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)).mean()

    @jax.jit
    def upd(pp_, os_, bt):
        g = jax.grad(lambda q: jax.vmap(seq_loss, in_axes=(None, 0))(q, bt).mean())(pp_)
        u, os_ = opt.update(g, os_, pp_); return optax.apply_updates(pp_, u), os_

    rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed + 1)
    dfront = 1.0; recent = []; steps = 0; it = 0; curve = []; nan_run = 0
    c = jax.vmap(world_reset, in_axes=(0, None))(jax.random.split(key, W), dfront)
    Ck = _ck.load(a.ckpt) if a.ckpt else None
    if Ck is not None:
        tj = lambda t_: jax.tree_util.tree_map(jnp.asarray, t_)  # noqa: E731
        p, ost, c, key = tj(Ck["p"]), tj(Ck["ost"]), tj(Ck["c"]), jnp.asarray(Ck["key"])
        dfront, recent, steps, it, curve = Ck["dfront"], Ck["recent"], Ck["steps"], Ck["it"], Ck["curve"]; rng.bit_generator.state = Ck["rng"]
        print(f"resumed at {steps:,} steps, frontier {dfront}", flush=True)
    t0 = time.time()
    while steps < a.steps:
        if a.ckpt and it % 20 == 0 and it > 0:
            _ck.save(a.ckpt, dict(p=p, ost=ost, c=c, key=key, dfront=dfront, recent=recent, steps=steps, it=it, curve=curve, rng=rng.bit_generator.state))
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                print(f"CHUNK_STOP at {steps:,}", flush=True); sys.exit(3)
        key, k = jax.random.split(key); c, R_ = roll(p, c, jax.random.split(k, W), jnp.float32(dfront))
        x, act, lp, v, r, dead, hin, ate_i, closed, gain = (np.asarray(z) for z in R_)      # (W, T, N, ...)
        # (W, T, N) -> sequences (W*N, T)
        sq = lambda z: np.swapaxes(z, 1, 2).reshape((W * N, T) + z.shape[3:])  # noqa: E731
        r_, v_, d_ = sq(r), sq(v), sq(dead)
        adv = gae(r_, v_, d_, 0.99, 0.95); ret = adv + v_; A_ = (adv - adv.mean()) / (adv.std() + 1e-8)
        first = np.concatenate([np.zeros((W * N, 1)), d_[:, :-1]], 1)
        bt = dict(x=jnp.asarray(sq(x)), h0=jnp.asarray(hin[:, 0].reshape(W * N, 128)), act=jnp.asarray(sq(act)), lp=jnp.asarray(sq(lp)),
                  adv=jnp.asarray(A_.astype(np.float32)), ret=jnp.asarray(ret.astype(np.float32)), first=jnp.asarray(first.astype(np.float32)))
        p_prev, o_prev = p, ost
        for _ in range(a.epochs):
            for mb in np.array_split(rng.permutation(W * N), a.minibatches):
                p, ost = upd(p, ost, {k_: v_[mb] for k_, v_ in bt.items()})
        if not all(bool(jnp.isfinite(l_).all()) for l_ in jax.tree_util.tree_leaves(p)):   # NaN guard
            p, ost = p_prev, o_prev; nan_run += 1; print(f"NAN_SKIP {it}", flush=True)
            if nan_run >= 10:
                print("NAN_STOP", flush=True); break
        else:
            nan_run = 0
        steps += W * N * T; it += 1
        n_close = int(closed.sum()); n_eat = int(ate_i.sum())
        recent.extend([1] * n_eat + [0] * max(n_close - n_eat, 0)); recent = recent[-500:]
        if len(recent) >= 500 and np.mean(recent) >= 0.8 and dfront < a.d_max:
            dfront += 1; recent = []
        if it % a.curve_every == 0:
            curve.append(dict(steps=steps, frontier=dfront, item_reach=n_eat / max(n_close, 1), food_per_step=float(gain.mean()), sec=round(time.time() - t0, 1)))
            print("CURVE", json.dumps(curve[-1]), flush=True)
    if a.save_params:
        open(a.save_params, "wb").write(flax.serialization.to_bytes(p))
    out = dict(cfg=vars(a), steps=steps, frontier=dfront, curve=curve, train_seconds=round(time.time() - t0, 1),
               throughput=steps / max(time.time() - t0, 1e-9))
    json.dump(out, open(a.out, "w"), indent=1); print("TRAINED", json.dumps(dict(steps=steps, frontier=dfront)), flush=True)


if __name__ == "__main__":
    main()
