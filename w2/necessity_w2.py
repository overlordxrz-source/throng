"""Cam Nov 15 (W2 port spec, item 1): NECESSITY CHECK in the new W2 variant. No training.

The picked W2 world (drain 0.0013, 8 patches, relocation 0.001; L 96, patch sigma 1). Each agent always has one
PERSONAL NAMED food item at distance d from where its bout started (d ~ uniform{1..d_max} per bout). The item is
announced every step through the agent's life code (token = code[sector], flag 1, offset to the bout start / 9), and is
VISIBLE ONLY WHEN ADJACENT. Eating it gives +0.1 energy (one bite x eff, as in the toy). It is replaced when eaten or
after 4d steps. Agents also forage the shared world as usual. Lives respawn on death; same worlds and keys for every
policy.

Policies:
- navigator (frozen W2 navigator 0);
- v2-best (the strongest code-free forager: v2's top elite, rebuilt as in the harness; oracle input silent);
- scripted true-code follower (walks the named sector's ray from its bout start; goes to the item when adjacent; eats
  on the item; eats world food it stands on).
Pass: follower food (energy gained per agent-step) >= 1.5x the better of v2-best and the navigator.
Note: item energy is added after env.step, so a same-step starvation death is not rescued (rare)."""
import argparse, glob, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import flax.serialization
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402
from w2.learnability import RecAgent  # noqa: E402
from w2.worlds import world_kw  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ckpt as _ck  # noqa: E402

MOVES = env.MOVES; EAT = env.EAT; R = 3
OFF = jnp.stack(jnp.meshgrid(jnp.arange(-R, R + 1), jnp.arange(-R, R + 1), indexing="ij"), -1).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", default="w2_out/nav_w13/*.msgpack"); ap.add_argument("--world", default="0.0013,8,0.001")
    ap.add_argument("--v2", default=os.path.expanduser("~/throng-backup/evo2/evo2.ckpt"))
    ap.add_argument("--W", type=int, default=32); ap.add_argument("--N", type=int, default=4); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d-max", type=int, default=12); ap.add_argument("--d-fixed", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = env.Cfg(L=96, N=a.N, patch_sigma=1.0, reproduce=False, A_max=3000, **world_kw(a.world)); L = cfg.L
    nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob(a.navs))[0], "rb").read())
    # v2-best rebuilt as in the harness: RecAgent(128), init seed 0, Dx = 100 + 11 + 6 + 1 + 8 (oracle), navigator trunk, evolved groups
    rnet = RecAgent(128); Dx = 126
    rp = rnet.init(jax.random.PRNGKey(0), jnp.zeros((128,)), jnp.zeros((Dx,)))
    pp = {k: dict(v) for k, v in rp["params"].items()}
    for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
        pp[k] = dict(nav["params"][k])
    top = jax.tree_util.tree_map(jnp.asarray, _ck.load(a.v2)["evo_pop"][0]); pp.update(top); v2p = {"params": pp}

    def torus(o):
        return (o + L // 2) % L - L // 2

    def place(key, start, n):
        k1, k2 = jax.random.split(key); sec = jax.random.randint(k1, (n,), 0, 8)
        d = jnp.full((n,), a.d_fixed, jnp.float32) if a.d_fixed else jax.random.randint(k2, (n,), 1, a.d_max + 1).astype(jnp.float32)
        ang = sec * (jnp.pi / 4); ipos = (start + jnp.stack([jnp.round(d * jnp.cos(ang)), jnp.round(d * jnp.sin(ang))], -1).astype(jnp.int32)) % L
        return sec, d, ipos

    def observe(s, item):
        o = env.observe(cfg, s)                                     # (N, 100): food window, agent window, energy, age
        cell = (s.pos[:, None, :] + OFF[None]) % L
        vis = jnp.all(cell == item["pos"][:, None, :], -1) & (jnp.abs(OFF).max(-1) <= 1)[None]        # the item, only when adjacent
        return o.at[:, :49].set(jnp.maximum(o[:, :49], vis.astype(jnp.float32)))

    def hear(s, item, code):
        tok = code[jnp.arange(a.N), item["sec"]]
        return jnp.concatenate([jax.nn.one_hot(tok, 8), jnp.ones((a.N, 1)), jnp.clip(torus(item["start"] - s.pos) / 9.0, -1, 1)], 1)

    def follower(s, item, o):
        on_item = jnp.all(s.pos == item["pos"], -1); here = s.food[s.pos[:, 0], s.pos[:, 1]] >= 0.2
        adj = torus(item["pos"] - s.pos); seen = jnp.abs(adj).max(-1) <= 1
        tw = lambda dd: jnp.where(jnp.abs(dd[:, 0]) >= jnp.abs(dd[:, 1]), jnp.where(dd[:, 0] < 0, 1, 2), jnp.where(dd[:, 1] > 0, 3, 4))  # noqa: E731
        ang = item["sec"] * (jnp.pi / 4); u = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1); off = torus(s.pos - item["start"]).astype(jnp.float32)
        cand = off[:, None, :] + MOVES[1:5][None].astype(jnp.float32)
        score = (cand * u[:, None]).sum(-1) - 2.0 * jnp.abs(cand[..., 0] * u[:, None, 1] - cand[..., 1] * u[:, None, 0])
        return jnp.where(on_item, EAT, jnp.where(seen, tw(adj), jnp.where(here, EAT, jnp.argmax(score, 1) + 1)))

    def run(policy, seed):
        key = jax.random.PRNGKey(seed); s = env.reset(cfg, key)
        code = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(jax.random.fold_in(key, 3), a.N))
        sec, d, ipos = place(jax.random.fold_in(key, 1), s.pos, a.N)
        item = dict(sec=sec, d=d, pos=ipos, start=s.pos, t=jnp.zeros(a.N)); h = jnp.zeros((a.N, 128)); pa = jnp.zeros(a.N, jnp.int32); pr = jnp.zeros(a.N)
        def body(c, k):
            s, item, h, pa, pr, code = c
            k1, k2, k3, k4 = jax.random.split(k, 4); o = observe(s, item); hr = hear(s, item, code)
            if policy == "navigator":
                lg, _ = nav_apply(o); act = jax.random.categorical(k1, lg); h2 = h
            elif policy == "v2best":
                x = jnp.concatenate([o, hr, jax.nn.one_hot(pa, 6), pr[:, None], jnp.zeros((a.N, 8))], 1)
                h2, lg, _, _ = jax.vmap(lambda hh, xx: rnet.apply(v2p, hh, xx))(h, x); act = jax.random.categorical(k1, lg)
            else:
                act = follower(s, item, o); h2 = h
            on_item = jnp.all(s.pos == item["pos"], -1) & (act == EAT) & s.alive
            s2, r, info = env.step(cfg, s, act, k2)
            gain = cfg.eff * info["take"] + 0.1 * on_item                       # energy gained this step (world + item)
            s2 = s2._replace(energy=jnp.clip(s2.energy + 0.1 * on_item * s2.alive, 0.0, 1.0))
            t2 = item["t"] + 1; renew = on_item | (t2 >= 4 * item["d"]) | ~s2.alive
            nsec, nd, npos = place(k3, s2.pos, a.N)
            item2 = dict(sec=jnp.where(renew, nsec, item["sec"]), d=jnp.where(renew, nd, item["d"]), pos=jnp.where(renew[:, None], npos, item["pos"]),
                         start=jnp.where(renew[:, None], s2.pos, item["start"]), t=jnp.where(renew, 0.0, t2))
            dead = ~s2.alive                                                    # respawn: a fresh life (new code, memory wiped)
            E = jnp.where(dead, cfg.E0, s2.energy); age = jnp.where(dead, 0, s2.age)
            s3 = s2._replace(energy=E, age=age, alive=jnp.ones(a.N, bool))
            ncode = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(k4, a.N))
            code2 = jnp.where(dead[:, None], ncode, code); h3 = jnp.where(dead[:, None], 0.0, h2)
            return (s3, item2, h3, act, r * 20.0, code2), (gain * s.alive, on_item, renew & ~dead, s.alive, dead)
        (_, _, _, _, _, _), (g, ate_i, ren, alv, dd) = jax.lax.scan(body, (s, item, h, pa, pr, code), jax.random.split(jax.random.fold_in(key, 2), a.steps))
        return g.sum(), alv.sum(), ate_i.sum(), ren.sum(), dd.sum()

    def nav_apply(o):
        return jax.vmap(lambda xx: Net().apply(nav, xx[None]))(o)[0][:, 0], None

    runj = jax.jit(run, static_argnums=(0,))
    res = {}
    for pol in ("navigator", "v2best", "follower"):
        tot = np.zeros(5)
        for w in range(a.W):
            tot += np.array([float(z) for z in runj(pol, a.seed * 1000 + w)])
        g, alv, ate_i, ren, dd = tot
        res[pol] = dict(food_per_step=g / alv, items_eaten=int(ate_i), item_reach_rate=ate_i / max(ren, 1), deaths=int(dd), agent_steps=int(alv))   # ren counts every closed bout
        print(pol, json.dumps(res[pol]), flush=True)
    best = max(res["navigator"]["food_per_step"], res["v2best"]["food_per_step"])
    out = dict(cfg=vars(a), results=res, premium=res["follower"]["food_per_step"] / best, best_code_free="v2best" if res["v2best"]["food_per_step"] >= res["navigator"]["food_per_step"] else "navigator")
    out["pass"] = bool(out["premium"] >= 1.5)
    json.dump(out, open(a.out, "w"), indent=1); print("NECESSITY", json.dumps({k: out[k] for k in ("premium", "best_code_free", "pass")}), flush=True)


if __name__ == "__main__":
    main()
