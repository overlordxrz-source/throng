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
Nov 15 ~03:55 (Cam): --hunger-gate (bouts open only when E < H = E_set - size; the world fix), --item-size, the
follower falls back to v2-best outside bouts, and the currency is PAID INTAKE per life (energy gained below the
setpoint; survival reported alongside). Gate off + --currency food + --follower-fallback none reproduces the original.
Note: item energy is added after env.step, so a same-step starvation death is not rescued (rare)."""
import argparse, glob, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import flax.serialization
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402
from w2.worlds import world_kw  # noqa: E402
from w2 import named_item as NI  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ckpt as _ck  # noqa: E402

EAT = env.EAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", default="w2_out/nav_w13/*.msgpack"); ap.add_argument("--world", default="0.0013,8,0.001")
    ap.add_argument("--v2", default=os.path.expanduser("~/throng-backup/evo2/evo2.ckpt"))
    ap.add_argument("--W", type=int, default=32); ap.add_argument("--N", type=int, default=4); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d-max", type=int, default=12); ap.add_argument("--d-fixed", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-params", default="", help="ORACLE ARM eval: also score a port_w2.py-trained agent (its weights)")
    ap.add_argument("--policy-oracle", action="store_true", help="the trained agent gets the oracle sector input")
    ap.add_argument("--item-size", type=float, default=0.1); ap.add_argument("--hunger-gate", action="store_true", help="bouts open only when E < E_set - size")
    ap.add_argument("--follower-fallback", default="", help="none | v2best (default: v2best when gated, none otherwise)")
    ap.add_argument("--currency", default="paid", choices=["paid", "food"]); ap.add_argument("--bar", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = env.Cfg(L=96, N=a.N, patch_sigma=1.0, reproduce=False, A_max=3000, **world_kw(a.world)); L = cfg.L
    nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob(a.navs))[0], "rb").read())
    # v2-best rebuilt as in the harness: RecAgent(128), init seed 0, Dx = 100 + 11 + 6 + 1 + 8 (oracle), navigator trunk, evolved groups
    rnet, v2p = NI.build_v2best(nav, _ck.load(a.v2)["evo_pop"][0])
    rp = rnet.init(jax.random.PRNGKey(0), jnp.zeros((128,)), jnp.zeros((126,)))
    trp = flax.serialization.from_bytes(rp, open(a.policy_params, "rb").read()) if a.policy_params else None

    S = a.item_size; H = cfg.E_set - S; gated = a.hunger_gate
    fb = a.follower_fallback or ("v2best" if gated else "none")
    dfn = (lambda k: jnp.full((a.N,), float(a.d_fixed))) if a.d_fixed else (lambda k: jax.random.randint(k, (a.N,), 1, a.d_max + 1).astype(jnp.float32))

    def nav_apply(o):
        return jax.vmap(lambda xx: Net().apply(nav, xx[None]))(o)[0][:, 0]

    def run(policy, seed):
        key = jax.random.PRNGKey(seed); s = env.reset(cfg, key)
        code = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(jax.random.fold_in(key, 3), a.N))
        item, _ = NI.open_bouts(NI.new_item(a.N), s, jax.random.fold_in(key, 1), dfn, L, (s.energy < H) if gated else jnp.ones(a.N, bool))
        h = jnp.zeros((a.N, 128)); pa = jnp.zeros(a.N, jnp.int32); pr = jnp.zeros(a.N)

        def body(c, k):
            s, item, h, pa, pr, code = c
            k1, k2, k3, k4 = jax.random.split(k, 4); o = NI.observe(cfg, s, item); hr = NI.hear(s, item, code, L)
            if policy == "navigator":
                act = jax.random.categorical(k1, nav_apply(o)); h2 = h
            elif policy in ("v2best", "follower") and not (policy == "follower" and fb == "none"):
                x = jnp.concatenate([o, hr, jax.nn.one_hot(pa, 6), pr[:, None], jnp.zeros((a.N, 8))], 1)
                h2, lg, _, _ = jax.vmap(lambda hh, xx: rnet.apply(v2p, hh, xx))(h, x); act = jax.random.categorical(k1, lg)
                if policy == "follower":
                    act = NI.follower(s, item, L, act)
            elif policy == "trained":
                x = jnp.concatenate([o, hr, jax.nn.one_hot(pa, 6), pr[:, None], NI.oracle(item, a.policy_oracle)], 1)
                h2, lg, _, _ = jax.vmap(lambda hh, xx: rnet.apply(trp, hh, xx))(h, x); act = jax.random.categorical(k1, lg)
            else:                                                               # the original follower (no fallback)
                act = NI.follower(s, item, L, jnp.zeros(a.N, jnp.int32)); h2 = h
            on_item = item["act"] & jnp.all(s.pos == item["pos"], -1) & (act == EAT) & s.alive
            s2, r, info = env.step(cfg, s, act, k2)
            gain = cfg.eff * info["take"] + S * on_item                         # energy gained this step (world + item)
            pd = NI.paid(s.energy, gain, cfg.E_set)                             # the part the setpoint reward pays for
            s2 = s2._replace(energy=jnp.clip(s2.energy + S * on_item * s2.alive, 0.0, 1.0))
            dead = ~s2.alive                                                    # respawn: a fresh life (new code, memory wiped)
            s3 = s2._replace(energy=jnp.where(dead, cfg.E0, s2.energy), age=jnp.where(dead, 0, s2.age), alive=jnp.ones(a.N, bool))
            item2, closed, opened = NI.close_and_open(item, on_item, dead, s3, k3, dfn, L, gated, H)
            ncode = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(k4, a.N))
            code2 = jnp.where(dead[:, None], ncode, code); h3 = jnp.where(dead[:, None], 0.0, h2)
            fullpaid = on_item & (s.energy + S <= cfg.E_set + 1e-6)
            return (s3, item2, h3, act, r * 20.0, code2), (gain * s.alive, on_item, closed & ~dead, s.alive, dead, pd * s.alive, opened, fullpaid, item["act"])
        _, Z = jax.lax.scan(body, (s, item, h, pa, pr, code), jax.random.split(jax.random.fold_in(key, 2), a.steps))
        return tuple(z.sum() for z in Z)

    runj = jax.jit(run, static_argnums=(0,))
    res = {}; PER = {}
    for pol in ("navigator", "v2best", "follower") + (("trained",) if trp is not None else ()):
        Zs = np.array([[float(q) for q in runj(pol, a.seed * 1000 + w)] for w in range(a.W)])     # (W, 9)
        g, ate_i, ren, alv, dd, pd, opn, fp, inb = Zs.sum(0); lives = a.W * a.N + dd
        PER[pol] = (Zs[:, 5], a.N + Zs[:, 4], Zs[:, 0], Zs[:, 3])     # per world: paid, lives, food, agent-steps
        res[pol] = dict(food_per_step=g / alv, paid_per_life=pd / lives, paid_per_step=pd / alv, lives=int(lives), deaths=int(dd),
                        mean_life_steps=alv / lives, items_eaten=int(ate_i), bouts_opened=int(opn), item_reach_rate=ate_i / max(ren, 1),
                        item_meals_fully_paid=fp / max(ate_i, 1), frac_steps_in_bout=inb / alv, agent_steps=int(alv))   # ren counts every closed bout
        print(pol, json.dumps(res[pol]), flush=True)
    cur = "paid_per_life" if a.currency == "paid" else "food_per_step"
    bk = "v2best" if res["v2best"][cur] >= res["navigator"][cur] else "navigator"
    out = dict(cfg=vars(a), H=H, follower_fallback=fb, currency=cur, results=res, premium=res["follower"][cur] / res[bk][cur], best_code_free=bk,
               premium_food=res["follower"]["food_per_step"] / max(res["v2best"]["food_per_step"], res["navigator"]["food_per_step"]),
               premium_paid=res["follower"]["paid_per_life"] / max(res["v2best"]["paid_per_life"], res["navigator"]["paid_per_life"]))
    out["pass"] = bool(out["premium"] >= a.bar)

    def metric(pol, ix):
        pdw, lv, fw, sw = (z[ix] for z in PER[pol])
        return pdw.sum() / lv.sum() if cur == "paid_per_life" else fw.sum() / sw.sum()
    if trp is not None:      # ORACLE ARM bar: trained >= 1.3x the best code-free; bootstrap over worlds (the lives / keys are paired by world)
        rb = np.random.default_rng(7); boots = []
        for _ in range(2000):
            ix = rb.integers(0, a.W, a.W); boots.append(metric("trained", ix) / metric(bk, ix))
        rt = res["trained"][cur] / res[bk][cur]
        out["oracle_arm"] = dict(currency=cur, ratio=rt, boot_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                 pass_=bool(rt >= 1.3 and np.percentile(boots, 2.5) > 1.0))
        print("ORACLE_ARM", json.dumps(out["oracle_arm"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1); print("NECESSITY", json.dumps({k: out[k] for k in ("premium", "best_code_free", "pass")}), flush=True)


if __name__ == "__main__":
    main()
