"""Cam Nov 15: the W2 PORT trainer (oracle arm first). The necessity-check variant of the picked W2 world: every agent
always has one personal named item at distance d from its bout start, announced through its life code, visible only
when adjacent, +0.1 energy, replaced when eaten or after 4d steps. Distance: the registered MIXED curriculum (d ~
uniform{1..frontier}; the frontier +1 when item reach over the last 500 bouts >= 80%; cap 24).

Learner: the harness agent (RecAgent: navigator trunk UNFROZEN, GRU, side head), KL 0; minibatched PPO (8 x 4),
entropy 0.01, gamma 0.99, lambda 0.95, the clipped log-ratio and the NaN guard. --oracle adds the named item's sector
(one-hot) as input (the ORACLE ARM). Reward = x20 drive reduction including the item's energy. Lives respawn on death
(memory wiped, new code). Adult lives only (no parent / juvenile phase): a disclosed simplification for the oracle arm.
Nov 15 ~03:55 (Cam): --hunger-gate / --item-size (the world fix, shared with necessity_w2 via w2/named_item.py);
--policy follower = the CURRICULUM POSITIVE CONTROL (scripted follower through the real gate, no updates); --eval-only
+ --init-params for readouts; reach split by bout-start energy. Gate off reproduces oracle #1's world.
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
from w2 import offset_wide as OW  # noqa: E402
from w2 import named_item as NI  # noqa: E402
from w2.gate import HonestGate  # noqa: E402
from w2 import lookahead as LA  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ckpt as _ck  # noqa: E402

EAT = env.EAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", default="w2_out/nav_w13/*.msgpack"); ap.add_argument("--world", default="0.0013,8,0.001")
    ap.add_argument("--W", type=int, default=16); ap.add_argument("--N", type=int, default=4); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--moves8", action="store_true", help="Cam Nov 17 ~13:00Z L2: 8-way movement (actions 6..9 = diagonal steps, a cardinal step's energy cost); prev-action one-hot columns 6..9 appended at x[126:130]")
    ap.add_argument("--offset-wide", action="store_true", help="Cam Nov 17 L4: in-bout offset input clip(off, +-48) / 24 (w2/offset_wide.py) instead of clip(off / 9, +-1)")
    ap.add_argument("--steps", type=float, default=3e7); ap.add_argument("--oracle", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--minibatches", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-max", type=int, default=24); ap.add_argument("--curve-every", type=int, default=20)
    ap.add_argument("--ckpt", default=""); ap.add_argument("--max-seconds", type=float, default=0); ap.add_argument("--save-params", default="")
    ap.add_argument("--item-size", type=float, default=0.1); ap.add_argument("--hunger-gate", action="store_true", help="bouts open only when E < E_set - size")
    ap.add_argument("--policy", default="learner", choices=["learner", "follower"], help="follower = the scripted curriculum positive control (no updates)")
    ap.add_argument("--v2", default=os.path.expanduser("~/throng-backup/evo2/evo2.ckpt")); ap.add_argument("--init-params", default="")
    ap.add_argument("--eval-only", action="store_true", help="no updates (readouts)"); ap.add_argument("--frontier0", type=float, default=1.0)
    ap.add_argument("--frontier-schedule", default="", help="DEAF TWIN: replay another run's frontier from its train JSON curve (latest logged value at or before the current step); the gate is off")
    ap.add_argument("--gate", default="old", choices=["old", "honest"], help="Cam Nov 15 ~12:20: honest = hungry-bout reach, two consecutive 2000-bout windows >= 0.8")
    ap.add_argument("--lookahead", type=float, default=0.0, help="Cam Nov 16 ~01:00: with this probability a bout's d is drawn from frontier+1 .. --la-max; the gate scores only d <= frontier")
    ap.add_argument("--la-max", type=int, default=12)
    ap.add_argument("--giveup", type=float, default=0.0, help="NEGATIVE gate control (with --policy follower): give up at random on this fraction of bouts")
    ap.add_argument("--ent-bout", type=float, default=None, help="Cam Nov 16 ~17:25Z O-X: the entropy coefficient on in-bout steps (x[108] = the bout flag); off-bout stays 0.01")
    ap.add_argument("--ent-anneal", type=float, default=0.0, help="Cam Nov 16 ~19:00Z O-X': with --ent-bout E, the in-bout coefficient goes linearly E -> 0.01 over this many steps, then 0.01")
    ap.add_argument("--reinit-out", action="store_true", help="Cam Nov 16 ~17:55Z (a'): Dense_2 (kernel + bias) keeps its fresh init (the navigator's own init scale); Dense_0/1/3 from the navigator")
    ap.add_argument("--no-nav", action="store_true", help="Cam Nov 16 ~17:55Z (b'): no navigator init; the whole trunk trains from its fresh init")
    ap.add_argument("--topo", action="store_true", help="Cam Nov 16 ~17:25Z O-T: the topographic tie in RecAgent (move logits += g P s, g learned, init 0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = a.item_size; gated = a.hunger_gate; nupd = a.eval_only or a.policy == "follower"
    cfg = env.Cfg(L=96, N=a.N, patch_sigma=1.0, reproduce=False, A_max=3000, moves8=a.moves8, **world_kw(a.world)); L, W, N, T = cfg.L, a.W, a.N, a.T
    nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob(a.navs))[0], "rb").read())
    rnet = RecAgent(128, topo=a.topo, n_act=10 if a.moves8 else 6); Dx = 100 + 11 + 6 + 1 + 8 + (4 if a.moves8 else 0)   # the harness layout (oracle slot always present; zero when off)
    assert not (a.moves8 and (a.policy == "follower" or not a.no_nav)), "--moves8 is built for the learner policy with --no-nav only"
    rp = rnet.init(jax.random.PRNGKey(a.seed), jnp.zeros((128,)), jnp.zeros((Dx,)))
    pp = {k: (dict(v) if hasattr(v, "items") else v) for k, v in rp["params"].items()}
    for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
        if a.no_nav or (a.reinit_out and k == "Dense_2"):              # (b') / (a'): keep the fresh init (flax Dense default = the navigator's own init)
            continue
        pp[k] = dict(nav["params"][k])
    p = {"params": pp}                                                # trunk UNFROZEN (nothing zeroed in the gradient)
    if a.init_params:
        p = flax.serialization.from_bytes(p, open(a.init_params, "rb").read())
    if a.policy == "follower":
        _, v2p = NI.build_v2best(nav, _ck.load(a.v2)["evo_pop"][0])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)); ost = opt.init(p)

    H = cfg.E_set - S

    def drive(E):
        return jnp.abs(cfg.E_set - E) ** cfg.drive_n

    def dfn_of(dfront):
        return LA.sampler(dfront, N, a.lookahead, a.la_max)                   # mixed curriculum: uniform{1..frontier} (+ lookahead bouts)

    def obs_x(s, item, code, pa, pr):
        o = NI.observe(cfg, s, item); hr = (OW.hear_wide if a.offset_wide else NI.hear)(s, item, code, L)
        x_ = jnp.concatenate([o, hr, jax.nn.one_hot(pa, 6), pr[:, None], NI.oracle(item, a.oracle)], 1)
        return jnp.concatenate([x_, jax.nn.one_hot(pa, 10)[:, 6:]], 1) if a.moves8 else x_

    def world_reset(key, dfront):
        s = env.reset(cfg, key)
        item, _ = NI.open_bouts(NI.new_item(N), s, jax.random.fold_in(key, 1), dfn_of(dfront), L, (s.energy < H) if gated else jnp.ones(N, bool))
        code = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(jax.random.fold_in(key, 3), N))
        return (s, item, jnp.zeros((N, 128)), jnp.zeros(N, jnp.int32), jnp.zeros(N), code)

    def rollout_world(pp_, c, key, dfront):
        def body(c, k):
            s, item, h, pa, pr, code = c
            k1, k2, k3, k4 = jax.random.split(k, 4); x = obs_x(s, item, code, pa, pr)
            item = dict(item, sat=item["sat"] | (item["act"] & (s.energy >= H)))   # the bout is no longer hungry once E reached H
            if a.policy == "follower":                                      # scripted: walks on a hearing, v2-best otherwise
                h2, lg, v, _ = jax.vmap(lambda hh, xx: rnet.apply(v2p, hh, xx))(h, x.at[:, -8:].set(0.0))
                fb = jax.random.categorical(k1, lg); act = NI.follower(s, item, L, fb)
                if a.giveup > 0:                                            # NEGATIVE control: a given-up bout is ignored (v2-best acts, never eats the item)
                    gave = item["gu"] & item["act"]; noeat = jnp.where(jnp.all(s.pos == item["pos"], -1) & (fb == EAT), 0, fb)
                    act = jnp.where(gave, noeat, act)
            else:
                h2, lg, v, _ = jax.vmap(lambda hh, xx: rnet.apply(pp_, hh, xx))(h, x); act = jax.random.categorical(k1, lg)
            lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act[:, None], 1)[:, 0]
            on_item = item["act"] & jnp.all(s.pos == item["pos"], -1) & (act == EAT)
            E0 = s.energy; s2, _, info = env.step(cfg, s, act, k2)
            E1 = jnp.clip(s2.energy + S * on_item * s2.alive, 0.0, 1.0); s2 = s2._replace(energy=E1)
            r = 20.0 * (drive(E0) - drive(E1))                              # x20 drive reduction, item energy included
            gain = cfg.eff * info["take"] + S * on_item; pd = NI.paid(E0, gain, cfg.E_set)
            dead = ~s2.alive
            s3 = s2._replace(energy=jnp.where(dead, cfg.E0, s2.energy), age=jnp.where(dead, 0, s2.age), alive=jnp.ones(N, bool))
            item2, closed, opened = NI.close_and_open(item, on_item, dead, s3, k3, dfn_of(dfront), L, gated, H)
            if a.giveup > 0:
                item2 = dict(item2, gu=jnp.where(opened, jax.random.uniform(jax.random.fold_in(k3, 5), (N,)) < a.giveup, item2["gu"]))
            hung = closed & ~dead & ~item["sat"]
            ncode = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(k4, N))
            code2 = jnp.where(dead[:, None], ncode, code); h3 = jnp.where(dead[:, None], 0.0, h2)
            return (s3, item2, h3, act, r, code2), (x, act, lp, v, r, dead.astype(jnp.float32), h, on_item & ~dead, closed & ~dead, gain, pd, item["es"], item["d"], item["act"], hung, on_item & ~dead & ~item["sat"])
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
        if a.ent_bout is None:
            return (pg + 0.5 * (v - bt["ret"]) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)).mean()
        ec = jnp.where(bt["x"][..., 108] > 0, bt["ecb"] if a.ent_anneal else a.ent_bout, 0.01)   # O-X / O-X': in-bout entropy coefficient
        return (pg + 0.5 * (v - bt["ret"]) ** 2 + ec * (jnp.exp(ls) * ls).sum(-1)).mean()

    @jax.jit
    def upd(pp_, os_, bt):
        g = jax.grad(lambda q: jax.vmap(seq_loss, in_axes=(None, 0))(q, bt).mean())(pp_)
        u, os_ = opt.update(g, os_, pp_); return optax.apply_updates(pp_, u), os_

    rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed + 1)
    dfront = a.frontier0; recent = []; SPLIT = []; GATE = HonestGate(); steps = 0; it = 0; curve = []; nan_run = 0
    c = jax.vmap(world_reset, in_axes=(0, None))(jax.random.split(key, W), dfront)
    Ck = _ck.load(a.ckpt) if a.ckpt else None
    if Ck is not None:
        tj = lambda t_: jax.tree_util.tree_map(jnp.asarray, t_)  # noqa: E731
        p, ost, c, key = tj(Ck["p"]), tj(Ck["ost"]), tj(Ck["c"]), jnp.asarray(Ck["key"])
        dfront, recent, steps, it, curve = Ck["dfront"], Ck["recent"], Ck["steps"], Ck["it"], Ck["curve"]; rng.bit_generator.state = Ck["rng"]
        if "gate" in Ck:
            GATE.load(Ck["gate"])
        print(f"resumed at {steps:,} steps, frontier {dfront}", flush=True)
    SCHED = [(c_["steps"], c_["frontier"]) for c_ in json.load(open(a.frontier_schedule))["curve"]] if a.frontier_schedule else None

    def sched_front(st_):
        fr = 1.0
        for s_, f_ in SCHED:
            if s_ <= st_:
                fr = f_
        return float(fr)
    t0 = time.time()
    while steps < a.steps:
        if SCHED is not None:
            dfront = sched_front(steps)
        if a.ckpt and it % 20 == 0 and it > 0:
            _ck.save(a.ckpt, dict(p=p, ost=ost, c=c, key=key, dfront=dfront, recent=recent, steps=steps, it=it, curve=curve, rng=rng.bit_generator.state, gate=GATE.state()))
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                print(f"CHUNK_STOP at {steps:,}", flush=True); sys.exit(3)
        key, k = jax.random.split(key); c, R_ = roll(p, c, jax.random.split(k, W), jnp.float32(dfront))
        x, act, lp, v, r, dead, hin, ate_i, closed, gain, pd, es, dd_, inb, hung, hate = (np.asarray(z) for z in R_)      # (W, T, N, ...)
        SPLIT.append(np.stack([es[closed > 0], ate_i[closed > 0], dd_[closed > 0]], 1))
        # (W, T, N) -> sequences (W*N, T)
        sq = lambda z: np.swapaxes(z, 1, 2).reshape((W * N, T) + z.shape[3:])  # noqa: E731
        r_, v_, d_ = sq(r), sq(v), sq(dead)
        adv = gae(r_, v_, d_, 0.99, 0.95); ret = adv + v_; A_ = (adv - adv.mean()) / (adv.std() + 1e-8)
        first = np.concatenate([np.zeros((W * N, 1)), d_[:, :-1]], 1)
        if a.ent_anneal:                                                    # O-X': the current in-bout coefficient, one per sequence
            ECB = 0.01 + (a.ent_bout - 0.01) * max(0.0, 1.0 - steps / a.ent_anneal)
        bt = dict(**(dict(ecb=jnp.full((W * N,), ECB, jnp.float32)) if a.ent_anneal else {}), x=jnp.asarray(sq(x)), h0=jnp.asarray(hin[:, 0].reshape(W * N, 128)), act=jnp.asarray(sq(act)), lp=jnp.asarray(sq(lp)),
                  adv=jnp.asarray(A_.astype(np.float32)), ret=jnp.asarray(ret.astype(np.float32)), first=jnp.asarray(first.astype(np.float32)))
        p_prev, o_prev = p, ost
        for _ in range(0 if nupd else a.epochs):
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
        n_la = int((hung * (dd_ > dfront)).sum())                          # hungry lookahead bouts this iteration (before any advance)
        if a.gate == "honest":
            inr = LA.in_range(dd_, dfront)                                  # lookahead bouts are not scored
            if SCHED is None and GATE.add(int((hate * inr).sum()), int((hung * inr).sum())) and dfront < a.d_max:
                dfront += 1; print(f"GATE_UP frontier -> {dfront:.0f} at {steps:,} (last windows {GATE.scores[-2:]})", flush=True)
        elif SCHED is None and len(recent) >= 500 and np.mean(recent) >= 0.8 and dfront < a.d_max:
            dfront += 1; recent = []
        if it % a.curve_every == 0:
            curve.append(dict(steps=steps, frontier=dfront, item_reach=n_eat / max(n_close, 1), food_per_step=float(gain.mean()), paid_per_step=float(pd.mean()), frac_in_bout=float(inb.mean()),
                              hungry_reach=float(hate.sum() / max(hung.sum(), 1)), n_hungry=int(hung.sum()), n_hungry_lookahead=n_la, gate_last=(GATE.scores[-1] if GATE.scores else None), sec=round(time.time() - t0, 1)))
            print("CURVE", json.dumps(curve[-1]), flush=True)
    if a.save_params:
        open(a.save_params, "wb").write(flax.serialization.to_bytes(p))
    Sp = np.concatenate(SPLIT[-200:]) if SPLIT else np.zeros((0, 3)); split = {}
    for nm, lo, hi in (("E<0.6", 0, 0.6), ("0.6<=E<H", 0.6, H), ("H<=E<0.75", H, 0.75), ("E>=0.75", 0.75, 9)):
        for dsel in ("d=1", "all"):
            m = (Sp[:, 0] >= lo) & (Sp[:, 0] < hi) & ((Sp[:, 2] == 1) if dsel == "d=1" else True)
            split[f"{nm} {dsel}"] = dict(reach=float(Sp[m, 1].mean()) if m.any() else None, bouts=int(m.sum()))
    print("REACH_BY_START_ENERGY (last <=200 iterations)", json.dumps(split), flush=True)
    out = dict(cfg=vars(a), steps=steps, frontier=dfront, curve=curve, reach_by_start_energy=split, gate_scores=GATE.scores, train_seconds=round(time.time() - t0, 1),
               throughput=steps / max(time.time() - t0, 1e-9))
    json.dump(out, open(a.out, "w"), indent=1); print("TRAINED", json.dumps(dict(steps=steps, frontier=dfront)), flush=True)


if __name__ == "__main__":
    main()
