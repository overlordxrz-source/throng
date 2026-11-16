"""DIAG COPY (Cam Nov 16 ~12:30Z: the pilot table on the relabel screen): learner_w2_rl.py + eval-only splits of the
closed bouts: demonstrated with the parent kept in / lost from the 7x7 view; later hearings with the head's top sector
right / wrong at the hearing (all bouts, and hungry bouts). Training is not changed, but this copy is for EVAL ONLY
(its carry has two extra fields, so it cannot resume a training checkpoint).

RELABEL VARIANT (Cam Nov 16 ~10:15Z, M-approved 10:46Z, stage 1 only): a copy of learner_w2_la.py with --head-relabel-token
(w2/relabel.py): the head target at every hearing of token X = the sector of a SOLVED record of X from this life, from the
young's own record buffer only; masked if none. Default off = learner_w2_la.py.

LOOKAHEAD VARIANT (Cam Nov 16 ~01:00): a copy of learner_w2_kv.py (left untouched while seed 2 runs) with
--lookahead p / --la-max (w2/lookahead.py: with p a bout's d is drawn from frontier+1 .. la_max; the honest gate scores
only d <= frontier) and --stop-at S (save checkpoint + params at S steps and exit with PAUSE_AT, for the registered
head check at 3e7). With --lookahead 0 and no --stop-at it is learner_w2_kv.py.

Original docstring: Cam Nov 15 ~05:50: the W2 LEARNER trainer and its evaluator (one step function for both, so training and the
registered evaluation see the same world).

World: the hunger-gated named-item world of w2/named_item.py (port_w2 / necessity_w2), item size 0.1, A_max 3000.
Agent (--learner):
  rec  = the harness RecAgent (port_w2's agent; with no extras this file reproduces port_w2 bit-exactly: check A3);
  ep   = w2/ep_agent.EpAgent: + the inborn episodic memory (S raw records per life, written at each bout close) with a
         learned read, + the hindsight head (trained only at the hearing step, weight --hindsight), --shared-tok option.
Demonstrator (--demonstrator, --demo-hearings K): on the first K hearings of a token in a life, a scripted parent
appears at the hearing place and walks to the named item (one step per step; it stays on it), visible as another
agent in the 7x7 view. (No young-seek reflex: following is learned, as in the toy.)
Policies (--policy): learner (PPO), follower (A1: the scripted true-code follower drives; with --head-only only the
head's loss trains), copyparent (A2: walks after the demonstrator).
Evaluation (--eval-necessity): the necessity lives / keys (world w keyed PRNGKey(seed*1000 + w), step keys
split(fold_in(key, 2), steps)), modes 0 true / 3 within-life shuffled code (a new code at every bout open) / 4 memory
wiped at every bout open; second half of life = bouts closing at age >= A_max / 2; head accuracy at the hearing step,
1st vs later hearings; reach by distance; mean lifespan; paid intake."""
import argparse, glob, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import flax.serialization
import optax
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402
from w2.learnability import RecAgent  # noqa: E402
from w2.ep_agent import EpAgent  # noqa: E402
from w2.ep_agent_dev import EpAgentDev  # noqa: E402
from w2.worlds import world_kw  # noqa: E402
from w2.gae import gae  # noqa: E402
from w2 import named_item as NI  # noqa: E402
from w2.gate import HonestGate  # noqa: E402
from w2 import lookahead as LA  # noqa: E402
from w2 import relabel as RL  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ckpt as _ck  # noqa: E402

EAT = env.EAT


def own_meal_mask(hear, closed, ate, dead):
    """(B, T) arrays. A hearing step is labelled only if its bout's next end within the sequence is the agent eating the
    item (closed & ate); an end by T_max or death, or no end within the sequence, leaves it unlabelled."""
    B_, T_ = hear.shape; cur = np.zeros(B_, bool); mk = np.zeros((B_, T_), bool)
    for t in range(T_ - 1, -1, -1):
        end = closed[:, t] | dead[:, t]; cur = np.where(end, ate[:, t] & closed[:, t], cur); mk[:, t] = hear[:, t] & cur
    return mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", default="w2_out/nav_w13/*.msgpack"); ap.add_argument("--world", default="0.0013,8,0.001")
    ap.add_argument("--W", type=int, default=32); ap.add_argument("--N", type=int, default=4); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--steps", type=float, default=3e7); ap.add_argument("--oracle", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--minibatches", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-max", type=int, default=24); ap.add_argument("--curve-every", type=int, default=20)
    ap.add_argument("--ckpt", default=""); ap.add_argument("--max-seconds", type=float, default=0); ap.add_argument("--save-params", default="")
    ap.add_argument("--item-size", type=float, default=0.1); ap.add_argument("--hunger-gate", action="store_true")
    ap.add_argument("--learner", default="rec", choices=["rec", "ep"]); ap.add_argument("--slots", type=int, default=128)
    ap.add_argument("--hindsight", type=float, default=0.0); ap.add_argument("--shared-tok", action="store_true")
    ap.add_argument("--gate", default="old", choices=["old", "honest"]); ap.add_argument("--parent-waits", action="store_true", help="Cam Nov 15 ~12:05: the scripted parent never gets more than 2 cells ahead of its young")
    ap.add_argument("--head-own-meal", action="store_true", help="Cam Nov 15 ~11:05: label the head only on bouts the agent ENDED BY EATING the item (own-experience hindsight, as in the toy); bouts closed by T_max / death or not closed within the chunk are unlabelled")
    ap.add_argument("--demonstrator", action="store_true"); ap.add_argument("--demo-hearings", type=int, default=1)
    ap.add_argument("--policy", default="learner", choices=["learner", "follower", "copyparent"]); ap.add_argument("--head-only", action="store_true")
    ap.add_argument("--init-params", default=""); ap.add_argument("--frontier0", type=float, default=1.0)
    ap.add_argument("--early-stop-frontier", type=float, default=0.0, help="stop (counted FAIL) if the frontier has not passed this by half the steps")
    ap.add_argument("--eval-necessity", default="", help="EVAL: comma list of modes (0,3,4) on the necessity keys; no training")
    ap.add_argument("--eval-W", type=int, default=32); ap.add_argument("--eval-steps", type=int, default=3000)
    ap.add_argument("--d-fixed", type=int, default=0); ap.add_argument("--eval-d-max", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookahead", type=float, default=0.0); ap.add_argument("--la-max", type=int, default=12); ap.add_argument("--stop-at", type=float, default=0.0)
    ap.add_argument("--head-relabel-token", action="store_true", help="Cam/M Nov 16: cross-hearing relabel of the head (own record buffer only)")
    a = ap.parse_args()
    S = a.item_size; gated = a.hunger_gate; ep = a.learner == "ep"; KV = not os.environ.get("KV_OFF")
    cfg = env.Cfg(L=96, N=a.N, patch_sigma=1.0, reproduce=False, A_max=3000, **world_kw(a.world)); L, W, N, T = cfg.L, a.W, a.N, a.T
    nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob(a.navs))[0], "rb").read())
    Dx = 100 + 11 + 6 + 1 + 8
    if ep:
        rnet = EpAgentDev(S=a.slots, use_hind=a.hindsight > 0, shared_tok=a.shared_tok); Hd = 128 + a.slots * 11 + 1   # same params as EpAgent
    else:
        rnet = RecAgent(128); Hd = 128
    rp = rnet.init(jax.random.PRNGKey(a.seed), jnp.zeros((Hd,)), jnp.zeros((Dx,)))
    pp = {k: dict(v) for k, v in rp["params"].items()}
    for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
        pp[k] = dict(nav["params"][k])
    p = {"params": pp}
    if a.init_params:
        p = flax.serialization.from_bytes(p, open(a.init_params, "rb").read())
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)); ost = opt.init(p)
    H = cfg.E_set - S; K = a.demo_hearings; HALF = cfg.A_max // 2

    def drive(E):
        return jnp.abs(cfg.E_set - E) ** cfg.drive_n

    def dfn_of(dfront):
        if a.d_fixed:
            return lambda k: jnp.full((N,), float(a.d_fixed))
        if a.eval_necessity:                                                # the necessity check's own sampler (same lives / keys)
            return lambda k: jax.random.randint(k, (N,), 1, a.eval_d_max + 1).astype(jnp.float32)
        return LA.sampler(dfront, N, a.lookahead, a.la_max)                # uniform{1..frontier} (+ lookahead bouts with p)

    def world_reset(key, dfront):
        s = env.reset(cfg, key)
        item, opened = NI.open_bouts(NI.new_item(N), s, jax.random.fold_in(key, 1), dfn_of(dfront), L, (s.energy < H) if gated else jnp.ones(N, bool))
        code = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(jax.random.fold_in(key, 3), N))
        tok = code[jnp.arange(N), item["sec"]]; oh = jax.nn.one_hot(tok, 8).astype(jnp.int32) * opened[:, None]
        dem_on = opened & a.demonstrator & (jnp.zeros(N, jnp.int32) < K)
        return dict(s=s, item=item, h=jnp.zeros((N, Hd)), pa=jnp.zeros(N, jnp.int32), pr=jnp.zeros(N), code=code,
                    dem_pos=s.pos, dem_on=dem_on, hcnt=oh, gen=jnp.zeros(N, bool), lost=jnp.zeros(N, bool), hok=jnp.zeros(N, bool))

    def step(pp_, c, k, dfront, mode):
        s, item, h, pa, pr, code = c["s"], c["item"], c["h"], c["pa"], c["pr"], c["code"]
        k1, k2, k3, k4 = jax.random.split(k, 4); ar = jnp.arange(N)
        o = NI.observe(cfg, s, item)
        if a.demonstrator:
            o = NI.mark_demo(o, s, c["dem_pos"], c["dem_on"] & item["act"], L)
        x = jnp.concatenate([o, NI.hear(s, item, code, L), jax.nn.one_hot(pa, 6), pr[:, None], NI.oracle(item, a.oracle)], 1)
        h2, lg, v, aux = jax.vmap(lambda hh, xx: rnet.apply(pp_, hh, xx))(h, x)
        hl = aux[0] if ep else jnp.zeros((N, 8))
        samp = jax.random.categorical(k1, lg)
        if a.policy == "follower":
            act = NI.follower(s, item, L, samp)
        elif a.policy == "copyparent":
            act = NI.copy_parent(s, item, c["dem_pos"], c["dem_on"], L, samp)
        else:
            act = samp
        lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act[:, None], 1)[:, 0]
        item = dict(item, sat=item["sat"] | (item["act"] & (s.energy >= H)))  # no longer a hungry bout once E reached H
        tok_now = code[ar, item["sec"]]; hear_now = item["act"] & (item["t"] == 0)
        first_hear = hear_now & (c["hcnt"][ar, tok_now] <= 1)
        on_item = item["act"] & jnp.all(s.pos == item["pos"], -1) & (act == EAT)
        E0 = s.energy; s2, _, info = env.step(cfg, s, act, k2)
        E1 = jnp.clip(s2.energy + S * on_item * s2.alive, 0.0, 1.0); s2 = s2._replace(energy=E1)
        r = 20.0 * (drive(E0) - drive(E1))
        gain = cfg.eff * info["take"] + S * on_item; pd = NI.paid(E0, gain, cfg.E_set)
        dead = ~s2.alive; age_close = s2.age
        s3 = s2._replace(energy=jnp.where(dead, cfg.E0, s2.energy), age=jnp.where(dead, 0, s2.age), alive=jnp.ones(N, bool))
        item2, closed, opened = NI.close_and_open(item, on_item, dead, s3, k3, dfn_of(dfront), L, gated, H)
        wflag = closed & ~dead
        if ep:                                                              # the inborn write: one raw record per closed bout
            wslot = h2[:, 128 + a.slots * 11].astype(jnp.int32) % a.slots
            disp = NI.torus(s3.pos - item["start"], L).astype(jnp.float32)
            wrec = jnp.concatenate([jax.nn.one_hot(tok_now, 8), disp / 24.0, on_item[:, None].astype(jnp.float32)], 1)
            h2 = NI.mem_write(h2, 128, a.slots, 11, wflag, tok_now, disp, on_item)
        else:
            wslot = jnp.zeros(N, jnp.int32); wrec = jnp.zeros((N, 11))
        ncode = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(k4, N))
        code2 = jnp.where(dead[:, None], ncode, code); h3 = jnp.where(dead[:, None], 0.0, h2)
        if mode == 3:                                                       # within-life shuffled code: a new code at every bout open
            rc = jax.vmap(lambda kk: jax.random.permutation(kk, 8))(jax.random.split(jax.random.fold_in(k4, 7), N))
            code2 = jnp.where(opened[:, None], rc, code2)
        if mode == 4 and ep:                                                # memory wiped at every bout open
            h3 = h3.at[:, 128:].set(jnp.where(opened[:, None], 0.0, h3[:, 128:]))
        hcnt = jnp.where(dead[:, None], 0, c["hcnt"]); tok_new = code2[ar, item2["sec"]]
        hcnt = hcnt + (jax.nn.one_hot(tok_new, 8).astype(jnp.int32) * opened[:, None])
        dem_on = jnp.where(opened, a.demonstrator & (jnp.take_along_axis(hcnt, tok_new[:, None], 1)[:, 0] <= K), c["dem_on"] & ~closed)
        dem_mv, _ = NI.step_toward(c["dem_pos"], item["pos"], L)
        if a.parent_waits:                                                  # the parent waits: never more than 2 cells (Chebyshev) ahead of its young
            dem_mv = jnp.where((jnp.abs(NI.torus(dem_mv - s3.pos, L)).max(-1) <= 2)[:, None], dem_mv, c["dem_pos"])
            # Cam Nov 15 ~13:30: and it steps BACK toward the young when the young is more than 2 cells away
            back, _ = NI.step_toward(c["dem_pos"], s3.pos, L)
            far = jnp.abs(NI.torus(c["dem_pos"] - s3.pos, L)).max(-1) > 2
            dem_mv = jnp.where(far[:, None], back, dem_mv)
        dem_pos = jnp.where(opened[:, None], s3.pos, jnp.where(c["dem_on"][:, None], dem_mv, c["dem_pos"]))
        gen_now = c["gen"] | (item["act"] & (info["take"] > 0))               # generic (world) food eaten during this bout
        gen2 = jnp.where(opened | closed, False, gen_now)
        doff = NI.torus(c["dem_pos"] - s.pos, L); inview = jnp.abs(doff).max(-1) <= NI.R
        lost_now = c["lost"] | (c["dem_on"] & item["act"] & ~inview)          # DIAG: the parent left the 7x7 view during this bout
        hok_now = jnp.where(hear_now, hl.argmax(-1) == item["sec"], c["hok"])  # DIAG: the head's top sector at the hearing was right
        c2 = dict(s=s3, item=item2, h=h3, pa=act, pr=r, code=code2, dem_pos=dem_pos, dem_on=dem_on, hcnt=hcnt, gen=gen2,
                  lost=jnp.where(opened | closed, False, lost_now), hok=jnp.where(opened | closed, False, hok_now))
        info_ = dict(x=x, act=act, lp=lp, v=v, r=r, dead=dead.astype(jnp.float32), h=h, ate=on_item & ~dead, closed=closed & ~dead, gain=gain, pd=pd,
                     es=item["es"], d=item["d"], inb=item["act"], dem=c["dem_on"] & item["act"], hear=hear_now, first=first_hear, sec=item["sec"], hl=hl, age=age_close, alive=s.alive,
                     wflag=wflag, wslot=wslot, wrec=wrec, hung=closed & ~dead & ~item["sat"], hate=on_item & ~dead & ~item["sat"],
                     demstep=c["dem_on"] & item["act"], demfar=c["dem_on"] & item["act"] & (jnp.abs(NI.torus(c["dem_pos"] - s.pos, L)).max(-1) > 2), cl_item=closed & on_item, cl_death=closed & dead, cl_tmax=closed & ~on_item & ~dead, gen=gen_now, lost=lost_now, hok=hok_now)
        return c2, info_

    def rollout_world(pp_, c, key, dfront):
        return jax.lax.scan(lambda cc, kk: step(pp_, cc, kk, dfront, 0), c, jax.random.split(key, T))
    roll = jax.jit(jax.vmap(rollout_world, in_axes=(None, 0, 0, None)))

    def kv_masks(bt):
        """(T, S + T) validity of [start-of-sequence slots ; this sequence's written records] at each replay step t:
        the memory the rollout read at step t, rebuilt exactly from write times, slots and death resets."""
        S_ = a.slots; tt = jnp.arange(T)
        mem0 = bt["h0"][128: 128 + S_ * 11].reshape(S_, 11); filled0 = mem0[:, :8].sum(-1) > 0
        lastreset = jax.lax.cummax(jnp.where(bt["first"] > 0, tt, -1))                       # wipe before step t
        ef = bt["wflag"] > 0; es = bt["wslot"]
        Es = jax.nn.one_hot(es, S_) * ef[:, None]                                           # (T, S)
        C = jnp.cumsum(Es, 0) - Es                                                          # writes to slot s at steps < t
        old_ok = filled0[None, :] & (lastreset[:, None] < 0) & (C <= 0)
        same = (es[:, None] == es[None, :]) & ef[None, :]                                   # [e, e']
        later = (tt[None, :, None] > tt[None, None, :]) & (tt[None, :, None] < tt[:, None, None])   # [t, e', e]: e < e' < t
        sup = jnp.any(later & jnp.swapaxes(same, 0, 1)[None], 1)                            # [t, e] superseded before t
        ev_ok = ef[None, :] & (tt[None, :] < tt[:, None]) & (tt[None, :] >= lastreset[:, None]) & ~sup
        return mem0, jnp.concatenate([old_ok, ev_ok], 1)

    def seq_loss_kv(pp_, bt):
        mem0, valid = kv_masks(bt)
        K0, V0 = rnet.apply(pp_, mem0, method=EpAgentDev.kv); Ke, Ve = rnet.apply(pp_, bt["wrec"], method=EpAgentDev.kv)
        K = jnp.concatenate([K0, Ke]); V = jnp.concatenate([V0, Ve])
        def body(hc, xa):
            xx, f, vm = xa; hc = jnp.where(f > 0, 0.0, hc)
            h2, lg, v, hl = rnet.apply(pp_, hc, xx, K, V, vm, method=EpAgentDev.step_kv); return h2, (lg, v, hl)
        _, (lg, v, hl) = jax.lax.scan(body, bt["h0"][:128], (bt["x"], bt["first"], valid))
        return lg, v, hl

    def seq_loss(pp_, bt):
        if ep and KV:
            lg, v, hl = seq_loss_kv(pp_, bt)
            return ppo_terms(lg, v, hl, bt)
        return seq_loss_old(pp_, bt)

    def seq_loss_old(pp_, bt):
        def body(hc, xa):
            xx, f, hm = xa; hc = jnp.where(f > 0, 0.0, hc)
            hh = jnp.concatenate([hc, hm[128:]]) if ep else hc               # the memory part is the world's (stored); the GRU part is replayed
            h2, lg, v, aux = rnet.apply(pp_, hh, xx); return h2[:128], (lg, v, aux[0] if ep else jnp.zeros(8))
        _, (lg, v, hl) = jax.lax.scan(body, bt["h0"][:128], (bt["x"], bt["first"], bt["hm"]))
        return ppo_terms(lg, v, hl, bt)

    def ppo_terms(lg, v, hl, bt):
        ls = jax.nn.log_softmax(lg); lp = jnp.take_along_axis(ls, bt["act"][:, None], 1)[:, 0]
        ratio = jnp.exp(jnp.clip(lp - bt["lp"], -20.0, 20.0)); adv = bt["adv"]
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(hl), bt["sec"][:, None], 1)[:, 0] * bt["hear"]
        hloss = a.hindsight * ce.sum() / jnp.maximum(bt["hear"].sum(), 1.0)
        if a.head_only:
            return hloss
        return (pg + 0.5 * (v - bt["ret"]) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)).mean() + hloss

    @jax.jit
    def upd(pp_, os_, bt):
        g = jax.grad(lambda q: jax.vmap(seq_loss, in_axes=(None, 0))(q, bt).mean())(pp_)
        u, os_ = opt.update(g, os_, pp_); return optax.apply_updates(pp_, u), os_

    # ------------------------------------------------------------------ evaluation on the necessity lives / keys
    if a.eval_necessity:
        def run_eval(pp_, seed, mode):
            key = jax.random.PRNGKey(seed); c = world_reset(key, float(a.eval_d_max))
            def body(acc, k):
                c, A = acc; c2, f = step(pp_, c, k, float(a.eval_d_max), mode)
                late = f["age"] >= HALF; ok = f["hl"].argmax(-1) == f["sec"]
                dh = jax.nn.one_hot(f["d"].astype(jnp.int32), 25)
                A = dict(A, alive=A["alive"] + f["alive"].sum(), deaths=A["deaths"] + f["dead"].sum(), oldage=A["oldage"] + (f["dead"] * (f["age"] >= cfg.A_max)).sum(), paid=A["paid"] + f["pd"].sum(),
                         ate=A["ate"] + f["ate"].sum(), closed=A["closed"] + f["closed"].sum(),
                         ate2=A["ate2"] + (f["ate"] & late).sum(), closed2=A["closed2"] + (f["closed"] & late).sum(),
                         ate2h=A["ate2h"] + (f["hate"] & late).sum(), closed2h=A["closed2h"] + (f["hung"] & late).sum(),
                         ateh=A["ateh"] + f["hate"].sum(), closedh=A["closedh"] + f["hung"].sum(),
                         demstep=A["demstep"] + f["demstep"].sum(), demfar=A["demfar"] + f["demfar"].sum(),
                         h1=A["h1"] + (ok & f["first"]).sum(), n1=A["n1"] + f["first"].sum(),
                         hL=A["hL"] + (ok & f["hear"] & ~f["first"]).sum(), nL=A["nL"] + (f["hear"] & ~f["first"]).sum(),
                         ateD=A["ateD"] + (f["ate"] & f["dem"]).sum(), closedD=A["closedD"] + (f["closed"] & f["dem"]).sum(),
                         cd=A["cd"] + (dh * f["closed"][:, None]).sum(0), ad=A["ad"] + (dh * f["ate"][:, None]).sum(0))
                for nm_, m_ in (("item", f["cl_item"]), ("tmax", f["cl_tmax"]), ("death", f["cl_death"])):
                    for dn_, dm_ in (("all", jnp.ones(N, bool)), ("dem", f["dem"])):
                        A = dict(A, **{f"c_{nm_}_{dn_}": A[f"c_{nm_}_{dn_}"] + (m_ & dm_).sum(), f"g_{nm_}_{dn_}": A[f"g_{nm_}_{dn_}"] + (m_ & dm_ & f["gen"]).sum()})
                endb = f["cl_item"] | f["cl_tmax"] | f["cl_death"]; nd = ~f["dem"]
                for nm_, m_ in (("dem_kept", f["dem"] & ~f["lost"]), ("dem_lost", f["dem"] & f["lost"]), ("late_hok", nd & f["hok"]), ("late_hbad", nd & ~f["hok"])):
                    A = dict(A, **{f"n_{nm_}": A[f"n_{nm_}"] + (endb & m_).sum(), f"i_{nm_}": A[f"i_{nm_}"] + (f["cl_item"] & m_).sum(),
                                   f"t_{nm_}": A[f"t_{nm_}"] + (f["cl_tmax"] & m_).sum(), f"d_{nm_}": A[f"d_{nm_}"] + (f["cl_death"] & m_).sum(),
                                   f"hn_{nm_}": A[f"hn_{nm_}"] + (f["hung"] & m_).sum(), f"hi_{nm_}": A[f"hi_{nm_}"] + (f["hate"] & m_).sum()})
                cnt = A["cnt"] + f["closed"].astype(jnp.float32); end = f["dead"] > 0          # records written in the current life (one per closed bout)
                A = dict(A, mx=jnp.maximum(A["mx"], jnp.max(cnt)), ovf=A["ovf"] + (end & (cnt > a.slots)).sum(),
                         over=A["over"] + jnp.where(end, jnp.maximum(cnt - a.slots, 0.0), 0.0).sum(), recs=A["recs"] + jnp.where(end, cnt, 0.0).sum(),
                         cnt=jnp.where(end, 0.0, cnt))
                return (c2, A), None
            z = jnp.float32(0.0)
            A0 = dict(**{f"{p_}_{n_}": z for p_ in ("n", "i", "t", "d", "hn", "hi") for n_ in ("dem_kept", "dem_lost", "late_hok", "late_hbad")}, ate2h=z, closed2h=z, ateh=z, closedh=z, demstep=z, demfar=z, **{f"{p_}_{n_}_{d_}": z for p_ in ("c", "g") for n_ in ("item", "tmax", "death") for d_ in ("all", "dem")}, alive=z, deaths=z, oldage=z, ateD=z, closedD=z, cnt=jnp.zeros(N), mx=z, ovf=z, over=z, recs=z, paid=z, ate=z, closed=z, ate2=z, closed2=z, h1=z, n1=z, hL=z, nL=z, cd=jnp.zeros(25), ad=jnp.zeros(25))
            (cF, A), _ = jax.lax.scan(body, (c, A0), jax.random.split(jax.random.fold_in(key, 2), a.eval_steps))
            cnt = A["cnt"]                                                   # lives still open at the end of the horizon
            A = dict(A, ovf=A["ovf"] + (cnt > a.slots).sum(), over=A["over"] + jnp.maximum(cnt - a.slots, 0.0).sum(), recs=A["recs"] + cnt.sum())
            return A
        out = dict(cfg=vars(a), modes={})
        for mode in [int(m) for m in a.eval_necessity.split(",")]:
            fn = jax.jit(lambda pp_, sd: run_eval(pp_, sd, mode))
            per = [jax.tree_util.tree_map(np.asarray, fn(p, a.seed * 1000 + w)) for w in range(a.eval_W)]
            tot = {k_: sum(q[k_] for q in per) for k_ in per[0] if k_ not in ('cnt', 'mx')}
            lives = a.eval_W * N + tot["deaths"] - tot["oldage"]   # an old-age death can only fall on the final step: it starts no counted life
            res = dict(mean_life_steps=float(tot["alive"] / lives), lives=float(lives), deaths=float(tot["deaths"]), paid_per_life=float(tot["paid"] / lives),
                       paid_per_step=float(tot["paid"] / tot["alive"]), item_reach=float(tot["ate"] / max(tot["closed"], 1)),
                       reach_2nd_half=float(tot["ate2"] / max(tot["closed2"], 1)),
                       hungry_reach=float(tot["ateh"] / max(tot["closedh"], 1)), hungry_reach_2nd_half=float(tot["ate2h"] / max(tot["closed2h"], 1)),
                       hungry_bouts_2nd_half=float(tot["closed2h"]),
                       demo_steps=float(tot["demstep"]), demo_frac_parent_over_2_cells=float(tot["demfar"] / max(tot["demstep"], 1)),
                       closures={f"{n_}_{d_}": [float(tot[f"c_{n_}_{d_}"]), float(tot[f"g_{n_}_{d_}"])] for n_ in ("item", "tmax", "death") for d_ in ("all", "dem")},
                       max_records_per_life=float(max(q["mx"] for q in per)), lives_overflowing=float(tot["ovf"]),
                       frac_lives_overflowing=float(tot["ovf"] / lives), frac_records_overwritten=float(tot["over"] / max(tot["recs"], 1)),
                       reach_demonstrated=float(tot["ateD"] / max(tot["closedD"], 1)), bouts_demonstrated=float(tot["closedD"]), bouts_2nd_half=float(tot["closed2"]),
                       head_acc_first=float(tot["h1"] / max(tot["n1"], 1)), head_n_first=float(tot["n1"]),
                       head_acc_later=float(tot["hL"] / max(tot["nL"], 1)), head_n_later=float(tot["nL"]),
                       diag={n_: {p_: float(tot[f"{p_}_{n_}"]) for p_ in ("n", "i", "t", "d", "hn", "hi")} for n_ in ("dem_kept", "dem_lost", "late_hok", "late_hbad")},
                       reach_by_d={int(d_): [round(float(tot["ad"][d_] / tot["cd"][d_]), 4), int(tot["cd"][d_])] for d_ in range(1, 25) if tot["cd"][d_] > 0},
                       per_world=[[float(q["alive"]), float(a.N + q["deaths"] - q["oldage"]), float(q["ate2"]), float(q["closed2"]), float(q["paid"]), float(q["ate2h"]), float(q["closed2h"])] for q in per])
            out["modes"][mode] = res; print("EVAL_MODE", mode, json.dumps({k_: v_ for k_, v_ in res.items() if k_ != "per_world"}), flush=True)
        json.dump(out, open(a.out, "w"), indent=1); return

    # ------------------------------------------------------------------ training
    rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed + 1)
    dfront = a.frontier0; recent = []; GATE = HonestGate(); steps = 0; it = 0; curve = []; nan_run = 0; hacc = [0, 0, 0, 0]
    c = jax.vmap(world_reset, in_axes=(0, None))(jax.random.split(key, W), dfront)
    Ck = _ck.load(a.ckpt) if a.ckpt else None
    if Ck is not None:
        tj = lambda t_: jax.tree_util.tree_map(jnp.asarray, t_)  # noqa: E731
        p, ost, c, key = tj(Ck["p"]), tj(Ck["ost"]), tj(Ck["c"]), jnp.asarray(Ck["key"])
        dfront, recent, steps, it, curve = Ck["dfront"], Ck["recent"], Ck["steps"], Ck["it"], Ck["curve"]; rng.bit_generator.state = Ck["rng"]
        if "gate" in Ck:
            GATE.load(Ck["gate"])
        print(f"resumed at {steps:,} steps, frontier {dfront}", flush=True)
    t0 = time.time(); nupd = a.policy != "learner" and not a.head_only
    while steps < a.steps:
        if a.ckpt and it % 20 == 0 and it > 0:
            _ck.save(a.ckpt, dict(p=p, ost=ost, c=c, key=key, dfront=dfront, recent=recent, steps=steps, it=it, curve=curve, rng=rng.bit_generator.state, gate=GATE.state()))
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                print(f"CHUNK_STOP at {steps:,}", flush=True); sys.exit(3)
        if a.stop_at and steps >= a.stop_at:
            if a.ckpt:
                _ck.save(a.ckpt, dict(p=p, ost=ost, c=c, key=key, dfront=dfront, recent=recent, steps=steps, it=it, curve=curve, rng=rng.bit_generator.state, gate=GATE.state()))
            if a.save_params:
                open(a.save_params, "wb").write(flax.serialization.to_bytes(p))
            print(f"PAUSE_AT {steps:,} frontier {dfront}", flush=True); sys.exit(5)
        if a.early_stop_frontier and steps >= a.steps / 2 and dfront <= a.early_stop_frontier:
            print(f"EARLY_STOP frontier {dfront} at {steps:,} (counted FAIL)", flush=True); break
        key, k = jax.random.split(key); c, R_ = roll(p, c, jax.random.split(k, W), jnp.float32(dfront))
        R_ = {k_: np.asarray(v_) for k_, v_ in R_.items()}                    # (W, T, N, ...)
        sq = lambda z: np.swapaxes(z, 1, 2).reshape((W * N, T) + z.shape[3:])  # noqa: E731
        r_, v_, d_ = sq(R_["r"]), sq(R_["v"]), sq(R_["dead"])
        adv = gae(r_, v_, d_, 0.99, 0.95); ret = adv + v_; A_ = (adv - adv.mean()) / (adv.std() + 1e-8)
        first = np.concatenate([np.zeros((W * N, 1)), d_[:, :-1]], 1)
        if a.head_relabel_token:                                             # targets from the young's own record buffer only (never the code table)
            h0f = R_["h"][:, 0].reshape(W * N, Hd); m0 = h0f[:, 128: 128 + a.slots * 11]; p0 = h0f[:, 128 + a.slots * 11]
            RT, RM, _, RST = RL.relabel(sq(R_["hear"]), sq(R_["x"])[..., 100:108], m0, p0, sq(R_["wflag"]), sq(R_["wrec"]), sq(R_["dead"]) > 0, a.slots)
            HEAR = RM; TGT = np.where(RM, RT, 0).astype(np.int32); RLST = RST
        else:
            HEAR = own_meal_mask(sq(R_["hear"]), sq(R_["closed"]), sq(R_["ate"]), sq(R_["dead"]) > 0) if a.head_own_meal else sq(R_["hear"])
            TGT = sq(R_["sec"]); RLST = None
        hm = sq(R_["h"]) if (ep and not KV) or os.environ.get("KV_COMPARE") else np.zeros((W * N, T, 1), np.float32)
        bt = dict(x=jnp.asarray(sq(R_["x"])), h0=jnp.asarray(R_["h"][:, 0].reshape(W * N, Hd)), act=jnp.asarray(sq(R_["act"])), lp=jnp.asarray(sq(R_["lp"])),
                  adv=jnp.asarray(A_.astype(np.float32)), ret=jnp.asarray(ret.astype(np.float32)), first=jnp.asarray(first.astype(np.float32)),
                  hm=jnp.asarray(hm), sec=jnp.asarray(TGT),
                  hear=jnp.asarray(HEAR.astype(np.float32)),
                  wflag=jnp.asarray(sq(R_["wflag"]).astype(np.float32)), wslot=jnp.asarray(sq(R_["wslot"])), wrec=jnp.asarray(sq(R_["wrec"])))
        if os.environ.get("KV_COMPARE"):                                     # the same batch through both replays: loss and gradient
            l_old, g_old = jax.value_and_grad(lambda q: jax.vmap(seq_loss_old, in_axes=(None, 0))(q, bt).mean())(p)
            l_new, g_new = jax.value_and_grad(lambda q: jax.vmap(seq_loss, in_axes=(None, 0))(q, bt).mean())(p)
            gd = max(float(jnp.abs(x_ - y_).max()) for x_, y_ in zip(jax.tree_util.tree_leaves(g_old), jax.tree_util.tree_leaves(g_new)))
            gm = max(float(jnp.abs(x_).max()) for x_ in jax.tree_util.tree_leaves(g_old))
            def memcheck(b1):                                                # rebuilt memory == stored memory at every step (order-free)
                mem0, valid = kv_masks(b1); ent = jnp.concatenate([mem0, b1["wrec"]])
                stored = b1["hm"][:, 128: 128 + a.slots * 11].reshape(T, a.slots, 11); sf = stored[:, :, :8].sum(-1) > 0
                c1 = valid.sum(1) - sf.sum(1); c2 = jnp.abs((valid[:, :, None] * ent[None]).sum(1) - (sf[:, :, None] * stored).sum(1)).max()
                return jnp.abs(c1).max(), c2
            mc = jax.vmap(memcheck)(bt); nw = int(bt["wflag"].sum()); nd = int(bt["first"].sum())
            print("KV_MEMCHECK it %d count mismatch %d ; content max|d| %.3g ; writes %d deaths %d" % (it, int(mc[0].max()), float(mc[1].max()), nw, nd), flush=True)
            print("KV_COMPARE it %d loss old %.7g new %.7g ; grad max|d| %.3g (max |grad| %.3g)" % (it, float(l_old), float(l_new), gd, gm), flush=True)
        p_prev, o_prev = p, ost
        for _ in range(0 if nupd else a.epochs):
            for mb in np.array_split(rng.permutation(W * N), a.minibatches):
                p, ost = upd(p, ost, {k_: v_[mb] for k_, v_ in bt.items()})
        if not all(bool(jnp.isfinite(l_).all()) for l_ in jax.tree_util.tree_leaves(p)):
            p, ost = p_prev, o_prev; nan_run += 1; print(f"NAN_SKIP {it}", flush=True)
            if nan_run >= 10:
                print("NAN_STOP", flush=True); break
        else:
            nan_run = 0
        steps += W * N * T; it += 1
        closed, ate_i = R_["closed"], R_["ate"]; n_close = int(closed.sum()); n_eat = int(ate_i.sum()); st_h = (int(R_["hate"].sum()), int(R_["hung"].sum()))
        ok = R_["hl"].argmax(-1) == R_["sec"]; fh, hh_ = R_["first"], R_["hear"] & ~R_["first"]
        hacc = [hacc[0] + int((ok & fh).sum()), hacc[1] + int(fh.sum()), hacc[2] + int((ok & hh_).sum()), hacc[3] + int(hh_.sum())]
        recent.extend([1] * n_eat + [0] * max(n_close - n_eat, 0)); recent = recent[-500:]
        inr = LA.in_range(R_["d"], dfront); n_la = int((R_["hung"] & ~inr).sum())   # lookahead bouts are not scored
        if a.gate == "honest":
            if GATE.add(int((R_["hate"] & inr).sum()), int((R_["hung"] & inr).sum())) and dfront < a.d_max:
                dfront += 1; print(f"GATE_UP frontier -> {dfront:.0f} at {steps:,} (last windows {GATE.scores[-2:]})", flush=True)
        elif len(recent) >= 500 and np.mean(recent) >= 0.8 and dfront < a.d_max:
            dfront += 1; recent = []
        if it % a.curve_every == 0:
            curve.append(dict(steps=steps, frontier=dfront, item_reach=n_eat / max(n_close, 1), food_per_step=float(R_["gain"].mean()), paid_per_step=float(R_["pd"].mean()),
                              frac_in_bout=float(R_["inb"].mean()), head_acc_first=hacc[0] / max(hacc[1], 1), head_acc_later=hacc[2] / max(hacc[3], 1),
                              n_first=hacc[1], n_later=hacc[3], n_hungry_lookahead=n_la, relabel=RLST, n_labels=int(np.asarray(HEAR).sum()), hungry_reach=st_h[0] / max(st_h[1], 1), gate_last=(GATE.scores[-1] if GATE.scores else None), sec=round(time.time() - t0, 1)))
            hacc = [0, 0, 0, 0]; print("CURVE", json.dumps(curve[-1]), flush=True)
    if a.save_params:
        open(a.save_params, "wb").write(flax.serialization.to_bytes(p))
    out = dict(cfg=vars(a), steps=steps, frontier=dfront, curve=curve, gate_scores=GATE.scores, train_seconds=round(time.time() - t0, 1), throughput=steps / max(time.time() - t0, 1e-9))
    json.dump(out, open(a.out, "w"), indent=1); print("TRAINED", json.dumps(dict(steps=steps, frontier=dfront)), flush=True)


if __name__ == "__main__":
    main()
