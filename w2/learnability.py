"""W2 learnability gate (Cam Nov 9 ~20:15; levers Nov 10 ~00:35): can a newborn learn to LISTEN within one lifetime?

Per world: 4 frozen speakers ("beacons": stationary, immortal, at the founders' cluster) broadcasting a fixed scripted
code (the 8-sector direction from the beacon to its nearest food patch), NL learners (fresh W2 navigators; new input
rows / heads at zero or fresh init; own per-agent learning; natural lives), and optionally NE elders (--copy):
immortal scripted listeners that forage by the heard code, the demonstrators the young copy.
A learner hears the nearest beacon within Chebyshev radius 9: one-hot(sector) + flag + offset / 9.
TEST: at death the learner's end-of-life weights are frozen. Food eaten over 300 steps from a start near the beacons
(learner alone, no elders), true code vs a cross-direction swap (every heard sector shifted by a random 1..7),
16 paired episodes. A life PASSES iff drop >= 30% AND paired drop > 2 SE. GATE PASS iff >= 50% of lives pass.
LEVERS:
  --juvenile J    provisioning: energy kept >= E0 for the first J steps of life; the young stays within hearing range
                  (default 500: the W2 default from Nov 10; the beacon world has no parents, so nobody pays)
  --pred C        predictive learning: a self-supervised head on the shared trunk (heard messages are inputs)
                  predicting the next view window (98), own energy change (1), and the max food in view 4 steps ahead
                  (1); MSE, weight C (0 = off)
  --replay K      sleep replay: after each rollout, K extra passes over the agent's own memory since birth (the last
                  --mem rollouts), predictive loss only
  --copy C        copying elders' actions (M approved, Nov 10): behaviour cloning on the (obs, action) of elders within
                  hearing range, young only (age < --young), weight C
  --sanity        positive control: the learners themselves are scripted listeners
usage: python w2/learnability.py --navs "w2_out/nav/*.msgpack" [--updates 150] [--pred 1.0] [--replay 8] [--copy 1.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env  # noqa: E402
from w2.learn import Net  # noqa: E402
from w2.worlds import world_kw  # noqa: E402

NB, R_HEAR, HD = 4, 9, 11


class NetP(nn.Module):
    """the navigator MLP (same layer names, so navigator weights load) + a predictive head on the shared trunk"""
    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(128, name="Dense_0")(x)); h = nn.tanh(nn.Dense(128, name="Dense_1")(h))
        return nn.Dense(6, name="Dense_2")(h), nn.Dense(1, name="Dense_3")(h)[..., 0], nn.Dense(100, name="pred")(h)


class RecAgent(nn.Module):
    """Stage 1 (Cam Nov 12): the navigator trunk (frozen; loaded from navigator 0) + a GRU memory that reads the heard
    token, the trunk's view features, the previous action and reward. A zero-initialised head adds to the policy and the
    value, so behaviour starts exactly as the navigator's. Three approved drives as auxiliary heads: the direction of
    the next meal (hindsight label), the parent's next move, the next view."""
    hd: int = 128
    bound: float = 0.0          # > 0: the side head's logit contribution is bound * tanh(raw / bound)

    @nn.compact
    def __call__(self, h, x):
        obs, side = x[..., :100], jnp.concatenate([x[..., 100:], x[..., 98:100]], -1)   # heard (+ speaker offset), prev act / reward, own energy + age
        t = nn.tanh(nn.Dense(128, name="Dense_0")(obs)); t = nn.tanh(nn.Dense(128, name="Dense_1")(t))
        lg_nav = nn.Dense(6, name="Dense_2")(t); v_nav = nn.Dense(1, name="Dense_3")(t)[..., 0]
        z = nn.tanh(nn.Dense(64, name="side_enc")(jnp.concatenate([side, t], -1)))
        h, y = nn.GRUCell(self.hd, name="gru")(h, z)
        zi = nn.initializers.zeros
        sh = nn.Dense(6, name="side_head", kernel_init=zi, bias_init=zi)(y)
        lg = lg_nav + (self.bound * jnp.tanh(sh / self.bound) if self.bound > 0 else sh)
        v = v_nav + nn.Dense(1, name="side_value", kernel_init=zi, bias_init=zi)(y)[..., 0]
        meal = nn.Dense(9, name="aux_meal")(y); other = nn.Dense(5, name="aux_other")(y)
        view = nn.Dense(49, name="aux_view")(jnp.concatenate([t, y], -1))
        return h, lg, v, (meal, other, view)


def sector(off):
    ang = jnp.arctan2(off[..., 1].astype(jnp.float32), off[..., 0].astype(jnp.float32))
    return jnp.mod(jnp.round(ang / (2 * jnp.pi / 8)).astype(jnp.int32), 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navs", required=True); ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--worlds", type=int, default=16); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--juvenile", type=int, default=500)
    ap.add_argument("--long-life", action="store_true"); ap.add_argument("--random-code", action="store_true")
    ap.add_argument("--pred", type=float, default=0.0); ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--mem", type=int, default=4); ap.add_argument("--copy", type=float, default=0.0)
    ap.add_argument("--young", type=int, default=1000); ap.add_argument("--n-elders", type=int, default=2)
    ap.add_argument("--n-learners", type=int, default=4)
    ap.add_argument("--eval-eps", type=int, default=16); ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--copy-replay", action="store_true", help="Cam Nov 10: the K replay passes also cover the copy loss (supervised)")
    ap.add_argument("--young-lr", type=float, default=1.0, help="young plasticity: update x this at birth, linear to x1 by the end of --juvenile")
    ap.add_argument("--diag", type=int, default=0, help="free diagnostic: on this many lives, does the predictive head use the message?")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="w2_out/learnability.json")
    ap.add_argument("--world", default="proto", help='"proto" or "d0,P,relocate" (Cam Nov 10 4b)')
    ap.add_argument("--beacon-spread", type=int, default=2, help="beacons at (+-k, +-k) around the cluster centre (Cam Nov 10 coverage: spread)")
    ap.add_argument("--beacon-walk", action="store_true", help="beacons random-walk at agent speed (one random move per step)")
    ap.add_argument("--seek-age", type=int, default=0, help="young-seek applies while age < this (0: during the juvenile period, as before); the same scaffolding in every arm, independent of the juvenile lever")
    ap.add_argument("--young-seek", action="store_true", help="a young that hears nothing steps toward the nearest beacon (was: stays)")
    ap.add_argument("--no-train", action="store_true", help="apparatus gate: no learning; per-life coverage + life test (+ blind on the same lives)")
    ap.add_argument("--test", default="shift", choices=["shift", "heard25"], help="heard25: swap = a derangement among the sectors that life heard >= --min-heard times")
    ap.add_argument("--min-heard", type=int, default=25); ap.add_argument("--test-window", type=int, default=300)
    ap.add_argument("--code-per-life", action="store_true", help="Nov 11 (exploratory): each life hears the beacons through its own random sector->token code; the scripted listener decodes with the inverse (stays a perfect decoder)")
    ap.add_argument("--ic-probe", action="store_true", help="Nov 12 stage-1 diagnostic (EXPLORE): with --no-train --code-per-life, record each life's input stream and train a GRU across lives to predict the true sector behind each heard token; accuracy vs tokens heard so far, against a memoryless baseline")
    ap.add_argument("--obs-listener", default="", choices=["", "observer", "true"], help="Cam Nov 12 plug-in check: lives run with navigator behaviour (+ parent); the frozen life test then runs the scripted listener acting through a per-life token->sector table: observer = the life's counting-decoder argmax (beacon cues + parent), true = the life's real inverse code (control)")
    ap.add_argument("--agent", default="ff", choices=["ff", "recurrent"], help="recurrent: Cam Nov 12 stage 1 (one shared GRU network; slow weights across lives; memory within a life)")
    ap.add_argument("--rec-hidden", type=int, default=128); ap.add_argument("--rec-lr", type=float, default=3e-4)
    ap.add_argument("--aux", type=float, default=0.5, help="weight of the three auxiliary drive losses")
    ap.add_argument("--readout-worlds", type=int, default=32); ap.add_argument("--readout-at", default="", help="agent-step milestones for stage-1 readouts, e.g. 1e8,3e8,1e9")
    ap.add_argument("--test-energy", type=float, default=0.0, help="probe episodes start the learner at this energy (0 = unchanged); stage-1 readouts use 0.3 if unset")
    ap.add_argument("--eval-updates", type=int, default=60, help="recurrent: updates with FROZEN weights after training; lives born then are tested")
    ap.add_argument("--code-family", default="A", choices=["A", "B"], help="per-life codes drawn from this family (A = training)")
    ap.add_argument("--crossfoster", action="store_true", help="Cam Nov 12 stage-3 instrument calibration (with --code-per-life --no-train): foster-coded vs birth-coded probes for a perfect foster-learner, a perfect birth-decoder and a deaf navigator")
    ap.add_argument("--parent-leave", type=int, default=0, help="age at which the parent leaves (0 = the juvenile period)")
    ap.add_argument("--eval-eligible-only", action="store_true", help="run the life test only on lives that heard >= --min-sectors sectors >= --min-heard x (eligibility rate recorded before the filter)")
    ap.add_argument("--parent", action="store_true", help="Cam Nov 12: one scripted parent per newborn (elder l pairs with learner l) stays within 2-3 cells through the juvenile period, then leaves; young-seek targets the parent; no demonstrators in the life test")
    ap.add_argument("--demonstrators", action="store_true", help="scripted elders (perfect followers of the heard code) present even without a copy loss (stage-1 demonstrator arm)")
    ap.add_argument("--io-decoder", action="store_true", help="with --ic-probe collection: run the IDEAL-OBSERVER counting decoder (no training) instead of the GRU probe (Cam Nov 12)")
    ap.add_argument("--ic-steps", type=int, default=400); ap.add_argument("--ic-len", type=int, default=640)
    ap.add_argument("--decode-identity", action="store_true", help="control for --code-per-life: the scripted listener ignores the code (decodes tokens as sectors); must FAIL when codes are on")
    ap.add_argument("--min-sectors", type=int, default=2, help="Cam Nov 10: a life is scorable only if it heard >= this many sectors >= --min-heard times (ruled: 5); pass rates over eligible lives")
    ap.add_argument("--score", default="window", choices=["window", "last-heard"], help="last-heard: in the heard25 test, only food eaten while the last heard TRUE sector is one the life heard >= --min-heard times counts")
    ap.add_argument("--test-start", default="fresh", choices=["fresh", "life"], help="life: each probe episode starts in the life's own world at its end (patch centres + beacon positions), the learner beside a random beacon")
    ap.add_argument("--snap-ages", default="", help="comma-separated ages: snapshot each life's weights there; the heard25 test also runs on them (steps to first reliance)")
    ap.add_argument("--forage-check", action="store_true", help="stage 1 (Cam Nov 12): with --ckpt, compare the trained agent with the navigator (side heads zeroed) on the same fresh worlds, natural energy: KM lifespan, food/step by age, deaths by age and cause; then exit")
    ap.add_argument("--forage-worlds", type=int, default=32); ap.add_argument("--forage-updates", type=int, default=120)
    ap.add_argument("--grad-diag", action="store_true", help="stage 1: with --ckpt, build one training batch, report per-loss gradient norms per module, side-head vs trunk logit norms, entropy, actions, value/advantage stats; then exit")
    ap.add_argument("--ppo-min-age", type=int, default=0, help="stage 1: steps of lives younger than this leave the PPO loss (diagnosis of the juvenile energy floor)")
    ap.add_argument("--kl-nav", type=float, default=0.0, help="stage 1: coefficient of a per-step KL(agent || frozen navigator) penalty in the loss")
    ap.add_argument("--kl-silent-only", action="store_true", help="stage 1: apply the --kl-nav penalty only on steps with no token heard")
    ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--side-bound", type=float, default=0.0, help="stage 1: bound the side head's logit contribution to +-B (B tanh(raw/B)); 0 = unbounded")
    ap.add_argument("--token-influence-every", type=int, default=0, help="stage 1: every N updates log token influence = KL(policy | true token || policy | swapped token) on heard steps, same memory")
    ap.add_argument("--mask-forced", action="store_true", help="stage 1: drop steps whose action the parent-seek reflex overrode from the PPO loss (they were not sampled from the policy)")
    ap.add_argument("--return-check", action="store_true", help="stage 1 (Cam Nov 12): with --ckpt, the training objective (discounted return of the actual per-step training reward) for the navigator, the checkpoint agent and scripted stand-still agents on the same fresh worlds; then exit")
    ap.add_argument("--redraw-probe", type=int, default=0, help="stage 1 (Cam Nov 12): with --ckpt, re-draw each life's code at this age (first rollout boundary after it); linear probe trained on pre-re-draw rows, scored on post rows by tokens heard since the re-draw, vs the wiped baseline; then exit")
    ap.add_argument("--readout-worlds-at", default="", help="stage 1: readout worlds per milestone, e.g. 32,32,64 (overrides --readout-worlds)")
    ap.add_argument("--forage-stop", action="store_true", help="stage 1 (Cam's foraging rule, in-run): after each milestone readout, forage-check the live weights; STOP at 1st milestone if the lower ratio < 0.5; at the 2nd if < 0.5 or still falling from the 50-90% band")
    ap.add_argument("--reward-variants", default="", help="stage 1 return check: extra rewards scored on the same trajectories, e.g. death:1,alive:0.001,topup")
    ap.add_argument("--wean-to", type=int, default=0, help="gradual weaning (Cam Nov 12): after the --juvenile top-up the parent supplies w*(d0+d_move) energy per step up to E0, w falling linearly to 0 at this age (set it to --parent-leave); 0 = off")
    ap.add_argument("--forage-at", default="", help="stage 1: forage checks (agent vs navigator) at these agent-step counts during training, e.g. 5e6,1e7")
    ap.add_argument("--final-return-check", action="store_true", help="stage 1: after training, the return check by age band on the final weights, into the results JSON")
    ap.add_argument("--forage-control", choices=["none", "damaged"], default="none")
    ap.add_argument("--diag-only", action="store_true", help="with --diag (and e.g. --ckpt of a finished run): run the predictive-head swap check and exit")
    ap.add_argument("--log-levers", action="store_true", help="diagnostic (Cam Nov 10, 4a): per update, each lever's own signal (pred loss, copy CE + rows, replay passes + loss, young-lr factor) to <out>.levers.jsonl")
    ap.add_argument("--ckpt", default=None, help="checkpoint path; resume from it if present (Cam Nov 10: --ckpt on for Modal runs)")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--stop-after", type=int, default=0, help="test only: exit abruptly after this many updates (simulated preemption)")
    ap.add_argument("--hearing-probe", action="store_true", help="Cam Nov 10 ~05:15: supervised probe of the hearing pathway")
    ap.add_argument("--probe-steps", type=int, default=300); ap.add_argument("--probe-lrs", default="1,4,16")
    ap.add_argument("--probe-collect", type=int, default=32, help="teacher rollouts (x T steps) per data set")
    ap.add_argument("--probe-pool", action="store_true", help="diagnostic: every learner trains on teacher data pooled over all training worlds (full sector coverage)")
    ap.add_argument("--probe-train", default="hear", choices=["hear", "all"], help="hear: heard rows + policy head only; all: every layer (diagnostic)")
    a = ap.parse_args()
    NL = a.n_learners; NE = NL if a.parent else (a.n_elders if (a.copy > 0 or a.demonstrators) else 0); M = NL + NE
    if a.world == "proto":
        cfg = env.Cfg(L=96, N=NB + M, P=6, patch_sigma=1.0, d0=0.004 if a.long_life else 0.008, reproduce=False, A_max=3000)
    else:
        cfg = env.Cfg(L=96, N=NB + M, patch_sigma=1.0, reproduce=False, A_max=3000, **world_kw(a.world))
    W, T = a.worlds, a.T; net = NetP()
    navs = sorted(glob.glob(a.navs)); K = len(navs)
    D0 = int(env.observe(cfg, env.reset(cfg, jax.random.PRNGKey(0))).shape[1])          # 100
    templ0 = Net().init(jax.random.PRNGKey(0), jnp.zeros((1, D0)))
    templ = net.init(jax.random.PRNGKey(1), jnp.zeros((1, D0 + HD)))

    def expand(p):          # hearing rows appended as zeros (the newborn starts exactly as its navigator); pred head fresh
        q = {k: dict(v) for k, v in templ["params"].items()}
        for k in ("Dense_0", "Dense_1", "Dense_2", "Dense_3"):
            q[k] = dict(p["params"][k])
        q["Dense_0"]["kernel"] = jnp.concatenate([p["params"]["Dense_0"]["kernel"], jnp.zeros((HD, 128))], 0)
        return {"params": q}
    NAV = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[expand(flax.serialization.from_bytes(templ0, open(f, "rb").read())) for f in navs])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))

    def beacons_reset(s, key):
        c = jax.random.randint(key, (2,), 0, cfg.L)
        k_ = a.beacon_spread
        bpos = (c[None] + jnp.array([[-k_, -k_], [-k_, k_], [k_, -k_], [k_, k_]])) % cfg.L
        mpos = (c[None] + jax.random.randint(jax.random.fold_in(key, 1), (M, 2), -3, 4)) % cfg.L
        if a.parent:                                   # each parent starts beside its newborn
            mpos = mpos.at[NL:].set((mpos[:NL] + jnp.array([1, 0])) % cfg.L)
        return s._replace(pos=jnp.concatenate([bpos, mpos], 0)), c

    def hear(s, shift, perm=None):
        """(M, HD) for the non-beacon agents: nearest beacon within radius; its sector (+ shift, or mapped through perm)
        one-hot, flag, offset / R"""
        sec = jnp.mod(sector(env.oracle_offset(cfg, s))[:NB] + shift, 8)
        if perm is not None:
            sec = perm[sec]
        lp, bp = s.pos[NB:], s.pos[:NB]
        d = (bp[None] - lp[:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
        cheb = jnp.abs(d).max(-1); j = jnp.argmin(cheb, 1); dj = d[jnp.arange(M), j]
        ok = cheb[jnp.arange(M), j] <= R_HEAR
        return jnp.concatenate([jax.nn.one_hot(sec[j], 8), jnp.ones((M, 1)), dj / R_HEAR], 1) * ok[:, None]

    def scripted(s, h, fallback, code=None, dec=None):
        """a listener that eats on food, heads for visible food, else walks the heard sector (decoded through the
        life's code when there is one)"""
        pos = s.pos[NB:]; here = s.food[pos[:, 0], pos[:, 1]]
        wv = env.window(cfg, s.food, pos, cfg.r).reshape(M, -1); best = jnp.argmax(wv, 1); seen = wv.max(1) > 0.2
        o_ = jnp.arange(-cfg.r, cfg.r + 1)
        tw = lambda dx, dy: jnp.where(jnp.abs(dx) >= jnp.abs(dy), jnp.where(dx < 0, 1, 2), jnp.where(dy > 0, 3, 4))  # noqa: E731
        vis = tw(o_[best // (2 * cfg.r + 1)], o_[best % (2 * cfg.r + 1)])
        idx = jnp.argmax(h[:, :8], 1)
        if dec is not None:                            # token -> sector through a given table (the observer listener)
            idx = dec[jnp.arange(M), idx]
        elif code is not None:                         # token -> sector through the inverse of the life's code
            idx = jnp.argsort(code, axis=1)[jnp.arange(M), idx]
        ang = idx * (2 * jnp.pi / 8)
        hd = tw(jnp.round(jnp.cos(ang) * 3), jnp.round(jnp.sin(ang) * 3))
        return jnp.where(here >= 0.2, env.EAT, jnp.where(seen, vis, jnp.where(h[:, 8] > 0.5, hd, fallback)))

    def step_all(P, s, key, shift, young, perm=None, mute=False, seek_ok=True, rand_fb=False, code=None, dec=None, lg_override=None):
        k1, k2 = jax.random.split(key)
        h = hear(s, shift, perm)
        if code is not None:                           # each agent hears the sector through its own code (M, 8)
            tok = code[jnp.arange(M), jnp.argmax(h[:, :8], 1)]
            h = h.at[:, :8].set(jax.nn.one_hot(tok, 8) * h[:, 8:9])
        if mute:                                  # silence: nobody hears anything (VoI baseline)
            h = jnp.zeros_like(h)
        o = jnp.concatenate([env.observe(cfg, s)[NB:], h], 1)                              # (M, D)
        if lg_override is not None:                    # stage 1: the recurrent agent's logits / values, computed outside
            lg, v = lg_override
        else:
            lg, v, _ = jax.vmap(lambda p, x: net.apply(p, x[None]))(P, o[:NL]); lg, v = lg[:, 0], v[:, 0]
        act_l = jax.random.categorical(k1, lg)
        fb_l = jax.random.randint(jax.random.fold_in(key, 79), (NL,), 1, 5) if rand_fb else act_l
        sc = scripted(s, h, jnp.concatenate([fb_l, jax.random.randint(k2, (NE,), 1, 5)]), None if a.decode_identity else code, dec)
        if a.sanity:
            act_l = sc[:NL]
        if a.parent and seek_ok:                          # Cam Nov 12: the young stay near THEIR parent while it is present
            dpar = (s.pos[NB + NL:] - s.pos[NB:NB + NL] + cfg.L // 2) % cfg.L - cfg.L // 2           # (NL, 2)
            seekp = jnp.where(jnp.abs(dpar[:, 0]) >= jnp.abs(dpar[:, 1]), jnp.where(dpar[:, 0] < 0, 1, 2), jnp.where(dpar[:, 1] > 0, 3, 4))
            farp = jnp.abs(dpar).max(-1) > cfg.r
            act_l = jnp.where((s.age[NB:NB + NL] < (a.parent_leave or a.juvenile)) & farp, seekp, act_l)
        elif a.young_seek and a.seek_age and seek_ok:     # scaffolding independent of the juvenile lever (Cam Nov 10)
            db = (s.pos[:NB][None] - s.pos[NB:NB + NL][:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
            dn_ = db[jnp.arange(NL), jnp.argmin(jnp.abs(db).max(-1), 1)]
            seek = jnp.where(jnp.abs(dn_[:, 0]) >= jnp.abs(dn_[:, 1]), jnp.where(dn_[:, 0] < 0, 1, 2), jnp.where(dn_[:, 1] > 0, 3, 4))
            far = jnp.abs(db).max(-1).min(1) > R_HEAR
            act_l = jnp.where((s.age[NB:NB + NL] < a.seek_age) & far, seek, act_l)
        elif a.juvenile:                                  # the young stay within hearing range of the beacons
            if a.young_seek:                              # ... by stepping toward the nearest beacon (beacons may wander)
                db = (s.pos[:NB][None] - s.pos[NB:NB + NL][:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
                dn_ = db[jnp.arange(NL), jnp.argmin(jnp.abs(db).max(-1), 1)]
                seek = jnp.where(jnp.abs(dn_[:, 0]) >= jnp.abs(dn_[:, 1]), jnp.where(dn_[:, 0] < 0, 1, 2), jnp.where(dn_[:, 1] > 0, 3, 4))
                far = jnp.abs(db).max(-1).min(1) > R_HEAR                  # geometry, never the message channel (Cam)
                act_l = jnp.where(young & far, seek, act_l)
            else:
                act_l = jnp.where(young & (h[:NL, 8] < 0.5), 0, act_l)
        lp = jnp.take_along_axis(jax.nn.log_softmax(lg), act_l[:, None], 1)[:, 0]
        act_e = sc[NL:]
        if a.parent:                                       # a parent more than 2 cells from its young steps back toward it
            dch = (s.pos[NB:NB + NL] - s.pos[NB + NL:] + cfg.L // 2) % cfg.L - cfg.L // 2
            back = jnp.where(jnp.abs(dch[:, 0]) >= jnp.abs(dch[:, 1]), jnp.where(dch[:, 0] < 0, 1, 2), jnp.where(dch[:, 1] > 0, 3, 4))
            act_e = jnp.where(jnp.abs(dch).max(-1) > 2, back, act_e)
        act_m = jnp.concatenate([act_l, act_e])
        act_b = jax.random.randint(jax.random.fold_in(key, 77), (NB,), 1, 5) if a.beacon_walk else jnp.zeros(NB, jnp.int32)
        s2, r, info = env.step(cfg, s, jnp.concatenate([act_b, act_m]), k2)
        if a.parent:                                       # dropout: after the juvenile period the parent is far away (gone)
            gone = s2.age[NB:NB + NL] >= (a.parent_leave or a.juvenile)
            s2 = s2._replace(pos=s2.pos.at[NB + NL:].set(jnp.where(gone[:, None], (s2.pos[NB:NB + NL] + cfg.L // 2) % cfg.L, s2.pos[NB + NL:])))
        E = s2.energy.at[:NB].set(1.0).at[NB + NL:].set(1.0)                              # beacons + elders immortal
        if a.juvenile:
            E = E.at[NB:NB + NL].set(jnp.where(young & s2.alive[NB:NB + NL], jnp.maximum(E[NB:NB + NL], cfg.E0), E[NB:NB + NL]))
        if a.wean_to:                                      # Cam Nov 12: gradual weaning. From the end of the top-up to --wean-to the
            agw = s2.age[NB:NB + NL]                       # parent supplies w * (d0 + d_move) per step (capped at E0), w 1 -> 0 linearly
            ww = jnp.clip((a.wean_to - agw) / max(a.wean_to - a.juvenile, 1), 0.0, 1.0)
            wm = (agw >= a.juvenile) & (agw < a.wean_to) & s2.alive[NB:NB + NL]
            El = E[NB:NB + NL]; E = E.at[NB:NB + NL].set(jnp.where(wm & (El < cfg.E0), jnp.minimum(El + ww * (cfg.d0 + cfg.d_move), cfg.E0), El))
        al = s2.alive.at[:NB].set(True).at[NB + NL:].set(True)
        s2 = s2._replace(energy=E, alive=al)
        dd = (s.pos[NB + NL:][None] - s.pos[NB:NB + NL][:, None] + cfg.L // 2) % cfg.L - cfg.L // 2
        near_e = jnp.abs(dd).max(-1) <= R_HEAR                                             # (NL, NE)
        return s2, (o[:NL], act_l, lp, v, r[NB:NB + NL] * 20.0, info["dead"][NB:NB + NL], info["take"][NB:NB + NL],
                    o[NL:], act_m[NL:], near_e)

    def roll(P, s, key, code=None):
        def body(s, k):
            k1, k2 = jax.random.split(k)
            shift = jax.random.randint(k2, (), 0, 8) if a.random_code else 0
            young = s.age[NB:NB + NL] < a.juvenile
            yc = s.age[NB:NB + NL] < a.young
            s2, rec = step_all(P, s, k1, shift, young, code=code)
            return s2, rec + (s.alive[NB:NB + NL], yc, s.pos[NB:])
        return jax.lax.scan(body, s, jax.random.split(key, T))
    roll_w = jax.jit(jax.vmap(roll))

    def pred_loss(p, o, alive):
        """targets from the agent's own consecutive steps: next view (98), energy change (1), max food in view 4 ahead (1)"""
        _, _, y = net.apply(p, o)
        nxt = o[1:, :98]; de = o[1:, 98] - o[:-1, 98]
        f4 = jnp.concatenate([o[4:, :49].max(1), jnp.zeros(4)])
        m4 = jnp.concatenate([alive[:-4] & alive[4:], jnp.zeros(4, bool)]).astype(jnp.float32)
        m1 = (alive[:-1] & alive[1:]).astype(jnp.float32)
        l1 = (((y[:-1, :98] - nxt) ** 2).mean(1) * m1).sum() / (m1.sum() + 1e-8)
        l2 = (((y[:-1, 98] - de * 20.0) ** 2) * m1).sum() / (m1.sum() + 1e-8)
        l3 = (((y[:, 99] - f4) ** 2) * m4).sum() / (m4.sum() + 1e-8)
        return l1 + l2 + l3

    def loss(p, o, a_, lp0, adv, ret, m, eo, ea, em):
        lg, v, _ = net.apply(p, o); ls = jax.nn.log_softmax(lg)
        lp = jnp.take_along_axis(ls, a_[:, None], 1)[:, 0]; ratio = jnp.exp(lp - lp0)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
        L = ((pg + 0.5 * (v - ret) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)) * m).sum() / (m.sum() + 1e-8)
        if a.pred:
            L = L + a.pred * pred_loss(p, o, m > 0)
        if a.copy and NE:
            lge, _, _ = net.apply(p, eo)
            ce = -jnp.take_along_axis(jax.nn.log_softmax(lge), ea[:, None], 1)[:, 0]
            L = L + a.copy * (ce * em).sum() / (em.sum() + 1e-8)
        return L

    @jax.jit
    def upd(P, OS, f, *b):
        def one(p, os_, f_, *bb):
            g = jax.grad(loss)(p, *bb); u, os_ = opt.update(g, os_, p)
            return optax.apply_updates(p, jax.tree_util.tree_map(lambda x: x * f_, u)), os_
        return jax.vmap(jax.vmap(one))(P, OS, f, *b)

    def replay_loss(q, o, m, eo, ea, em):
        L = a.pred * pred_loss(q, o, m > 0) if a.pred else 0.0
        if a.copy_replay and a.copy and NE:
            lge, _, _ = net.apply(q, eo)
            ce = -jnp.take_along_axis(jax.nn.log_softmax(lge), ea[:, None], 1)[:, 0]
            L = L + a.copy * (ce * em).sum() / (em.sum() + 1e-8)
        return L

    @jax.jit
    def upd_replay(P, OS, f, o, m, eo, ea, em):
        def one(p, os_, f_, o, m, eo, ea, em):
            g = jax.grad(replay_loss)(p, o, m, eo, ea, em); u, os_ = opt.update(g, os_, p)
            return optax.apply_updates(p, jax.tree_util.tree_map(lambda x: x * f_, u)), os_
        return jax.vmap(jax.vmap(one))(P, OS, f, o, m, eo, ea, em)

    def eval_life(p, centres, key, shift_mode, perm=None, bpos=None, smask=None, mute=False, rand_fb=False, code=None, dec=None):
        s = env.reset(cfg, key)
        s = s._replace(centres=centres, cap=env.capacity(cfg, centres), food=env.capacity(cfg, centres))
        s, _ = beacons_reset(s, jax.random.fold_in(key, 3))
        if bpos is not None:                      # --test-start life: the life's beacons; the learner beside one of them
            j = jax.random.randint(jax.random.fold_in(key, 11), (), 0, NB)
            lp0 = (bpos[j] + jax.random.randint(jax.random.fold_in(key, 12), (2,), -3, 4)) % cfg.L
            s = s._replace(pos=s.pos.at[:NB].set(bpos).at[NB].set(lp0))
        s = s._replace(alive=jnp.array([True] * NB + [True] + [False] * (M - 1)))
        if a.test_energy:                         # Cam Nov 12: hungry start, identical for every arm (scoring unchanged)
            s = s._replace(energy=s.energy.at[NB].set(a.test_energy))
        sh = jax.random.randint(jax.random.fold_in(key, 9), (), 1, 8) * shift_mode
        P1 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NL,) + x.shape), p)
        def body(c, k):
            s, eaten, last = c
            if smask is not None:                     # the TRUE sector heard now (before any swap), for attribution
                ht = hear(s, 0)[0]; last = jnp.where(ht[8] > 0.5, jnp.argmax(ht[:8]), last)
            s2, rec = step_all(P1, s, k, sh, jnp.zeros(NL, bool), perm, mute, seek_ok=False, rand_fb=rand_fb, code=code, dec=dec)
            s2 = s2._replace(alive=s2.alive.at[NB + 1:].set(False))                      # only this learner (no elders)
            credit = 1.0 if smask is None else jnp.where(last >= 0, smask[jnp.maximum(last, 0)], 0.0)
            return (s2, eaten + rec[6][0] * credit, last), None
        (s, eaten, _), _ = jax.lax.scan(body, (s, 0.0, jnp.int32(-1)), jax.random.split(jax.random.fold_in(key, 5), a.test_window))
        return eaten
    ev = jax.jit(jax.vmap(jax.vmap(eval_life, in_axes=(None, None, 0, None)), in_axes=(0, 0, 0, None)))
    def make_evp(mute=False, rand_fb=False):   # the heard25 test: per-life, per-episode sector permutation (identity = true message); traced on first call
        life_start = a.test_start == "life"; lh = a.score == "last-heard"
        if a.code_per_life and EVAL_CODE.get("dec") is not None:   # + a per-life decode table (observer listener)
            def oned(p, c, k, q, b, m, cd, dc):
                return eval_life(p, c, k, 0, q, b if life_start else None, m if lh else None, mute, rand_fb, cd, dc)
            fd = jax.jit(jax.vmap(jax.vmap(oned, in_axes=(None, None, 0, 0, None, None, None, None)), in_axes=(0, 0, 0, 0, 0, 0, 0, 0)))
            return lambda p, c, k, q, b, m, cd=None: fd(p, c, k, q, b, m, jnp.asarray(EVAL_CODE["all"] if cd is None else cd),
                                                        jnp.asarray(EVAL_CODE["dec"] if cd is None else EVAL_CODE["dec_sub"]))
        if a.code_per_life:                            # the life's own code (M, 8) rides along per life
            def onec(p, c, k, q, b, m, cd):
                return eval_life(p, c, k, 0, q, b if life_start else None, m if lh else None, mute, rand_fb, cd)
            fc = jax.jit(jax.vmap(jax.vmap(onec, in_axes=(None, None, 0, 0, None, None, None)), in_axes=(0, 0, 0, 0, 0, 0, 0)))
            return lambda p, c, k, q, b, m, cd=None: fc(p, c, k, q, b, m, jnp.asarray(EVAL_CODE["all"] if cd is None else cd))
        def one(p, c, k, q, b, m):
            return eval_life(p, c, k, 0, q, b if life_start else None, m if lh else None, mute, rand_fb)
        f = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, None, 0, 0, None, None)), in_axes=(0, 0, 0, 0, 0, 0)))
        return lambda p, c, k, q, b, m, cd=None: f(p, c, k, q, b, m)


    def crossfoster_calibration(lives, scorable, Pz, C, keys, SM):
        """Stage-3 instrument (Cam Nov 12). Per life: foster code CF (heard all life) and a birth code CB that differs
        from CF on every well-heard sector. Probes: the same scripted beacons encode with CF (foster probe) or CB (birth
        probe), same keys, no derangement, last-heard credit. Uses-foster = pass(foster probes vs birth probes); uses-
        birth = the reverse. Listeners: perfect foster-learner (decodes inv CF), perfect birth-decoder (decodes inv CB),
        deaf navigator."""
        n = len(lives); E_ = a.eval_eps; rb = np.random.default_rng(5150)
        CF = np.stack([x["code"] for x in lives]); CB = CF.copy()
        for i, x in enumerate(lives):
            Ss = np.where(x["heard"] >= a.min_heard)[0]
            while True:
                q = rb.permutation(8)
                if np.all(q[Ss] != CF[i, Ss]):
                    break
            CB[i] = q
        def mk(c):
            A_ = np.tile(np.arange(8), (n, M, 1)); A_[:, 0] = c; return jnp.asarray(A_)
        CLF, CLB = mk(CF), mk(CB)
        DF = np.tile(np.arange(8), (n, M, 1)); DF[:, 0] = np.argsort(CF, 1); DB = np.tile(np.arange(8), (n, M, 1)); DB[:, 0] = np.argsort(CB, 1)
        ident = jnp.asarray(np.tile(np.arange(8), (n, E_, 1)))
        out = dict(lives=n, eligible=int(np.asarray(scorable).sum()))
        sv = a.sanity
        for name, dec, scripted_ in (("perfect_foster_learner", DF, True), ("perfect_birth_decoder", DB, True), ("deaf_navigator", None, False)):
            a.sanity = scripted_
            if dec is not None:
                f = jax.jit(jax.vmap(jax.vmap(lambda p, c, k, q, m, cd, dc: eval_life(p, c, k, 0, q, None, m, False, False, cd, dc),
                                              in_axes=(None, None, 0, 0, None, None, None)), in_axes=(0, 0, 0, 0, 0, 0, 0)))
                ef = np.asarray(f(Pz, C, keys, ident, SM, CLF, jnp.asarray(dec))); eb = np.asarray(f(Pz, C, keys, ident, SM, CLB, jnp.asarray(dec)))
            else:
                f = jax.jit(jax.vmap(jax.vmap(lambda p, c, k, q, m, cd: eval_life(p, c, k, 0, q, None, m, False, False, cd),
                                              in_axes=(None, None, 0, 0, None, None)), in_axes=(0, 0, 0, 0, 0, 0)))
                ef = np.asarray(f(Pz, C, keys, ident, SM, CLF)); eb = np.asarray(f(Pz, C, keys, ident, SM, CLB))
            el = np.asarray(scorable)
            fos = np.array([_life_pass(ef[i], eb[i]) for i in range(n)]); bir = np.array([_life_pass(eb[i], ef[i]) for i in range(n)])
            out[name] = dict(uses_foster=float(fos[el].mean()), uses_foster_ci95=_wilson(int(fos[el].sum()), int(el.sum())),
                             uses_birth=float(bir[el].mean()), uses_birth_ci95=_wilson(int(bir[el].sum()), int(el.sum())),
                             eaten_foster_probes=float(ef[el].mean()), eaten_birth_probes=float(eb[el].mean()))
        a.sanity = sv
        pf, pb, dn = out["perfect_foster_learner"], out["perfect_birth_decoder"], out["deaf_navigator"]
        out["calibrated"] = bool(pf["uses_foster"] >= 0.9 and pf["uses_birth"] <= 0.1 and pb["uses_birth"] >= 0.9 and pb["uses_foster"] <= 0.1
                                 and dn["uses_foster"] <= 0.1 and dn["uses_birth"] <= 0.1)
        return out

    rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed)
    states = []
    for w in range(W):
        s0 = env.reset(cfg, jax.random.PRNGKey(1000 + w)); s0, _ = beacons_reset(s0, jax.random.PRNGKey(2000 + w)); states.append(s0)
    S = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *states)
    nav_of = rng.integers(0, K, (W, NL))
    P = jax.tree_util.tree_map(lambda x: x[nav_of], NAV); OS = jax.vmap(jax.vmap(opt.init))(P)
    if a.hearing_probe:
        return hearing_probe(a, cfg, net, opt, NAV, nav_of, S, roll_w, ev, hear, W, NL, T, M)
    mem_o, mem_m, mem_e = [], [], []            # replay memory: the last a.mem rollouts, (W, NL, T, D) each (+ elder data)
    since_birth = np.zeros((W, NL), int)        # rollouts since birth (replay uses own memory only)
    gsteps = np.zeros((W, NL), int)
    cnt = np.zeros((W, NL, 8), np.int64)       # per-life heard-sector counts (coverage)
    SNAP = [int(x) for x in a.snap_ages.split(",")] if a.snap_ages else []
    crng = np.random.default_rng(a.seed + 555)        # own stream: the main rng is untouched
    CODES = np.stack([np.stack([crng.permutation(8) for _ in range(M)]) for _ in range(W)]) if a.code_per_life else None
    EVAL_CODE = {"all": None}; OBS_INFO = {}
    if a.code_per_life:
        assert a.test == "heard25", "--code-per-life needs --test heard25"
    snaps = {}; snap_gs = {}; prev_age = np.zeros((W, NL), np.int64)
    lives = []
    if a.ic_probe:
        assert a.no_train and a.code_per_life and (a.io_decoder or not a.sanity), "--ic-probe needs --no-train --code-per-life (navigator behaviour; --sanity only as the decoder's control)"
        assert not (a.obs_listener and a.sanity), "--obs-listener: lives run with navigator behaviour (the test then uses the decode table)"
        IC = dict(o=[], act=[], r=[], alive=[], codes=[], lid=[], take=[], ea=[], pos=[]); life_id = np.zeros((W, NL), np.int64)

    def stage1_recurrent():
        import time as _time
        from w2.fastlearner import code_families
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        import ckpt as _ck
        assert a.parent and a.test == "heard25", "stage 1 runs in the parent arm with the heard25 test"
        print("RESOLVED_CONFIG", json.dumps(vars(a), default=str), flush=True)
        FA, FB = code_families(); FAM = FA if a.code_family == "A" else FB
        Hd = a.rec_hidden; rnet = RecAgent(Hd, a.side_bound); Dx = 100 + HD + 6 + 1
        rp = rnet.init(jax.random.PRNGKey(a.seed), jnp.zeros((Hd,)), jnp.zeros((Dx,)))
        nav0 = jax.tree_util.tree_map(lambda x: x[0], NAV)["params"]
        pp = {k: dict(v) for k, v in rp["params"].items()}
        pp["Dense_0"] = {"kernel": nav0["Dense_0"]["kernel"][:100], "bias": nav0["Dense_0"]["bias"]}
        for k_ in ("Dense_1", "Dense_2", "Dense_3"):
            pp[k_] = dict(nav0[k_])
        rp = {"params": pp}; FROZEN = ("Dense_0", "Dense_1", "Dense_2", "Dense_3")
        ropt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.rec_lr)); ros = ropt.init(rp)
        cr = np.random.default_rng(a.seed + 555)
        LEAVE = a.parent_leave or a.juvenile
        HEADM = np.array([1 if np.round(np.cos(k * np.pi / 4) * 3) < 0 else 2 if abs(np.round(np.cos(k * np.pi / 4) * 3)) >= abs(np.round(np.sin(k * np.pi / 4) * 3))
                          else (3 if np.round(np.sin(k * np.pi / 4) * 3) > 0 else 4) for k in range(8)]) - 1     # sector -> move class 0..3

        def obs_x(s, code, pa, pr, perm=None):
            h_ = hear(s, 0, perm)
            tk = code[jnp.arange(M), jnp.argmax(h_[:, :8], 1)]
            h_ = h_.at[:, :8].set(jax.nn.one_hot(tk, 8) * h_[:, 8:9])
            o = jnp.concatenate([env.observe(cfg, s)[NB:], h_], 1)[:NL]
            return jnp.concatenate([o, jax.nn.one_hot(pa, 6), pr[:, None]], 1)

        def roll_r(rp, s, Hh, pa, pr, key, code, probe):
            def body(c, k):
                s, Hh, pa, pr = c
                x = obs_x(s, code, pa, pr)
                h2, lg, v, _ = jax.vmap(lambda h, xx: rnet.apply(rp, h, xx))(Hh, x)
                young = s.age[NB:NB + NL] < a.juvenile
                s2, rec = step_all(None, s, k, 0, young, code=code, lg_override=(lg, v))
                al = s.alive[NB:NB + NL]
                out = (x, rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], al, rec[8], s.pos[NB:])
                if probe:                                  # memory state after reading this step, and the memory-wiped state
                    hz, _, _, _ = jax.vmap(lambda xx: rnet.apply(rp, jnp.zeros((Hd,)), xx))(x)
                    true_sec = jnp.argmax(hear(s, 0)[:NL, :8], 1)
                    out = out + (h2, hz, true_sec)
                dpf = (s.pos[NB + NL:] - s.pos[NB:NB + NL] + cfg.L // 2) % cfg.L - cfg.L // 2   # the parent-seek reflex overrode this step's action
                out = out + ((s.age[NB:NB + NL] < LEAVE) & (jnp.abs(dpf).max(-1) > cfg.r),)
                return (s2, jnp.where(al[:, None], h2, Hh), rec[1], rec[4]), out
            (s, Hh, pa, pr), R = jax.lax.scan(body, (s, Hh, pa, pr), jax.random.split(key, T))
            return s, Hh, pa, pr, R
        roll_rw = jax.jit(jax.vmap(roll_r, in_axes=(None, 0, 0, 0, 0, 0, 0, None)), static_argnums=(7,))

        def ce(lg, y):
            return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[..., None], -1)[..., 0]

        def seq_loss(rp, x, h0, act, lp0, adv, ret, m, ym, mm, yo, mo, yv, mv):
            def body(h, xx):
                h2, lg, v, aux = rnet.apply(rp, h, xx); return h2, (lg, v, aux)
            _, (lg, v, (am, ao, av)) = jax.lax.scan(body, h0, x)
            ls = jax.nn.log_softmax(lg); lp = jnp.take_along_axis(ls, act[:, None], 1)[:, 0]; ratio = jnp.exp(lp - lp0)
            pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)
            Lp = ((pg + 0.5 * (v - ret) ** 2 + 0.01 * (jnp.exp(ls) * ls).sum(-1)) * m).sum() / (m.sum() + 1e-8)
            if a.kl_nav:                                   # KL to the frozen navigator's policy (the trunk alone), on every live step
                P_ = rp["params"]; tn = jnp.tanh(x[:, :100] @ P_["Dense_0"]["kernel"] + P_["Dense_0"]["bias"])
                tn = jnp.tanh(tn @ P_["Dense_1"]["kernel"] + P_["Dense_1"]["bias"]); lsn = jax.nn.log_softmax(tn @ P_["Dense_2"]["kernel"] + P_["Dense_2"]["bias"])
                mk = m * (x[:, 108] < 0.5) if a.kl_silent_only else m
                Lp = Lp + a.kl_nav * ((jnp.exp(ls) * (ls - lsn)).sum(-1) * mk).sum() / (mk.sum() + 1e-8)
            Lm = (ce(am, ym) * mm).sum() / (mm.sum() + 1e-8); Lo = (ce(ao, yo) * mo).sum() / (mo.sum() + 1e-8)
            Lv = (((av - yv) ** 2).mean(-1) * mv).sum() / (mv.sum() + 1e-8)
            return Lp + a.aux * (Lm + Lo + Lv), jnp.stack([Lp, Lm, Lo, Lv, mm.sum(), mo.sum()])

        @jax.jit
        def upd(rp, ros, *b):
            def tot(q):
                L, parts = jax.vmap(seq_loss, in_axes=(None,) + (0,) * 13)(q, *b); return L.mean(), parts
            (L, parts), g = jax.value_and_grad(tot, has_aux=True)(rp)
            g = {"params": {k_: (jax.tree_util.tree_map(jnp.zeros_like, v_) if k_ in FROZEN else v_) for k_, v_ in g["params"].items()}}
            u, ros = ropt.update(g, ros, rp); return optax.apply_updates(rp, u), ros, parts

        def sec_np(dx, dy):
            return np.mod(np.round(np.arctan2(dy, dx) / (np.pi / 4)), 8).astype(np.int64)

        def respawn(S_, Hh, PA, PR, dead, rng_):
            dm = jnp.asarray(dead)
            Hh = jnp.where(dm[..., None], 0.0, Hh); PA = jnp.where(dm, 0, PA); PR = jnp.where(dm, 0.0, PR)
            npos = (np.asarray(S_.pos)[:, :NB].mean(1, keepdims=True).astype(int) + rng_.integers(-3, 4, (S_.pos.shape[0], NL, 2))) % cfg.L
            pos_ = np.asarray(S_.pos).copy(); E = np.asarray(S_.energy).copy(); age = np.asarray(S_.age).copy(); alv = np.asarray(S_.alive).copy()
            sl = slice(NB, NB + NL)
            pos_[:, sl][dead] = npos[dead]; E[:, sl][dead] = cfg.E0; age[:, sl][dead] = 0; alv[:, sl][dead] = True
            pos_[:, NB + NL:][dead] = (npos[dead] + np.array([1, 0])) % cfg.L
            return S_._replace(pos=jnp.asarray(pos_), energy=jnp.asarray(E), age=jnp.asarray(age), alive=jnp.asarray(alv)), Hh, PA, PR

        def fresh_worlds(base, Wn):
            st = []
            for w in range(Wn):
                s0 = env.reset(cfg, jax.random.PRNGKey(base + w)); s0, _ = beacons_reset(s0, jax.random.PRNGKey(base + 500 + w)); st.append(s0)
            return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *st)

        # ---- the life test: the frozen heard25 probes / derangements / keys / credit, memory carried in; hungry start
        def eval_rec(rp, centres, key, perm, smask, code_l, h0, e0):
            s = env.reset(cfg, key)
            s = s._replace(centres=centres, cap=env.capacity(cfg, centres), food=env.capacity(cfg, centres))
            s, _ = beacons_reset(s, jax.random.fold_in(key, 3))
            s = s._replace(alive=jnp.array([True] * NB + [True] + [False] * (M - 1)))
            s = s._replace(energy=jnp.where(e0 > 0, s.energy.at[NB].set(e0), s.energy))
            code = jnp.tile(jnp.arange(8), (M, 1)).at[0].set(code_l)
            def body(c, k):
                s, h, pa, pr, eaten, last = c
                ht = hear(s, 0)[0]; last = jnp.where(ht[8] > 0.5, jnp.argmax(ht[:8]), last)
                x = obs_x(s, code, jnp.zeros(NL, jnp.int32).at[0].set(pa), jnp.zeros(NL).at[0].set(pr), perm)
                h2, lg, v, _ = rnet.apply(rp, h, x[0])
                LG = jnp.zeros((NL, 6)).at[0].set(lg); V = jnp.zeros(NL).at[0].set(v)
                s2, rec = step_all(None, s, k, 0, jnp.zeros(NL, bool), perm, False, seek_ok=False, code=code, lg_override=(LG, V))
                s2 = s2._replace(alive=s2.alive.at[NB + 1:].set(False))
                credit = jnp.where(last >= 0, smask[jnp.maximum(last, 0)], 0.0)
                return (s2, h2, rec[1][0], rec[4][0], eaten + rec[6][0] * credit, last), None
            (_, _, _, _, eaten, _), _ = jax.lax.scan(body, (s, h0, jnp.int32(0), 0.0, 0.0, jnp.int32(-1)),
                                                     jax.random.split(jax.random.fold_in(key, 5), a.test_window))
            return eaten
        evr = jax.jit(jax.vmap(jax.vmap(eval_rec, in_axes=(None, None, 0, 0, None, None, None, None)), in_axes=(None, 0, 0, 0, 0, 0, 0, None)))

        def linear_probe(F, Y, LIV, K_, ncls, seed):
            """softmax regression on frozen features; train on 70% of lives, test on the rest; accuracy by tokens heard"""
            r_ = np.random.default_rng(seed); ul = np.unique(LIV); r_.shuffle(ul); te = np.isin(LIV, ul[: max(1, len(ul) * 3 // 10)])
            Xtr, Ytr, Xte, Yte, Kte = jnp.asarray(F[~te]), jnp.asarray(Y[~te]), jnp.asarray(F[te]), Y[te], K_[te]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Wt = jnp.zeros((F.shape[1], ncls)); bt = jnp.zeros(ncls); po = optax.adam(1e-2); pst = po.init((Wt, bt))
            @jax.jit
            def st_(pr_, pst):
                def lf(pr_):
                    Wq, bq = pr_; lg = ((Xtr - mu) / sd) @ Wq + bq; return ce(lg, Ytr).mean() + 1e-4 * (Wq ** 2).sum()
                g = jax.grad(lf)(pr_); u, pst = po.update(g, pst, pr_); return optax.apply_updates(pr_, u), pst
            prm = (Wt, bt)
            for _ in range(300):
                prm, pst = st_(prm, pst)
            pred = np.asarray((((Xte - mu) / sd) @ prm[0] + prm[1]).argmax(-1)); hit = pred == Yte
            bins = [(1, 5), (6, 15), (16, 40), (41, 100), (101, 300), (301, 10 ** 6)]
            return dict(acc_all=float(hit.mean()), n_test=int(len(hit)),
                        by_tokens_heard={f"{lo}-{hi}": (round(float(hit[(Kte >= lo) & (Kte <= hi)].mean()), 4) if ((Kte >= lo) & (Kte <= hi)).any() else None) for lo, hi in bins})

        def readout(rp, tag):
            """frozen weights on a fresh world batch: lives born here are tested (hungry start primary, natural secondary;
            memory carried vs wiped), and a linear probe reads the memory (vs wiped and shuffled-code controls)"""
            Wr = int(a.readout_worlds_at.split(",")[len(READS)]) if a.readout_worlds_at else a.readout_worlds   # Cam Nov 12: halved early readouts
            Sr = fresh_worlds(91_000 + 1000 * len(READS), Wr); rr = np.random.default_rng(a.seed + 101)
            Hr = jnp.zeros((Wr, NL, Hd)); PAr = jnp.zeros((Wr, NL), jnp.int32); PRr = jnp.zeros((Wr, NL))
            CODr = np.tile(np.arange(8), (Wr, M, 1)); CODr[:, :NL] = FAM[rr.integers(0, len(FAM), (Wr, NL))]
            cnt_r = np.zeros((Wr, NL, 8), np.int64); kh = np.zeros((Wr, NL), np.int64); lidr = np.zeros((Wr, NL), np.int64)
            lives_x, PF, key_r = [], dict(h=[], hz=[], sec=[], k=[], life=[], code=[]), jax.random.PRNGKey(a.seed + 202)
            for it in range(a.eval_updates):
                key_r, k = jax.random.split(key_r)
                Sr, Hr, PAr, PRr, R = roll_rw(rp, Sr, Hr, PAr, PRr, jax.random.split(k, Wr), jnp.asarray(CODr), True)
                x, al = np.asarray(R[0]), np.asarray(R[7]); h2, hz, tsec = np.asarray(R[10]), np.asarray(R[11]), np.asarray(R[12])
                hm = (x[..., 108] > 0.5) & al
                for t in range(T):                         # probe rows: heard steps, with the life's running count of tokens heard
                    kh += hm[:, t]
                    if it % 3 == 0 and hm[:, t].any():      # subsample every 3rd update to bound memory
                        ww, ll = np.where(hm[:, t])
                        PF["h"].append(h2[ww, t, ll]); PF["hz"].append(hz[ww, t, ll]); PF["sec"].append(tsec[ww, t, ll])
                        PF["k"].append(kh[ww, ll]); PF["life"].append(ww * 10_000 + ll * 1000 + lidr[ww, ll])
                cnt_r += np.einsum("wtl,wtlk->wlk", hm.astype(np.int64), (x[..., 100:108] > 0.5).astype(np.int64))
                dead = ~np.asarray(Sr.alive)[:, NB:NB + NL]; ages = np.asarray(Sr.age)[:, NB:NB + NL]
                for w, l in zip(*np.where(dead)):
                    lives_x.append(dict(lifespan=int(ages[w, l]), heard=cnt_r[w, l][CODr[w, l]], code=CODr[w, l].copy(), h=np.asarray(Hr[w, l]),
                                        centres=np.asarray(Sr.centres[w]), tokens=int(kh[w, l])))
                    CODr[w, l] = FAM[rr.integers(0, len(FAM))]; cnt_r[w, l] = 0; kh[w, l] = 0; lidr[w, l] += 1
                if dead.any():
                    Sr, Hr, PAr, PRr = respawn(Sr, Hr, PAr, PRr, dead, rr)
            out = dict(tag=tag, lives=len(lives_x))
            if lives_x:
                n = len(lives_x); E_ = a.eval_eps; rq = np.random.default_rng(4242)
                ident = np.tile(np.arange(8), (n, E_, 1)); der = ident.copy(); el = np.zeros(n, bool)
                for i, x_ in enumerate(lives_x):
                    Ss = np.where(x_["heard"] >= a.min_heard)[0]; el[i] = len(Ss) >= max(2, a.min_sectors)
                    if len(Ss) < 2:
                        continue
                    for e in range(E_):
                        while True:
                            q = rq.permutation(Ss)
                            if np.all(q != Ss):
                                break
                        der[i, e, Ss] = q
                keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(7000 + i), E_))(jnp.arange(n))
                Cn = jnp.asarray(np.stack([x_["centres"] for x_ in lives_x])); SM = jnp.asarray(np.stack([(x_["heard"] >= a.min_heard).astype(np.float32) for x_ in lives_x]))
                CL = jnp.asarray(np.stack([x_["code"] for x_ in lives_x])); HL = jnp.asarray(np.stack([x_["h"] for x_ in lives_x]))
                gm = np.array([int((x_["heard"] >= a.min_heard).sum()) for x_ in lives_x]); tok = np.array([x_["tokens"] for x_ in lives_x])
                tests = {}
                for start, e0 in (("hungry", a.test_energy or 0.3), ("natural", 0.0)):
                    for name, h0s in (("memory_carried", HL), ("memory_wiped", jnp.zeros_like(HL))):
                        tru = np.asarray(evr(rp, Cn, keys, jnp.asarray(ident), SM, CL, h0s, e0)); swp = np.asarray(evr(rp, Cn, keys, jnp.asarray(der), SM, CL, h0s, e0))
                        ok = np.array([_life_pass(tru[i], swp[i]) for i in range(n)])
                        tests[f"{start}_{name}"] = dict(
                            eligible=int(el.sum()), pass_frac=float(ok[el].mean()) if el.any() else None, ci95=_wilson(int(ok[el].sum()), int(el.sum())),
                            stratified={lab: (int((el & mk).sum()), float(ok[el & mk].mean()) if (el & mk).any() else None) for lab, mk in (("5", gm == 5), ("6", gm == 6), ("7-8", gm >= 7))},
                            by_tokens_heard_in_life={f"{lo}-{hi}": (int((el & (tok >= lo) & (tok <= hi)).sum()), float(ok[el & (tok >= lo) & (tok <= hi)].mean()) if (el & (tok >= lo) & (tok <= hi)).any() else None)
                                                     for lo, hi in ((0, 300), (301, 600), (601, 1200), (1201, 10 ** 6))},
                            eaten_true=float(tru[el].mean()) if el.any() else None, eaten_swap=float(swp[el].mean()) if el.any() else None)
                out.update(eligible=int(el.sum()), eligibility_rate=float(el.mean()), tests=tests)
            if PF["h"]:
                Fh, Fz = np.concatenate(PF["h"]), np.concatenate(PF["hz"]); Ysec = np.concatenate(PF["sec"]); Kk = np.concatenate(PF["k"]); LV = np.concatenate(PF["life"])
                Yshuf = np.empty_like(Ysec); rs = np.random.default_rng(7)
                for lv in np.unique(LV):                   # shuffled-code control: each life's labels through a random permutation
                    mk = LV == lv; Yshuf[mk] = rs.permutation(8)[Ysec[mk]]
                out["linear_probe"] = {"sector_memory": linear_probe(Fh, Ysec, LV, Kk, 8, 1), "sector_wiped": linear_probe(Fz, Ysec, LV, Kk, 8, 1),
                                       "sector_shuffled_code": linear_probe(Fh, Yshuf, LV, Kk, 8, 1),
                                       "move_memory": linear_probe(Fh, HEADM[Ysec], LV, Kk, 4, 1), "move_wiped": linear_probe(Fz, HEADM[Ysec], LV, Kk, 4, 1),
                                       "rows": int(len(Ysec)), "chance_sector": 0.125}
            return out

        def km(L, C):
            """Kaplan-Meier median of lifespans L with right-censoring flags C"""
            L, C = np.asarray(L), np.asarray(C, bool); Sv = 1.0
            for t_ in np.unique(L[~C]):
                n_ = (L >= t_).sum(); d_ = ((L == t_) & ~C).sum(); Sv *= 1 - d_ / n_
                if Sv <= 0.5:
                    return int(t_)
            return None

        def km_surv(L, C, t0):
            """Kaplan-Meier survival past age t0"""
            L, C = np.asarray(L), np.asarray(C, bool); Sv = 1.0
            for t_ in np.unique(L[~C]):
                if t_ > t0:
                    break
                Sv *= 1 - ((L == t_) & ~C).sum() / (L >= t_).sum()
            return float(Sv)

        def forage(rp_, Wf, nup, base, rew=None):
            """frozen weights, natural energy, fresh worlds (same seeds for every arm): every life born in the window;
            lifespan (KM, censored at the end), cause of death, food per step by age"""
            Sf = fresh_worlds(base, Wf); rf = np.random.default_rng(a.seed + 303); kf = jax.random.PRNGKey(base + 9)
            Hf = jnp.zeros((Wf, NL, Hd)); PAf = jnp.zeros((Wf, NL), jnp.int32); PRf = jnp.zeros((Wf, NL))
            CODf = np.tile(np.arange(8), (Wf, M, 1)); CODf[:, :NL] = FAM[rf.integers(0, len(FAM), (Wf, NL))]
            born = np.asarray(Sf.age)[:, NB:NB + NL] == 0
            AB = [0, 250, 500, 1000, 1500, 2000, 3000, 10 ** 6]; FS = np.zeros((Wf, len(AB) - 1)); NS = np.zeros((Wf, len(AB) - 1))
            lives = []                                     # (world, lifespan, censored, cause)
            for _ in range(nup):
                kf, k = jax.random.split(kf); age0 = np.asarray(Sf.age)[:, NB:NB + NL]
                Sf, Hf, PAf, PRf, R = roll_rw(rp_, Sf, Hf, PAf, PRf, jax.random.split(k, Wf), jnp.asarray(CODf), False)
                take, al = np.asarray(R[6]), np.asarray(R[7])                       # (Wf, T, NL)
                if rew is not None:                        # rew[w][l] = (birth step, [rewards of the current life])
                    rr_ = np.asarray(R[4])
                    for w in range(Wf):
                        for l in range(NL):
                            rew[w][l][1].extend(rr_[w, al[w, :, l], l].tolist()); rew[w][l][2].extend(np.asarray(R[0])[w, al[w, :, l], l, 98].tolist())
                ageT = age0[:, None, :] + np.arange(T)[None, :, None]
                bi = np.searchsorted(AB, ageT, side="right") - 1
                for j in range(len(AB) - 1):
                    mk = al & (bi == j) & born[:, None, :]
                    FS[:, j] += (take * mk).sum((1, 2)); NS[:, j] += mk.sum((1, 2))
                dead = ~np.asarray(Sf.alive)[:, NB:NB + NL]; agd = np.asarray(Sf.age)[:, NB:NB + NL]; Ed = np.asarray(Sf.energy)[:, NB:NB + NL]
                for w, l in zip(*np.where(dead)):
                    if born[w, l]:
                        lives.append((w, int(agd[w, l]), False, "starvation" if Ed[w, l] <= 0 else ("age" if agd[w, l] >= cfg.A_max else "other")))
                        if rew is not None:
                            rew["done"].append((w, rew[w][l][0], np.array(rew[w][l][1]), lives[-1][3], np.array(rew[w][l][2])))
                    if rew is not None:
                        rew[w][l] = ((_ + 1) * T, [], [])
                    CODf[w, l] = FAM[rf.integers(0, len(FAM))]; born[w, l] = True
                if dead.any():
                    Sf, Hf, PAf, PRf = respawn(Sf, Hf, PAf, PRf, dead, rf)
            agd = np.asarray(Sf.age)[:, NB:NB + NL]
            for w, l in zip(*np.where(born)):
                lives.append((w, int(agd[w, l]), True, "censored"))
            return lives, FS, NS, AB

        def redraw_probe(rp, AGE, Wr, nup):
            Sr = fresh_worlds(93_000, Wr); rr = np.random.default_rng(a.seed + 404); kr = jax.random.PRNGKey(a.seed + 505)
            Hr = jnp.zeros((Wr, NL, Hd)); PAr = jnp.zeros((Wr, NL), jnp.int32); PRr = jnp.zeros((Wr, NL))
            CODr = np.tile(np.arange(8), (Wr, M, 1)); CODr[:, :NL] = FAM[rr.integers(0, len(FAM), (Wr, NL))]
            kb = np.zeros((Wr, NL), np.int64); ka = np.zeros((Wr, NL), np.int64); post = np.zeros((Wr, NL), bool); lid = np.zeros((Wr, NL), np.int64)
            born = np.asarray(Sr.age)[:, NB:NB + NL] == 0
            F = dict(h=[], hz=[], sec=[], kb=[], ka=[], post=[], life=[])
            for _ in range(nup):
                ages = np.asarray(Sr.age)[:, NB:NB + NL]
                for w, l in zip(*np.where(born & ~post & (ages >= AGE))):          # re-draw: a new code, different on every sector
                    while True:
                        c2 = FAM[rr.integers(0, len(FAM))]
                        if np.all(c2 != CODr[w, l]):
                            break
                    CODr[w, l] = c2; post[w, l] = True; ka[w, l] = 0
                kr, k = jax.random.split(kr)
                Sr, Hr, PAr, PRr, R = roll_rw(rp, Sr, Hr, PAr, PRr, jax.random.split(k, Wr), jnp.asarray(CODr), True)
                x, al = np.asarray(R[0]), np.asarray(R[7]); h2, hz, tsec = np.asarray(R[10]), np.asarray(R[11]), np.asarray(R[12])
                hm = (x[..., 108] > 0.5) & al & born[:, None, :]
                for t in range(T):
                    kb += hm[:, t]; ka += hm[:, t] & post
                    if hm[:, t].any():
                        ww, ll = np.where(hm[:, t])
                        F["h"].append(h2[ww, t, ll]); F["hz"].append(hz[ww, t, ll]); F["sec"].append(tsec[ww, t, ll]); F["kb"].append(kb[ww, ll])
                        F["ka"].append(ka[ww, ll]); F["post"].append(post[ww, ll]); F["life"].append(ww * 10_000 + ll * 1000 + lid[ww, ll])
                dead = ~np.asarray(Sr.alive)[:, NB:NB + NL]
                for w, l in zip(*np.where(dead)):
                    CODr[w, l] = FAM[rr.integers(0, len(FAM))]; kb[w, l] = 0; ka[w, l] = 0; post[w, l] = False; lid[w, l] += 1; born[w, l] = True
                if dead.any():
                    Sr, Hr, PAr, PRr = respawn(Sr, Hr, PAr, PRr, dead, rr)
            Fh, Fz = np.concatenate(F["h"]), np.concatenate(F["hz"]); Y = np.concatenate(F["sec"]); KB = np.concatenate(F["kb"]); KA = np.concatenate(F["ka"])
            PO = np.concatenate(F["post"]); LV = np.concatenate(F["life"])
            r_ = np.random.default_rng(1); ul = np.unique(LV); r_.shuffle(ul); te = np.isin(LV, ul[: max(1, len(ul) * 3 // 10)])
            bins = [(1, 5), (6, 15), (16, 40), (41, 100), (101, 300), (301, 10 ** 6)]
            def fit_score(Fx):
                tr_ = ~te & ~PO; Xtr, Ytr = jnp.asarray(Fx[tr_]), jnp.asarray(Y[tr_]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                po = optax.adam(1e-2); prm = (jnp.zeros((Fx.shape[1], 8)), jnp.zeros(8)); pst = po.init(prm)
                @jax.jit
                def st_(prm, pst):
                    g = jax.grad(lambda q: ce(((Xtr - mu) / sd) @ q[0] + q[1], Ytr).mean() + 1e-4 * (q[0] ** 2).sum())(prm); u, pst = po.update(g, pst, prm)
                    return optax.apply_updates(prm, u), pst
                for _ in range(300):
                    prm, pst = st_(prm, pst)
                hit = np.asarray((((jnp.asarray(Fx[te]) - mu) / sd) @ prm[0] + prm[1]).argmax(-1)) == Y[te]
                po_, kb_, ka_ = PO[te], KB[te], KA[te]
                def bn(mk, kk):
                    return {f"{lo}-{hi}": ((round(float(hit[mk & (kk >= lo) & (kk <= hi)].mean()), 4), int((mk & (kk >= lo) & (kk <= hi)).sum())) if (mk & (kk >= lo) & (kk <= hi)).any() else None) for lo, hi in bins}
                return dict(pre_by_tokens_since_birth=bn(~po_, kb_), post_by_tokens_since_redraw=bn(po_, ka_), post_by_tokens_since_birth=bn(po_, kb_),
                            pre_all=float(hit[~po_].mean()) if (~po_).any() else None, post_all=float(hit[po_].mean()) if po_.any() else None)
            out = dict(redraw_age=AGE, worlds=Wr, steps=nup * T, rows=int(len(Y)), rows_post=int(PO.sum()), lives=int(len(ul)), at_update=it, at_agent_steps=steps_train,
                       memory=fit_score(Fh), wiped=fit_score(Fz), chance=0.125)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1, default=float)
            print("REDRAW_PROBE", json.dumps(out, default=float), flush=True)

        def return_check(rp, write=True):
            """Cam Nov 12: is freezing what the training objective asks for? The actual training reward (x20 drive
            reduction, move cost included) along each complete life born in the window, discounted at several gammas"""
            def biased(i):
                return {"params": {**rpn["params"], "side_head": {"kernel": jnp.zeros_like(rp["params"]["side_head"]["kernel"]),
                                                                  "bias": jnp.zeros(6).at[i].set(1e3)}}}
            rpn = {"params": {k_: (jax.tree_util.tree_map(jnp.zeros_like, v_) if k_ in ("side_head", "side_value") else v_) for k_, v_ in rp["params"].items()}}
            arms = dict(navigator=rpn, agent_ckpt=rp, stand_still=biased(0), stand_and_eat=biased(5))
            Wf, nup = a.forage_worlds, a.forage_updates; last_birth = nup * T - cfg.A_max     # every life born by then is complete
            G = [0.99, 0.995, 0.998, 0.999, 0.9995, 1.0]; out = dict(worlds=Wf, steps=nup * T, train_gamma=0.99, train_lambda=0.95, reward_scale=20.0,
                                                                   reward="drive(E_prev) - drive(E), drive = |E_set - E|^drive_n; energy update includes d_move on moves",
                                                                   cfg=dict(E_set=cfg.E_set, drive_n=cfg.drive_n, d0=cfg.d0, d_move=cfg.d_move, E0=cfg.E0, eff=cfg.eff, inclusive=cfg.inclusive, A_max=cfg.A_max),
                                                                   juvenile_topup_age=a.juvenile, at_update=it)
            for name, q in arms.items():
                rew = {w: {l: (0, [], []) for l in range(NL)} for w in range(Wf)}; rew["done"] = []
                lives, FS, NS, AB = forage(q, Wf, nup, 77_000, rew)
                done = [d_ for d_ in rew["done"] if d_[1] <= last_birth]
                res = dict(lives_complete=len(done), mean_lifespan=float(np.mean([len(d_[2]) for d_ in done])) if done else None,
                           old_age_deaths=sum(1 for d_ in done if d_[3] == "age"),
                           undiscounted_life_return_mean=float(np.mean([d_[2].sum() for d_ in done])) if done else None)
                for g in G:
                    g0 = [float((d_[2] * g ** np.arange(len(d_[2]))).sum()) for d_ in done]
                    gt = []                                 # return-to-go from every live step of complete lives
                    for d_ in done:
                        rv = d_[2]; acc = np.zeros(len(rv)); c = 0.0
                        for t_ in range(len(rv) - 1, -1, -1):
                            c = rv[t_] + g * c; acc[t_] = c
                        gt.append(acc)
                    ga = np.concatenate([np.arange(len(d_[2])) for d_ in done]) if done else np.zeros(1)
                    gt = np.concatenate(gt) if gt else np.zeros(1)
                    byage = {f"{lo}-{hi - 1}": (round(float(gt[(ga >= lo) & (ga < hi)].mean()), 4) if ((ga >= lo) & (ga < hi)).any() else None)
                             for lo, hi in ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000))}
                    bw = {}                                 # bootstrap CI over worlds for the from-birth return
                    rb = np.random.default_rng(5); ws_ = np.array([d_[0] for d_ in done]); g0a = np.array(g0)
                    bs = [g0a[np.concatenate([np.where(ws_ == w_)[0] for w_ in rb.integers(0, Wf, Wf)])].mean() for _ in range(300)]
                    res[f"gamma_{g}"] = dict(return_from_birth=float(g0a.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                             return_to_go_per_step=float(gt.mean()), return_to_go_by_age=byage)
                res["adult_food_per_step"] = float(FS[:, [j for j in range(len(AB) - 1) if AB[j] >= LEAVE]].sum() / max(NS[:, [j for j in range(len(AB) - 1) if AB[j] >= LEAVE]].sum(), 1))
                if a.reward_variants:                       # Cam Nov 12: alternative rewards scored on the SAME trajectories
                    def drv(E):
                        return np.abs(cfg.E_set - np.clip(E, 0, 1)) ** cfg.drive_n
                    VAR = {}
                    for spec in ["current"] + a.reward_variants.split(","):
                        kind, val = (spec.split(":") + ["0"])[:2]; val = float(val); seqs = []
                        for d_ in done:
                            rv = d_[2].copy()
                            if kind == "death" and d_[3] == "starvation":
                                rv[-1] -= 20.0 * val                # terminal penalty (drive units, x20 like the reward)
                            elif kind == "alive":
                                rv = rv + 20.0 * val                # per-step alive bonus
                            elif kind == "topup":                   # drive reduction on the energy AFTER the juvenile top-up
                                E_ = d_[4]; rv = rv.copy(); rv[:-1] = 20.0 * (drv(E_[:-1]) - drv(E_[1:]))
                            seqs.append(rv)
                        g = 0.99; ages_ = np.concatenate([np.arange(len(q)) for q in seqs]); gt_ = []
                        for q in seqs:
                            acc = np.zeros(len(q)); c = 0.0
                            for t_ in range(len(q) - 1, -1, -1):
                                c = q[t_] + g * c; acc[t_] = c
                            gt_.append(acc)
                        gt_ = np.concatenate(gt_)
                        VAR[spec] = dict(return_from_birth=float(np.mean([(q * g ** np.arange(len(q))).sum() for q in seqs])),
                                         by_age={f"{lo}-{hi - 1}": (round(float(gt_[(ages_ >= lo) & (ages_ < hi)].mean()), 4) if ((ages_ >= lo) & (ages_ < hi)).any() else None)
                                                 for lo, hi in ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000))})
                    res["reward_variants_gamma_0.99"] = VAR
                out[name] = res; print("return", name, json.dumps(res)[:400], flush=True)
            if write:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1, default=float)
            print("RETURN_CHECK", json.dumps(out, default=float)[:3000], flush=True)
            return out

        def forage_check(rp, write=True):
            rpn = {"params": {k_: (jax.tree_util.tree_map(jnp.zeros_like, v_) if k_ in ("side_head", "side_value") else v_) for k_, v_ in rp["params"].items()}}
            Wf, nup, res = a.forage_worlds, a.forage_updates, {}
            for name, q in (("agent", rp), ("navigator", rpn)):
                t1 = _time.time(); res[name] = forage(q, Wf, nup, 77_000); print(f"forage {name}: {_time.time() - t1:.0f}s", flush=True)
            AB = res["agent"][3]; adult = [j for j in range(len(AB) - 1) if AB[j] >= LEAVE]
            def summ(lives, FS, NS, wsel):
                Lw = [x for w_ in wsel for x in lives if x[0] == w_]; L_ = [x[1] for x in Lw]; C_ = [x[2] for x in Lw]   # resampled worlds count with multiplicity
                fs, ns = FS[wsel].sum(0), NS[wsel].sum(0)
                return dict(km_median=km(L_, C_), adult_food_per_step=float(fs[adult].sum() / max(ns[adult].sum(), 1)), fs=fs, ns=ns, Lw=Lw)
            allw = np.arange(Wf); out = dict(worlds=Wf, steps=nup * T, adult_from_age=LEAVE, A_max=cfg.A_max, at_update=it, at_agent_steps=steps_train)
            for name in res:
                sm = summ(*res[name][:3], allw); Lw = sm["Lw"]
                deaths = [x for x in Lw if not x[2]]
                out[name] = dict(lives=len(Lw), deaths=len(deaths), censored=len(Lw) - len(deaths), km_median_lifespan=sm["km_median"],
                                 survival_past_1000=km_surv([x[1] for x in Lw], [x[2] for x in Lw], 1000),
                                 survival_past_1300=km_surv([x[1] for x in Lw], [x[2] for x in Lw], 1300),
                                 adult_food_per_step=sm["adult_food_per_step"],
                                 food_per_step_by_age={f"{AB[j]}-{AB[j + 1] - 1}": (round(float(sm["fs"][j] / sm["ns"][j]), 6) if sm["ns"][j] else None, int(sm["ns"][j])) for j in range(len(AB) - 1)},
                                 death_age_hist_100={int(b_): int(c_) for b_, c_ in zip(*np.unique([x[1] // 100 * 100 for x in deaths], return_counts=True))},
                                 deaths_by_age_and_cause={f"{AB[j]}-{AB[j + 1] - 1}": {c: sum(1 for x in deaths if AB[j] <= x[1] < AB[j + 1] and x[3] == c) for c in ("starvation", "age", "other")} for j in range(len(AB) - 1)})
            rb = np.random.default_rng(11); rk, rf_ = [], []           # bootstrap over worlds (same worlds in both arms)
            for _ in range(500):
                ws = rb.integers(0, Wf, Wf); sa, sn = summ(*res["agent"][:3], ws), summ(*res["navigator"][:3], ws)
                if sa["km_median"] and sn["km_median"]:
                    rk.append(sa["km_median"] / sn["km_median"])
                if sn["adult_food_per_step"] > 0:
                    rf_.append(sa["adult_food_per_step"] / sn["adult_food_per_step"])
            ka, kn = out["agent"]["km_median_lifespan"], out["navigator"]["km_median_lifespan"]
            fa, fn_ = out["agent"]["adult_food_per_step"], out["navigator"]["adult_food_per_step"]
            out["ratio_km_median"] = dict(point=(ka / kn if ka and kn else None), ci95=[float(np.percentile(rk, 2.5)), float(np.percentile(rk, 97.5))] if rk else None)
            out["ratio_adult_food_per_step"] = dict(point=(fa / fn_ if fn_ else None), ci95=[float(np.percentile(rf_, 2.5)), float(np.percentile(rf_, 97.5))] if rf_ else None)
            pts = [r_["point"] for r_ in (out["ratio_km_median"], out["ratio_adult_food_per_step"]) if r_["point"] is not None]
            worst = min(pts) if pts else None
            out["rule_registered"] = "lower of the two ratios: <0.5 STOP and diagnose; 0.5-0.9 continue, recheck at 300M; >=0.9 carry on"
            out["decision"] = None if worst is None else ("STOP" if worst < 0.5 else ("CONTINUE_RECHECK_300M" if worst < 0.9 else "CARRY_ON"))
            if write:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1, default=float)
            print("FORAGE_CHECK", json.dumps(out, default=float), flush=True)
            return out

        @jax.jit
        def _tinf(rp, x, h0, m, key):
            def fw(x1, h1):
                def body(h, xx):
                    h2, lg, _, _ = rnet.apply(rp, h, xx); return h2, (h, lg)
                return jax.lax.scan(body, h1, x1)[1]
            hp, lgT = jax.vmap(fw)(x, h0)                               # pre-step memory and the policy with the true token
            tok = jnp.argmax(x[..., 100:108], -1); sh_ = jax.random.randint(key, tok.shape, 1, 8); tok2 = (tok + sh_) % 8
            heard = x[..., 108] > 0.5
            x2 = x.at[..., 100:108].set(jnp.where(heard[..., None], jax.nn.one_hot(tok2, 8), x[..., 100:108]))
            lgS = jax.vmap(jax.vmap(lambda h, xx: rnet.apply(rp, h, xx)[1]))(hp, x2)
            lT, lS = jax.nn.log_softmax(lgT), jax.nn.log_softmax(lgS); kl = (jnp.exp(lT) * (lT - lS)).sum(-1)
            w_ = heard * (m > 0); return (kl * w_).sum() / (w_.sum() + 1e-8), w_.sum()
        def token_influence(rp, b, key):
            v_, n_ = _tinf(rp, b[0], b[1], b[6], key); return dict(kl=float(v_), heard_steps=int(n_))

        def grad_diag(rp, b, adv, ret, v, al, act):
            """Cam Nov 12 diagnosis: which loss moves the shared memory, how big the learned policy term is next to the
            navigator's, and what the policy now does"""
            def part(q, i):
                _, P_ = jax.vmap(seq_loss, in_axes=(None,) + (0,) * 13)(q, *b); return P_[:, i].mean()
            groups = {"shared_memory(side_enc+gru)": ("side_enc", "gru"), "side_head": ("side_head",), "side_value": ("side_value",),
                      "aux_heads": ("aux_meal", "aux_other", "aux_view")}
            gn = {}
            for nm, i, wgt in (("ppo", 0, 1.0), ("meal", 1, a.aux), ("other", 2, a.aux), ("view", 3, a.aux)):
                g = jax.grad(part)(rp, i)["params"]
                gn[nm] = {gk: float(wgt * np.sqrt(sum(float((leaf ** 2).sum()) for m_ in mods for leaf in jax.tree_util.tree_leaves(g[m_])))) for gk, mods in groups.items()}
            rpn = {"params": {k_: (jax.tree_util.tree_map(jnp.zeros_like, v_) if k_ in ("side_head", "side_value") else v_) for k_, v_ in rp["params"].items()}}
            def fw(q, x, h0):
                def body(h, xx):
                    h2, lg, vv, _ = rnet.apply(q, h, xx); return h2, (lg, vv)
                return jax.lax.scan(body, h0, x)[1]
            (lgA, vA), (lgN, vN) = (jax.jit(jax.vmap(fw, in_axes=(None, 0, 0)))(q, b[0], b[1]) for q in (rp, rpn))
            m = np.asarray(b[6]) > 0; lgA, lgN, vA = np.asarray(lgA)[m], np.asarray(lgN)[m], np.asarray(vA)[m]
            side = lgA - lgN
            def ent(lg):
                p_ = np.exp(lg - lg.max(-1, keepdims=True)); p_ /= p_.sum(-1, keepdims=True); return p_, float(-(p_ * np.log(p_ + 1e-12)).sum(-1).mean())
            pA, eA = ent(lgA); pN, eN = ent(lgN)
            kl = float((pA * (np.log(pA + 1e-12) - np.log(pN + 1e-12))).sum(-1).mean())
            names = ["stay", "up", "down", "right", "left", "eat"]; alm = np.asarray(al, bool)
            out = dict(at_update=it, at_agent_steps=steps_train, aux_coef=a.aux,
                       grad_norm_by_loss_and_module=gn,
                       logit_norm=dict(trunk_navigator=float(np.linalg.norm(lgN - lgN.mean(-1, keepdims=True), axis=-1).mean()),
                                       side_head=float(np.linalg.norm(side - side.mean(-1, keepdims=True), axis=-1).mean())),
                       side_head_mean_contribution={n_: round(float(c_), 4) for n_, c_ in zip(names, (side - side.mean(-1, keepdims=True)).mean(0))},
                       policy_entropy=dict(agent=eA, navigator=eN, max=float(np.log(6))), kl_agent_to_navigator=kl,
                       action_prob_mean=dict(agent={n_: round(float(c_), 4) for n_, c_ in zip(names, pA.mean(0))}, navigator={n_: round(float(c_), 4) for n_, c_ in zip(names, pN.mean(0))}),
                       actions_taken={n_: round(float((np.asarray(act)[alm] == i_).mean()), 4) for i_, n_ in enumerate(names)},
                       value=dict(mean=float(vA.mean()), sd=float(vA.std()), navigator_value_mean=float(np.asarray(vN)[m].mean())),
                       returns=dict(mean=float(np.asarray(ret)[alm].mean()), sd=float(np.asarray(ret)[alm].std())),
                       advantage_raw=dict(mean=float(np.asarray(adv)[alm].mean()), sd=float(np.asarray(adv)[alm].std()), abs_mean=float(np.abs(np.asarray(adv)[alm]).mean())),
                       explained_variance=float(1 - np.var((np.asarray(ret) - np.asarray(v))[alm]) / (np.var(np.asarray(ret)[alm]) + 1e-12)))
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1, default=float)
            print("GRAD_DIAG", json.dumps(out, default=float), flush=True)

        # ---- training, with milestone readouts and checkpoints
        S_ = S; Hh = jnp.zeros((W, NL, Hd)); PA = jnp.zeros((W, NL), jnp.int32); PR = jnp.zeros((W, NL))
        COD = np.tile(np.arange(8), (W, M, 1)); COD[:, :NL] = FAM[cr.integers(0, len(FAM), (W, NL))]
        ag = np.asarray(S_.age).copy(); ag[:, NB:NB + NL] = cr.integers(0, 2500, (W, NL))   # staggered ages: juveniles always present
        S_ = S_._replace(age=jnp.asarray(ag))
        log, READS, key_ = [], [], jax.random.PRNGKey(a.seed + 7)
        MIL = [float(x) for x in a.readout_at.split(",")] if a.readout_at else []
        FMIL = [float(x) for x in a.forage_at.split(",")] if a.forage_at else []; FORAGES = []
        it, t_train, steps_train = 0, 0.0, 0; rng_main = rng
        if a.ckpt:
            Ck = _ck.load(a.ckpt)
            if Ck is not None:
                tj = lambda t_: jax.tree_util.tree_map(jnp.asarray, t_)  # noqa: E731
                rp, ros, S_, Hh, PA, PR = tj(Ck["rp"]), tj(Ck["ros"]), tj(Ck["S"]), tj(Ck["Hh"]), tj(Ck["PA"]), tj(Ck["PR"])
                COD, cr, key_, it, t_train, steps_train, log, READS = Ck["COD"], Ck["cr"], jnp.asarray(Ck["key"]), Ck["it"], Ck["t"], Ck["steps"], Ck["log"], Ck["reads"]
                rng_main.bit_generator.state = Ck["rng_state"]; MIL[:] = Ck["mil"]
                if "fmil" in Ck:
                    FMIL[:] = Ck["fmil"]; FORAGES[:] = Ck["forages"]
                print(f"resumed stage 1 at update {it}, {steps_train:,} agent-steps", flush=True)
                if any(r_.get("forage", {}).get("decision") == "STOP" for r_ in READS) and not (a.forage_check or a.return_check or a.redraw_probe or a.grad_diag):
                    print("FORAGE_STOP already decided in this checkpoint: not training on", flush=True); return
        if a.return_check:
            return return_check(rp)
        if a.redraw_probe:
            return redraw_probe(rp, a.redraw_probe, a.forage_worlds, a.forage_updates)
        if a.forage_check:
            if a.forage_control == "damaged":           # negative control: the side head pushes hard toward action 0
                rp = {"params": {**rp["params"], "side_head": {"kernel": rp["params"]["side_head"]["kernel"], "bias": rp["params"]["side_head"]["bias"].at[0].add(4.0)}}}
            print("forage-check on", "ckpt" if a.ckpt else "FRESH INIT (control: agent == navigator)", a.forage_control, flush=True)
            return forage_check(rp)
        def save_out():
            thr = steps_train / max(t_train, 1e-9)
            out = dict(train_updates_done=it, train_agent_steps=steps_train, train_seconds=round(t_train, 1),
                       train_throughput_agent_steps_per_s=round(thr, 1), worlds=W, T=T, hidden=Hd, readouts=READS,
                       lever_check=(dict(first=log[0], last=log[-1]) if log else None), forages=FORAGES)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(dict(summary=out, log=log), open(a.out, "w"), indent=1, default=float)
            return out
        while it < a.updates:
            t0 = _time.time(); key_, k = jax.random.split(key_); H0 = Hh
            S_, Hh, PA, PR, R = roll_rw(rp, S_, Hh, PA, PR, jax.random.split(k, W), jnp.asarray(COD), False)
            x, act, lp, v, r, dn, take, al, ea, pos = (np.asarray(z) for z in R[:10]); forced = np.asarray(R[10])
            adv = np.zeros_like(r); last = 0.0
            for t in reversed(range(T)):
                nv = v[:, t + 1] if t + 1 < T else v[:, t]; nt = 1.0 - dn[:, t]
                last = r[:, t] + a.gamma * nv * nt - v[:, t] + a.gamma * a.lam * nt * last; adv[:, t] = last
            ret = adv + v; m = (al & ~forced).astype(np.float32) if a.mask_forced else al.astype(np.float32)
            if a.ppo_min_age:
                m = m * ((np.asarray(R[0])[..., 99] * cfg.A_max) >= a.ppo_min_age - 0.5)
            A_ = (adv - (adv * m).sum() / max(m.sum(), 1)) / (adv.std() + 1e-8)
            lp_ = pos[:, :, :NL]; ym = np.full((W, T, NL), 8, np.int64); nxt = np.full((W, NL), -1)
            for t in reversed(range(T)):
                has = nxt >= 0
                if has.any():
                    tgt = np.take_along_axis(lp_, np.maximum(nxt, 0)[:, None, :, None].repeat(2, 3), 1)[:, 0]
                    d = (tgt - lp_[:, t] + cfg.L // 2) % cfg.L - cfg.L // 2
                    ym[:, t] = np.where(has & (np.abs(d).sum(-1) > 0), sec_np(d[..., 0], d[..., 1]), 8)
                nxt = np.where(take[:, t] > 0, t, nxt)
            mm = (ym < 8) & al
            dpar = (pos[:, :, NL:] - lp_ + cfg.L // 2) % cfg.L - cfg.L // 2
            yo = np.where(np.isin(ea, [1, 2, 3, 4]), ea, 0).astype(np.int64)
            agesT = np.asarray(S_.age)[:, NB:NB + NL]
            mo = (np.abs(dpar).max(-1) <= cfg.r) & al
            yv = np.concatenate([x[:, 1:, :, :49], x[:, -1:, :, :49]], 1); mv = np.concatenate([al[:, 1:] & al[:, :-1], np.zeros_like(al[:, :1])], 1)
            tr = lambda z: jnp.asarray(np.swapaxes(np.asarray(z), 1, 2).reshape((W * NL, T) + np.asarray(z).shape[3:]))  # noqa: E731
            b = (tr(x), jnp.asarray(np.asarray(H0).reshape(W * NL, Hd)), tr(act), tr(lp), tr(A_.astype(np.float32)), tr(ret.astype(np.float32)),
                 tr(m), tr(ym), tr(mm.astype(np.float32)), tr(yo), tr(mo.astype(np.float32)), tr(yv.astype(np.float32)), tr(mv.astype(np.float32)))
            if a.grad_diag:
                return grad_diag(rp, b, adv, ret, v, al, act)
            tinf = token_influence(rp, b, jax.random.fold_in(k, 99)) if (a.token_influence_every and it % a.token_influence_every == 0) else None
            for _ in range(a.epochs):
                rp, ros, parts = upd(rp, ros, *b)
            parts = np.asarray(parts)                     # per-sequence [ppo, meal, other, view, n_meal, n_other]
            wts = parts[:, 4:6]
            jax.block_until_ready(rp); dt = _time.time() - t0
            if it > 0:
                t_train += dt
            steps_train += W * T * NL
            log.append(dict(update=it, loss_ppo=float(parts[:, 0].mean()),
                            loss_meal=float((parts[:, 1] * wts[:, 0]).sum() / max(wts[:, 0].sum(), 1)),
                            loss_other=float((parts[:, 2] * wts[:, 1]).sum() / max(wts[:, 1].sum(), 1)), loss_view=float(parts[:, 3].mean()),
                            labels_meal=int(wts[:, 0].sum()), labels_other=int(wts[:, 1].sum()),
                            juvenile_frac=float((agesT < LEAVE).mean()), eaten_per_step=float(take[al].mean()) if al.any() else None,
                            forced_frac=float(forced[al].mean()) if al.any() else None, token_influence=tinf, sec=round(dt, 3)))
            dead = ~np.asarray(S_.alive)[:, NB:NB + NL]
            for w, l in zip(*np.where(dead)):
                COD[w, l] = FAM[cr.integers(0, len(FAM))]
            if dead.any():
                S_, Hh, PA, PR = respawn(S_, Hh, PA, PR, dead, rng_main)
            it += 1
            if (it % 10) == 0:
                print(f"stage1 update {it}: {json.dumps(log[-1])}", flush=True)
            while MIL and steps_train >= MIL[0]:
                tagm = f"{MIL[0]:.0e}"; MIL.pop(0); t1 = _time.time()
                READS.append(dict(at_agent_steps=steps_train, **readout(rp, tagm), readout_seconds=round(_time.time() - t1, 1))); save_out()
                print("STAGE1_READOUT", json.dumps(READS[-1])[:2500], flush=True)
                if a.forage_stop:                          # Cam's foraging rule, enforced in the run (live milestone weights)
                    rpf = rp
                    if a.forage_control == "damaged":      # test only
                        rpf = {"params": {**rp["params"], "side_head": {"kernel": rp["params"]["side_head"]["kernel"], "bias": rp["params"]["side_head"]["bias"].at[0].add(4.0)}}}
                    fo = forage_check(rpf, write=False)
                    pts_ = [r_["point"] for r_ in (fo["ratio_km_median"], fo["ratio_adult_food_per_step"]) if r_["point"] is not None]
                    worst = min(pts_) if pts_ else None; READS[-1]["forage"] = dict(worst_ratio=worst, check=fo)
                    nmil = len(READS); prev = READS[-2]["forage"]["worst_ratio"] if nmil >= 2 and "forage" in READS[-2] else None
                    stop = worst is not None and ((nmil == 1 and worst < 0.5) or
                                                  (nmil == 2 and (worst < 0.5 or (prev is not None and 0.5 <= prev < 0.9 and worst < prev))))
                    READS[-1]["forage"]["decision"] = "STOP" if stop else ("RECHECK_NEXT" if worst is not None and worst < 0.9 else "CARRY_ON")
                    save_out(); print("FORAGE_RULE", json.dumps(dict(milestone=nmil, worst_ratio=worst, prev=prev, decision=READS[-1]["forage"]["decision"])), flush=True)
                    if stop:
                        if a.ckpt:
                            _ck.save(a.ckpt, dict(rp=rp, ros=ros, S=S_, Hh=Hh, PA=PA, PR=PR, COD=COD, cr=cr, key=key_, it=it, t=t_train, steps=steps_train,
                                                  log=log, reads=READS, rng_state=rng_main.bit_generator.state, mil=[]))
                        print("FORAGE_STOP: the run stops here under the registered foraging rule", flush=True)
                        return
            while FMIL and steps_train >= FMIL[0]:          # Cam Nov 13: forage checks along training, for the slope
                FMIL.pop(0); fo = forage_check(rp, write=False)
                FORAGES.append(dict(at_agent_steps=steps_train, adult_food_ratio=fo["ratio_adult_food_per_step"], km_ratio=fo["ratio_km_median"],
                                    agent=dict(survival_past_1000=fo["agent"]["survival_past_1000"], survival_past_1300=fo["agent"]["survival_past_1300"],
                                               adult_food_per_step=fo["agent"]["adult_food_per_step"]))); save_out()
            if a.ckpt and (it % a.ckpt_every == 0):
                _ck.save(a.ckpt, dict(rp=rp, ros=ros, S=S_, Hh=Hh, PA=PA, PR=PR, COD=COD, cr=cr, key=key_, it=it, t=t_train, steps=steps_train,
                                      log=log, reads=READS, rng_state=rng_main.bit_generator.state, mil=list(MIL), fmil=list(FMIL), forages=FORAGES))
            if a.stop_after and it == a.stop_after:
                os._exit(3)
        if not READS and a.eval_updates:
            READS.append(dict(at_agent_steps=steps_train, **readout(rp, "final")))
        out = save_out()
        if a.final_return_check:                           # Cam Nov 13: return-to-go by age band on the final weights
            out["return_check"] = return_check(rp, write=False)
            json.dump(dict(summary=out, log=log), open(a.out, "w"), indent=1, default=float)
        print("STAGE1_THROUGHPUT", json.dumps(dict(train_throughput_agent_steps_per_s=out["train_throughput_agent_steps_per_s"])), flush=True)
        print("STAGE1", json.dumps(out, default=float)[:3000], flush=True)

    if a.agent == "recurrent":
        return stage1_recurrent()
    it0 = 0
    if a.ckpt:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        import ckpt as _ck
        Ck = _ck.load(a.ckpt)
        if Ck is not None:
            tj = lambda t: jax.tree_util.tree_map(jnp.asarray, t)  # noqa: E731
            P, OS, S = tj(Ck["P"]), tj(Ck["OS"]), tj(Ck["S"]); key = jnp.asarray(Ck["key"]); rng = Ck["rng"]
            mem_o, mem_m = Ck["mem_o"], Ck["mem_m"]; mem_e = [(jnp.asarray(e0), jnp.asarray(e1), e2) for e0, e1, e2 in Ck["mem_e"]]
            since_birth, gsteps, cnt, lives, nav_of = Ck["since_birth"], Ck["gsteps"], Ck["cnt"], Ck["lives"], Ck["nav_of"]
            snaps, prev_age, snap_gs = Ck.get("snaps", {}), Ck.get("prev_age", prev_age), Ck.get("snap_gs", {})
            if a.code_per_life:
                CODES, crng = Ck["codes"], Ck["crng"]
            it0 = Ck["it"] + 1; print(f"resumed from update {Ck['it'] + 1}", flush=True)
    for it in range(it0, a.updates):
        key, k = jax.random.split(key)
        S, (o, act, lp, v, r, dn, take, eo, ea, ne, alive, yc, mpos) = (roll_w(P, S, jax.random.split(k, W), jnp.asarray(CODES))
                                                                if a.code_per_life else roll_w(P, S, jax.random.split(k, W)))
        o, act, lp, v, r, dn, alive = (np.asarray(x) for x in (o, act, lp, v, r, dn, alive))   # (W, T, NL, ...)
        if a.ic_probe:                                 # (W, T, NL, ...) per update, with the codes and life ids in force
            IC["o"].append(o.astype(np.float32)); IC["act"].append(np.asarray(act)); IC["r"].append(r.astype(np.float32))
            IC["alive"].append(alive); IC["codes"].append(CODES[:, :NL].copy()); IC["lid"].append(life_id.copy())
            IC["take"].append(np.asarray(take)); IC["ea"].append(np.asarray(ea)); IC["pos"].append(np.asarray(mpos))
        hm = (o[..., 108] > 0.5) & alive
        cnt += np.einsum("wtl,wtlk->wlk", hm.astype(np.int64), (o[..., 100:108] > 0.5).astype(np.int64))
        adv = np.zeros_like(r); last = 0.0
        for t in reversed(range(T)):
            nv = v[:, t + 1] if t + 1 < T else v[:, t]; nt = 1.0 - dn[:, t]
            last = r[:, t] + 0.99 * nv * nt - v[:, t] + 0.99 * 0.95 * nt * last; adv[:, t] = last
        ret = adv + v; m = alive.astype(np.float32)
        A = (adv - (adv * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)) / (adv.std() + 1e-8)
        tr = lambda x: jnp.asarray(np.swapaxes(np.asarray(x), 1, 2))  # noqa: E731
        Dd = o.shape[-1]
        if NE:
            eo_, ea_ = np.asarray(eo), np.asarray(ea)                                       # (W, T, NE, D), (W, T, NE)
            em = (np.asarray(ne) & np.asarray(yc)[..., None] & alive[..., None]).astype(np.float32)   # (W, T, NL, NE)
            EO = jnp.asarray(np.broadcast_to(eo_[:, None], (W, NL) + eo_.shape[1:]).reshape(W, NL, T * NE, Dd))
            EA = jnp.asarray(np.broadcast_to(ea_[:, None], (W, NL) + ea_.shape[1:]).reshape(W, NL, T * NE))
            EM = jnp.asarray(np.swapaxes(em, 1, 2).reshape(W, NL, T * NE))
        else:
            EO = jnp.zeros((W, NL, 1, Dd)); EA = jnp.zeros((W, NL, 1), jnp.int32); EM = jnp.zeros((W, NL, 1))
        agel = np.asarray(S.age)[:, NB:NB + NL]
        fy = jnp.asarray(1.0 + (a.young_lr - 1.0) * np.clip(1.0 - agel / max(a.juvenile, 1), 0.0, 1.0), jnp.float32)
        if a.log_levers:                              # each lever's own signal, on this rollout, before the updates
            if "lev_f" not in dir():
                def _lev(p, o, m, eo, ea, em):
                    pl = pred_loss(p, o, m > 0) if a.pred else 0.0
                    if a.copy and NE:
                        lge, _, _ = net.apply(p, eo)
                        ce = -jnp.take_along_axis(jax.nn.log_softmax(lge), ea[:, None], 1)[:, 0]
                        cl = (ce * em).sum() / (em.sum() + 1e-8)
                    else:
                        cl = 0.0
                    return jnp.asarray(pl), jnp.asarray(cl), em.sum()
                lev_f = jax.jit(jax.vmap(jax.vmap(_lev)))
            pl_, cl_, cn_ = (np.asarray(x) for x in lev_f(P, tr(o), tr(m), EO, EA, EM))
            young_m = agel < a.juvenile if a.juvenile else np.zeros_like(agel, bool)
            rec_ = dict(update=it, pred_loss_mean=float(pl_.mean()), copy_ce_mean=float(cl_[cn_ > 0].mean()) if (cn_ > 0).any() else None,
                        copy_rows=float(cn_.sum()), learners_copying=int((cn_ > 0).sum()),
                        replay_passes=int(a.replay * len(mem_o)) if (a.replay and (a.pred or a.copy_replay)) else 0,
                        young_lr_factor_young=float(np.asarray(fy)[young_m].mean()) if young_m.any() else None,
                        young_lr_factor_all=float(np.asarray(fy).mean()), mean_age=float(agel.mean()))
            open(a.out + ".levers.jsonl", "a").write(json.dumps(rec_) + "\n")
        for _ in range(0 if a.no_train else a.epochs):
            P, OS = upd(P, OS, fy, tr(o), tr(act), tr(lp), tr(A.astype(np.float32)), tr(ret.astype(np.float32)), tr(m), EO, EA, EM)
        gsteps += a.epochs
        since_birth += 1
        if a.log_levers:                              # after the PPO epochs, before replay (isolates replay's effect)
            pl1, cl1, cn1 = (np.asarray(x) for x in lev_f(P, tr(o), tr(m), EO, EA, EM))
            open(a.out + ".levers.jsonl", "a").write(json.dumps(dict(update=it, after_epochs=True, pred_loss_mean=float(pl1.mean()),
                                                                      copy_ce_mean=float(cl1[cn1 > 0].mean()) if (cn1 > 0).any() else None)) + "\n")
        if a.replay and (a.pred or a.copy_replay) and not a.no_train:
            mem_o.append(np.swapaxes(o, 1, 2)); mem_m.append(np.swapaxes(m, 1, 2)); mem_e.append((EO, EA, np.asarray(EM)))
            mem_o, mem_m, mem_e = mem_o[-a.mem:], mem_m[-a.mem:], mem_e[-a.mem:]
            for _ in range(a.replay):
                for j in range(len(mem_o)):          # own memory since birth only: older rollouts masked per learner
                    age_ok = (len(mem_o) - j) <= since_birth
                    eoj, eaj, emj = mem_e[j]
                    P, OS = upd_replay(P, OS, fy, jnp.asarray(mem_o[j]), jnp.asarray(mem_m[j] * age_ok[..., None]),
                                       eoj, eaj, jnp.asarray(emj * age_ok[..., None]))
            gsteps += a.replay * len(mem_o)
            if a.log_levers:                          # replay's own effect: the same rollout's losses after the replay passes
                pl2, cl2, cn2 = (np.asarray(x) for x in lev_f(P, tr(o), tr(m), EO, EA, EM))
                open(a.out + ".levers.jsonl", "a").write(json.dumps(dict(update=it, after_replay=True, pred_loss_mean=float(pl2.mean()),
                                                                          copy_ce_mean=float(cl2[cn2 > 0].mean()) if (cn2 > 0).any() else None)) + "\n")
        dead = ~np.asarray(S.alive)[:, NB:NB + NL]
        ages = np.asarray(S.age)[:, NB:NB + NL]
        for k_ in SNAP:                             # weights snapshot when a life first reaches age k_
            for w, l in zip(*np.where((prev_age < k_) & (ages >= k_) & ~dead)):
                snaps.setdefault((int(w), int(l)), []).append((k_, jax.tree_util.tree_map(lambda x: np.asarray(x[w, l]), P)))
                snap_gs.setdefault((int(w), int(l)), {})[str(k_)] = int(gsteps[w, l])
        prev_age = np.where(dead, 0, ages)
        for w, l in zip(*np.where(dead)):
            hc = cnt[w, l][CODES[w, l]] if a.code_per_life else cnt[w, l].copy()     # token counts -> sector space
            lives.append(dict(lifespan=int(ages[w, l]), grad_steps=int(gsteps[w, l]), heard=hc, bpos=np.asarray(S.pos)[w, :NB].copy(),
                              code=CODES[w, l].copy() if a.code_per_life else None,
                              key=(int(w), int(l), int(life_id[w, l])) if a.ic_probe else None,
                              params=jax.tree_util.tree_map(lambda x: x[w, l], P), centres=np.asarray(S.centres[w]),
                              snaps=snaps.pop((int(w), int(l)), []), snap_gsteps=snap_gs.pop((int(w), int(l)), {})))
            nav_of[w, l] = rng.integers(0, K)
            if a.code_per_life:
                CODES[w, l] = crng.permutation(8)       # a new life, a new code
            if a.ic_probe:
                life_id[w, l] += 1
        if dead.any():
            fresh = jax.tree_util.tree_map(lambda x: x[nav_of], NAV); dm = jnp.asarray(dead)
            P = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), P, fresh)
            OS = jax.tree_util.tree_map(lambda x, y: jnp.where(dm.reshape(dm.shape + (1,) * (x.ndim - 2)), y, x), OS,
                                        jax.vmap(jax.vmap(opt.init))(fresh))
            since_birth[dead] = 0; gsteps[dead] = 0; cnt[dead] = 0
            npos = (np.asarray(S.pos)[:, :NB].mean(1, keepdims=True).astype(int) + rng.integers(-3, 4, (W, NL, 2))) % cfg.L
            pos = np.asarray(S.pos).copy(); E = np.asarray(S.energy).copy(); age = np.asarray(S.age).copy(); al = np.asarray(S.alive).copy()
            sl = slice(NB, NB + NL)
            pos[:, sl][dead] = npos[dead]; E[:, sl][dead] = cfg.E0; age[:, sl][dead] = 0; al[:, sl][dead] = True
            if a.parent:                               # a newborn's parent arrives beside it
                pos[:, NB + NL:][dead] = (npos[dead] + np.array([1, 0])) % cfg.L
            S = S._replace(pos=jnp.asarray(pos), energy=jnp.asarray(E), age=jnp.asarray(age), alive=jnp.asarray(al))
        if a.ckpt and (it + 1) % a.ckpt_every == 0 and it + 1 < a.updates:
            _ck.save(a.ckpt, dict(it=it, P=P, OS=OS, S=S, key=key, rng=rng, mem_o=mem_o, mem_m=mem_m, mem_e=mem_e,
                                  since_birth=since_birth, gsteps=gsteps, cnt=cnt, lives=lives, nav_of=nav_of, snaps=snaps, prev_age=prev_age, snap_gs=snap_gs,
                                  codes=CODES, crng=crng))
        if a.stop_after and it + 1 == a.stop_after:
            os._exit(3)
    if a.eval_eligible_only and lives:
        n_all = len(lives)
        lives = [x for x in lives if int((x["heard"] >= a.min_heard).sum()) >= a.min_sectors]
        OBS_INFO.update(lives_before_filter=n_all, eligible_before_filter=len(lives), eligibility_rate_all=len(lives) / n_all)
        if not lives:
            print("no eligible lives"); return
    if a.ic_probe and a.io_decoder and not a.obs_listener:
        return ideal_observer(a, IC, W, NL, NE, T, cfg)
    if a.ic_probe and not a.obs_listener:
        return incontext_probe(a, IC, W, NL, T)
    if not lives:
        print("no completed lives"); return
    diag = None
    if a.diag and a.pred:
        def pred_trace(p, centres, key):
            """a true-message test episode; at each step the head's food-in-view-4-ahead prediction with the heard
            sector as heard and with it shifted by +4 (on the SAME observation), plus the realised target"""
            s = env.reset(cfg, key)
            s = s._replace(centres=centres, cap=env.capacity(cfg, centres), food=env.capacity(cfg, centres))
            s, _ = beacons_reset(s, jax.random.fold_in(key, 3))
            s = s._replace(alive=jnp.array([True] * NB + [True] + [False] * (M - 1)))
            P1 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NL,) + x.shape), p)
            def body(s, k):
                h = hear(s, 0); o = jnp.concatenate([env.observe(cfg, s)[NB:], h], 1)[0]
                osw = o.at[100:108].set(jnp.roll(o[100:108], 4))
                _, _, y = net.apply(p, o[None]); _, _, ys = net.apply(p, osw[None])
                s2, _ = step_all(P1, s, k, 0, jnp.zeros(NL, bool))
                s2 = s2._replace(alive=s2.alive.at[NB + 1:].set(False))
                return s2, (y[0, 99], ys[0, 99], o[:49].max(), o[108] > 0.5)
            _, (yt, ys, fv, heard) = jax.lax.scan(body, s, jax.random.split(jax.random.fold_in(key, 5), 300))
            return yt, ys, fv, heard
        idx = np.linspace(0, len(lives) - 1, min(a.diag, len(lives))).astype(int)
        errs_t, errs_s = [], []
        for i in idx:
            for e in range(4):
                yt, ys, fv, hd = (np.asarray(x) for x in jax.jit(pred_trace)(lives[i]["params"], jnp.asarray(lives[i]["centres"]),
                                                                              jax.random.PRNGKey(9000 + 10 * int(i) + e)))
                tgt = np.concatenate([fv[4:], np.full(4, np.nan)]); mk = hd & ~np.isnan(tgt)
                if mk.sum() > 5:
                    errs_t.append(np.mean((yt[mk] - tgt[mk]) ** 2)); errs_s.append(np.mean((ys[mk] - tgt[mk]) ** 2))
        et, es_ = np.array(errs_t), np.array(errs_s); dd = es_ - et
        diag = dict(n=len(et), mse_true=float(et.mean()), mse_swapped=float(es_.mean()),
                    swap_increase=float(dd.mean() / max(et.mean(), 1e-9)), paired_z=float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)))
        print("DIAG predictive head uses the message?", json.dumps(diag), flush=True)
        if a.diag_only:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(diag, open(a.out, "w"), indent=1); return
    Pz = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[x["params"] for x in lives])
    C = jnp.asarray(np.stack([x["centres"] for x in lives]))
    keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(7000 + i), a.eval_eps))(jnp.arange(len(lives)))
    scorable = np.ones(len(lives), bool)
    if a.test == "heard25":
        rq = np.random.default_rng(4242); E_ = a.eval_eps
        ident = np.tile(np.arange(8), (len(lives), E_, 1)); der = ident.copy()
        for i, x in enumerate(lives):
            Sset = np.where(x["heard"] >= a.min_heard)[0]
            if len(Sset) < max(2, a.min_sectors):
                scorable[i] = False
            if len(Sset) < 2:
                continue
            for e in range(E_):
                while True:
                    q = rq.permutation(Sset)
                    if np.all(q != Sset):
                        break
                der[i, e, Sset] = q
        if a.code_per_life:                            # learner = agent 0 of M in the life test; the rest are dead
            CL = np.tile(np.arange(8), (len(lives), M, 1)); CL[:, 0] = np.stack([x["code"] for x in lives]); EVAL_CODE["all"] = CL
        if a.obs_listener:                             # Cam Nov 12 plug-in check: scripted listener through a per-life table
            DECS = observer_decs(a, IC, W, NL, NE, T, cfg) if a.obs_listener == "observer" else {}
            DL = np.tile(np.arange(8), (len(lives), M, 1)); nodec = 0
            for i_, x in enumerate(lives):
                if a.obs_listener == "true":
                    DL[i_, 0] = np.argsort(x["code"])
                elif x["key"] in DECS:
                    DL[i_, 0] = DECS[x["key"]]
                else:
                    nodec += 1; DL[i_, 0] = np.random.default_rng(i_).integers(0, 8, 8)
            EVAL_CODE["dec"] = DL; a.sanity = True; OBS_INFO.update(mode=a.obs_listener, lives_without_stream=nodec)
        evp = make_evp(); BP = jnp.asarray(np.stack([x["bpos"] for x in lives]))
        SM = jnp.asarray(np.stack([(x["heard"] >= a.min_heard).astype(np.float32) for x in lives]))
        true = np.asarray(evp(Pz, C, keys, jnp.asarray(ident), BP, SM)); swap = np.asarray(evp(Pz, C, keys, jnp.asarray(der), BP, SM))
    else:
        true = np.asarray(ev(Pz, C, keys, 0)); swap = np.asarray(ev(Pz, C, keys, 1))
    blind = silent = blind_swap = None; evaluate_voi_r = None
    if a.no_train and a.sanity and a.test == "heard25":   # the blind comparison on the SAME lives / keys: the navigator policy
        a.sanity = False
        evn = make_evp()
        blind = np.asarray(evn(Pz, C, keys, jnp.asarray(np.tile(np.arange(8), (len(lives), a.eval_eps, 1))), BP, SM))
        blind_swap = np.asarray(evn(Pz, C, keys, jnp.asarray(der), BP, SM))        # negative control: the non-listener's life test
        a.sanity = True
        I8 = jnp.asarray(np.tile(np.arange(8), (len(lives), a.eval_eps, 1)))
        silent = np.asarray(make_evp(mute=True)(Pz, C, keys, I8, BP, SM))
        # appetite-free VoI (Nov 10): the scripted listener's own uninformed move (random) when it hears nothing, both arms
        true_r = np.asarray(make_evp(rand_fb=True)(Pz, C, keys, I8, BP, SM)); silent_r = np.asarray(make_evp(mute=True, rand_fb=True)(Pz, C, keys, I8, BP, SM))
        evaluate_voi_r = (true_r, silent_r)
    res = []
    for i, x in enumerate(lives):
        t, s_ = true[i], swap[i]; d = t - s_; se = d.std(ddof=1) / np.sqrt(len(d))
        drop = 1 - s_.sum() / max(t.sum(), 1e-9)
        res.append(dict(lifespan=x["lifespan"], grad_steps=x["grad_steps"], eaten_true=float(t.mean()), eaten_swap=float(s_.mean()),
                        drop=float(drop), passed=bool(scorable[i] and drop >= 0.30 and d.mean() > 2 * se), scorable=bool(scorable[i]),
                        distinct_heard=int((x["heard"] > 0).sum()), sectors_ge_min=int((x["heard"] >= a.min_heard).sum()),
                        eaten_blind=float(blind[i].mean()) if blind is not None else None))
    frac = np.mean([r_["passed"] for r_ in res])
    summ = dict(variant=dict(juvenile=a.juvenile, long_life=a.long_life, pred=a.pred, replay=a.replay, copy=a.copy,
                             copy_replay=a.copy_replay, young_lr=a.young_lr,
                             n_elders=NE, sanity=a.sanity, random_code=a.random_code, updates=a.updates, T=T, epochs=a.epochs),
                lives=len(res), pass_frac=float(frac), median_drop=float(np.median([r_["drop"] for r_ in res])),
                median_lifespan=float(np.median([r_["lifespan"] for r_ in res])),
                median_grad_steps_per_life=float(np.median([r_["grad_steps"] for r_ in res])),
                eaten_true_mean=float(np.mean([r_["eaten_true"] for r_ in res])),
                eaten_swap_mean=float(np.mean([r_["eaten_swap"] for r_ in res])), gate="PASS" if frac >= 0.5 else "FAIL",
                diag=diag)
    if a.no_train:
        dh = np.array([r_["distinct_heard"] for r_ in res]); gm = np.array([r_["sectors_ge_min"] for r_ in res])
        summ["apparatus"] = dict(
            test_start=a.test_start, score=a.score, beacon_spread=a.beacon_spread, beacon_walk=a.beacon_walk, young_seek=a.young_seek, world=a.world, test=a.test,
            test_window=a.test_window, eval_eps=a.eval_eps, policy="scripted listener" if a.sanity else "navigator (no training)",
            coverage_median_distinct=float(np.median(dh)), coverage_median_sectors_ge_min=float(np.median(gm)),
            coverage_hist_distinct=np.bincount(dh, minlength=9).tolist(),
            gate_i=bool(np.median(dh) >= 4 and np.median(gm) >= 3), scorable_frac=float(scorable.mean()))
        if blind is not None:
            eb = float(np.mean([r_["eaten_blind"] for r_ in res])); summ["apparatus"]["eaten_blind_mean"] = eb
            summ["apparatus"]["scripted_over_blind"] = float(summ["eaten_true_mean"] / max(eb, 1e-9))
            summ["apparatus"]["gate_ii"] = bool(frac >= 0.9 and summ["eaten_true_mean"] >= 2 * eb)
            es = float(silent.mean()); summ["apparatus"]["eaten_scripted_silence_mean"] = es
            summ["apparatus"]["scripted_true_over_silence"] = float(summ["eaten_true_mean"] / max(es, 1e-9))
    if a.test == "heard25":
        summ["heard25"] = heard25_report(a, lives, res, scorable, true, swap, der, blind, blind_swap, silent)
        if OBS_INFO:
            summ["heard25"]["obs_listener"] = OBS_INFO
        if a.crossfoster:
            summ["crossfoster"] = crossfoster_calibration(lives, scorable, Pz, C, keys, SM)
        if a.parent:
            summ["heard25"]["parent"] = dict(leave_age=a.parent_leave or a.juvenile)
        if evaluate_voi_r is not None:
            el = np.asarray(scorable); tm, sm = evaluate_voi_r[0][el].mean(1), evaluate_voi_r[1][el].mean(1); d = tm - sm
            summ["heard25"]["voi_random_fallback"] = dict(eaten_true=float(tm.mean()), eaten_silence=float(sm.mean()),
                                                          ratio=float(tm.mean() / max(sm.mean(), 1e-9)),
                                                          paired_z=float(d.mean() / max(d.std(ddof=1) / np.sqrt(len(d)), 1e-12)))
        if SNAP:                                     # the same test on the age snapshots: steps to first reliance
            byage = {}; first = np.full(len(lives), np.inf)
            for k_ in SNAP:
                idx = [i for i, x in enumerate(lives) if scorable[i] and any(kk == k_ for kk, _ in x["snaps"])]
                if not idx:
                    byage[str(k_)] = dict(n=0); continue
                Pk = jax.tree_util.tree_map(lambda *z: jnp.stack([jnp.asarray(q) for q in z]), *[dict(x["snaps"])[k_] for x in (lives[i] for i in idx)])
                ii = np.array(idx); cdi = EVAL_CODE["all"][ii] if a.code_per_life else None
                if EVAL_CODE.get("dec") is not None:
                    EVAL_CODE["dec_sub"] = EVAL_CODE["dec"][ii]
                tk = np.asarray(evp(Pk, C[ii], keys[ii], jnp.asarray(ident[ii]), BP[ii], SM[ii], cdi))
                sk = np.asarray(evp(Pk, C[ii], keys[ii], jnp.asarray(der[ii]), BP[ii], SM[ii], cdi))
                pk = np.array([_life_pass(tk[j], sk[j]) for j in range(len(idx))])
                for j, i in enumerate(idx):
                    res[i].setdefault("snap_pass", {})[str(k_)] = bool(pk[j])     # per-life, for paired age trends
                    res[i].setdefault("snap_gsteps", {})[str(k_)] = lives[i]["snap_gsteps"].get(str(k_))
                    if pk[j]:
                        first[i] = min(first[i], k_)
                byage[str(k_)] = dict(n=len(idx), pass_frac=float(pk.mean()), ci95=_wilson(int(pk.sum()), len(idx)))
            endp = np.array([r_["passed"] for r_ in res])
            for i in range(len(lives)):
                if scorable[i] and endp[i]:
                    first[i] = min(first[i], lives[i]["lifespan"])
            fe = first[np.asarray(scorable)]
            summ["heard25"]["by_age"] = byage
            summ["heard25"]["first_reliance"] = dict(ever_frac=float(np.isfinite(fe).mean()) if len(fe) else None,
                                                     median_age_among_ever=float(np.median(fe[np.isfinite(fe)])) if np.isfinite(fe).any() else None,
                                                     median_lifespan=float(np.median([x["lifespan"] for x in lives])))
        summ["gate"] = "PASS" if summ["heard25"]["learner_pass_frac_eligible"] >= 0.5 else "FAIL"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summ, lives=res), open(a.out, "w"), indent=1, default=lambda z: z.tolist() if hasattr(z, "tolist") else str(z))
    print(json.dumps(summ))


def _life_pass(t, s_):
    d = t - s_; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
    return bool((1 - s_.sum() / max(t.sum(), 1e-9)) >= 0.30 and d.mean() > 2 * se)


def _wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def heard25_report(a, lives, res, scorable, true, swap, der, blind, blind_swap, silent):
    """Cam Nov 10 rulings: last-heard metric; a life is eligible iff it heard >= min-sectors sectors >= min-heard x; pass
    rates over ELIGIBLE lives (>= 50 needed), eligibility rate, stratified 5 / 6 / 7-8; secondary (no bar): probes whose
    swap moved the well-heard sectors >= 90 degrees on average; apparatus: negative control (non-listener) and VoI (true vs silence)."""
    el = np.asarray(scorable); n_el = int(el.sum()); gm = np.array([r_["sectors_ge_min"] for r_ in res])
    passed = np.array([r_["passed"] for r_ in res])
    out = dict(min_sectors=a.min_sectors, min_heard=a.min_heard, score=a.score, lives=len(res), eligible=n_el,
               eligibility_rate=float(el.mean()), enough_eligible=n_el >= 50,
               learner_pass_frac_eligible=float(passed[el].mean()) if n_el else float("nan"),
               learner_pass_ci95=_wilson(int(passed[el].sum()), n_el))
    out["stratified"] = {k: dict(n=int(m.sum()), pass_frac=float(passed[m].mean()) if m.any() else None)
                         for k, m in (("5", el & (gm == 5)), ("6", el & (gm == 6)), ("7-8", el & (gm >= 7)))}
    # secondary: per life, only the probes whose derangement moved every well-heard sector >= 90 degrees (>= 2 sectors)
    sec = []
    for i, x in enumerate(lives):
        if not el[i]:
            continue
        S_ = np.where(x["heard"] >= a.min_heard)[0]
        mv = np.abs(der[i][:, S_] - S_[None]); mv = np.minimum(mv, 8 - mv)
        ok = mv.mean(1) >= 2                        # mean move over the well-heard sectors >= 90 degrees (2 sector steps)
        if ok.sum() >= 4:
            sec.append(_life_pass(true[i][ok], swap[i][ok]))
    out["secondary_swap_ge90deg"] = dict(lives_with_ge4_such_probes=len(sec), pass_frac=float(np.mean(sec)) if sec else None)
    if blind_swap is not None:
        neg = np.array([_life_pass(blind[i], blind_swap[i]) for i in range(len(lives))])
        out["negative_control_nonlistener"] = dict(pass_frac=float(neg[el].mean()) if n_el else None,
                                                   ci95=_wilson(int(neg[el].sum()), n_el), ok=bool(neg[el].mean() <= 0.10) if n_el else False)
        out["scripted_control"] = dict(pass_frac=out["learner_pass_frac_eligible"], ci95=out["learner_pass_ci95"],
                                       ok=bool(out["learner_pass_frac_eligible"] >= 0.90))
    if silent is not None:
        tm, sm = true[el].mean(1), silent[el].mean(1); d = tm - sm; se = d.std(ddof=1) / np.sqrt(len(d))
        ratio = float(tm.mean() / max(sm.mean(), 1e-9))
        out["voi_true_vs_silence"] = dict(eaten_true=float(tm.mean()), eaten_silence=float(sm.mean()), ratio=ratio,
                                          paired_z=float(d.mean() / max(se, 1e-12)), ok=bool(ratio >= 1.5 and d.mean() > 2 * se))
    if blind_swap is not None and silent is not None:
        out["gate_ii"] = bool(out["scripted_control"]["ok"] and out["negative_control_nonlistener"]["ok"]
                              and out["voi_true_vs_silence"]["ok"] and out["enough_eligible"])
    return out


def hearing_probe(a, cfg, net, opt, NAV, nav_of, S, roll_w, ev, hear, W, NL, T, M):
    """Cam Nov 10 ~05:15 (4a). Train ONLY the hearing rows (Dense_0 kernel rows 100:111) + the policy head (Dense_2)
    of each learner, starting from its navigator, by cross-entropy on the scripted listener's actions given
    (obs, heard), Adam at the arm's rate (3e-4) x a multiplier, --probe-steps steps of batch 128 from the learner's own
    world's teacher data. Report held-out action accuracy (other worlds) and the harness's unchanged life-test gate.
    Pipeline checks: the input layout, the heard sector vs an independent beacon -> nearest-patch sector, the
    gradient reaching the heard rows, the heard input changing the logits after training."""
    P0 = jax.tree_util.tree_map(lambda x: x[nav_of], NAV)
    a.sanity = True                                           # the scripted listener drives the collection rollouts

    def collect(S, seed):
        key = jax.random.PRNGKey(seed); Os, As, Ms = [], [], []
        for _ in range(a.probe_collect):
            key, k = jax.random.split(key)
            S, rec = roll_w(P0, S, jax.random.split(k, W))
            Os.append(np.asarray(rec[0])); As.append(np.asarray(rec[1])); Ms.append(np.asarray(rec[10]))
        return np.concatenate(Os, 1), np.concatenate(As, 1), np.concatenate(Ms, 1)       # (W, Tn, NL, ...)

    def fresh_worlds(base):
        st = []
        for w in range(W):
            s0 = env.reset(cfg, jax.random.PRNGKey(base + w))
            c = jax.random.randint(jax.random.PRNGKey(base + 500 + w), (2,), 0, cfg.L)
            bpos = (c[None] + jnp.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])) % cfg.L
            mpos = (c[None] + jax.random.randint(jax.random.PRNGKey(base + 900 + w), (M, 2), -3, 4)) % cfg.L
            st.append(s0._replace(pos=jnp.concatenate([bpos, mpos], 0)))
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *st)

    O, A, Mk = collect(S, 11)
    Oh, Ah, Mh = collect(fresh_worlds(30_000), 12)                                     # held-out worlds
    a.sanity = False                                          # the life test (traced below) uses the learners' own policy
    D = O.shape[-1]
    heard = (O[..., 108] > 0.5) & Mk; heard_h = (Oh[..., 108] > 0.5) & Mh
    # code-driven states: nothing to eat underfoot (centre cell < 0.2) and no food seen (> 0.2) in view
    def code_driven(o):
        r = cfg.r; wv = o[..., :(2 * r + 1) ** 2]; centre = wv[..., ((2 * r + 1) ** 2) // 2]
        return (centre < 0.2) & (wv.max(-1) <= 0.2)
    cd_h = heard_h & code_driven(Oh)

    # ---- pipeline checks
    chk = dict(input_dim=int(D), heard_dims="100:108 one-hot sector, 108 flag, 109:111 offset / R_HEAR",
               kernel_rows=int(P0["params"]["Dense_0"]["kernel"].shape[-2]))
    oh = O[heard]; chk["onehot_sums_to_1"] = bool(np.allclose(oh[:, 100:108].sum(1), 1.0))
    chk["offset_within_unit"] = bool(np.abs(oh[:, 109:111]).max() <= 1.0 + 1e-6)
    chk["unheard_all_zero"] = bool(np.abs(O[(O[..., 108] < 0.5) & Mk][:, 100:111]).max() == 0.0) if ((O[..., 108] < 0.5) & Mk).any() else True
    chk["heard_frac_alive_states"] = float(heard.sum() / max(Mk.sum(), 1))
    chk["code_driven_frac_heard_heldout"] = float(cd_h.sum() / max(heard_h.sum(), 1))
    # independent sector: each world's beacons -> nearest patch, unclipped torus offset, vs the beacon's broadcast
    s_ = jax.tree_util.tree_map(lambda x: x[0], S)
    bp = np.asarray(s_.pos[:NB]); cen = np.asarray(s_.centres); L = cfg.L
    d = (cen[None] - bp[:, None] + L // 2) % L - L // 2; i = np.abs(d).sum(-1).argmin(1); off = d[np.arange(NB), i]
    true_sec = np.mod(np.round(np.arctan2(off[:, 1], off[:, 0]) / (2 * np.pi / 8)).astype(int), 8)
    bc_sec = np.asarray(sector(env.oracle_offset(cfg, s_))[:NB])
    chk["beacon_offsets_unclipped"] = off.tolist(); chk["r_oracle"] = int(cfg.r_oracle)
    chk["broadcast_sector"] = bc_sec.tolist(); chk["true_direction_sector"] = true_sec.tolist()
    chk["broadcast_eq_true_direction"] = float((bc_sec == true_sec).mean())
    print("PIPELINE", json.dumps(chk), flush=True)

    tm = {"params": {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} for k, v in P0["params"].items()}}
    for k in tm["params"]:
        for kk in tm["params"][k]:
            tm["params"][k][kk] = jnp.ones_like(tm["params"][k][kk]) if k == "Dense_2" else tm["params"][k][kk]
    tm["params"]["Dense_0"]["kernel"] = jnp.zeros_like(tm["params"]["Dense_0"]["kernel"]).at[..., 100:, :].set(1.0)
    if a.probe_train == "all":
        tm = jax.tree_util.tree_map(jnp.ones_like, tm)
    tm1 = jax.tree_util.tree_map(lambda x: x[0, 0], tm)
    # target consistency: on heard code-driven states the scripted action must be the heading of the heard sector
    def heading(sec):
        ang = sec * (2 * np.pi / 8); dx = np.round(np.cos(ang) * 3); dy = np.round(np.sin(ang) * 3)
        return np.where(np.abs(dx) >= np.abs(dy), np.where(dx < 0, 1, 2), np.where(dy > 0, 3, 4))
    Ocd, Acd = Oh[cd_h], Ah[cd_h]
    chk["target_eq_heading_of_heard_sector_codedriven"] = float((heading(Ocd[:, 100:108].argmax(1)) == Acd).mean()) if len(Acd) else None
    chk["codedriven_action_hist"] = np.bincount(Acd, minlength=6).tolist() if len(Acd) else None
    # first-layer saturation of the navigator trunk on heard states
    p00 = jax.tree_util.tree_map(lambda x: x[0, 0], P0)
    pre = np.asarray(jnp.asarray(oh[:2000]) @ p00["params"]["Dense_0"]["kernel"] + p00["params"]["Dense_0"]["bias"])
    chk["dense0_tanh_saturated_frac(|tanh|>0.99)"] = float((np.abs(np.tanh(pre)) > 0.99).mean())
    chk["dense0_preact_abs_median"] = float(np.median(np.abs(pre)))
    print("PIPELINE2", json.dumps(chk), flush=True)

    def ce(p, o, y):
        lg, _, _ = net.apply(p, o)
        return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], 1)[:, 0].mean()

    def one(p, os_, o, y, f):
        g = jax.tree_util.tree_map(lambda x, m: x * m, jax.grad(ce)(p, o, y), tm1)
        u, os_ = opt.update(g, os_, p)
        return optax.apply_updates(p, jax.tree_util.tree_map(lambda x: x * f, u)), os_
    upd = jax.jit(jax.vmap(jax.vmap(one, in_axes=(0, 0, 0, 0, None)), in_axes=(0, 0, 0, 0, None)))
    g0 = jax.grad(ce)(jax.tree_util.tree_map(lambda x: x[0, 0], P0), jnp.asarray(oh[:128]), jnp.asarray(A[heard][:128]))
    chk_g = float(jnp.abs(g0["params"]["Dense_0"]["kernel"][100:]).max())
    print("PIPELINE grad into heard rows at step 0 (max |g|):", chk_g, flush=True)

    rng = np.random.default_rng(a.seed + 77)
    Opool, Apool = O[heard], A[heard]
    idx = [[np.where(heard[w, :, l])[0] for l in range(NL)] for w in range(W)]
    Ohh, Ahh = Oh[heard_h], Ah[heard_h]; CDm = code_driven(Ohh)
    sub = rng.choice(len(Ohh), min(4000, len(Ohh)), replace=False); Ohs, Ahs, CDs = jnp.asarray(Ohh[sub]), Ahh[sub], CDm[sub]
    acc_f = jax.jit(jax.vmap(jax.vmap(lambda p: net.apply(p, Ohs)[0].argmax(-1))))

    def accuracy(P):
        pr = np.asarray(acc_f(P)); hit = pr == Ahs[None, None]
        return float(hit.mean()), float(hit[..., CDs].mean()) if CDs.any() else float("nan")

    out = dict(checks=chk, grad_heard_rows_step0=chk_g, n_train_heard=int(heard.sum()), n_heldout_heard=int(heard_h.sum()),
               n_heldout_code_driven=int(cd_h.sum()))
    a0 = accuracy(P0); out["navigator_accuracy"] = dict(all=a0[0], code_driven=a0[1])
    print("navigator (untrained) held-out accuracy all / code-driven:", a0, flush=True)
    C = jnp.asarray(np.repeat(np.asarray(S.centres), NL, 0))
    keys = jax.vmap(lambda i: jax.random.split(jax.random.PRNGKey(7000 + i), a.eval_eps))(jnp.arange(W * NL))

    def gate(P):
        Pz = jax.tree_util.tree_map(lambda x: x.reshape((W * NL,) + x.shape[2:]), P)
        t = np.asarray(ev(Pz, C, keys, 0)); s2 = np.asarray(ev(Pz, C, keys, 1)); dd = t - s2
        se = dd.std(1, ddof=1) / np.sqrt(dd.shape[1]); drop = 1 - s2.sum(1) / np.maximum(t.sum(1), 1e-9)
        ok = (drop >= 0.30) & (dd.mean(1) > 2 * se)
        return dict(pass_frac=float(ok.mean()), median_drop=float(np.median(drop)), eaten_true=float(t.mean()), eaten_swap=float(s2.mean()))
    out["navigator_gate"] = gate(P0); print("navigator gate", json.dumps(out["navigator_gate"]), flush=True)
    out["runs"] = {}
    for fm in [float(x) for x in a.probe_lrs.split(",")]:
        P = P0; OS = jax.vmap(jax.vmap(opt.init))(P0); curve = []
        for st in range(a.probe_steps):
            if a.probe_pool:
                j = rng.choice(len(Opool), (W, NL, 128)); ob = jnp.asarray(Opool[j]); yb = jnp.asarray(Apool[j])
            else:
                b = np.stack([np.stack([rng.choice(idx[w][l], 128) if len(idx[w][l]) else np.zeros(128, int) for l in range(NL)]) for w in range(W)])
                ob = jnp.asarray(O[np.arange(W)[:, None, None], b, np.arange(NL)[None, :, None]])
                yb = jnp.asarray(A[np.arange(W)[:, None, None], b, np.arange(NL)[None, :, None]])
            P, OS = upd(P, OS, ob, yb, fm)
            if (st + 1) in (50, 100, 200, 300, 1000, 2000, a.probe_steps):
                curve.append((st + 1,) + accuracy(P))
        acc = accuracy(P); g = gate(P)
        # the heard input reaches the policy: flip the heard sector by 4 on held-out code-driven states
        o_sw = Ohs.at[:, 100:108].set(jnp.roll(Ohs[:, 100:108], 4, axis=1))
        l1 = np.asarray(jax.vmap(jax.vmap(lambda p: net.apply(p, Ohs)[0]))(P)); l2 = np.asarray(jax.vmap(jax.vmap(lambda p: net.apply(p, o_sw)[0]))(P))
        # coverage: which heard sectors did each learner see in training? accuracy on held-out code-driven states split
        # by whether the state's sector was among them
        seen = np.zeros((W, NL, 8), bool)
        for w in range(W):
            for l in range(NL):
                if a.probe_pool:
                    seen[w, l, np.unique(Opool[:, 100:108].argmax(1))] = True
                elif len(idx[w][l]):
                    seen[w, l, np.unique(O[w, idx[w][l], l, 100:108].argmax(1))] = True
        secs = np.asarray(Ohs[:, 100:108]).argmax(1)
        prd = np.asarray(acc_f(P)); hit = prd == Ahs[None, None]
        sm = seen[:, :, secs] & CDs[None, None]; um = (~seen[:, :, secs]) & CDs[None, None]
        cov = dict(distinct_sectors_heard_per_learner_median=float(np.median(seen.sum(-1))),
                   distinct_sectors_hist=np.bincount(seen.sum(-1).ravel(), minlength=9).tolist(),
                   acc_codedriven_seen_sector=float(hit[sm].mean()) if sm.any() else None,
                   acc_codedriven_unseen_sector=float(hit[um].mean()) if um.any() else None,
                   frac_codedriven_unseen=float(um.sum() / max(sm.sum() + um.sum(), 1)))
        # training-set accuracy (pooled mode: on a sample of the pooled training states; code-driven split)
        if a.probe_pool:
            jt = rng.choice(len(Opool), min(4000, len(Opool)), replace=False)
            Ot, At = jnp.asarray(Opool[jt]), Apool[jt]; CDt = code_driven(Opool[jt])
            pt = np.asarray(jax.vmap(jax.vmap(lambda p: net.apply(p, Ot)[0].argmax(-1)))(P)); ht = pt == At[None, None]
            cov["train_acc_all"] = float(ht.mean()); cov["train_acc_code_driven"] = float(ht[..., CDt].mean())
        r = dict(coverage=cov, lr=3e-4 * fm, steps=a.probe_steps, heldout_accuracy_all=acc[0], heldout_accuracy_code_driven=acc[1],
                 curve=curve, gate=g, gate_pass=g["pass_frac"] >= 0.5, accuracy_pass=acc[0] >= 0.9,
                 logit_change_on_sector_flip=float(np.abs(l1 - l2).mean()))
        out["runs"][f"x{fm:g}"] = r; print("PROBE", json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


def incontext_probe(a, IC, W, NL, T):
    """Nov 12 (EXPLORE; Cam work order iii, advisor's cheapest decisive test). Each life hears through its own random
    sector->token code. Can a GRU, trained ACROSS lives with direct supervision, work out a life's code from that
    life's own input stream? Input per step: the 100-dim view/body obs, the heard token (one-hot + flag + offset),
    the previous action, the reward. Target: the TRUE sector behind the heard token (on heard steps). Held-out lives
    are scored by the number of tokens heard so far in the life. Baseline: a memoryless MLP on the same inputs (it can
    use only what the current step shows)."""
    o = np.concatenate(IC["o"], 1); act = np.concatenate(IC["act"], 1); r = np.concatenate(IC["r"], 1)
    alive = np.concatenate(IC["alive"], 1)
    U = len(IC["codes"]); codes = np.repeat(np.stack(IC["codes"], 1), T, 1)          # (W, U*T, NL, 8)
    lid = np.repeat(np.stack(IC["lid"], 1), T, 1)                                      # (W, U*T, NL)
    seqs = []
    Lm = a.ic_len
    for w in range(W):
        for l in range(NL):
            for li in np.unique(lid[w, :, l]):
                m = (lid[w, :, l] == li) & alive[w, :, l]
                idx = np.where(m)[0][:Lm]
                if len(idx) < 100:
                    continue
                ob = o[w, idx, l]; heard = ob[:, 108] > 0.5
                tok = ob[:, 100:108].argmax(1); inv = np.argsort(codes[w, idx[0], l]); sec = inv[tok]
                pa = np.concatenate([[0], act[w, idx[:-1], l]]); pr = np.concatenate([[0.0], r[w, idx[:-1], l]])
                x = np.concatenate([ob[:, :111], np.eye(6)[pa], pr[:, None]], 1).astype(np.float32)
                X = np.zeros((Lm, x.shape[1]), np.float32); X[:len(idx)] = x
                Y = np.zeros(Lm, np.int64); Y[:len(idx)] = sec
                Mk = np.zeros(Lm, np.float32); Mk[:len(idx)] = heard
                K = np.zeros(Lm, np.int64); K[:len(idx)] = np.cumsum(heard)
                seqs.append((X, Y, Mk, K))
    rng = np.random.default_rng(a.seed + 9)
    order = rng.permutation(len(seqs)); nte = max(len(seqs) // 5, 1)
    te, trn = [seqs[i] for i in order[:nte]], [seqs[i] for i in order[nte:]]
    st = lambda S_, j: jnp.asarray(np.stack([x[j] for x in S_]))  # noqa: E731
    Xtr, Ytr, Mtr = st(trn, 0), st(trn, 1), st(trn, 2); Xte, Yte, Mte, Kte = st(te, 0), st(te, 1), st(te, 2), np.stack([x[3] for x in te])

    class GRUNet(nn.Module):
        @nn.compact
        def __call__(self, xs):                                  # xs (L, D)
            cell = nn.GRUCell(128); h = jnp.zeros((128,))
            enc = nn.Dense(128); out = nn.Dense(8)
            def body(h, x):
                h, _ = cell(h, nn.tanh(enc(x))); return h, out(h)
            _, lg = nn.scan(lambda m, c, x: body(c, x), variable_broadcast="params", split_rngs={"params": False})(self, h, xs)
            return lg

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, xs):
            return nn.Dense(8)(nn.tanh(nn.Dense(128)(nn.tanh(nn.Dense(128)(xs)))))

    out = dict(lives_train=len(trn), lives_test=len(te), heard_steps_test=int(np.asarray(Mte).sum()),
               chance_uniform=0.125, chance_majority=float(np.bincount(np.asarray(Yte)[np.asarray(Mte) > 0], minlength=8).max() / max(np.asarray(Mte).sum(), 1)))
    for name, M_ in (("gru", GRUNet()), ("memoryless", MLP())):
        p = M_.init(jax.random.PRNGKey(0), Xtr[0]); opt = optax.adam(1e-3); os_ = opt.init(p)
        def lossf(p, X, Y, Mk):
            lg = jax.vmap(lambda x: M_.apply(p, x))(X)
            ce = -jnp.take_along_axis(jax.nn.log_softmax(lg), Y[..., None], -1)[..., 0]
            return (ce * Mk).sum() / (Mk.sum() + 1e-8)
        @jax.jit
        def upd(p, os_, X, Y, Mk):
            l, g = jax.value_and_grad(lossf)(p, X, Y, Mk); u, os_ = opt.update(g, os_, p); return optax.apply_updates(p, u), os_, l
        for s_ in range(a.ic_steps):
            bi = rng.integers(0, Xtr.shape[0], 32)
            p, os_, l = upd(p, os_, Xtr[bi], Ytr[bi], Mtr[bi])
        pred = np.asarray(jax.jit(jax.vmap(lambda x: M_.apply(p, x)))(Xte)).argmax(-1)
        hit = (pred == np.asarray(Yte)); mk = np.asarray(Mte) > 0
        bins = [(1, 5), (6, 15), (16, 40), (41, 100), (101, 10_000)]
        out[name] = dict(final_train_loss=float(l), acc_all=float(hit[mk].mean()),
                         by_tokens_heard={f"{lo}-{hi}": dict(n=int((mk & (Kte >= lo) & (Kte <= hi)).sum()),
                                                            acc=float(hit[mk & (Kte >= lo) & (Kte <= hi)].mean()) if (mk & (Kte >= lo) & (Kte <= hi)).any() else None)
                                          for lo, hi in bins})
    print("ICPROBE", json.dumps(out), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


def ideal_observer(a, IC, W, NL, NE, T, cfg):
    """Cam Nov 12 (EXPLORE): is the evidence for a life's code in its stream at all, with no learning? For each token a
    life hears, accumulate votes for sectors from (a) food visible in view when it is heard (direction to the best
    cell), (b) hindsight: the displacement from where it was heard to the life's next meal (available only once that
    meal happens), (c) demonstrators within view (Chebyshev <= view radius) and their next move direction. Online:
    each heard token is decoded as the argmax of the votes available BEFORE that moment (ties and no votes ->
    random), and scored against the true sector. Accuracy vs tokens heard, per evidence set."""
    o = np.concatenate(IC["o"], 1); alive = np.concatenate(IC["alive"], 1); take = np.concatenate(IC["take"], 1)
    pos = np.concatenate(IC["pos"], 1); ea = np.concatenate(IC["ea"], 1) if NE else None
    codes = np.repeat(np.stack(IC["codes"], 1), T, 1); lid = np.repeat(np.stack(IC["lid"], 1), T, 1)
    L = cfg.L; r_ = cfg.r; o_ = np.arange(-r_, r_ + 1)
    def sec_of(dx, dy):
        return int(np.mod(np.round(np.arctan2(dy, dx) / (2 * np.pi / 8)), 8))
    MOVE = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
    def _head(sc_):                                       # the move a scripted follower makes for sector sc_
        dx, dy = np.round(np.cos(sc_ * np.pi / 4) * 3), np.round(np.sin(sc_ * np.pi / 4) * 3)
        return (1 if dx < 0 else 2) if abs(dx) >= abs(dy) else (3 if dy > 0 else 4)
    HEAD = np.array([_head(k) for k in range(8)])
    sets = {"oracle_control": "o", "a_visible_food": "a", "b_hindsight_meal": "b", "a+b": "ab",
            "A_visible_food_from_beacon": "A", "B_hindsight_meal_from_beacon": "B", "A+B_from_beacon": "AB"}
    if NE:
        sets.update({"c_demonstrators": "c", "a+b+c": "abc", "A+B+c": "ABc"})
    rng = np.random.default_rng(a.seed + 21)
    rec = {k: [] for k in sets}; heard_per_life = []; vis = []
    for w in range(W):
        for l in range(NL):
            for li in np.unique(lid[w, :, l]):
                idx = np.where((lid[w, :, l] == li) & alive[w, :, l])[0]
                if len(idx) < 100:
                    continue
                ob = o[w, idx, l]; hd = ob[:, 108] > 0.5; tok = ob[:, 100:108].argmax(1)
                inv = np.argsort(codes[w, idx[0], l]); tsec = inv[tok]
                p = pos[w, idx, l]; eat = take[w, idx, l] > 0
                heard_per_life.append(int(hd.sum()))
                # evidence events: (time available, token, sector, source)
                ev = []
                meal_t = np.where(eat)[0]
                for i in np.where(hd)[0]:
                    ev.append((i + 1, tok[i], int(tsec[i]), "o"))     # positive control: the true sector, after the fact
                    wv = ob[i, :(2 * r_ + 1) ** 2]
                    dj = ob[i, 109:111] * R_HEAR                   # the speaker's offset (beacon - listener), an input
                    if wv.max() > 0.2:
                        b = int(wv.argmax()); fx, fy = o_[b // (2 * r_ + 1)], o_[b % (2 * r_ + 1)]
                        ev.append((i, tok[i], sec_of(fx, fy), "a"))
                        if abs(fx - dj[0]) + abs(fy - dj[1]) > 0:
                            ev.append((i, tok[i], sec_of(fx - dj[0], fy - dj[1]), "A"))
                    nm = meal_t[meal_t > i]
                    if len(nm):
                        j = nm[0]; d = (p[j] - p[i] + L // 2) % L - L // 2
                        if np.abs(d).sum() > 0:
                            ev.append((j, tok[i], sec_of(d[0], d[1]), "b"))
                        db_ = d - dj
                        if np.abs(db_).sum() > 0:
                            ev.append((j, tok[i], sec_of(db_[0], db_[1]), "B"))
                    if NE:
                        pe = pos[w, idx[i], NL:]; dd = (pe - p[i] + L // 2) % L - L // 2
                        near = np.abs(dd).max(1) <= r_; vis.append(bool(near.any()))
                        for e in np.where(near)[0]:
                            m = int(ea[w, idx[i], e])
                            if m in MOVE:
                                ev.append((i + 1, tok[i], sec_of(*MOVE[m]), "c"))
                ev.sort(key=lambda x: x[0])
                for name, srcs in sets.items():
                    V = np.zeros((8, 8)); k = 0; e_i = 0
                    for i in np.where(hd)[0]:
                        while e_i < len(ev) and ev[e_i][0] < i:      # only evidence available before this moment
                            if ev[e_i][3] in srcs:
                                V[ev[e_i][1], ev[e_i][2]] += 1
                            e_i += 1
                        k += 1; v = V[tok[i]]
                        guess = rng.choice(np.flatnonzero(v == v.max()))
                        rec[name].append((k, guess == tsec[i], ob[i, 99] * cfg.A_max, HEAD[guess] == HEAD[tsec[i]]))
    nmed = float(np.median(heard_per_life))
    bins = [(1, 5), (6, 15), (16, 40), (41, 100), (101, 300), (301, 100_000)]
    out = dict(lives=len(heard_per_life), median_tokens_heard_per_life=nmed, demonstrator_in_view_frac=float(np.mean(vis)) if vis else None,
               chance=0.125, sets={})
    for name in sets:
        R = np.array(rec[name], float); Kk, H, Ag, HE = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
        late = (Kk >= nmed / 2) & (Kk <= nmed)
        agew = {"age_200-250": (200, 250), "age_450-500": (450, 500), f"after_age_{a.juvenile}": (a.juvenile, 1e9)}
        out["sets"].setdefault(name, {})["by_age"] = {k_: (round(float(H[(Ag > lo) & (Ag <= hi)].mean()), 4) if ((Ag > lo) & (Ag <= hi)).any() else None) for k_, (lo, hi) in agew.items()}
        out["sets"][name]["heading_equivalent"] = dict(acc_all=float(HE.mean()), acc_late_median_life=float(HE[late].mean()) if late.any() else None,
            by_age={k_: (round(float(HE[(Ag > lo) & (Ag <= hi)].mean()), 4) if ((Ag > lo) & (Ag <= hi)).any() else None) for k_, (lo, hi) in agew.items()})
        out["sets"][name].update(dict(acc_all=float(H.mean()), acc_late_median_life=float(H[late].mean()) if late.any() else None,
                                 by_tokens_heard={f"{lo}-{hi}": round(float(H[(Kk >= lo) & (Kk <= hi)].mean()), 4) if ((Kk >= lo) & (Kk <= hi)).any() else None for lo, hi in bins}))
    print("IDEALOBS", json.dumps(out), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


def observer_decs(a, IC, W, NL, NE, T, cfg, srcs="ABc"):
    """per-life token -> sector table at the END of each life: the argmax of all counting-decoder evidence from the
    given sources (A: visible food from the beacon, B: next meal from the beacon, c: demonstrators/parent moves) over
    the life; tokens with no evidence -> a random sector. Keys = (world, learner, life id)."""
    o = np.concatenate(IC["o"], 1); alive = np.concatenate(IC["alive"], 1); take = np.concatenate(IC["take"], 1)
    pos = np.concatenate(IC["pos"], 1); ea = np.concatenate(IC["ea"], 1) if NE else None
    lid = np.repeat(np.stack(IC["lid"], 1), T, 1)
    L = cfg.L; r_ = cfg.r; o_ = np.arange(-r_, r_ + 1); rng = np.random.default_rng(a.seed + 31)
    sec_of = lambda dx, dy: int(np.mod(np.round(np.arctan2(dy, dx) / (2 * np.pi / 8)), 8))  # noqa: E731
    MOVE = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
    out = {}
    for w in range(W):
        for l in range(NL):
            for li in np.unique(lid[w, :, l]):
                idx = np.where((lid[w, :, l] == li) & alive[w, :, l])[0]
                if len(idx) == 0:
                    continue
                ob = o[w, idx, l]; hd = ob[:, 108] > 0.5; tok = ob[:, 100:108].argmax(1)
                p = pos[w, idx, l]; meal_t = np.where(take[w, idx, l] > 0)[0]; V = np.zeros((8, 8))
                for i in np.where(hd)[0]:
                    wv = ob[i, :(2 * r_ + 1) ** 2]; dj = ob[i, 109:111] * R_HEAR
                    if "A" in srcs and wv.max() > 0.2:
                        b = int(wv.argmax()); fx, fy = o_[b // (2 * r_ + 1)], o_[b % (2 * r_ + 1)]
                        if abs(fx - dj[0]) + abs(fy - dj[1]) > 0:
                            V[tok[i], sec_of(fx - dj[0], fy - dj[1])] += 1
                    nm = meal_t[meal_t > i]
                    if "B" in srcs and len(nm):
                        d = (p[nm[0]] - p[i] + L // 2) % L - L // 2 - dj
                        if np.abs(d).sum() > 0:
                            V[tok[i], sec_of(d[0], d[1])] += 1
                    if "c" in srcs and NE:
                        pe = pos[w, idx[i], NL:]; dd = (pe - p[i] + L // 2) % L - L // 2
                        for e in np.where(np.abs(dd).max(1) <= r_)[0]:
                            m = int(ea[w, idx[i], e])
                            if m in MOVE:
                                V[tok[i], sec_of(*MOVE[m])] += 1
                out[(int(w), int(l), int(li))] = np.array([rng.choice(np.flatnonzero(V[t] == V[t].max())) for t in range(8)])
    return out


if __name__ == "__main__":
    main()
