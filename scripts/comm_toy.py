"""CPU toy of the comm testbed (Cam, 2026-11-07): reproduce the live-wire token collapse cheaply, then test
positive signalling / positive listening (Eccles et al. 2019) before any GPU money.

KEPT IDENTICAL to scripts/comm_testbed.py (ceadaf8): the world rules (four production hearths via
generate_hearth_positions, sender = informed / listener = uninformed, one active hearth, Craft within radius 1 of it
deposits 2.7, depositor respawns away from hearths, active re-rolls on ANY deposit in the world or after REROLL
steps, listener deposit pays the sender 2.7, cost per non-deposit Craft), the symmetric one-step-delayed wire
(each agent's broadcast = its sampled tokens' codebook vectors, heard by its partner), the token policy
(rl_jax.token_logits, tau 0.08, log-prob in the PPO ratio, entropy coef 0.01), the VQ loss
(network_jax.vector_quantize_signals, beta 0.25, vq_coef 0.1), rl_jax.ppo_loss itself, C' (learner_testbed.
gae_bootstrap: GAE 0.95 + bootstrap, gamma 0.999), PopArt (rl_jax.PopArt, beta 0.1), vf 0.5, ent 0.02, Adam 3e-4,
the comms pathway (separate feed-forward branch from the obs, as network_jax), the /2 action-logit temperature,
the per-group (head_action / head_value / trunk) gradient clip at 2.0, minibatch 512, one epoch, the paired
token-swap eval and its apparatus check, the hearth_rel formula, the 4 noise dims.

CHANGED (the toy): G = 12 (hearths at 3/9), REROLL = 96 (256 x 12/32, rounded), T = 128 rollout steps, a small
MLP + GRU (64) instead of AgentNetworkJax, a 59-dim obs (pos, hearth_rel, informed flag, need one-hot, 4 noise
dims, 40-dim wire) instead of the production layout, no alarm head, the whole rollout jitted with lax.scan.

Arms: ORACLE (wire zeroed, both agents informed) | NOSIG (wire zeroed) | COMM (+ --ps / --pl aux losses).
  --ps C   positive signalling: + C * sum_slots MI(token; active hearth) on SENDER rows (batch estimate,
           H(mean p) - sum_T w_T H(mean p | T); maximised).
  --pl C   positive listening: + C * mean L1(pi(a | wire), stopgrad pi(a | wire = 0)) on LISTENER rows (maximised).
"""
from __future__ import annotations

import argparse
from collections import deque
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import flax.linen as nn  # noqa: E402
import flax.serialization  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from jax_sim import rl_jax  # noqa: E402
from jax_sim.action_space import mask_disabled_actions  # noqa: E402
from jax_sim.grid_jax import generate_hearth_positions  # noqa: E402
from jax_sim.network_jax import vector_quantize_signals  # noqa: E402
from jax_sim.rl_jax import PopArt, ppo_loss, token_logits  # noqa: E402
from learner_testbed import ACTION_CRAFT, DELTAS, gae_bootstrap  # noqa: E402

R_DEP = 2.7
N_ACT, V, SIG = 12, 64, 40
GAMMA, LAM, LR, CLIP, ENT, VQC, VF, MB = 0.999, 0.95, 3e-4, 0.2, 0.02, 0.1, 0.5, 512
TAU, TOK_ENT = 0.08, 0.01
# obs layout (toy)
O_POS, O_HR, O_INF, O_NEED, O_NOISE, O_WIRE, D = 0, 2, 10, 11, 15, 19, 59


class Outs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ToyNet(nn.Module):
    hidden: int = 64
    credit: str = "joint"     # "A": + V_shared(s) head; "B": + per-slot message-conditioned critic q_i(s, m_i)
    wire_blind: int = 0       # C1: 1 = the message pathway's input has the heard wire zeroed (no path from the wire to the
                              # tokens: the pathway is feed-forward from the obs and shares nothing with the trunk);
                              # 2 = MUTANT for the apparatus check (one wire row left live)

    @nn.compact
    def __call__(self, carries, obs):
        x = nn.relu(nn.Dense(self.hidden, name="enc_0")(obs))
        x = nn.relu(nn.Dense(self.hidden, name="enc_1")(x))
        new_c, h = nn.GRUCell(self.hidden, name="gru")(carries, x)
        logits = nn.Dense(N_ACT, name="head_action")(h) / 2.0       # production temperature 2.0
        v = nn.Dense(1, name="head_value")(h)[..., 0]
        # Comms pathway as in production (network_jax: gwt_comms_1 -> head_signal_slot{0,1,2}): a separate
        # feed-forward branch from the CURRENT obs, independent of the recurrent trunk; z_e[:, :8] = 0.
        obs_c = obs
        if self.wire_blind == 1:
            obs_c = obs.at[:, O_WIRE:].set(0.0)
        elif self.wire_blind == 2:   # mutant: every wire dim zeroed except one (the first codebook dim)
            obs_c = obs.at[:, O_WIRE:].set(0.0).at[:, O_WIRE + 8].set(obs[:, O_WIRE + 8])
        h_c = nn.relu(nn.Dense(self.hidden, name="gwt_comms_1")(obs_c))
        zs, lvq = [jnp.zeros((obs.shape[0], 8))], []
        for i, (lo, hi, k) in enumerate(rl_jax.TOKEN_SLOTS):
            z = nn.Dense(hi - lo, name=f"head_signal_slot{i}")(h_c)
            cb = nn.Embed(V, hi - lo, name=k).embedding
            _, _, l = vector_quantize_signals(z, cb, beta=0.25)
            zs.append(z); lvq.append(l)
        z_e = jnp.concatenate(zs, -1)
        vsh = nn.Dense(1, name="head_value_shared")(h)[..., 0] if self.credit == "A" else None
        qs = [nn.Dense(V, name=f"head_q_slot{i}")(h) for i in range(3)] if self.credit == "B" else None
        return new_c, Outs(action_logits=logits, values=v, z_e=z_e, loss_vq=jnp.stack(lvq, -1), alarm_out=None,
                           values_shared=vsh, q_slots=qs)


class CentralCritic(nn.Module):
    """T4 (Cam 2026-11-08): MAPPO-style centralized critic for message credit, TRAINING TIME ONLY. Input = global
    features of a world at a step: sender obs, listener obs, listener GRU state, active-fire one-hot (positions are
    inside both obs). mode "V": V_shared(g); mode "Q": per-slot q_i(g, m_i) over the 64 codes."""
    mode: str = "V"

    @nn.compact
    def __call__(self, g):
        x = nn.relu(nn.Dense(128, name="c0")(g))
        x = nn.relu(nn.Dense(128, name="c1")(x))
        if self.mode == "V":
            return nn.Dense(1, name="out")(x)[..., 0]
        return [nn.Dense(V, name=f"q{i}")(x) for i in range(3)]


def main():
    global VQC
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["COMM", "NOSIG", "ORACLE"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--reroll", type=int, default=96)
    ap.add_argument("--craft-cost", type=float, default=0.45)
    ap.add_argument("--ps", type=float, default=0.0)
    ap.add_argument("--pl", type=float, default=0.0)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--save-params", default=None)
    ap.add_argument("--swap-pairs", type=int, default=10)
    ap.add_argument("--ps-anneal", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="c_ps decays linearly from --ps to 0 over updates START..END, then stays 0")
    ap.add_argument("--eval-at", type=int, nargs="*", default=[], help="extra eval (MI + swaps) after these updates")
    ap.add_argument("--erode-tok-ent", type=float, default=None, help="erosion: token entropy coef from --erode-from on")
    ap.add_argument("--erode-from", type=int, default=1500)
    ap.add_argument("--reset-from", type=int, default=None, help="T2/T3 turnover: first reset update")
    ap.add_argument("--reset-every", type=int, default=250)
    ap.add_argument("--reset-until", type=int, default=2750)
    ap.add_argument("--imitate", action="store_true", help="T3: a reset sender head first imitates the previous head")
    ap.add_argument("--imitate-steps", type=int, default=500)
    ap.add_argument("--imitate-lr", type=float, default=1e-3)
    ap.add_argument("--imitate-window", type=int, default=50)
    ap.add_argument("--wire-blind", type=int, default=0, help="C1: 1 = wire-blind message head from the start")
    ap.add_argument("--comp", action="store_true",
                    help="step 6: the target is a (fire, material) pair (16 meanings); material one-hot in the sender's 4 noise "
                         "dims; 4 Craft-material actions (5, 9, 10, 11); a delivery needs the active fire AND the material")
    ap.add_argument("--held-out", default="1,11", help="step 6: pair ids (fire*4 + material) never active in training")
    ap.add_argument("--vq-vocab", type=int, default=0, help="step 6: VQ codes per slot reachable (0 = all 64)")
    ap.add_argument("--selftest-comp", action="store_true")
    ap.add_argument("--mutant-leak-heldout", action="store_true", help="TEST ONLY: training pool includes held-out pairs")
    ap.add_argument("--channel", default="vq", choices=["vq", "sym"],
                    help="step 2: sym = 3 discrete symbols from a vocabulary of --sym-vocab, broadcast one-hot (fixed orthogonal "
                         "codewords a*e_k, the rest unreachable; no VQ loss)")
    ap.add_argument("--sym-vocab", type=int, default=8)
    ap.add_argument("--sym-scale", type=float, default=0.3)
    ap.add_argument("--selftest-sym", action="store_true")
    ap.add_argument("--prox-world-check", default=None, help="step 3 free check: saved actor params path; print and exit")
    ap.add_argument("--selftest-wireblind", action="store_true", help="C1 apparatus: token TV under random heard input")
    ap.add_argument("--snapshot-at", type=int, default=None, help="step B: pickle the full training state after this update")
    ap.add_argument("--snapshot-path", default=None)
    ap.add_argument("--resume", default=None, help="step B: resume from a full-state snapshot")
    ap.add_argument("--branch", default="none", choices=["none", "frozen_listener", "stopgrad", "wireblind"],
                    help="step B arm, active after the resume point")
    ap.add_argument("--prox", type=int, default=None,
                    help="step 3: no assigned sender -- an agent sees the active fire only within this Chebyshev radius of it; "
                         "symmetric shared reward; PS on whoever is informed")
    ap.add_argument("--hearths", default=None, help="step 3b: custom fire positions 'x,y;x,y;x,y;x,y' (default: the generator's lattice)")
    ap.add_argument("--lookout", default=None,
                    help="step 3b: 'x,y,r' -- an agent sees the lit fire only while within Chebyshev r of (x,y); symmetric roles and "
                         "shared reward as step 3")
    ap.add_argument("--lookout-oracle", action="store_true",
                    help="step 3b VOI (Cam 2026-11-08): the non-lookout agent is also fed the lit fire while its partner stands at the lookout")
    ap.add_argument("--mutant-lookout-everywhere", action="store_true", help="apparatus mutant: everyone counts as at the lookout")
    ap.add_argument("--selftest-lookout", action="store_true", help="step 3b apparatus: geometry + need dims only at the lookout")
    ap.add_argument("--selftest-leak", action="store_true", help="Step A (Cam 2026-11-08): wire -> message-parameter gradient paths")
    ap.add_argument("--save-at", type=int, nargs="*", default=[], help="save actor (+ central critic) params at these updates")
    ap.add_argument("--save-prefix", default=None)
    ap.add_argument("--qcheck", default=None, help="T4 check 3 (b)/(c): path prefix of saved params at one update; print and exit")
    ap.add_argument("--credit", default="joint", choices=["joint", "A", "B", "Ag", "Bg"],
                    help="token credit: joint (baseline) | A separate V_shared stream | B COMA-style Q_shared(s, m)")
    ap.add_argument("--selftest-credit", action="store_true")
    ap.add_argument("--diag-from", type=int, default=None, help="erosion diagnostic (Cam 2026-11-08): log from this update")
    ap.add_argument("--diag-to", type=int, default=None)
    ap.add_argument("--diag-every", type=int, default=10, help="reliance probe every N updates in the window")
    ap.add_argument("--diag-pairs", type=int, default=4)
    ap.add_argument("--erode-wd", type=float, default=None,
                    help="erosion: decoupled weight decay on the message head (gwt_comms_1, head_signal_slot*) per minibatch step from --erode-from on")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--speak-test", action="store_true",
                    help="north star (Cam, 2026-11-08): build a dictionary (fire -> modal sender triple) from frozen control "
                         "rollouts, then 'speak to them': replace the sender's broadcast by the dictionary word for fire "
                         "X = T + s and count listener Crafts at X, vs the control and vs the sender's own words for X")
    ap.add_argument("--msg-grad", default="all", choices=["all", "sender"],
                    help="Cam Nov 9: 'sender' = from-scratch stop-grad: the message head trains on sender rows only (PS unscaled)")
    ap.add_argument("--selftest-sg2", action="store_true", help="apparatus check of --msg-grad sender at update 1 (Cam Nov 9)")
    ap.add_argument("--mi-null", type=int, default=0, help="with --same-spot: joint-MI shuffle null (N label permutations); skips the swaps")
    ap.add_argument("--same-spot", action="store_true", help="Cam Nov 9: same-spot vs cross-spot empirical swap + joint / pre-Craft MI")
    ap.add_argument("--spawn-box", default=None, help="x0,y0,x1,y1 inclusive: start / respawn cells restricted to this box")
    ap.add_argument("--export-m", default=None, help="for M: write dictionary + one episode transcript (CSV/JSON) to this dir")
    ap.add_argument("--grad-share", action="store_true",
                    help="Cam Nov 9 item 4: listener vs sender rows' share of the message-head gradient at the loaded params")
    ap.add_argument("--grad-share-u", type=int, default=2000, help="the update the loaded params come from (sets c_ps / tok-ent)")
    ap.add_argument("--grad-share-rollouts", type=int, default=4)
    ap.add_argument("--grad-share-snap", default=None, help="full-state snapshot: exact params + PopArt stats")
    ap.add_argument("--grad-share-r0", type=int, default=0, help="first rollout index (split-half noise check)")
    ap.add_argument("--grad-share-save", default=None, help="save the MSGK vectors c_S, c_L, g_all (.npz)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--wire-init-scale", type=float, default=1.0,
                    help="after warm start, scale the (never-trained) wire-input rows of enc_0 and gwt_comms_1 (fidelity calibration)")
    ap.add_argument("--measure-sens", action="store_true", help="print listener sensitivity at the loaded params and exit")
    a = ap.parse_args()

    G, W, T, H = a.grid, a.worlds, a.steps, 64
    N = 2 * W
    HEARTHS = jnp.asarray(np.asarray(generate_hearth_positions(G)) if a.hearths is None else
                          np.array([[int(v) for v in q.split(",")] for q in a.hearths.split(";")], np.int32))
    LOOK = None if a.lookout is None else [int(v) for v in a.lookout.split(",")]
    SYMR = a.prox is not None or LOOK is not None        # symmetric roles (steps 3 / 3b)
    is_listener = jnp.tile(jnp.array([False, True]), W)
    live_wire = a.arm == "COMM"
    oracle = a.arm == "ORACLE"
    informed = jnp.ones(N, bool) if oracle else ~is_listener
    COMP = a.comp
    NCLS = 16 if COMP else 4
    CRAFT_ACTS = jnp.asarray([5, 9, 10, 11])
    HELD = [int(x) for x in a.held_out.split(",")] if COMP else []
    TRAIN_POOL = jnp.asarray([q for q in range(16) if (q not in HELD or a.mutant_leak_heldout)]) if COMP else None
    fire_of = (lambda act_: act_ // 4) if COMP else (lambda act_: act_)
    DEL = jnp.asarray(DELTAS)
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    hn = np.asarray(HEARTHS)
    dd = np.abs(cells[:, None, :] - hn[None]); dd = np.minimum(dd, G - dd).max(-1)
    VALID = jnp.asarray(cells[(dd > 1).all(1)])           # respawn cells: not within radius 1 of any hearth
    if a.spawn_box is not None:   # retuned 3b world (Cam, Nov 9): start and respawn only inside the fire cluster's region
        bx0, by0, bx1, by1 = (int(v) for v in a.spawn_box.split(","))
        vv = np.asarray(VALID)
        VALID = jnp.asarray(vv[(vv[:, 0] >= bx0) & (vv[:, 0] <= bx1) & (vv[:, 1] >= by0) & (vv[:, 1] <= by1)])

    model = ToyNet(H, credit=a.credit, wire_blind=a.wire_blind)

    def apply_fn(params, carries, obs, n_layers=1, detach_value=False, **kw):
        return model.apply(params, carries, obs)

    key = jax.random.PRNGKey(a.seed)
    key, k_init = jax.random.split(key)
    params = model.init(k_init, jnp.zeros((1, H)), jnp.zeros((1, D)))
    if a.warm_start:
        if a.credit == "joint":
            params = flax.serialization.from_bytes(params, Path(a.warm_start).read_bytes())
        else:   # the warm start has no credit heads: load the shared structure, keep the new heads at init
            base = ToyNet(H).init(k_init, jnp.zeros((1, H)), jnp.zeros((1, D)))
            base = flax.serialization.from_bytes(base, Path(a.warm_start).read_bytes())
            pp = dict(params["params"]); pp.update(base["params"]); params = {"params": pp}
        if a.wire_init_scale != 1.0:
            # The wire was zero throughout ORACLE training, so these rows are still at init (verified bit-identical).
            # Rescaling them sets the listener's start sensitivity to the real network's without scaling the input
            # (an input gain would also scale the learning signal under Adam).
            pp = {k: dict(v) for k, v in params["params"].items()}
            for n in ("enc_0", "gwt_comms_1"):
                pp[n]["kernel"] = jnp.asarray(pp[n]["kernel"]).at[O_WIRE:].multiply(a.wire_init_scale)
            params = {"params": pp}
    if a.channel == "sym":
        # Symbols as actions with a codebook-distance policy over FIXED orthogonal codewords c_k = a e_k (k < vocab):
        # -||z - c_k||^2 / tau = (2a/tau) z_k + const, i.e. a linear symbol head with temperature tau / (2a); the
        # broadcast is a * one-hot. Rows >= vocab are pushed far away (never chosen). No VQ loss, so the codebooks
        # get no gradient and stay fixed.
        VQC = 0.0
        pp = {k: dict(v) if isinstance(v, dict) else v for k, v in params["params"].items()}
        for lo, hi, ck in rl_jax.TOKEN_SLOTS:
            d = hi - lo
            E = np.full((V, d), 1e3 / np.sqrt(d), np.float32)
            E[:a.sym_vocab] = a.sym_scale * np.eye(a.sym_vocab, d, dtype=np.float32)
            pp[ck] = dict(pp[ck], embedding=jnp.asarray(E))
        params = {"params": pp}
    if a.vq_vocab and a.channel == "vq":
        # step 6: only the first vq_vocab codewords per slot are reachable (the rest pushed unreachably far); the
        # reachable ones stay learned VQ codewords
        pp = {k: dict(v) if isinstance(v, dict) else v for k, v in params["params"].items()}
        for lo, hi, ck in rl_jax.TOKEN_SLOTS:
            E = np.array(pp[ck]["embedding"], np.float32)
            E[a.vq_vocab:] = 1e3 / np.sqrt(hi - lo)
            pp[ck] = dict(pp[ck], embedding=jnp.asarray(E))
        params = {"params": pp}
    opt = optax.adam(LR)
    opt_state = opt.init(params)
    popart = PopArt(beta=0.1)
    popart_s = PopArt(beta=0.1, head_key="head_value_shared")   # credit A
    Q_SCALE = R_DEP                                                 # credit B: q targets in units of one delivery
    CENTRAL = a.credit in ("Ag", "Bg")
    if CENTRAL:                    # T4: centralized critic, separate params and optimizer, training time only
        DG = D + D + H + 4
        ccrit = CentralCritic("V" if a.credit == "Ag" else "Q")
        cparams = ccrit.init(jax.random.PRNGKey(9_999 + a.seed), jnp.zeros((1, DG)))
        copt = optax.adam(LR); cstate = copt.init(cparams)
        popart_g = PopArt(beta=0.1, head_key="out")

    # ---------------------------------------------------------------- env (jnp)
    def near(pos):                                                     # (n, 4) Chebyshev radius 1, torus
        d = jnp.abs(pos[:, None, :] - HEARTHS[None]); d = jnp.minimum(d, G - d).max(-1)
        return d <= 1

    def at_look(pos):
        dl = jnp.abs(pos - jnp.asarray(LOOK[:2])[None]); dl = jnp.minimum(dl, G - dl).max(-1)
        return jnp.ones(pos.shape[0], bool) if a.mutant_lookout_everywhere else dl <= LOOK[2]

    def build_obs(pos, active, wire, k):
        diff = (HEARTHS[None] - pos[:, None, :] + G / 2) % G - G / 2
        hr = (diff / (G / 2.0)).reshape(N, 8)
        if LOOK is not None:       # step 3b: knowledge from standing at the lookout (or, VOI oracle, the partner standing there)
            at = at_look(pos)
            inf_ = (at | at.reshape(W, 2)[:, ::-1].reshape(N)) if a.lookout_oracle else at
        elif a.prox is not None:     # step 3: knowledge from proximity to the active fire
            dh = jnp.abs(HEARTHS[jnp.repeat(active, 2)] - pos); dh = jnp.minimum(dh, G - dh).max(-1)
            inf_ = dh <= a.prox
        else:
            inf_ = informed
        need = jax.nn.one_hot(jnp.repeat(fire_of(active), 2), 4) * inf_[:, None]
        heard = wire.reshape(W, 2, -1)[:, ::-1].reshape(N, -1) if live_wire else jnp.zeros((N, SIG))
        noise = (jax.nn.one_hot(jnp.repeat(active % 4, 2), 4) * inf_[:, None]) if COMP else jax.random.normal(k, (N, 4))
        return jnp.concatenate([pos / G, hr, inf_[:, None].astype(jnp.float32), need, noise, heard], -1)

    def policy(params, carries, obs, k):
        new_c, o = apply_fn(params, carries, obs)
        logits = mask_disabled_actions(o.action_logits, axis=-1)
        k1, k3 = jax.random.split(k)
        act = jax.random.categorical(k1, logits)
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), act[:, None], -1)[:, 0]
        p = params["params"]
        toks, bc, ent = [], [jnp.zeros((N, 8))], []
        for i, lg in enumerate(token_logits(params, o.z_e, TAU)):
            t = jax.random.categorical(jax.random.fold_in(k3, i), lg)
            lsm = jax.nn.log_softmax(lg, -1)
            lp = lp + jnp.take_along_axis(lsm, t[:, None], -1)[:, 0]
            ent.append(-jnp.sum(jnp.exp(lsm) * lsm, -1))
            toks.append(t); bc.append(p[rl_jax.TOKEN_SLOTS[i][2]]["embedding"][t])
        return (new_c, logits, act, lp, o.values, o.loss_vq.sum(-1), jnp.stack(toks, -1),
                jnp.concatenate(bc, -1), jnp.stack(ent, -1).sum(-1), o)

    def tv(l1, l2):
        p1, p2 = jax.nn.softmax(l1, -1), jax.nn.softmax(l2, -1)
        return 0.5 * jnp.abs(p1 - p2).sum(-1)

    branch_hold = {"pL": None}

    @functools.partial(jax.jit, static_argnames=("swap", "shift"))
    def rollout(params, state, key, swap=False, table=None, pool=None, shift=1):
        """lax.scan over T steps. swap=True (eval): the listener hears what its sender WOULD broadcast for
        T' = (T + 1) % 4 (same pre-step carry, same key); the sender's own action/reward come from the true T."""
        cb = [params["params"][k]["embedding"] for _, _, k in rl_jax.TOKEN_SLOTS]

        def step(st, k):
            pos, active, since, wire, carries = st
            k_obs, k_act, k_rs, k_rr, k_rand = jax.random.split(k, 5)
            obs = build_obs(pos, active, wire, k_obs)
            new_c, logits, act, lp, val, lvq, toks, bc, ent, o_pol = policy(params, carries, obs, k_act)
            if branch_hold["pL"] is not None:   # step B (ii): the listener acts with frozen params
                outL = policy(branch_hold["pL"], carries, obs, k_act)
                selL = lambda x, y: jnp.where(is_listener.reshape((N,) + (1,) * (x.ndim - 1)), y, x)  # noqa: E731
                new_c, logits, act, lp, val, lvq, toks, bc, ent = [selL(x, y) for x, y in
                                                                   zip((new_c, logits, act, lp, val, lvq, toks, bc, ent), outL[:9])]
            # credit A/B only: on the joint (default) path nothing extra is computed, so its compiled graph -- and
            # its results, bit for bit -- stay as before (an unconditional extra log_softmax changed them at 1e-7)
            lp_act = (jnp.take_along_axis(jax.nn.log_softmax(logits, -1), act[:, None], -1)[:, 0]
                      if a.credit != "joint" else jnp.zeros(N))
            crit = o_pol.values_shared if a.credit == "A" else (
                sum(jnp.take_along_axis(o_pol.q_slots[i], toks[:, i:i + 1], -1)[:, 0] for i in range(3))
                if a.credit == "B" else jnp.zeros(N))
            # sender's counterfactual broadcast for T' (paired key) -- used by the swap eval and the swap-diff metric
            obs_sw = build_obs(pos, (active + shift) % 4, wire, k_obs)
            _, _, _, _, _, _, toks_sw, bc_sw, _, _ = policy(params, carries, obs_sw, k_act)
            swap_diff = jnp.any(toks_sw != toks, -1) & ~is_listener
            # listener sensitivity: received wire replaced by a uniform random token triple (diagnostic only)
            rt = jax.random.randint(k_rand, (N, 3), 0, V)
            rbc = jnp.concatenate([jnp.zeros((N, 8))] + [cb[i][rt[:, i]] for i in range(3)], -1)
            w_r = jnp.where(is_listener[:, None], wire, rbc)          # replace the SENDER's entry (partner of listener)
            _, o_r = apply_fn(params, carries, build_obs(pos, active, w_r, k_obs))
            sens = tv(logits, mask_disabled_actions(o_r.action_logits, axis=-1))
            # swap modes (eval only): "constructed" = the sender's own policy sample for T' (same carry/key);
            # "empirical" = a full token triple drawn from the triples this sender emitted for T' in the paired
            # control rollout (Cam, 2026-11-07: guarantees the listener hears an in-distribution token)
            if swap == "empirical":
                k_tab = jax.random.fold_in(k_rand, 7)          # fold_in: leaves every training RNG stream unchanged
                idx = jax.random.randint(k_tab, (N,), 0, table.shape[1])
                trip = table[jnp.repeat((active + shift) % 4, 2), idx]
                bc_emp = jnp.concatenate([jnp.zeros((N, 8))] + [cb[i][trip[:, i]] for i in range(3)], -1)
                bc_used, trip_used = bc_emp, trip
            else:
                bc_used, trip_used = bc_sw, toks_sw
            if SYMR:   # step 3: every currently informed speaker's broadcast is swapped
                inf_now = obs[:, O_INF] > 0.5
                bc_out = jnp.where(inf_now[:, None], bc_used, bc) if swap else bc
            elif swap == "shuffle":   # step 6 null: each listener hears ANOTHER world's sender broadcast
                bs = bc.reshape(W, 2, -1)
                bc_out = bs.at[:, 0].set(jnp.roll(bs[:, 0], 1, axis=0)).reshape(N, -1)
            else:
                bc_out = jnp.where(is_listener[:, None], bc, bc_used) if swap else bc
            # movement toward T' and T (listeners)
            mv = DEL[act].astype(jnp.float32)
            def mu(tgt):
                h = HEARTHS[jnp.repeat(tgt, 2)]
                dv = (h - pos + G / 2) % G - G / 2
                nr = jnp.linalg.norm(dv, axis=1)
                m = is_listener & (act >= 1) & (act <= 4) & (nr > 0)
                return jnp.where(m, (mv * dv).sum(1) / jnp.maximum(nr, 1e-6), 0.0), m
            fire = fire_of(active)
            mu_sw, m_sw = mu((fire + shift) % 4)
            mu_tr, m_tr = mu(fire)
            nr_ = near(pos)
            craft = jnp.isin(act, CRAFT_ACTS) if COMP else act == ACTION_CRAFT
            at_active = nr_[jnp.arange(N), jnp.repeat(fire, 2)]
            dep = craft & at_active
            if COMP:   # step 6: the delivery also needs the target material's Craft action
                dep = dep & (act == CRAFT_ACTS[jnp.repeat(active % 4, 2)])
            att = craft & nr_.any(1)
            att_Tp = craft & nr_[jnp.arange(N), jnp.repeat((fire + shift) % 4, 2)]     # Craft at the T' hearth
            r = jnp.where(dep, R_DEP, 0.0) - a.craft_cost * (craft & ~dep)
            ldep_w = dep[is_listener]
            if SYMR:   # step 3: symmetric shared reward -- either agent's delivery also pays its partner
                r = r + R_DEP * dep.reshape(W, 2)[:, ::-1].reshape(N)
            else:
                r = r + jnp.where(~is_listener, R_DEP * jnp.repeat(ldep_w, 2), 0.0)   # shared reward to the sender
            newpos = (pos + DEL[act]) % G
            rs = VALID[jax.random.randint(k_rs, (N,), 0, VALID.shape[0])]
            pos2 = jnp.where(dep[:, None], rs, newpos)
            dep_w = dep.reshape(W, 2).any(1)
            since2 = since + 1
            reroll = dep_w | (since2 >= a.reroll)
            if COMP:   # re-roll to a different pair from the pool (training pool, or the held-out pool in the frozen test)
                pl = TRAIN_POOL if pool is None else pool
                cur = jnp.argmax(pl[None, :] == active[:, None], -1)
                nxt = pl[(cur + jax.random.randint(k_rr, (W,), 1, max(int(pl.shape[0]), 2))) % pl.shape[0]]
                active2 = jnp.where(reroll, nxt, active)
            else:
                active2 = jnp.where(reroll, (active + jax.random.randint(k_rr, (W,), 1, 4)) % 4, active)
            since2 = jnp.where(reroll, 0, since2)
            wire2 = bc_out if live_wire else jnp.zeros_like(bc_out)
            heard_next = wire2.reshape(W, 2, -1)[:, ::-1].reshape(N, -1)
            r_shared = jnp.where(~is_listener, R_DEP * jnp.repeat(ldep_w, 2), 0.0)
            extra = dict(inf=obs[:, O_INF] > 0.5, used_bc=bc_used) if SYMR else {}
            if LOOK is not None:
                extra["atl"] = at_look(pos)
            rec = dict(**extra, lp_act=lp_act, crit=crit, r_shared=r_shared.astype(jnp.float32), obs=obs, actions=act, log_probs=lp, rewards=r.astype(jnp.float32), values=val,
                       carries=carries, loss_vq=lvq, token_actions=toks, ent=ent, dep=dep, att=att,
                       swap_diff=swap_diff, sens=sens, att_Tp=att_Tp, mu_sw=mu_sw, m_sw=m_sw, mu_tr=mu_tr, m_tr=m_tr,
                       active=active, heard_next=heard_next, bc_sw=bc_used, toks_sw=toks_sw, trip_used=trip_used)
            return (pos2, active2, since2, wire2, new_c), rec

        ks = jax.random.split(key, T)
        st, recs = jax.lax.scan(step, state, ks)
        return st, recs

    def init_state(r_key, pool=None):
        k1, k2 = jax.random.split(r_key)
        pos = VALID[jax.random.randint(k1, (N,), 0, VALID.shape[0])]
        if COMP:
            pl = TRAIN_POOL if pool is None else pool
            act0 = pl[jax.random.randint(k2, (W,), 0, pl.shape[0])]
        else:
            act0 = jax.random.randint(k2, (W,), 0, 4)
        return (pos, act0, jnp.zeros(W, jnp.int32), jnp.zeros((N, SIG)), jnp.zeros((N, H)))

    # ---------------------------------------------------------------- aux losses
    def ps_term(params, obs, carries, role_mask=True):
        """MI(token; active hearth) on sender rows, summed over the 3 slots (nats); batch estimate."""
        _, o = apply_fn(params, carries, obs)
        snd = obs[:, O_INF] > 0.5
        if oracle:
            snd = snd & False
        if not role_mask:          # mutation-test only
            snd = jnp.ones_like(snd)
        Tl = jnp.argmax(obs[:, O_NEED:O_NEED + 4], -1)
        if COMP:
            Tl = Tl * 4 + jnp.argmax(obs[:, O_NOISE:O_NOISE + 4], -1)
        w_s = snd.astype(jnp.float32)
        mi = 0.0
        for lg in token_logits(params, o.z_e, TAU):
            p = jax.nn.softmax(lg, -1)
            pbar = (p * w_s[:, None]).sum(0) / (w_s.sum() + 1e-8)
            h_all = -(pbar * jnp.log(pbar + 1e-10)).sum()
            h_cond = 0.0
            for tt in range(NCLS):
                wt = w_s * (Tl == tt)
                pt = (p * wt[:, None]).sum(0) / (wt.sum() + 1e-8)
                h_cond = h_cond + (wt.sum() / (w_s.sum() + 1e-8)) * -(pt * jnp.log(pt + 1e-10)).sum()
            mi = mi + h_all - h_cond
        return mi

    def pl_term(params, obs, carries, role_mask=True):
        """mean L1 between the listener's action distribution with the wire and (stop-grad) with the wire zeroed."""
        _, o = apply_fn(params, carries, obs)
        obs0 = obs.at[:, O_WIRE:].set(0.0)
        _, o0 = apply_fn(params, carries, obs0)
        p = jax.nn.softmax(mask_disabled_actions(o.action_logits, axis=-1), -1)
        p0 = jax.lax.stop_gradient(jax.nn.softmax(mask_disabled_actions(o0.action_logits, axis=-1), -1))
        lst = (obs[:, O_INF] < 0.5).astype(jnp.float32)
        if not role_mask:          # mutation-test only
            lst = jnp.ones_like(lst)
        return (jnp.abs(p - p0).sum(-1) * lst).sum() / (lst.sum() + 1e-8)

    def huber(e, d=0.5):
        e = jnp.abs(e)
        return jnp.where(e < d, 0.5 * e * e, d * (e - 0.5 * d))

    def credit_terms(params, mb):
        """Credit fixes A / B: the token's own clipped surrogate on its own advantage, + the credit critic's loss."""
        _, o = apply_fn(params, mb["carries"], mb["obs"])
        lgs = token_logits(params, o.z_e, TAU)
        lsm = [jax.nn.log_softmax(lg, -1) for lg in lgs]
        lp_new = sum(jnp.take_along_axis(lsm[i], mb["token_actions"][:, i:i + 1], -1)[:, 0] for i in range(3))
        ent = sum(-jnp.sum(jnp.exp(l_) * l_, -1) for l_ in lsm)
        snd = (mb["obs"][:, O_INF] > 0.5).astype(jnp.float32)
        if a.credit in ("A", "Ag"):
            adv_t = mb["adv_tok"]
            crit_loss = (jnp.mean(huber(o.values_shared - (mb["ret_s"] - mb["mu_s"]) / mb["sig_s"])) if a.credit == "A"
                         else jnp.array(0.0))     # Ag: the central critic is trained separately (crit_step)
        else:
            q = ([jax.lax.stop_gradient(qq) for qq in o.q_slots] if a.credit == "B"
                 else [jax.lax.stop_gradient(mb["qg"][:, i]) for i in range(3)])
            pi = [jax.lax.stop_gradient(jnp.exp(l_)) for l_ in lsm]
            A_i = [jnp.take_along_axis(q[i], mb["token_actions"][:, i:i + 1], -1)[:, 0] - jnp.sum(pi[i] * q[i], -1)
                   for i in range(3)]
            raw = sum(A_i)                                            # COMA counterfactual, additive over slots
            mu_ = (raw * snd).sum() / (snd.sum() + 1e-8)
            sd_ = jnp.sqrt((((raw - mu_) ** 2) * snd).sum() / (snd.sum() + 1e-8)) + 1e-8
            adv_t = jnp.clip((raw - mu_) / sd_, -10, 10) * snd
            if a.credit == "B":
                q_taken = sum(jnp.take_along_axis(o.q_slots[i], mb["token_actions"][:, i:i + 1], -1)[:, 0] for i in range(3))
                crit_loss = (huber(q_taken - mb["g_s"] / Q_SCALE) * snd).sum() / (snd.sum() + 1e-8)
            else:
                crit_loss = jnp.array(0.0)
        ratio = jnp.clip(jnp.exp(lp_new - mb["lp_tok_old"]), 0.0, 10.0)
        pg = -jnp.minimum(ratio * adv_t, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv_t)
        return jnp.mean(pg), jnp.mean(ent), crit_loss, adv_t

    def total_loss(params, mb, vs, vsh, ps_c, te_c, ps_force=None):
        if a.credit != "joint":
            loss, mets = ppo_loss(params, apply_fn, mb["obs"], mb["actions"], None, mb["lp_act"], mb["adv"],
                                  mb["ret"], mb["carries"], 1, mb["values"], clip_eps=CLIP, vf_coef=VF, ent_coef=ENT,
                                  vq_coef=VQC, value_scale=vs, value_clip=False, value_shift=vsh, popart_mode=True,
                                  token_actions=None)
            tpg, tent, closs, _ = credit_terms(params, mb)
            tec = te_c if a.erode_tok_ent is not None else TOK_ENT
            loss = loss + tpg - tec * tent + VF * closs
            mets = dict(mets, tok_pg=tpg, tok_ent=tent, credit_critic_loss=closs)
            ps = ps_term(params, mb["obs"], mb["carries"]) if a.ps else jnp.array(0.0)
            ps_coef = ps_c if a.ps_anneal else a.ps
            if ps_force is not None:   # apparatus / --msg-grad sender only; the default path never sets it
                ps_coef = ps_force
            mets = dict(mets, aux_ps=ps, aux_pl=jnp.array(0.0), ps_coef=ps_coef)
            return loss - ps_coef * ps, mets
        loss, mets = ppo_loss(params, apply_fn, mb["obs"], mb["actions"], None, mb["log_probs"], mb["adv"],
                              mb["ret"], mb["carries"], 1, mb["values"], clip_eps=CLIP, vf_coef=VF, ent_coef=ENT,
                              vq_coef=VQC, value_scale=vs, value_clip=False, value_shift=vsh, popart_mode=True,
                              alive=mb.get("alive"),
                              token_actions=mb["token_actions"], token_tau=TAU,
                              token_ent_coef=(te_c if a.erode_tok_ent is not None else TOK_ENT))
        ps = ps_term(params, mb["obs"], mb["carries"]) if a.ps else jnp.array(0.0)
        pl = pl_term(params, mb["obs"], mb["carries"]) if a.pl else jnp.array(0.0)
        mets = dict(mets, aux_ps=ps, aux_pl=pl)
        ps_coef = ps_c if a.ps_anneal else a.ps            # constant path unchanged when not annealing
        if ps_force is not None:   # apparatus / --msg-grad sender only; the default path never sets it
            ps_coef = ps_force
        mets = dict(mets, ps_coef=ps_coef)
        return loss - ps_coef * ps - a.pl * pl, mets

    def _clip_group(g, max_norm=2.0):
        norm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
        s = jnp.minimum(1.0, max_norm / (norm + 1e-6))
        return jax.tree_util.tree_map(lambda x: x * s, g)

    @jax.jit
    def mb_step(params, opt_state, mb, vs, vsh, ps_c, te_c):
        (loss, mets), grads = jax.value_and_grad(total_loss, has_aux=True)(params, mb, vs, vsh, ps_c, te_c)
        g = dict(grads["params"])
        crit_keys = [k for k in g if k == "head_value_shared" or k.startswith("head_q_slot")]
        groups = [["head_action"], ["head_value"], *([crit_keys] if crit_keys else []),
                  [k for k in g if k not in ("head_action", "head_value", *crit_keys)]]
        for grp in groups:
            c = _clip_group({k: g[k] for k in grp})
            g.update(c)
        grads = {"params": g}
        upd, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, mets

    MSGK = ("gwt_comms_1", "head_signal_slot0", "head_signal_slot1", "head_signal_slot2", "codebook_0", "codebook_1", "codebook_2")

    @jax.jit
    def mb_step_stopgrad(params, opt_state, mb, vs, vsh, ps_c, te_c):
        """step B (iii): the message parameters get gradient from SENDER rows only (the listener's token PG, VQ and
        token-entropy terms cannot reach them); every other parameter gets the usual all-row gradient. Sender-row
        grads are rescaled by n_sender / n_rows so their magnitude matches the all-row mean."""
        (loss, mets), g_all = jax.value_and_grad(total_loss, has_aux=True)(params, mb, vs, vsh, ps_c, te_c)
        snd = (mb["obs"][:, O_INF] > 0.5).astype(jnp.float32)
        mb_s = dict(mb, alive=snd)
        g_snd = jax.grad(lambda p_: total_loss(p_, mb_s, vs, vsh, ps_c, te_c)[0])(params)
        scale = snd.sum() / snd.shape[0]
        g = dict(g_all["params"])
        for k in MSGK:
            g[k] = jax.tree_util.tree_map(lambda x: x * scale, g_snd["params"][k])
        groups = [["head_action"], ["head_value"], [k for k in g if k not in ("head_action", "head_value")]]
        for grp in groups:
            g.update(_clip_group({k: g[k] for k in grp}))
        upd, opt_state = opt.update({"params": g}, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, mets

    def sg2_grads(params, mb, vs, vsh, ps_c, te_c, version="fixed", leak_row=-1):
        """Cam Nov 9 (from-scratch stop-grad, PS fix): every parameter gets the usual all-row gradient, EXCEPT the message
        head (MSGK), which gets (n_sender / n_rows) x the sender-row gradient of the loss WITHOUT the PS term, plus the PS
        term's own gradient UNSCALED (the PS term is a batch statistic over sender rows, not a row mean). version="old" =
        the resume-branch composition (the scaled sender-row gradient INCLUDING PS), kept for the apparatus check only.
        leak_row >= 0 (mutant, apparatus only): that minibatch row is counted as a sender row."""
        g_all = jax.grad(lambda p_: total_loss(p_, mb, vs, vsh, ps_c, te_c)[0])(params)
        snd = (mb["obs"][:, O_INF] > 0.5).astype(jnp.float32)
        if leak_row >= 0:
            snd = snd.at[leak_row].set(1.0)
        scale = snd.sum() / snd.shape[0]
        mb_s = dict(mb, alive=snd)
        if version == "old":
            g_s = jax.grad(lambda p_: total_loss(p_, mb_s, vs, vsh, ps_c, te_c)[0])(params)
            msg = {k: jax.tree_util.tree_map(lambda x: x * scale, g_s["params"][k]) for k in MSGK}
        else:
            g_s0 = jax.grad(lambda p_: total_loss(p_, mb_s, vs, vsh, ps_c, te_c, ps_force=0.0)[0])(params)
            ps_eff = ps_c if a.ps_anneal else a.ps                 # the coefficient total_loss itself applies
            g_ps = (jax.grad(lambda p_: -ps_eff * ps_term(p_, mb["obs"], mb["carries"]))(params) if a.ps else None)
            msg = {k: jax.tree_util.tree_map(lambda x, y: x * scale + y, g_s0["params"][k], g_ps["params"][k]) if g_ps is not None
                   else jax.tree_util.tree_map(lambda x: x * scale, g_s0["params"][k]) for k in MSGK}
        g = dict(g_all["params"]); g.update(msg)
        return {"params": g}

    @jax.jit
    def mb_step_sg2(params, opt_state, mb, vs, vsh, ps_c, te_c):
        (loss, mets) = total_loss(params, mb, vs, vsh, ps_c, te_c)
        g = dict(sg2_grads(params, mb, vs, vsh, ps_c, te_c)["params"])
        groups = [["head_action"], ["head_value"], [k for k in g if k not in ("head_action", "head_value")]]
        for grp in groups:
            g.update(_clip_group({k: g[k] for k in grp}))
        upd, opt_state = opt.update({"params": g}, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, mets

    def wireblind(params):
        pp = dict(params["params"])
        k0 = jnp.asarray(pp["gwt_comms_1"]["kernel"])
        pp["gwt_comms_1"] = dict(pp["gwt_comms_1"], kernel=k0.at[O_WIRE:].set(0.0))
        return {"params": pp}

    # ---------------------------------------------------------------- self-test of the aux losses
    if a.selftest:
        st_, R0 = rollout(params, init_state(jax.random.PRNGKey(7)), jax.random.PRNGKey(8))
        obs = jnp.asarray(R0["obs"]).reshape(-1, D)[:2048]
        car = jnp.asarray(R0["carries"]).reshape(-1, H)[:2048]
        lst_rows = np.asarray(obs[:, O_INF] < 0.5)
        ok = True
        for name, fn, own_rows in (("PS", ps_term, ~lst_rows), ("PL", pl_term, lst_rows)):
            g_obs = np.asarray(jax.grad(lambda o: fn(params, o, car))(obs))
            other = np.abs(g_obs[~own_rows]).max(); own = np.abs(g_obs[own_rows]).max()
            g_p = jax.grad(lambda p: fn(p, obs, car))(params)
            gn = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g_p))))
            p2 = jax.tree_util.tree_map(lambda x, g: x + 1e-2 * g / (gn + 1e-12), params, g_p)
            before, after = float(fn(params, obs, car)), float(fn(p2, obs, car))
            res = dict(term=name, grad_obs_own_rows=float(own), grad_obs_other_rows=float(other), grad_norm=gn,
                       value_before=before, value_after_ascent=after)
            print(json.dumps(res), flush=True)
            ok &= other == 0.0 and own > 0 and gn > 0 and after > before
        # mutation: PS computed on LISTENER rows instead must be caught by the role check
        def ps_mut(p, o, c):
            return ps_term(p, o, c, role_mask=False)
        g_m = np.asarray(jax.grad(lambda o: ps_mut(params, o, car))(obs))
        caught = np.abs(g_m[lst_rows]).max() > 0
        print(json.dumps(dict(mutant="PS without the sender-row mask", caught=bool(caught))), flush=True)
        ok &= bool(caught)
        g_m2 = np.asarray(jax.grad(lambda o: pl_term(params, o, car, role_mask=False))(obs))
        caught2 = np.abs(g_m2[~lst_rows]).max() > 0
        print(json.dumps(dict(mutant="PL without the listener-row mask", caught=bool(caught2))), flush=True)
        ok &= bool(caught2)
        print("SELFTEST", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)

    if a.measure_sens:
        vals = []
        for k in range(4):
            _, Rm = rollout(params, init_state(jax.random.PRNGKey(50 + k)), jax.random.PRNGKey(60 + k))
            vals.append(float(np.asarray(Rm["sens"])[:, np.asarray(is_listener)].mean()))
        print(json.dumps(dict(wire_init_scale=a.wire_init_scale, listener_sens_tv=float(np.mean(vals)))), flush=True)
        raise SystemExit(0)

    # ---------------------------------------------------------------- coefficient calibration (registered rule)
    if a.calibrate:
        # c_aux = 0.25 * ||grad PPO|| / ||grad aux||, full-parameter L2 norms (pre-clip), mean over 4 minibatches
        # of 512 from one COMM rollout at the warm-start params.
        _, Rc = rollout(params, init_state(jax.random.PRNGKey(3)), jax.random.PRNGKey(4))
        Rc = jax.tree_util.tree_map(np.asarray, Rc)
        vals = popart.denormalise(Rc["values"])
        adv, ret = gae_bootstrap(Rc["rewards"], vals, np.zeros_like(Rc["rewards"]), vals[-1], GAMMA, LAM)
        params_c = popart.update_and_preserve(ret, params)
        M = T * N
        flat = {k: Rc[k].reshape((M,) + Rc[k].shape[2:]) for k in ("obs", "actions", "log_probs", "carries",
                                                                    "token_actions")}
        flat["values"], flat["adv"], flat["ret"] = vals.reshape(M), adv.reshape(M), ret.reshape(M).astype(np.float32)
        perm = np.random.default_rng(0).permutation(M)
        gn = lambda g: float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g))))  # noqa: E731
        r_ps, r_pl = [], []
        for i in range(4):
            mb = {k: jnp.asarray(v[perm[i * MB:(i + 1) * MB]]) for k, v in flat.items()}
            g_ppo = jax.grad(lambda p: ppo_loss(
                p, apply_fn, mb["obs"], mb["actions"], None, mb["log_probs"], mb["adv"], mb["ret"], mb["carries"], 1,
                mb["values"], clip_eps=CLIP, vf_coef=VF, ent_coef=ENT, vq_coef=VQC, value_scale=popart.sigma,
                value_clip=False, value_shift=popart.mu, popart_mode=True, token_actions=mb["token_actions"],
                token_tau=TAU, token_ent_coef=TOK_ENT)[0])(params_c)
            g_ps = jax.grad(lambda p: ps_term(p, mb["obs"], mb["carries"]))(params_c)
            g_pl = jax.grad(lambda p: pl_term(p, mb["obs"], mb["carries"]))(params_c)
            r_ps.append(gn(g_ppo) / gn(g_ps)); r_pl.append(gn(g_ppo) / gn(g_pl))
        print(json.dumps(dict(calibration="0.25 * |grad ppo| / |grad aux|", c_ps=0.25 * float(np.mean(r_ps)),
                              c_pl=0.25 * float(np.mean(r_pl)), ratios_ps=r_ps, ratios_pl=r_pl)), flush=True)
        raise SystemExit(0)

    # ---------------------------------------------------------------- loop
    def emit(rec):
        rec = dict(arm=a.arm, seed=a.seed, ps=a.ps, pl=a.pl, tag=a.tag, craft_cost=a.craft_cost,
                   warm_start=bool(a.warm_start), **rec)
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    HEADS = ("gwt_comms_1", "head_signal_slot0", "head_signal_slot1", "head_signal_slot2")
    diag_state = {}

    def head_vec(tree):
        t = tree["params"] if "params" in tree else tree
        return np.concatenate([np.asarray(x, np.float64).ravel() for h in HEADS for x in jax.tree_util.tree_leaves(t[h])])

    def reliance_probe(params):
        # light reliance probe: a.diag_pairs paired control / empirical-swap rollouts (own keys), deposits and
        # crafts at the swapped hearth
        cs, ss, rc, rsw = [], [], [], []
        for k in range(a.diag_pairs):
            st0 = init_state(jax.random.PRNGKey(30_000 + k))
            _, Rc = rollout(params, st0, jax.random.PRNGKey(40_000 + k), swap=False)
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            trip_c, labc = Rc["token_actions"][:, ~lis], Rc["active"]
            table = np.zeros((4, 512, 3), np.int32); rng_p = np.random.default_rng(k)
            for t in range(4):
                tt = trip_c[labc == t]
                table[t] = tt[rng_p.integers(0, len(tt), 512)] if len(tt) else 0
            _, Rk = rollout(params, st0, jax.random.PRNGKey(40_000 + k), swap="empirical", table=jnp.asarray(table))
            Rk = jax.tree_util.tree_map(np.asarray, Rk)
            cs.append(Rc["dep"][:, lis].sum() / W); ss.append(Rk["dep"][:, lis].sum() / W)
            rc.append(Rc["att_Tp"][:, lis].sum() / W); rsw.append(Rk["att_Tp"][:, lis].sum() / W)
        c, s_ = float(np.mean(cs)), float(np.mean(ss))
        return dict(rel_dep_control=c, rel_dep_swap=s_, rel_drop=1 - s_ / max(c, 1e-9),
                    rel_crafts_Tp_control=float(np.mean(rc)), rel_crafts_Tp_swap=float(np.mean(rsw)))

    def evaluate(params, u):
        """MI over the last 10 updates' sender tokens + paired swaps in two modes. Uses its own RNGs, so an eval in
        the middle of training does not perturb the training streams."""
        from comm_testbed import mi_miller_madow
        ev_rng = np.random.default_rng(1000 + 7919 * a.seed + u)
        tok, Tt = (np.concatenate(list(mi_buf_tok)), np.concatenate(list(mi_buf_T))) if mi_buf_tok else (None, None)
        for i in (range(3) if tok is not None else []):
            mi = mi_miller_madow(tok[:, i], Tt, V, NCLS)
            null = np.array([mi_miller_madow(tok[:, i], ev_rng.permutation(Tt), V, NCLS) for _ in range(200)])
            emit(dict(phase="mi", update=u, slot=i, mi_bits=mi, null_p99=float(np.percentile(null, 99)), n=int(len(Tt))))
        if a.arm != "COMM":
            return
        enc = lambda x: (x[..., 0] * V + x[..., 1]) * V + x[..., 2]  # noqa: E731
        if SYMR:
            evaluate_prox(params, u, ev_rng); return
        if COMP:
            evaluate_comp(params, u, ev_rng); return
        for k in range(a.swap_pairs):
            st0 = init_state(jax.random.PRNGKey(10_000 + k))
            _, Rc = rollout(params, st0, jax.random.PRNGKey(20_000 + k), swap=False)
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            trip_c, lab = Rc["token_actions"][:, ~lis], Rc["active"]                 # (T, W, 3), (T, W)
            sets = [set(enc(trip_c[lab == t]).tolist()) for t in range(4)]
            s_all = set().union(*sets)
            table = np.zeros((4, 512, 3), np.int32)
            for t in range(4):
                tt = trip_c[lab == t]
                if len(tt) == 0:
                    print(f"[APPARATUS] FATAL: no control-rollout tokens for hearth {t}", flush=True); raise SystemExit(1)
                table[t] = tt[ev_rng.integers(0, len(tt), 512)]
            ctrl = (Rc["mu_sw"].sum() / max(Rc["m_sw"].sum(), 1), Rc["mu_tr"].sum() / max(Rc["m_tr"].sum(), 1),
                    float(Rc["dep"][:, lis].sum() / W), float(Rc["att_Tp"][:, lis].sum() / W))
            for mode in ("constructed", "empirical"):
                _, Rk = rollout(params, st0, jax.random.PRNGKey(20_000 + k), swap=mode, table=jnp.asarray(table))
                Rk = jax.tree_util.tree_map(np.asarray, Rk)
                if not np.allclose(Rk["heard_next"][:, lis], Rk["bc_sw"][:, ~lis]):   # listener hears the swap token
                    print("[APPARATUS] FATAL: swap wiring", flush=True); raise SystemExit(1)
                used, lab_sw = Rk["trip_used"][:, ~lis], (Rk["active"] + 1) % 4
                e_used = enc(used)
                in_Tp = float(np.mean([e in sets[t] for e, t in zip(e_used.ravel().tolist(), lab_sw.ravel().tolist())]))
                in_any = float(np.mean([e in s_all for e in e_used.ravel().tolist()]))
                emit(dict(phase="swap", update=u, mode=mode, pair=k, mu_Tprime_control=float(ctrl[0]),
                          mu_Tprime_swap=float(Rk["mu_sw"].sum() / max(Rk["m_sw"].sum(), 1)),
                          mu_T_control=float(ctrl[1]), mu_T_swap=float(Rk["mu_tr"].sum() / max(Rk["m_tr"].sum(), 1)),
                          listener_dep_control=ctrl[2], listener_dep_swap=float(Rk["dep"][:, lis].sum() / W),
                          swapped_token_differs_frac=float(np.any(used != Rk["token_actions"][:, ~lis], -1).mean()),
                          swap_token_seen_for_Tprime=in_Tp, swap_token_seen_at_all=in_any,
                          listener_crafts_at_Tprime_control=ctrl[3],
                          listener_crafts_at_Tprime_swap=float(Rk["att_Tp"][:, lis].sum() / W)))

    def evaluate_comp(params, u, ev_rng):
        """step 6: (a) topographic similarity between meaning distance (Hamming over (fire, material)) and message
        distance (Hamming over the 3 slots of each meaning's modal token triple), training meanings, vs a 1,000-perm
        null over the message-to-meaning assignment; (b) held-out generalisation: frozen rollouts where only the
        held-out pairs are ever active -- listener success per Craft attempt at a fire (both parts right; chance 1/16),
        with real messages vs a shuffled-message null (each listener hears another world's sender)."""
        from scipy.stats import spearmanr
        toks, labs = [], []
        for k in range(4):
            _, Rc = rollout(params, init_state(jax.random.PRNGKey(60_000 + k)), jax.random.PRNGKey(61_000 + k))
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            toks.append(Rc["token_actions"][:, ~lis].reshape(-1, 3)); labs.append(Rc["active"].reshape(-1))
        tok, lab = np.concatenate(toks), np.concatenate(labs)
        modal = {}
        for q in np.unique(lab):
            tq = tok[lab == q]
            if len(tq) >= 20:
                vals, cnts = np.unique(tq, axis=0, return_counts=True); modal[int(q)] = vals[cnts.argmax()]
        qs = sorted(modal)
        pairs = [(i, j) for i in range(len(qs)) for j in range(i + 1, len(qs))]
        mdist = np.array([(qs[i] // 4 != qs[j] // 4) + (qs[i] % 4 != qs[j] % 4) for i, j in pairs])
        msgs = np.stack([modal[q] for q in qs])
        gd = lambda M: np.array([int((M[i] != M[j]).sum()) for i, j in pairs])  # noqa: E731
        rho = float(spearmanr(mdist, gd(msgs)).correlation) if len(pairs) > 2 else float("nan")
        null = [spearmanr(mdist, gd(msgs[ev_rng.permutation(len(qs))])).correlation for _ in range(1000)]
        null = np.nan_to_num(np.array(null, float))
        emit(dict(phase="topsim", update=u, rho=rho, null_p95=float(np.percentile(null, 95)), n_meanings=len(qs),
                  distinct_messages=int(len(np.unique(msgs, axis=0)))))
        HP = jnp.asarray(HELD)
        c = dict(sr=0, ar=0, ss=0, as_=0)
        for k in range(10):
            st0 = init_state(jax.random.PRNGKey(62_000 + k), pool=HP)
            for mode, kk in ((False, "r"), ("shuffle", "s")):
                _, Rh = rollout(params, st0, jax.random.PRNGKey(63_000 + k), swap=mode, pool=HP)
                Rh = jax.tree_util.tree_map(np.asarray, Rh)
                if not np.isin(Rh["active"], HELD).all():
                    print("[APPARATUS] FATAL: held-out eval saw a training pair", flush=True); raise SystemExit(1)
                c["s" + kk] += int(Rh["dep"][:, lis].sum()); c[("a" + kk) if kk == "r" else "as_"] += int(Rh["att"][:, lis].sum())
        _, Rt = rollout(params, init_state(jax.random.PRNGKey(64_000)), jax.random.PRNGKey(65_000))
        Rt = jax.tree_util.tree_map(np.asarray, Rt)
        emit(dict(phase="heldout", update=u, succ_real=c["sr"], att_real=c["ar"], succ_shuf=c["ss"], att_shuf=c["as_"],
                  rate_real=c["sr"] / max(c["ar"], 1), rate_shuf=c["ss"] / max(c["as_"], 1),
                  train_rate=float(Rt["dep"][:, lis].sum() / max(Rt["att"][:, lis].sum(), 1))))

    def evaluate_prox(params, u, ev_rng):
        """step 3 swap: every currently informed speaker's broadcast is replaced by a full token triple drawn from the
        triples informed agents emitted for T' in the paired control rollout. Redirection = Crafts at the T' hearth by
        agents uninformed at that step; deposits = all deposits."""
        partner = np.arange(N).reshape(W, 2)[:, ::-1].reshape(N)
        for k in range(a.swap_pairs):
            st0 = init_state(jax.random.PRNGKey(10_000 + k))
            _, Rc = rollout(params, st0, jax.random.PRNGKey(20_000 + k), swap=False)
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            lab = np.repeat(Rc["active"], 2, axis=1); inf = Rc["inf"]
            table = np.zeros((4, 512, 3), np.int32)
            for t in range(4):
                tt = Rc["token_actions"][inf & (lab == t)]
                if len(tt) == 0:
                    print(f"[APPARATUS] FATAL: no informed control tokens for hearth {t}", flush=True); raise SystemExit(1)
                table[t] = tt[ev_rng.integers(0, len(tt), 512)]
            _, Rk = rollout(params, st0, jax.random.PRNGKey(20_000 + k), swap="empirical", table=jnp.asarray(table))
            Rk = jax.tree_util.tree_map(np.asarray, Rk)
            pinf = Rk["inf"][:, partner]          # the partner (speaker) was informed -> what this agent hears is the swap
            if not np.allclose(Rk["heard_next"][pinf], Rk["used_bc"][:, partner][pinf]):
                print("[APPARATUS] FATAL: prox swap wiring", flush=True); raise SystemExit(1)
            red = lambda R_: float((R_["att_Tp"] & ~R_["inf"]).sum() / W)  # noqa: E731
            emit(dict(phase="swap", update=u, mode="empirical", pair=k, prox=True,
                      listener_crafts_at_Tprime_control=red(Rc), listener_crafts_at_Tprime_swap=red(Rk),
                      listener_dep_control=float(Rc["dep"].sum() / W), listener_dep_swap=float(Rk["dep"].sum() / W),
                      informed_frac=float(Rc["inf"].mean()),
                      mu_Tprime_control=0.0, mu_Tprime_swap=0.0, swap_token_seen_for_Tprime=1.0, swap_token_seen_at_all=1.0))

    def mi_plugin(tok, Tl):
        # plug-in MI(token; T), mean over the 3 slots, bits (cheap per-update trace; numpy only, no RNG)
        out = []
        for i in range(3):
            j = np.zeros((V, NCLS)); np.add.at(j, (tok[:, i], Tl), 1); pj = j / j.sum()
            px, py = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True); nz = pj > 0
            out.append(float((pj[nz] * np.log2(pj[nz] / (px @ py)[nz])).sum()))
        return float(np.mean(out))

    def gfeat(obs, carries, active):
        """global critic features for SENDER rows: sender obs, listener obs, listener GRU state, active-fire one-hot.
        obs (..., N, D), carries (..., N, H), active (..., W) -> (..., W, DG)."""
        return jnp.concatenate([obs[..., 0::2, :], obs[..., 1::2, :], carries[..., 1::2, :],
                                jax.nn.one_hot(active, 4)], -1)

    def central_streams(R, o_last, params, M, rew, mutant, cp=None, listener_blind=False):
        cp = cparams if cp is None else cp
        G = gfeat(jnp.asarray(R["obs"]), jnp.asarray(R["carries"]), jnp.asarray(R["active"]))       # (T, W, DG)
        GL = gfeat(central_last["obs"][None], central_last["carries"][None], central_last["active"][None])[0]
        if listener_blind:   # mutant for the self-test: a LOCAL critic (listener and fire inputs zeroed)
            mask = jnp.concatenate([jnp.ones(D), jnp.zeros(D + H + 4)])
            G, GL = G * mask, GL * mask
        r_s = (rew if mutant in ("full_adv", "total_reward") else R["r_shared"])[:, 0::2]          # sender rows
        out = {"lp_act": R["lp_act"].reshape(M).astype(np.float32),
               "lp_tok_old": (R["log_probs"] - R["lp_act"]).reshape(M).astype(np.float32)}
        Tn = R["obs"].shape[0]
        gf = np.zeros((Tn, N, DG), np.float32); gf[:, 0::2] = np.asarray(G)
        out["gfeat"] = gf.reshape(M, DG)
        if a.credit == "Ag":
            v = popart_g.denormalise(np.asarray(ccrit.apply(cp, G)))
            v_last = popart_g.denormalise(np.asarray(ccrit.apply(cp, GL)))
            adv_s, ret_s = gae_raw(r_s, v, v_last)
            z = np.clip((adv_s - adv_s.mean()) / (adv_s.std() + 1e-8), -10, 10)
            full = np.zeros((Tn, N)); full[:, 0::2] = z
            rr = np.zeros((Tn, N)); rr[:, 0::2] = ret_s
            out["adv_tok"] = full.reshape(M).astype(np.float32); out["ret_g"] = rr.reshape(M).astype(np.float32)
            out["_ret_g_full"] = ret_s
            out["_v_norm"] = np.asarray(ccrit.apply(cp, G))
        else:
            qs = [np.asarray(q) for q in ccrit.apply(cp, G)]                                         # 3 x (T, W, 64)
            tok_s = R["token_actions"][:, 0::2]
            q_taken = sum(np.take_along_axis(qs[i], tok_s[..., i:i + 1], -1)[..., 0] for i in range(3))
            lg_last = token_logits(params, o_last.z_e, TAU)
            ql = ccrit.apply(cp, GL)
            boot = sum(np.asarray(jnp.sum(jax.nn.softmax(lg_last[i][0::2], -1) * ql[i], -1)) for i in range(3))
            _, g_s = gae_raw(r_s, q_taken * Q_SCALE, boot * Q_SCALE)
            qg = np.zeros((Tn, N, 3, V), np.float32); qg[:, 0::2] = np.stack(qs, -2)
            gg = np.zeros((Tn, N), np.float32); gg[:, 0::2] = g_s
            out["qg"] = qg.reshape(M, 3, V); out["g_s"] = gg.reshape(M)
        return out

    @jax.jit
    def crit_step(cp, cst, gb, tokb, tgt, snd, mu, sig):
        def loss(cp_):
            if a.credit == "Ag":
                e = ccrit.apply(cp_, gb) - (tgt - mu) / sig
            else:
                q = ccrit.apply(cp_, gb)
                e = sum(jnp.take_along_axis(q[i], tokb[:, i:i + 1], -1)[:, 0] for i in range(3)) - tgt / Q_SCALE
            return (huber(e) * snd).sum() / (snd.sum() + 1e-8)
        l, g = jax.value_and_grad(loss)(cp)
        upd, cst = copt.update(g, cst, cp)
        return optax.apply_updates(cp, upd), cst, l

    central_last = {}

    def credit_streams(R, o_last, params, M, rewards_override=None, mutant=None):
        """Flat per-row streams for credit A / B. rewards_override replaces the TOTAL rewards (test: the token streams
        must not change when only the sender's own rewards change). mutant (tests only): "full_adv" -> A's token
        advantage from the total reward; "total_reward" -> B's Q target from the total reward."""
        rew = R["rewards"] if rewards_override is None else rewards_override
        if CENTRAL:
            return central_streams(R, o_last, params, M, rew, mutant)
        out = {"lp_act": R["lp_act"].reshape(M).astype(np.float32),
               "lp_tok_old": (R["log_probs"] - R["lp_act"]).reshape(M).astype(np.float32)}
        snd_rows = np.broadcast_to(~lis, R["r_shared"].shape)
        r_stream = rew if mutant in ("full_adv", "total_reward") else R["r_shared"]
        if a.credit == "A":
            vs_den = popart_s.denormalise(R["crit"])
            vs_last = popart_s.denormalise(np.asarray(o_last.values_shared))
            adv_s, ret_s = gae_raw(r_stream, vs_den, vs_last)
            m_, sd_ = adv_s[snd_rows].mean(), adv_s[snd_rows].std() + 1e-8
            out["adv_tok"] = np.where(snd_rows, np.clip((adv_s - m_) / sd_, -10, 10), 0.0).reshape(M).astype(np.float32)
            out["ret_s"] = ret_s.reshape(M).astype(np.float32)
            out["_ret_s_full"] = ret_s
        else:
            lg_last = token_logits(params, o_last.z_e, TAU)
            boot = sum(np.asarray(jnp.sum(jax.nn.softmax(lg_last[i], -1) * o_last.q_slots[i], -1)) for i in range(3))
            _, g_s = gae_raw(r_stream, R["crit"] * Q_SCALE, boot * Q_SCALE)
            out["g_s"] = g_s.reshape(M).astype(np.float32)
        return out

    def credit_selftest(params, flat, R, o_last, M):
        """Gradient tests + mutants for credit A / B (Cam 2026-11-08), on the first minibatch of update 1."""
        idx = np.arange(MB)
        mb = {k: jnp.asarray(v[idx]) for k, v in flat.items()}
        gn = lambda g, keys: float(np.sqrt(sum(float(jnp.sum(x ** 2)) for k in keys  # noqa: E731
                                               for x in jax.tree_util.tree_leaves(g["params"][k]))))
        MSG = ["gwt_comms_1", "head_signal_slot0", "head_signal_slot1", "head_signal_slot2"]
        CRIT = ["head_value_shared"] if a.credit == "A" else [f"head_q_slot{i}" for i in range(3)]
        ok = True
        g_tok = jax.grad(lambda p_: credit_terms(p_, mb)[0])(params)
        g_crit = jax.grad(lambda p_: credit_terms(p_, mb)[2])(params)
        g_mov = jax.grad(lambda p_: ppo_loss(p_, apply_fn, mb["obs"], mb["actions"], None, mb["lp_act"], mb["adv"],
                         mb["ret"], mb["carries"], 1, mb["values"], clip_eps=CLIP, vf_coef=VF, ent_coef=ENT, vq_coef=0.0,
                         value_scale=popart.sigma, value_clip=False, value_shift=popart.mu, popart_mode=True,
                         token_actions=None)[0])(params)
        res = dict(credit=a.credit,
                   tok_grad_message_heads=gn(g_tok, MSG), tok_grad_action_value_heads=gn(g_tok, ["head_action", "head_value"]),
                   tok_grad_critic=gn(g_tok, CRIT), crit_grad_critic=gn(g_crit, CRIT), crit_grad_message=gn(g_crit, MSG),
                   move_grad_message=gn(g_mov, MSG), move_grad_critic=gn(g_mov, CRIT))
        ok &= res["tok_grad_message_heads"] > 0 and res["tok_grad_action_value_heads"] == 0 and res["tok_grad_critic"] == 0
        ok &= res["crit_grad_critic"] > 0 and res["crit_grad_message"] == 0
        ok &= res["move_grad_message"] == 0 and res["move_grad_critic"] == 0
        # own-reward independence of the token stream, and its mutant
        rng_t = np.random.default_rng(0)
        own_noise = rng_t.normal(size=R["rewards"].shape).astype(np.float32) * (~lis)[None, :]
        alt = credit_streams(R, o_last, params, M, rewards_override=R["rewards"] + own_noise)
        base_ = credit_streams(R, o_last, params, M)
        key_ = "adv_tok" if a.credit == "A" else "g_s"
        res["stream_unchanged_by_own_reward"] = bool(np.array_equal(alt[key_], base_[key_]))
        mut = "full_adv" if a.credit == "A" else "total_reward"
        m1 = credit_streams(R, o_last, params, M, rewards_override=R["rewards"], mutant=mut)
        m2 = credit_streams(R, o_last, params, M, rewards_override=R["rewards"] + own_noise, mutant=mut)
        res["mutant_caught"] = bool(not np.array_equal(m1[key_], m2[key_]))
        ok &= res["stream_unchanged_by_own_reward"] and res["mutant_caught"]
        if a.credit == "B":   # counterfactual baseline: E_pi[A_i] = 0 per row and slot; mutant without baseline
            _, o = apply_fn(params, mb["carries"], mb["obs"])
            lgs = token_logits(params, o.z_e, TAU)
            ev, ev_mut = 0.0, 0.0
            for i in range(3):
                pi = jax.nn.softmax(lgs[i], -1); q = o.q_slots[i]
                Aall = q - jnp.sum(pi * q, -1, keepdims=True)
                ev = max(ev, float(jnp.max(jnp.abs(jnp.sum(pi * Aall, -1)))))
                ev_mut = max(ev_mut, float(jnp.max(jnp.abs(jnp.sum(pi * q, -1)))))
            res["baseline_expectation_max"] = ev; res["no_baseline_mutant_expectation"] = ev_mut
            ok &= ev < 1e-5 and ev_mut > 1e-3
        print(json.dumps(res), flush=True)
        print("SELFTEST-CREDIT", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)

    # ---- T2 / T3 generational turnover
    reset_sched = {}
    if a.reset_from is not None:
        for j, uu in enumerate(range(a.reset_from, a.reset_until + 1, a.reset_every)):
            reset_sched[uu] = "sender" if j % 2 == 0 else "listener"
    turn_state = {"maps": []}
    imit_buf = deque(maxlen=a.imitate_window)

    def code_map(tok, lab):
        """per-slot decoder token -> fire (argmax of co-occurrence counts); unseen tokens -> -1 (a miss)"""
        maps = []
        for i in range(3):
            cnt = np.zeros((V, 4)); np.add.at(cnt, (tok[:, i], lab), 1)
            mp = np.where(cnt.sum(1) > 0, cnt.argmax(1), -1); maps.append(mp)
        return maps

    def turnover_record(u):
        tok, lab = np.concatenate(list(mi_buf_tok)), np.concatenate(list(mi_buf_T))
        cur = code_map(tok, lab)
        rec = dict(phase="turnover", update=u, mi_plugin=mi_plugin(tok, lab),
                   next_reset=reset_sched.get(u), self_decode=float(np.mean([np.mean(cur[i][tok[:, i]] == lab) for i in range(3)])))
        if turn_state["maps"]:
            prev = turn_state["maps"][-1][1]
            per = [float(np.mean(prev[i][tok[:, i]] == lab)) for i in range(3)]
            rec.update(continuity=float(np.mean(per)), continuity_slots=per, continuity_vs_update=turn_state["maps"][-1][0])
        turn_state["maps"].append((u, cur))
        emit(rec)

    def do_reset(kind, params, opt_state, u):
        fresh = model.init(jax.random.fold_in(jax.random.PRNGKey(4242 + a.seed), u), jnp.zeros((1, H)), jnp.zeros((1, D)))["params"]
        pp = {k: dict(v) if isinstance(v, dict) else v for k, v in params["params"].items()}
        adam = opt_state[0]
        mu = {k: dict(v) if isinstance(v, dict) else v for k, v in adam.mu["params"].items()}
        nu = {k: dict(v) if isinstance(v, dict) else v for k, v in adam.nu["params"].items()}
        if kind == "sender":       # the message head: gwt_comms_1 + head_signal_slot* (codebooks kept)
            for hk in HEADS:
                pp[hk] = fresh[hk]
                mu[hk] = jax.tree_util.tree_map(jnp.zeros_like, mu[hk]); nu[hk] = jax.tree_util.tree_map(jnp.zeros_like, nu[hk])
        else:                      # the listener's wire-input layer: enc_0 rows reading the wire (same 0.06 init scale)
            k0 = jnp.asarray(pp["enc_0"]["kernel"])
            pp["enc_0"] = dict(pp["enc_0"], kernel=k0.at[O_WIRE:].set(jnp.asarray(fresh["enc_0"]["kernel"])[O_WIRE:] * a.wire_init_scale))
            for st in (mu, nu):
                st["enc_0"] = dict(st["enc_0"], kernel=jnp.asarray(st["enc_0"]["kernel"]).at[O_WIRE:].set(0.0))
        params = {"params": pp}
        opt_state = (adam._replace(mu={"params": mu}, nu={"params": nu}),) + tuple(opt_state[1:])
        info = dict(phase="reset", update=u, kind=kind)
        if kind == "sender" and a.imitate and imit_buf:
            params, acc0, acc1 = imitate(params)
            info.update(imitation_acc_before=acc0, imitation_acc_after=acc1, imitation_rows=int(sum(len(b[1]) for b in imit_buf)))
        emit(info)
        return params, opt_state

    def imitate(params):
        """T3 newborn imitation: supervised cross-entropy of the new message head on the previous head's (obs, token)
        pairs from its last imitate_window updates; only the message head moves; separate Adam."""
        X = np.concatenate([b[0] for b in imit_buf]); Y = np.concatenate([b[1] for b in imit_buf])
        head = {k: params["params"][k] for k in HEADS}
        iopt = optax.adam(a.imitate_lr); ist = iopt.init(head)

        def ce(hd, xb, yb):
            pp_ = dict(params["params"]); pp_.update(hd)
            _, o = apply_fn({"params": pp_}, jnp.zeros((xb.shape[0], H)), xb)
            lsm = [jax.nn.log_softmax(lg, -1) for lg in token_logits({"params": pp_}, o.z_e, TAU)]
            return -jnp.mean(sum(jnp.take_along_axis(lsm[i], yb[:, i:i + 1], -1)[:, 0] for i in range(3)))

        @jax.jit
        def step(hd, st, xb, yb):
            g = jax.grad(ce)(hd, xb, yb)
            upd, st = iopt.update(g, st, hd)
            return optax.apply_updates(hd, upd), st

        def acc(hd):
            ix = np.random.default_rng(0).integers(0, len(X), 4096)
            pp_ = dict(params["params"]); pp_.update(hd)
            _, o = apply_fn({"params": pp_}, jnp.zeros((4096, H)), jnp.asarray(X[ix]))
            pred = np.stack([np.asarray(jnp.argmax(lg, -1)) for lg in token_logits({"params": pp_}, o.z_e, TAU)], -1)
            return float((pred == Y[ix]).all(1).mean())
        a0 = acc(head)
        rng_i = np.random.default_rng(777)
        for _ in range(a.imitate_steps):
            ix = rng_i.integers(0, len(X), 512)
            head, ist = step(head, ist, jnp.asarray(X[ix]), jnp.asarray(Y[ix]))
        a1 = acc(head)
        pp = dict(params["params"]); pp.update(head)
        return {"params": pp}, a0, a1

    def central_selftest(params, flat, R, o_last, M):
        """T4 gradient tests + mutants (Cam 2026-11-08), first minibatch of update 1."""
        ok = True
        res = dict(credit=a.credit)
        # actors unchanged: the actor param tree is exactly the joint (baseline) network's
        base_keys = set(ToyNet(H).init(jax.random.PRNGKey(0), jnp.zeros((1, H)), jnp.zeros((1, D)))["params"])
        res["actor_tree_unchanged"] = set(params["params"]) == base_keys
        ok &= res["actor_tree_unchanged"]
        mb = {k: jnp.asarray(v[np.arange(MB)]) for k, v in flat.items()}
        MSG = ["gwt_comms_1", "head_signal_slot0", "head_signal_slot1", "head_signal_slot2"]
        gn = lambda g, keys: float(np.sqrt(sum(float(jnp.sum(x ** 2)) for k in keys  # noqa: E731
                                               for x in jax.tree_util.tree_leaves(g["params"][k]))))
        g_tok = jax.grad(lambda p_: credit_terms(p_, mb)[0])(params)
        res["tok_grad_message_heads"] = gn(g_tok, MSG); res["tok_grad_action_value"] = gn(g_tok, ["head_action", "head_value"])
        ok &= res["tok_grad_message_heads"] > 0 and res["tok_grad_action_value"] == 0
        # the critic learns from its own loss
        snd_m = (mb["obs"][:, O_INF] > 0.5).astype(jnp.float32)
        tgt = mb["ret_g"] if a.credit == "Ag" else mb["g_s"]
        _, _, l0 = crit_step(cparams, cstate, mb["gfeat"], mb["token_actions"], tgt, snd_m, popart_g.mu, popart_g.sigma)
        res["critic_loss_finite"] = bool(np.isfinite(float(l0)))
        # the critic uses GLOBAL inputs: perturbing only the listener/fire part of g changes its output; the local
        # (listener-blind) mutant does not react
        G = gfeat(jnp.asarray(R["obs"][:4]), jnp.asarray(R["carries"][:4]), jnp.asarray(R["active"][:4]))
        noise = jax.random.normal(jax.random.PRNGKey(5), G.shape) * jnp.concatenate([jnp.zeros(D), jnp.ones(D + H + 4)])
        out = lambda g_: ccrit.apply(cparams, g_)  # noqa: E731
        diff = lambda a_, b_: float(jnp.max(jnp.abs(a_ - b_))) if a.credit == "Ag" else max(  # noqa: E731
            float(jnp.max(jnp.abs(x - y))) for x, y in zip(a_, b_))
        res["critic_reacts_to_listener_inputs"] = diff(out(G), out(G + noise))
        mask = jnp.concatenate([jnp.ones(D), jnp.zeros(D + H + 4)])
        res["local_mutant_reacts"] = diff(out(G * mask), out((G + noise) * mask))
        ok &= res["critic_reacts_to_listener_inputs"] > 1e-6 and res["local_mutant_reacts"] == 0.0
        # token stream independent of the sender's own reward; the full-reward mutant is caught
        own_noise = np.random.default_rng(0).normal(size=R["rewards"].shape).astype(np.float32) * (~lis)[None, :]
        key_ = "adv_tok" if a.credit == "Ag" else "g_s"
        b0 = central_streams(R, o_last, params, M, R["rewards"], None)[key_]
        b1 = central_streams(R, o_last, params, M, R["rewards"] + own_noise, None)[key_]
        m1 = central_streams(R, o_last, params, M, R["rewards"], "full_adv")[key_]
        m2 = central_streams(R, o_last, params, M, R["rewards"] + own_noise, "full_adv")[key_]
        res["stream_unchanged_by_own_reward"] = bool(np.array_equal(b0, b1)); res["mutant_caught"] = bool(not np.array_equal(m1, m2))
        ok &= res["stream_unchanged_by_own_reward"] and res["mutant_caught"]
        if a.credit == "Bg":
            q = mb["qg"]; _, o = apply_fn(params, mb["carries"], mb["obs"]); lgs = token_logits(params, o.z_e, TAU)
            ev = max(float(jnp.max(jnp.abs(jnp.sum(jax.nn.softmax(lgs[i], -1) * (q[:, i] - jnp.sum(jax.nn.softmax(lgs[i], -1) * q[:, i], -1, keepdims=True)), -1)))) for i in range(3))
            evm = max(float(jnp.max(jnp.abs(jnp.sum(jax.nn.softmax(lgs[i], -1) * q[:, i], -1) * snd_m))) for i in range(3))
            res["baseline_expectation_max"] = ev; res["no_baseline_mutant_expectation"] = evm
            ok &= ev < 1e-5 and evm > 1e-6
        print(json.dumps(res), flush=True)
        print("SELFTEST-CENTRAL", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)

    def gae_raw(r, v, v_last):
        # GAE(lambda) with bootstrap, no normalisation (the credit streams; same recursion as gae_bootstrap)
        r = np.asarray(r, np.float64); v = np.asarray(v, np.float64)
        adv_ = np.zeros_like(r); c = np.zeros(r.shape[1]); nv = np.asarray(v_last, np.float64)
        for t in range(r.shape[0] - 1, -1, -1):
            c = r[t] + GAMMA * nv - v[t] + GAMMA * LAM * c
            adv_[t] = c; nv = v[t]
        return adv_, adv_ + v

    if a.qcheck:   # T4 free check (b)/(c): Q over all tokens on ~500 sampled sender states at the saved params
        assert a.credit == "Bg"
        params = flax.serialization.from_bytes(params, Path(f"{a.qcheck}_actor.msgpack").read_bytes())
        cparams = flax.serialization.from_bytes(cparams, Path(f"{a.qcheck}_critic.msgpack").read_bytes())
        lisq = np.asarray(is_listener)
        Gs, toks = [], []
        for k in range(2):
            st0 = init_state(jax.random.PRNGKey(70_000 + k))
            stq, Rq = rollout(params, st0, jax.random.PRNGKey(80_000 + k))
            Rq = jax.tree_util.tree_map(np.asarray, Rq)
            Gs.append(np.asarray(gfeat(jnp.asarray(Rq["obs"]), jnp.asarray(Rq["carries"]), jnp.asarray(Rq["active"]))).reshape(-1, DG))
            toks.append(Rq["token_actions"][:, ~lisq].reshape(-1, 3))
        Gs, toks = np.concatenate(Gs), np.concatenate(toks)
        ix = np.random.default_rng(1).choice(len(Gs), 500, replace=False)
        q = [np.asarray(x) for x in ccrit.apply(cparams, jnp.asarray(Gs[ix]))]                       # 3 x (500, 64)
        sent = toks[ix]
        top = float(np.mean([np.mean(np.argmax(q[i], -1) == sent[:, i]) for i in range(3)]))
        spread = float(np.mean([np.std(q[i], -1).mean() for i in range(3)]))
        # sent token's counterfactual advantage under the current policy (raw, units of one delivery), sum over slots
        obs_s = np.zeros((500, D), np.float32)
        obs_s[:] = Gs[ix][:, :D]
        _, o = apply_fn(params, jnp.zeros((500, H)), jnp.asarray(obs_s))   # message pathway is feed-forward from obs
        pis = [np.asarray(jax.nn.softmax(lg, -1)) for lg in token_logits(params, o.z_e, TAU)]
        adv = float(np.mean(sum(np.take_along_axis(q[i], sent[:, i:i + 1], -1)[:, 0] - (pis[i] * q[i]).sum(-1) for i in range(3))))
        print(json.dumps(dict(phase="qcheck", prefix=a.qcheck, arm=a.arm, sent_top_share=top, q_spread=spread,
                              mean_sent_adv=adv, n=500)), flush=True)
        raise SystemExit(0)

    if a.selftest_sg2:
        """Cam Nov 9 apparatus check for --msg-grad sender, on an update-1 minibatch (the training batch recipe):
        (1) the listener rows' contribution to the message-head gradient is exactly 0: perturbing every listener row's
            advantage / return / actions / log-prob / value leaves the MSGK gradient bit-identical; flag-off and a mutant
            that leaks ONE listener row into the sender set must change it;
        (2) the PS term's gradient on MSGK (gradient at c_ps minus at c_ps = 0) is allclose to flag-off (rtol 1e-4,
            atol 1e-6 x max|g|); the old scaled-PS composition must FAIL that check."""
        assert a.ps > 0 and a.ps_anneal, "run with --ps and --ps-anneal (the protocol config; c_ps is then honoured)"
        _, Rq = rollout(params, init_state(jax.random.PRNGKey(41)), jax.random.PRNGKey(42))
        Rq = jax.tree_util.tree_map(np.asarray, Rq)
        vals = popart.denormalise(Rq["values"])
        advq, retq = gae_bootstrap(Rq["rewards"], vals, np.zeros_like(Rq["rewards"]), vals[-1], GAMMA, LAM)
        params = popart.update_and_preserve(retq, params)
        M = T * N
        flat = {k: Rq[k].reshape((M,) + Rq[k].shape[2:]) for k in ("obs", "actions", "log_probs", "carries", "token_actions")}
        flat["values"] = vals.reshape(M).astype(np.float32)
        flat["adv"], flat["ret"] = advq.reshape(M).astype(np.float32), retq.reshape(M).astype(np.float32)
        idx = np.random.default_rng(43).permutation(M)[:MB]
        mb = {k: jnp.asarray(v[idx]) for k, v in flat.items()}
        lisr = np.asarray(mb["obs"][:, O_INF] < 0.5)
        r_ = np.random.default_rng(44)
        mb_p = dict(mb)
        for k in ("adv", "ret", "values", "log_probs"):
            v = np.array(mb[k]); v[lisr] = r_.normal(size=lisr.sum()).astype(v.dtype) * 5; mb_p[k] = jnp.asarray(v)
        v = np.array(mb["actions"]); v[lisr] = r_.integers(0, 5, lisr.sum()); mb_p["actions"] = jnp.asarray(v)
        v = np.array(mb["token_actions"]); v[lisr] = r_.integers(0, V, (lisr.sum(), 3)); mb_p["token_actions"] = jnp.asarray(v)
        args = (popart.sigma, popart.mu, a.ps, TOK_ENT)
        vec = lambda g: np.concatenate([np.asarray(x, np.float64).ravel() for k in MSGK for x in jax.tree_util.tree_leaves(g["params"][k])])  # noqa: E731
        full = lambda m_, ps_: jax.grad(lambda p_: total_loss(p_, m_, popart.sigma, popart.mu, ps_, TOK_ENT, ps_force=ps_)[0])(params)  # noqa: E731
        res = {}
        g1, g1p = vec(sg2_grads(params, mb, *args)), vec(sg2_grads(params, mb_p, *args))
        res["listener_perturbed_identical"] = bool(np.array_equal(g1, g1p))
        res["flag_off_changes"] = bool(not np.array_equal(vec(full(mb, a.ps)), vec(full(mb_p, a.ps))))
        lr0 = int(np.where(lisr)[0][0])
        gm, gmp = vec(sg2_grads(params, mb, *args, leak_row=lr0)), vec(sg2_grads(params, mb_p, *args, leak_row=lr0))
        res["leak_one_row_mutant_changes"] = bool(not np.array_equal(gm, gmp))
        ps_ref = vec(full(mb, a.ps)) - vec(full(mb, 0.0))
        ps_new = vec(sg2_grads(params, mb, *args)) - vec(sg2_grads(params, mb, popart.sigma, popart.mu, 0.0, TOK_ENT))
        ps_old = (vec(sg2_grads(params, mb, *args, version="old")) -
                  vec(sg2_grads(params, mb, popart.sigma, popart.mu, 0.0, TOK_ENT, version="old")))
        atol = 1e-6 * float(np.abs(ps_ref).max())
        res["ps_grad_allclose_fixed"] = bool(np.allclose(ps_new, ps_ref, rtol=1e-4, atol=atol))
        res["ps_grad_allclose_old"] = bool(np.allclose(ps_old, ps_ref, rtol=1e-4, atol=atol))
        res["ps_norm_ratio_fixed"] = float(np.linalg.norm(ps_new) / np.linalg.norm(ps_ref))
        res["ps_norm_ratio_old"] = float(np.linalg.norm(ps_old) / np.linalg.norm(ps_ref))
        res["sender_row_fraction"] = float(1 - lisr.mean())
        ok = (res["listener_perturbed_identical"] and res["flag_off_changes"] and res["leak_one_row_mutant_changes"]
              and res["ps_grad_allclose_fixed"] and not res["ps_grad_allclose_old"])
        res["PASS"] = bool(ok)
        print(json.dumps(res, indent=1), flush=True)
        raise SystemExit(0 if ok else 1)

    if a.selftest_leak:
        """Step A: gradient of the LISTENER's losses (token PPO term zeroed) w.r.t. the shared message parameters,
        on one wire-on rollout batch, with the listener's heard wire as recorded vs zeroed (same batch otherwise).
        A nonzero difference = the wire reaches the message parameters through the listener's losses. Also: the
        listener's own token policy-gradient path, and the forward coupling (sender tokens vs heard wire)."""
        assert a.arm == "COMM"
        lisq = np.asarray(is_listener)
        _, Rq = rollout(params, init_state(jax.random.PRNGKey(31)), jax.random.PRNGKey(32))
        Rq = jax.tree_util.tree_map(np.asarray, Rq)
        vals = popart.denormalise(Rq["values"])
        advq, _ = gae_bootstrap(Rq["rewards"], vals, np.zeros_like(Rq["rewards"]), vals[-1], GAMMA, LAM)
        Lrows = lambda x: x[:, lisq].reshape((-1,) + x.shape[2:])  # noqa: E731
        obsL, carL, tokL, advL = (jnp.asarray(Lrows(Rq["obs"])), jnp.asarray(Lrows(Rq["carries"])),
                                  jnp.asarray(Lrows(Rq["token_actions"])), jnp.asarray(Lrows(advq)))
        obsL0 = obsL.at[:, O_WIRE:].set(0.0)
        MSG = ["gwt_comms_1", "head_signal_slot0", "head_signal_slot1", "head_signal_slot2", "codebook_0", "codebook_1", "codebook_2"]

        def parts(p_, o_in):
            _, o = apply_fn(p_, carL, o_in)
            lsm = [jax.nn.log_softmax(lg, -1) for lg in token_logits(p_, o.z_e, TAU)]
            lp = sum(jnp.take_along_axis(lsm[i], tokL[:, i:i + 1], -1)[:, 0] for i in range(3))
            return dict(vq=VQC * jnp.mean(o.loss_vq.sum(-1)),
                        tok_ent=-TOK_ENT * jnp.mean(sum(-jnp.sum(jnp.exp(l_) * l_, -1) for l_ in lsm)),
                        tok_pg_listener=-jnp.mean(advL * lp),
                        move=jnp.mean(-jax.nn.log_softmax(mask_disabled_actions(o.action_logits, -1), -1)[:, 0] * advL)
                        + VF * jnp.mean(o.values ** 2))

        def gvec(term, o_in):
            g = jax.grad(lambda p_: parts(p_, o_in)[term])(params)
            return np.concatenate([np.asarray(x).ravel() for k in MSG for x in jax.tree_util.tree_leaves(g["params"][k])])
        res = {}
        for term in ("vq", "tok_ent", "tok_pg_listener", "move"):
            g1, g0 = gvec(term, obsL), gvec(term, obsL0)
            res[term] = dict(grad_norm_wire=float(np.linalg.norm(g1)), grad_norm_wire_zeroed=float(np.linalg.norm(g0)),
                             diff_norm=float(np.linalg.norm(g1 - g0)))
        # codebook specifically (the codebook vectors ARE the wire content)
        def gcb(term, o_in):
            g = jax.grad(lambda p_: parts(p_, o_in)[term])(params)
            return float(np.sqrt(sum(float(jnp.sum(x ** 2)) for k in ("codebook_0", "codebook_1", "codebook_2")
                                     for x in jax.tree_util.tree_leaves(g["params"][k]))))
        res["vq_codebook_grad_norm"] = gcb("vq", obsL)
        # forward coupling: the sender's token distribution vs the heard wire (recorded vs zeroed vs random tokens)
        obsS = jnp.asarray(Rq["obs"][:, ~lisq].reshape(-1, D))
        cb = [params["params"][k]["embedding"] for _, _, k in rl_jax.TOKEN_SLOTS]
        rt = jax.random.randint(jax.random.PRNGKey(3), (obsS.shape[0], 3), 0, V)
        rbc = jnp.concatenate([jnp.zeros((obsS.shape[0], 8))] + [cb[i][rt[:, i]] for i in range(3)], -1)
        def tokdist(o_in):
            _, o = apply_fn(params, jnp.zeros((o_in.shape[0], H)), o_in)
            return [jax.nn.softmax(lg, -1) for lg in token_logits(params, o.z_e, TAU)]
        p_rec, p_zero, p_rand = tokdist(obsS), tokdist(obsS.at[:, O_WIRE:].set(0.0)), tokdist(obsS.at[:, O_WIRE:].set(rbc))
        tv = lambda P, Q: float(np.mean([0.5 * np.abs(np.asarray(P[i]) - np.asarray(Q[i])).sum(-1).mean() for i in range(3)]))  # noqa: E731
        am = lambda P, Q: float(np.mean([np.mean(np.argmax(np.asarray(P[i]), -1) != np.argmax(np.asarray(Q[i]), -1)) for i in range(3)]))  # noqa: E731
        res["forward_sender_token_TV_recorded_vs_zeroed_wire"] = tv(p_rec, p_zero)
        res["forward_sender_token_TV_recorded_vs_random_wire"] = tv(p_rec, p_rand)
        res["forward_sender_argmax_changed_frac_random_wire"] = am(p_rec, p_rand)
        res["shared_parameters_sender_listener"] = "all (one ToyNet; the role is set only by the informed flag in obs)"
        print(json.dumps(res, indent=1), flush=True)
        raise SystemExit(0)

    if a.grad_share:
        """Cam Nov 9 item 4 (free, eval-only): the listener rows' share of the message-head gradient. On fresh rollouts
        at the loaded params, the first-epoch training gradient (total_loss, ratio = 1, mean over every minibatch of a
        random permutation, pre-clip) is split exactly by rows: every term of total_loss is a row-masked mean, so
        g_all = f_S g_S + f_L g_L with g_X = the gradient with alive = X's rows and f_X = X's row fraction (the PS term is
        not row-masked; c_ps = 0 at u >= 1,500 under the erosion protocol, so it is absent). Message head = MSGK (the
        step-B (iii) stop-grad key set). PopArt stats: exact from --grad-share-snap (a full-state snapshot); otherwise
        (actor-only saves) fitted by OLS of the GAE returns on the raw value-head output, iterated 5x (disclosed)."""
        lisg = np.asarray(is_listener)
        if a.grad_share_snap:
            import pickle
            snap = pickle.load(open(a.grad_share_snap, "rb"))
            params = jax.tree_util.tree_map(jnp.asarray, snap["params"])
            popart.mu, popart.nu, popart.initialised = snap["popart"]
            gs_u, pa_src = snap["u"], "snapshot (exact)"
        else:
            gs_u, pa_src = a.grad_share_u, "OLS fit (approximate)"
        ps_g = a.ps
        if a.ps_anneal:
            s0_, e0_ = a.ps_anneal
            ps_g = a.ps * float(np.clip((e0_ - gs_u) / (e0_ - s0_), 0.0, 1.0)) if gs_u > s0_ else a.ps
        te_g = a.erode_tok_ent if (a.erode_tok_ent is not None and gs_u >= a.erode_from) else TOK_ENT
        assert ps_g == 0.0, "the PS term is not row-masked; the split is exact only with c_ps = 0"
        sel = lambda g: np.concatenate([np.asarray(x, np.float64).ravel() for k in MSGK  # noqa: E731
                                        for x in jax.tree_util.tree_leaves(g["params"][k])])
        sel_h = lambda g: np.concatenate([np.asarray(x, np.float64).ravel() for k in MSGK if not k.startswith("codebook")  # noqa: E731
                                          for x in jax.tree_util.tree_leaves(g["params"][k])])
        gfn = jax.jit(jax.grad(lambda p_, mb, sg, mu: total_loss(p_, mb, sg, mu, ps_g, te_g)[0]))
        acc = {"all": 0.0, "S": 0.0, "L": 0.0}; acc_h = {"all": 0.0, "S": 0.0, "L": 0.0}; nmb = 0
        st_g = init_state(jax.random.PRNGKey(60_000 + gs_u))
        for rr in range(a.grad_share_r0, a.grad_share_r0 + a.grad_share_rollouts):
            st_g, R = rollout(params, st_g, jax.random.PRNGKey(61_000 + 7 * gs_u + rr))
            R = jax.tree_util.tree_map(np.asarray, R)
            obs_last = build_obs(st_g[0], st_g[1], st_g[3], jax.random.PRNGKey(62_000 + rr))
            _, o_last = apply_fn(params, st_g[4], obs_last); vh_last = np.asarray(o_last.values)
            if not a.grad_share_snap:          # fit (mu, sigma): returns ~ sigma * v_hat + mu
                mu_, sg_ = float(R["rewards"].mean() / (1 - GAMMA)), 1.0
                for _ in range(5):
                    vals = R["values"] * sg_ + mu_
                    _, ret_ = gae_bootstrap(R["rewards"], vals, np.zeros_like(R["rewards"]), vh_last * sg_ + mu_, GAMMA, LAM)
                    sg_, mu_ = np.polyfit(R["values"].ravel(), ret_.ravel(), 1)
                    sg_ = float(max(sg_, 1e-4)); mu_ = float(mu_)
                popart.mu, popart.nu, popart.initialised = mu_, sg_ ** 2 + mu_ ** 2, True
            values = popart.denormalise(R["values"])
            adv, ret = gae_bootstrap(R["rewards"], values, np.zeros_like(R["rewards"]), popart.denormalise(vh_last), GAMMA, LAM)
            M = T * N
            flat = {k: R[k].reshape((M,) + R[k].shape[2:]) for k in ("obs", "actions", "log_probs", "carries", "token_actions")}
            flat["values"] = values.reshape(M).astype(np.float32)
            flat["adv"], flat["ret"] = adv.reshape(M).astype(np.float32), ret.reshape(M).astype(np.float32)
            snd_row = np.broadcast_to(~lisg, (T, N)).reshape(M)
            perm = np.random.default_rng(63_000 + rr).permutation(M)
            for i in range(M // MB):
                idx = perm[i * MB:(i + 1) * MB]
                mb = {k: jnp.asarray(v[idx]) for k, v in flat.items()}
                fS = float(snd_row[idx].mean())
                for nm, msk, f in (("all", None, 1.0), ("S", snd_row[idx], fS), ("L", ~snd_row[idx], 1.0 - fS)):
                    g = gfn(params, mb if msk is None else dict(mb, alive=jnp.asarray(msk, jnp.float32)), popart.sigma, popart.mu)
                    acc[nm] = acc[nm] + f * sel(g); acc_h[nm] = acc_h[nm] + f * sel_h(g)
                nmb += 1
        out = dict(phase="grad_share", update=gs_u, popart=pa_src, popart_mu=popart.mu, popart_sigma=popart.sigma,
                   rollouts=a.grad_share_rollouts, minibatches=nmb, c_ps=ps_g)
        for lab, A_ in (("msgk", acc), ("head_no_codebook", acc_h)):
            gA, cS, cL = (A_[k] / nmb for k in ("all", "S", "L"))
            nS, nL = float(np.linalg.norm(cS)), float(np.linalg.norm(cL))
            out[lab] = dict(norm_all=float(np.linalg.norm(gA)), norm_S=nS, norm_L=nL,
                            share_L=nL / (nS + nL + 1e-30), share_L_sq=nL ** 2 / (nS ** 2 + nL ** 2 + 1e-30),
                            cos_LS=float(np.dot(cL, cS) / (nS * nL + 1e-30)),
                            proj_share_L=float(np.dot(cL, gA) / (np.dot(gA, gA) + 1e-30)),
                            split_residual=float(np.linalg.norm(gA - cS - cL) / (np.linalg.norm(gA) + 1e-30)))
        if a.grad_share_save:
            np.savez(a.grad_share_save, **{k: acc[k] / nmb for k in ("all", "S", "L")})
        emit(out)
        raise SystemExit(0)

    if a.selftest_lookout:
        """step 3b apparatus: (1) geometry -- 4-neighbour steps to Chebyshev <= 1 of each fire, from the lookout centre and
        from the cluster centre; (2) on a rollout, the informed bit and the need dims are nonzero exactly where the
        lookout predicate (recomputed here from positions) says so -- self at the lookout, or (VOI oracle) self or
        partner. The mutant (everyone at the lookout) must FAIL."""
        assert LOOK is not None
        hn, L = np.asarray(HEARTHS), np.array(LOOK[:2])

        def steps(p_, h_):
            d = np.abs(p_ - h_); d = np.minimum(d, G - d); return int(max(d[0] - 1, 0) + max(d[1] - 1, 0))
        ctr = np.rint(hn.mean(0)).astype(int)
        geo = dict(lookout_to_fire=[steps(L, h) for h in hn], centre=ctr.tolist(), centre_to_fire=[steps(ctr, h) for h in hn])
        _, Rc = rollout(params, init_state(jax.random.PRNGKey(5)), jax.random.PRNGKey(6))
        ob = np.asarray(Rc["obs"])                                            # (T, N, D)
        pos = np.rint(ob[..., O_POS:O_POS + 2] * G).astype(int) % G
        dl = np.abs(pos - L); dl = np.minimum(dl, G - dl).max(-1); at = dl <= LOOK[2]
        want = (at | at.reshape(T, W, 2)[..., ::-1].reshape(T, N)) if a.lookout_oracle else at
        got = ob[..., O_INF] > 0.5
        need_nz = np.abs(ob[..., O_NEED:O_NEED + 4]).sum(-1) > 0
        res = dict(geo, agent_steps=int(at.size), at_lookout=int(at.sum()), informed=int(got.sum()), want_informed=int(want.sum()),
                   informed_match=bool((got == want).all()), need_match=bool((need_nz == want).all()))
        ok = res["informed_match"] and res["need_match"] and 0 < at.sum() < at.size
        print(json.dumps(res)); print("SELFTEST-LOOKOUT", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)
    if a.prox_world_check:
        """Cam 2026-11-08 (step 3): per fire period (active hearth until a delivery or a re-roll), who delivers, and
        whether the partner of the first-informed agent became informed on its own before the delivery."""
        assert a.prox is not None
        params = flax.serialization.from_bytes(params, Path(a.prox_world_check).read_bytes())
        n_ep = by_first = by_partner = partner_found = 0; delays = []
        for k in range(4):
            _, Rq = rollout(params, init_state(jax.random.PRNGKey(90_000 + k)), jax.random.PRNGKey(91_000 + k))
            Rq = jax.tree_util.tree_map(np.asarray, Rq)
            inf, dep, act = Rq["inf"], Rq["dep"], Rq["active"]
            for w in range(W):
                t0 = 0
                for t in range(1, T + 1):
                    end = t == T or dep[t - 1, 2 * w:2 * w + 2].any() or (t < T and act[t, w] != act[t - 1, w])
                    if not end:
                        continue
                    seg = slice(t0, t)
                    iA = np.nonzero(inf[seg, 2 * w])[0]; iB = np.nonzero(inf[seg, 2 * w + 1])[0]
                    d = dep[seg, 2 * w:2 * w + 2]
                    if d.any():   # a delivered period
                        n_ep += 1
                        tdel = np.nonzero(d.any(1))[0][0]; who = int(d[tdel].argmax())
                        firsts = [iA[0] if len(iA) else 10**9, iB[0] if len(iB) else 10**9]
                        first = int(np.argmin(firsts)); other = 1 - first
                        by_first += who == first; by_partner += who == other
                        partner_found += (firsts[other] <= tdel)
                        if firsts[first] < 10**9:
                            delays.append(tdel - firsts[first])
                    t0 = t
        print(json.dumps(dict(phase="prox_world_check", params=a.prox_world_check, delivered_periods=n_ep,
                              frac_delivered_by_first_informed=by_first / max(n_ep, 1),
                              frac_delivered_by_partner=by_partner / max(n_ep, 1),
                              frac_partner_informed_on_its_own_before_delivery=partner_found / max(n_ep, 1),
                              median_steps_first_informed_to_delivery=float(np.median(delays)) if delays else None)), flush=True)
        raise SystemExit(0)

    if a.selftest_comp:
        """step 6 apparatus: (1) every delivery is at the active fire with the target material's Craft action; (2) the
        material one-hot is present for informed rows only; (3) no held-out pair is ever active in training rollouts."""
        assert COMP
        lisq = np.asarray(is_listener); ok = True; res = {}
        _, Rq = rollout(params, init_state(jax.random.PRNGKey(71)), jax.random.PRNGKey(72))
        Rq = jax.tree_util.tree_map(np.asarray, Rq)
        act, dep, actv = Rq["actions"], Rq["dep"], np.repeat(Rq["active"], 2, axis=1)
        pos = Rq["obs"][:, :, O_POS:O_POS + 2] * G
        hf = np.asarray(HEARTHS)[actv // 4]
        dd = np.abs(np.rint(pos) - hf); dd = np.minimum(dd, G - dd).max(-1)
        ok_dep = bool(((act[dep] == np.asarray(CRAFT_ACTS)[actv[dep] % 4]) & (dd[dep] <= 1)).all())
        mat = Rq["obs"][:, :, O_NOISE:O_NOISE + 4]
        ok_mat = bool((mat[:, lisq] == 0).all() and np.allclose(mat[:, ~lisq].sum(-1), 1))
        ok_held = not np.isin(Rq["active"], HELD).any()
        res.update(deposits=int(dep.sum()), deposit_rule_ok=ok_dep, material_dims_ok=ok_mat, no_heldout_active=ok_held)
        ok = ok_dep and ok_mat and (ok_held or a.mutant_leak_heldout)
        print(json.dumps(res)); print("SELFTEST-COMP", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)

    if a.selftest_sym:
        """step 2 apparatus: (1) no probability on unreachable codewords; (2) a broadcast is a * one-hot of the symbol;
        (3) one training update leaves the codebooks bit-identical; (4) a mutant with all codewords equal (constant
        broadcast) must fail check (2)."""
        assert a.channel == "sym"
        lisq = np.asarray(is_listener)
        _, Rs = rollout(params, init_state(jax.random.PRNGKey(51)), jax.random.PRNGKey(52))
        Rs = jax.tree_util.tree_map(np.asarray, Rs)
        O = jnp.asarray(Rs["obs"].reshape(-1, D)[:4096]); C = jnp.asarray(Rs["carries"].reshape(-1, H)[:4096])
        _, o = apply_fn(params, C, O)
        probs = [np.asarray(jax.nn.softmax(lg, -1)) for lg in token_logits(params, o.z_e, TAU)]
        unreach = max(float(p_[:, a.sym_vocab:].sum(-1).max()) for p_ in probs)
        toks = Rs["token_actions"].reshape(-1, 3)
        def bcast(par, t):
            return np.concatenate([np.asarray(par["params"][k]["embedding"])[t[:, i]] for i, (_, _, k) in enumerate(rl_jax.TOKEN_SLOTS)], -1)
        want = np.concatenate([a.sym_scale * np.eye(a.sym_vocab, hi - lo)[toks[:, i]] for i, (lo, hi, _) in enumerate(rl_jax.TOKEN_SLOTS)], -1)
        ok_bc = bool(np.allclose(bcast(params, toks), want))
        mut = {"params": {k: (dict(v, embedding=jnp.ones_like(v["embedding"]) * 0.3) if k.startswith("codebook") else v)
                          for k, v in params["params"].items()}}
        mut_caught = not np.allclose(bcast(mut, toks), want)
        # one real update
        cb0 = {k: np.asarray(params["params"][k]["embedding"]) for _, _, k in rl_jax.TOKEN_SLOTS}
        vals = popart.denormalise(Rs["values"]); adv, ret = gae_bootstrap(Rs["rewards"], vals, np.zeros_like(Rs["rewards"]), vals[-1], GAMMA, LAM)
        Mq = T * N
        fl = {k: Rs[k].reshape((Mq,) + Rs[k].shape[2:]) for k in ("obs", "actions", "log_probs", "carries", "token_actions")}
        fl["values"] = vals.reshape(Mq).astype(np.float32); fl["adv"], fl["ret"] = adv.reshape(Mq), ret.reshape(Mq).astype(np.float32)
        mb = {k: jnp.asarray(v[:MB]) for k, v in fl.items()}
        p2, _, _ = mb_step(params, opt_state, mb, popart.sigma, popart.mu, a.ps, TOK_ENT)
        cb_fixed = all(np.array_equal(cb0[k], np.asarray(p2["params"][k]["embedding"])) for k in cb0)
        head_moved = not np.array_equal(np.asarray(params["params"]["head_signal_slot0"]["kernel"]), np.asarray(p2["params"]["head_signal_slot0"]["kernel"]))
        res = dict(unreachable_prob_max=unreach, broadcast_is_scaled_onehot=ok_bc, constant_codebook_mutant_caught=mut_caught,
                   codebooks_fixed_after_update=cb_fixed, symbol_head_moved=head_moved)
        ok = unreach < 1e-6 and ok_bc and mut_caught and cb_fixed and head_moved
        print(json.dumps(res)); print("SELFTEST-SYM", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)

    if a.selftest_wireblind:
        """C1 apparatus: the sender's token distribution must be exactly unchanged when its heard wire is replaced by
        random token triples (wire-blind), while a mutant with one live wire row must change it."""
        lisq = np.asarray(is_listener)
        _, Rw = rollout(params, init_state(jax.random.PRNGKey(41)), jax.random.PRNGKey(42))
        obsS = jnp.asarray(np.asarray(Rw["obs"])[:, ~lisq].reshape(-1, D))
        cb = [params["params"][k]["embedding"] for _, _, k in rl_jax.TOKEN_SLOTS]
        rt = jax.random.randint(jax.random.PRNGKey(3), (obsS.shape[0], 3), 0, V)
        rbc = jnp.concatenate([jnp.zeros((obsS.shape[0], 8))] + [cb[i][rt[:, i]] for i in range(3)], -1)
        res = {}
        for name, wb in (("wire_blind", 1), ("mutant_one_live_row", 2), ("normal_head", 0)):
            m_ = ToyNet(H, credit=a.credit, wire_blind=wb)
            def dist(o_in):
                _, o = m_.apply(params, jnp.zeros((o_in.shape[0], H)), o_in)
                return [np.asarray(jax.nn.softmax(lg, -1)) for lg in token_logits(params, o.z_e, TAU)]
            P, Q = dist(obsS), dist(obsS.at[:, O_WIRE:].set(rbc))
            res[name] = dict(tv=float(np.mean([0.5 * np.abs(P[i] - Q[i]).sum(-1).mean() for i in range(3)])),
                             max_abs=float(max(np.abs(P[i] - Q[i]).max() for i in range(3))),
                             argmax_changed=float(np.mean([np.mean(P[i].argmax(-1) != Q[i].argmax(-1)) for i in range(3)])))
        ok = res["wire_blind"]["max_abs"] == 0.0 and res["mutant_one_live_row"]["max_abs"] > 0.0
        print(json.dumps(res, indent=1)); print("SELFTEST-WIREBLIND", "PASS" if ok else "FAIL", flush=True)
        raise SystemExit(0 if ok else 1)

    np_rng = np.random.default_rng(a.seed)
    if a.same_spot:            # Cam Nov 9 ~02:40 check (a) + (c): same-spot swap control and joint / pre-Craft MI; frozen params
        from comm_testbed import mi_miller_madow
        lis = np.asarray(is_listener)
        ev_rng = np.random.default_rng(5_000 + a.seed)
        enc = lambda x: (x[..., 0] * V + x[..., 1]) * V + x[..., 2]  # noqa: E731

        def joint_mi(e, lab):
            _, ei = np.unique(e, return_inverse=True)
            J = np.zeros((ei.max() + 1, NCLS)); np.add.at(J, (ei, lab), 1); P = J / J.sum()
            px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True); nz = P > 0
            mi = float((P[nz] * np.log2(P[nz] / (px @ py)[nz])).sum())
            return mi, mi - (nz.sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2 * len(lab) * np.log(2)), int(ei.max() + 1)
        all_tok, all_lab, pre_tok, pre_lab = [], [], [], []
        for k in range(a.swap_pairs):
            st0 = init_state(jax.random.PRNGKey(10_000 + k))
            _, Rc = rollout(params, st0, jax.random.PRNGKey(20_000 + k), swap=False)
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            trip_c, lab = Rc["token_actions"][:, ~lis], Rc["active"]                 # (T, W, 3), (T, W)
            all_tok.append(trip_c.reshape(-1, 3)); all_lab.append(lab.reshape(-1))
            # steps before the listener's first Craft within each lit period (a period = a run of one lit spot in a world)
            craftL = Rc["actions"][:, lis] == ACTION_CRAFT                            # (T, W)
            for w_ in range(W):
                seen = False
                for t_ in range(T):
                    if t_ > 0 and lab[t_, w_] != lab[t_ - 1, w_]:
                        seen = False
                    if not seen:
                        pre_tok.append(trip_c[t_, w_]); pre_lab.append(lab[t_, w_])
                    if craftL[t_, w_]:
                        seen = True
            if a.arm != "COMM" or a.mi_null:
                continue
            table = np.zeros((4, 512, 3), np.int32)
            for t in range(4):
                tt = trip_c[lab == t]
                table[t] = tt[ev_rng.integers(0, len(tt), 512)]
            row = dict(phase="same_spot", pair=k, listener_dep_control=float(Rc["dep"][:, lis].sum() / W))
            for nm, sh in (("cross", 1), ("same", 0)):
                _, Rk = rollout(params, st0, jax.random.PRNGKey(20_000 + k), swap="empirical", table=jnp.asarray(table), shift=sh)
                Rk = jax.tree_util.tree_map(np.asarray, Rk)
                if not np.allclose(Rk["heard_next"][:, lis], Rk["bc_sw"][:, ~lis]):
                    print("[APPARATUS] FATAL: swap wiring", flush=True); raise SystemExit(1)
                row[f"listener_dep_{nm}"] = float(Rk["dep"][:, lis].sum() / W)
                row[f"swapped_differs_{nm}"] = float(np.any(Rk["trip_used"][:, ~lis] != Rk["token_actions"][:, ~lis], -1).mean())
            emit(row)
        tok, lab = np.concatenate(all_tok), np.concatenate(all_lab)
        ptok, plab = np.stack(pre_tok), np.asarray(pre_lab)
        jm, jmm, nd = joint_mi(enc(tok), lab); pm, pmm, pnd = joint_mi(enc(ptok), plab)
        emit(dict(phase="mi_joint", n=int(len(lab)), slot_mi_mm=[float(mi_miller_madow(tok[:, i], lab, V, NCLS)) for i in range(3)],
                  joint_mi_plugin=jm, joint_mi_mm=jmm, distinct_triples=nd,
                  pre_craft_n=int(len(plab)), pre_craft_slot_mi_mm=[float(mi_miller_madow(ptok[:, i], plab, V, NCLS)) for i in range(3)],
                  pre_craft_joint_mi_plugin=pm, pre_craft_joint_mi_mm=pmm, pre_craft_distinct=pnd))
        if a.mi_null:          # Cam Nov 9 ~02:55: spot labels permuted at the same N, same estimators
            nrng = np.random.default_rng(6_000 + a.seed); et = enc(tok)
            nl_ = np.array([joint_mi(et, nrng.permutation(lab))[:2] for _ in range(a.mi_null)])
            emit(dict(phase="mi_null", perms=a.mi_null, n=int(len(lab)), joint_mi_plugin=jm, joint_mi_mm=jmm,
                      null_plugin_mean=float(nl_[:, 0].mean()), null_plugin_p95=float(np.percentile(nl_[:, 0], 95)),
                      null_mm_mean=float(nl_[:, 1].mean()), null_mm_p95=float(np.percentile(nl_[:, 1], 95)),
                      null_mm_max=float(nl_[:, 1].max())))
        raise SystemExit(0)

    if a.export_m:             # for M (Cam, Nov 9): dictionary table + one episode transcript; frozen params, $0
        import csv
        lis = np.asarray(is_listener)
        outd = Path(a.export_m); outd.mkdir(parents=True, exist_ok=True)
        toks, labs = [], []
        for k in range(4):     # the speak test's dictionary keys, so the words reproduce its dictionary exactly
            _, Rc = rollout(params, init_state(jax.random.PRNGKey(70_000 + k)), jax.random.PRNGKey(71_000 + k))
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            toks.append(Rc["token_actions"][:, ~lis].reshape(-1, 3)); labs.append(Rc["active"].reshape(-1))
        tok, lab = np.concatenate(toks), np.concatenate(labs)
        HN = np.asarray(HEARTHS)
        slotw = np.array([[np.bincount(tok[lab == t, i], minlength=V).argmax() for i in range(3)] for t in range(4)])
        # naive-Bayes reader over slots (add-0.5 smoothing), fitted on the same dictionary rollouts
        cnt = np.stack([np.stack([np.bincount(tok[lab == t, i], minlength=V) for i in range(3)]) for t in range(4)]) + 0.5
        logp = np.log(cnt / cnt.sum(-1, keepdims=True)); prior = np.log(np.bincount(lab, minlength=4) / len(lab))
        def read(tr):
            s_ = prior + np.array([sum(logp[t, i, tr[i]] for i in range(3)) for t in range(4)])
            p_ = np.exp(s_ - s_.max()); p_ /= p_.sum()
            return int(p_.argmax()), float(p_.max())
        dic_rows = []
        for t in range(4):
            m = lab == t; vals, cnts = np.unique(tok[m], axis=0, return_counts=True); o = np.argsort(-cnts)
            other = [u for u in range(4) if u != t]
            row = dict(lit_spot=t, spot_xy=HN[t].tolist(), messages_sampled=int(m.sum()),
                       most_common_triple=vals[o[0]].tolist(), most_common_triple_share=float(cnts[o[0]] / m.sum()),
                       distinct_triples=int(len(vals)),
                       top5_triples=[dict(triple=vals[j].tolist(), share=float(cnts[j] / m.sum())) for j in o[:5]],
                       slot_words=slotw[t].tolist(),
                       slot_word_share=[float(np.mean(tok[m, i] == slotw[t, i])) for i in range(3)],
                       # how often each slot says one of the OTHER spots' words (0 = never confused)
                       slot_uses_other_spots_word=[float(np.mean(np.isin(tok[m, i], slotw[other, i]))) for i in range(3)],
                       read_back_correct=float(np.mean([read(x)[0] == t for x in tok[m][:4000]])))
            dic_rows.append(row)
        meta = dict(checkpoint=a.warm_start, arm=a.arm, wire_blind=a.wire_blind, grid=G, worlds=W, steps=T,
                    vocab_per_slot=V, note="sender = the informed agent (sees which spot is lit); listener = uninformed partner, "
                    "hears the sender's 3-token message one step later. slot words = each slot's most common token for "
                    "that spot. read_back_correct = how often a simple reader (per-slot frequencies) recovers the lit "
                    "spot from a single message.")
        json.dump(dict(meta=meta, dictionary=dic_rows), open(outd / "dictionary.json", "w"), indent=1)
        with open(outd / "dictionary.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["lit_spot", "spot_x", "spot_y", "most_common_triple", "share_of_messages", "distinct_triples",
                         "slot1_word", "slot1_share", "slot2_word", "slot2_share", "slot3_word", "slot3_share", "read_back_correct"])
            for r_ in dic_rows:
                wr.writerow([r_["lit_spot"], *r_["spot_xy"], "-".join(map(str, r_["most_common_triple"])),
                             round(r_["most_common_triple_share"], 4), r_["distinct_triples"],
                             *sum(([r_["slot_words"][i], round(r_["slot_word_share"][i], 4)] for i in range(3)), []),
                             round(r_["read_back_correct"], 4)])
        # one episode: a fresh rollout; the world with the most listener deliveries; every step of that world
        _, Re = rollout(params, init_state(jax.random.PRNGKey(95_000)), jax.random.PRNGKey(95_001))
        Re = jax.tree_util.tree_map(np.asarray, Re)
        dl = Re["dep"][:, lis].sum(0); w = int(dl.argmax()); si, li = 2 * w, 2 * w + 1
        NAMES = {0: "stay", 1: "move x-1", 2: "move x+1", 3: "move y+1", 4: "move y-1", ACTION_CRAFT: "craft"}
        posf = lambda t_, i_: [int(round(v)) for v in (Re["obs"][t_, i_, :2] * G)]  # noqa: E731
        def dist(p_, h_):
            d_ = np.abs(np.asarray(p_) - h_); d_ = np.minimum(d_, G - d_); return int(max(d_.max() - 1, 0))
        steps = []
        for t_ in range(T):
            lit = int(Re["active"][t_, w]); h = HN[lit]
            sp, lp_ = posf(t_, si), posf(t_, li)
            trip = Re["token_actions"][t_, si].tolist(); heard = Re["token_actions"][t_ - 1, si].tolist() if t_ > 0 else None
            rd = read(trip); hr = read(heard) if heard is not None else (None, None)
            la = int(Re["actions"][t_, li]); sa = int(Re["actions"][t_, si])
            nxt = posf(t_ + 1, li) if t_ + 1 < T else None
            steps.append(dict(step=t_, lit_spot=lit, lit_spot_xy=h.tolist(), sender_xy=sp, sender_saw=f"spot {lit} is lit",
                              sender_sent=trip, sender_message_reads_as=rd[0], reader_confidence=round(rd[1], 3),
                              sender_action=NAMES.get(sa, f"idle ({sa})"), listener_xy=lp_, listener_heard=heard,
                              heard_reads_as=hr[0], listener_action=NAMES.get(la, f"idle ({la})"),
                              listener_steps_to_lit_spot=dist(lp_, h),
                              listener_delivered=bool(Re["dep"][t_, li]), sender_delivered=bool(Re["dep"][t_, si]),
                              listener_next_xy=nxt))
        json.dump(dict(meta=dict(meta, world=w, listener_deliveries_in_episode=int(dl[w]), key="95000/95001",
                                 note=meta["note"] + " A delivery respawns the agent elsewhere and re-lights a different "
                                 "spot; an unlit-for-too-long spot also re-rolls. 'idle (k)' = an action with no effect in this world. "
                                 "listener_steps_to_lit_spot = moves needed to stand next to the lit spot (0 = can craft there)."), steps=steps),
                  open(outd / "transcript.json", "w"), indent=1)
        with open(outd / "transcript.csv", "w", newline="") as f:
            wr = csv.writer(f); keys = list(steps[0].keys()); wr.writerow(keys)
            for s_ in steps:
                wr.writerow([("-".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v)) for v in (s_[k] for k in keys)])
        emit(dict(phase="export_m", outdir=str(outd), world=w, listener_deliveries=int(dl[w]),
                  words=[r_["most_common_triple"] for r_ in dic_rows]))
        raise SystemExit(0)

    if a.speak_test:           # exploratory, frozen params, its own keys (70_000+ / 80_000+)
        lis = np.asarray(is_listener)
        from comm_testbed import mi_miller_madow
        toks, labs = [], []
        for k in range(4):
            _, Rc = rollout(params, init_state(jax.random.PRNGKey(70_000 + k)), jax.random.PRNGKey(71_000 + k))
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            toks.append(Rc["token_actions"][:, ~lis].reshape(-1, 3)); labs.append(Rc["active"].reshape(-1))
        tok, lab = np.concatenate(toks), np.concatenate(labs)
        enc = (tok[:, 0] * V + tok[:, 1]) * V + tok[:, 2]
        dic, purity = np.zeros((4, 3), np.int32), []
        for t in range(4):
            vals, cnts = np.unique(tok[lab == t], axis=0, return_counts=True)
            dic[t] = vals[cnts.argmax()]; purity.append(float(cnts.max() / cnts.sum()))
        # per-slot dictionary: each slot's modal token for fire t (the triples are diverse even when every slot is pure)
        dslot = np.array([[np.bincount(tok[lab == t, i], minlength=V).argmax() for i in range(3)] for t in range(4)], np.int32)
        spur = [[float(np.mean(tok[lab == t, i] == dslot[t, i])) for i in range(3)] for t in range(4)]
        emit(dict(phase="dictionary", words=dic.tolist(), purity=purity, distinct=int(len(np.unique(dic, axis=0))),
                  slot_words=dslot.tolist(), slot_purity=spur,
                  mi_slot=[float(mi_miller_madow(tok[:, i], lab, V, NCLS)) for i in range(3)], n=int(len(lab))))
        # joint-triple MI (plug-in; the triple alphabet is V^3, so Miller-Madow is reported alongside)
        _, ei = np.unique(enc, return_inverse=True)
        J = np.zeros((ei.max() + 1, 4)); np.add.at(J, (ei, lab), 1); P = J / J.sum()
        px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True); nz = P > 0
        mi_j = float((P[nz] * np.log2(P[nz] / (px @ py)[nz])).sum())
        emit(dict(phase="triple_mi", mi_triple_plugin=mi_j,
                  mi_triple_mm=mi_j - ((nz.sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2 * len(lab) * np.log(2))),
                  distinct_triples=int(ei.max() + 1)))
        for k in range(a.swap_pairs):
            st0, kk = init_state(jax.random.PRNGKey(80_000 + k)), jax.random.PRNGKey(81_000 + k)
            _, Rc = rollout(params, st0, kk)
            Rc = jax.tree_util.tree_map(np.asarray, Rc)
            trip_c, labc = Rc["token_actions"][:, ~lis], Rc["active"]
            own = np.zeros((4, 512, 3), np.int32)
            for t in range(4):
                tt = trip_c[labc == t]
                own[t] = tt[np.random.default_rng(k).integers(0, len(tt), 512)] if len(tt) else dic[t]
            word = np.repeat(dic[:, None, :], 512, 1); wslot = np.repeat(dslot[:, None, :], 512, 1)
            babble = np.random.default_rng(90_000 + k).integers(0, V, (4, 512, 3)).astype(np.int32)
            for sh in range(4):     # say X = T + sh (sh = 0: the truth)
                row = dict(phase="speak", pair=k, shift=sh)
                for name, sw, tab in (("control", False, None), ("own_words", "empirical", own), ("dictionary", "empirical", word),
                                       ("slot_dictionary", "empirical", wslot), ("babble", "empirical", babble)):
                    _, Rk = rollout(params, st0, kk, swap=sw, table=None if tab is None else jnp.asarray(tab), shift=sh)
                    Rk = jax.tree_util.tree_map(np.asarray, Rk)
                    row[f"crafts_at_X_{name}"] = float(Rk["att_Tp"][:, lis].sum() / W)
                    row[f"dep_{name}"] = float(Rk["dep"][:, lis].sum() / W)
                    row[f"crafts_any_{name}"] = float(Rk["att"][:, lis].sum() / W)   # listener Crafts at any fire
                emit(row)
        return
    if a.eval_only:            # exploratory: evaluate the loaded params (no training, no MI)
        lis = np.asarray(is_listener); mi_buf_tok, mi_buf_T = deque(), deque()
        evaluate(params, 0); return
    key, k_s = jax.random.split(key)
    state = init_state(k_s)
    lis = np.asarray(is_listener)
    mi_buf_tok, mi_buf_T = deque(maxlen=10), deque(maxlen=10)
    u_start = 1
    elder_buf = None
    if a.resume:
        import pickle
        snap = pickle.load(open(a.resume, "rb"))
        params = jax.tree_util.tree_map(jnp.asarray, snap["params"])
        opt_state = jax.tree_util.tree_map(jnp.asarray, snap["opt_state"])
        popart.mu, popart.nu, popart.initialised = snap["popart"]
        state = tuple(jnp.asarray(x) for x in snap["state"]); key = jnp.asarray(snap["key"])
        np_rng.bit_generator.state = snap["np_rng"]
        mi_buf_tok.extend(snap["mi_tok"]); mi_buf_T.extend(snap["mi_T"])
        turn_state["maps"] = snap["maps"]; imit_buf.extend(snap["imit"]); elder_buf = snap["elder"]
        u_start = snap["u"] + 1
        if a.branch == "frozen_listener":
            branch_hold["pL"] = jax.tree_util.tree_map(jnp.asarray, snap["params"])
        if a.branch == "wireblind":
            params = wireblind(params)
        emit(dict(phase="resume", update=snap["u"], branch=a.branch))
    for u in range(u_start, a.updates + 1):
        t0 = time.time()
        key, k_r, k_l = jax.random.split(key, 3)
        state, R = rollout(params, state, k_r)
        R = jax.tree_util.tree_map(np.asarray, R)
        if COMP and np.isin(R["active"], HELD).any():
            print(f"[APPARATUS] FATAL: a held-out pair was active in training at update {u}", flush=True); raise SystemExit(1)
        values = popart.denormalise(R["values"])
        obs_last = build_obs(state[0], state[1], state[3], k_l)
        _, o_last = apply_fn(params, state[4], obs_last)
        v_last = popart.denormalise(np.asarray(o_last.values))
        adv, ret = gae_bootstrap(R["rewards"], values, np.zeros_like(R["rewards"]), v_last, GAMMA, LAM)
        params = popart.update_and_preserve(ret, params)
        M = T * N
        flat = {k: R[k].reshape((M,) + R[k].shape[2:]) for k in ("obs", "actions", "log_probs", "carries",
                                                                  "token_actions")}
        flat["values"] = values.reshape(M).astype(np.float32)
        flat["adv"], flat["ret"] = adv.reshape(M), ret.reshape(M).astype(np.float32)
        if a.credit != "joint":
            if CENTRAL:
                central_last.update(obs=jnp.asarray(obs_last), carries=state[4], active=state[1])
            flat.update(credit_streams(R, o_last, params, M))
            if CENTRAL:
                flat.pop("_v_norm", None)
                if a.credit == "Ag":
                    cparams = popart_g.update_and_preserve(flat.pop("_ret_g_full"), cparams)
                if a.selftest_credit:
                    central_selftest(params, flat, R, o_last, M)
            if a.credit == "A":
                params = popart_s.update_and_preserve(flat.pop("_ret_s_full"), params)
                flat["mu_s"] = np.full(M, popart_s.mu, np.float32); flat["sig_s"] = np.full(M, popart_s.sigma, np.float32)
            if a.selftest_credit:
                credit_selftest(params, flat, R, o_last, M)
        perm = np_rng.permutation(M)
        ps_c = a.ps
        te_c = a.erode_tok_ent if (a.erode_tok_ent is not None and u >= a.erode_from) else TOK_ENT
        if a.ps_anneal:
            s0_, e0_ = a.ps_anneal
            ps_c = a.ps * float(np.clip((e0_ - u) / (e0_ - s0_), 0.0, 1.0)) if u > s0_ else a.ps
        msum = {}
        diag = a.diag_from is not None and a.diag_from <= u <= a.diag_to
        if diag:
            th = head_vec(params)
            if u == a.diag_from:
                diag_state["theta0"] = th
            gsum = 0.0
            for i in range(4):   # PPO-loss gradient on the message head, mean over the first 4 minibatches
                mb = {k: jnp.asarray(v[perm[i * MB:(i + 1) * MB]]) for k, v in flat.items()}
                g = jax.grad(lambda p_: total_loss(p_, mb, popart.sigma, popart.mu, ps_c, te_c)[0])(params)
                gsum = gsum + head_vec(g)
            g_head = gsum / 4
            back = diag_state["theta0"] - th
            opt_disp = np.zeros_like(th); dec_disp = np.zeros_like(th)
        for i in range(M // MB):
            idx = perm[i * MB:(i + 1) * MB]
            mb = {k: jnp.asarray(v[idx]) for k, v in flat.items()}
            if diag:
                before = head_vec(params)
            if a.resume and a.branch == "frozen_listener":
                mb = dict(mb, alive=(mb["obs"][:, O_INF] > 0.5).astype(jnp.float32))   # the frozen listener does not train
            if a.resume and a.branch == "stopgrad":
                params, opt_state, mets = mb_step_stopgrad(params, opt_state, mb, popart.sigma, popart.mu, ps_c, te_c)
            elif a.msg_grad == "sender":
                params, opt_state, mets = mb_step_sg2(params, opt_state, mb, popart.sigma, popart.mu, ps_c, te_c)
            else:
                params, opt_state, mets = mb_step(params, opt_state, mb, popart.sigma, popart.mu, ps_c, te_c)
            if a.resume and a.branch == "wireblind":
                params = wireblind(params)
            if CENTRAL:
                snd_m = (mb["obs"][:, O_INF] > 0.5).astype(jnp.float32)
                tgt = mb["ret_g"] if a.credit == "Ag" else mb["g_s"]
                cparams, cstate, closs_ = crit_step(cparams, cstate, mb["gfeat"], mb["token_actions"], tgt, snd_m,
                                                   popart_g.mu, popart_g.sigma)
                mets = dict(mets, central_critic_loss=closs_)
            if diag:
                mid = head_vec(params); opt_disp += mid - before
            if a.erode_wd is not None and u >= a.erode_from:
                pp = dict(params["params"])
                for hk in ("gwt_comms_1", "head_signal_slot0", "head_signal_slot1", "head_signal_slot2"):
                    pp[hk] = jax.tree_util.tree_map(lambda x: x * (1.0 - a.erode_wd), pp[hk])
                params = {"params": pp}
            if diag:
                dec_disp += head_vec(params) - mid
            for k, v in mets.items():
                msum[k] = msum.get(k, 0.0) + float(v)
        mets = {k: v / (M // MB) for k, v in msum.items()}
        dep, att, ent = R["dep"], R["att"], R["ent"]
        if diag:
            nb = float(np.linalg.norm(back))
            cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))  # noqa: E731
            # advantage-variance share of the token action from the shared vs the sender's own reward: discounted
            # (gamma * lambda) forward sums of each reward component on sender rows, no baseline
            rs = R["r_shared"][:, ~lis]; ro = R["rewards"][:, ~lis] - rs
            def disc(x):
                out = np.zeros_like(x); c = np.zeros(x.shape[1], np.float64)
                for t in range(x.shape[0] - 1, -1, -1):
                    c = x[t] + GAMMA * LAM * c; out[t] = c
                return out
            vs_, vo_ = float(np.var(disc(rs))), float(np.var(disc(ro)))
            drec = dict(phase="diag", update=u, grad_head_norm=float(np.linalg.norm(g_head)),
                        decay_pull_norm=float(a.erode_wd * np.linalg.norm(th)) if a.erode_wd else 0.0,
                        theta_head_norm=float(np.linalg.norm(th)), dist_from_1500=nb,
                        cos_negrad_back=(cos(-g_head, back) if nb > 0 else None),
                        opt_disp_norm=float(np.linalg.norm(opt_disp)), decay_disp_norm=float(np.linalg.norm(dec_disp)),
                        cos_optdisp_back=(cos(opt_disp, back) if nb > 0 else None),
                        shared_var=vs_, own_var=vo_, shared_share=vs_ / (vs_ + vo_ + 1e-30),
                        mi_plugin=mi_plugin(R["token_actions"][:, ~lis].reshape(-1, 3), R["active"].reshape(-1)))
            if (u - a.diag_from) % a.diag_every == 0 or u == a.diag_to:
                drec.update(reliance_probe(params))
            emit(drec)
        ev = 1 - np.var(ret - values) / (np.var(ret) + 1e-8)
        rec = dict(phase="train", update=u,
                   listener_dep_per_world=float(dep[:, lis].sum() / W), sender_dep_per_world=float(dep[:, ~lis].sum() / W),
                   listener_attempts=int(att[:, lis].sum()), sender_attempts=int(att[:, ~lis].sum()),
                   listener_crafts=int((R["actions"][:, lis] == ACTION_CRAFT).sum()),
                   listener_correct_frac=float(dep[:, lis].sum() / max(att[:, lis].sum(), 1)),
                   token_entropy_sender=float(ent[:, ~lis].mean()), token_entropy_listener=float(ent[:, lis].mean()),
                   mi_plugin=(mi_plugin(R["token_actions"][R["inf"]], np.repeat(R["active"], 2, axis=1)[R["inf"]])
                              if SYMR else mi_plugin(R["token_actions"][:, ~lis].reshape(-1, 3), R["active"].reshape(-1))),
                   informed_frac=(float(R["inf"].mean()) if SYMR else None),
                   **(dict(lookout_occ=float(R["atl"].mean()), lookout_any=float(R["atl"].reshape(T, W, 2).any(-1).mean()),
                           agent_dep_per_world=float(R["dep"].sum() / W)) if LOOK is not None else {}),
                   tok_ent_coef=float(te_c),
                   swap_diff=float(R["swap_diff"][:, ~lis].mean()), listener_sens_tv=float(R["sens"][:, lis].mean()),
                   ev=float(ev), aux_ps=mets.get("aux_ps", 0.0), aux_pl=mets.get("aux_pl", 0.0),
                   ppo_entropy=mets.get("ppo_entropy"), ps_coef=float(ps_c), t_update=round(time.time() - t0, 2))
        emit(rec)
        if SYMR:
            infm = R["inf"]; labm = np.repeat(R["active"], 2, axis=1)
            mi_buf_tok.append(R["token_actions"][infm]); mi_buf_T.append(labm[infm])
        else:
            mi_buf_tok.append(R["token_actions"][:, ~lis].reshape(-1, 3)); mi_buf_T.append(R["active"].reshape(-1))
        if a.reset_from is not None and a.imitate:
            imit_buf.append((R["obs"][:, ~lis].reshape(-1, D), R["token_actions"][:, ~lis].reshape(-1, 3)))
        if u in a.eval_at and u != a.updates:
            evaluate(params, u)
        if u in a.save_at and a.save_prefix:
            Path(a.save_prefix).parent.mkdir(parents=True, exist_ok=True)
            Path(f"{a.save_prefix}_u{u}_actor.msgpack").write_bytes(flax.serialization.to_bytes(params))
            if CENTRAL:
                Path(f"{a.save_prefix}_u{u}_critic.msgpack").write_bytes(flax.serialization.to_bytes(cparams))
                Path(f"{a.save_prefix}_u{u}_popart_g.json").write_text(json.dumps(dict(mu=popart_g.mu, nu=popart_g.nu)))
        if a.reset_from is not None and (u in a.eval_at or u == a.updates):
            turnover_record(u)
        if a.reset_from is not None and u in reset_sched and not (a.resume and u == a.updates):
            if u == a.snapshot_at or (elder_buf is None and reset_sched[u] == "sender" and a.imitate):
                elder_buf = [(np.array(o_), np.array(t_)) for o_, t_ in imit_buf]
            params, opt_state = do_reset(reset_sched[u], params, opt_state, u)
        if a.snapshot_at is not None and u == a.snapshot_at:
            import pickle
            snap = dict(u=u, params=jax.tree_util.tree_map(np.asarray, params), opt_state=jax.tree_util.tree_map(np.asarray, opt_state),
                        popart=(popart.mu, popart.nu, popart.initialised), state=tuple(np.asarray(x) for x in state),
                        key=np.asarray(key), np_rng=np_rng.bit_generator.state, mi_tok=list(mi_buf_tok), mi_T=list(mi_buf_T),
                        maps=turn_state["maps"], imit=list(imit_buf), elder=elder_buf)
            Path(a.snapshot_path).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(snap, open(a.snapshot_path, "wb"))
            emit(dict(phase="snapshot", update=u, path=a.snapshot_path))
        if elder_buf and ((a.resume and u == a.updates) or (not a.resume and (u in a.eval_at or u == a.updates))):
            # copy accuracy of the current head vs the elder's (obs, token) pairs from before the first sender reset
            X = np.concatenate([b[0] for b in elder_buf]); Y = np.concatenate([b[1] for b in elder_buf])
            ix = np.random.default_rng(0).integers(0, len(X), 4096)
            _, o = apply_fn(params, jnp.zeros((4096, H)), jnp.asarray(X[ix]))
            pred = np.stack([np.asarray(jnp.argmax(lg, -1)) for lg in token_logits(params, o.z_e, TAU)], -1)
            emit(dict(phase="copy_acc", update=u, branch=a.branch, copy_acc=float((pred == Y[ix]).all(1).mean()),
                      copy_acc_slot=float((pred == Y[ix]).mean())))

    if a.save_params:
        Path(a.save_params).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save_params).write_bytes(flax.serialization.to_bytes(params))
    evaluate(params, a.updates)

if __name__ == "__main__":
    main()
