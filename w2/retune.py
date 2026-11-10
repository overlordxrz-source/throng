"""W2 world retune (Cam Nov 10 ~02:25): new gates on MEDIANS, juvenile provisioning on, parents paying.
Retune the WORLD (drain, patch count, relocation), never the agents: the same learner (w2/learn.py shared-brain PPO,
400 updates) trains a blind and an oracle policy per world config; then each policy is evaluated FROZEN in a world with
reproduction (setpoint rule) and parent-paid provisioning (J 500): 16 worlds x 64 slots, 32 founders, 6,000 steps.
Per life: lifespan, offspring, energy intake per step (lives still alive at the end count as censored at their
current values).
  P1m: learned blind median lifespan >= 3x random's.
  P4:  learned blind median lifespan >= 2,000 steps.
  P2m: median offspring per life, oracle >= 2x blind (if both medians are 0: median energy intake per step,
       oracle >= 2x blind).
usage: python w2/retune.py --d0 0.004 --P 12 --relocate 0.0005
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2 import env, learn  # noqa: E402


def evaluate(cfg, params, net, oracle, policy, worlds=16, T=6000, seed=5, founders=32):
    def one(k):
        s = env.reset(cfg, k)
        s = s._replace(alive=jnp.arange(cfg.N) < founders, rel=0.5 * (1 - jnp.eye(cfg.N)) * (jnp.arange(cfg.N) < founders)[:, None] * (jnp.arange(cfg.N) < founders)[None, :])
        def body(c, kk):
            s, kids, intake, life, lifeid = c
            if policy == "random":
                a = jax.random.randint(kk, (cfg.N,), 0, 6)
            else:
                lg, _ = net.apply(params, env.observe(cfg, s, oracle)); a = jax.random.categorical(kk, lg)
            s2, _, info = env.step(cfg, s, a, kk)
            births = info["births"]; par = s2.parent
            kids = kids + jnp.zeros(cfg.N).at[jnp.clip(par, 0, cfg.N - 1)].add(births.astype(jnp.float32))
            intake = intake + cfg.eff * info["take"]; life = life + s.alive
            ended = info["dead"]
            rec = (ended, life, kids, intake, s2.parent >= 0)
            kids = jnp.where(births | ended, 0.0, kids); intake = jnp.where(births | ended, 0.0, intake)
            life = jnp.where(births | ended, 0, life)
            return (s2, kids, intake, life, lifeid), rec
        z = jnp.zeros(cfg.N)
        (s, kids, intake, life, _), (ended, L, Kd, In, Off) = jax.lax.scan(body, (s, z, z, jnp.zeros(cfg.N, jnp.int32), z),
                                                                     jax.random.split(jax.random.fold_in(k, 3), T))
        return ended, L, Kd, In, Off, s.alive, life, kids, intake, s.parent >= 0
    out = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(seed), worlds))
    ended, L, Kd, In, Off, alive, life, kids, intake, off_end = (np.asarray(x) for x in out)
    lifes = np.concatenate([L[ended], life[alive]]); offs = np.concatenate([Kd[ended], kids[alive]])
    ints = np.concatenate([In[ended], intake[alive]]); per = ints / np.maximum(lifes, 1)
    isoff = np.concatenate([Off[ended], off_end[alive]])             # lives born in the run (provisioned) vs founders
    def st(mask):
        if mask.sum() == 0:
            return dict(lives=0)
        return dict(lives=int(mask.sum()), median_life=float(np.median(lifes[mask])), mean_life=float(lifes[mask].mean()),
                    median_offspring=float(np.median(offs[mask])), mean_offspring=float(offs[mask].mean()),
                    median_intake_per_step=float(np.median(per[mask])))
    return dict(offspring_lives=st(isoff), all_lives=st(np.ones_like(isoff, bool)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=float, default=0.008); ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=1.0); ap.add_argument("--relocate", type=float, default=0.0)
    ap.add_argument("--updates", type=int, default=400); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = dict(L=96, P=a.P, patch_sigma=a.sigma, d0=a.d0, relocate=a.relocate)
    tcfg = env.Cfg(N=64, reproduce=False, A_max=100_000, **base)                     # training world (as before)
    ecfg = env.Cfg(N=64, reproduce=True, rep_mode="setpoint", juvenile=500, A_max=100_000, **base)   # evaluation world
    pb, nb = learn.train(False, a.updates, 0, juvenile=500, cfg=tcfg)
    po, no = learn.train(True, a.updates, 0, juvenile=500, cfg=tcfg)
    R = dict(random=evaluate(ecfg, pb, nb, False, "random"), blind=evaluate(ecfg, pb, nb, False, "policy"),
             oracle=evaluate(ecfg, po, no, True, "policy"))
    # gates on the lives born during the evaluation (provisioned newborns); random has none, so P1m uses its all-lives median
    b, o = R["blind"]["offspring_lives"], R["oracle"]["offspring_lives"]; r = R["random"]["all_lives"]
    ok_b, ok_o = b.get("lives", 0) >= 20, o.get("lives", 0) >= 20
    p1m = ok_b and b["median_life"] >= 3 * r["median_life"]; p4 = ok_b and b["median_life"] >= 2000
    if ok_b and ok_o and b["median_offspring"] > 0:
        p2m = o["median_offspring"] >= 2 * b["median_offspring"]; p2_basis = "offspring"
    elif ok_b and ok_o:
        p2m = o["median_intake_per_step"] > 0 and o["median_intake_per_step"] >= 2 * b["median_intake_per_step"]
        p2_basis = "intake/step (offspring medians 0)"
    else:
        p2m = False; p2_basis = "too few offspring lives (< 20)"
    res = dict(config=base, results=R, P1m=bool(p1m), P4=bool(p4), P2m=bool(p2m), P2m_basis=p2_basis)
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
