"""Cam Nov 17 G2: fairness by sector, pure geometry with the world's own rules (named_item.place formula, the
close rule t2 >= TO*d, eat = on the cell AND EAT, step_toward / follower from named_item). No food, energy or agents."""
import sys, json, types
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy")
import jax, jax.numpy as jnp, numpy as np
from w2 import named_item as NI
L = 96; EAT = NI.EAT

# check the placement formula used here against NI.place on random keys
start = jnp.full((64, 2), 40, jnp.int32)
for d in (3, 12, 21):
    sec, dd, ipos = NI.place(jax.random.PRNGKey(d), start, lambda k: jnp.full((64,), float(d)), L)
    for s_, p_ in zip(np.asarray(sec), np.asarray(ipos)):
        a = s_ * np.pi / 4; assert tuple(p_ - 40) == (round(float(jnp.round(d * jnp.cos(a)))), round(float(jnp.round(d * jnp.sin(a))))), (d, s_, p_)

def run(policy, sec, d, TO):
    st = jnp.array([[40, 40]], jnp.int32); a_ = sec * np.pi / 4
    ip = (st + jnp.stack([jnp.round(d * jnp.cos(a_)), jnp.round(d * jnp.sin(a_))], -1).astype(jnp.int32)[None]) % L
    item = dict(NI.new_item(1), sec=jnp.array([sec], jnp.int32), d=jnp.array([float(d)]), pos=ip, start=st, act=jnp.array([True]))
    pos = st; t = 0
    while True:
        if policy == "ideal":
            on = bool(jnp.all(pos == ip)); act = EAT if on else int(NI.step_toward(pos, ip, L)[1][0])
        else:
            s = types.SimpleNamespace(pos=pos, food=jnp.zeros((L, L)))
            act = int(NI.follower(s, item, L, jnp.zeros(1, jnp.int32))[0])
        ate = bool(jnp.all(pos == ip)) and act == EAT
        t += 1
        if ate or t >= TO * d:
            return dict(ate=ate, steps=t, budget=TO * d, manhattan=int(np.abs(np.asarray(ip - st)).sum()))
        pos = (pos + NI.MOVES[act]) % L

out = {}
for TO in (4.0, 1.2):
    for pol in ("ideal", "sector_only"):
        for d in (3, 12, 21):
            rows = [run(pol, k, d, TO) for k in range(8)]
            card = [r for k, r in enumerate(rows) if k % 2 == 0]; diag = [r for k, r in enumerate(rows) if k % 2 == 1]
            ratio = lambda R: np.mean([r["steps"] / r["budget"] for r in R if r["ate"]]) if any(r["ate"] for r in R) else float("nan")  # noqa: E731
            out[f"TO{TO}_{pol}_d{d}"] = dict(reach=[int(r["ate"]) for r in rows], steps=[r["steps"] for r in rows], manhattan=[r["manhattan"] for r in rows],
                                             steps_over_timeout_card=round(ratio(card), 3), steps_over_timeout_diag=round(ratio(diag), 3),
                                             diag_over_card=round(ratio(diag) / ratio(card), 3))
            print(f"timeout {TO}d {pol:11s} d={d:2d} reach {[int(r['ate']) for r in rows]} steps {[r['steps'] for r in rows]} budget {TO*d:g} "
                  f"card {out[f'TO{TO}_{pol}_d{d}']['steps_over_timeout_card']} diag {out[f'TO{TO}_{pol}_d{d}']['steps_over_timeout_diag']} ratio {out[f'TO{TO}_{pol}_d{d}']['diag_over_card']}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
