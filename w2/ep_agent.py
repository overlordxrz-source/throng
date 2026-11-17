"""Cam Nov 15 ~05:50: the W2 LEARNER agent. The harness agent (navigator trunk, GRU, zero-initialised side head) plus
the M-approved inborn EPISODIC MEMORY (S raw records per life, written by the world loop at each bout close: heard
token one-hot, displacement from the hearing place / 24, ate flag) read by learned key / value attention queried by
the current input, and the approved hindsight drive (a next-meal-sector head, trained only at the hearing step, whose
softmax feeds the policy). --shared-tok option: query and keys share one token embedding (the toy's screen-2 prior).

Carry h = [GRU state (hd) | memory (S * R) | write pointer (1)]. The memory part is never changed by the network.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn


class EpAgent(nn.Module):  # noqa: D101
    S: int = 128; R: int = 11; hd: int = 128; hind: bool = True; shared_tok: bool = False; tok_at: int = 100
    n_act: int = 6              # Cam Nov 17 L2: 10 with 8-way movement

    @nn.compact
    def __call__(self, h, x):
        hg, memf, ptr = h[: self.hd], h[self.hd: self.hd + self.S * self.R], h[self.hd + self.S * self.R:]
        mem = memf.reshape(self.S, self.R); filled = mem[:, :8].sum(-1) > 0
        obs, side = x[..., :100], jnp.concatenate([x[..., 100:], x[..., 98:100]], -1)
        t = nn.tanh(nn.Dense(128, name="Dense_0")(obs)); t = nn.tanh(nn.Dense(128, name="Dense_1")(t))
        lg_nav = nn.Dense(self.n_act, name="Dense_2")(t); v_nav = nn.Dense(1, name="Dense_3")(t)[..., 0]
        z = nn.tanh(nn.Dense(64, name="side_enc")(jnp.concatenate([side, t], -1)))
        hg2, y = nn.GRUCell(self.hd, name="gru")(hg, z)
        qz = nn.tanh(nn.Dense(64, name="f1")(x))
        if self.shared_tok:
            E = nn.Dense(16, use_bias=False, name="tokemb")
            q = jnp.concatenate([E(x[self.tok_at: self.tok_at + 8]), nn.Dense(16, name="q")(qz)])
            k = jnp.concatenate([E(mem[:, :8]), nn.Dense(16, name="k")(mem)], -1)
        else:
            q = nn.Dense(32, name="q")(qz); k = nn.Dense(32, name="k")(mem)
        vv = nn.Dense(32, name="v")(mem)
        att = jnp.where(filled, k @ q / jnp.sqrt(32.0), -1e9); w = jax.nn.softmax(att) * filled.any()
        rd = w @ vv
        y2 = nn.tanh(nn.Dense(64, name="f2")(jnp.concatenate([y, rd])))
        hl = nn.Dense(8, name="hind")(jnp.concatenate([y2, rd])) if self.hind else jnp.zeros(8)
        feat = jnp.concatenate([y2, jax.nn.softmax(hl)])
        zi = nn.initializers.zeros
        lg = lg_nav + nn.Dense(self.n_act, name="side_head", kernel_init=zi, bias_init=zi)(feat)
        v = v_nav + nn.Dense(1, name="side_value", kernel_init=zi, bias_init=zi)(feat)[..., 0]
        return jnp.concatenate([hg2, memf, ptr]), lg, v, (hl, w)
