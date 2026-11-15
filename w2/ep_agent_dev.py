"""DEV (Cam Nov 15 ~09:30, the stage-2 cost lever): EpAgent in setup style, the SAME parameter tree as w2/ep_agent.py,
with two extra entry points so the PPO replay can compute each record's key / value once per sequence:
  kv(mem)                      -> K (n, 32), V (n, 32) for raw records (n, R)
  step_kv(h, x, K, V, valid)   -> the same outputs as __call__, reading from precomputed K / V under a validity mask.
__call__(h, x) is the original forward (reads the memory inside h)."""
import jax
import jax.numpy as jnp
import flax.linen as nn


class EpAgentDev(nn.Module):  # noqa: D101
    S: int = 128; R: int = 11; hd: int = 128; use_hind: bool = True; shared_tok: bool = False; tok_at: int = 100

    def setup(self):
        zi = nn.initializers.zeros
        self.Dense_0 = nn.Dense(128); self.Dense_1 = nn.Dense(128); self.Dense_2 = nn.Dense(6); self.Dense_3 = nn.Dense(1)
        self.side_enc = nn.Dense(64); self.gru = nn.GRUCell(self.hd); self.f1 = nn.Dense(64)
        if self.shared_tok:
            self.tokemb = nn.Dense(16, use_bias=False); self.q = nn.Dense(16); self.k = nn.Dense(16)
        else:
            self.q = nn.Dense(32); self.k = nn.Dense(32)
        self.v = nn.Dense(32); self.f2 = nn.Dense(64)
        if self.use_hind:
            self.hind = nn.Dense(8)
        self.side_head = nn.Dense(6, kernel_init=zi, bias_init=zi); self.side_value = nn.Dense(1, kernel_init=zi, bias_init=zi)

    def kv(self, mem):
        if self.shared_tok:
            K = jnp.concatenate([self.tokemb(mem[..., :8]), self.k(mem)], -1)
        else:
            K = self.k(mem)
        return K, self.v(mem)

    def _query(self, x):
        qz = jnp.tanh(self.f1(x))
        if self.shared_tok:
            return jnp.concatenate([self.tokemb(x[self.tok_at: self.tok_at + 8]), self.q(qz)])
        return self.q(qz)

    def _trunk(self, hg, x):
        obs, side = x[..., :100], jnp.concatenate([x[..., 100:], x[..., 98:100]], -1)
        t = jnp.tanh(self.Dense_0(obs)); t = jnp.tanh(self.Dense_1(t))
        lg_nav = self.Dense_2(t); v_nav = self.Dense_3(t)[..., 0]
        z = jnp.tanh(self.side_enc(jnp.concatenate([side, t], -1)))
        hg2, y = self.gru(hg, z)
        return hg2, y, lg_nav, v_nav

    def _head(self, y, rd, lg_nav, v_nav):
        y2 = jnp.tanh(self.f2(jnp.concatenate([y, rd])))
        hl = self.hind(jnp.concatenate([y2, rd])) if self.use_hind else jnp.zeros(8)
        feat = jnp.concatenate([y2, jax.nn.softmax(hl)])
        return lg_nav + self.side_head(feat), v_nav + self.side_value(feat)[..., 0], hl

    def step_kv(self, hg, x, K, V, valid):
        hg2, y, lg_nav, v_nav = self._trunk(hg, x)
        q = self._query(x)
        att = jnp.where(valid, K @ q / jnp.sqrt(32.0), -1e9); w = jax.nn.softmax(att) * valid.any()
        rd = w @ V
        lg, v, hl = self._head(y, rd, lg_nav, v_nav)
        return hg2, lg, v, hl

    def __call__(self, h, x):
        hg, memf, ptr = h[: self.hd], h[self.hd: self.hd + self.S * self.R], h[self.hd + self.S * self.R:]
        mem = memf.reshape(self.S, self.R); filled = mem[:, :8].sum(-1) > 0
        K, V = self.kv(mem)
        hg2, lg, v, hl = self.step_kv(hg, x, K, V, filled)
        return jnp.concatenate([hg2, memf, ptr]), lg, v, (hl, None)
