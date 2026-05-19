"""
agents/network.py — Actor-Critic brain for MAPPO (Phase 2).

Parameter sharing: one set of weights per team.
All agents on a team run the SAME network with DIFFERENT carries (memory).

Input: single flat observation vector x (shape: obs_dim,)
  [own_state(5) | neighbor_sigs(K*sd) | local_symbols(W*symd) | local_presence(W*2) | own_signal(sd)]

Outputs:
  action_logits  (n_actions,)   — discrete movement policy
  signal_out     (signal_dim,)  — continuous broadcast vector [-1, 1]
  symbol_write   (symbol_dim,)  — continuous culture write vector [-1, 1]
  value          ()             — scalar state-value estimate V(s)

Brain depth n_layers is a SHARED team-level integer.
All max_layers attention blocks exist in the params; blocks i >= n_layers
are gated to identity via jnp.where so depth can grow without re-init.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


class _AttentionBlock(nn.Module):
    token_dim: int
    n_heads:   int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        attn = nn.MultiHeadDotProductAttention(
            num_heads   = self.n_heads,
            qkv_features = self.token_dim,
            out_features = self.token_dim,
        )(x, x)
        x = nn.LayerNorm()(x + attn)
        ff = nn.gelu(nn.Dense(self.token_dim * 4)(x))
        ff = nn.Dense(self.token_dim)(ff)
        return nn.LayerNorm()(x + ff)


class AgentNetwork(nn.Module):
    hidden_dim:  int = 128
    token_dim:   int = 64
    n_heads:     int = 4
    max_layers:  int = 6      # pre-allocates weights for up to 6 layers
    signal_dim:  int = 16
    symbol_dim:  int = 8
    neighbor_k:  int = 6
    obs_radius:  int = 1
    n_actions:   int = 5

    @property
    def _window(self) -> int:
        return (2 * self.obs_radius + 1) ** 2

    @nn.compact
    def __call__(
        self,
        carry:    jnp.ndarray,   # (hidden_dim,)
        x:        jnp.ndarray,   # (obs_dim,)   flat observation
        n_layers: int,           # shared team brain depth (static or jnp scalar)
    ) -> Tuple[jnp.ndarray, Tuple]:
        W    = self._window
        K    = self.neighbor_k
        sd   = self.signal_dim
        symd = self.symbol_dim

        # ── Split flat observation ────────────────────────────────────────────
        c = 0
        own  = x[c : c + 5];                    c += 5
        nb   = x[c : c + K * sd].reshape(K, sd); c += K * sd
        syms = x[c : c + W * symd].reshape(W, symd); c += W * symd
        pres = x[c : c + W * 2].reshape(W, 2);  c += W * 2
        sig  = x[c : c + sd];                   c += sd

        # ── Project each group to token_dim via Dense ─────────────────────────
        t_own  = nn.Dense(self.token_dim, name="emb_own")(own)[None]        # (1, D)
        t_nb   = nn.Dense(self.token_dim, name="emb_nb")(nb)                # (K, D)
        t_sym  = nn.Dense(self.token_dim, name="emb_sym")(syms)             # (W, D)
        t_pres = nn.Dense(self.token_dim, name="emb_pres")(pres)            # (W, D)
        t_sig  = nn.Dense(self.token_dim, name="emb_sig")(sig)[None]        # (1, D)
        t_mem  = nn.Dense(self.token_dim, name="emb_mem")(carry)[None]      # (1, D)

        tokens = jnp.concatenate([t_own, t_nb, t_sym, t_pres, t_sig, t_mem], axis=0)

        # ── Variable-depth attention: gate layers i >= n_layers to identity ───
        for i in range(self.max_layers):
            processed = _AttentionBlock(
                token_dim = self.token_dim,
                n_heads   = self.n_heads,
                name      = f"attn_{i}",
            )(tokens)
            tokens = jnp.where(i < n_layers, processed, tokens)

        readout   = tokens.mean(axis=0)                                     # (D,)
        new_carry = jnp.tanh(nn.Dense(self.hidden_dim, name="carry_proj")(readout))

        # ── Output heads ─────────────────────────────────────────────────────
        action_logits = nn.Dense(self.n_actions,   name="head_action")(readout)
        signal_out    = jnp.tanh(nn.Dense(self.signal_dim,  name="head_signal")(readout))
        symbol_write  = jnp.tanh(nn.Dense(self.symbol_dim,  name="head_symbol")(readout))
        value         = nn.Dense(1, name="head_value")(readout)[0]          # scalar

        return new_carry, (action_logits, signal_out, symbol_write, value)


# ── Per-layer introspection ───────────────────────────────────────────────────

def layer_entropy(
    model:    "AgentNetwork",
    params,
    carries:  "jnp.ndarray",   # (N, hidden_dim)
    obs:      "jnp.ndarray",   # (N, obs_dim)
    n_layers: int,
) -> dict:
    """
    Per-active-layer output statistics measured by running the network at each
    depth d = 1..n_layers and observing how the output changes.

    Returns dict with keys "layer_1" .. "layer_N", each containing:
      action_entropy — mean H(action distribution); higher = more uncertain/exploratory
      signal_norm    — mean L2 norm of emitted signal vector
      logit_delta    — mean abs logit change vs previous depth (0 for layer 1)

    A new layer showing non-zero logit_delta + different entropy vs the layer below
    it means the layer is doing real computation, not just passing through.
    """
    import numpy as np

    stats  = {}
    prev_logits = None
    for li in range(1, n_layers + 1):
        fwd_i = make_forward_fn(model, li)
        _, (logits_i, sigs_i, _, _) = fwd_i(params, carries, obs)
        ln = np.array(logits_i)                                  # (N, 5)
        sn = np.array(sigs_i)                                    # (N, sig_dim)

        p   = jax.nn.softmax(jnp.array(ln), axis=-1)
        ent = float(-jnp.sum(p * jnp.log(p + 1e-12), axis=-1).mean())

        delta = 0.0
        if prev_logits is not None:
            delta = float(np.abs(ln - prev_logits).mean())
        prev_logits = ln

        stats[f"layer_{li}"] = {
            "action_entropy": ent,
            "signal_norm":    float(np.linalg.norm(sn, axis=-1).mean()),
            "logit_delta":    delta,
        }
    return stats


# ── Observation dimension helper ─────────────────────────────────────────────

def compute_obs_dim(config: dict) -> int:
    K    = config["neighbor_k"]
    sd   = config["signal_dim"]
    symd = config.get("symbol_dim", 8)
    r    = config["local_obs_radius"]
    W    = (2 * r + 1) ** 2
    return 5 + K * sd + W * symd + W * 2 + sd


# ── Forward function factory (cached per n_layers) ───────────────────────────

@lru_cache(maxsize=8)
def make_forward_fn(model: AgentNetwork, n_layers: int):
    """Return a jit-compiled forward function for this brain depth.

    Signature: (params, carries, obs) -> (new_carries, (action_logits, signal_out, symbol_write, values))
    - params   : shared PyTree (NOT per-agent)
    - carries  : (max_pop, hidden_dim) per-agent memory
    - obs      : (max_pop, obs_dim)    per-agent observations

    Cached per n_layers value so brain growth just looks up a new fn.
    """
    def forward(params, carries, obs):
        def _agent(carry, x):
            return model.apply(params, carry, x, n_layers)
        return jax.vmap(_agent)(carries, obs)

    return jax.jit(forward)
