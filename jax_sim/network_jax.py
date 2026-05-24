"""
jax_sim/network_jax.py — Flax transformer with discrete token outputs.

Architecture (same as PyTorch version):
  - Input tokens: own_state(6), nb_signals(K, sig_dim), local_symbols(W, sym_dim),
    env_channels(W, 7), own_signal(sig_dim), episodic_memory(mem_slots, sig_dim+2),
    cultural_fast(W, sym_dim), cultural_slow(W, sym_dim)
  - Embed each token → d_model=128
  - Transformer blocks (4-6 layers, 4 heads)
  - Output heads: action(5), signal(vocab=256), symbol_write(sym_dim),
    value(1), tom(K, 5), culture_fast(sym_dim), culture_slow(sym_dim)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple


class AgentNetworkJax(nn.Module):
    """
    Transformer-based agent brain.
    All hyperparameters are module attributes for easy instantiation.
    """
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    obs_dim: int = 2285
    signal_dim: int = 32
    symbol_dim: int = 16
    vocab_size: int = 256
    max_layers: int = 6
    memory_slots: int = 0

    def setup(self):
        d = self.hidden_dim
        sym_d = self.symbol_dim

        # Token embeddings
        self.emb_own = nn.Dense(d)          # own_state (6) -> token
        self.emb_nb = nn.Dense(d)           # nb_signals -> token per neighbour
        self.emb_sym = nn.Dense(d)          # local_symbols -> token per cell
        self.emb_env = nn.Dense(d)          # env_channels -> token per cell
        self.emb_sig = nn.Dense(d)          # own_signal -> token
        self.emb_mem = nn.Dense(d)          # episodic memory -> token per slot
        self.emb_cult = nn.Dense(d)         # cultural memory -> token per cell

        # Positional encoding for transformer
        self.pos_enc = nn.Embed(num_embeddings=64, features=d)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d, self.n_heads) for _ in range(self.max_layers)
        ]

        # Output heads
        self.head_action = nn.Dense(5)           # 5 actions
        self.head_signal = nn.Dense(self.vocab_size)  # discrete vocab
        self.head_symbol = nn.Dense(sym_d)       # symbol write
        self.head_value = nn.Dense(1)            # value estimate
        self.head_tom = nn.Dense(5)             # Theory-of-Mind per neighbour
        self.head_culture_fast = nn.Dense(sym_d)
        self.head_culture_slow = nn.Dense(sym_d)

    def __call__(
        self,
        carries: jnp.ndarray,   # (N, hidden_dim)
        obs: jnp.ndarray,       # (N, obs_dim)
        n_layers: int,
        nb_gain: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Tuple]:
        """
        Forward pass.
        Returns: (new_carries, outputs_tuple)
        outputs_tuple = (
            action_logits, signal_logits, symbol_write, values,
            tom_logits, token_ids, signal_probs, culture_fast, culture_slow
        )
        """
        jax.debug.print("[NET] carries NaN={n} inf={i} max={m}", n=jnp.isnan(carries).any(), i=jnp.isinf(carries).any(), m=jnp.max(jnp.abs(carries)))
        N = obs.shape[0]
        d = self.hidden_dim
        sym_d = self.symbol_dim
        K = 6  # neighbour_k (hardcoded for speed; should come from config)
        W = 25  # (2*r+1)**2 with r=2 (5x5 patch)

        # Split observation vector
        own_state = obs[:, :6]
        nb_sigs = obs[:, 6:6 + K * self.signal_dim].reshape(N, K, self.signal_dim)
        loc_sym = obs[:, 6 + K * self.signal_dim:6 + K * self.signal_dim + W * sym_d].reshape(N, W, sym_d)
        env_ch = 7
        loc_env = obs[:, 6 + K * self.signal_dim + W * sym_d:6 + K * self.signal_dim + W * sym_d + W * env_ch].reshape(N, W, env_ch)
        own_sig = obs[:, 6 + K * self.signal_dim + W * sym_d + W * env_ch:6 + K * self.signal_dim + W * sym_d + W * env_ch + self.signal_dim]

        idx = 6 + K * self.signal_dim + W * sym_d + W * env_ch + self.signal_dim

        # Optional memory buffer
        if self.memory_slots > 0:
            mem = obs[:, idx:idx + self.memory_slots * (self.signal_dim + 2)].reshape(N, self.memory_slots, self.signal_dim + 2)
            idx += self.memory_slots * (self.signal_dim + 2)
        else:
            mem = None

        loc_cult_fast = obs[:, idx:idx + W * sym_d].reshape(N, W, sym_d)
        idx += W * sym_d
        loc_cult_slow = obs[:, idx:idx + W * sym_d].reshape(N, W, sym_d)

        # Embed tokens
        t1 = self.emb_own(own_state)[:, None, :]  # (N, 1, d)
        t2 = self.emb_nb(nb_sigs)                 # (N, K, d)
        t3 = self.emb_sym(loc_sym)                # (N, W, d)
        t4 = self.emb_env(loc_env)                # (N, W, d)
        t5 = self.emb_sig(own_sig)[:, None, :]    # (N, 1, d)
        tokens = [t1, t2, t3, t4, t5]
        if mem is not None:
            tokens.append(self.emb_mem(mem))      # (N, mem_slots, d)
        tokens.append(self.emb_cult(loc_cult_fast))     # (N, W, d)
        tokens.append(self.emb_cult(loc_cult_slow))    # (N, W, d)
        jax.debug.print("[NET] emb_own NaN={n}", n=jnp.isnan(t1).any())
        jax.debug.print("[NET] emb_nb NaN={n}", n=jnp.isnan(t2).any())
        jax.debug.print("[NET] emb_sym NaN={n}", n=jnp.isnan(t3).any())
        jax.debug.print("[NET] emb_env NaN={n}", n=jnp.isnan(t4).any())
        jax.debug.print("[NET] emb_sig NaN={n}", n=jnp.isnan(t5).any())

        x = jnp.concatenate(tokens, axis=1)  # (N, T, d)
        T = x.shape[1]
        jax.debug.print("[NET] T={t}", t=T)

        # Add positional encoding
        pos_ids = jnp.arange(T)
        pos_emb = self.pos_enc(pos_ids)[None, :, :]
        jax.debug.print("[NET] pos_emb NaN={n}", n=jnp.isnan(pos_emb).any())
        x = x + pos_emb  # (N, T, d)
        jax.debug.print("[NET] after_pos NaN={n}", n=jnp.isnan(x).any())

        # Carry fusion: add carry as a global bias to all tokens
        carry_broadcast = carries[:, None, :]
        jax.debug.print("[NET] carry_broadcast NaN={n}", n=jnp.isnan(carry_broadcast).any())
        x = x + carry_broadcast  # (N, T, d)
        jax.debug.print("[NET] after_fusion NaN={n}", n=jnp.isnan(x).any())

        # Transformer blocks (only first n_layers are active)
        for i in range(n_layers):
            x = self.blocks[i](x)
            jax.debug.print("[NET] after_block{i} NaN={n}", i=i, n=jnp.isnan(x).any())

        # Pool across tokens for global representation
        pooled = x.mean(axis=1)  # (N, d)
        jax.debug.print("[NET] pooled NaN={n}", n=jnp.isnan(pooled).any())

        # Update carries with pooled representation
        new_carries = 0.9 * carries + 0.1 * pooled  # soft update

        # Output heads
        action_logits = self.head_action(pooled)           # (N, 5)
        signal_logits = self.head_signal(pooled)            # (N, vocab_size)
        symbol_write = self.head_symbol(pooled)              # (N, sym_d)
        values = self.head_value(pooled).squeeze(-1)        # (N,)
        jax.debug.print("[NET] action_logits NaN={n}", n=jnp.isnan(action_logits).any())
        jax.debug.print("[NET] values NaN={n}", n=jnp.isnan(values).any())
        tom_logits = self.head_tom(pooled)[:, None, :]     # (N, 1, 5) — simplified; real version needs K
        tom_logits = jnp.broadcast_to(tom_logits, (N, K, 5))  # (N, K, 5)
        culture_fast = self.head_culture_fast(pooled)       # (N, sym_d)
        culture_slow = self.head_culture_slow(pooled)       # (N, sym_d)

        # Discrete token sampling
        token_ids = jnp.argmax(signal_logits, axis=-1)      # (N,)
        signal_probs = jax.nn.softmax(signal_logits, axis=-1)  # (N, vocab_size)

        return new_carries, (
            action_logits, signal_logits, symbol_write, values,
            tom_logits, token_ids, signal_probs, culture_fast, culture_slow,
        )


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""
    hidden_dim: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (N, T, d)
        d = self.hidden_dim

        # Pre-norm + attention
        x_norm = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=d,
            out_features=d,
            dropout_rate=0.0,
        )(x_norm, x_norm)
        x = x + attn_out

        # Pre-norm + MLP
        x_norm = nn.LayerNorm()(x)
        mlp_out = nn.Dense(d * 4)(x_norm)
        mlp_out = nn.relu(mlp_out)
        mlp_out = nn.Dense(d)(mlp_out)
        x = x + mlp_out

        return x
