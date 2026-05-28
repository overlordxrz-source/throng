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
from flax.core import freeze, unfreeze
from typing import Any, Optional, Tuple

# Params created only in auxiliary_heads (not touched by __call__ during init).
AUX_HEAD_KEYS = ("head_fwd_1", "head_fwd_2", "head_self_pred")


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
        self.pos_enc = nn.Embed(num_embeddings=256, features=d)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d, self.n_heads) for _ in range(self.max_layers)
        ]

        # Output heads
        self.final_norm = nn.LayerNorm()
        self.head_action = nn.Dense(5)           # 5 actions
        self.head_signal = nn.Dense(self.vocab_size)  # discrete vocab
        self.head_symbol = nn.Dense(sym_d)       # symbol write
        self.head_value = nn.Dense(1, kernel_init=nn.initializers.normal(0.01), bias_init=nn.initializers.zeros)  # zero init for stable value learning
        self.head_tom = nn.Dense(5)             # Theory-of-Mind per neighbour
        self.head_culture_fast = nn.Dense(sym_d)
        self.head_culture_slow = nn.Dense(sym_d)

        # Forward dynamics head: predicts carry_{t+1} from carry_t + action_onehot
        self.head_fwd_1 = nn.Dense(self.hidden_dim * 4)
        self.head_fwd_2 = nn.Dense(self.hidden_dim)

        # Self-prediction head (Phase 9.1): predicts own next action from carry_t
        self.head_self_pred = nn.Dense(5)

    def __call__(
        self,
        carries: jnp.ndarray,   # (N, hidden_dim)
        obs: jnp.ndarray,       # (N, obs_dim)
        n_layers: int,
        nb_gain: Optional[jnp.ndarray] = None,
        detach_value: bool = False,
    ) -> Tuple[jnp.ndarray, Tuple]:
        """
        Forward pass.
        Returns: (new_carries, outputs_tuple)
        outputs_tuple = (
            action_logits, signal_logits, symbol_write, values,
            tom_logits, token_ids, signal_probs, culture_fast, culture_slow
        )
        """
        N = obs.shape[0]
        d = self.hidden_dim
        sym_d = self.symbol_dim
        K = 6  # neighbour_k (hardcoded for speed; should come from config)
        W = 25  # (2*r+1)**2 with r=2 (5x5 patch)

        # Split observation vector
        own_state = obs[:, :6]
        nb_sigs = obs[:, 6:6 + K * self.signal_dim].reshape(N, K, self.signal_dim)
        loc_sym = obs[:, 6 + K * self.signal_dim:6 + K * self.signal_dim + W * sym_d].reshape(N, W, sym_d)
        env_ch = 8
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

        x = jnp.concatenate(tokens, axis=1)  # (N, T, d)
        T = x.shape[1]

        # Add positional encoding
        pos_ids = jnp.arange(T)
        x = x + self.pos_enc(pos_ids)[None, :, :]  # (N, T, d)

        # Carry fusion: add carry as a global bias to all tokens
        x = x + carries[:, None, :]  # (N, T, d)

        # Transformer blocks (only first n_layers are active)
        for i in range(n_layers):
            x = self.blocks[i](x)

        # Pool across tokens for global representation
        pooled = x.mean(axis=1)  # (N, d)
        pooled = self.final_norm(pooled)

        # Update carries with pooled representation
        new_carries = 0.9 * carries + 0.1 * pooled  # soft update

        # Detach pooled for value head so value gradients don't corrupt shared representation
        value_input = jax.lax.stop_gradient(pooled) if detach_value else pooled

        # Output heads
        action_logits = self.head_action(pooled) / 2.0   # (N, 5)  temperature=2.0 for exploration
        signal_logits = self.head_signal(pooled)            # (N, vocab_size)
        symbol_write = self.head_symbol(pooled)              # (N, sym_d)
        values = self.head_value(value_input).squeeze(-1)  # (N,)  unbounded, Huber loss prevents explosion
        tom_logits = self.head_tom(pooled)[:, None, :]     # (N, 1, 5) — simplified; real version needs K
        tom_logits = jnp.broadcast_to(tom_logits, (N, K, 5))  # (N, K, 5)
        culture_fast = self.head_culture_fast(pooled)       # (N, sym_d)
        culture_slow = self.head_culture_slow(pooled)       # (N, sym_d)

        # Discrete token sampling
        token_ids = jnp.argmax(signal_logits, axis=-1)      # (N,)
        signal_probs = jax.nn.softmax(signal_logits, axis=-1)  # (N, vocab_size)

        # Register auxiliary-head params in the same init as __call__ (Flax idiom).
        if self.is_initializing():
            _action_oh = jnp.zeros((N, 5), dtype=obs.dtype)
            self.auxiliary_heads(carries, _action_oh)

        return new_carries, (
            action_logits, signal_logits, symbol_write, values,
            tom_logits, token_ids, signal_probs, culture_fast, culture_slow,
        )

    def forward_dynamics(
        self,
        carry_t: jnp.ndarray,    # (N, hidden_dim)
        action_oh: jnp.ndarray,  # (N, 5)
    ) -> jnp.ndarray:
        """
        Predict carry_{t+1} from carry_t + action_onehot.
        IMPORTANT: caller must stop_gradient the target carry_{t+1}.
        """
        inp = jnp.concatenate([carry_t, action_oh], axis=-1)
        h = nn.relu(self.head_fwd_1(inp))
        return self.head_fwd_2(h)

    def auxiliary_heads(
        self,
        carry_t: jnp.ndarray,    # (N, hidden_dim)
        action_oh: jnp.ndarray,  # (N, 5)  — action taken at t
    ) -> tuple:
        """
        Compute both auxiliary predictions from carry_t in one forward pass.

        Returns:
          carry_pred       (N, hidden_dim) — predicted carry_{t+1} (forward dynamics)
          self_pred_logits (N, 5)          — predicted action_{t+1} (self-prediction)

        Caller must stop_gradient carry_{t+1} before computing fwd loss.
        Self-prediction accuracy > 0.20 (random baseline) signals the agent
        has built a model of its own future behaviour.
        """
        # Forward dynamics
        fwd_inp = jnp.concatenate([carry_t, action_oh], axis=-1)
        carry_pred = self.head_fwd_2(nn.relu(self.head_fwd_1(fwd_inp)))

        # Self-prediction: what action will I take next?
        self_pred_logits = self.head_self_pred(carry_t)

        return carry_pred, self_pred_logits


def init_agent_params(
    model: AgentNetworkJax,
    rng: jax.Array,
    carry: jnp.ndarray,
    obs: jnp.ndarray,
    n_layers: int,
) -> Any:
    """
    Initialize full parameter tree (main + auxiliary heads).

    Auxiliary heads are touched inside ``__call__`` when ``is_initializing()``.
    """
    return sanitize_agent_params(model.init(rng, carry, obs, n_layers)["params"])


def params_apply_variables(params: Any) -> dict:
    """Wrap a params PyTree for ``Module.apply`` (explicit params collection)."""
    return {"params": params}


def sanitize_agent_params(params: Any) -> Any:
    """
    Fix param trees corrupted by merging full Flax variable dicts.

    A mistaken merge can leave an empty top-level ``params`` collection next to
    real module keys (``emb_own``, …). ``apply`` then uses the empty collection
    and raises ScopeCollectionNotFound even though ``emb_own`` looks fine in a
    flat dict inspection.
    """
    flat = unfreeze(params)
    if "params" not in flat or not isinstance(flat["params"], dict):
        return params
    inner = flat["params"]
    if inner and ("emb_own" in inner or "head_action" in inner):
        merged = dict(inner)
        for k in AUX_HEAD_KEYS:
            if k in flat and k not in merged:
                merged[k] = flat[k]
        return freeze(merged)
    if not inner:
        cleaned = {k: v for k, v in flat.items() if k != "params"}
        return freeze(cleaned)
    return params


def ensure_aux_head_params(
    model: AgentNetworkJax,
    params: Any,
    rng: jax.Array,
    hidden_dim: int,
) -> Any:
    """Fill missing auxiliary-head params when resuming an older checkpoint."""
    flat = unfreeze(params)
    if all(k in flat for k in AUX_HEAD_KEYS):
        return params
    carry = jnp.zeros((1, hidden_dim))
    action_oh = jnp.zeros((1, 5), dtype=jnp.float32)
    aux_only = model.init(
        rng, carry, action_oh, method=model.auxiliary_heads
    )["params"]
    aux_flat = unfreeze(aux_only)
    for k in AUX_HEAD_KEYS:
        if k not in flat:
            flat[k] = aux_flat[k]
    print("[JAX] Merged fresh auxiliary-head params into restored checkpoint")
    return sanitize_agent_params(freeze(flat))


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
