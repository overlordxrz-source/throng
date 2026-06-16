"""
jax_sim/network_jax.py — Flax transformer with discrete token outputs.

Architecture (same as PyTorch version):
  - Input tokens: own_state(6), nb_signals(K, sig_dim), local_symbols(W, sym_dim),
    env_channels(W, 7), own_signal(sig_dim), episodic_memory(mem_slots, sig_dim+2),
    cultural_fast(W, sym_dim), cultural_slow(W, sym_dim)
  - Embed each token → d_model=128
  - Transformer blocks (4-6 layers, 4 heads)
  - Output heads: action(5), signal via VQ codebook(vocab_size × signal_dim),
    symbol_write(sym_dim), value(1), tom(K, 5), culture_fast(sym_dim), culture_slow(sym_dim)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.core.frozen_dict import FrozenDict
from typing import Any, List, Optional, Tuple
from jax_sim.obs_layout import make_obs_layout

# Params created only in auxiliary_heads (not touched by __call__ during init).
AUX_HEAD_KEYS = (
    "head_fwd_1",
    "head_fwd_2",
    "head_self_pred",
    "head_fwd_dyn_1",
    "head_fwd_dyn_2",
    "head_confidence_1",
    "head_confidence_2",
    "head_proprio",
)

# Top-level module keys grafted from a fresh init when missing in Orbax checkpoints.
CHECKPOINT_GRAFT_TOP_KEYS = AUX_HEAD_KEYS + (
    "nb_cross_attn",
    "head_vqel_recon_1",
    "head_vqel_recon_2",
)


def vector_quantize_signals(
    z_e: jnp.ndarray,
    codebook: jnp.ndarray,
    beta: float = 0.25,
    dead_code_reset: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Hardened Discrete Bottleneck (Phase 17).
    Sever the continuous gradient path by using a straight-through estimator
    over a frozen codebook.

    z_e: (N, signal_dim) continuous pre-quantization vectors.
    codebook: (vocab_size, signal_dim) embedding table.
    """
    vocab_size = codebook.shape[0]

    # Calculate logits based on negative distance
    diff = z_e[:, None, :] - jax.lax.stop_gradient(codebook[None, :, :])
    dist_sq = jnp.sum(diff * diff, axis=-1)
    
    # Gumbel Softmax (STE approximation)
    logits = -dist_sq
    y_soft = jax.nn.softmax(logits)
    token_ids = jnp.argmin(dist_sq, axis=-1)
    y_hard = jax.nn.one_hot(token_ids, vocab_size, dtype=y_soft.dtype)
    
    # Straight-through estimator
    y = y_soft + jax.lax.stop_gradient(y_hard - y_soft)
    
    # Output is mapped back to 32D through the FROZEN codebook.
    signal_out = y @ jax.lax.stop_gradient(codebook)
    
    # Set VQ loss to zero to prevent old VQ dynamics from pulling the codebook
    loss_vq = jnp.zeros(z_e.shape[0])
    
    return signal_out, token_ids, loss_vq


def dead_code_reset_codebook_params(
    params: Any,
    token_ids: jnp.ndarray,
    z_e: jnp.ndarray,
    vocab_size: int,
    rng: jax.Array,
    codebook_key: str = "codebook",
) -> Any:
    """
  Persist dead-code reset into the codebook embedding (runs after PPO on CPU/GPU).

  token_ids: (M,) int — tokens used in the rollout window (alive agents only).
  z_e: (M, signal_dim) matching encoder outputs for those rows.
  codebook_key: ``codebook`` (blue) or ``red_codebook`` (Phase 12 predator).
    """
    if z_e.shape[0] == 0:
        return params
    flat = unfreeze(params)
    cb = flat[codebook_key]["embedding"]
    usage = jnp.bincount(token_ids, length=vocab_size)
    dead_mask = usage == 0
    n_pool = z_e.shape[0]
    rand_idx = jax.random.randint(rng, (vocab_size,), 0, n_pool)
    replacement = jax.lax.stop_gradient(z_e[rand_idx])
    flat[codebook_key] = {
        **flat[codebook_key],
        "embedding": jnp.where(dead_mask[:, None], replacement, cb),
    }
    # Match container type — optax Adam state must stay aligned with params tree.
    return freeze(flat) if isinstance(params, FrozenDict) else flat


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
    vocab_size: int = 64
    vq_beta: float = 0.25
    vq_dead_code_reset: bool = True
    max_layers: int = 6
    memory_slots: int = 0
    fwd_env_dim: int = 200   # W × 8 flattened loc_env (set from config in main_jax)
    cross_attn_enabled: bool = False
    cross_attn_num_heads: int = 4
    neighbor_k: int = 6
    local_cells: int = 25
    env_channels: int = 9
    own_state_dim: int = 10
    n_actions: int = 8

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
        self.head_action = nn.Dense(self.n_actions)           # N actions (N, S, E, W, Stay, Strike, Push, Guard, [Build])
        self.head_signal = nn.Dense(self.signal_dim)  # continuous z_e pre-VQ
        self.codebook = nn.Embed(self.vocab_size, self.signal_dim)
        self.head_alarm = nn.Dense(2)            # Phase 17.5: 1-bit discrete alarm channel
        self.head_symbol = nn.Dense(sym_d)       # symbol write
        self.head_value = nn.Dense(1, kernel_init=nn.initializers.normal(0.01), bias_init=nn.initializers.zeros)  # zero init for stable value learning
        self.head_tom = nn.Dense(self.n_actions)             # Theory-of-Mind per neighbour
        self.head_culture_fast = nn.Dense(sym_d)
        self.head_culture_slow = nn.Dense(sym_d)

        # Phase 16.6 GWT Router for Blue
        self.gwt_comms_1 = nn.Dense(d)

        # Forward dynamics head: predicts next-step flat loc_env from carry_t + action
        self.head_fwd_1 = nn.Dense(self.hidden_dim * 4)
        self.head_fwd_2 = nn.Dense(self.fwd_env_dim)

        # Self-prediction head (Phase 9.1): predicts own next action from carry_t
        self.head_self_pred = nn.Dense(self.n_actions)

        # Latent forward dynamics (Phase 11 / 9.2): carry_{t+1} from [carry_t, action_t]
        self.head_fwd_dyn_1 = nn.Dense(self.hidden_dim * 4)
        self.head_fwd_dyn_2 = nn.Dense(self.hidden_dim)

        # Phase 9.1 — epistemic confidence: predict carry_fwd MSE from [carry_t, action_t]
        self.head_confidence_1 = nn.Dense(self.hidden_dim * 4)
        self.head_confidence_2 = nn.Dense(1)
        # Phase 14.1a scaffold — keep decoder shallow so codebook carries semantics.
        self.head_vqel_recon_1 = nn.Dense(self.hidden_dim)
        self.head_vqel_recon_2 = nn.Dense(self._obs_layout().spatial_ego_dim)
        # Phase 14.1b — proprioceptive disentanglement (energy from carry, not VQ wire).
        self.head_proprio = nn.Dense(1)

        if self.cross_attn_enabled:
            self.nb_cross_attn = NeighborCrossAttention(
                hidden_dim=d,
                num_heads=self.cross_attn_num_heads,
            )

    def _obs_layout(self):
        return make_obs_layout(
            signal_dim=self.signal_dim,
            symbol_dim=self.symbol_dim,
            memory_slots=self.memory_slots,
            neighbor_k=self.neighbor_k,
            local_cells=self.local_cells,
            env_channels=self.env_channels,
            own_state_dim=self.own_state_dim,
        )

    def extract_spatial_ego(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Phase 14.1a target: own_state + local env channels (flattened).
        This is the monologue reconstruction target before social dialogue.
        """
        layout = self._obs_layout()
        own_state = obs[:, layout.own_state_start : layout.own_state_end]
        loc_env_flat = obs[:, layout.loc_env_start : layout.loc_env_end]
        return jnp.concatenate([own_state, loc_env_flat], axis=-1)

    def __call__(
        self,
        carries: jnp.ndarray,   # (N, hidden_dim)
        obs: jnp.ndarray,       # (N, obs_dim)
        n_layers: int,
        nb_gain: Optional[jnp.ndarray] = None,
        detach_value: bool = False,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Tuple]:
        """
        Forward pass.
        Returns: (new_carries, outputs_tuple)
        outputs_tuple = (
            action_logits, signal_out, symbol_write, values,
            tom_logits, token_ids, loss_vq, z_e, culture_fast, culture_slow,
        )
        """
        N = obs.shape[0]
        d = self.hidden_dim
        sym_d = self.symbol_dim
        layout = self._obs_layout()
        K = layout.neighbor_k
        W = layout.local_cells

        # Split observation vector
        own_state = obs[:, layout.own_state_start : layout.own_state_end]
        nb_sigs = obs[:, layout.nb_sigs_start : layout.nb_sigs_end].reshape(
            N, K, self.signal_dim
        )
        nb_alarms = obs[:, layout.nb_alarms_start : layout.nb_alarms_end].reshape(
            N, K, 2
        )
        nb_combined = jnp.concatenate([nb_sigs, nb_alarms], axis=-1)
        loc_sym = obs[:, layout.loc_sym_start : layout.loc_sym_end].reshape(
            N, W, sym_d
        )
        loc_env = obs[:, layout.loc_env_start : layout.loc_env_end].reshape(
            N, W, layout.env_channels
        )
        own_sig = obs[:, layout.own_sig_start : layout.own_sig_end]
        own_alarm = obs[:, layout.own_alarm_start : layout.own_alarm_end]
        own_combined = jnp.concatenate([own_sig, own_alarm], axis=-1)

        idx = layout.mem_start

        # Optional memory buffer
        if self.memory_slots > 0:
            mem = obs[:, idx : idx + self.memory_slots * (self.signal_dim + 2)].reshape(
                N, self.memory_slots, self.signal_dim + 2
            )
            idx += self.memory_slots * (self.signal_dim + 2)
        else:
            mem = None

        loc_cult_fast = obs[:, idx:idx + W * sym_d].reshape(N, W, sym_d)
        idx += W * sym_d
        loc_cult_slow = obs[:, idx:idx + W * sym_d].reshape(N, W, sym_d)

        # Embed tokens
        self_encoded = self.emb_own(own_state)    # (N, d)
        t1 = self_encoded[:, None, :]             # (N, 1, d)
        nb_kv = self.emb_nb(nb_combined)          # (N, K, d)
        if self.cross_attn_enabled:
            processed = self.nb_cross_attn(self_encoded, carries, nb_kv)
            t2 = processed[:, None, :]            # (N, 1, d) — attended Other
        else:
            t2 = nb_kv                            # (N, K, d) — per-neighbor tokens
        t3 = self.emb_sym(loc_sym)                # (N, W, d)
        t4 = self.emb_env(loc_env)                # (N, W, d)
        t5 = self.emb_sig(own_combined)[:, None, :] # (N, 1, d)
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
        action_logits = self.head_action(pooled) / 2.0   # (N, 8)  temperature=2.0 for exploration
        
        # Phase 16.6 GWT Router for Blue
        # Zero out age(0), mat(1), energy(2), layers(3) to completely sever the metabolic leak
        exteroceptive_obs = obs.at[:, :4].set(0.0)
        h_comms = nn.relu(self.gwt_comms_1(exteroceptive_obs))
        z_e = self.head_signal(h_comms)                 # (N, signal_dim)
        codebook_w = self.codebook.embedding           # (vocab_size, signal_dim)
        signal_out, token_ids, loss_vq = vector_quantize_signals(
            z_e,
            codebook_w,
            beta=self.vq_beta,
            dead_code_reset=self.vq_dead_code_reset,
        )
        
        # Phase 17.5.1: Timescale Grammar 1-bit discrete alarm channel
        alarm_logits = self.head_alarm(h_comms) # (N, 2)
        # alarm_out is now raw logits, we sample in the environment step
        alarm_out = alarm_logits
        
        symbol_write = self.head_symbol(pooled)              # (N, sym_d)
        feral_mask = jax.lax.stop_gradient(obs[:, 2] < 0.20)
        symbol_write = jnp.where(feral_mask[:, None], 0.0, symbol_write)
        signal_out = jnp.where(feral_mask[:, None], 0.0, signal_out)
        
        # Feral mask forces the alarm logit to highly favor class 0 (silent)
        alarm_silence = jnp.array([10.0, -10.0], dtype=alarm_out.dtype) 
        alarm_out = jnp.where(feral_mask[:, None], alarm_silence[None, :], alarm_out)
        values = self.head_value(value_input).squeeze(-1)  # (N,)  unbounded, Huber loss prevents explosion
        tom_logits = self.head_tom(pooled)[:, None, :]     # (N, 1, 8) — simplified; real version needs K
        tom_logits = jnp.broadcast_to(tom_logits, (N, K, self.n_actions))  # (N, K, n_actions)
        culture_fast = self.head_culture_fast(pooled)       # (N, sym_d)
        culture_slow = self.head_culture_slow(pooled)       # (N, sym_d)

        # Register auxiliary-head params in the same init as __call__ (Flax idiom).
        if self.is_initializing():
            _action_oh = jnp.zeros((N, self.n_actions), dtype=obs.dtype)
            self.auxiliary_heads(carries, _action_oh)
            _zq_seed = jnp.zeros_like(z_e)
            self.head_vqel_recon_2(nn.relu(self.head_vqel_recon_1(_zq_seed)))

        return new_carries, (
            action_logits, signal_out, symbol_write, values,
            tom_logits, token_ids, alarm_out, loss_vq, z_e, culture_fast, culture_slow,
        )

    def monologue_forward(
        self,
        carries: jnp.ndarray,
        obs: jnp.ndarray,
        n_layers: int,
    ) -> tuple:
        """
        Phase 14.1a scaffold: reconstruct spatial ego state through discrete codebook.
        Returns (z_q, token_ids, spatial_ego_hat, spatial_ego_target, loss_vq, z_e).
        """
        layout = self._obs_layout()
        obs_masked = obs.at[:, layout.nb_sigs_start : layout.nb_sigs_end].set(0.0)
        obs_masked = obs_masked.at[:, layout.own_sig_start : layout.own_sig_end].set(0.0)

        _, outs = self(carries, obs_masked, n_layers)
        token_ids = outs[5]
        alarm_out = outs[6]
        loss_vq = outs[7]
        z_e = outs[8]
        z_q = z_e + jax.lax.stop_gradient(outs[1] - z_e)
        spatial_ego_hat = self.head_vqel_recon_2(nn.relu(self.head_vqel_recon_1(z_q)))
        spatial_ego_target = self.extract_spatial_ego(obs)
        return z_q, token_ids, spatial_ego_hat, spatial_ego_target, loss_vq, z_e

    def forward_dynamics(
        self,
        carry_t: jnp.ndarray,    # (N, hidden_dim)
        action_oh: jnp.ndarray,  # (N, 8)
    ) -> jnp.ndarray:
        """
        Predict flat loc_env_{t+1} from carry_t + action_onehot.
        IMPORTANT: caller must stop_gradient the target loc_env_{t+1}.
        """
        inp = jnp.concatenate([carry_t, action_oh], axis=-1)
        h = nn.relu(self.head_fwd_1(inp))
        return self.head_fwd_2(h)

    def carry_forward_dynamics(
        self,
        carry_t: jnp.ndarray,    # (N, hidden_dim)
        action_oh: jnp.ndarray,  # (N, 8)
    ) -> jnp.ndarray:
        """
        Predict carry_{t+1} from carry_t + action_onehot.
        Caller must stop_gradient carry_{t+1} before computing MSE loss.
        """
        fwd_inp = jnp.concatenate([carry_t, action_oh], axis=-1)
        h = nn.relu(self.head_fwd_dyn_1(fwd_inp))
        return self.head_fwd_dyn_2(h)

    def predict_carry_fwd_confidence(
        self,
        carry_t: jnp.ndarray,
        action_oh: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict expected carry forward MSE from [carry_t, action_t] (Phase 9.1).
        Returns (N,) non-negative scalars (softplus).
        """
        fwd_inp = jnp.concatenate([carry_t, action_oh], axis=-1)
        h = nn.relu(self.head_confidence_1(fwd_inp))
        return nn.softplus(self.head_confidence_2(h)).squeeze(-1)

    def auxiliary_heads(
        self,
        carry_t: jnp.ndarray,    # (N, hidden_dim)
        action_oh: jnp.ndarray,  # (N, 8)  — action taken at t
    ) -> tuple:
        """
        Compute auxiliary predictions from carry_t in one forward pass.

        Returns:
          env_pred         (N, fwd_env_dim) — predicted flat loc_env_{t+1}
          self_pred_logits (N, 8)           — predicted action_{t+1}
          carry_pred       (N, hidden_dim)  — predicted carry_{t+1}
          conf_pred        (N,)             — predicted carry_fwd MSE (Phase 9.1)

        Caller must stop_gradient loc_env_{t+1} and carry_{t+1} before loss.
        """
        fwd_inp = jnp.concatenate([carry_t, action_oh], axis=-1)
        env_pred = self.head_fwd_2(nn.relu(self.head_fwd_1(fwd_inp)))
        self_pred_logits = self.head_self_pred(carry_t)
        carry_pred = self.carry_forward_dynamics(carry_t, action_oh)
        conf_pred = self.predict_carry_fwd_confidence(carry_t, action_oh)
        energy_pred = self.predict_proprio_energy(carry_t)
        return env_pred, self_pred_logits, carry_pred, conf_pred, energy_pred

    def predict_proprio_energy(self, carry_t: jnp.ndarray) -> jnp.ndarray:
        """Predict next-step energy from carry (Phase 14.1b proprio aux)."""
        return self.head_proprio(carry_t).squeeze(-1)

    def value_from_carry(self, carry: jnp.ndarray) -> jnp.ndarray:
        """Latent value readout for K-step imagination (carry-only; frozen at inference)."""
        return self.head_value(carry).squeeze(-1)


# Predator (Red) comms — separate codebook + cross-attn; no P11 aux / imagination heads.
PREDATOR_GRAFT_TOP_KEYS = ("dcvq", "simvq_W", "red_nb_cross_attn", "head_proprio", "gwt_comms_1", "carry_gru")
PREDATOR_VQ_COLD_RESTART_KEYS = ("gwt_comms_1", "head_signal", "dcvq", "simvq_W", "red_codebook", "red_nb_cross_attn")

# SimVQ W bound — stateless forward pass only (no new params; Orbax ckpt-compatible).
SIMVQ_W_CLIP_DEFAULT = 2.0
SIMVQ_OUT_SCALE_DEFAULT = 2.0


def simvq_bounded_project(
    z: jnp.ndarray,
    W: jnp.ndarray,
    w_clip: float = SIMVQ_W_CLIP_DEFAULT,
    out_scale: float = SIMVQ_OUT_SCALE_DEFAULT,
) -> jnp.ndarray:
    """Clip W at matmul time, then tanh-scale the projection (P15 stabilization)."""
    W_eff = jnp.clip(W, -w_clip, w_clip)
    return jnp.tanh(jnp.matmul(z, W_eff)) * out_scale


class DCVQ(nn.Module):
    n_e: int
    e_dim: int
    num_subspaces: int
    beta: float = 0.25

    def setup(self):
        assert self.e_dim % self.num_subspaces == 0
        self.sub_dim = self.e_dim // self.num_subspaces
        self.embeddings = self.param('subspace_embeddings',
                                     nn.initializers.normal(stddev=1.0 / self.sub_dim),
                                     (self.num_subspaces, self.n_e, self.sub_dim))

    def __call__(self, z):
        batch_size = z.shape[0]
        z_split = z.reshape((batch_size, self.num_subspaces, self.sub_dim))
        z_sq = jnp.sum(z_split**2, axis=-1, keepdims=True)
        e_sq = jnp.sum(self.embeddings**2, axis=-1)
        e_sq_broad = jnp.expand_dims(e_sq, axis=0)
        dot_prod = jnp.einsum('bgd,gkd->bgk', z_split, self.embeddings)
        distances = z_sq + e_sq_broad - 2.0 * dot_prod
        indices = jnp.argmin(distances, axis=-1)
        batch_idx = jnp.arange(batch_size)[:, None]
        group_idx = jnp.arange(self.num_subspaces)[None, :]
        z_q_split = self.embeddings[group_idx, indices]
        z_q_concat = z_q_split.reshape((batch_size, self.e_dim))
        z_q_ste = z + jax.lax.stop_gradient(z_q_concat - z)
        loss_codebook = jnp.sum((jax.lax.stop_gradient(z_split) - z_q_split) ** 2, axis=(-1, -2))
        loss_commit = self.beta * jnp.sum((z_split - jax.lax.stop_gradient(z_q_split)) ** 2, axis=(-1, -2))
        vq_loss = loss_codebook + loss_commit
        return z_q_ste, indices, vq_loss


class PredatorNetworkJax(nn.Module):
    """
    Lean predator policy (Phase 14.3 GWT Router): VQ comms via GWT structural mask.

    Two pathways:
      h_policy — full pooled transformer output (interoceptive + exteroceptive)
                 → action logits, value, symbol write, ToM, culture, proprio.
      h_comms  — lightweight Dense(d)+ReLU on energy-masked obs (exteroceptive ONLY)
                 + optional cross-attention over neighbor signals
                 → z_e → VQ bottleneck → red_signal_out.

    The GWT mask (obs[:, 0] = 0) physically severs the energy/metabolic gradient
    from the communication head, forcing the VQ codebook to maximise channel
    capacity over exteroceptive data (blue geometry, neighbor signals).

    No carry_fwd, confidence, or imagination heads — catch PPO forges language.
    """

    hidden_dim: int = 128
    neighbor_k: int = 6
    local_obs_radius: int = 2
    n_heads: int = 4
    n_layers: int = 4
    signal_dim: int = 32
    symbol_dim: int = 16
    vocab_size: int = 64
    vq_beta: float = 0.25
    vq_dead_code_reset: bool = True
    simvq_w_clip: float = SIMVQ_W_CLIP_DEFAULT
    simvq_out_scale: float = SIMVQ_OUT_SCALE_DEFAULT
    vq_noise_scale: float = 0.0
    max_layers: int = 6
    memory_slots: int = 0
    cross_attn_enabled: bool = True
    cross_attn_num_heads: int = 4
    n_actions: int = 8
    env_channels: int = 9
    own_state_dim: int = 10

    def _obs_layout(self):
        return make_obs_layout(
            signal_dim=self.signal_dim,
            symbol_dim=self.symbol_dim,
            memory_slots=self.memory_slots,
            neighbor_k=self.neighbor_k,
            local_cells=(2 * self.local_obs_radius + 1) ** 2,
            env_channels=self.env_channels,
            own_state_dim=self.own_state_dim,
        )

    def setup(self):
        d = self.hidden_dim
        sym_d = self.symbol_dim
        K = self.neighbor_k
        W = (2 * self.local_obs_radius + 1) ** 2

        self.emb_own = nn.Dense(d)
        self.emb_nb = nn.Dense(d)
        self.emb_sym = nn.Dense(d)
        self.emb_env = nn.Dense(d)
        self.emb_sig = nn.Dense(d)
        self.emb_mem = nn.Dense(d)
        self.emb_cult = nn.Dense(d)
        self.pos_enc = nn.Embed(num_embeddings=256, features=d)
        self.blocks = [
            TransformerBlock(d, self.n_heads) for _ in range(self.max_layers)
        ]
        self.final_norm = nn.LayerNorm()
        # Phase 15.5: GRUCell to replace EMA, protecting magnitude from BPTT explosion
        self.carry_gru = nn.GRUCell(features=d, name="carry_gru")
        self.head_action = nn.Dense(self.n_actions)           # N actions
        self.head_signal = nn.Dense(self.signal_dim)
        
        # Phase 14.4: DCVQ + SimVQ instead of red_codebook
        self.num_subspaces = 4
        self.dcvq = DCVQ(
            n_e=self.vocab_size, 
            e_dim=self.signal_dim, 
            num_subspaces=self.num_subspaces, 
            beta=self.vq_beta
        )
        self.simvq_W = self.param('simvq_W', 
                                  nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"), 
                                  (self.signal_dim, self.signal_dim))
        
        self.head_symbol = nn.Dense(sym_d)
        self.head_value = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.zeros,
        )
        self.head_tom = nn.Dense(self.n_actions)
        self.head_culture_fast = nn.Dense(sym_d)
        self.head_culture_slow = nn.Dense(sym_d)
        self.head_proprio = nn.Dense(1)
        # Phase 15.3 Semantic Retention Loss (SRL)
        self.head_retention = nn.Dense(self.signal_dim * self.neighbor_k)
        
        # Phase 14.3 GWT Router — exteroceptive-only comms embedding (energy masked).
        # Input dim = full obs_dim; energy feature is zeroed before this layer.
        self.gwt_comms_1 = nn.Dense(d)
        if self.cross_attn_enabled:
            self.red_nb_cross_attn = NeighborCrossAttention(
                hidden_dim=d,
                num_heads=self.cross_attn_num_heads,
            )

    def red_auxiliary_heads(self, carry_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict next-step energy and lag-5 neighbor signals from carry (Phase 14.1b + 15.3)."""
        energy_pred = self.head_proprio(carry_t).squeeze(-1)
        retention_pred = self.head_retention(carry_t)
        return energy_pred, retention_pred

    def __call__(
        self,
        carries: jnp.ndarray,
        obs: jnp.ndarray,
        n_layers: int,
        nb_gain: Optional[jnp.ndarray] = None,
        detach_value: bool = False,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Tuple]:
        del nb_gain
        N = obs.shape[0]
        sym_d = self.symbol_dim
        K = self.neighbor_k
        W = (2 * self.local_obs_radius + 1) ** 2
        layout = self._obs_layout()

        own_state = obs[:, layout.own_state_start : layout.own_state_end]
        nb_sigs = obs[:, layout.nb_sigs_start : layout.nb_sigs_end].reshape(
            N, K, self.signal_dim
        )
        nb_alarms = obs[:, layout.nb_alarms_start : layout.nb_alarms_end].reshape(
            N, K, 2
        )
        nb_combined = jnp.concatenate([nb_sigs, nb_alarms], axis=-1)
        loc_sym = obs[:, layout.loc_sym_start : layout.loc_sym_end].reshape(
            N, W, sym_d
        )
        env_ch = self.env_channels
        loc_env = obs[:, layout.loc_env_start : layout.loc_env_end].reshape(
            N, W, env_ch
        )
        own_sig = obs[:, layout.own_sig_start : layout.own_sig_end]
        own_alarm = obs[:, layout.own_alarm_start : layout.own_alarm_end]
        own_combined = jnp.concatenate([own_sig, own_alarm], axis=-1)

        idx = layout.mem_start
        if self.memory_slots > 0:
            mem = obs[:, idx : idx + self.memory_slots * (self.signal_dim + 2)].reshape(
                N, self.memory_slots, self.signal_dim + 2
            )
            idx += self.memory_slots * (self.signal_dim + 2)
        else:
            mem = None

        loc_cult_fast = obs[:, idx : idx + W * sym_d].reshape(N, W, sym_d)
        idx += W * sym_d
        loc_cult_slow = obs[:, idx : idx + W * sym_d].reshape(N, W, sym_d)

        self_encoded = self.emb_own(own_state)
        t1 = self_encoded[:, None, :]
        nb_kv = self.emb_nb(nb_combined)
        if self.cross_attn_enabled:
            processed = self.red_nb_cross_attn(self_encoded, carries, nb_kv)
            t2 = processed[:, None, :]
        else:
            t2 = nb_kv
        t3 = self.emb_sym(loc_sym)
        t4 = self.emb_env(loc_env)
        t5 = self.emb_sig(own_combined)[:, None, :]
        tokens = [t1, t2, t3, t4, t5]
        if mem is not None:
            tokens.append(self.emb_mem(mem))
        tokens.append(self.emb_cult(loc_cult_fast))
        tokens.append(self.emb_cult(loc_cult_slow))

        x = jnp.concatenate(tokens, axis=1)
        T = x.shape[1]
        pos_ids = jnp.arange(T)
        x = x + self.pos_enc(pos_ids)[None, :, :]
        x = x + carries[:, None, :]

        for i in range(n_layers):
            x = self.blocks[i](x)

        pooled = self.final_norm(x.mean(axis=1))
        # Phase 15.5: GRUCell protects representations from BPTT magnitude explosion
        new_carries, _ = self.carry_gru(carries, pooled)

        # --- Phase 14.3 GWT Router ---
        # h_policy: full pooled transformer output (interoceptive + exteroceptive).
        # Drives action, value, symbol, ToM, culture, and proprio heads.
        h_policy = pooled

        # h_comms: exteroceptive-ONLY pathway.
        # Zero out age(0), mat(1), energy(2), layers(3) to completely sever the metabolic leak
        exteroceptive_obs = obs.at[:, :4].set(0.0)
        h_comms = nn.relu(self.gwt_comms_1(exteroceptive_obs))
        if self.cross_attn_enabled:
            # Query = h_comms (exteroceptive embedding).
            # KV   = nb_kv   (neighbor signal embeddings).
            # Carry is zeroed: recurrent state carries metabolic history; exclude it.
            h_comms = self.red_nb_cross_attn(
                h_comms, jnp.zeros_like(carries), nb_kv
            )

        value_input = jax.lax.stop_gradient(h_policy) if detach_value else h_policy

        action_logits = self.head_action(h_policy) / 2.0
        # z_e sourced from comms pathway (exteroceptive mask enforced — GWT)
        z_e = self.head_signal(h_comms)
        
        z_e_noisy = z_e
        if not deterministic and self.vq_noise_scale > 0.0:
            noise_key = self.make_rng('dropout')
            z_e_noisy = z_e + jax.random.normal(noise_key, z_e.shape) * self.vq_noise_scale
        
        # Phase 14.4 Contingencies: DCVQ + SimVQ (P15: bounded W projection)
        signal_out, indices, loss_vq = self.dcvq(z_e_noisy)
        signal_out = simvq_bounded_project(
            signal_out, self.simvq_W, self.simvq_w_clip, self.simvq_out_scale
        )
        token_ids = indices[:, 0]  # Export first subspace token for telemetry compatibility
        
        symbol_write = self.head_symbol(h_policy)
        feral_mask = jax.lax.stop_gradient(obs[:, 2] < 0.20)
        symbol_write = jnp.where(feral_mask[:, None], 0.0, symbol_write)
        values = self.head_value(value_input).squeeze(-1)
        tom_logits = self.head_tom(h_policy)[:, None, :]
        tom_logits = jnp.broadcast_to(tom_logits, (N, K, self.n_actions))
        culture_fast = self.head_culture_fast(h_policy)
        culture_slow = self.head_culture_slow(h_policy)

        if self.is_initializing():
            self.head_proprio(h_policy)
            self.head_retention(h_policy)

        return new_carries, (
            action_logits,
            signal_out,
            symbol_write,
            values,
            tom_logits,
            token_ids,
            loss_vq,
            z_e,
            culture_fast,
            culture_slow,
        )


def init_predator_params(
    model: PredatorNetworkJax,
    rng: jax.Array,
    carry: jnp.ndarray,
    obs: jnp.ndarray,
    n_layers: int,
) -> Any:
    return sanitize_agent_params(model.init(rng, carry, obs, n_layers)["params"])


def ensure_predator_params(
    model: PredatorNetworkJax,
    params: Any,
    rng: jax.Array,
    hidden_dim: int,
    obs_dim: int,
    n_layers: int,
) -> Any:
    """Fill missing dcvq / simvq_W / red_nb_cross_attn / gwt_comms_1 when resuming older predator ckpts."""
    flat = unfreeze(params)
    pad_emb_own(flat, model.own_state_dim) # Phase 17 own_state grafting
    pad_head_action(flat, model.n_actions) # Phase 16 parameter grafting
    pad_gwt_comms_1(flat, obs_dim)  # Phase 16 obs grafting for GWT Router
    pad_auxiliary_heads(flat, hidden_dim, model.n_actions) # Phase 16 auxiliary grafting
    needs_codebook = "dcvq" not in flat or "simvq_W" not in flat or (
        "head_signal" in flat
        and flat["head_signal"]["kernel"].shape[-1] != model.signal_dim
    )
    needs_cross = bool(
        getattr(model, "cross_attn_enabled", False) and "red_nb_cross_attn" not in flat
    )
    needs_proprio = "head_proprio" not in flat
    # Phase 14.3 GWT Router — graft comms embedding if missing from P14.2 ckpt.
    needs_gwt = "gwt_comms_1" not in flat
    # Phase 15.5 GRUCell upgrade
    needs_gru = "carry_gru" not in flat
    if not needs_codebook and not needs_cross and not needs_proprio and not needs_gwt and not needs_gru:
        return sanitize_agent_params(freeze(flat))
    carry = jnp.zeros((1, hidden_dim))
    obs = jnp.zeros((1, obs_dim))
    fresh = model.init(rng, carry, obs, n_layers)["params"]
    fresh_flat = unfreeze(fresh)
    if needs_codebook:
        for k in ("dcvq", "simvq_W", "head_signal"):
            if k in fresh_flat:
                flat[k] = fresh_flat[k]
        if "red_codebook" in flat:
            del flat["red_codebook"]
        print("[JAX] Merged fresh dcvq + simvq_W + head_signal (Phase 14.4) into predator params")
    if needs_cross and "red_nb_cross_attn" in fresh_flat:
        flat["red_nb_cross_attn"] = fresh_flat["red_nb_cross_attn"]
        print("[JAX] Merged fresh red_nb_cross_attn (Phase 12) into predator params")
    if needs_proprio and "head_proprio" in fresh_flat:
        flat["head_proprio"] = fresh_flat["head_proprio"]
        print("[JAX] Merged fresh head_proprio (Phase 14.1b) into predator params")
    if needs_gwt and "gwt_comms_1" in fresh_flat:
        flat["gwt_comms_1"] = fresh_flat["gwt_comms_1"]
        print("[JAX] Merged fresh gwt_comms_1 (Phase 14.3 GWT Router) into predator params")
    if needs_gru and "carry_gru" in fresh_flat:
        flat["carry_gru"] = fresh_flat["carry_gru"]
        print("[JAX] Merged fresh carry_gru (Phase 15.5 GRUCell upgrade) into predator params")
    return sanitize_agent_params(freeze(flat))


def reset_predator_vq_on_resume(
    model: PredatorNetworkJax,
    params: Any,
    rng: jax.Array,
    hidden_dim: int,
    obs_dim: int,
    n_layers: int,
) -> Any:
    """Reinitialize full comms bottleneck (GWT → z_e → VQ); preserve policy/value/attn."""
    flat = unfreeze(params)
    carry = jnp.zeros((1, hidden_dim))
    obs = jnp.zeros((1, obs_dim))
    fresh_flat = unfreeze(model.init(rng, carry, obs, n_layers)["params"])
    reset = False
    for k in PREDATOR_VQ_COLD_RESTART_KEYS:
        if k in fresh_flat:
            flat[k] = fresh_flat[k]
            reset = True
    if not reset:
        return params
    print(
        "[JAX] Red VQ cold-restart: gwt_comms_1 + head_signal + codebook + simvq_W + red_nb_cross_attn reinitialized "
        "(policy/value preserved)",
        flush=True,
    )
    return sanitize_agent_params(freeze(flat))


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


def make_model_apply(model: AgentNetworkJax):
    """Return ``apply(params, carries, obs, n_layers, ...)`` with correct Flax variables."""
    def apply_fn(params, carries, obs, n_layers, detach_value: bool = False, deterministic: bool = False, rngs=None):
        return model.apply(
            params_apply_variables(params),
            carries,
            obs,
            n_layers,
            detach_value=detach_value,
            deterministic=deterministic,
            rngs=rngs,
        )
    return apply_fn


def make_vqel_monologue_apply(model: AgentNetworkJax):
    """Return apply bound to ``monologue_forward`` (Phase 14.1b)."""
    def apply_fn(params, carries, obs, n_layers):
        return model.apply(
            params_apply_variables(params),
            carries,
            obs,
            n_layers,
            method=model.monologue_forward,
        )
    return apply_fn


def _is_nested_param_dict(node: Any) -> bool:
    """True if this dict is a Flax submodule collection (not a single kernel/bias leaf)."""
    if not isinstance(node, dict):
        return False
    return any(isinstance(v, dict) for v in node.values())


def graft_missing_param_subtrees(
    source: dict,
    template: dict,
    prefix: str = "",
) -> List[str]:
    """
    Copy parameter subtrees present in ``template`` but missing from ``source``.

    Recurses into nested Flax modules (e.g. ``nb_cross_attn/q_norm``, ``nb_cross_mha``).
    Returns dotted paths of injected top-level or nested keys for logging.
    """
    injected: List[str] = []
    for key, tgt_val in template.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if key not in source:
            source[key] = jax.tree_util.tree_map(lambda x: x, tgt_val)
            injected.append(path)
            continue
        src_val = source[key]
        if not isinstance(tgt_val, dict) and hasattr(tgt_val, "shape") and hasattr(src_val, "shape"):
            if tgt_val.shape != src_val.shape:
                if key == "kernel" and len(tgt_val.shape) == 2 and len(src_val.shape) == 2:
                    if tgt_val.shape[1] == src_val.shape[1] and tgt_val.shape[0] > src_val.shape[0]:
                        import jax.numpy as jnp
                        diff = tgt_val.shape[0] - src_val.shape[0]
                        padding = jnp.zeros((diff, tgt_val.shape[1]), dtype=src_val.dtype)
                        source[key] = jnp.concatenate([src_val, padding], axis=0)
                        injected.append(f"{path} (zero-padded {diff} inputs)")
                        continue
                # If shapes mismatch and not handled above, reset to template
                source[key] = jax.tree_util.tree_map(lambda x: x, tgt_val)
                injected.append(f"{path} (shape mismatch {src_val.shape} -> {tgt_val.shape}, reinitialized)")
                continue

        if (
            isinstance(tgt_val, dict)
            and isinstance(src_val, dict)
        ):
            injected.extend(graft_missing_param_subtrees(src_val, tgt_val, path))
    return injected


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


def pad_emb_own(flat_params: dict, target_dim: int = 10) -> None:
    """Pad emb_own weights from 6 to target_dim to accommodate Phase 17 entropy addition."""
    if "emb_own" not in flat_params:
        return
    eo = flat_params["emb_own"]
    kernel = eo["kernel"]
    if kernel.shape[0] < target_dim:
        missing = target_dim - kernel.shape[0]
        out_dim = kernel.shape[1]
        padded_kernel = jnp.concatenate([
            kernel,
            jnp.zeros((missing, out_dim), dtype=kernel.dtype)
        ], axis=0)
        flat_params["emb_own"] = dict(eo)  # make a copy to avoid mutating frozen dicts accidentally
        flat_params["emb_own"]["kernel"] = padded_kernel
        print(f"[JAX] Grafting padding to emb_own: expanded inputs from {kernel.shape[0]} to {target_dim}", flush=True)


def pad_head_action(flat_params: dict, target_actions: int = 8) -> None:
    """Pad or truncate head_action weights/biases to match target_actions."""
    if "head_action" not in flat_params:
        return
    ha = flat_params["head_action"]
    kernel = ha["kernel"]
    bias = ha["bias"]
    
    if kernel.shape[1] < target_actions:
        d = kernel.shape[0]
        missing = target_actions - kernel.shape[1]
        padded_kernel = jnp.concatenate([
            kernel,
            jnp.zeros((d, missing), dtype=kernel.dtype)
        ], axis=1)
        padded_bias = jnp.concatenate([
            bias,
            jnp.zeros((missing,), dtype=bias.dtype)
        ], axis=0)
        flat_params["head_action"] = dict(ha)
        flat_params["head_action"]["kernel"] = padded_kernel
        flat_params["head_action"]["bias"] = padded_bias
        print(f"[JAX] Grafting padding to head_action: expanded from {kernel.shape[1]} to {target_actions} actions", flush=True)
    elif kernel.shape[1] > target_actions:
        truncated_kernel = kernel[:, :target_actions]
        truncated_bias = bias[:target_actions]
        flat_params["head_action"] = dict(ha)
        flat_params["head_action"]["kernel"] = truncated_kernel
        flat_params["head_action"]["bias"] = truncated_bias
        print(f"[JAX] Truncating head_action: reduced from {kernel.shape[1]} to {target_actions} actions", flush=True)



def pad_gwt_comms_1(flat_params: dict, target_channels: int) -> None:
    """Pad gwt_comms_1 kernel from older observation dimensions (2335, 2360) to target_channels."""
    if "gwt_comms_1" not in flat_params:
        return
    gc1 = flat_params["gwt_comms_1"]
    kernel = gc1["kernel"]
    hidden_dim = kernel.shape[1]
    
    # 1. Pad own_state from 6 to 10 if needed
    if kernel.shape[0] in (2335, 2360) and target_channels >= 2364:
        missing_own = 4
        padded_kernel = jnp.concatenate([
            kernel[:6, :],
            jnp.zeros((missing_own, hidden_dim), dtype=kernel.dtype),
            kernel[6:, :]
        ], axis=0)
        kernel = padded_kernel
        print(f"[JAX] Grafting own_state padding to gwt_comms_1: expanded +{missing_own}", flush=True)

    # 2. Pad loc_env from 9 to 10 channels if needed
    if kernel.shape[0] == 2339 and target_channels == 2364:
        K = 6
        W = 25
        sig_dim = 32
        sym_dim = 16
        idx_start = 10 + K * sig_dim + W * sym_dim  # 602
        
        new_kernel = jnp.zeros((target_channels, hidden_dim), dtype=kernel.dtype)
        insert_indices = jnp.array([idx_start + 9 + i * 10 for i in range(W)])
        
        mask = jnp.ones(target_channels, dtype=bool)
        mask = mask.at[insert_indices].set(False)
        new_kernel = new_kernel.at[mask].set(kernel)
        kernel = new_kernel
        print(f"[JAX] Grafting interleaved padding to gwt_comms_1: expanded +25", flush=True)
        
    flat_params["gwt_comms_1"] = {
        "kernel": kernel,
        "bias": gc1.get("bias", jnp.zeros(hidden_dim, dtype=kernel.dtype)),
    }


def pad_auxiliary_heads(flat_params: dict, hidden_dim: int, target_actions: int = 8) -> None:
    """Pad or truncate auxiliary heads to handle the target_actions vector length."""
    
    # 1. Output heads: pad or truncate axis=1 (like head_action)
    for k in ["head_self_pred", "head_tom"]:
        if k in flat_params:
            ha = flat_params[k]
            d = ha["kernel"]
            bias = ha.get("bias", jnp.zeros(d.shape[1], dtype=d.dtype))
            if d.shape[1] < target_actions:
                missing = target_actions - d.shape[1]
                padded_kernel = jnp.concatenate([d, jnp.zeros((d.shape[0], missing), dtype=d.dtype)], axis=1)
                padded_bias = jnp.concatenate([bias, jnp.zeros((missing,), dtype=bias.dtype)], axis=0)
                flat_params[k] = {"kernel": padded_kernel, "bias": padded_bias}
                print(f"[JAX] Grafting padding to {k}: expanded outputs to {target_actions}")
            elif d.shape[1] > target_actions:
                flat_params[k] = {"kernel": d[:, :target_actions], "bias": bias[:target_actions]}
                print(f"[JAX] Truncating {k}: reduced outputs to {target_actions}")

    # 2. Input heads: pad or truncate axis=0 because action_oh is concatenated at the END of carry_t
    for k in ["head_fwd_1", "head_fwd_dyn_1", "head_confidence_1"]:
        if k in flat_params:
            ha = flat_params[k]
            d = ha["kernel"]
            bias = ha.get("bias", jnp.zeros(d.shape[1], dtype=d.dtype))
            target_dim = hidden_dim + target_actions
            if d.shape[0] < target_dim:
                missing = target_dim - d.shape[0]
                padded_kernel = jnp.concatenate([d, jnp.zeros((missing, d.shape[1]), dtype=d.dtype)], axis=0)
                flat_params[k] = {"kernel": padded_kernel, "bias": bias}
                print(f"[JAX] Grafting padding to {k}: expanded inputs to {target_dim}")
            elif d.shape[0] > target_dim:
                flat_params[k] = {"kernel": d[:target_dim, :], "bias": bias}
                print(f"[JAX] Truncating {k}: reduced inputs to {target_dim}")


def pad_head_fwd_2(flat_params: dict, target_outputs: int = 250) -> None:
    """Pad head_fwd_2 outputs from 225 to 250 by interleaving zeros for the 10th env channel."""
    if "head_fwd_2" not in flat_params:
        return
    hf2 = flat_params["head_fwd_2"]
    kernel = hf2["kernel"]
    if kernel.shape[1] < target_outputs:
        hidden_dim = kernel.shape[0]
        
        W = 25 # (2*2 + 1)**2
        
        new_kernel = jnp.zeros((hidden_dim, target_outputs), dtype=kernel.dtype)
        bias = hf2.get("bias", jnp.zeros(kernel.shape[1], dtype=kernel.dtype))
        new_bias = jnp.zeros(target_outputs, dtype=bias.dtype)
        
        insert_indices = jnp.array([9 + i * 10 for i in range(W)])
        
        mask = jnp.ones(target_outputs, dtype=bool)
        mask = mask.at[insert_indices].set(False)
        
        new_kernel = new_kernel.at[:, mask].set(kernel)
        new_bias = new_bias.at[mask].set(bias)
        
        flat_params["head_fwd_2"] = {
            "kernel": new_kernel,
            "bias": new_bias,
        }
        print(f"[JAX] Grafting interleaved padding to head_fwd_2: expanded outputs from {kernel.shape[1]} to {target_outputs}", flush=True)


def ensure_aux_head_params(
    model: AgentNetworkJax,
    params: Any,
    rng: jax.Array,
    hidden_dim: int,
    obs_dim: int = 0,
    n_layers: int = 4,
) -> Any:
    """Fill missing auxiliary-head / VQ / monologue params when resuming."""
    flat = unfreeze(params)
    pad_emb_own(flat, model.own_state_dim) # Phase 17 own_state grafting
    pad_head_action(flat, model.n_actions) # Phase 16 parameter grafting
    pad_gwt_comms_1(flat, obs_dim)         # Phase 17 GWT Router grafting
    pad_auxiliary_heads(flat, hidden_dim, model.n_actions) # Phase 16 auxiliary grafting
    pad_head_fwd_2(flat)   # Phase 16 env prediction grafting
    needs_vq = (
        "codebook" not in flat
        or (
            "head_signal" in flat
            and flat["head_signal"]["kernel"].shape[-1] != model.signal_dim
        )
    )
    needs_cross_attn = bool(
        getattr(model, "cross_attn_enabled", False) and "nb_cross_attn" not in flat
    )
    needs_vqel_recon = (
        "head_vqel_recon_1" not in flat or "head_vqel_recon_2" not in flat
    )
    needs_gwt = "gwt_comms_1" not in flat

    if (
        all(k in flat for k in AUX_HEAD_KEYS)
        and not needs_vq
        and not needs_cross_attn
        and not needs_vqel_recon
        and not needs_gwt
    ):
        return sanitize_agent_params(freeze(flat))
    carry = jnp.zeros((1, hidden_dim))
    if needs_vq and obs_dim > 0:
        obs = jnp.zeros((1, obs_dim))
        fresh = model.init(rng, carry, obs, n_layers)["params"]
        fresh_flat = unfreeze(fresh)
        for k in ("codebook", "head_signal"):
            flat[k] = fresh_flat[k]
        print("[JAX] Merged fresh VQ params (codebook, head_signal) into restored checkpoint")
    if needs_vqel_recon and obs_dim > 0:
        obs = jnp.zeros((1, obs_dim))
        fresh = model.init(rng, carry, obs, n_layers)["params"]
        fresh_flat = unfreeze(fresh)
        for k in ("head_vqel_recon_1", "head_vqel_recon_2"):
            flat[k] = fresh_flat[k]
        print("[JAX] Merged fresh VQEL monologue decoder heads into restored checkpoint")
    action_oh = jnp.zeros((1, model.n_actions), dtype=jnp.float32)
    aux_only = model.init(
        rng, carry, action_oh, method=model.auxiliary_heads
    )["params"]
    aux_flat = unfreeze(aux_only)
    for k in AUX_HEAD_KEYS:
        if k not in flat:
            flat[k] = aux_flat[k]
            print(f"[JAX] Merged fresh {k} params into restored checkpoint")
    if needs_cross_attn and obs_dim > 0:
        obs = jnp.zeros((1, obs_dim))
        fresh = model.init(rng, carry, obs, n_layers)["params"]
        fresh_flat = unfreeze(fresh)
        if "nb_cross_attn" in fresh_flat:
            flat["nb_cross_attn"] = fresh_flat["nb_cross_attn"]
            print("[JAX] Merged fresh nb_cross_attn (Phase 9.4) into restored checkpoint")
            
    if needs_gwt and obs_dim > 0:
        obs = jnp.zeros((1, obs_dim))
        fresh = model.init(rng, carry, obs, n_layers)["params"]
        fresh_flat = unfreeze(fresh)
        if "gwt_comms_1" in fresh_flat:
            flat["gwt_comms_1"] = fresh_flat["gwt_comms_1"]
            print("[JAX] Merged fresh gwt_comms_1 (Phase 16.6 GWT Router) into restored checkpoint")
            
    return sanitize_agent_params(freeze(flat))


class NeighborCrossAttention(nn.Module):
    """
    Phase 9.4 — Cross-Attention Receiver.

    Query: encoded self (own_state embedding + carry).
    Key/Value: neighbor signal embeddings (the swarm \"Other\").
    Output: self_encoded + attention_out (residual on self).
    """

    hidden_dim: int
    num_heads: int

    @nn.compact
    def __call__(
        self,
        self_encoded: jnp.ndarray,
        carry: jnp.ndarray,
        neighbor_kv: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        self_encoded: (N, d)
        carry: (N, d)
        neighbor_kv: (N, K, d) — embedded neighbor signals
        Returns processed_signals: (N, d)
        """
        q = nn.LayerNorm(name="q_norm")(self_encoded + carry)[:, None, :]
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            name="nb_cross_mha",
        )(q, neighbor_kv)
        return self_encoded + attn_out.squeeze(1)


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
