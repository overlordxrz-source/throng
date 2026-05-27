"""
agents/network_torch.py — PyTorch Actor-Critic brain for MAPPO.

Mirrors the Flax AgentNetwork architecture exactly:
  - Observation split into typed tokens (own_state, neighbour_sigs,
    local_symbols, local_presence, own_signal, carry_memory)
  - Variable-depth transformer: all max_layers blocks pre-allocated,
    only first n_layers executed (brain growth = increment n_layers)
  - Carry = tanh(Linear(readout)) — simple deterministic update
  - Output heads: action_logits(5), signal(sig_dim), symbol(sym_dim), value(1)

TorchBrain: thin wrapper that
  - holds model + Adam optimizer
  - provides numpy-in / numpy-out forward() for the sim loop
  - provides ppo_update() consuming RolloutBuffer.get() dicts
  - moves tensors to MPS (or CUDA or CPU) automatically
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ── Device selection ──────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _best_device()


# ── Transformer block ─────────────────────────────────────────────────────────

class _AttentionBlock(nn.Module):
    def __init__(self, token_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn  = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(token_dim)
        self.ff1   = nn.Linear(token_dim, token_dim * 4)
        self.ff2   = nn.Linear(token_dim * 4, token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff = F.gelu(self.ff1(x))
        ff = self.ff2(ff)
        return self.norm2(x + ff)


# ── Agent network ─────────────────────────────────────────────────────────────

class SignalGate(nn.Module):
    """
    Learned cross-attention gate: own_state queries neighbour signals
    to restore information lost to sensor masking.

    Makes the signal channel structurally load-bearing: if signals are
    zeroed, the gate cannot restore masked dims and fitness craters.
    """

    def __init__(self, token_dim: int, signal_dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.attn = nn.MultiheadAttention(
            token_dim, n_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        own_token: torch.Tensor,   # (N, 1, token_dim)  — masked own-state
        nb_tokens: torch.Tensor,   # (N, K, token_dim)  — neighbour signals
    ) -> torch.Tensor:
        """Returns restoration vector added to own_token."""
        restored, _ = self.attn(own_token, nb_tokens, nb_tokens)
        return self.norm(own_token + restored)  # (N, 1, token_dim)


class AgentNetworkTorch(nn.Module):
    """
    Parameter-shared transformer brain. One instance per team; all agents
    on the team run the same weights with different carry tensors.

    Phase 4: Discrete token bottleneck — signal head outputs vocab_size logits,
    sampled via Gumbel-Softmax (differentiable in train, hard argmax in eval).
    Tokens are embedded back to signal_dim continuous vectors for neighbour
    consumption, keeping the rest of the pipeline unchanged.
    """

    def __init__(
        self,
        hidden_dim:     int  = 64,
        token_dim:      int  = 32,
        n_heads:        int  = 2,
        max_layers:     int  = 6,
        signal_dim:     int  = 16,
        symbol_dim:     int  = 8,
        neighbor_k:     int  = 6,
        obs_radius:     int  = 1,
        n_actions:      int  = 5,
        tom_independent: bool = False,
        signal_gate:    bool = False,
        vocab_size:     int  = 64,
        gumbel_tau:     float = 0.5,
        memory_slots:   int  = 0,
        memory_slot_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim     = hidden_dim
        self.token_dim      = token_dim
        self.max_layers     = max_layers
        self.signal_dim     = signal_dim
        self.symbol_dim     = symbol_dim
        self.neighbor_k     = neighbor_k
        self.obs_radius     = obs_radius
        self.n_actions      = n_actions
        self.tom_independent = tom_independent
        self.signal_gate_enabled = signal_gate
        self.vocab_size     = vocab_size
        self.gumbel_tau     = gumbel_tau
        self.memory_slots   = memory_slots
        self.memory_slot_dim = memory_slot_dim

        # Signal gate — restores masked sensor dims from neighbour signals
        if signal_gate:
            self.gate = SignalGate(token_dim, signal_dim, n_heads)
        else:
            self.gate = None

        # Embedding layers — one per observation segment
        # Phase 7: pres expanded to 7 (added shelter, contested, scent)
        self.emb_own  = nn.Linear(6,          token_dim)
        self.emb_nb   = nn.Linear(signal_dim,  token_dim)   # per neighbour
        self.emb_sym  = nn.Linear(symbol_dim,  token_dim)   # per symbol cell
        self.emb_pres = nn.Linear(7,           token_dim)   # per cell: [blue, red, wall, resource, shelter, contested, scent]
        self.emb_sig  = nn.Linear(signal_dim,  token_dim)
        self.emb_mem  = nn.Linear(hidden_dim,  token_dim)
        # Phase 7: episodic memory tokens
        if memory_slots > 0:
            self.emb_epi = nn.Linear(memory_slot_dim, token_dim)
        else:
            self.emb_epi = None

        # All max_layers blocks pre-allocated (depth gated at runtime)
        self.attn_blocks = nn.ModuleList([
            _AttentionBlock(token_dim, n_heads) for _ in range(max_layers)
        ])

        # Carry update
        self.carry_proj = nn.Linear(token_dim, hidden_dim)

        # Output heads
        self.head_action = nn.Linear(token_dim, n_actions)
        # Phase 4: signal head now outputs discrete vocab logits
        self.head_signal = nn.Linear(token_dim, vocab_size)
        self.token_embed = nn.Embedding(vocab_size, signal_dim)
        self.head_symbol = nn.Linear(token_dim, symbol_dim)
        self.head_value  = nn.Linear(token_dim, 1)
        # Phase 7.5: dual cultural memory write heads (fast = recent, slow = long-term)
        self.head_culture_fast = nn.Linear(token_dim, symbol_dim)
        self.head_culture_slow = nn.Linear(token_dim, symbol_dim)

        # Theory-of-mind head: predicts each neighbour's action from
        # this agent's carry (world model) + received neighbour signals.
        if tom_independent:
            # Per-neighbour: concat(carry, sig_i) → 5 logits, applied independently
            # for each of the K neighbours.  Forces the head to use the specific
            # neighbour's signal rather than a global average over all K signals.
            self.tom_head = nn.Sequential(
                nn.Linear(hidden_dim + signal_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_actions),
            )
        else:
            # Joint (legacy): carry + all K signals flattened → K*5 logits.
            self.tom_head = nn.Sequential(
                nn.Linear(hidden_dim + signal_dim * neighbor_k, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, neighbor_k * n_actions),
            )

    @property
    def _W(self) -> int:
        return (2 * self.obs_radius + 1) ** 2

    def forward(
        self,
        carries: torch.Tensor,   # (N, hidden_dim)
        obs:     torch.Tensor,   # (N, obs_dim)
        n_layers: int,
        nb_gain: Optional[torch.Tensor] = None,  # (N,) per-agent signal reception gain
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        N   = obs.shape[0]
        W   = self._W
        K   = self.neighbor_k
        sd  = self.signal_dim
        syd = self.symbol_dim

        # ── Split flat observation ──────────────────────────────────────────
        c = 0
        own  = obs[:, c:c + 6];                                  c += 6
        nb   = obs[:, c:c + K * sd].view(N, K, sd);             c += K * sd
        syms = obs[:, c:c + W * syd].view(N, W, syd);           c += W * syd
        pres = obs[:, c:c + W * 7].view(N, W, 7);               c += W * 7  # Phase 7: 7 env channels
        sig  = obs[:, c:c + sd];                                 c += sd
        # Phase 7: episodic memory buffer
        epi = None
        if self.memory_slots > 0:
            epi = obs[:, c:c + self.memory_slots * self.memory_slot_dim].view(
                N, self.memory_slots, self.memory_slot_dim
            )
            c += self.memory_slots * self.memory_slot_dim
        # Phase 7.5: dual cultural memory grids (fast + slow)
        cult_fast = obs[:, c:c + W * syd].view(N, W, syd);   c += W * syd
        cult_slow = obs[:, c:c + W * syd].view(N, W, syd);   c += W * syd

        # ── Within-lifetime Hebbian: scale incoming signals by per-agent gain ─
        if nb_gain is not None:
            nb = nb * nb_gain[:, None, None]                    # (N, K, sd) * (N, 1, 1)

        # ── Embed each group ────────────────────────────────────────────────
        t_own  = self.emb_own(own).unsqueeze(1)                  # (N, 1, D)
        t_nb   = self.emb_nb(nb.reshape(N * K, sd)).view(N, K, self.token_dim)
        t_sym  = self.emb_sym(syms.reshape(N * W, syd)).view(N, W, self.token_dim)
        t_pres = self.emb_pres(pres.reshape(N * W, 7)).view(N, W, self.token_dim)
        t_sig  = self.emb_sig(sig).unsqueeze(1)                  # (N, 1, D)
        t_mem  = self.emb_mem(carries).unsqueeze(1)              # (N, 1, D)
        t_cult_f = self.emb_sym(cult_fast.reshape(N * W, syd)).view(N, W, self.token_dim)
        t_cult_s = self.emb_sym(cult_slow.reshape(N * W, syd)).view(N, W, self.token_dim)

        # ── Signal gate: restore masked own-state from neighbour signals ───
        if self.gate is not None:
            t_own = self.gate(t_own, t_nb)

        token_list = [t_own, t_nb, t_sym, t_pres, t_sig, t_mem, t_cult_f, t_cult_s]
        if epi is not None and self.emb_epi is not None:
            t_epi = self.emb_epi(
                epi.reshape(N * self.memory_slots, self.memory_slot_dim)
            ).view(N, self.memory_slots, self.token_dim)
            token_list.append(t_epi)
        tokens = torch.cat(token_list, dim=1)

        # ── Variable-depth attention (break at n_layers) ────────────────────
        for i, block in enumerate(self.attn_blocks):
            if i >= n_layers:
                break
            tokens = block(tokens)

        readout     = tokens.mean(dim=1)                          # (N, D)
        new_carries = torch.tanh(self.carry_proj(readout))        # (N, hidden)

        # ── Output heads ────────────────────────────────────────────────────
        action_logits = self.head_action(readout)                 # (N, 5)
        symbol_write  = torch.tanh(self.head_symbol(readout))           # (N, sym_dim)
        culture_fast  = torch.tanh(self.head_culture_fast(readout))       # (N, sym_dim)
        culture_slow  = torch.tanh(self.head_culture_slow(readout))       # (N, sym_dim)
        values        = self.head_value(readout).squeeze(-1)            # (N,)

        # ── Theory-of-mind head ───────────────────────────────────────────────
        if self.tom_independent:
            _tom_in = torch.cat([
                new_carries[:, None, :].expand(-1, K, -1),
                nb,
            ], dim=-1)  # (N, K, hidden_dim + signal_dim)
            tom_logits = self.tom_head(_tom_in.reshape(N * K, -1)).view(N, K, self.n_actions)
        else:
            _tom_in = torch.cat([new_carries, nb.view(N, K * sd)], dim=-1)
            tom_logits = self.tom_head(_tom_in).view(N, K, self.n_actions)

        # ── Discrete token bottleneck (Phase 4) ─────────────────────────────
        signal_logits = self.head_signal(readout)                 # (N, vocab_size)
        if self.training:
            # Differentiable soft sampling via Gumbel-Softmax during PPO update
            soft_tokens = F.gumbel_softmax(signal_logits, tau=self.gumbel_tau, hard=True)
            # soft_tokens (N, V) @ token_embed.weight (V, signal_dim) → (N, signal_dim)
            signal_out = soft_tokens @ self.token_embed.weight
        else:
            # Hard argmax during simulation — no gradients needed
            token_ids = signal_logits.argmax(dim=-1)              # (N,)
            signal_out = self.token_embed(token_ids)                # (N, signal_dim)
            # Also return token IDs for logging/analysis
            return new_carries, (action_logits, signal_out, symbol_write, values, tom_logits, token_ids, None, culture_fast, culture_slow)

        # During training: return signal_logits as 7th element for signal entropy bonus
        return new_carries, (action_logits, signal_out, symbol_write, values, tom_logits, None, signal_logits, culture_fast, culture_slow)


# ── Layer-entropy introspection ───────────────────────────────────────────────

def layer_entropy_torch(
    model:    AgentNetworkTorch,
    carries:  np.ndarray,   # (N, hidden_dim)
    obs:      np.ndarray,   # (N, obs_dim)
    n_layers: int,
    device:   torch.device,
    nb_gain:  Optional[np.ndarray] = None,
) -> dict:
    """
    Run forward at each depth 1..n_layers, measure per-layer output stats.
    Returns dict {layer_i: {action_entropy, signal_norm, logit_delta}}.
    """
    stats = {}
    prev_logits = None
    c_t = torch.tensor(carries, dtype=torch.float32, device=device)
    o_t = torch.tensor(obs,     dtype=torch.float32, device=device)

    g_t = (torch.tensor(nb_gain, dtype=torch.float32, device=device)
           if nb_gain is not None else None)
    with torch.no_grad():
        for li in range(1, n_layers + 1):
            _, (logits_i, sigs_i, _, _, _, _, _, _, _) = model(c_t, o_t, li, g_t)
            ln = logits_i.cpu().numpy()
            sn = sigs_i.cpu().numpy()

            p   = np.exp(ln - ln.max(axis=-1, keepdims=True))
            p  /= p.sum(axis=-1, keepdims=True)
            ent = float(-np.sum(p * np.log(p + 1e-12), axis=-1).mean())
            delta = 0.0 if prev_logits is None else float(np.abs(ln - prev_logits).mean())
            prev_logits = ln

            stats[f"layer_{li}"] = {
                "action_entropy": ent,
                "signal_norm":    float(np.linalg.norm(sn, axis=-1).mean()),
                "logit_delta":    delta,
            }
    return stats


# ── Observation dimension helper ──────────────────────────────────────────────

def compute_obs_dim_torch(config: dict) -> int:
    K    = config["neighbor_k"]
    sd   = config["signal_dim"]
    symd = config.get("symbol_dim", 8)
    r    = config["local_obs_radius"]
    W    = (2 * r + 1) ** 2
    # Phase 8: env channels = 8 (blue_pres, red_pres, wall, resource, shelter, contested, scent, puzzle)
    env_ch = 8
    base = 6 + K * sd + W * symd + W * env_ch + sd
    # Phase 7: episodic memory buffer
    mem_slots = int(config.get("memory_buffer_size", 0))
    if mem_slots > 0 and config.get("memory_buffer_enabled", False):
        base += mem_slots * (sd + 2)
    # Phase 7.5: dual cultural memory grids (fast + slow)
    base += W * symd * 2
    return base


# ── TorchBrain — the main interface for main.py ───────────────────────────────

class TorchBrain:
    """
    Wrapper around AgentNetworkTorch + Adam optimizer.
    All public methods accept / return numpy arrays.
    Internally converts to torch tensors on self.device.
    """

    def __init__(
        self,
        config: dict,
        device: Optional[torch.device] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        self.device = device or DEVICE
        self.config = config

        _hd = hidden_dim if hidden_dim is not None else config["agent_hidden_dim"]
        _mem_slots = int(config.get("memory_buffer_size", 0))
        _mem_slot_dim = 0
        if _mem_slots > 0 and config.get("memory_buffer_enabled", False):
            _mem_slot_dim = config["signal_dim"] + 2

        self.model = AgentNetworkTorch(
            hidden_dim      = _hd,
            token_dim       = config["brain_token_dim"],
            n_heads         = config["brain_n_heads"],
            max_layers      = config["brain_max_layers"],
            signal_dim      = config["signal_dim"],
            symbol_dim      = config.get("symbol_dim", 8),
            neighbor_k      = config["neighbor_k"],
            obs_radius      = config["local_obs_radius"],
            tom_independent = bool(config.get("tom_head_independent", False)),
            signal_gate     = bool(config.get("signal_gate_enabled", False)),
            vocab_size      = int(config.get("vocab_size", 64)),
            gumbel_tau      = float(config.get("gumbel_tau", 0.5)),
            memory_slots    = _mem_slots,
            memory_slot_dim = _mem_slot_dim,
        ).to(self.device)

        _policy_params = [
            p for n, p in self.model.named_parameters()
            if not n.startswith("tom_head")
        ]
        self.optimizer = Adam(_policy_params, lr=float(config["ppo_lr"]))

        _tom_lr = float(config.get("tom_lr", 1e-4))
        self.tom_optimizer = Adam(self.model.tom_head.parameters(), lr=_tom_lr)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        carries:  np.ndarray,   # (N, hidden_dim) float32
        obs:      np.ndarray,   # (N, obs_dim)    float32
        n_layers: int,
        nb_gain:  Optional[np.ndarray] = None,  # (N,) float32 per-agent signal gain
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (new_carries, action_logits, signal_out, symbol_write, values, tom_logits, token_ids, culture_fast, culture_slow)
        all as float32 numpy arrays.  tom_logits shape: (N, K, 5). token_ids: (N,) int64 or -1.
        """
        c_t = torch.tensor(carries, dtype=torch.float32, device=self.device)
        o_t = torch.tensor(obs,     dtype=torch.float32, device=self.device)
        g_t = (torch.tensor(nb_gain, dtype=torch.float32, device=self.device)
               if nb_gain is not None else None)

        new_c, (logits, sigs, syms, vals, tom_log, token_ids, _sig_logits, cult_f, cult_s) = self.model(c_t, o_t, n_layers, g_t)

        return (
            new_c.cpu().numpy(),
            logits.cpu().numpy(),
            sigs.cpu().numpy(),
            syms.cpu().numpy(),
            vals.cpu().numpy(),
            tom_log.cpu().numpy(),
            token_ids.cpu().numpy() if token_ids is not None else np.full(carries.shape[0], -1, dtype=np.int64),
            cult_f.cpu().numpy(),
            cult_s.cpu().numpy(),
        )

    # ------------------------------------------------------------------

    def ppo_update(
        self,
        buffer_data: Dict[str, np.ndarray],
        last_value:  np.ndarray,   # (N,)
        n_layers:    int,
        rng:         np.random.Generator,
    ) -> dict:
        from agents.rl_torch import ppo_update_torch
        return ppo_update_torch(self, buffer_data, last_value, n_layers, rng)

    # ------------------------------------------------------------------

    def layer_entropy(
        self,
        carries:  np.ndarray,
        obs:      np.ndarray,
        n_layers: int,
        nb_gain:  Optional[np.ndarray] = None,
    ) -> dict:
        return layer_entropy_torch(self.model, carries, obs, n_layers, self.device, nb_gain)

    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "model":         self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "tom_optimizer": self.tom_optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        missing, unexpected = self.model.load_state_dict(d["model"], strict=False)
        tom_keys = [k for k in missing if k.startswith("tom_head")]
        non_tom_missing = [k for k in missing if not k.startswith("tom_head")]
        if non_tom_missing:
            raise RuntimeError(f"Unexpected missing keys in checkpoint: {non_tom_missing}")
        if tom_keys:
            print(f"  [ckpt] tom_head not in checkpoint — initialising fresh ({len(tom_keys)} keys)")
        self.optimizer.load_state_dict(d["optimizer"])
        if "tom_optimizer" in d:
            self.tom_optimizer.load_state_dict(d["tom_optimizer"])

    def grow(self, new_n_layers: int) -> None:
        """No weight changes needed — just documents the new depth for logging."""
        pass   # TorchBrain is stateless w.r.t. n_layers; caller tracks it
