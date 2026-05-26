"""
jax_sim/population_jax.py — Fixed-size population arrays with jnp.where masking.

All fields are JAX arrays of shape (max_pop, ...).  Dead agents are masked out
by the `alive` boolean array.  No dynamic resizing — the array shape is fixed
at init time and never changes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Optional, Tuple


class PopState:
    """
    Fixed-size population state.  All agents live in arrays of shape (max_pop, ...).
    The `alive` mask determines which slots are active.
    """
    def __init__(
        self,
        max_pop: int,
        hidden_dim: int,
        signal_dim: int,
        n_teams: int = 2,
        memory_slots: int = 0,
    ) -> None:
        self.max_pop = max_pop
        self.hidden_dim = hidden_dim
        self.signal_dim = signal_dim
        self.memory_slots = memory_slots

        # Core state
        self.positions = jnp.zeros((max_pop, 2), dtype=jnp.int32)
        self.ages = jnp.zeros(max_pop, dtype=jnp.int32)
        self.alive = jnp.zeros(max_pop, dtype=jnp.bool_)
        self.energy = jnp.zeros(max_pop, dtype=jnp.float32)
        self.team = jnp.zeros(max_pop, dtype=jnp.int8)
        self.n_layers = jnp.ones(max_pop, dtype=jnp.int8)  # current brain depth

        # Neural state
        self.carries = jnp.zeros((max_pop, hidden_dim), dtype=jnp.float32)
        self.signals = jnp.zeros((max_pop, signal_dim), dtype=jnp.float32)
        self.nb_gain = jnp.ones(max_pop, dtype=jnp.float32)

        # Episodic memory (optional)
        if memory_slots > 0:
            self.memory_buffer = jnp.zeros(
                (max_pop, memory_slots, signal_dim + 2), dtype=jnp.float32
            )
        else:
            self.memory_buffer = None

        # Book-keeping
        self.offspring_count = jnp.zeros(max_pop, dtype=jnp.int32)
        self.steps_since_catch = jnp.zeros(max_pop, dtype=jnp.int32)
        self.lineage_ids = jnp.zeros(max_pop, dtype=jnp.int32)
        self.next_lineage_id = jnp.int32(1)

    # ── PyTree registration ───────────────────────────────────────────────

    def tree_flatten(self):
        children = [
            self.positions, self.ages, self.alive, self.energy, self.team,
            self.n_layers, self.carries, self.signals, self.nb_gain,
            self.offspring_count, self.steps_since_catch, self.lineage_ids,
            self.next_lineage_id,
        ]
        if self.memory_buffer is not None:
            children.append(self.memory_buffer)
        aux = (self.max_pop, self.hidden_dim, self.signal_dim, self.memory_slots)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        max_pop, hidden_dim, signal_dim, memory_slots = aux
        pop = cls.__new__(cls)
        pop.max_pop = max_pop
        pop.hidden_dim = hidden_dim
        pop.signal_dim = signal_dim
        pop.memory_slots = memory_slots
        (pop.positions, pop.ages, pop.alive, pop.energy, pop.team,
         pop.n_layers, pop.carries, pop.signals, pop.nb_gain,
         pop.offspring_count, pop.steps_since_catch, pop.lineage_ids,
         pop.next_lineage_id) = children[:13]
        if memory_slots > 0:
            pop.memory_buffer = children[13]
        else:
            pop.memory_buffer = None
        return pop

    def replace(self, **kwargs):
        """Immutable update: return new PopState with replaced fields."""
        pop = PopState.__new__(PopState)
        pop.max_pop = self.max_pop
        pop.hidden_dim = self.hidden_dim
        pop.signal_dim = self.signal_dim
        pop.memory_slots = self.memory_slots
        pop.positions = kwargs.get("positions", self.positions)
        pop.ages = kwargs.get("ages", self.ages)
        pop.alive = kwargs.get("alive", self.alive)
        pop.energy = kwargs.get("energy", self.energy)
        pop.team = kwargs.get("team", self.team)
        pop.n_layers = kwargs.get("n_layers", self.n_layers)
        pop.carries = kwargs.get("carries", self.carries)
        pop.signals = kwargs.get("signals", self.signals)
        pop.nb_gain = kwargs.get("nb_gain", self.nb_gain)
        pop.offspring_count = kwargs.get("offspring_count", self.offspring_count)
        pop.steps_since_catch = kwargs.get("steps_since_catch", self.steps_since_catch)
        pop.lineage_ids = kwargs.get("lineage_ids", self.lineage_ids)
        pop.next_lineage_id = kwargs.get("next_lineage_id", self.next_lineage_id)
        pop.memory_buffer = kwargs.get("memory_buffer", self.memory_buffer)
        return pop


jax.tree_util.register_pytree_node(
    PopState,
    lambda p: p.tree_flatten(),
    lambda aux, children: PopState.tree_unflatten(aux, children)
)


# ── Population helpers ───────────────────────────────────────────────────

def init_population(
    max_pop: int,
    hidden_dim: int,
    signal_dim: int,
    grid_size: int,
    team_id: int,
    key: jax.Array,
    n_agents: int = 500,
    memory_slots: int = 0,
) -> PopState:
    """Create a fresh population with n_agents alive at random positions."""
    pop = PopState(max_pop, hidden_dim, signal_dim, memory_slots=memory_slots)
    keys = jax.random.split(key, 3)

    # Random positions for first n_agents
    pos_y = jax.random.randint(keys[0], (n_agents,), 0, grid_size)
    pos_x = jax.random.randint(keys[1], (n_agents,), 0, grid_size)
    positions = jnp.stack([pos_y, pos_x], axis=1)

    alive = jnp.arange(max_pop) < n_agents

    # Update state
    pop = pop.replace(
        positions=pop.positions.at[:n_agents].set(positions),
        alive=alive,
        energy=jnp.where(alive, 1.0, 0.0),
        team=jnp.where(alive, jnp.int8(team_id), jnp.int8(0)),
    )
    return pop


def kill_agents(pop: PopState, mask: jnp.ndarray) -> PopState:
    """mask: (max_pop,) bool — True = die."""
    new_alive = pop.alive & ~mask
    # Zero out dead agents' neural state
    new_carries = jnp.where(mask[:, None], 0.0, pop.carries)
    new_signals = jnp.where(mask[:, None], 0.0, pop.signals)
    new_energy = jnp.where(mask, 0.0, pop.energy)
    return pop.replace(
        alive=new_alive,
        carries=new_carries,
        signals=new_signals,
        energy=new_energy,
    )


def apply_auto_reproduce(
    pop: PopState, 
    key: jax.Array, 
    grid_size: int, 
    min_pop: int, 
    energy_thresh: float = 0.8,
    energy_cost: float = 0.4
) -> PopState:
    """
    JAX-compatible reproduction with static shapes.
    1. Enforces min_pop by cloning random alive agents.
    2. Allows agents with energy >= energy_thresh to clone themselves.
    """
    n = pop.max_pop
    alive_count = jnp.sum(pop.alive)
    
    # 1. Parents who reproduce due to high energy
    energy_repro_mask = pop.alive & (pop.energy >= energy_thresh)
    
    # 2. How many shortfall to reach min_pop?
    shortfall = jnp.maximum(0, min_pop - alive_count)
    
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    
    # Safe random parent sampling for shortfall
    parent_weights = jnp.where(pop.alive, 1.0, 0.0)
    safe_parent_weights = jnp.where(jnp.sum(parent_weights) > 0, parent_weights, jnp.ones(n))
    shortfall_parents = jax.random.choice(k1, jnp.arange(n), shape=(n,), p=safe_parent_weights)
    
    # Map which dead slots get a spawn
    dead_mask = ~pop.alive
    
    # Assign a random priority to dead slots to pick which ones get resurrected
    dead_prio = jax.random.uniform(k2, (n,))
    dead_prio = jnp.where(dead_mask, dead_prio, -1.0)
    sorted_dead_idx = jnp.argsort(dead_prio)[::-1]
    
    num_energy_repro = jnp.sum(energy_repro_mask)
    total_spawns = jnp.minimum(shortfall + num_energy_repro, jnp.sum(dead_mask))
    
    # Boolean mask of size (n,) indicating which elements in sorted_dead_idx are activated
    activated_ranks_mask = jnp.arange(n) < total_spawns
    
    # Inverse map to see which actual slots (0..n-1) are activated
    activate_mask = jnp.zeros(n, dtype=bool)
    activate_mask = activate_mask.at[sorted_dead_idx].set(activated_ranks_mask)
    activate_mask = activate_mask & dead_mask
    
    # Extract indices of energy parents and push them to the front (since valid >= 0, invalid = -1)
    energy_parents = jnp.where(energy_repro_mask, jnp.arange(n), -1)
    energy_parents_sorted = jnp.sort(energy_parents)[::-1]
    
    # Assign parents for each rank
    ranks = jnp.arange(n)
    parent_for_rank = jnp.where(
        ranks < shortfall, 
        shortfall_parents, 
        energy_parents_sorted[jnp.clip(ranks - shortfall, 0, n-1)]
    )
    
    # Map assigned parents back to the activated dead slots
    assigned_parents = jnp.zeros(n, dtype=jnp.int32)
    assigned_parents = assigned_parents.at[sorted_dead_idx].set(parent_for_rank)
    
    # Clone logic
    new_pos_y = jax.random.randint(k3, (n,), 0, grid_size)
    new_pos_x = jax.random.randint(k4, (n,), 0, grid_size)
    new_pos = jnp.stack([new_pos_y, new_pos_x], axis=1)
    
    parent_carries = pop.carries[assigned_parents]
    parent_signals = pop.signals[assigned_parents]
    parent_teams = pop.team[assigned_parents]
    parent_layers = pop.n_layers[assigned_parents]
    
    new_alive = jnp.where(activate_mask, True, pop.alive)
    new_positions = jnp.where(activate_mask[:, None], new_pos, pop.positions)
    new_ages = jnp.where(activate_mask, 0, pop.ages)
    
    # Energy updates: new agents get 1.0, reproducing agents lose energy_cost
    new_energy = jnp.where(activate_mask, 1.0, pop.energy)
    new_energy = jnp.where(energy_repro_mask & ~activate_mask, new_energy - energy_cost, new_energy)
    
    new_teams = jnp.where(activate_mask, parent_teams, pop.team)
    new_layers = jnp.where(activate_mask, parent_layers, pop.n_layers)
    new_carries = jnp.where(activate_mask[:, None], parent_carries, pop.carries)
    new_signals = jnp.where(activate_mask[:, None], parent_signals, pop.signals)
    new_nb_gain = jnp.where(activate_mask, 1.0, pop.nb_gain)
    
    pop = pop.replace(
        alive=new_alive,
        positions=new_positions,
        ages=new_ages,
        energy=new_energy,
        team=new_teams,
        n_layers=new_layers,
        carries=new_carries,
        signals=new_signals,
        nb_gain=new_nb_gain
    )
    
    if pop.memory_buffer is not None:
        parent_mem = pop.memory_buffer[assigned_parents]
        new_mem = jnp.where(activate_mask[:, None, None], parent_mem, pop.memory_buffer)
        pop = pop.replace(memory_buffer=new_mem)
        
    return pop


def update_memory_buffer(
    pop: PopState,
    recv_signals: jnp.ndarray,  # (max_pop, signal_dim) — mean of neighbour signals
    actions: jnp.ndarray,         # (max_pop,) int32
    alive: jnp.ndarray,          # (max_pop,) bool
) -> PopState:
    """Shift memory buffer and write new entry at slot 0."""
    if pop.memory_buffer is None:
        return pop

    sig_dim = pop.signal_dim
    # Shift buffer: [t-1, t-2, ...] <- [t, t-1, ...]
    shifted = jnp.roll(pop.memory_buffer, shift=1, axis=1)

    # Write new entry at slot 0
    new_entry = jnp.zeros((pop.max_pop, sig_dim + 2), dtype=jnp.float32)
    new_entry = new_entry.at[:, :sig_dim].set(recv_signals)
    new_entry = new_entry.at[:, sig_dim].set(actions.astype(jnp.float32))
    new_entry = new_entry.at[:, sig_dim + 1].set(alive.astype(jnp.float32))

    new_buffer = shifted.at[:, 0].set(new_entry)
    new_buffer = jnp.where(alive[:, None, None], new_buffer, 0.0)

    return pop.replace(memory_buffer=new_buffer)


def apply_mind_meld(
    pop: PopState,
    grid_size: int,
    radius: int = 1,
    rate: float = 0.1,
    direction: str = "older_to_younger",
) -> PopState:
    """
    Agents within `radius` blend their carries.
    """
    pos = pop.positions
    diff = jnp.abs(pos[:, None, :] - pos[None, :, :])
    diff = jnp.minimum(diff, grid_size - diff)
    dist = jnp.max(diff, axis=-1)  # Chebyshev
    
    alive_mask = pop.alive[:, None] & pop.alive[None, :]
    adj = (dist <= radius) & alive_mask
    adj = adj.at[jnp.diag_indices(pop.max_pop)].set(False)
    
    if direction == "older_to_younger":
        # adj[i, j] True if j is adjacent to i and j is older than i (so i learns from j)
        older_mask = pop.ages[None, :] > pop.ages[:, None]
        adj = adj & older_mask
        
    n_nb = jnp.sum(adj, axis=1, keepdims=True)
    has_nb = n_nb > 0
    
    nb_carries = jnp.dot(adj.astype(jnp.float32), pop.carries) / jnp.maximum(n_nb, 1.0)
    
    new_carries = jnp.where(
        has_nb,
        (1.0 - rate) * pop.carries + rate * nb_carries,
        pop.carries
    )
    
    return pop.replace(carries=new_carries)
