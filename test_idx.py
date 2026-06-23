from jax_sim.obs_layout import make_obs_layout

layout = make_obs_layout(
    signal_dim=40,
    symbol_dim=16,
    memory_slots=20,
    neighbor_k=6,
    local_cells=25,
    env_channels=15,
    own_state_dim=22,
)
print("mem_start:", layout.mem_start)
print("loc_cult_slow_start:", layout.loc_cult_slow_start)
