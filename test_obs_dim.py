import yaml
from jax_sim.obs_layout import make_obs_layout

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

layout = make_obs_layout(
    signal_dim=config["signal_dim"],
    symbol_dim=config.get("symbol_dim", 8),
    memory_slots=config.get("memory_slots", 0),
    neighbor_k=config["neighbor_k"],
    local_cells=(2 * config["local_obs_radius"] + 1)**2,
    env_channels=int(config.get("env_channels", 10)),
    own_state_dim=int(config.get("own_state_dim", 10)),
)
print("Total dim:", layout.total_dim)
print("Layout dict:", layout.__dict__)
