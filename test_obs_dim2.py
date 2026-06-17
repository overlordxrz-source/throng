import yaml
from agents.network_torch import compute_obs_dim_torch
config = yaml.safe_load(open("config_phase7.yaml"))
print(compute_obs_dim_torch(config))
print(f"K = {config.get('neighbor_k')}, r = {config.get('local_obs_radius')}")
