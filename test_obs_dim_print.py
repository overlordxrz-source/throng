import yaml
from agents.network_torch import compute_obs_dim_torch
config = yaml.safe_load(open("config_phase7.yaml"))
d = 0
d += 1 # age
print(f"age: {d}")
d += 2 # norm_x, norm_y
d += 2 # vx, vy
d += 1 # energy
K = int(config["neighbor_k"])
d += K * int(config["signal_dim"])
print(f"K*sig: {K * int(config['signal_dim'])}")
r = int(config["local_obs_radius"])
W = (2 * r + 1) ** 2
print(f"W: {W}")
d += W * int(config["symbol_dim"])
print(f"sym: {W * int(config['symbol_dim'])}")
d += W * int(config.get("env_channels", 9))
print(f"env: {W * int(config.get('env_channels', 9))}")
d += int(config.get("memory_slots", 0))
d += int(config.get("phase9_canvas", {}).get("cross_attn_num_heads", 4)) * int(config["signal_dim"])
print(f"cross_attn: {int(config.get('phase9_canvas', {}).get('cross_attn_num_heads', 4)) * int(config['signal_dim'])}")
print(f"total: {d}")
