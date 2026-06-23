import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("config env_channels:", config.get("env_channels"))
print("config own_state_dim:", config.get("own_state_dim"))
print("config symbol_dim:", config.get("symbol_dim"))
