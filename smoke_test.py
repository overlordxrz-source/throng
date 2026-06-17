import yaml
from jax_sim.train_entry import run_simulation
with open("config_phase7.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["checkpoint_dir"] = "/tmp/throng-checkpoints"
cfg["ppo_minibatch_size"] = 128
cfg["ppo_rollout_steps"] = 128
run_simulation(cfg, seed=42, n_steps=128)
