import time, numpy as np, yaml, torch
from agents.network_torch import TorchBrain
from agents.rl_torch import ppo_update_torch

cfg = yaml.safe_load(open('config.yaml'))
T, N, obs_dim = 512, 400, 207
rng = np.random.default_rng(42)

hd = int(cfg['agent_hidden_dim'])
buf = {
    'obs':         np.random.randn(T, N, obs_dim).astype('float32'),
    'actions':     np.random.randint(0, 5, (T, N)).astype('int32'),
    'log_probs':   np.random.randn(T, N).astype('float32'),
    'values':      np.random.randn(T, N).astype('float32'),
    'rewards':     np.random.randn(T, N).astype('float32'),
    'dones':       np.zeros((T, N), dtype='float32'),
    'alive':       np.zeros((T, N), dtype='float32'),
    'warmup_ok':   np.ones((T, N), dtype='float32'),
    'tom_targets': np.random.randint(-1, 5, (T, N, 6)).astype('int64'),
    'carries':     np.random.randn(T, N, hd).astype('float32') * 0.1,
}
buf['alive'][:, np.random.choice(N, 30, replace=False)] = 1.0
last_val = np.zeros(N, dtype='float32')

for label, device in [('MPS', None), ('CPU', torch.device('cpu'))]:
    brain = TorchBrain(cfg, device=device)
    ppo_update_torch(brain, buf, last_val, 6, rng)  # warmup
    t0 = time.perf_counter()
    stats = ppo_update_torch(brain, buf, last_val, 6, rng)
    elapsed = time.perf_counter() - t0
    print(f"{label}  PPO epochs=2  alive=30  T=512: {elapsed:.2f}s  "
          f"clip={stats['ppo_clip_frac']:.3f}")

    carries = np.zeros((N, cfg['agent_hidden_dim']), dtype='float32')
    obs = np.random.randn(N, obs_dim).astype('float32')
    for _ in range(5): brain.forward(carries, obs, 6)
    t0 = time.perf_counter()
    for _ in range(100): brain.forward(carries, obs, 6)
    print(f"{label}  forward N=400: {(time.perf_counter()-t0)/100*1000:.1f}ms/call")
