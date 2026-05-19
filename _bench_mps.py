"""
_bench_mps.py — Benchmark PyTorch MPS vs CPU for THRONG forward pass + PPO.

Usage:
    python _bench_mps.py

Reports:
  - Device detected
  - Forward pass time (N=400 agents, T=1 step)
  - Full PPO cycle time (T=128 steps, N=400 agents, minibatch=2048)
  - Estimated steps/second
"""

import time
import numpy as np
import torch
import yaml

from agents.network_torch import TorchBrain, DEVICE, compute_obs_dim_torch


def bench(device_label: str, device: torch.device) -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    brain = TorchBrain(config, device=device)
    brain.model.eval()

    N       = config["population_size"]
    T       = config["ppo_rollout_steps"]
    obs_dim = compute_obs_dim_torch(config)
    hid_dim = config["agent_hidden_dim"]
    n_layers = int(config.get("n_layers", 2))
    rng     = np.random.default_rng(0)

    carries = np.zeros((N, hid_dim), dtype=np.float32)
    obs_np  = rng.random((N, obs_dim), dtype=np.float32)

    # ── Warmup ───────────────────────────────────────────────────────────────
    for _ in range(3):
        brain.forward(carries, obs_np, n_layers)

    # ── Forward pass benchmark ────────────────────────────────────────────────
    REPS = 50
    t0 = time.perf_counter()
    for _ in range(REPS):
        brain.forward(carries, obs_np, n_layers)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000 / REPS

    # ── PPO cycle benchmark ───────────────────────────────────────────────────
    # Build a fake buffer
    buf = {
        "obs":       rng.random((T, N, obs_dim), dtype=np.float32),
        "actions":   rng.integers(0, 5, (T, N)).astype(np.int32),
        "log_probs": rng.random((T, N), dtype=np.float32).astype(np.float32) * -2,
        "values":    rng.random((T, N), dtype=np.float32),
        "rewards":   rng.random((T, N), dtype=np.float32) * 0.1,
        "dones":     np.zeros((T, N), dtype=np.float32),
        "alive":     np.ones((T, N),  dtype=np.float32),
        "warmup_ok": np.ones((T, N),  dtype=np.float32),
    }
    last_val = np.zeros(N, dtype=np.float32)

    # Warmup PPO
    brain.ppo_update(buf, last_val, n_layers, rng)

    PPO_REPS = 5
    t0 = time.perf_counter()
    for _ in range(PPO_REPS):
        brain.ppo_update(buf, last_val, n_layers, rng)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    ppo_ms = (time.perf_counter() - t0) * 1000 / PPO_REPS

    # Steps per second: one full sim step = 1 forward pass (~14ms at 400 agents)
    # PPO fires every T=128 steps, so amortised PPO cost = ppo_ms / 128
    step_ms = fwd_ms + ppo_ms / T
    steps_per_sec = 1000.0 / step_ms

    print(f"\n{'─'*50}")
    print(f"  Device:          {device_label} ({device})")
    print(f"  Forward pass:    {fwd_ms:.2f} ms  (N={N})")
    print(f"  PPO cycle:       {ppo_ms:.1f} ms  (T={T}, N={N})")
    print(f"  Steps/sec (est): {steps_per_sec:.1f}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    print(f"\nTHRONG MPS Benchmark")
    print(f"PyTorch {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Auto-selected device: {DEVICE}")

    bench("auto (best)", DEVICE)

    cpu = torch.device("cpu")
    if DEVICE.type != "cpu":
        bench("cpu (baseline)", cpu)
