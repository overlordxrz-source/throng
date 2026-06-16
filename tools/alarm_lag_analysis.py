#!/usr/bin/env python3
"""
tools/alarm_lag_analysis.py — Temporal lag analysis for the alarm-energy correlation.

Disambiguates two causal hypotheses:
  Direction A: Low energy at T → alarm at T+1 (genuine distress signal)
  Direction B: Alarm at T → low energy at T+1 (reverse causality from metabolic cost)

Runs the simulation from a checkpoint and tracks per-agent alarm state and energy
across consecutive timesteps. Uses logistic regression with Granger-style lag terms
and a chi-squared test to determine which direction dominates.

Usage:
    cd /root/throng && python3 tools/alarm_lag_analysis.py \
        --checkpoint /mnt/throng-runs/checkpoints --steps 500
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint._src.metadata import sharding as orbax_sharding

def mock_to_jax_sharding(self):
    return jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ('x',)),
        jax.sharding.PartitionSpec()
    )
orbax_sharding.NamedShardingMetadata.to_jax_sharding = mock_to_jax_sharding
orbax_sharding.SingleDeviceShardingMetadata.to_jax_sharding = mock_to_jax_sharding

import jax_sim.observations_jax as _obs_mod
original_build_obs = _obs_mod.build_observations_jax
def patched_build_obs(*args, **kwargs):
    obs = original_build_obs(*args, **kwargs)
    N = obs.shape[0]
    with open(ROOT / "config_phase7.yaml") as f:
        cfg = yaml.safe_load(f)
    env_channels = int(cfg.get("env_channels", 10))
    if env_channels == 9 and obs.shape[1] >= (598 + 250):
        part1 = obs[:, :598]
        loc_env = obs[:, 598:598+250].reshape((N, 25, 10))
        loc_env_9 = loc_env[:, :, :9].reshape((N, 225))
        part3 = obs[:, 598+250:]
        return jnp.concatenate([part1, loc_env_9, part3], axis=1)
    return obs
_obs_mod.build_observations_jax = patched_build_obs

from agents.network_torch import compute_obs_dim_torch, compute_fwd_env_dim
from jax_sim.grid_jax import GridState
from jax_sim.main_jax import _normalize_config, make_sim_step
from jax_sim.network_jax import AgentNetworkJax, PredatorNetworkJax, make_model_apply
from jax_sim.population_jax import init_population
from flax.core.frozen_dict import unfreeze, freeze


def get_red_dists(b_pos, b_alive, r_pos, r_alive, gs):
    dy = np.abs(b_pos[:, None, 0] - r_pos[None, :, 0])
    dx = np.abs(b_pos[:, None, 1] - r_pos[None, :, 1])
    dy = np.minimum(dy, gs - dy)
    dx = np.minimum(dx, gs - dx)
    dist = np.maximum(dy, dx)
    dist = np.where(r_alive[None, :], dist, 9999)
    return np.min(dist, axis=1)


def run_lag_analysis(checkpoint_dir: str, steps: int = 500):
    with open(ROOT / "config_phase7.yaml") as f:
        config = yaml.safe_load(f)
    config = _normalize_config(config)
    config["ppo_rollout_steps"] = 1

    gs = int(config["grid_size"])
    max_pop = int(config["max_pop"])
    max_pop_red = int(config["max_pop_red"])
    hidden_d = int(config["hidden_dim"])
    n_layers = int(config["n_layers"])
    sig_d = int(config["signal_dim"])
    fwd_env_dim = compute_fwd_env_dim(config)

    model = AgentNetworkJax(
        hidden_dim=hidden_d,
        n_heads=int(config["n_heads"]),
        n_layers=n_layers,
        obs_dim=0,
        signal_dim=sig_d,
        symbol_dim=int(config["symbol_dim"]),
        vocab_size=int(config["vocab_size"]),
        memory_slots=0,
        fwd_env_dim=fwd_env_dim,
        cross_attn_enabled=config.get("phase9_canvas", {}).get("cross_attn_enabled", False),
        neighbor_k=int(config["neighbor_k"]),
        n_actions=int(config.get("n_actions", 8)),
        env_channels=int(config.get("env_channels", 10)),
    )
    model_apply = make_model_apply(model)

    _p14t = config.get("phase14_transcendental", {})
    model_red = PredatorNetworkJax(
        hidden_dim=int(_p14t.get("hidden_dim", 128)),
        neighbor_k=int(config["neighbor_k"]),
        local_obs_radius=int(config["local_obs_radius"]),
        n_heads=int(config["n_heads"]),
        n_layers=n_layers,
        signal_dim=sig_d,
        symbol_dim=int(config["symbol_dim"]),
        vocab_size=int(_p14t.get("red_vocab_size", 64)),
        vq_beta=float(config.get("vq_beta", 0.25)),
        vq_dead_code_reset=bool(config.get("vq_dead_code_reset", True)),
        cross_attn_enabled=bool(_p14t.get("cross_attn_enabled", False)),
        cross_attn_num_heads=int(config.get("phase9_canvas", {}).get("cross_attn_num_heads", 4)),
        n_actions=int(config.get("n_actions", 8)),
        env_channels=int(config.get("env_channels", 10)),
    )
    r_model_apply = make_model_apply(model_red)

    ckpt_mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer())
    _ckpt_latest = ckpt_mngr.latest_step()
    if _ckpt_latest is None:
        sys.exit(f"No valid checkpoint found in {checkpoint_dir}")
    raw_restored = ckpt_mngr.restore(_ckpt_latest)
    b_params = freeze(unfreeze(raw_restored)["b_params"])
    r_params = freeze(unfreeze(raw_restored)["r_params"])

    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 10)
    grid = GridState(gs, symbol_dim=int(config["symbol_dim"]))
    grid = grid.replace(walls=jnp.zeros((gs, gs), dtype=jnp.bool_))

    b_pop = init_population(max_pop, hidden_d, sig_d, gs, 0, keys[0], max_pop, 20)
    r_pop = init_population(max_pop_red, int(_p14t.get("hidden_dim", 128)),
                            sig_d, gs, 1, keys[1], max_pop_red, 20)
    b_carries = jnp.zeros((max_pop, hidden_d))
    r_carries = jnp.zeros((max_pop_red, int(_p14t.get("hidden_dim", 128))))

    sim_step = make_sim_step(config, model, model_apply, r_model_apply=r_model_apply)

    # ── Per-agent tracking across timesteps ──────────────────────────────
    # We track (alarm_t, energy_t, red_dist_t) for each agent across steps.
    # At step T+1, we pair with agent's state at step T to build lag features.
    prev_alarm = None   # (max_pop,) bool
    prev_energy = None  # (max_pop,) float
    prev_alive = None   # (max_pop,) bool
    prev_red_dist = None  # (max_pop,) float

    # Lag-1 paired records: (alarm_T, alarm_T+1, energy_T, energy_T+1, red_dist_T)
    lag_alarm_t = []
    lag_alarm_tp1 = []
    lag_energy_t = []
    lag_energy_tp1 = []
    lag_red_dist_t = []

    print(f"Running Lag Analysis on checkpoint {_ckpt_latest} for {steps} steps...")
    carry = (grid, b_pop, r_pop, b_carries, r_carries, b_params, r_params)

    for i in range(steps):
        step_key = jax.random.fold_in(keys[2], i)
        carry, rollout = sim_step(carry, (step_key, i))

        b_pop_out = carry[1]
        r_pop_out = carry[2]

        alive = np.asarray(b_pop_out.alive)
        alarms = np.asarray(b_pop_out.alarms)
        alarm_fired = alarms[:, 1] > 0.5
        energy = np.asarray(b_pop_out.energy)

        b_pos = np.asarray(b_pop_out.positions)
        r_pos = np.asarray(r_pop_out.positions)
        r_alive = np.asarray(r_pop_out.alive)
        red_dist = get_red_dists(b_pos, alive, r_pos, r_alive, gs)

        # Pair with previous step if available
        if prev_alarm is not None:
            # Only include agents alive in BOTH steps
            both_alive = alive & prev_alive
            idx = np.where(both_alive)[0]
            if len(idx) > 0:
                lag_alarm_t.extend(prev_alarm[idx])
                lag_alarm_tp1.extend(alarm_fired[idx])
                lag_energy_t.extend(prev_energy[idx])
                lag_energy_tp1.extend(energy[idx])
                lag_red_dist_t.extend(prev_red_dist[idx])

        # Store current state for next step
        prev_alarm = alarm_fired.copy()
        prev_energy = energy.copy()
        prev_alive = alive.copy()
        prev_red_dist = red_dist.copy()

        if (i + 1) % 50 == 0:
            print(f"  Step {i + 1}/{steps}...")

    lag_alarm_t = np.array(lag_alarm_t, dtype=bool)
    lag_alarm_tp1 = np.array(lag_alarm_tp1, dtype=bool)
    lag_energy_t = np.array(lag_energy_t, dtype=np.float32)
    lag_energy_tp1 = np.array(lag_energy_tp1, dtype=np.float32)
    lag_red_dist_t = np.array(lag_red_dist_t, dtype=np.float32)

    N = len(lag_alarm_t)
    print(f"\nTotal paired agent-step records: {N:,}")

    # ── Test 1: Transition probabilities (model-free) ────────────────────
    print("\n" + "=" * 70)
    print(" TEST 1: CONDITIONAL TRANSITION PROBABILITIES")
    print("=" * 70)

    # Direction A: P(alarm_{T+1} | energy_T <= 0.3) vs P(alarm_{T+1} | energy_T > 0.3)
    low_e = lag_energy_t <= 0.3
    high_e = lag_energy_t > 0.3
    p_alarm_given_low_e = np.mean(lag_alarm_tp1[low_e]) if low_e.sum() > 0 else float("nan")
    p_alarm_given_high_e = np.mean(lag_alarm_tp1[high_e]) if high_e.sum() > 0 else float("nan")
    print(f"\n  Direction A: Does low energy CAUSE alarm?")
    print(f"    P(Alarm_{'{T+1}'} | Energy_T <= 0.3) = {p_alarm_given_low_e:.4f}  (n={int(low_e.sum()):,})")
    print(f"    P(Alarm_{'{T+1}'} | Energy_T >  0.3) = {p_alarm_given_high_e:.4f}  (n={int(high_e.sum()):,})")
    diff_a = p_alarm_given_low_e - p_alarm_given_high_e
    print(f"    Δ = {diff_a:+.4f}")

    # Direction B: E[energy_{T+1} | alarm_T=1] vs E[energy_{T+1} | alarm_T=0]
    alarmed = lag_alarm_t
    silent = ~lag_alarm_t
    e_given_alarm = np.mean(lag_energy_tp1[alarmed]) if alarmed.sum() > 0 else float("nan")
    e_given_silent = np.mean(lag_energy_tp1[silent]) if silent.sum() > 0 else float("nan")
    print(f"\n  Direction B: Does alarm CAUSE energy drop?")
    print(f"    E[Energy_{'{T+1}'} | Alarm_T = 1] = {e_given_alarm:.4f}  (n={int(alarmed.sum()):,})")
    print(f"    E[Energy_{'{T+1}'} | Alarm_T = 0] = {e_given_silent:.4f}  (n={int(silent.sum()):,})")
    diff_b = e_given_alarm - e_given_silent
    print(f"    Δ = {diff_b:+.4f}")

    # ── Test 2: Granger-style logistic regression ────────────────────────
    print("\n" + "=" * 70)
    print(" TEST 2: GRANGER-STYLE LOGISTIC REGRESSION")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chi2

    # Direction A: alarm_{T+1} ~ energy_T + alarm_T + red_dist_T
    # Control: alarm_{T+1} ~ alarm_T + red_dist_T (no energy)
    y_a = lag_alarm_tp1.astype(np.float64)

    if len(np.unique(y_a)) < 2:
        print("  [SKIP] Alarm outcomes are all-same — cannot run regression.")
    else:
        def zscore(x):
            s = x.std()
            return (x - x.mean()) / (s + 1e-8)

        X_ctrl_a = np.column_stack([
            lag_alarm_t.astype(np.float64),
            zscore(lag_red_dist_t.astype(np.float64)),
        ])
        X_full_a = np.column_stack([
            lag_alarm_t.astype(np.float64),
            zscore(lag_red_dist_t.astype(np.float64)),
            zscore(lag_energy_t.astype(np.float64)),
        ])

        lr_ctrl_a = LogisticRegression(max_iter=500, C=1.0).fit(X_ctrl_a, y_a)
        lr_full_a = LogisticRegression(max_iter=500, C=1.0).fit(X_full_a, y_a)

        def log_likelihood(model, X, y):
            proba = model.predict_proba(X)
            return float(np.sum(
                y * np.log(proba[:, 1] + 1e-12) +
                (1 - y) * np.log(proba[:, 0] + 1e-12)
            ))

        ll_ctrl_a = log_likelihood(lr_ctrl_a, X_ctrl_a, y_a)
        ll_full_a = log_likelihood(lr_full_a, X_full_a, y_a)
        lrt_a = 2 * (ll_full_a - ll_ctrl_a)
        p_a = float(1 - chi2.cdf(lrt_a, df=1))
        beta_energy = float(lr_full_a.coef_[0][2])

        print(f"\n  Direction A: alarm_{{T+1}} ~ alarm_T + red_dist_T + energy_T")
        print(f"    LRT χ²(1) = {lrt_a:.4f}   p = {p_a:.6f}")
        print(f"    β(energy_T) = {beta_energy:+.4f}")
        if p_a < 0.05:
            direction = "LOWER energy → MORE alarm" if beta_energy < 0 else "HIGHER energy → MORE alarm"
            print(f"    *** SIGNIFICANT: Energy at T Granger-causes Alarm at T+1 ({direction}) ***")
        else:
            print(f"    Not significant — energy at T does not predict alarm at T+1")

    # Direction B: low_energy_{T+1} ~ alarm_T + energy_T + red_dist_T
    # Control: low_energy_{T+1} ~ energy_T + red_dist_T (no alarm)
    y_b = (lag_energy_tp1 <= 0.3).astype(np.float64)

    if len(np.unique(y_b)) < 2:
        print("  [SKIP] Energy outcomes are all-same — cannot run regression.")
    else:
        X_ctrl_b = np.column_stack([
            zscore(lag_energy_t.astype(np.float64)),
            zscore(lag_red_dist_t.astype(np.float64)),
        ])
        X_full_b = np.column_stack([
            zscore(lag_energy_t.astype(np.float64)),
            zscore(lag_red_dist_t.astype(np.float64)),
            lag_alarm_t.astype(np.float64),
        ])

        lr_ctrl_b = LogisticRegression(max_iter=500, C=1.0).fit(X_ctrl_b, y_b)
        lr_full_b = LogisticRegression(max_iter=500, C=1.0).fit(X_full_b, y_b)

        ll_ctrl_b = log_likelihood(lr_ctrl_b, X_ctrl_b, y_b)
        ll_full_b = log_likelihood(lr_full_b, X_full_b, y_b)
        lrt_b = 2 * (ll_full_b - ll_ctrl_b)
        p_b = float(1 - chi2.cdf(lrt_b, df=1))
        beta_alarm = float(lr_full_b.coef_[0][2])

        print(f"\n  Direction B: low_energy_{{T+1}} ~ energy_T + red_dist_T + alarm_T")
        print(f"    LRT χ²(1) = {lrt_b:.4f}   p = {p_b:.6f}")
        print(f"    β(alarm_T) = {beta_alarm:+.4f}")
        if p_b < 0.05:
            direction = "Alarm → MORE low-energy" if beta_alarm > 0 else "Alarm → LESS low-energy"
            print(f"    *** SIGNIFICANT: Alarm at T Granger-causes low energy at T+1 ({direction}) ***")
        else:
            print(f"    Not significant — alarm at T does not predict low energy at T+1")

    # ── Test 3: Energy-binned alarm transition matrix ────────────────────
    print("\n" + "=" * 70)
    print(" TEST 3: ENERGY-BINNED ALARM TRANSITION MATRIX")
    print("=" * 70)

    bins = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
    bin_labels = ["0.00-0.15", "0.15-0.30", "0.30-0.50", "0.50-0.70", "0.70-1.00"]
    print(f"\n  {'Energy Bin':<12} | {'N':<8} | {'P(Alarm T)':<12} | {'P(Alarm T+1)':<14} | {'Δ Alarm':<10}")
    print(f"  {'-'*66}")

    for b in range(len(bins) - 1):
        mask = (lag_energy_t >= bins[b]) & (lag_energy_t < bins[b + 1])
        n = int(mask.sum())
        if n < 20:
            continue
        p_at = np.mean(lag_alarm_t[mask])
        p_atp1 = np.mean(lag_alarm_tp1[mask])
        delta = p_atp1 - p_at
        print(f"  {bin_labels[b]:<12} | {n:<8,} | {p_at:<12.4f} | {p_atp1:<14.4f} | {delta:<+10.4f}")

    # ── Test 4: Metabolic cost accounting ────────────────────────────────
    print("\n" + "=" * 70)
    print(" TEST 4: METABOLIC COST ACCOUNTING")
    print("=" * 70)

    metabolic_cost = 0.006  # from config
    # Among agents who alarmed at T, what is the energy delta vs those who didn't?
    delta_e_alarm = np.mean(lag_energy_tp1[alarmed] - lag_energy_t[alarmed]) if alarmed.sum() > 0 else float("nan")
    delta_e_silent = np.mean(lag_energy_tp1[silent] - lag_energy_t[silent]) if silent.sum() > 0 else float("nan")

    print(f"  Alarm metabolic cost per step: {metabolic_cost}")
    print(f"  Mean ΔEnergy (Alarm=1): {delta_e_alarm:+.6f}  (n={int(alarmed.sum()):,})")
    print(f"  Mean ΔEnergy (Alarm=0): {delta_e_silent:+.6f}  (n={int(silent.sum()):,})")
    print(f"  Difference: {delta_e_alarm - delta_e_silent:+.6f}")
    print(f"  Expected from cost alone: {-metabolic_cost:+.6f}")

    excess = (delta_e_alarm - delta_e_silent) - (-metabolic_cost)
    print(f"  Excess beyond metabolic cost: {excess:+.6f}")
    if abs(excess) < metabolic_cost * 0.5:
        print("  → Energy difference is FULLY EXPLAINED by metabolic cost (Direction B: reverse causality)")
    else:
        print("  → Energy difference EXCEEDS metabolic cost — additional causal mechanism present")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    if 'p_a' in dir() and 'p_b' in dir():
        if p_a < 0.05 and p_b >= 0.05:
            print("  CONCLUSION: Direction A — Low energy CAUSES alarm firing (DISTRESS SIGNAL)")
        elif p_b < 0.05 and p_a >= 0.05:
            print("  CONCLUSION: Direction B — Alarm CAUSES low energy (METABOLIC COST ARTIFACT)")
        elif p_a < 0.05 and p_b < 0.05:
            print("  CONCLUSION: BIDIRECTIONAL — Both causal directions are significant.")
            print("  (Likely the metabolic cost creates a feedback loop with a genuine distress signal)")
        else:
            print("  CONCLUSION: Neither direction significant — alarm and energy are contemporaneously")
            print("  correlated but neither Granger-causes the other at lag-1.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    run_lag_analysis(args.checkpoint, args.steps)
