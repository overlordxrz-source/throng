"""Step 0 equivalence (registered bar, docs/RESEARCH_PROTOCOL.md 2026-11-08): the jitted rollout vs the Python
path, COMM arm at ORACLE0 params, 3 fixed seeds x 512 steps, identical injected draws and key schedule.
PASS = actions, rewards and positions identical at every step for every agent, all 3 seeds.

usage: equiv_comm_jit.py <ORACLE0.msgpack> <workdir>
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
G, W, T, HR = 32, 32, 512, 1
N = 2 * W


def valid_cells():
    sys.path.insert(0, str(ROOT))
    from jax_sim.grid_jax import generate_hearth_positions
    H = np.asarray(generate_hearth_positions(G))
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    d = np.abs(cells[:, None, :] - H[None]); d = np.minimum(d, G - d).max(-1)
    return cells[(d > HR).all(1)]


def main(warm, work):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    V = valid_cells()
    common = ["--arm", "COMM", "--tau", "0.08", "--craft-cost", "0.1", "--warm-start", warm]
    all_ok = True
    for seed in (0, 1, 2):
        rng = np.random.default_rng(1000 + seed)
        base = str(work / f"seed{seed}")
        np.savez(base + ".draws.npz", init_pos=V[rng.integers(0, len(V), N)], init_active=rng.integers(0, 4, W),
                 resp=V[rng.integers(0, len(V), (T, N))], inc=rng.integers(1, 4, (T, W)), key_seed=seed)
        for script, flag in (("comm_testbed.py", "--dump-equiv"), ("comm_testbed_jit.py", "--equiv")):
            r = subprocess.run([PY, str(ROOT / "scripts" / script), *common, flag, base], capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-4000:]); raise SystemExit(f"{script} failed")
        ref, jit = np.load(base + ".ref.npz"), np.load(base + ".jit.npz")
        from jax_sim.obs_layout import make_obs_layout  # noqa: F401  (layout only to locate pos dims)
        s0 = None
        import yaml
        from jax_sim.main_jax import _normalize_config
        cfg = _normalize_config(yaml.safe_load((ROOT / "config.yaml").read_text()))
        from jax_sim.obs_layout import make_obs_layout as mol
        L = mol(signal_dim=cfg["signal_dim"], symbol_dim=cfg["symbol_dim"], memory_slots=cfg.get("memory_slots", 0),
                neighbor_k=cfg["neighbor_k"], local_cells=(2 * cfg["local_obs_radius"] + 1) ** 2,
                env_channels=int(cfg.get("env_channels", 15)), own_state_dim=int(cfg.get("own_state_dim", 29)))
        s0 = L.own_state_start
        fields = {
            "actions": (ref["actions"], jit["actions"]),
            "rewards": (ref["rewards"], jit["rewards"]),
            "positions": (np.rint(ref["obs"][:, :, s0 + 4:s0 + 6] * G), np.rint(jit["obs"][:, :, s0 + 4:s0 + 6] * G)),
            "obs": (ref["obs"], jit["obs"]),
            "token_actions": (ref["token_actions"], jit["token_actions"]),
            "log_probs": (ref["log_probs"], jit["log_probs"]),
        }
        line = []
        for name, (x, y) in fields.items():
            assert x.shape == y.shape, (name, x.shape, y.shape)
            same = np.array_equal(x, y)
            if same:
                line.append(f"{name} identical")
            else:
                bad = np.nonzero((x != y).reshape(T, -1).any(1))[0]
                line.append(f"{name} DIFFER first@{bad[0]} ({len(bad)} steps; max|d| {np.abs(x.astype(float) - y).max():.3g})")
            if name in ("actions", "rewards", "positions"):
                all_ok &= same
        deps = int((ref["rewards"] > 2).sum())
        print(f"seed {seed}: " + " | ".join(line) + f" | listener-or-sender deposits in rollout: {deps}", flush=True)
    print("EQUIVALENCE (registered bar: actions, rewards, positions exact):", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
