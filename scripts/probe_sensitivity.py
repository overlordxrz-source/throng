"""Step 1 (Cam, 2026-11-07): listener action sensitivity to the received token, start (ORACLE0) vs end (final params).

For each params file: a 96-step COMM-arm rollout (W worlds, real obs builder logic copied from comm_testbed.py at
ceadaf8, the policy's own actions, the live wire). At every step, for each LISTENER, recompute its action
distribution with the received broadcast replaced by (a) the broadcast its sender would emit for a different active
hearth T' (what a meaningful channel would change) and (b) a uniformly random token triple. Same carry, same obs
otherwise. Report mean TV and mean / p99 max|dlogit| over listener steps. Also per-role token entropy of the policy
(sender vs listener) and the fraction of sender steps whose sampled token changes under T -> T' (paired keys).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jax, jax.numpy as jnp, yaml, flax.serialization  # noqa: E402
from jax_sim import rl_jax  # noqa: E402
from jax_sim.action_space import mask_disabled_actions  # noqa: E402
from jax_sim.grid_jax import generate_hearth_positions  # noqa: E402
from jax_sim.main_jax import _normalize_config  # noqa: E402
from jax_sim.network_jax import AgentNetworkJax, init_agent_params, make_model_apply  # noqa: E402
from jax_sim.obs_layout import make_obs_layout  # noqa: E402
from jax_sim.rl_jax import token_logits  # noqa: E402
from comm_testbed import hearth_rel_multi  # noqa: E402
from learner_testbed import DELTAS  # noqa: E402

TAU, G, W, STEPS = 0.08, 32, 32, 96
cfg = _normalize_config(yaml.safe_load((ROOT / "config.yaml").read_text()))
nl, hidden = int(cfg["n_layers"]), int(cfg["hidden_dim"])
N = 2 * W
HEARTHS = np.asarray(generate_hearth_positions(G))
L = make_obs_layout(signal_dim=cfg["signal_dim"], symbol_dim=cfg["symbol_dim"], memory_slots=cfg.get("memory_slots", 0),
                    neighbor_k=cfg["neighbor_k"], local_cells=(2 * cfg["local_obs_radius"] + 1) ** 2,
                    env_channels=int(cfg.get("env_channels", 15)), own_state_dim=int(cfg.get("own_state_dim", 29)))
model = AgentNetworkJax(
    hidden_dim=hidden, n_heads=cfg["n_heads"], n_layers=nl, obs_dim=L.total_dim, signal_dim=cfg["signal_dim"],
    symbol_dim=cfg["symbol_dim"], vocab_size=cfg["vocab_size"], vq_beta=float(cfg.get("vq_beta", 0.25)),
    vq_dead_code_reset=bool(cfg.get("vq_dead_code_reset", True)), memory_slots=cfg.get("memory_slots", 0),
    fwd_env_dim=L.loc_env_end - L.loc_env_start,
    cross_attn_enabled=bool((cfg.get("phase9_canvas") or {}).get("cross_attn_enabled", False)),
    cross_attn_num_heads=int((cfg.get("phase9_canvas") or {}).get("cross_attn_num_heads", cfg["n_heads"])),
    env_channels=int(cfg.get("env_channels", 15)), own_state_dim=L.own_state_dim, n_actions=int(cfg["n_actions"]),
    local_cells=L.local_cells, neighbor_k=cfg["neighbor_k"])
apply_fn = make_model_apply(model)
ns, ne, s0 = L.nb_sigs_start, L.nb_sigs_start + int(cfg["signal_dim"]), L.own_state_start
is_listener = np.tile([False, True], W)
V = int(cfg["vocab_size"])


@jax.jit
def fwd(params, carries, obs, k):
    new_c, o = apply_fn(params, carries, obs, nl)
    logits = mask_disabled_actions(o.action_logits, axis=-1)
    p = params["params"] if "params" in params else params
    bc, toks, ent = [jnp.zeros((obs.shape[0], 8))], [], []
    for i, lg in enumerate(token_logits(params, o.z_e, TAU)):
        t = jax.random.categorical(jax.random.fold_in(k, i), lg)
        lsm = jax.nn.log_softmax(lg, -1)
        ent.append(-jnp.sum(jnp.exp(lsm) * lsm, -1))
        toks.append(t); bc.append(p[rl_jax.TOKEN_SLOTS[i][2]]["embedding"][t])
    return new_c, logits, jnp.concatenate(bc, -1), jnp.stack(toks, -1), jnp.stack(ent, -1).sum(-1)


def build_obs(pos, active, wire, rng):
    obs = np.zeros((N, L.total_dim), np.float32)
    obs[:, s0 + 2] = 1.0; obs[:, s0 + 3] = nl / 6.0
    obs[:, s0 + 4] = pos[:, 0] / G; obs[:, s0 + 5] = pos[:, 1] / G; obs[:, s0 + 6] = 1.0
    obs[:, s0 + 13: s0 + 21] = hearth_rel_multi(pos, HEARTHS, G)
    informed = ~is_listener
    obs[:, s0 + 12] = informed
    obs[:, s0 + 21: s0 + 25] = np.eye(4, dtype=np.float32)[np.repeat(active, 2)] * informed[:, None]
    obs[:, s0 + 25: s0 + 29] = rng.normal(size=(N, 4))
    obs[:, ns:ne] = wire.reshape(W, 2, -1)[:, ::-1].reshape(N, -1)
    return obs


def probe(path, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_agent_params(model, key, jnp.zeros((1, hidden)), jnp.zeros((1, L.total_dim)), nl)
    params = flax.serialization.from_bytes(params, Path(path).read_bytes())
    p = params["params"] if "params" in params else params
    cbs = [np.asarray(p[k]["embedding"]) for _, _, k in rl_jax.TOKEN_SLOTS]
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, G, (N, 2)); active = rng.integers(0, 4, W); wire = np.zeros((N, 40), np.float32)
    carries = jnp.zeros((N, hidden))
    tv_sw, tv_rand, dl_sw, dl_rand, ent_s, ent_l, tok_changed = [], [], [], [], [], [], []
    for t in range(STEPS):
        k = jax.random.PRNGKey(10_000 * seed + t)
        obs = build_obs(pos, active, wire, rng)
        new_c, logits, bc, toks, ent = map(np.asarray, fwd(params, carries, jnp.asarray(obs), k))
        # sender's broadcast for T' with the same carry and key (paired)
        sw_obs = obs.copy(); sw_obs[:, s0 + 21: s0 + 25] = np.eye(4, dtype=np.float32)[np.repeat((active + 1) % 4, 2)] * (~is_listener)[:, None]
        _, _, bc_sw, toks_sw, _ = map(np.asarray, fwd(params, carries, jnp.asarray(sw_obs), k))
        tok_changed.append(float((toks_sw[~is_listener] != toks[~is_listener]).any(1).mean()))
        # listener with the received wire replaced: (a) sender's T' broadcast from the PREVIOUS step's state is not
        # available, so use this step's T' broadcast as the counterfactual message; (b) random token triple
        rand_bc = np.concatenate([np.zeros((N, 8), np.float32)] + [cb[rng.integers(0, V, N)] for cb in cbs], -1)
        for alt, tv_l, dl_l in ((bc_sw, tv_sw, dl_sw), (rand_bc, tv_rand, dl_rand)):
            w2 = wire.copy(); w2[~is_listener] = alt[~is_listener]
            o2 = obs.copy(); o2[:, ns:ne] = w2.reshape(W, 2, -1)[:, ::-1].reshape(N, -1)
            _, lg2, _, _, _ = map(np.asarray, fwd(params, carries, jnp.asarray(o2), k))
            a1, a2 = logits[is_listener], lg2[is_listener]
            p1 = np.exp(a1 - a1.max(1, keepdims=True)); p1 /= p1.sum(1, keepdims=True)
            p2 = np.exp(a2 - a2.max(1, keepdims=True)); p2 /= p2.sum(1, keepdims=True)
            tv_l.append(0.5 * np.abs(p1 - p2).sum(1))
            fin = np.isfinite(a1) & np.isfinite(a2)
            dl_l.append(np.where(fin, np.abs(a1 - a2), 0).max(1))
        ent_s.append(ent[~is_listener].mean()); ent_l.append(ent[is_listener].mean())
        # move with the sampled policy actions (dynamics only to get realistic carries/positions; no reward needed)
        act = np.asarray(jax.random.categorical(jax.random.fold_in(k, 99), jnp.asarray(logits)))
        pos = (pos + DELTAS[act]) % G
        if t % 32 == 31:
            active = (active + rng.integers(1, 4, W)) % 4
        wire, carries = bc, jnp.asarray(new_c)
    cat = np.concatenate
    return dict(params=Path(path).name, tv_Tprime=float(cat(tv_sw).mean()), tv_random=float(cat(tv_rand).mean()),
                maxdlogit_Tprime_mean=float(cat(dl_sw).mean()), maxdlogit_random_mean=float(cat(dl_rand).mean()),
                maxdlogit_random_p99=float(np.percentile(cat(dl_rand), 99)),
                token_entropy_sender=float(np.mean(ent_s)), token_entropy_listener=float(np.mean(ent_l)),
                sender_token_changes_under_Tprime=float(np.mean(tok_changed)))


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(json.dumps(probe(f)), flush=True)
