"""Item 5 profile: forward + backward of the learner replay (scan over T steps, KV memory) for EpAgentDev vs EpAgentDevD4,
plus per-component timings of the D4 step (jitted, CPU). Microbenchmark only (no training)."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy")
from w2.ep_agent_dev import EpAgentDev
from w2.d4 import EpAgentDevD4
S, T, B, Dx = 96, 64, 16, 126
key = jax.random.PRNGKey(0)
def bench(net, HG, reps=5):
    p = net.init(key, jnp.zeros((HG + S * 11 + 1,)), jnp.zeros((Dx,)))
    xs = jax.random.normal(key, (B, T, Dx)); mem = jax.random.normal(key, (B, S + T, 11)); valid = jnp.ones((B, T, S + T), bool)
    def loss(p):
        def one(x, m, v):
            K, V = net.apply(p, m, method=type(net).kv)
            def body(h, xv):
                xx, vv = xv; h2, lg, val, hl = net.apply(p, h, xx, K, V, vv, method=type(net).step_kv); return h2, lg.sum() + val + hl.sum()
            _, o = jax.lax.scan(body, jnp.zeros(HG), (x, v)); return o.sum()
        return jax.vmap(one)(xs, mem, valid).sum()
    g = jax.jit(jax.grad(loss)); jax.block_until_ready(g(p)); t0 = time.time()
    for _ in range(reps): jax.block_until_ready(g(p))
    return (time.time() - t0) / reps
tb = bench(EpAgentDev(S=S, shared_tok=True), 128); td = bench(EpAgentDevD4(S=S, shared_tok=True), 8 * 128)
print(f"replay fwd+bwd (B {B} x T {T}): base {tb*1e3:.1f} ms, D4 {td*1e3:.1f} ms, ratio x{td/tb:.2f}")
# MAC count per step (forward), by component
H, Z, A = 128, 64, 32; n = S + T; side_in = (Dx - 100) + 2
base = dict(trunk=100*128 + 128*128, side_enc=(side_in + 128) * Z, gru=3 * (Z + H) * H, f1=Dx * Z, q_att=Z * 16 + n * A * 2, f2=(H + A) * 64, heads=(64 + A) * 8 + (64 + 8) * 7)
d4 = {k: v * 8 for k, v in base.items()}; d4["trunk"] += base["trunk"]
tot_b, tot_d = sum(base.values()), sum(d4.values())
print("MACs/step base %d, D4 %d (x%.2f)" % (tot_b, tot_d, tot_d / tot_b)); print("D4 share by component:", {k: round(v / tot_d, 3) for k, v in d4.items()})
