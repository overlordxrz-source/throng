"""Read-only watcher (Cam ~17:35Z item 4): at every new chunk checkpoint of the oracle-screen arms, record steps, frontier,
O-T's g, and the trunk's drift from the pretrained navigator (relative L2 change of Dense_0-2 kernels + biases)."""
import glob, json, os, pickle, sys, time
import numpy as np
import flax.serialization, jax, jax.numpy as jnp
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy")
from w2.learn import Net
O = os.path.dirname(os.path.abspath(__file__))
nav = flax.serialization.from_bytes(Net().init(jax.random.PRNGKey(1), jnp.zeros((1, 100))), open(sorted(glob.glob("/Users/overlord/CascadeProjects/throng-toy/w2_out/nav_w13/*.msgpack"))[0], "rb").read())["params"]
def drift(P):
    out = {}
    for k in ("Dense_0", "Dense_1", "Dense_2"):
        for w in ("kernel", "bias"):
            a, b = np.asarray(P[k][w], np.float64), np.asarray(nav[k][w], np.float64)
            out[f"{k}.{w}"] = round(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)), 4)
    out["Dense_2.bias(stay,-x,+x,+y,-y,eat) delta"] = [round(float(v), 3) for v in np.asarray(P["Dense_2"]["bias"]) - np.asarray(nav["Dense_2"]["bias"])]
    return out
seen = {}
while True:
    for A in ("X", "T"):
        f = f"{O}/{A}.ckpt"
        if os.path.exists(f) and os.path.getmtime(f) != seen.get(A):
            seen[A] = os.path.getmtime(f)
            try:
                C = pickle.load(open(f, "rb")); P = C["p"]["params"]
                row = dict(arm=A, steps=int(C["steps"]), frontier=float(C["dfront"]), g=(float(P["topo_g"]) if "topo_g" in P else None), drift=drift(P), t=time.strftime("%H:%M:%S"))
                open(f"{O}/snap_g.jsonl", "a").write(json.dumps(row) + "\n")
            except Exception as e:
                open(f"{O}/snap_g.jsonl", "a").write(json.dumps(dict(arm=A, error=str(e))) + "\n")
    if os.path.exists(f"{O}/queue_os.out") and "OSDONE" in open(f"{O}/queue_os.out").read():
        break
    time.sleep(30)
