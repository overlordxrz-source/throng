import msgpack
import os

def patch(d):
    if isinstance(d, dict):
        return {k: patch(v) for k, v in d.items()}
    if isinstance(d, list):
        return [patch(v) for v in d]
    if isinstance(d, str):
        return d.replace('cuda', 'cpu')
    return d

for filename in ['runs/jax_run/checkpoints/1860/default/_sharding', 'runs/jax_run/checkpoints/1860/default/_METADATA']:
    if not os.path.exists(filename): continue
    with open(filename, 'rb') as f:
        data = msgpack.unpackb(f.read())
    data = patch(data)
    with open(filename, 'wb') as f:
        f.write(msgpack.packb(data))
print("Patched successfully!")
