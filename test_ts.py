import tensorstore as ts
import numpy as np

spec = {
    'driver': 'zarr',
    'kvstore': {
        'driver': 'ocdbt',
        'path': 'runs/jax_run/checkpoints/1860/default/b_params/codebook/embedding',
    }
}
try:
    dataset = ts.open(spec).result()
    cb = dataset.read().result()
    print(cb.shape)
except Exception as e:
    print(e)
