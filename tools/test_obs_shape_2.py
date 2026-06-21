import jax
import sys
import os
sys.path.append('.')
from scripts.modal_train import build_cfg
from jax_sim.train_entry import run_simulation

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
cfg = build_cfg()
cfg['checkpoint_dir'] = '/tmp/throng-runs/checkpoints'
os.makedirs('/tmp/throng-runs/checkpoints', exist_ok=True)
os.makedirs('/tmp/throng-runs/runs', exist_ok=True)
try:
    run_simulation(cfg, seed=42, n_steps=1)
except Exception as e:
    import traceback
    traceback.print_exc()
