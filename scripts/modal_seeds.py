"""Cam Nov 21 ~21:40Z item 5 (PREPARED, NOT LAUNCHED; launches only after seed 9's verdict + Cam's go): Stage-1 confirm
seeds on Modal L4, one app per seed (throng-seed<S>) so each has its own per-item spend watch. The seed-9 recipe exactly
(scripts/modal_seed/run_seed_modal.sh = docs/data/port_w2/j1/run_seed.sh with container paths, 3000 s chunks and
idempotent legs), then the 3-table verdict in the container. Code = throng-toy comm-toy a7087f2 (SHA-checked); the J1
reference and twin JSONs + the helper scripts are baked into the image. Outputs: volume throng-runs-b1, seeds/s<S>/.

    MODAL_PROFILE=bazingagi1 SEED_APP=throng-seed10 modal run --detach scripts/modal_seeds.py --seed 10
    ~/throng-smoke/item_watch_b1.sh throng-seed10 1.053 <cap>
Copy back:  MODAL_PROFILE=bazingagi1 modal volume get throng-runs-b1 seeds/s10 ~/throng-backup/modal_b1/seeds/
"""
from __future__ import annotations

import os

import modal

SHA = "a7087f23e5fd5aa2dfa01ff079d984a6c843ec88"
REPO = "https://github.com/overlordxrz-source/throng.git"
VOL = "/mnt/runs"
HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(os.path.dirname(HERE), "docs", "data")
J1 = os.path.expanduser("~/throng-backup/scratch/j1")
AUX = {  # container path -> local source
    "/root/aux/run_seed_modal.sh": os.path.join(HERE, "modal_seed", "run_seed_modal.sh"),
    "/root/aux/decomp_d12.py": os.path.join(HERE, "modal_seed", "decomp_d12.py"),
    "/root/aux/verdict3.py": os.path.join(DOCS, "port_w2", "j1", "verdict3.py"),
    "/root/aux/head_later.py": os.path.join(DOCS, "port_w2", "e1", "head_later.py"),
    "/root/aux/bar3.py": os.path.join(DOCS, "learner_w2", "bar3.py"),
    "/root/aux/bysec.py": os.path.join(DOCS, "learner_w2", "relabel_screen", "policy_diag", "bysec.py"),
    "/root/aux/j1_ref_mean.json": os.path.join(J1, "j1_ref_mean.json"),
    "/root/aux/tw_eval_d12.json": os.path.join(J1, "tw_eval_d12.json"),
}
app = modal.App(os.environ.get("SEED_APP", "throng-seed"))
volume = modal.Volume.from_name("throng-runs-b1", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "zsh")
         .run_commands(f"git clone {REPO} /root/tt", f"cd /root/tt && git checkout {SHA}", f'cd /root/tt && test "$(git rev-parse HEAD)" = "{SHA}"')
         .pip_install("jax[cuda12]==0.10.0", "flax==0.12.7", "optax==0.2.8", "numpy==2.5.4"))
for dst, src in AUX.items():
    image = image.add_local_file(src, dst, copy=True)


@app.function(image=image, gpu="L4", cpu=4.0, memory=8192, timeout=24 * 60 * 60, retries=modal.Retries(max_retries=2, initial_delay=10.0),
              volumes={VOL: volume})
def run_seed(seed: int) -> str:
    import subprocess
    import time
    os.chdir("/root/tt")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == SHA
    out = f"{VOL}/seeds/s{seed}"; os.makedirs(out, exist_ok=True)
    log = open(f"{out}/run.out", "a"); log.write(f"=== container start {time.strftime('%H:%M:%S')} seed {seed}\n"); log.flush()
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used", "--format=csv,noheader", "-l", "30"],
                           stdout=open(f"{out}/gpu_util.csv", "a"), stderr=subprocess.STDOUT)
    p = subprocess.Popen(["zsh", "/root/aux/run_seed_modal.sh", str(seed)], stdout=log, stderr=subprocess.STDOUT)
    while p.poll() is None:
        time.sleep(60); log.flush(); volume.commit()
    smi.kill(); log.write(f"=== container exit {p.returncode} {time.strftime('%H:%M:%S')}\n"); log.close(); volume.commit()
    if p.returncode != 0:
        raise RuntimeError(f"seed {seed}: script exit {p.returncode} (retry resumes from the volume)")
    return f"seed {seed}: done"


@app.local_entrypoint()
def main(seed: int):
    print(run_seed.remote(seed))
