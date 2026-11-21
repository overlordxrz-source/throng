"""Cam Nov 21 ~21:40Z SPEED SMOKE on bazingagi1: the EXACT seed-9 learner recipe (relabel + no-nav + 4-way + target-KL .015 +
--d4-side; batch / minibatches unchanged; the --early-stop-frontier stopping rule omitted), seed 9, 2e6 steps, on
(a) one CPU container (8 cores, 8 GiB) and (b) one L4 GPU (4 cores, 8 GiB). Code = throng-toy comm-toy at a7087f2 (cloned,
SHA-checked). Outputs on the volume throng-runs-b1 under s9smoke/<tag>/.

    MODAL_PROFILE=bazingagi1 S9_APP=throng-s9smoke-cpu modal run --detach scripts/modal_s9smoke.py::cpu
    MODAL_PROFILE=bazingagi1 S9_APP=throng-s9smoke-l4  modal run --detach scripts/modal_s9smoke.py::l4
Per-item watch: ~/throng-smoke/item_watch_b1.sh <app> <rate $/h> <cap>.
"""
from __future__ import annotations

import os

import modal

SHA = "a7087f23e5fd5aa2dfa01ff079d984a6c843ec88"
REPO = "https://github.com/overlordxrz-source/throng.git"
VOL = "/mnt/runs"
app = modal.App(os.environ.get("S9_APP", "throng-s9smoke"))
volume = modal.Volume.from_name("throng-runs-b1", create_if_missing=True)
_base = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .run_commands(f"git clone {REPO} /root/tt", f"cd /root/tt && git checkout {SHA}", f'cd /root/tt && test "$(git rev-parse HEAD)" = "{SHA}"'))
img_cpu = _base.pip_install("jax==0.10.0", "flax==0.12.7", "optax==0.2.8", "numpy==2.5.4")
img_gpu = _base.pip_install("jax[cuda12]==0.10.0", "flax==0.12.7", "optax==0.2.8", "numpy==2.5.4")

XT = ["--learner", "ep", "--slots", "96", "--shared-tok", "--hindsight", "0.5", "--head-own-meal", "--demonstrator", "--parent-waits",
      "--gate", "honest", "--hunger-gate", "--item-size", "0.1", "--W", "32", "--lookahead", "0.25", "--la-max", "12", "--head-relabel-token",
      "--d4-side"]
X = XT + ["--no-nav", "--target-kl", "0.015"]


def _run(tag: str, steps: str, pin: int = 0) -> str:
    import subprocess
    import time
    os.chdir("/root/tt")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == SHA
    out = f"{VOL}/s9smoke/{tag}"; os.makedirs(out, exist_ok=True)
    log = open(f"{out}/run.log", "a")
    pre = ["taskset", "-c", f"0-{pin - 1}"] if pin else []                 # Cam 21:50Z: pin XLA / JAX to the allotted cores
    env = dict(os.environ, **({"OMP_NUM_THREADS": str(pin), "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true"} if pin else {}))
    dev = subprocess.run([*pre, "python", "-c", "import jax, os; print(jax.devices(), jax.default_backend(), os.cpu_count(), 'affinity', len(os.sched_getaffinity(0)))"], capture_output=True, text=True, env=env)
    log.write(f"=== devices {dev.stdout.strip()} {dev.stderr.strip()[-300:]}\n")
    cmd = [*pre, "python", "w2/learner_w2_rl2.py", *X, "--steps", steps, "--seed", "9", "--ckpt", f"{out}/s9.ckpt", "--save-params", f"{out}/s9.params",
           "--out", f"{out}/s9_train.json"]
    t0 = time.time(); log.write(f"=== start {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n"); log.flush()
    smi = None
    if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0:   # Cam 21:55Z: GPU utilisation every 5 s
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used", "--format=csv,noheader", "-l", "5"],
                               stdout=open(f"{out}/gpu_util.csv", "a"), stderr=subprocess.STDOUT)
    cpu_s = subprocess.Popen(["sh", "-c", f"while true; do echo $(date +%T) $(cat /proc/loadavg); sleep 10; done >> {out}/loadavg.txt"])
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    while p.poll() is None:
        time.sleep(30); log.flush(); volume.commit()
    cpu_s.kill(); smi and smi.kill()
    log.write(f"=== exit {p.returncode} {time.strftime('%H:%M:%S')} wall {time.time() - t0:.1f}s\n"); log.close(); volume.commit()
    return f"{tag}: exit {p.returncode}"


@app.function(image=img_cpu, cpu=8.0, memory=8192, timeout=45 * 60, retries=0, volumes={VOL: volume})
def run_cpu(steps: str) -> str:
    return _run("cpu8", steps)


@app.function(image=img_cpu, cpu=8.0, memory=8192, timeout=40 * 60, retries=0, volumes={VOL: volume})
def run_cpu_pin(steps: str) -> str:
    return _run("cpu8pin", steps, pin=8)


@app.function(image=img_gpu, gpu="L4", cpu=4.0, memory=8192, timeout=27 * 60, retries=0, volumes={VOL: volume})
def run_l4(steps: str) -> str:
    return _run("l4", steps)


@app.function(image=img_gpu, gpu="T4", cpu=4.0, memory=8192, timeout=27 * 60, retries=0, volumes={VOL: volume})
def run_t4(steps: str) -> str:
    return _run("t4", steps)


@app.local_entrypoint()
def cpu(steps: str = "2e6"):
    print(run_cpu.remote(steps))


@app.local_entrypoint()
def cpupin(steps: str = "1e6"):
    print(run_cpu_pin.remote(steps))


@app.local_entrypoint()
def t4(steps: str = "2e6"):
    print(run_t4.remote(steps))


@app.local_entrypoint()
def l4(steps: str = "2e6"):
    print(run_l4.remote(steps))
