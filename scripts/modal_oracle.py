"""Cam Nov 22 ~03:35Z item 6, FAIL branch (PREPARED; real launches need a registered --run-id): fallback (3) ORACLE on
Modal L4 -- 4-way oracle (b') + target-KL .015 + --d4-side + --offset-proj, one app per seed (throng-f3o<S>), legs of 1e7 to
5e7 (E1 gate) with tables every leg (scripts/modal_seed/run_oracle_modal.sh). Code = throng-toy branch offset-proj abed942
(SHA-checked). Same safety pattern as modal_seeds.py: server-side timeout = cap / $1.053 per h, 1 retry + an in-container
ledger counting all attempts toward one cap, the cap baked into the image env, and --run-id matching exactly one protocol
line `RUN-ID: <id> | seed <S> | recipe fallback3-oracle@abed942 | cap $<C>` or it refuses before any container.

    MODAL_PROFILE=bazingagi1 F3_APP=throng-f3o0 SEED_CAP=<cap> modal run --detach scripts/modal_oracle.py --seed 0 --run-id <id>
    MODAL_PROFILE=bazingagi1 F3_APP=throng-f3smoke SEED_CAP=0.30 modal run --detach scripts/modal_oracle.py::smoke
"""
from __future__ import annotations

import os

import modal

SHA = "abed942907474470edb60104584131a52516ed8f"   # throng-toy offset-proj (a7087f2 + --offset-proj)
REPO = "https://github.com/overlordxrz-source/throng.git"
VOL = "/mnt/runs"
HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(os.path.dirname(HERE), "docs", "data")
J1 = os.path.expanduser("~/throng-backup/scratch/j1")
AUX = {  # container path -> local source
    "/root/aux/run_oracle_modal.sh": os.path.join(HERE, "modal_seed", "run_oracle_modal.sh"),
    "/root/aux/decide.py": os.path.join(HERE, "modal_seed", "decide.py"),
    "/root/aux/bysec_o.py": os.path.join(DOCS, "port_w2", "oracle_screen", "bysec_o.py"),
}
app = modal.App(os.environ.get("F3_APP", "throng-f3o"))
# Cam Nov 22 ~03:35Z standing rule: SERVER-SIDE timeout = item cap / hourly rate (L4 $0.80 + 4 cores x $0.0473 + 8 GiB x
# $0.008 = $1.053 / h); at most 1 retry, and an in-container LEDGER on the volume makes every attempt's runtime count
# toward the same cap (an attempt starts with remaining = cap time - used, and is killed at 0).
RATE = 1.053
CAP = float(os.environ.get("SEED_CAP", "5.5")); CAP_S = int(CAP / RATE * 3600)
volume = modal.Volume.from_name("throng-runs-b1", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "zsh")
         .run_commands(f"git clone {REPO} /root/tt", f"cd /root/tt && git checkout {SHA}", f'cd /root/tt && test "$(git rev-parse HEAD)" = "{SHA}"')
         .pip_install("jax[cuda12]==0.10.0", "flax==0.12.7", "optax==0.2.8", "numpy==2.5.4"))
image = image.env({"SEED_CAP": str(CAP)})                               # the SAME cap inside the container (the ledger), fixed after the smoke
for dst, src in AUX.items():
    image = image.add_local_file(src, dst, copy=True)


@app.function(image=image, gpu="L4", cpu=4.0, memory=8192, timeout=CAP_S, retries=modal.Retries(max_retries=1, initial_delay=10.0),
              volumes={VOL: volume})
def run_oracle(seed: int, env: dict | None = None, kill_after_leg1_s: int = 0) -> str:
    import json
    import signal
    import subprocess
    import time
    os.chdir("/root/tt")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == SHA
    env = dict(env or {}); tag = env.get("OUTTAG", f"o{seed}"); out = f"{VOL}/oracle/{tag}"; os.makedirs(out, exist_ok=True)
    led = f"{out}/ledger.json"; volume.reload()
    used = json.load(open(led))["used_s"] if os.path.exists(led) else 0.0; remaining = CAP_S - used
    log = open(f"{out}/run.out", "a"); log.write(f"=== container start {time.strftime('%H:%M:%S')} seed {seed} cap ${CAP} = {CAP_S}s used {used:.0f}s remaining {remaining:.0f}s env {env}\n"); log.flush()
    if remaining <= 0:
        log.write("=== LEDGER: cap time used up; not starting\n"); log.close(); volume.commit(); return f"seed {seed}: cap time used up"
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used", "--format=csv,noheader", "-l", "30"],
                           stdout=open(f"{out}/gpu_util.csv", "a"), stderr=subprocess.STDOUT)
    p = subprocess.Popen(["zsh", "/root/aux/run_oracle_modal.sh", str(seed)], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **env),
                         start_new_session=True)
    t0 = time.time(); killed = None; leg1_seen = None
    while p.poll() is None:
        time.sleep(10); log.flush(); used_now = used + time.time() - t0
        json.dump(dict(used_s=used_now, cap_s=CAP_S), open(led, "w")); volume.commit()
        if used_now >= CAP_S:
            os.killpg(p.pid, signal.SIGKILL); killed = "LEDGER cap time reached"; break
        if kill_after_leg1_s:                                               # smoke only: a hard kill mid leg 2
            if leg1_seen is None and f"F3 TABLE s{seed} 1e7" in open(f"{out}/run.out").read():
                leg1_seen = time.time()
            if leg1_seen and time.time() - leg1_seen >= kill_after_leg1_s:
                os.killpg(p.pid, signal.SIGKILL); killed = f"SMOKE hard kill {kill_after_leg1_s}s into leg 2"; break
    p.wait(); smi.kill(); json.dump(dict(used_s=used + time.time() - t0, cap_s=CAP_S), open(led, "w"))
    log.write(f"=== container exit {p.returncode} {killed or ''} {time.strftime('%H:%M:%S')}\n"); log.close(); volume.commit()
    if killed:
        return f"seed {seed}: {killed}"
    if p.returncode != 0:
        raise RuntimeError(f"seed {seed}: script exit {p.returncode} (retry resumes from the volume)")
    return f"seed {seed}: done"


SMOKE_ENV = dict(OUTTAG="smoke_o0b", LEGSTEP="204800", NLEG="2", CHUNK="60")   # 25-iteration legs: port_w2 checkpoints every 20 iterations, so leg 1 leaves a ckpt at it 20 (the first smoke used 20-iteration legs: no ckpt, leg 2 retrained from scratch)


RECIPE = "fallback3-oracle@abed942"   # oracle (b') + 4-way + target-KL .015 + --d4-side + --offset-proj (run_oracle_modal.sh)
PROTOCOL = os.path.join(os.path.dirname(HERE), "docs", "RESEARCH_PROTOCOL.md")


def _registered(run_id: str, seed: int) -> str:
    """Cam Nov 22 ~04:20Z: a REAL run needs a protocol line `RUN-ID: <id> | seed <S> | recipe <R> | cap $<C>` registered
    BEFORE launch, matching this seed, this recipe and the cap in effect (SEED_CAP). Returns '' if OK, else the reason."""
    import re
    if not run_id:
        return "no --run-id given"
    pat = re.compile(r"RUN-ID: (\S+) \| seed (\d+) \| recipe (\S+) \| cap \$([0-9.]+)")
    hits = [m for m in pat.finditer(open(PROTOCOL).read()) if m.group(1) == run_id]
    if len(hits) != 1:
        return f"run-id {run_id!r} registered {len(hits)} times in the protocol (need exactly 1)"
    m = hits[0]
    if int(m.group(2)) != seed or m.group(3) != RECIPE or abs(float(m.group(4)) - CAP) > 1e-9:
        return f"run-id {run_id!r} is registered as seed {m.group(2)} / {m.group(3)} / cap ${m.group(4)}; this launch is seed {seed} / {RECIPE} / cap ${CAP}"
    return ""


@app.local_entrypoint()
def main(seed: int, run_id: str = ""):
    """REAL run. Refuses (before any container exists) unless --run-id matches a registered protocol line."""
    why = _registered(run_id, seed)
    if why:
        print(f"REFUSED: {why}. Nothing launched."); return
    print(f"launching run-id {run_id}: seed {seed}, {RECIPE}, cap ${CAP} (server-side timeout {CAP_S} s)")
    print(run_oracle.remote(seed))


@app.local_entrypoint()
def smoke():
    """In-container smoke (cap $0.30): 2 tiny legs, 60 s chunks; call 1 hard-killed 30 s after its leg-1 table; call 2 must
    skip leg 1 and resume leg 2 from the checkpoint; files copied back by hand after."""
    print(run_oracle.remote(0, SMOKE_ENV, kill_after_leg1_s=20))
    print(run_oracle.remote(0, SMOKE_ENV, 0))
