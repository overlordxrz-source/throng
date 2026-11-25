"""Cam Nov 24 ~23:55Z chain (b): TWIN-K2 on Modal L4 (scripts/modal_seed/run_twin_modal.sh): the deaf twin in the K = 2
world, 3e7, then its 3000-step life eval. Code = throng-toy v3-k2 (SHA-checked; rl2 rec + --d4-side + --frontier-schedule,
bit-exact to port_w2 with the demonstrator off). Standing rules: server-side timeout = cap / $1.053 per h; at most 1 retry;
an in-container ledger counts every attempt toward one cap; a REAL run needs a registered `RUN-ID: <id> | seed 0 |
recipe twin-k2@<sha7> | cap $<C>` line.

    MODAL_PROFILE=bazingagi1 TW_APP=throng-twk2 SEED_CAP=1.5 modal run --detach scripts/modal_twin.py::main --run-id TWIN-K2
    MODAL_PROFILE=bazingagi1 TW_APP=throng-twk2smoke SEED_CAP=0.30 modal run --detach scripts/modal_twin.py::smoke
Copy back:  MODAL_PROFILE=bazingagi1 modal volume get throng-runs-b1 /twin/k2 ~/throng-backup/modal_b1/twk2/
"""
from __future__ import annotations

import os

import modal

SHA = os.environ.get("TW_SHA", "")                 # the v3-k2 commit (full SHA), passed at launch and checked in the image
REPO = "https://github.com/overlordxrz-source/throng.git"
VOL = "/mnt/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App(os.environ.get("TW_APP", "throng-twk2"))
RATE = 1.053
CAP = float(os.environ.get("SEED_CAP", "1.5")); CAP_S = int(CAP / RATE * 3600)
volume = modal.Volume.from_name("throng-runs-b1", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "zsh")
         .run_commands(f"git clone {REPO} /root/tt", f"cd /root/tt && git checkout {SHA}", f'cd /root/tt && test "$(git rev-parse HEAD)" = "{SHA}"')
         .pip_install("jax[cuda12]==0.10.0", "flax==0.12.7", "optax==0.2.8", "numpy==2.5.4"))
image = image.env({"SEED_CAP": str(CAP), "TW_SHA": SHA}).add_local_file(os.path.join(HERE, "modal_seed", "run_twin_modal.sh"), "/root/aux/run_twin_modal.sh", copy=True)


@app.function(image=image, gpu="L4", cpu=4.0, memory=8192, timeout=CAP_S, retries=modal.Retries(max_retries=1, initial_delay=10.0),
              volumes={VOL: volume})
def run_twin(env: dict | None = None, kill_after_leg1_s: int = 0) -> str:
    import json
    import signal
    import subprocess
    import time
    os.chdir("/root/tt")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == SHA
    env = dict(env or {}); tag = env.get("OUTTAG", "k2"); out = f"{VOL}/twin/{tag}"; os.makedirs(out, exist_ok=True)
    led = f"{out}/ledger.json"; volume.reload()
    used = json.load(open(led))["used_s"] if os.path.exists(led) else 0.0; remaining = CAP_S - used
    log = open(f"{out}/run.out", "a"); log.write(f"=== container start {time.strftime('%H:%M:%S')} twin cap ${CAP} = {CAP_S}s used {used:.0f}s remaining {remaining:.0f}s sha {SHA[:7]} env {env}\n"); log.flush()
    if remaining <= 0:
        log.write("=== LEDGER: cap time used up; not starting\n"); log.close(); volume.commit(); return "twin: cap time used up"
    p = subprocess.Popen(["zsh", "/root/aux/run_twin_modal.sh"], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **env), start_new_session=True)
    t0 = time.time(); killed = None; leg1_seen = None
    while p.poll() is None:
        time.sleep(10); log.flush(); used_now = used + time.time() - t0
        json.dump(dict(used_s=used_now, cap_s=CAP_S), open(led, "w")); volume.commit()
        if used_now >= CAP_S:
            os.killpg(p.pid, signal.SIGKILL); killed = "LEDGER cap time reached"; break
        if kill_after_leg1_s:                                               # smoke only: a hard kill mid leg 2
            if leg1_seen is None and "TWIN 1e7 done" in open(f"{out}/run.out").read():
                leg1_seen = time.time()
            if leg1_seen and time.time() - leg1_seen >= kill_after_leg1_s:
                os.killpg(p.pid, signal.SIGKILL); killed = f"SMOKE hard kill {kill_after_leg1_s}s into leg 2"; break
    p.wait(); json.dump(dict(used_s=used + time.time() - t0, cap_s=CAP_S), open(led, "w"))
    log.write(f"=== container exit {p.returncode} {killed or ''} {time.strftime('%H:%M:%S')}\n"); log.close(); volume.commit()
    if killed:
        return f"twin: {killed}"
    if p.returncode != 0:
        raise RuntimeError(f"twin: script exit {p.returncode} (retry resumes from the volume)")
    return "twin: done"


RECIPE = f"twin-k2@{SHA[:7]}"
PROTOCOL = os.path.join(os.path.dirname(HERE), "docs", "RESEARCH_PROTOCOL.md")
SMOKE_ENV = dict(OUTTAG="smoke_k2", LEGSTEP="163840", NLEG="2", CHUNK="60")   # 20-iteration legs (rl2 --stop-at saves its own checkpoint)


def _registered(run_id: str) -> str:
    import re
    if not run_id:
        return "no --run-id given"
    if len(SHA) != 40:
        return "TW_SHA is not a full commit SHA"
    hits = [m for m in re.finditer(r"RUN-ID: (\S+) \| seed (\d+) \| recipe (\S+) \| cap \$([0-9.]+)", open(PROTOCOL).read()) if m.group(1) == run_id]
    if len(hits) != 1:
        return f"run-id {run_id!r} registered {len(hits)} times in the protocol (need exactly 1)"
    m = hits[0]
    if int(m.group(2)) != 0 or m.group(3) != RECIPE or abs(float(m.group(4)) - CAP) > 1e-9:
        return f"run-id {run_id!r} is registered as seed {m.group(2)} / {m.group(3)} / cap ${m.group(4)}; this launch is seed 0 / {RECIPE} / cap ${CAP}"
    return ""


@app.local_entrypoint()
def main(run_id: str = ""):
    """REAL run. Refuses (before any container exists) unless --run-id matches a registered protocol line."""
    why = _registered(run_id)
    if why:
        print(f"REFUSED: {why}. Nothing launched."); return
    print(f"launching run-id {run_id}: {RECIPE}, cap ${CAP} (server-side timeout {CAP_S} s)")
    print(run_twin.remote({}))


@app.local_entrypoint()
def smoke():
    """In-container smoke: 2 tiny legs; call 1 hard-killed 30 s after leg 1; call 2 must skip leg 1, resume leg 2 from the
    checkpoint, finish, and write the life eval."""
    if len(SHA) != 40:
        print("REFUSED: TW_SHA is not a full commit SHA"); return
    print(run_twin.remote(SMOKE_ENV, kill_after_leg1_s=30))
    print(run_twin.remote(SMOKE_ENV, 0))
