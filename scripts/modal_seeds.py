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

VARIANT = os.environ.get("SEED_VARIANT", "")       # "" = the seed-9 recipe (a7087f2); "f3" = fallback (3) learner (offset-proj abed942); "f3k2" = F3-K2 under rule v3 (v3-k2)
K2_SHA = "52d53b6a66e273eefd7acff73b12bb2a2b603d25"                                  # throng-toy v3-k2 (v3 eval metric + rl2 rec D4 / frontier-schedule)
SHA = {"": "a7087f23e5fd5aa2dfa01ff079d984a6c843ec88", "f3": "abed942907474470edb60104584131a52516ed8f", "f3k2": K2_SHA}[VARIANT]
VARIANT_ENV = {"": {}, "f3": dict(XT_EXTRA="--offset-proj", REF_FILE="/mnt/runs/oracle/o0/f3_ref_mean.json", TWIN_FILE="/mnt/runs/oracle/o0/tw_eval_d12.json"),
               "f3k2": dict(XT_EXTRA="--offset-proj --demo-hearings 2", VERDICT="v3", REF_FILE="/mnt/runs/k2/ref_k2_mean.json", TWIN_FILE="/mnt/runs/twin/k2/tw_k2_eval_d12.json")}[VARIANT]
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
    "/root/aux/bar3v3.py": os.path.join(DOCS, "port_w2", "v3", "bar3v3.py"),
    "/root/aux/verdict3v3.py": os.path.join(DOCS, "port_w2", "v3", "verdict3v3.py"),
    "/root/aux/j1_ref_mean.json": os.path.join(J1, "j1_ref_mean.json"),
    "/root/aux/tw_eval_d12.json": os.path.join(J1, "tw_eval_d12.json"),
}
app = modal.App(os.environ.get("SEED_APP", "throng-seed"))
# Cam Nov 22 ~03:35Z standing rule: SERVER-SIDE timeout = item cap / hourly rate (L4 $0.80 + 4 cores x $0.0473 + 8 GiB x
# $0.008 = $1.053 / h); at most 1 retry, and an in-container LEDGER on the volume makes every attempt's runtime count
# toward the same cap (an attempt starts with remaining = cap time - used, and is killed at 0).
RATE = 1.053
CAP = float(os.environ.get("SEED_CAP", "5.5")); CAP_S = int(CAP / RATE * 3600)
volume = modal.Volume.from_name("throng-runs-b1", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "zsh")
         .run_commands(f"git clone {REPO} /root/tt", f"cd /root/tt && git checkout {SHA}", f'cd /root/tt && test "$(git rev-parse HEAD)" = "{SHA}"')
         .pip_install("jax[cuda12]==0.10.0", "flax==0.12.7", "optax==0.2.8", "numpy==2.5.4"))
image = image.env({"SEED_CAP": str(CAP), "SEED_VARIANT": VARIANT})                               # the SAME cap inside the container (the ledger), fixed after the smoke
for dst, src in AUX.items():
    image = image.add_local_file(src, dst, copy=True)


@app.function(image=image, gpu="L4", cpu=4.0, memory=8192, timeout=CAP_S, retries=modal.Retries(max_retries=1, initial_delay=10.0),
              volumes={VOL: volume})
def run_seed(seed: int, env: dict | None = None, kill_after_leg1_s: int = 0) -> str:
    import json
    import signal
    import subprocess
    import time
    os.chdir("/root/tt")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == SHA
    env = dict(env or {}); tag = env.get("OUTTAG", f"s{seed}"); out = f"{VOL}/seeds/{tag}"; os.makedirs(out, exist_ok=True)
    led = f"{out}/ledger.json"; volume.reload()
    used = json.load(open(led))["used_s"] if os.path.exists(led) else 0.0; remaining = CAP_S - used
    log = open(f"{out}/run.out", "a"); log.write(f"=== container start {time.strftime('%H:%M:%S')} seed {seed} cap ${CAP} = {CAP_S}s used {used:.0f}s remaining {remaining:.0f}s env {env}\n"); log.flush()
    if remaining <= 0:
        log.write("=== LEDGER: cap time used up; not starting\n"); log.close(); volume.commit(); return f"seed {seed}: cap time used up"
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used", "--format=csv,noheader", "-l", "30"],
                           stdout=open(f"{out}/gpu_util.csv", "a"), stderr=subprocess.STDOUT)
    p = subprocess.Popen(["zsh", "/root/aux/run_seed_modal.sh", str(seed)], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **env),
                         start_new_session=True)
    t0 = time.time(); killed = None; leg1_seen = None
    while p.poll() is None:
        time.sleep(10); log.flush(); used_now = used + time.time() - t0
        json.dump(dict(used_s=used_now, cap_s=CAP_S), open(led, "w")); volume.commit()
        if used_now >= CAP_S:
            os.killpg(p.pid, signal.SIGKILL); killed = "LEDGER cap time reached"; break
        if kill_after_leg1_s:                                               # smoke only: a hard kill mid leg 2
            if leg1_seen is None and f"J1L S{seed} 1e7 WATCH" in open(f"{out}/run.out").read():
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


SMOKE_ENV = dict(OUTTAG="smoke_s9", SMOKE="1", LEGSTEP="163840", NLEG="2", CHUNK="60", EXTRA_TR="--early-stop-step 1e12")


RECIPE = {"": "seed9-recipe@a7087f2", "f3": "fallback3-learner@abed942", "f3k2": f"f3k2-learner@{K2_SHA[:7]}"}[VARIANT]   # + --offset-proj and the F3 reference / twin for "f3"
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
    print(run_seed.remote(seed, VARIANT_ENV))


@app.local_entrypoint()
def smoke():
    """Cam Nov 22 ~03:35Z item 4: tiny legs (163,840 steps), 2 legs, 60 s chunks; call 1 is hard-killed 30 s into leg 2;
    call 2 must skip leg 1 and resume leg 2 from the checkpoint; then the verdict step runs on a synthetic table."""
    env = {**SMOKE_ENV, **VARIANT_ENV, **({"OUTTAG": "smoke_f3l"} if VARIANT == "f3" else {}),
           **(dict(OUTTAG="smoke_f3k2", V3_KS="1 2 2", V3_ES="3000", TWIN_FILE="/mnt/runs/oracle/o0/tw_eval_d12.json") if VARIANT == "f3k2" else {})}   # v3 smoke: the real v3 block on tiny evals; the old twin file for plumbing only
    print(run_seed.remote(9, env, kill_after_leg1_s=30))
    print(run_seed.remote(9, env, 0))


@app.local_entrypoint()
def capcheck():
    """The in-container ledger uses the SAME cap as the server-side timeout: a ledger already past the cap must refuse."""
    print(run_seed.remote(9, dict(SMOKE_ENV, OUTTAG="smoke_capcheck"), 0))
