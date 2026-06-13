# Modal Notebook — Phase 9.4 launch

**Symptom:** `can't cd to /root/throng`, `run_bg.py: No such file or directory`  
**Cause:** The repo is **not** on the GPU disk by default — only the volume (`/mnt/throng-runs`) persists. You must **clone** every new Modal machine.

Mount volume **`throng-runs`** at **`/mnt/throng-runs`** before running cells.

---

## Cell 1 — Universal Launch (Clone, Pull, and Popen)

Run this cell to clone the repo (if missing), pull the latest branch, configure the environment, and safely launch `run_bg.py` in an isolated process group.

```python
import subprocess, os, time

# ── CONFIG — only edit this block when switching phases or accounts ──────────
BRANCH   = "feature/phase17-5-timescale-alarm"
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
REPO_DIR = "/root/throng"
LOG_PATH = "/mnt/throng-runs/train.log"
# ─────────────────────────────────────────────────────────────────────────────

# 1. Kill any existing run
subprocess.run("pkill -9 -f run_bg.py", shell=True)
time.sleep(2)

# 2. Clone if missing, pull if exists
if not os.path.exists(REPO_DIR):
    subprocess.run(f"git clone {REPO_URL} {REPO_DIR}", shell=True, check=True)
subprocess.run(
    f"cd {REPO_DIR} && git fetch origin && "
    f"git checkout {BRANCH} && git pull origin {BRANCH}",
    shell=True, check=True
)

# 3. Verify SHA (catch stale cache issues)
sha = subprocess.run("git -C /root/throng log --oneline -1",
                     shell=True, capture_output=True, text=True)
print("SHA:", sha.stdout.strip())

# 4. Configure env
env = os.environ.copy()
env["TF_GPU_ALLOCATOR"]               = "cuda_malloc_async"
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"
env["JAX_COMPILATION_CACHE_DIR"]      = "/tmp/throng_jax_cache"

# 5. Launch — NEVER remove start_new_session=True
proc = subprocess.Popen(
    ["python", "-u", "run_bg.py"],
    cwd=REPO_DIR,
    env=env,
    stdout=open(LOG_PATH, "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)
time.sleep(5)

# 6. Verify exactly one training process
r = subprocess.run("pgrep -a -f run_bg.py", shell=True, capture_output=True, text=True)
print("Processes:\n", r.stdout.strip())
print(f"PID: {proc.pid}")
assert r.stdout.count("python -u run_bg.py") == 1, "⚠️ Multiple run_bg.py processes!"
```

---

## Cell 3 — Tail log

```python
!tail -n 80 -f /mnt/throng-runs/train.log
```

`Ctrl+C` stops **tail only**, not training.

---

## What you should see in the log

```text
[JAX] Checkpoint on volume: latest PPO update = 390
[JAX] Orbax strict match failed (schema evolution). Merging new heads manually...
[JAX] Injected randomly initialized nb_cross_attn into b_params
[JAX] Phase9.4 cross-attn receiver: heads=4 ...
[JAX] Restored params from step 390.
```

Resume from **`393`** only if you intend the post-imagination weights. Prefer **`390`**.

---

## Do NOT use (broken on fresh Modal disk)

```bash
cd /root/throng   # fails if never cloned
sed -i ... config_phase7.yaml   # fails if repo missing
python /root/throng/run_bg.py   # fails if repo missing
```
