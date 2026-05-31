# Modal Notebook — Phase 9.4 launch

**Symptom:** `can't cd to /root/throng`, `run_bg.py: No such file or directory`  
**Cause:** The repo is **not** on the GPU disk by default — only the volume (`/mnt/throng-runs`) persists. You must **clone** every new Modal machine.

Mount volume **`throng-runs`** at **`/mnt/throng-runs`** before running cells.

---

## Cell 1 — Clone repo + enable cross-attention (Python)

Run this **first**. Do not use bare `cd /root/throng` without the clone block.

```python
import os, subprocess, sys
from pathlib import Path

REPO = Path("/root/throng")
BRANCH = "feature/phase9-canvas"
REMOTE = "https://github.com/overlordxrz-source/throng.git"

if not REPO.exists():
    subprocess.run(
        ["git", "clone", "-b", BRANCH, REMOTE, str(REPO)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPO), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO), "pull", "origin", BRANCH], check=True)

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
    text=True,
).strip()
print(f"throng @ {BRANCH} git={sha}")

assert (REPO / "run_bg.py").is_file(), "run_bg.py missing — clone failed"
assert (REPO / "config_phase7.yaml").is_file()

import yaml
cfg_path = REPO / "config_phase7.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
p9 = cfg.setdefault("phase9_canvas", {})
p9["cross_attn_enabled"] = True
p9["cross_attn_num_heads"] = 4
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

Path("/mnt/throng-runs/checkpoints").mkdir(parents=True, exist_ok=True)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/throng_jax_cache"
Path("/tmp/throng_jax_cache").mkdir(parents=True, exist_ok=True)

print("cross_attn_enabled:", p9["cross_attn_enabled"])
print("run_bg exists:", REPO / "run_bg.py")
```

---

## Cell 2 — Start training (Python `Popen`, not shell `cd`)

Modal Jupyter often breaks on `!cd ... && nohup`. Use **`subprocess.Popen`**:

```python
import subprocess, sys
from pathlib import Path

REPO = Path("/root/throng")
if not (REPO / "run_bg.py").is_file():
    raise FileNotFoundError("Run Cell 1 first — /root/throng not cloned")

# Stop a prior bad launch if any
subprocess.run(["pkill", "-f", "run_bg.py"], check=False)

log_path = "/mnt/throng-runs/train.log"
log = open(log_path, "a")
proc = subprocess.Popen(
    [sys.executable, "-u", str(REPO / "run_bg.py")],
    cwd=str(REPO),
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"Igniting Phase 9.4. PID={proc.pid} log={log_path}")
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
