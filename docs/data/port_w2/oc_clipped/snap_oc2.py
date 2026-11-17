"""Lighter replacement of snap_oc.py (that one unpickled every checkpoint and cost ~0.7 CPU): read the step count from the
log's last CURVE line; when it passes the next multiple of 1e7, copy the (atomic) checkpoint. Copy precision: one save
interval (20 iterations, 163,840 steps)."""
import os, re, shutil, time
O = os.path.dirname(os.path.abspath(__file__)); nxt = 6
done = lambda: any(w in open(os.path.join(O, "run_oc.out")).read() for w in ("OC_DONE", "CRASH"))  # noqa: E731
while nxt <= 10 and not done():
    with open(os.path.join(O, "oc.log"), "rb") as f:
        f.seek(max(0, os.path.getsize(os.path.join(O, "oc.log")) - 4000)); tail = f.read().decode(errors="ignore")
    m = re.findall(r'"steps": (\d+)', tail)
    if m and int(m[-1]) >= nxt * 1e7:
        dst = os.path.join(O, f"oc_ckpt_{nxt}e7.ckpt"); shutil.copyfile(os.path.join(O, "oc.ckpt"), dst)
        print(f"SNAP {dst} log steps {int(m[-1]):,}", flush=True); nxt += 1
    time.sleep(20)
print("SNAP_DONE", flush=True)
