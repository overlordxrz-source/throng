"""A5 audit of --head-observed-meal (Cam Nov 16 ~07:30Z check (b)): every new label must trace to a frame where the parent's
meal is in the young's OWN obs. Reads the --a5-log npz of a flag-on run."""
import sys, json
import numpy as np
sys.path.insert(0, "/Users/overlord/CascadeProjects/throng-toy")
from w2 import observed_meal as OM  # noqa: E402
Z = np.load(sys.argv[1]); L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
x, pos, dpos, ipos = Z["x"], Z["pos"], Z["dpos"], Z["ipos"]
if not L:
    L = 64
tor = lambda o: (o + L // 2) % L - L // 2  # noqa: E731
off = tor(dpos - pos); R = 3; idx = (off[:, 0] + R) * 7 + (off[:, 1] + R)
n = len(x); r = dict(events=n)
r["event_parent_on_item_cell"] = int((dpos == ipos).all(-1).sum())
r["event_cell_adjacent_to_young"] = int((np.abs(off).max(-1) <= 1).sum())
r["event_parent_mark_in_agent_window"] = int((x[np.arange(n), 49 + idx] >= 1.0).sum())
r["event_item_in_food_channel"] = int((x[np.arange(n), idx] >= 1.0).sum())
r["event_all_four"] = int(((dpos == ipos).all(-1) & (np.abs(off).max(-1) <= 1) & (x[np.arange(n), 49 + idx] >= 1.0) & (x[np.arange(n), idx] >= 1.0)).sum())
oc = Z["obsd_close"]; r["observed_records"] = int(len(oc))
r["record_disp_equals_item_minus_start"] = int((oc == tor(Z["ipos_close"] - Z["istart_close"])).all(-1).sum())
r["record_sector_equals_named_sector"] = int((OM.sector_of(oc) == Z["sec_close"]).sum())
r["observed_head_labels"] = int(Z["n_obs_lab"].sum()); r["target_mismatches"] = int(Z["n_mis"].sum())
r["PASS"] = bool(n > 0 and r["event_all_four"] == n and r["observed_records"] > 0 and r["record_disp_equals_item_minus_start"] == r["observed_records"]
                 and r["record_sector_equals_named_sector"] == r["observed_records"] and r["target_mismatches"] == 0 and r["observed_head_labels"] > 0)
print(json.dumps(r, indent=1))
