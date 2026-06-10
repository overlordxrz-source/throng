import time
import json
import glob
from pathlib import Path
from visualization.dashboard import DashboardUpdate, _dashboard_process_main, DashboardProcess
import threading
import multiprocessing as mp

def tail_logs(run_dir):
    run_path = Path(run_dir)
    # Find the most recently created log file
    log_files = sorted(glob.glob(str(run_path / "*.jsonl")), key=os.path.getmtime)
    if not log_files:
        print("No log files found!")
        return

    latest_log = log_files[-1]
    print(f"Tailing {latest_log}...")

    steps, pops, mean_fit, max_fit, mean_e, mi_hist = [], [], [], [], [], []

    q = mp.Queue()
    proc = mp.Process(target=_dashboard_process_main, args=(q, 5.0), daemon=True)
    proc.start()

    with open(latest_log, "r") as f:
        # read existing
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("type") == "step_metrics":
                    steps.append(rec["step"])
                    pops.append(rec.get("population", 0))
                    mean_fit.append(rec.get("mean_fitness", 0.0))
                    max_fit.append(rec.get("max_fitness",  0.0))
                    mean_e.append(rec.get("mean_energy",   0.0))
                elif rec.get("type") == "mi_snapshot":
                    mi_hist.append(rec)
            except:
                pass
        
        if steps:
            u = DashboardUpdate(
                step=steps[-1], population_history=pops.copy(), step_history=steps.copy(),
                mean_fitness_hist=mean_fit.copy(), max_fitness_hist=max_fit.copy(),
                mean_energy_hist=mean_e.copy(), mi_history=mi_hist.copy(), top_lineages=[]
            )
            q.put(u)

        # tail new lines
        while True:
            line = f.readline()
            if not line:
                time.sleep(1.0)
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "step_metrics":
                    steps.append(rec["step"])
                    pops.append(rec.get("population", 0))
                    mean_fit.append(rec.get("mean_fitness", 0.0))
                    max_fit.append(rec.get("max_fitness",  0.0))
                    mean_e.append(rec.get("mean_energy",   0.0))
                elif rec.get("type") == "mi_snapshot":
                    mi_hist.append(rec)
                
                u = DashboardUpdate(
                    step=steps[-1], population_history=pops.copy(), step_history=steps.copy(),
                    mean_fitness_hist=mean_fit.copy(), max_fitness_hist=max_fit.copy(),
                    mean_energy_hist=mean_e.copy(), mi_history=mi_hist.copy(), top_lineages=[]
                )
                # Keep lists bounded
                if len(steps) > 500:
                    steps.pop(0); pops.pop(0); mean_fit.pop(0); max_fit.pop(0); mean_e.pop(0)
                
                q.put(u)
            except:
                pass

if __name__ == "__main__":
    import os
    tail_logs("/mnt/throng-runs")
