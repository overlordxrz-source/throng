import time
import json
import glob
import os
import re
from pathlib import Path
from visualization.dashboard import DashboardUpdate, _dashboard_process_main, DashboardProcess
import threading
import multiprocessing as mp

def tail_logs(log_file):
    if not os.path.exists(log_file):
        print(f"File {log_file} does not exist!")
        return

    print(f"Tailing {log_file}...")

    steps, pops, mean_fit, max_fit, mean_e, mi_hist = [], [], [], [], [], []

    q = mp.Queue()
    proc = mp.Process(target=_dashboard_process_main, args=(q, 5.0), daemon=True)
    proc.start()

    step_regex = re.compile(r"\[step\s+(\d+)\] .*\| blue=(\d+) red=")
    energy_regex = re.compile(r"Energy:\s+mean=([\d\.]+)")

    with open(log_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(1.0)
                continue
            
            m_step = step_regex.search(line)
            if m_step:
                steps.append(int(m_step.group(1)))
                pops.append(int(m_step.group(2)))
                # we don't have fitness here, so just 0
                mean_fit.append(0.0)
                max_fit.append(0.0)
                # Next line might be Energy, but we'll append a dummy energy for now, which will be overwritten
                mean_e.append(0.0)
            
            m_en = energy_regex.search(line)
            if m_en and len(mean_e) > 0:
                mean_e[-1] = float(m_en.group(1))
            
            if m_step or m_en:
                if not steps: continue
                u = DashboardUpdate(
                    step=steps[-1], population_history=pops.copy(), step_history=steps.copy(),
                    mean_fitness_hist=mean_fit.copy(), max_fitness_hist=max_fit.copy(),
                    mean_energy_hist=mean_e.copy(), mi_history=mi_hist.copy(), top_lineages=[]
                )
                if len(steps) > 500:
                    steps.pop(0); pops.pop(0); mean_fit.pop(0); max_fit.pop(0); mean_e.pop(0)
                
                q.put(u)

if __name__ == "__main__":
    tail_logs("/mnt/throng-runs/train.log")
