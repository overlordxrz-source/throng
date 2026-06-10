import time
import json
import os
import re
from pathlib import Path
from visualization.dashboard import DashboardUpdate, _dashboard_process_main, DashboardProcess
import multiprocessing as mp

def tail_logs(log_file):
    print(f"Waiting for {log_file} to exist...")
    while not os.path.exists(log_file):
        time.sleep(1.0)

    print(f"Tailing {log_file}...")

    steps, pops, mean_fit, max_fit, mean_e, mi_hist = [], [], [], [], [], []

    q = mp.Queue()
    proc = mp.Process(target=_dashboard_process_main, args=(q, 5.0), daemon=True)
    proc.start()

    step_regex = re.compile(r"\[step\s+(\d+)\] .*\| blue=(\d+) red=")
    energy_regex = re.compile(r"Energy:\s+mean=([\d\.]+)")
    # Capture entropy so we can repurpose the fitness chart to show Entropy!
    entropy_regex = re.compile(r"Reward:\s+mean=.*\|\s+Entropy:\s+([\d\.]+)")
    red_vq_regex = re.compile(r"RedVQ:\s+loss=([\d\.]+)")

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
                mean_fit.append(0.0) # will be overwritten by entropy
                max_fit.append(0.0)  # will be overwritten by red vq loss
                mean_e.append(0.0)   # will be overwritten by energy
            
            m_en = energy_regex.search(line)
            if m_en and len(mean_e) > 0:
                mean_e[-1] = float(m_en.group(1))
            
            m_ent = entropy_regex.search(line)
            if m_ent and len(mean_fit) > 0:
                # Store Blue Entropy in the mean_fitness slot
                mean_fit[-1] = float(m_ent.group(1))

            m_rvq = red_vq_regex.search(line)
            if m_rvq and len(max_fit) > 0:
                # Store Red VQ Loss in the max_fitness slot
                max_fit[-1] = float(m_rvq.group(1))
            
            # Whenever we hit the AuxLoss or VQ line (which is near the end of a block), we push the update
            if "VQ:" in line and steps:
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
