"""
Trace when danger encoding emerged in Phase 6.
Runs decode_signals.py on sliding windows to see when red_dist MI
replaced own_energy MI and when TOPO_SIM became significant.
"""
import subprocess
import sys

corpus = "/kaggle/working/throng/runs_large/run_20260522_110122/signal_corpus.jsonl"
windows = [
    (100000, 110000, "100-110k"),
    (120000, 130000, "120-130k"),
    (140000, 150000, "140-150k"),
    (150001, 157201, "150-157k"),
]

print("🔬 Tracing emergence of danger-encoded communication\n")
for min_step, max_step, label in windows:
    print(f"\n{'='*60}")
    print(f"  WINDOW: {label}  (steps {min_step}-{max_step})")
    print(f"{'='*60}")
    cmd = [
        "python", "/kaggle/working/throng/tools/decode_signals.py", corpus,
        "--min-step", str(min_step), "--max-step", str(max_step)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Extract just the summary lines we care about
    for line in proc.stdout.splitlines():
        if any(k in line for k in ["TOPO_SIM", "MI_max", "red_dist", "energy", "SCOUT vs BLIND", "LAG-1"]):
            print(line)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[:500])
