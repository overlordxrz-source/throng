"""
visualization/dashboard.py — Web-based Analytics Dashboard.

Runs a lightweight HTTP server in a separate daemon process.
Serves a modern web UI built with HTML/JS/CSS (Chart.js) from the `web` folder.
The main process passes data via a multiprocessing Queue.

For offline analysis (--analyze mode), reads JSONL logs and serves the static 
run data through the same beautiful web interface.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import json
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

@dataclass
class DashboardUpdate:
    """Snapshot of simulation state pushed to the dashboard subprocess."""
    step:               int
    population_history: List[int]
    step_history:       List[int]
    mean_fitness_hist:  List[float]
    max_fitness_hist:   List[float]
    mean_energy_hist:   List[float]
    mi_history:         List[Dict]
    top_lineages:       List[Dict]
    signal_vectors:     Optional[np.ndarray] = None
    cluster_labels:     Optional[np.ndarray] = None
    env_features:       Optional[np.ndarray] = None


class NumpyEncoder(json.JSONEncoder):
    """Encodes NumPy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Global holder for the latest update received via queue
_latest_update = None

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve static web files from visualization/web
        web_dir = Path(__file__).parent / "web"
        super().__init__(*args, directory=str(web_dir), **kwargs)

    def do_GET(self):
        if self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            # Prevent caching so frontend gets real-time data
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            
            if _latest_update is None:
                self.wfile.write(b'{}')
            else:
                data_dict = asdict(_latest_update)
                # Remove massive arrays to keep JSON payload lightweight
                data_dict.pop('signal_vectors', None)
                data_dict.pop('cluster_labels', None)
                data_dict.pop('env_features', None)
                
                payload = json.dumps(data_dict, cls=NumpyEncoder).encode('utf-8')
                self.wfile.write(payload)
        else:
            super().do_GET()

    # Mute standard HTTP server logging to avoid terminal spam
    def log_message(self, format, *args):
        pass


# ── Subprocess entry point ────────────────────────────────────────────────────

def _dashboard_process_main(data_queue: mp.Queue, update_interval: float) -> None:
    global _latest_update
    
    server = HTTPServer(('0.0.0.0', 8050), DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    print("\n[Dashboard] 🌐 Web UI Live: http://localhost:8050\n", flush=True)

    while True:
        try:
            # Drain queue — keep only most recent
            while True:
                _latest_update = data_queue.get_nowait()
        except Exception:
            pass
        
        time.sleep(1.0)


# ── Public API ────────────────────────────────────────────────────────────────

class DashboardProcess:
    """Manages the dashboard as a non-blocking daemon subprocess."""

    def __init__(self, update_interval: float = 5.0) -> None:
        self._queue            = mp.Queue(maxsize=4)
        self._process: Optional[mp.Process] = None
        self._update_interval  = update_interval

    def start(self) -> None:
        self._process = mp.Process(
            target=_dashboard_process_main,
            args=(self._queue, self._update_interval),
            daemon=True,
        )
        self._process.start()

    def push(self, update: DashboardUpdate) -> None:
        try:
            self._queue.put_nowait(update)
        except Exception:
            pass

    def stop(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)


# ── Offline analysis mode ─────────────────────────────────────────────────────

def run_offline_analysis(run_dir: str) -> None:
    """Serve saved run data statically via the web UI. Called via `python main.py --analyze`."""
    import glob
    global _latest_update
    
    run_path  = Path(run_dir)
    log_files = sorted(glob.glob(str(run_path / "*.jsonl")))
    steps, pops, mean_fit, max_fit, mean_e, mi_hist = [], [], [], [], [], []

    for lf in log_files:
        with open(lf) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "step_metrics":
                    steps.append(rec["step"])
                    pops.append(rec.get("population", 0))
                    mean_fit.append(rec.get("mean_fitness", 0.0))
                    max_fit.append(rec.get("max_fitness",  0.0))
                    mean_e.append(rec.get("mean_energy",   0.0))
                elif rec.get("type") == "mi_snapshot":
                    mi_hist.append(rec)

    if not steps:
        print(f"No step_metrics found in {run_dir}")
        return

    _latest_update = DashboardUpdate(
        step=steps[-1], population_history=pops, step_history=steps,
        mean_fitness_hist=mean_fit, max_fitness_hist=max_fit,
        mean_energy_hist=mean_e, mi_history=mi_hist, top_lineages=[],
    )
    
    print(f"\n[Dashboard] 🌐 Offline Analysis Ready: http://localhost:8050")
    print(f"              Serving {len(steps)} steps from {run_dir}")
    print("Press Ctrl+C to stop the server.\n")
    
    server = HTTPServer(('0.0.0.0', 8050), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down offline server.")
        server.server_close()
