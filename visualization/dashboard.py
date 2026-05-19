"""
visualization/dashboard.py — Matplotlib analytics dashboard.

Runs in a separate daemon process so it never blocks the simulation.
The main process passes data via a multiprocessing Queue.

Layout (2×2 compact grid — fits on any screen):
  [0,0] Population + mean energy  (dual y-axis)
  [0,1] Fitness curves            (mean / max / mean-energy)
  [1,0] MI heatmap                (signal_dim × env_features, imshow)
  [1,1] Lineage timeline bars     (top-10 by longevity, coloured alive/dead)

For offline analysis (--analyze mode), reads JSONL logs and plots same layout.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "grid":    "#21262d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "blue":    "#58a6ff",
    "green":   "#3fb950",
    "orange":  "#f0b429",
    "red":     "#f85149",
    "purple":  "#bc8cff",
    "yellow":  "#e3b341",
}


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


def _pick_backend() -> None:
    """Try GUI backends in order; fall back to Agg silently."""
    import matplotlib
    for backend in ["macosx", "Qt5Agg", "GTK3Agg", "TkAgg"]:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.close(fig)
            return
        except Exception:
            continue
    matplotlib.use("Agg")


def _ax_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply compact dark-theme styling to a freshly cleared axis."""
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["muted"], labelsize=7, length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(C["border"])
    ax.grid(True, color=C["grid"], linewidth=0.4, alpha=0.8)
    if title:
        ax.set_title(title, color=C["text"], fontsize=8.5, fontweight="bold", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["muted"], fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["muted"], fontsize=7)


def _legend(ax, **kw) -> None:
    ax.legend(fontsize=6.5, facecolor=C["bg"], labelcolor=C["text"],
              framealpha=0.85, edgecolor=C["border"], **kw)


# ── Panel drawers ─────────────────────────────────────────────────────────────

def _draw_population(ax, u: DashboardUpdate) -> None:
    # Remove any twin axes from previous redraws before clearing
    fig = ax.get_figure()
    for other in list(fig.axes):
        if other is not ax and other.get_shared_x_axes().joined(ax, other):
            other.remove()
    ax.cla()

    steps = np.array(u.step_history)
    pop   = np.array(u.population_history)
    eng   = np.array(u.mean_energy_hist)

    ax2 = ax.twinx()
    ax2.set_facecolor(C["panel"])
    ax2.tick_params(colors=C["orange"], labelsize=7, length=3)
    for spine in ax2.spines.values():
        spine.set_edgecolor(C["border"])

    ax.fill_between(steps, pop, alpha=0.18, color=C["blue"])
    ax.plot(steps, pop, color=C["blue"], linewidth=1.4, label="population")
    ax2.plot(steps, eng, color=C["orange"], linewidth=1.1,
             linestyle="--", alpha=0.9, label="mean energy")

    _ax_style(ax, title="Population & Energy", ylabel="Agents")
    ax2.set_ylabel("Mean energy", color=C["orange"], fontsize=7)
    ax.yaxis.label.set_color(C["blue"])
    ax.tick_params(axis="y", colors=C["blue"])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, facecolor=C["bg"],
              labelcolor=C["text"], framealpha=0.85, edgecolor=C["border"],
              loc="upper left")


def _draw_fitness(ax, u: DashboardUpdate) -> None:
    ax.cla()
    _ax_style(ax, title="Fitness  (age × consumed)", ylabel="Score")
    steps = np.array(u.step_history)
    if len(steps) == 0:
        return
    mean_f = np.array(u.mean_fitness_hist)
    max_f  = np.array(u.max_fitness_hist)

    ax.fill_between(steps, mean_f, alpha=0.15, color=C["green"])
    ax.plot(steps, mean_f, color=C["green"],  linewidth=1.4, label="mean")
    ax.plot(steps, max_f,  color=C["yellow"], linewidth=1.0,
            linestyle="--", alpha=0.85, label="max")
    _legend(ax, loc="upper left")

    # Annotate latest values
    if len(mean_f):
        ax.annotate(f"{mean_f[-1]:.0f}", xy=(steps[-1], mean_f[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=C["green"], fontsize=6.5)


def _draw_mi_heatmap(ax, u: DashboardUpdate) -> None:
    """
    Render MI as a 2-D heatmap: rows = signal dims, cols = env features.
    Uses the most recent MI snapshot.
    """
    import matplotlib.pyplot as plt
    ax.cla()
    _ax_style(ax, title="Mutual Information  (signal × env)")

    feat_names = ["resource", "neighbors", "energy", "dist_red"]

    if not u.mi_history:
        ax.text(0.5, 0.5, "Waiting for first MI snapshot…\n(runs every 1 000 steps)",
                transform=ax.transAxes, ha="center", va="center",
                color=C["muted"], fontsize=8)
        return

    # Latest snapshot
    latest = u.mi_history[-1]
    mi_mat = np.array(latest.get("mi_matrix", []))   # (signal_dim, n_features)
    if mi_mat.ndim != 2 or mi_mat.shape[0] == 0:
        ax.text(0.5, 0.5, "Malformed MI data", transform=ax.transAxes,
                ha="center", va="center", color=C["muted"], fontsize=8)
        return

    im = ax.imshow(mi_mat, aspect="auto", cmap="magma",
                   vmin=0, vmax=max(mi_mat.max(), 0.01),
                   interpolation="nearest")
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, fontsize=6.5, color=C["muted"], rotation=20, ha="right")
    ax.set_ylabel("Signal dim", color=C["muted"], fontsize=7)
    ax.tick_params(axis="y", colors=C["muted"], labelsize=6.5)
    ax.tick_params(axis="x", colors=C["muted"])

    # Remove any stale colorbar axes from previous redraws before adding a new one
    try:
        fig = ax.get_figure()
        # Colorbar axes are thin axes that share the same figure but not the main grid
        for other_ax in fig.axes:
            if other_ax is not ax and getattr(other_ax, "_label", "") == "throng_mi_cb":
                other_ax.remove()
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MI (nats)")
        cb.ax._label = "throng_mi_cb"  # tag so we can find and remove it next time
        cb.ax.tick_params(labelsize=6, colors=C["muted"])
        cb.set_label("MI (nats)", color=C["muted"], fontsize=6)
    except Exception:
        pass

    # MI trend line as inset text
    if len(u.mi_history) > 1:
        trend = np.array([np.array(h.get("mi_matrix", [[0]])).mean()
                          for h in u.mi_history[-20:]])
        direction = "↑" if trend[-1] > trend[0] else "↓"
        ax.set_title(
            f"Mutual Information  (signal × env)   mean={mi_mat.mean():.3f} {direction}",
            color=C["text"], fontsize=8.5, fontweight="bold", pad=4,
        )


def _draw_lineage(ax, u: DashboardUpdate) -> None:
    ax.cla()
    _ax_style(ax, title="Lineage Timeline  (top 10)")

    recs = u.top_lineages[:10]
    if not recs:
        ax.text(0.5, 0.5, "No lineage data yet",
                transform=ax.transAxes, ha="center", va="center",
                color=C["muted"], fontsize=8)
        return

    names  = [f"L{r.get('lineage_id', i)}" for i, r in enumerate(recs)]
    starts = [r.get("birth_step",     0)    for r in recs]
    ends   = [r.get("last_seen_step", u.step) for r in recs]
    alive  = [r.get("is_alive", False)       for r in recs]

    y = np.arange(len(names))
    for i, (s, e, a) in enumerate(zip(starts, ends, alive)):
        span   = max(e - s, 1)
        colour = C["green"] if a else C["red"]
        ax.barh(y[i], span, left=s, height=0.55, color=colour, alpha=0.80)
        ax.text(s + span + u.step * 0.003, y[i], f"{span:,}",
                va="center", fontsize=5.5, color=C["muted"])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5, color=C["text"])
    ax.set_xlabel("Simulation step", color=C["muted"], fontsize=7)
    ax.tick_params(axis="x", colors=C["muted"], labelsize=7)
    ax.axvline(u.step, color=C["muted"], linewidth=0.7, linestyle=":")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["green"], label="alive"),
                        Patch(color=C["red"],   label="extinct")],
              fontsize=6, facecolor=C["bg"], labelcolor=C["text"],
              framealpha=0.8, edgecolor=C["border"], loc="lower right")


def _redraw(fig, axes, u: DashboardUpdate) -> None:
    import traceback
    try:
        _draw_population(axes[0], u)
        _draw_fitness(axes[1], u)
        _draw_mi_heatmap(axes[2], u)
        _draw_lineage(axes[3], u)
        fig.suptitle(
            f"THRONG  ·  step {u.step:,}",
            color=C["text"], fontsize=10, fontweight="bold", y=0.995,
        )
        fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.11,
                            hspace=0.42, wspace=0.38)
        fig.canvas.draw_idle()
    except Exception:
        print("[dashboard] redraw error:\n" + traceback.format_exc(), flush=True)


# ── Subprocess entry point ────────────────────────────────────────────────────

def _dashboard_process_main(data_queue: mp.Queue, update_interval: float) -> None:
    _pick_backend()

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(11, 5.2), facecolor=C["bg"])
    try:
        fig.canvas.manager.set_window_title("THRONG — Analytics")
    except Exception:
        pass

    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [
        fig.add_subplot(gs[0, 0]),  # population + energy
        fig.add_subplot(gs[0, 1]),  # fitness
        fig.add_subplot(gs[1, 0]),  # MI heatmap
        fig.add_subplot(gs[1, 1]),  # lineage
    ]
    fig.patch.set_facecolor(C["bg"])

    plt.show(block=False)

    latest: Optional[DashboardUpdate] = None
    last_redraw = 0.0

    while True:
        try:
            if not plt.fignum_exists(fig.number):
                break
        except Exception:
            break

        # Drain queue — keep only most recent
        try:
            while True:
                latest = data_queue.get_nowait()
        except Exception:
            pass

        now = time.time()
        if latest is not None and (now - last_redraw) >= update_interval:
            _redraw(fig, axes, latest)
            last_redraw = now

        try:
            fig.canvas.flush_events()
        except Exception as exc:
            print(f"[dashboard] flush_events error: {exc}", flush=True)
            break
        time.sleep(0.05)

    plt.close(fig)


# ── Public API ────────────────────────────────────────────────────────────────

class DashboardProcess:
    """Manages the dashboard as a non-blocking daemon subprocess."""

    def __init__(self, update_interval: float = 15.0) -> None:
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
    """Plot saved run data statically.  Called via `python main.py --analyze`."""
    import glob, json
    _pick_backend()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

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

    u = DashboardUpdate(
        step=steps[-1], population_history=pops, step_history=steps,
        mean_fitness_hist=mean_fit, max_fitness_hist=max_fit,
        mean_energy_hist=mean_e, mi_history=mi_hist, top_lineages=[],
    )
    fig = plt.figure(figsize=(11, 5.2), facecolor=C["bg"], constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    fig.patch.set_facecolor(C["bg"])
    _redraw(fig, axes, u)
    plt.show()
