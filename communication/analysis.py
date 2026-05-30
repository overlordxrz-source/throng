"""
communication/analysis.py — Mutual information tracking and signal clustering.

This is the core science output of THRONG.

Mutual information (MI) between a signal dimension and an environmental
variable tells us whether that dimension is carrying information about the
environment — i.e. whether the agent is "talking about" something real.

We use sklearn's mutual_info_regression because:
1. It handles continuous-valued signals and targets
2. It is non-parametric (k-NN based) so it doesn't assume linearity
3. It runs fast enough for our ≤500 sample windows

The analysis runs in a background thread so it never stalls the simulation.
Results are posted to a thread-safe queue read by the dashboard.

Signal clustering (k-means on signal vectors) reveals whether agents have
converged on discrete "words" (tight clusters) or are still broadcasting
noise (diffuse clusters).  Cluster stability over time = proto-language.
"""

from __future__ import annotations

import json
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


# Feature names for logging and plotting
ENV_FEATURE_NAMES = [
    "local_resource",
    "neighbor_count",
    "own_energy",
    "dist_to_nearest_red",
]


@dataclass
class MISnapshot:
    """One MI analysis result."""
    step:            int
    n_samples:       int
    # Shape (signal_dim, n_env_features) — MI between each signal dim and each feature
    mi_matrix:       np.ndarray
    # k-means cluster centres on signal vectors, shape (k, signal_dim)
    cluster_centres: np.ndarray
    # Cluster labels for each sample, shape (n_samples,)
    cluster_labels:  np.ndarray
    # Recent signal vectors for UMAP, shape (n_samples, signal_dim)
    signal_vectors:  np.ndarray
    # Context features for colouring UMAP, shape (n_samples, n_features)
    env_features:    np.ndarray


def topographic_similarity(
    signals:   np.ndarray,   # (N, signal_dim) — one signal per context
    contexts:  np.ndarray,   # (N, context_dim) — e.g. [bearing, distance] per signal
) -> float:
    """
    Topographic similarity r_s (Chaabouni et al. 2020):
    Spearman rank correlation between pairwise cosine distances in signal space
    and pairwise Euclidean distances in context space.

    r_s > 0.3 = compositional language (signal geometry mirrors referent geometry).
    r_s < 0.1 = correlated noise.

    Requires N >= 20 distinct contexts.
    """
    N = len(signals)
    if N < 10:
        return float("nan")

    # Normalise signals for cosine distance
    norms = np.linalg.norm(signals, axis=1, keepdims=True) + 1e-8
    s_norm = signals / norms

    sig_dists  = []
    ctx_dists  = []
    for i in range(N):
        for j in range(i + 1, N):
            sig_dists.append(1.0 - float(np.dot(s_norm[i], s_norm[j])))    # cosine dist
            ctx_dists.append(float(np.linalg.norm(contexts[i] - contexts[j])))

    if len(sig_dists) < 2:
        return float("nan")

    r_s, _ = spearmanr(sig_dists, ctx_dists)
    return float(r_s)


def granger_causality_lags(
    neighbor_signals: np.ndarray,  # (T, N, K*sig_dim) — OTHER agents' signals received by n
    actions:          np.ndarray,  # (T, N) int32 actions
    alive:            np.ndarray,  # (T, N) bool
    lags:             list = None, # e.g. [1, 2, 3, 5, 10]
) -> Dict[int, float]:
    """
    Cross-agent Granger causality: do NEIGHBOR signals at t-k predict
    focal agent's action at t, beyond what the agent's own past actions explain?

    Restricted:  action_n_t = f(action_n_{t-1..k})
    Full:        action_n_t = f(action_n_{t-1..k}, mean_neighbour_signal_{t-1..k})

    F-statistic peak at lag k* = system's communication latency.
    k=1 peak → within-step signal integration (fast).
    k=3-5 peak → multi-hop propagation through population.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]

    T, N, nb_sig_dim = neighbor_signals.shape
    results = {}

    for k in lags:
        if T <= k + 1:
            results[k] = float("nan")
            continue

        X_act_r = []
        X_full  = []
        y       = []

        for t in range(k, T):
            for n in range(N):
                if not alive[t, n]:
                    continue
                past_acts     = actions[t - k:t, n].astype(np.float32)
                # Mean neighbour signal over past k steps (averaged across K dims per step)
                past_nb_sigs  = neighbor_signals[t - k:t, n].mean(axis=0).astype(np.float32)

                X_act_r.append(past_acts)
                X_full.append(np.concatenate([past_acts, past_nb_sigs]))
                y.append(float(actions[t, n]))

        min_obs = nb_sig_dim + k + 5
        if len(y) < min_obs:
            results[k] = float("nan")
            continue

        X_r = np.array(X_act_r)
        X_f = np.array(X_full)
        y_a = np.array(y)

        try:
            lr_r = LinearRegression(fit_intercept=True).fit(X_r, y_a)
            lr_f = LinearRegression(fit_intercept=True).fit(X_f, y_a)

            rss_r   = float(np.sum((y_a - lr_r.predict(X_r)) ** 2))
            rss_f   = float(np.sum((y_a - lr_f.predict(X_f)) ** 2))
            n_obs   = len(y_a)
            p_extra = nb_sig_dim
            q_full  = X_f.shape[1] + 1

            if rss_f < 1e-10 or n_obs <= q_full or p_extra == 0:
                results[k] = float("nan")
                continue

            f_stat = ((rss_r - rss_f) / p_extra) / (rss_f / (n_obs - q_full))
            results[k] = float(f_stat)
        except Exception:
            results[k] = float("nan")

    return results


class CommunicationAnalyser:
    """
    Background-thread mutual information analyser.

    The simulation calls `record_sample()` each step.  Every
    `analysis_interval` steps it fires a background thread that computes MI
    and posts an MISnapshot to `result_queue`.  The dashboard reads from
    this queue without blocking the simulation.
    """

    def __init__(
        self,
        signal_dim:        int   = 16,
        analysis_interval: int   = 1000,
        window:            int   = 500,
        cluster_k:         int   = 8,
        result_queue:      Optional[queue.Queue] = None,
    ) -> None:
        self.signal_dim        = signal_dim
        self.analysis_interval = analysis_interval
        self.window            = window
        self.cluster_k         = cluster_k
        self.result_queue      = result_queue or queue.Queue()

        # Rolling buffer (deque semantics via list with fixed size)
        self._signal_buf:  List[np.ndarray] = []  # (signal_dim,) each
        self._feature_buf: List[np.ndarray] = []  # (n_features,) each

        self._lock = threading.Lock()
        self._last_analysis_step = -1

    # ------------------------------------------------------------------
    # Sample recording (called from simulation loop, must be fast)
    # ------------------------------------------------------------------

    def record_samples(
        self,
        signals:       np.ndarray,   # (n_alive, signal_dim)
        local_resource: np.ndarray,  # (n_alive,)
        neighbor_count: np.ndarray,  # (n_alive,)
        own_energy:     np.ndarray,  # (n_alive,)
        dist_to_red:    np.ndarray,  # (n_alive,)
    ) -> None:
        """
        Append a batch of per-agent samples to the rolling buffer.
        Keeps only the most recent `window` samples total.
        """
        n = len(signals)
        if n == 0:
            return

        features = np.stack([
            local_resource, neighbor_count, own_energy, dist_to_red
        ], axis=1).astype(np.float32)  # (n, 4)

        with self._lock:
            for i in range(n):
                self._signal_buf.append(signals[i])
                self._feature_buf.append(features[i])

            # Trim to window size
            if len(self._signal_buf) > self.window:
                excess = len(self._signal_buf) - self.window
                self._signal_buf  = self._signal_buf[excess:]
                self._feature_buf = self._feature_buf[excess:]

    # ------------------------------------------------------------------
    # Trigger analysis (called from simulation loop, spawns background thread)
    # ------------------------------------------------------------------

    def maybe_analyse(self, step: int) -> None:
        """
        If `analysis_interval` steps have elapsed since last analysis,
        spawn a background thread to run MI computation.
        Non-blocking — returns immediately.
        """
        if step - self._last_analysis_step < self.analysis_interval:
            return
        if len(self._signal_buf) < 20:
            return

        self._last_analysis_step = step

        # Snapshot the buffer (shallow copy is fine — we're appending, not mutating)
        with self._lock:
            signals  = np.array(self._signal_buf)    # (n, signal_dim)
            features = np.array(self._feature_buf)   # (n, n_features)

        n_samples = len(signals)
        t = threading.Thread(
            target=self._analyse_and_post,
            args=(step, signals, features, n_samples),
            daemon=True,
        )
        t.start()

    # ------------------------------------------------------------------
    # Background analysis (runs in separate thread)
    # ------------------------------------------------------------------

    def _analyse_and_post(
        self,
        step:     int,
        signals:  np.ndarray,   # (n_samples, signal_dim) float32
        features: np.ndarray,   # (n_samples, n_features)
        n_samples: int,
    ) -> None:
        """
        Compute MI between each signal dimension and each env feature.
        Signals are continuous float vectors; MI estimated via k-NN regression.
        """
        signals_2d = signals if signals.ndim == 2 else signals.reshape(-1, 1)
        signal_dim = signals_2d.shape[1]
        n_features = features.shape[1]
        mi_matrix  = np.zeros((signal_dim, n_features), dtype=np.float32)

        for feat_idx in range(n_features):
            target = features[:, feat_idx]
            if target.std() < 1e-6:
                continue
            mi = mutual_info_regression(
                signals_2d, target,
                discrete_features=False,
                random_state=42,
            )
            mi_matrix[:, feat_idx] = mi.astype(np.float32)

        snapshot = MISnapshot(
            step            = step,
            n_samples       = n_samples,
            mi_matrix       = mi_matrix,
            cluster_centres = np.zeros((1, signal_dim), dtype=np.float32),
            cluster_labels  = np.zeros(n_samples, dtype=np.int32),
            signal_vectors  = signals_2d.copy(),
            env_features    = features.copy(),
        )
        self.result_queue.put(snapshot)


class SignalCorpusWriter:
    """
    Periodically samples alive blue agents and writes one JSONL record per
    sample to disk for offline language analysis.

    Each record captures:
      step, agent_id, emitted signal, chosen action, scout flag,
      nearest-red distance + bearing, local resource, own energy,
      neighbour density.

    The file accumulates across restarts (opened in append mode) so a single
    corpus spans multiple resume sessions.
    """

    def __init__(
        self,
        path:          str,
        sample_frac:   float = 0.08,
        every_n_steps: int   = 20,
        rng:           Optional[np.random.Generator] = None,
    ) -> None:
        self.path          = path
        self.sample_frac   = sample_frac
        self.every_n_steps = every_n_steps
        self._rng          = rng or np.random.default_rng()
        self._last_step    = -(every_n_steps + 1)
        self._fh           = open(path, "a", buffering=8192)

    def maybe_record(
        self,
        step:              int,
        alive_idx:         np.ndarray,
        signals:           np.ndarray,
        actions:           np.ndarray,
        is_scout:          np.ndarray,
        nearest_red_dist:  np.ndarray,
        nearest_red_bear:  np.ndarray,
        local_resource:    np.ndarray,
        own_energy:        np.ndarray,
        neighbor_count:    np.ndarray,
        token_ids:         Optional[np.ndarray] = None,
        nb_scout_sig_lag1:  Optional[np.ndarray] = None,
        nb_scout_dist_lag1: Optional[np.ndarray] = None,
        nb_scout_token_lag1: Optional[np.ndarray] = None,
    ) -> None:
        """Write sampled records; no-op if called more often than every_n_steps.

        nb_scout_sig_lag1:  (n_alive, signal_dim) — mean signal of scouts within
            alarm_scout_range at T-1. nan rows = no scout in range.
        nb_scout_dist_lag1: (n_alive,) — mean nearest-red distance of those same
            scouts at T-1. Controls for scout spatial proximity in LRT.
        """
        if step - self._last_step < self.every_n_steps:
            return
        self._last_step = step
        n = len(alive_idx)
        if n == 0:
            return
        k   = max(1, int(n * self.sample_frac))
        sel = self._rng.choice(n, size=min(k, n), replace=False)
        lines = []
        for i in sel:
            idx = int(alive_idx[i])
            rec = {
                "step":      step,
                "agent":     idx,
                "sig":       [round(float(v), 5) for v in signals[idx]],
                "action":    int(actions[idx]),
                "scout":     bool(is_scout[i]),
                "red_dist":  round(float(nearest_red_dist[i]), 3),
                "red_bear":  round(float(nearest_red_bear[i]), 2),
                "resource":  round(float(local_resource[i]), 4),
                "energy":    round(float(own_energy[i]), 4),
                "neighbors": round(float(neighbor_count[i]), 4),
            }
            if token_ids is not None:
                rec["vq_token"] = int(token_ids[idx])
            if nb_scout_sig_lag1 is not None:
                row = nb_scout_sig_lag1[i]
                rec["nb_scout_sig_lag1"] = (
                    None if not np.isfinite(row).all()
                    else [round(float(v), 5) for v in row]
                )
            if nb_scout_dist_lag1 is not None:
                d = float(nb_scout_dist_lag1[i])
                rec["nb_scout_dist_lag1"] = None if not np.isfinite(d) else round(d, 3)
            if nb_scout_token_lag1 is not None:
                tok = int(nb_scout_token_lag1[i])
                rec["nb_scout_token_lag1"] = None if tok < 0 else tok
            lines.append(json.dumps(rec))
        self._fh.write("\n".join(lines) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
