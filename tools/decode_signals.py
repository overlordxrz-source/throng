"""
tools/decode_signals.py — Offline language analysis of the signal corpus.

Usage:
    python tools/decode_signals.py runs/run_XXXXXXXX_XXXXXX/signal_corpus.jsonl
    python tools/decode_signals.py --red
    python tools/decode_signals.py --red /path/to/signal_corpus_red.jsonl --min-step 1000

What this script does:
  1. Loads the corpus and prints basic stats.
  2. K-means clusters signal vectors → proto-vocabulary.
  3. For each cluster: shows mean context features, dominant action, scout %.
  4. Per-dimension Spearman correlations with every context feature.
  5. Mutual information between each signal dim and each context feature.
  6. For each signal dim: labels it by its strongest MI feature.

Interpretation guide:
  - A dim strongly correlated with red_dist → encodes danger proximity.
  - A dim correlated with red_bear → encodes danger direction.
  - A dim correlated with resource → encodes food availability.
  - Clusters with scout=True majority emitting distinct signals → alarm calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("sklearn required: pip install scikit-learn")

ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY"}
CONTEXT_KEYS = ["red_dist", "red_bear", "resource", "energy", "neighbors"]
RED_CONTEXT_KEYS = ["blue_dist", "blue_bear", "resource", "energy", "neighbors"]
RED_CORPUS_DEFAULT = "/mnt/throng-runs/signal_corpus_red.jsonl"
# Phase 12.2 pincer test: emitter blue_dist when token was transmitted
RED_CHASE_BLUE_DIST_MAX = 2.0
RED_SEARCH_BLUE_DIST_MIN = 5.0


# ── Loading ────────────────────────────────────────────────────────────────────

def load_corpus(path: str) -> dict:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        sys.exit(f"No valid records in {path}")

    signals  = np.array([r["sig"]    for r in records], dtype=np.float32)
    actions  = np.array([r["action"] for r in records], dtype=np.int32)
    scouts   = np.array([r["scout"]  for r in records], dtype=bool)
    steps    = np.array([r["step"]   for r in records], dtype=np.int64)

    ctx = {}
    for k in CONTEXT_KEYS:
        raw = np.array([r.get(k, float("nan")) for r in records], dtype=np.float32)
        ctx[k] = raw

    # Lag-1 scout signal field (may be absent in older corpus files)
    sig_dim = signals.shape[1]
    nb_lag1_raw = [r.get("nb_scout_sig_lag1", None) for r in records]
    has_lag1 = any(v is not None for v in nb_lag1_raw)
    if has_lag1:
        nb_lag1 = np.array(
            [v if v is not None else [float("nan")] * sig_dim for v in nb_lag1_raw],
            dtype=np.float32,
        )
    else:
        nb_lag1 = None

    # Lag-1 scout nearest-red distance (control variable for LRT)
    nb_dist_lag1_raw = [r.get("nb_scout_dist_lag1", None) for r in records]
    has_dist_lag1 = any(v is not None for v in nb_dist_lag1_raw)
    if has_dist_lag1:
        nb_dist_lag1 = np.array(
            [v if v is not None else float("nan") for v in nb_dist_lag1_raw],
            dtype=np.float32,
        )
    else:
        nb_dist_lag1 = None

    vq_tokens = np.array(
        [r.get("vq_token", -1) for r in records], dtype=np.int32
    )
    nb_tok_lag1_raw = [r.get("nb_scout_token_lag1", None) for r in records]
    has_tok_lag1 = any(v is not None for v in nb_tok_lag1_raw)
    if has_tok_lag1:
        nb_tok_lag1 = np.array(
            [v if v is not None else -1 for v in nb_tok_lag1_raw],
            dtype=np.int32,
        )
    else:
        nb_tok_lag1 = None

    return {
        "signals":      signals,
        "actions":      actions,
        "scouts":       scouts,
        "steps":        steps,
        "ctx":          ctx,
        "nb_lag1":      nb_lag1,
        "nb_dist_lag1": nb_dist_lag1,
        "vq_tokens":    vq_tokens,
        "nb_tok_lag1":  nb_tok_lag1,
        "n":            len(records),
    }


def load_red_corpus(path: str) -> dict:
    """Load Phase 12.1 red corpus (hunter / blue_dist schema)."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        sys.exit(f"No valid records in {path}")

    signals = np.array([r["sig"] for r in records], dtype=np.float32)
    actions = np.array([r["action"] for r in records], dtype=np.int32)
    hunters = np.array([r["hunter"] for r in records], dtype=bool)
    steps = np.array([r["step"] for r in records], dtype=np.int64)

    ctx = {}
    for k in RED_CONTEXT_KEYS:
        raw = np.array([r.get(k, float("nan")) for r in records], dtype=np.float32)
        ctx[k] = raw

    sig_dim = signals.shape[1]
    nb_lag1_raw = [r.get("nb_hunter_sig_lag1", None) for r in records]
    has_lag1 = any(v is not None for v in nb_lag1_raw)
    if has_lag1:
        nb_lag1 = np.array(
            [v if v is not None else [float("nan")] * sig_dim for v in nb_lag1_raw],
            dtype=np.float32,
        )
    else:
        nb_lag1 = None

    nb_dist_lag1_raw = [r.get("nb_hunter_dist_lag1", None) for r in records]
    has_dist_lag1 = any(v is not None for v in nb_dist_lag1_raw)
    if has_dist_lag1:
        nb_dist_lag1 = np.array(
            [v if v is not None else float("nan") for v in nb_dist_lag1_raw],
            dtype=np.float32,
        )
    else:
        nb_dist_lag1 = None

    vq_tokens = np.array(
        [r.get("vq_token", -1) for r in records], dtype=np.int32
    )
    nb_tok_lag1_raw = [r.get("nb_hunter_token_lag1", None) for r in records]
    has_tok_lag1 = any(v is not None for v in nb_tok_lag1_raw)
    if has_tok_lag1:
        nb_tok_lag1 = np.array(
            [v if v is not None else -1 for v in nb_tok_lag1_raw],
            dtype=np.int32,
        )
    else:
        nb_tok_lag1 = None

    return {
        "signals": signals,
        "actions": actions,
        "hunters": hunters,
        "steps": steps,
        "ctx": ctx,
        "nb_lag1": nb_lag1,
        "nb_dist_lag1": nb_dist_lag1,
        "vq_tokens": vq_tokens,
        "nb_tok_lag1": nb_tok_lag1,
        "n": len(records),
    }


def _apply_step_filter(
    data: dict,
    min_step: int,
    max_step: int,
    *,
    emitter_key: str = "scouts",
) -> None:
    """In-place step filter for blue (scouts) or red (hunters) corpora."""
    keep = np.ones(len(data["steps"]), dtype=bool)
    if min_step > 0:
        keep &= data["steps"] >= min_step
    if max_step > 0:
        keep &= data["steps"] <= max_step
    if keep.all():
        return
    for key in ("signals", "actions", emitter_key, "steps"):
        data[key] = data[key][keep]
    data["ctx"] = {k: v[keep] for k, v in data["ctx"].items()}
    if data.get("nb_lag1") is not None:
        data["nb_lag1"] = data["nb_lag1"][keep]
    if data.get("nb_dist_lag1") is not None:
        data["nb_dist_lag1"] = data["nb_dist_lag1"][keep]
    data["vq_tokens"] = data["vq_tokens"][keep]
    if data.get("nb_tok_lag1") is not None:
        data["nb_tok_lag1"] = data["nb_tok_lag1"][keep]
    data["n"] = int(keep.sum())


# ── Per-dimension correlation table ───────────────────────────────────────────

def dim_correlations(
    signals: np.ndarray,
    ctx: dict,
    feat_keys: Optional[list[str]] = None,
) -> np.ndarray:
    """Returns (signal_dim, n_features) Spearman r matrix."""
    sig_dim = signals.shape[1]
    feat_keys = feat_keys or CONTEXT_KEYS
    mat = np.zeros((sig_dim, len(feat_keys)), dtype=np.float32)
    for fi, k in enumerate(feat_keys):
        y = ctx[k]
        valid = np.isfinite(y)
        if valid.sum() < 10:
            continue
        for di in range(sig_dim):
            r, _ = spearmanr(signals[valid, di], y[valid])
            mat[di, fi] = float(r) if np.isfinite(r) else 0.0
    return mat


# ── Mutual information table ───────────────────────────────────────────────────

def dim_mi(
    signals: np.ndarray,
    ctx: dict,
    feat_keys: Optional[list[str]] = None,
) -> np.ndarray:
    """Returns (signal_dim, n_features) MI matrix."""
    sig_dim   = signals.shape[1]
    feat_keys = feat_keys or CONTEXT_KEYS
    mat = np.zeros((sig_dim, len(feat_keys)), dtype=np.float32)
    for fi, k in enumerate(feat_keys):
        y = ctx[k]
        valid = np.isfinite(y)
        if valid.sum() < 20 or y[valid].std() < 1e-6:
            continue
        mi = mutual_info_regression(
            signals[valid], y[valid],
            discrete_features=False,
            random_state=42,
        )
        mat[:, fi] = mi.astype(np.float32)
    return mat


# ── Cluster analysis ───────────────────────────────────────────────────────────

def cluster_analysis(
    signals: np.ndarray,
    actions: np.ndarray,
    scouts:  np.ndarray,
    ctx:     dict,
    k:       int = 8,
    feat_keys: Optional[list[str]] = None,
    emitter_pct_label: str = "Scout",
) -> None:
    feat_keys = feat_keys or CONTEXT_KEYS
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(signals)

    print(f"\n{'─'*70}")
    print(f"  CLUSTER VOCABULARY  (k={k})")
    print(f"{'─'*70}")
    header = (f"{'Clust':>6}  {'N':>5}  {emitter_pct_label + '%':>7}  "
              f"{'DomAct':>7}  "
              + "  ".join(f"{k[:8]:>8}" for k in feat_keys))
    print(header)
    print("─" * len(header))

    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        n        = int(mask.sum())
        scout_pct = 100 * scouts[mask].mean()
        act_counts = np.bincount(actions[mask], minlength=5)
        dom_act  = ACTION_NAMES[int(act_counts.argmax())]
        ctx_vals = []
        for ck in feat_keys:
            v = ctx[ck][mask]
            v = v[np.isfinite(v)]
            ctx_vals.append(f"{v.mean():>8.3f}" if len(v) > 0 else f"{'nan':>8}")
        print(f"  {c:>4d}  {n:>5d}  {scout_pct:>6.1f}%  "
              f"{dom_act:>7}  " + "  ".join(ctx_vals))


def red_vq_token_pincer_test(
    actions: np.ndarray,
    hunters: np.ndarray,
    ctx: dict,
    vq_tokens: np.ndarray,
    nb_tok_lag1: Optional[np.ndarray],
    nb_dist_lag1: Optional[np.ndarray],
    min_token_n: int = 25,
    min_elig: int = 50,
    chase_max: float = RED_CHASE_BLUE_DIST_MAX,
    search_min: float = RED_SEARCH_BLUE_DIST_MIN,
) -> None:
    """
    Phase 12.2 — Red VQ pincer test (χ²).

    Chase tokens: emitted when emitter blue_dist <= chase_max (close pursuit).
    Search tokens: emitted when emitter blue_dist > search_min (wide search).

    Among receiver reds pursuing (N/S/E/W) with a lag-1 hunter token, does
    pursuit direction differ between Chase vs Search codebook subsets?
    """
    from scipy.stats import chi2_contingency

    A_STAY = 4
    dir_names = {0: "N", 1: "S", 2: "E", 3: "W"}
    dist_key = "blue_dist"

    print(f"\n{'─'*70}")
    print(f"  RED VQ PINCER TEST  (lag-1 hunter token → receiver pursuit direction)")
    print(f"{'─'*70}")

    if nb_tok_lag1 is None:
        print("  nb_hunter_token_lag1 absent — accumulate red corpus after 12.1 restart.")
        return

    if dist_key not in ctx:
        print(f"  {dist_key} missing from red corpus.")
        return

    valid_tok = vq_tokens >= 0
    token_stats: dict[int, tuple[float, int]] = {}
    for t in range(int(vq_tokens.max()) + 1 if valid_tok.any() else 0):
        mask = valid_tok & (vq_tokens == t)
        if int(mask.sum()) < 5:
            continue
        token_stats[t] = (float(ctx[dist_key][mask].mean()), int(mask.sum()))

    if not token_stats:
        print("  No vq_token field in red corpus — accumulate data after training fix.")
        return

    print(f"\n  Per-token emitter context  (mean blue_dist when token is transmitted):")
    print(f"  {'tok':>4}  {'N':>6}  {'blue_dist':>10}")
    print(f"  {'-'*26}")
    for t in sorted(token_stats, key=lambda x: token_stats[x][0]):
        mu, n = token_stats[t]
        print(f"  {t:>4}  {n:>6}  {mu:>10.2f}")

    chase_set = {
        t for t, (mu, n) in token_stats.items()
        if mu <= chase_max and n >= min_token_n
    }
    search_set = {
        t for t, (mu, n) in token_stats.items()
        if mu >= search_min and n >= min_token_n
    }

    # Closest vs farthest frequent tokens (pairwise sanity check)
    t_chase = int(min(token_stats, key=lambda x: token_stats[x][0]))
    t_search = int(max(token_stats, key=lambda x: token_stats[x][0]))

    receivers = ~hunters
    pursuing = receivers & (actions != A_STAY)
    has_lag = nb_tok_lag1 >= 0
    if nb_dist_lag1 is not None:
        has_lag &= np.isfinite(nb_dist_lag1)
    eligible = pursuing & has_lag

    n_elig = int(eligible.sum())
    print(f"\n  Receiver reds pursuing with lag-1 hunter token: {n_elig:,}")
    print(f"  Chase tokens (mean blue_dist ≤ {chase_max}): {sorted(chase_set) or '—'}")
    print(f"  Search tokens (mean blue_dist > {search_min}): {sorted(search_set) or '—'}")

    if n_elig < min_elig:
        print(f"  Too few eligible records (< {min_elig}) — accumulate more red corpus.")
        return

    lag_chase = np.isin(nb_tok_lag1, list(chase_set))
    lag_search = np.isin(nb_tok_lag1, list(search_set))

    if t_chase in token_stats and t_search in token_stats:
        for label, tok in [
            (f"CHASE (low blue_dist, tok mean≤{chase_max})", t_chase),
            (f"SEARCH (high blue_dist, tok mean≥{search_min})", t_search),
        ]:
            sub = eligible & (nb_tok_lag1 == tok)
            ns = int(sub.sum())
            if ns < 10:
                print(f"\n  Token {tok} ({label}): n={ns} — too few receivers")
                continue
            print(f"\n  Receivers with lag-1 hunter token {tok} ({label}, "
                  f"emitter blue_dist≈{token_stats[tok][0]:.1f})  n={ns}")
            for a, nm in dir_names.items():
                cnt = int((actions[sub] == a).sum())
                print(f"    {nm}: {cnt:5d}  ({100 * cnt / max(ns, 1):5.1f}%)")

        sub_c = eligible & (nb_tok_lag1 == t_chase)
        sub_s = eligible & (nb_tok_lag1 == t_search)
        if int(sub_c.sum()) >= 10 and int(sub_s.sum()) >= 10:
            cats = sorted(dir_names.keys())
            table = np.zeros((2, len(cats)), dtype=np.int64)
            for j, a in enumerate(cats):
                table[0, j] = int((actions[sub_c] == a).sum())
                table[1, j] = int((actions[sub_s] == a).sum())
            chi2, p, _, _ = chi2_contingency(table)
            print(f"\n  χ²(token {t_chase} vs {t_search} pursuit directions) = "
                  f"{chi2:.2f}  p={p:.4f}")

    if chase_set and search_set:
        sub_c = eligible & lag_chase
        sub_s = eligible & lag_search
        print(f"\n  Pincer sets: chase {sorted(chase_set)}  search {sorted(search_set)}")
        print(f"  Receivers lag-1 chase-token: {int(sub_c.sum()):,}  "
              f"search-token: {int(sub_s.sum()):,}")
        if int(sub_c.sum()) >= 10 and int(sub_s.sum()) >= 10:
            cats = sorted(dir_names.keys())
            table = np.zeros((2, len(cats)), dtype=np.int64)
            for j, a in enumerate(cats):
                table[0, j] = int((actions[sub_c] == a).sum())
                table[1, j] = int((actions[sub_s] == a).sum())
            chi2, p, _, _ = chi2_contingency(table)
            print(f"  χ²(chase-set vs search-set pursuit directions) = {chi2:.2f}  p={p:.4f}")
            if p < 0.05:
                print("  → Significant: chase vs search tokens associate with "
                      "different pursuit mix (pincer / coordination hypothesis).")
            else:
                print("  → Not significant at p<0.05 (may need more red corpus).")
    else:
        print("\n  Could not form chase/search token sets at min_token_n — "
              "accumulate more emissions per token.")


# ── Lag-1 partial regression (Cam's communication test) ───────────────────────

def lag1_regression(
    signals:      np.ndarray,
    actions:      np.ndarray,
    scouts:       np.ndarray,
    ctx:          dict,
    nb_lag1:      np.ndarray,
    nb_dist_lag1: Optional[np.ndarray],
    sig_dim:      int,
    *,
    red:          bool = False,
    dist_ctrl_key: Optional[str] = None,
) -> None:
    """
    Partial regression test for genuine cross-agent communication.

    Among blind agents that had a scout within alarm range at T-1:

      Control: flee ~ scout_red_dist_lag1
          (controls for scout spatial proximity to danger at T-1)
      Full:    flee ~ scout_red_dist_lag1 + scout_sig_lag1 (all dims)

    If the LRT is significant after this control, the scout's SIGNAL
    is predicting blind agent flee beyond spatial proximity alone.
    That is communication, not confound.

    Note: controlling for blind agent's own red_dist is wrong —
    blind agents have red_dist > 8 by definition (constant), so it
    adds no information. The real confound is the SCOUT's proximity.
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chi2

    title = (
        "LAG-1 HUNTER → RECEIVER PURSUIT  (red partial regression)"
        if red else
        "LAG-1 COMMUNICATION TEST  (Cam's partial regression)"
    )
    listener = "Non-hunter" if red else "Blind"
    emitter = "hunter" if red else "scout"
    outcome = "pursuit" if red else "flee"

    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")

    # Filter: listeners with valid lag-1 emitter signal
    blind    = ~scouts
    has_sig  = blind & np.isfinite(nb_lag1).all(axis=1)

    # Also require emitter dist lag-1 if available
    if nb_dist_lag1 is not None:
        has_dist = np.isfinite(nb_dist_lag1)
        eligible = has_sig & has_dist
        ctrl_label = (
            "hunter_blue_dist_lag1" if red else "scout_red_dist_lag1"
        )
    else:
        eligible   = has_sig
        fallback_key = dist_ctrl_key or ("blue_dist" if red else "red_dist")
        ctrl_label = f"{listener.lower()}_{fallback_key} (fallback)"

    n_eligible = int(eligible.sum())
    print(f"  {listener} agents with lag-1 {emitter} signal + dist: "
          f"{n_eligible:,} / {blind.sum():,}")
    print(f"  Control variable: {ctrl_label}")

    if n_eligible < 50:
        print("  Too few eligible records (< 50) — accumulate more corpus data.")
        return

    y    = (actions[eligible] != 4).astype(np.float32)  # 1 = move, 0 = stay
    if len(np.unique(y)) < 2:
        print(f"  Outcome is all-{outcome} among eligible listeners — skip regression.")
        return

    lag1 = nb_lag1[eligible]

    if nb_dist_lag1 is not None:
        ctrl = nb_dist_lag1[eligible]
    else:
        fk = dist_ctrl_key or ("blue_dist" if red else "red_dist")
        ctrl = ctx[fk][eligible]

    def zscore(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / (std + 1e-8)

    ctrl_z = zscore(ctrl).reshape(-1, 1)
    lag1_z = np.column_stack([zscore(lag1[:, d]) for d in range(sig_dim)])

    # Control model: flee ~ scout_red_dist_lag1
    lr_ctrl = LogisticRegression(max_iter=500, C=1.0).fit(ctrl_z, y)
    ll_ctrl = float(np.sum(
        y * np.log(lr_ctrl.predict_proba(ctrl_z)[:, 1] + 1e-12) +
        (1 - y) * np.log(lr_ctrl.predict_proba(ctrl_z)[:, 0] + 1e-12)
    ))

    # Full model: flee ~ scout_red_dist_lag1 + scout_sig_lag1
    X_full  = np.hstack([ctrl_z, lag1_z])
    lr_full = LogisticRegression(max_iter=500, C=1.0).fit(X_full, y)
    ll_full = float(np.sum(
        y * np.log(lr_full.predict_proba(X_full)[:, 1] + 1e-12) +
        (1 - y) * np.log(lr_full.predict_proba(X_full)[:, 0] + 1e-12)
    ))

    lrt_stat = 2 * (ll_full - ll_ctrl)
    lrt_df   = sig_dim
    lrt_p    = float(1 - chi2.cdf(lrt_stat, df=lrt_df))

    print(f"\n  Omnibus likelihood-ratio test  (all dims vs control):")
    print(f"    LRT χ²({lrt_df}) = {lrt_stat:.3f}   p = {lrt_p:.4f}")
    verdict = (
        f"SIGNIFICANT ← {emitter} signal predicts {outcome} beyond proximity"
        if lrt_p < 0.05 else
        "not significant — no communication above spatial confound"
    )
    print(f"    {verdict}")

    print(f"\n  Per-dim lag-1 β in full model (control={ctrl_label}):")
    coefs = lr_full.coef_[0]
    for d in range(sig_dim):
        beta = float(coefs[1 + d])
        print(f"    dim{d}: β = {beta:>+.4f}")

    # ── Per-dim LRT χ²(1): flee ~ ctrl + dim_d alone ──────────────────────
    print(f"\n  Per-dim LRT  χ²(1)  (one dim at a time — avoids collinearity dilution):")
    print(f"  {'dim':>4}  {'β':>8}  {'χ²(1)':>8}  {'p':>8}  result")
    print(f"  {'-'*50}")
    for d in range(sig_dim):
        X_one = np.hstack([ctrl_z, lag1_z[:, d:d+1]])
        lr_one = LogisticRegression(max_iter=500, C=1.0).fit(X_one, y)
        ll_one = float(np.sum(
            y * np.log(lr_one.predict_proba(X_one)[:, 1] + 1e-12) +
            (1 - y) * np.log(lr_one.predict_proba(X_one)[:, 0] + 1e-12)
        ))
        chi1 = 2 * (ll_one - ll_ctrl)
        p1   = float(1 - chi2.cdf(chi1, df=1))
        beta1 = float(lr_one.coef_[0][1])
        sig  = "*** SIGNIFICANT" if p1 < 0.05 else ("~ marginal" if p1 < 0.15 else "")
        print(f"  dim{d}  {beta1:>+8.4f}  {chi1:>8.3f}  {p1:>8.4f}  {sig}")

    print(f"\n  Baseline {outcome} rate: {y.mean():.3f}")
    print(f"  β > 0 = higher {emitter} signal → more {outcome}  |  β < 0 = suppresses")
    print(f"  Per-dim p < 0.05 = that specific dim carries communication signal")


# ── Topographic similarity (offline, with bounds + emitted vs transmitted) ─────

def _sample_pairs(sigs: np.ndarray, meanings: np.ndarray,
                  n_pairs: int, rng: np.random.Generator):
    """Return (sig_dist, meaning_dist) arrays for random pairs."""
    n = len(sigs)
    pi = rng.integers(0, n, size=n_pairs * 2)
    pj = rng.integers(0, n, size=n_pairs * 2)
    keep = pi != pj
    pi, pj = pi[keep][:n_pairs], pj[keep][:n_pairs]
    return (np.linalg.norm(sigs[pi] - sigs[pj], axis=1),
            np.linalg.norm(meanings[pi] - meanings[pj], axis=1))


def topographic_similarity(
    signals:  np.ndarray,
    actions:  np.ndarray,
    scouts:   np.ndarray,
    ctx:      dict,
    nb_lag1:  np.ndarray,
    sig_dim:  int,
    n_pairs:  int = 5000,
    n_null:   int = 200,
    seed:     int = 42,
    *,
    bear_key: str = "red_bear",
    dist_key: str = "red_dist",
    emitter_label: str = "scout",
    listener_label: str = "blind",
) -> None:
    """
    TOPO_SIM with three components (Cam Q10/Q11):

    1. Transmitted TOPO_SIM: nb_scout_sig_lag1 → blind flee direction.
       Answers: does the communication channel work?

    2. Emitted TOPO_SIM: scout raw signal → scout red_bear (bearing).
       Answers: does the production-side vocabulary encode direction?

    3. Bounds analysis:
       - Null r_s: shuffle directions → expected r_s under no structure.
       - Ceiling r_s: perfect 2D circular code for 4 directions → max
         achievable r_s given this 4-class discrete meaning space.

    Angular meaning space (2D circular): mirrors actual bearing distances.
    N/S and E/W are 180° apart; N/E, N/W, S/E, S/W are 90° apart.
    This gives 3 distinct distance levels vs only 2 in 4D one-hot.
    """
    from scipy.stats import spearmanr as sp_r

    A_STAY = 4
    # 2D circular direction embedding — mirrors angular distances
    dir_circ = {
        0: np.array([0.,  1.]),   # N = 0°
        1: np.array([0., -1.]),   # S = 180°
        2: np.array([1.,  0.]),   # E = 90°
        3: np.array([-1., 0.]),   # W = 270°
    }

    rng = np.random.default_rng(seed)

    print(f"\n{'─'*70}")
    print(f"  TOPOGRAPHIC SIMILARITY  (transmitted + emitted, with bounds)")
    print(f"{'─'*70}")

    # ── 1. TRANSMITTED: nb_scout_sig_lag1 → blind flee direction ──────────
    blind    = ~scouts
    fleeing  = blind & (actions != A_STAY)
    has_sig  = np.isfinite(nb_lag1).all(axis=1)
    elig_t   = fleeing & has_sig
    n_elig_t = int(elig_t.sum())
    print(f"\n  [TRANSMITTED]  {listener_label} fleeing agents with lag-1: {n_elig_t:,}")

    if n_elig_t >= 100:
        t_sigs = nb_lag1[elig_t]
        t_dirs = actions[elig_t]
        t_dirs_valid = np.array([d for d in t_dirs if d in dir_circ])
        t_sigs_valid = t_sigs[[d in dir_circ for d in t_dirs]]
        t_meaning = np.array([dir_circ[int(d)] for d in t_dirs_valid], dtype=np.float32)

        # Subsample for speed
        cap = min(len(t_sigs_valid), 3000)
        if len(t_sigs_valid) > cap:
            idx = rng.choice(len(t_sigs_valid), size=cap, replace=False)
            t_sigs_valid = t_sigs_valid[idx]
            t_meaning    = t_meaning[idx]

        sd_t, md_t = _sample_pairs(t_sigs_valid, t_meaning, n_pairs, rng)
        r_t, p_t   = sp_r(sd_t, md_t)
        print(f"  Transmitted TOPO_SIM  r_s = {r_t:.4f}   p = {p_t:.2e}")

        # ── Null distribution via shuffled directions ──────────────────
        null_rs = []
        for _ in range(n_null):
            shuf = t_meaning[rng.permutation(len(t_meaning))]
            sd_n, md_n = _sample_pairs(t_sigs_valid, shuf, n_pairs, rng)
            r_n, _ = sp_r(sd_n, md_n)
            null_rs.append(r_n)
        null_rs = np.array(null_rs)
        print(f"  Null r_s (shuffled, n={n_null}):  "
              f"mean={null_rs.mean():.4f}  std={null_rs.std():.4f}  "
              f"95th={np.percentile(null_rs, 95):.4f}")

        # ── Ceiling: perfect circular code → same data, oracle signals ──
        # Build oracle: signal = direction 2D coords + orthogonal padding
        oracle_sig = np.hstack([t_meaning,
                                np.zeros((len(t_meaning), max(0, sig_dim - 2)),
                                         dtype=np.float32)])
        sd_c, md_c = _sample_pairs(oracle_sig, t_meaning, n_pairs, rng)
        r_ceil, _  = sp_r(sd_c, md_c)
        print(f"  Ceiling r_s (perfect circular code): {r_ceil:.4f}")

        pct_ceil = 100 * r_t / r_ceil if r_ceil > 0 else float("nan")
        pct_null = (r_t - null_rs.mean()) / (null_rs.std() + 1e-8)
        print(f"  Observed = {pct_ceil:.1f}% of ceiling  |  "
              f"{pct_null:.1f}σ above null")
        if p_t < 0.05 and pct_ceil > 30:
            print(f"  → MEANINGFUL topographic structure (>{30:.0f}% of ceiling)")
        elif p_t < 0.05:
            print(f"  → Weak but real structure (significant, <30% of ceiling)")
        else:
            print(f"  → Not significant")
    else:
        r_t = float("nan")
        print("  Too few records.")

    # ── 2. EMITTED: emitter raw signal → threat/prey bearing ─────────────
    print(f"\n  [EMITTED]  {emitter_label} raw signal → {bear_key} (production vocabulary)")
    scout_mask = (
        scouts
        & np.isfinite(ctx[bear_key])
        & np.isfinite(ctx[dist_key])
    )
    scout_mask &= ctx[dist_key] < 8.0
    n_elig_e = int(scout_mask.sum())
    print(f"  {emitter_label.capitalize()} records with target visible (dist<8): {n_elig_e:,}")

    if n_elig_e >= 100:
        e_sigs  = signals[scout_mask]
        e_bear  = ctx[bear_key][scout_mask]   # degrees 0-360

        # Angular distance: min(|a-b|, 360-|a-b|) → normalize to [0, 1]
        cap = min(n_elig_e, 3000)
        if n_elig_e > cap:
            idx = rng.choice(n_elig_e, size=cap, replace=False)
            e_sigs = e_sigs[idx]
            e_bear = e_bear[idx]

        pi = rng.integers(0, len(e_sigs), size=n_pairs * 2)
        pj = rng.integers(0, len(e_sigs), size=n_pairs * 2)
        keep = pi != pj
        pi, pj = pi[keep][:n_pairs], pj[keep][:n_pairs]
        sd_e  = np.linalg.norm(e_sigs[pi] - e_sigs[pj], axis=1)
        db    = np.abs(e_bear[pi] - e_bear[pj])
        md_e  = np.minimum(db, 360 - db) / 180.0   # [0, 1]

        r_e, p_e = sp_r(sd_e, md_e)
        print(f"  Emitted TOPO_SIM      r_s = {r_e:.4f}   p = {p_e:.2e}")

        if not np.isnan(r_t):
            gap = r_e - r_t
            if gap > 0.02:
                print(f"  Gap (emitted − transmitted) = +{gap:.4f}  "
                      f"→ information lost in transmission (signal averaging?)")
            elif gap < -0.02:
                print(f"  Gap (emitted − transmitted) = {gap:.4f}  "
                      f"→ channel amplifies or aligns signal (unexpected)")
            else:
                print(f"  Gap (emitted − transmitted) = {gap:+.4f}  "
                      f"→ channel is faithful")
    else:
        r_e = float("nan")
        print("  Too few scout records with visible red.")


# ── Categorical vocabulary tests (Cam Q12 follow-up) ──────────────────────────

def _cardinal_bin(bearing_deg: np.ndarray) -> np.ndarray:
    """Map 0-360° bearing to cardinal index: 0=N 1=E 2=S 3=W."""
    b = bearing_deg % 360
    return np.where(b < 45, 0,
           np.where(b < 135, 1,
           np.where(b < 225, 2,
           np.where(b < 315, 3, 0)))).astype(np.int32)


def _cardinal_dist(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Ordinal cardinal distance: 0=same 1=adjacent 2=opposite."""
    diff = np.abs(c1.astype(np.int32) - c2.astype(np.int32)) % 4
    return np.minimum(diff, 4 - diff)   # 0, 1, or 2


def categorical_vocabulary_tests(
    signals: np.ndarray,
    scouts:  np.ndarray,
    ctx:     dict,
    sig_dim: int,
    n_pairs: int = 5000,
    seed:    int = 42,
    *,
    bear_key: str = "red_bear",
    dist_key: str = "red_dist",
    emitter_label: str = "scout",
    visible_dist_max: float = 8.0,
) -> None:
    """
    Three tests for discrete 4-symbol vocabulary (Cam round 7):

    1. Categorical emitted TOPO_SIM — scout signal vs cardinal bin distance
       (0=same, 1=adjacent, 2=opposite). Should jump vs continuous r_s=0.018.

    2. K-means k=4 on scout signals near reds — cluster membership vs
       cardinal direction of threat. Chi-squared contingency.

    3. Within-bin vs across-bin signal variance — lower within-bin variance
       = categorical encoding confirmed structurally.
    """
    from scipy.stats import spearmanr as sp_r, chi2_contingency

    CARD_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}

    print(f"\n{'─'*70}")
    print(f"  CATEGORICAL VOCABULARY TESTS  (4-symbol lexicon)")
    print(f"{'─'*70}")

    # Filter: emitters with visible prey/threat
    scout_vis = (scouts
                 & np.isfinite(ctx[bear_key])
                 & np.isfinite(ctx[dist_key])
                 & (ctx[dist_key] < visible_dist_max))
    n_sv = int(scout_vis.sum())
    print(f"\n  {emitter_label.capitalize()}s with visible target "
          f"({dist_key}<{visible_dist_max}): {n_sv:,}")
    if n_sv < 100:
        print("  Too few — cannot run tests.")
        return

    sv_sigs  = signals[scout_vis]
    sv_bear  = ctx[bear_key][scout_vis]
    sv_card  = _cardinal_bin(sv_bear)

    rng = np.random.default_rng(seed)
    cap = min(n_sv, 3000)
    if n_sv > cap:
        idx = rng.choice(n_sv, size=cap, replace=False)
        sv_sigs = sv_sigs[idx]
        sv_card = sv_card[idx]

    # ── Test 1: Categorical emitted TOPO_SIM ──────────────────────────────
    print(f"\n  TEST 1 — Categorical emitted TOPO_SIM  (ordinal cardinal distance)")
    pi = rng.integers(0, len(sv_sigs), size=n_pairs * 2)
    pj = rng.integers(0, len(sv_sigs), size=n_pairs * 2)
    keep = pi != pj
    pi, pj = pi[keep][:n_pairs], pj[keep][:n_pairs]
    sd   = np.linalg.norm(sv_sigs[pi] - sv_sigs[pj], axis=1)
    cd   = _cardinal_dist(sv_card[pi], sv_card[pj])
    r_s, p_v = sp_r(sd, cd)
    print(f"  r_s = {r_s:.4f}   p = {p_v:.2e}")
    if p_v < 0.05 and r_s > 0.05:
        print(f"  → Categorical encoding confirmed at production  (+{r_s-0.018:.3f} vs continuous r_s=0.018)")
    elif p_v < 0.05:
        print(f"  → Significant but weak")
    else:
        print(f"  → Not significant")

    # ── Test 2: K-means k=4 → cluster vs cardinal direction ──────────────
    print(f"\n  TEST 2 — K-means k=4 on {emitter_label} signals, "
          f"cluster vs cardinal direction")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(sv_sigs)
    km = KMeans(n_clusters=4, n_init=20, random_state=seed).fit(X)
    labels = km.labels_

    # Contingency table: cluster (4) × cardinal (4)
    contingency = np.zeros((4, 4), dtype=np.int32)
    for cl, cd_val in zip(labels, sv_card):
        contingency[cl, cd_val] += 1

    chi2, p_chi, dof, _ = chi2_contingency(contingency)
    print(f"  Contingency table (cluster × cardinal N/E/S/W):")
    print(f"  {'':>8}  " + "  ".join(f"{CARD_NAMES[c]:>6}" for c in range(4)))
    for cl in range(4):
        dom = int(np.argmax(contingency[cl]))
        print(f"  cluster{cl}  " +
              "  ".join(f"{contingency[cl, c]:>6}" for c in range(4)) +
              f"  → dom={CARD_NAMES[dom]}")
    print(f"  χ²({dof}) = {chi2:.1f}   p = {p_chi:.2e}")
    if p_chi < 0.05:
        # Measure alignment: fraction of each cluster matching dominant direction
        n_match = sum(contingency[cl, np.argmax(contingency[cl])]
                      for cl in range(4))
        pct = 100 * n_match / len(labels)
        print(f"  {pct:.1f}% of {emitter_label}s in cluster matching dominant cardinal")
        if p_chi < 1e-6:
            print(f"  → 4-symbol lexicon CONFIRMED (clusters align with cardinal threat direction)")
        else:
            print(f"  → Clusters partially align with cardinal directions")
    else:
        print(f"  → Not significant — no cluster-direction alignment")

    # ── Test 3: Within-bin vs across-bin signal variance ─────────────────
    print(f"\n  TEST 3 — Within-bin vs across-bin signal variance")
    total_var = sv_sigs.var(axis=0).mean()
    within_vars = []
    print(f"  {'bin':>4}  {'n':>5}  {'within_var':>12}  {'vs_total':>10}")
    for c in range(4):
        mask = sv_card == c
        n_c  = int(mask.sum())
        if n_c >= 5:
            wv = sv_sigs[mask].var(axis=0).mean()
            within_vars.append(wv)
            ratio = wv / total_var
            print(f"  {CARD_NAMES[c]:>4}  {n_c:>5}  {wv:>12.5f}  {ratio:>9.2%}")
    if within_vars:
        mean_wv = np.mean(within_vars)
        reduction = 1 - mean_wv / total_var
        print(f"  Total var = {total_var:.5f}   Mean within-bin var = {mean_wv:.5f}")
        print(f"  Variance reduction when conditioned on cardinal: {reduction:.1%}")
        if reduction > 0.10:
            print(f"  → Categorical encoding STRUCTURALLY confirmed ({reduction:.0%} reduction)")
        elif reduction > 0.02:
            print(f"  → Modest signal concentration within cardinal bins")
        else:
            print(f"  → No structural categorical encoding")


# ── Lag-1 direction LRT (Cam's multinomial test) ──────────────────────────────

def lag1_direction_lrt(
    actions:      np.ndarray,
    scouts:       np.ndarray,
    nb_lag1:      np.ndarray,
    nb_dist_lag1: Optional[np.ndarray],
    sig_dim:      int,
    *,
    red:          bool = False,
) -> None:
    """
    Multinomial direction LRT among blind agents that are fleeing.

    Target: flee direction (N/S/E/W) among blind agents with action != STAY.
    Control: scout_red_dist_lag1  (scout spatial proximity confound)
    Test:    ctrl + dim_d alone per dim

    A scout signal encoding 'red is East' should predict Westward flight
    specifically. This has variance even at 80%+ flee rate.

    df per dim = n_classes - 1 (adding one predictor to multinomial logit).
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chi2

    A_STAY = 4
    dir_names = {0: "N", 1: "S", 2: "E", 3: "W"}

    motion = "pursuit" if red else "flee"
    listener = "Non-hunter" if red else "Blind"
    title = (
        f"LAG-1 DIRECTION LRT  (multinomial {motion} direction, red)"
        if red else
        "LAG-1 DIRECTION LRT  (multinomial flee direction)"
    )

    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")

    blind   = ~scouts
    fleeing = blind & (actions != A_STAY)
    has_sig = np.isfinite(nb_lag1).all(axis=1)

    if nb_dist_lag1 is not None:
        has_dist = np.isfinite(nb_dist_lag1)
        eligible = fleeing & has_sig & has_dist
        ctrl_label = (
            "hunter_blue_dist_lag1" if red else "scout_red_dist_lag1"
        )
    else:
        eligible   = fleeing & has_sig
        ctrl_label = f"{listener.lower()}_dist (fallback)"

    n_elig = int(eligible.sum())
    n_flee = int(fleeing.sum())
    n_blind = int(blind.sum())

    print(f"  {listener} agents total:   {n_blind:,}")
    print(f"  {listener} agents {motion}: {n_flee:,}  "
          f"({100*n_flee/max(n_blind,1):.1f}%  ← {motion} baseline)")
    print(f"  Eligible for LRT:     {n_elig:,}  (have lag-1 fields)")
    print(f"  Control variable:     {ctrl_label}")

    if n_elig < 50:
        print("  Too few eligible records — accumulate more corpus data.")
        return

    y    = actions[eligible]      # direction labels: 0=N,1=S,2=E,3=W
    lag1 = nb_lag1[eligible]
    ctrl = (nb_dist_lag1[eligible] if nb_dist_lag1 is not None
            else np.zeros(n_elig, dtype=np.float32))

    def zscore(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / (std + 1e-8)

    ctrl_z = zscore(ctrl).reshape(-1, 1)
    lag1_z = np.column_stack([zscore(lag1[:, d]) for d in range(sig_dim)])

    n_classes = len(np.unique(y))
    df_per_dim = n_classes - 1   # extra params when adding 1 predictor to multinomial

    print(f"\n  {motion.capitalize()} direction distribution:")
    for a, nm in dir_names.items():
        cnt = int((y == a).sum())
        print(f"    {nm}: {cnt:5d}  ({100*cnt/max(n_elig,1):5.1f}%)")

    # Control model
    lr_ctrl = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr_ctrl.fit(ctrl_z, y)
    ll_ctrl = float(np.sum(
        np.log(lr_ctrl.predict_proba(ctrl_z)[np.arange(n_elig), y] + 1e-12)
    ))

    print(f"\n  Per-dim direction LRT  χ²({df_per_dim})  (one dim at a time):")
    print(f"  {'dim':>4}  {'χ²':>8}  {'p':>8}  result")
    print(f"  {'-'*44}")
    for d in range(sig_dim):
        X_one = np.hstack([ctrl_z, lag1_z[:, d:d+1]])
        lr_one = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        lr_one.fit(X_one, y)
        ll_one = float(np.sum(
            np.log(lr_one.predict_proba(X_one)[np.arange(n_elig), y] + 1e-12)
        ))
        chi_val = max(0.0, 2 * (ll_one - ll_ctrl))
        p_val   = float(1 - chi2.cdf(chi_val, df=df_per_dim))
        sig     = "*** SIGNIFICANT" if p_val < 0.05 else ("~ marginal" if p_val < 0.15 else "")

        # Show per-direction weight for this dim (coef sign tells direction)
        coef_row = lr_one.coef_[:, 1]   # coef for lag1_dim (column 1)
        coef_str = "  ".join(
            f"{dir_names[c]}:{coef_row[i]:+.3f}"
            for i, c in enumerate(lr_ctrl.classes_)
        )
        print(f"  dim{d}  {chi_val:>8.3f}  {p_val:>8.4f}  {sig}")
        print(f"        coefs: {coef_str}")

    verb = "pursuits" if red else "flights"
    print(f"\n  Coef interpretation: positive coef for direction X = higher signal → more X {verb}")
    print(f"  Significant dim + coef pattern matching bearing = signal encodes direction")


# ── Phase 12.2 red decode pipeline ─────────────────────────────────────────────

def decode_red_schema(
    data: dict,
    *,
    k: int = 8,
    min_step: int = 0,
    max_step: int = 0,
) -> None:
    """Full offline decode for signal_corpus_red.jsonl (Phase 12.2)."""
    _apply_step_filter(data, min_step, max_step, emitter_key="hunters")

    n = data["n"]
    signals = data["signals"]
    actions = data["actions"]
    hunters = data["hunters"]
    ctx = data["ctx"]
    sig_dim = signals.shape[1]
    feat_keys = RED_CONTEXT_KEYS

    print(f"\n{'='*70}")
    print(f"  RED CORPUS SUMMARY  (Phase 12.2)")
    print(f"{'='*70}")
    print(f"  Records  : {n:,}")
    print(f"  Steps    : {data['steps'].min():,} – {data['steps'].max():,}")
    print(f"  Signal dim: {sig_dim}")
    print(f"  Hunters  : {hunters.sum():,}  ({100 * hunters.mean():.1f}%)")
    print(
        "  Actions  : "
        + "  ".join(
            f"{ACTION_NAMES[a]}={int((actions == a).sum())}" for a in range(5)
        )
    )

    print(f"\n{'─'*70}")
    print(f"  SIGNAL RANGE PER DIM")
    print(f"{'─'*70}")
    for di in range(sig_dim):
        v = signals[:, di]
        print(
            f"  dim{di}: mean={v.mean():+.4f}  std={v.std():.4f}  "
            f"min={v.min():+.4f}  max={v.max():+.4f}"
        )

    cor_mat = dim_correlations(signals, ctx, feat_keys=feat_keys)
    print(f"\n{'─'*70}")
    print(f"  SPEARMAN r(signal_dim, context_feature)")
    print(f"{'─'*70}")
    hdr = f"  {'dim':>5}  " + "  ".join(f"{k[:9]:>9}" for k in feat_keys)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for di in range(sig_dim):
        vals = "  ".join(
            f"{cor_mat[di, fi]:>+9.3f}" for fi in range(len(feat_keys))
        )
        print(f"  dim{di:>2}  {vals}")

    mi_mat = dim_mi(signals, ctx, feat_keys=feat_keys)
    print(f"\n{'─'*70}")
    print(f"  MUTUAL INFORMATION I(signal_dim ; context_feature)")
    print(f"{'─'*70}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    best_label = {}
    for di in range(sig_dim):
        vals = "  ".join(
            f"{mi_mat[di, fi]:>9.4f}" for fi in range(len(feat_keys))
        )
        best_fi = int(mi_mat[di].argmax())
        best_label[di] = (
            feat_keys[best_fi] if mi_mat[di, best_fi] > 0.005 else "noise"
        )
        print(f"  dim{di:>2}  {vals}  ← {best_label[di]}")

    print(f"\n{'─'*70}")
    print(f"  DIMENSION ROLES (by peak MI)")
    print(f"{'─'*70}")
    for di in range(sig_dim):
        print(f"  dim{di} → {best_label[di]}")

    if hunters.any() and (~hunters).any():
        print(f"\n{'─'*70}")
        print(f"  HUNTER vs RECEIVER SIGNAL DIFFERENCE")
        print(f"{'─'*70}")
        for di in range(sig_dim):
            mu_h = float(signals[hunters, di].mean())
            mu_r = float(signals[~hunters, di].mean())
            print(
                f"  dim{di}: hunter={mu_h:+.4f}  receiver={mu_r:+.4f}  "
                f"Δ={mu_h - mu_r:+.4f}"
            )

    cluster_analysis(
        signals,
        actions,
        hunters,
        ctx,
        k=k,
        feat_keys=feat_keys,
        emitter_pct_label="Hunter",
    )

    nb_lag1 = data.get("nb_lag1")
    nb_dist_lag1 = data.get("nb_dist_lag1")
    if nb_lag1 is not None:
        lag1_regression(
            signals,
            actions,
            hunters,
            ctx,
            nb_lag1,
            nb_dist_lag1,
            sig_dim,
            red=True,
            dist_ctrl_key="blue_dist",
        )
        topographic_similarity(
            signals,
            actions,
            hunters,
            ctx,
            nb_lag1,
            sig_dim,
            bear_key="blue_bear",
            dist_key="blue_dist",
            emitter_label="hunter",
            listener_label="receiver",
        )
        categorical_vocabulary_tests(
            signals,
            hunters,
            ctx,
            sig_dim,
            bear_key="blue_bear",
            dist_key="blue_dist",
            emitter_label="hunter",
        )
        lag1_direction_lrt(
            actions, hunters, nb_lag1, nb_dist_lag1, sig_dim, red=True,
        )
        red_vq_token_pincer_test(
            actions,
            hunters,
            ctx,
            data["vq_tokens"],
            data.get("nb_tok_lag1"),
            nb_dist_lag1,
        )
    else:
        print(f"\n{'─'*70}")
        print(f"  LAG-1 RED COMMUNICATION TEST")
        print(f"{'─'*70}")
        print("  nb_hunter_sig_lag1 absent — restart with red_corpus_enabled.")

    print(f"\n{'='*70}")
    print("  Done (red). Interpret:")
    print("  • dim with high MI(blue_dist) → encodes prey proximity")
    print("  • dim with high MI(blue_bear) → encodes prey bearing")
    print("  • hunter cluster with distinct signal → possible hunt call")
    print("  • lag-1 β significant after hunter_blue_dist_lag1 → coordination")
    print("  • Red VQ pincer χ²: chase tokens → different pursuit mix")
    print(f"{'='*70}\n")


# ── VQ token vocabulary: flee direction vs lag-1 scout token ───────────────────

def vq_token_direction_test(
    actions:      np.ndarray,
    scouts:       np.ndarray,
    ctx:          dict,
    vq_tokens:    np.ndarray,
    nb_tok_lag1:  Optional[np.ndarray],
    nb_dist_lag1: Optional[np.ndarray],
    min_token_n:  int = 25,
    min_elig:     int = 50,
) -> None:
    """
    Codebook-level alarm test: do high-alert VQ tokens (low emitter red_dist)
  predict different flee directions among blind listeners than safe tokens?

    Uses lag-1 mode scout token among agents within alarm_scout_range at T-1.
    """
    from scipy.stats import chi2_contingency

    A_STAY = 4
    dir_names = {0: "N", 1: "S", 2: "E", 3: "W"}

    print(f"\n{'─'*70}")
    print(f"  VQ TOKEN DIRECTION TEST  (lag-1 scout token → blind flee direction)")
    print(f"{'─'*70}")

    if nb_tok_lag1 is None:
        print("  nb_scout_token_lag1 absent — re-run corpus after vq_token logging.")
        return

    valid_tok = vq_tokens >= 0
    token_stats: dict[int, tuple[float, int]] = {}
    for t in range(int(vq_tokens.max()) + 1 if valid_tok.any() else 0):
        mask = valid_tok & (vq_tokens == t)
        if int(mask.sum()) < 5:
            continue
        token_stats[t] = (float(ctx["red_dist"][mask].mean()), int(mask.sum()))

    if not token_stats:
        print("  No vq_token field in corpus — accumulate data after training fix.")
        return

    print(f"\n  Per-token emitter context  (mean red_dist when token is transmitted):")
    print(f"  {'tok':>4}  {'N':>6}  {'red_dist':>9}")
    print(f"  {'-'*24}")
    for t in sorted(token_stats, key=lambda x: token_stats[x][0]):
        mu, n = token_stats[t]
        print(f"  {t:>4}  {n:>6}  {mu:>9.2f}")

    dists = np.array([token_stats[t][0] for t in token_stats])
    t_lo = int(min(token_stats, key=lambda x: token_stats[x][0]))
    t_hi = int(max(token_stats, key=lambda x: token_stats[x][0]))
    alert_cut = float(np.percentile(dists, 25))
    safe_cut = float(np.percentile(dists, 75))
    alert_set = {t for t, (mu, n) in token_stats.items() if mu <= alert_cut and n >= min_token_n}
    safe_set = {t for t, (mu, n) in token_stats.items() if mu >= safe_cut and n >= min_token_n}

    blind = ~scouts
    fleeing = blind & (actions != A_STAY)
    has_lag = nb_tok_lag1 >= 0
    if nb_dist_lag1 is not None:
        has_lag &= np.isfinite(nb_dist_lag1)
    eligible = fleeing & has_lag

    n_elig = int(eligible.sum())
    print(f"\n  Blind fleeing with lag-1 scout token: {n_elig:,}")
    if n_elig < min_elig:
        print(f"  Too few eligible records (< {min_elig}) — accumulate more corpus.")
        return

    y = actions[eligible]
    lag_tok = nb_tok_lag1[eligible]
    # Full-length masks (same shape as eligible) for bitwise & with eligible
    lag_alert = np.isin(nb_tok_lag1, list(alert_set))
    lag_safe = np.isin(nb_tok_lag1, list(safe_set))

    # Pairwise: lowest vs highest mean-red_dist frequent tokens
    if t_lo in token_stats and t_hi in token_stats:
        for label, tok in [("ALERT (low red_dist)", t_lo), ("SAFE (high red_dist)", t_hi)]:
            sub = eligible & (nb_tok_lag1 == tok)
            ns = int(sub.sum())
            if ns < 10:
                print(f"\n  Token {tok} ({label}): n={ns} — too few listeners")
                continue
            print(f"\n  Listeners with lag-1 scout token {tok} ({label}, "
                  f"emitter red_dist≈{token_stats[tok][0]:.1f})  n={ns}")
            for a, nm in dir_names.items():
                cnt = int((actions[sub] == a).sum())
                print(f"    {nm}: {cnt:5d}  ({100*cnt/max(ns,1):5.1f}%)")

        sub_lo = eligible & (nb_tok_lag1 == t_lo)
        sub_hi = eligible & (nb_tok_lag1 == t_hi)
        if int(sub_lo.sum()) >= 10 and int(sub_hi.sum()) >= 10:
            cats = sorted(dir_names.keys())
            table = np.zeros((2, len(cats)), dtype=np.int64)
            for j, a in enumerate(cats):
                table[0, j] = int((actions[sub_lo] == a).sum())
                table[1, j] = int((actions[sub_hi] == a).sum())
            chi2, p, _, _ = chi2_contingency(table)
            print(f"\n  χ²(token {t_lo} vs {t_hi} flee directions) = {chi2:.2f}  p={p:.4f}")

    if alert_set and safe_set:
        sub_a = eligible & lag_alert
        sub_s = eligible & lag_safe
        print(f"\n  Quartile sets: alert tokens {sorted(alert_set)}  safe tokens {sorted(safe_set)}")
        print(f"  Listeners lag-1 alert-token: {int(sub_a.sum()):,}  safe-token: {int(sub_s.sum()):,}")
        if int(sub_a.sum()) >= 10 and int(sub_s.sum()) >= 10:
            cats = sorted(dir_names.keys())
            table = np.zeros((2, len(cats)), dtype=np.int64)
            for j, a in enumerate(cats):
                table[0, j] = int((actions[sub_a] == a).sum())
                table[1, j] = int((actions[sub_s] == a).sum())
            chi2, p, _, _ = chi2_contingency(table)
            print(f"  χ²(alert-set vs safe-set flee directions) = {chi2:.2f}  p={p:.4f}")
            if p < 0.05:
                print("  → Significant: codebook alert tokens associate with different flee mix.")
            else:
                print("  → Not significant at p<0.05 (may need more corpus).")


# ── Withdrawal comparison ──────────────────────────────────────────────────────

def _quick_withdrawal_metrics(corpus_path: str, seed: int = 42) -> dict:
    """
    Extract the two primary withdrawal metrics from a corpus file:
      1. k-means contingency χ² (production-side vocabulary structure)
      2. Direction LRT χ²(3) per dim (channel causation)
    Returns a dict; safe to call on partial corpora.
    """
    from scipy.stats import chi2_contingency
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data   = load_corpus(corpus_path)
    sigs   = data["signals"]
    acts   = data["actions"]
    scouts = data["scouts"]
    ctx    = data["ctx"]
    nb_lag1      = data.get("nb_lag1")
    nb_dist_lag1 = data.get("nb_dist_lag1")
    sig_dim      = sigs.shape[1]
    rng          = np.random.default_rng(seed)
    result: dict = {"n": data["n"], "sig_dim": sig_dim}

    # ── 1. K-means contingency χ² (production vocabulary) ────────────────
    scout_vis = (
        scouts
        & np.isfinite(ctx["red_bear"])
        & np.isfinite(ctx["red_dist"])
        & (ctx["red_dist"] < 8.0)
    )
    n_sv = int(scout_vis.sum())
    result["n_scouts_vis"] = n_sv
    if n_sv >= 100:
        sv_sigs = sigs[scout_vis]
        sv_bear = ctx["red_bear"][scout_vis]
        sv_card = _cardinal_bin(sv_bear)
        cap = min(n_sv, 3000)
        if n_sv > cap:
            idx = rng.choice(n_sv, size=cap, replace=False)
            sv_sigs = sv_sigs[idx]
            sv_card = sv_card[idx]
        X = StandardScaler().fit_transform(sv_sigs)
        km = KMeans(n_clusters=4, n_init=20, random_state=seed).fit(X)
        contingency = np.zeros((4, 4), dtype=np.int32)
        for cl, cd in zip(km.labels_, sv_card):
            contingency[cl, cd] += 1
        km_chi2, km_p, km_dof, _ = chi2_contingency(contingency)
        result["km_chi2"]       = float(km_chi2)
        result["km_p"]          = float(km_p)
        result["km_contingency"] = contingency
    else:
        result["km_chi2"] = float("nan")
        result["km_p"]    = float("nan")

    # ── 2. Direction LRT χ²(3) per dim (channel causation) ───────────────
    result["lrt_chi2"] = [float("nan")] * sig_dim
    result["lrt_p"]    = [float("nan")] * sig_dim
    result["lrt_coefs"] = [None] * sig_dim
    if nb_lag1 is not None:
        from scipy.stats import chi2 as chi2_dist
        A_STAY = 4
        CARD_NAMES = ["N", "S", "E", "W"]
        blind   = ~scouts
        fleeing = blind & (acts != A_STAY)
        has_sig = np.isfinite(nb_lag1).all(axis=1)
        elig    = fleeing & has_sig
        if nb_dist_lag1 is not None:
            has_dist = np.isfinite(nb_dist_lag1)
            elig &= has_dist
        n_elig = int(elig.sum())
        result["n_lrt_elig"] = n_elig
        if n_elig >= 50:
            y  = acts[elig].astype(np.int32)
            nb = nb_lag1[elig]
            ctrl = (nb_dist_lag1[elig].reshape(-1, 1)
                    if nb_dist_lag1 is not None
                    else np.zeros((n_elig, 1), dtype=np.float32))
            ctrl_z = (ctrl - ctrl.mean(axis=0)) / (ctrl.std(axis=0) + 1e-8)
            valid  = np.isin(y, [0, 1, 2, 3])
            y, nb, ctrl_z = y[valid], nb[valid], ctrl_z[valid]
            from sklearn.linear_model import LogisticRegression
            lr_ctrl = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            lr_ctrl.fit(ctrl_z, y)
            lp_ctrl  = lr_ctrl.predict_log_proba(ctrl_z)  # (n, n_cls)
            cls_idx  = {c: i for i, c in enumerate(lr_ctrl.classes_)}
            ll_ctrl  = float(np.sum(lp_ctrl[np.arange(len(y)),
                                            [cls_idx.get(yi, 0) for yi in y]]))
            for d in range(sig_dim):
                feat = np.hstack([ctrl_z, nb[:, d:d+1]])
                lr_one = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
                lr_one.fit(feat, y)
                lp_one = lr_one.predict_log_proba(feat)
                ci_one = {c: i for i, c in enumerate(lr_one.classes_)}
                ll_one = float(np.sum(lp_one[np.arange(len(y)),
                                             [ci_one.get(yi, 0) for yi in y]]))
                chi1 = max(0.0, 2 * (ll_one - ll_ctrl))
                p1   = float(1 - chi2_dist.cdf(chi1, df=3))
                result["lrt_chi2"][d]  = float(chi1)
                result["lrt_p"][d]     = float(p1)
                result["lrt_coefs"][d] = {
                    k: float(lr_one.coef_[i, -1])
                    for i, k in enumerate(CARD_NAMES)
                }
    return result


def withdrawal_comparison(baseline_path: str, withdrawal_path: str) -> None:
    """
    Compare pre- and post-withdrawal metrics side-by-side.
    Primary metrics:
      1. k-means contingency χ² — production-side vocabulary structure
      2. Direction LRT χ²(3) per dim — channel causation
    Verdict emitted automatically if +10k data is present.
    """
    print(f"\n{'═'*70}")
    print(f"  SCAFFOLDING WITHDRAWAL COMPARISON")
    print(f"{'═'*70}")
    print(f"  Baseline : {baseline_path}")
    print(f"  Withdrawal: {withdrawal_path}")

    print("\n  Loading baseline … ", end="", flush=True)
    base = _quick_withdrawal_metrics(baseline_path)
    print(f"({base['n']:,} records)")
    print("  Loading withdrawal … ", end="", flush=True)
    with_ = _quick_withdrawal_metrics(withdrawal_path)
    print(f"({with_['n']:,} records)")

    sig_dim = base["sig_dim"]

    # ── K-means contingency χ² ────────────────────────────────────────────
    print(f"\n  {'─'*66}")
    print(f"  METRIC 1 — K-means contingency χ²  (PRODUCTION vocabulary)")
    print(f"  {'─'*66}")
    bkm = base.get("km_chi2", float("nan"))
    wkm = with_.get("km_chi2", float("nan"))
    bp  = base.get("km_p",    float("nan"))
    wp  = with_.get("km_p",   float("nan"))
    delta_km = wkm - bkm if not (np.isnan(wkm) or np.isnan(bkm)) else float("nan")
    print(f"  Baseline    χ²(9) = {bkm:>8.1f}   p = {bp:.2e}")
    print(f"  Post-w/d    χ²(9) = {wkm:>8.1f}   p = {wp:.2e}   Δ = {delta_km:+.1f}")
    if not np.isnan(delta_km):
        pct_change = 100 * delta_km / (bkm + 1e-8)
        if pct_change < -20:
            km_verdict = f"⚠ DEGRADING  ({pct_change:.0f}% drop — vocabulary collapsing at production)"
        elif pct_change < -5:
            km_verdict = f"~ Softening  ({pct_change:.0f}% drop — monitor)"
        else:
            km_verdict = f"✓ PERSISTING ({pct_change:+.0f}% change — vocabulary structure held)"
        print(f"  → {km_verdict}")

    # ── Direction LRT χ²(3) per dim ───────────────────────────────────────
    print(f"\n  {'─'*66}")
    print(f"  METRIC 2 — Direction LRT χ²(3) per dim  (CHANNEL causation)")
    print(f"  {'─'*66}")
    print(f"  {'dim':>4}  {'baseline':>10}  {'post-w/d':>10}  {'Δ':>8}  verdict")
    lrt_verdicts = []
    for d in range(sig_dim):
        bc = base["lrt_chi2"][d]
        wc = with_["lrt_chi2"][d]
        if np.isnan(bc) or np.isnan(wc):
            print(f"  dim{d}  {'n/a':>10}  {'n/a':>10}  {'n/a':>8}")
            lrt_verdicts.append(None)
            continue
        dchi = wc - bc
        pct  = 100 * dchi / (bc + 1e-8)
        if pct < -30:
            v = "⚠ degrading"
        elif pct < -10:
            v = "~ softening"
        else:
            v = "✓ held"
        lrt_verdicts.append(v)
        print(f"  dim{d}  {bc:>10.1f}  {wc:>10.1f}  {dchi:>+8.1f}  {v}  ({pct:+.0f}%)")

    # ── Automated verdict ─────────────────────────────────────────────────
    print(f"\n  {'─'*66}")
    print(f"  WITHDRAWAL VERDICT")
    print(f"  {'─'*66}")
    n_lrt_ok = sum(1 for v in lrt_verdicts if v is not None and "✓" in v)
    km_ok    = not np.isnan(delta_km) and delta_km / (bkm + 1e-8) > -0.20

    if km_ok and n_lrt_ok >= sig_dim // 2:
        print(f"  ✅ VOCABULARY PERSISTS  (k-means held, {n_lrt_ok}/{sig_dim} LRT dims held)")
        print(f"  → Communication is self-sustaining. LAUNCH LARGE RUN NOW.")
    elif not km_ok and n_lrt_ok >= sig_dim // 2:
        print(f"  ⚠ EARLY-STAGE COLLAPSE  (k-means degrading, LRT channel coasting)")
        print(f"  → Production vocabulary eroding. Continue monitoring at +20k.")
    elif km_ok and n_lrt_ok < sig_dim // 2:
        print(f"  ⚠ CHANNEL WEAKENING  (k-means held, LRT dims dropping)")
        print(f"  → Monitor at +20k before deciding.")
    else:
        print(f"  ❌ VOCABULARY COLLAPSING  (both metrics degrading)")
        print(f"  → ToM reward was load-bearing. Include in large run config.")
    print(f"{'═'*70}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Decode Throng signal corpus")
    ap.add_argument(
        "corpus",
        nargs="?",
        default=None,
        help="Path to signal_corpus.jsonl (blue) or signal_corpus_red.jsonl (--red)",
    )
    ap.add_argument(
        "--red",
        action="store_true",
        help="Phase 12.2: decode red predator corpus (default path on volume)",
    )
    ap.add_argument("--k", type=int, default=8, help="Number of clusters")
    ap.add_argument("--min-step", type=int, default=0,
                    help="Ignore records before this step")
    ap.add_argument("--max-step", type=int, default=0,
                    help="Ignore records after this step (0 = no limit)")
    ap.add_argument("--baseline", default=None,
                    help="Pre-withdrawal corpus for comparison (skips full analysis)")
    args = ap.parse_args()

    if args.red:
        corpus_path = args.corpus or RED_CORPUS_DEFAULT
        if args.baseline is not None:
            sys.exit("--baseline is only supported for blue corpus decode.")
        if not Path(corpus_path).exists():
            sys.exit(f"Red corpus not found: {corpus_path}")
        print(f"\nLoading RED corpus from {corpus_path} …")
        data = load_red_corpus(corpus_path)
        decode_red_schema(
            data, k=args.k, min_step=args.min_step, max_step=args.max_step,
        )
        return

    if args.corpus is None:
        ap.error("corpus path required unless --red is set")

    if args.baseline is not None:
        withdrawal_comparison(args.baseline, args.corpus)
        return

    print(f"\nLoading corpus from {args.corpus} …")
    data = load_corpus(args.corpus)

    keep = np.ones(len(data["steps"]), dtype=bool)
    if args.min_step > 0:
        keep &= data["steps"] >= args.min_step
    if args.max_step > 0:
        keep &= data["steps"] <= args.max_step
    if not keep.all():
        for key in ("signals", "actions", "scouts", "steps"):
            data[key] = data[key][keep]
        data["ctx"] = {k: v[keep] for k, v in data["ctx"].items()}
        data["nb_lag1"]      = data["nb_lag1"][keep]      if data["nb_lag1"]      is not None else None
        data["nb_dist_lag1"] = data["nb_dist_lag1"][keep] if data["nb_dist_lag1"] is not None else None
        data["vq_tokens"]    = data["vq_tokens"][keep]
        data["nb_tok_lag1"]  = data["nb_tok_lag1"][keep]  if data["nb_tok_lag1"]  is not None else None
        data["n"]   = int(keep.sum())

    n       = data["n"]
    signals = data["signals"]
    actions = data["actions"]
    scouts  = data["scouts"]
    ctx     = data["ctx"]
    sig_dim = signals.shape[1]

    print(f"\n{'='*70}")
    print(f"  CORPUS SUMMARY")
    print(f"{'='*70}")
    print(f"  Records  : {n:,}")
    print(f"  Steps    : {data['steps'].min():,} – {data['steps'].max():,}")
    print(f"  Signal dim: {sig_dim}")
    print(f"  Scouts   : {scouts.sum():,}  ({100*scouts.mean():.1f}%)")
    print(f"  Actions  : " +
          "  ".join(f"{ACTION_NAMES[a]}={int((actions==a).sum())}" for a in range(5)))

    print(f"\n{'─'*70}")
    print(f"  SIGNAL RANGE PER DIM")
    print(f"{'─'*70}")
    for di in range(sig_dim):
        v = signals[:, di]
        print(f"  dim{di}: mean={v.mean():+.4f}  std={v.std():.4f}  "
              f"min={v.min():+.4f}  max={v.max():+.4f}")

    # ── Correlation table ──────────────────────────────────────────────────
    cor_mat = dim_correlations(signals, ctx)
    print(f"\n{'─'*70}")
    print(f"  SPEARMAN r(signal_dim, context_feature)")
    print(f"{'─'*70}")
    hdr = f"  {'dim':>5}  " + "  ".join(f"{k[:9]:>9}" for k in CONTEXT_KEYS)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for di in range(sig_dim):
        vals = "  ".join(f"{cor_mat[di, fi]:>+9.3f}" for fi in range(len(CONTEXT_KEYS)))
        print(f"  dim{di:>2}  {vals}")

    # ── MI table ──────────────────────────────────────────────────────────
    mi_mat = dim_mi(signals, ctx)
    print(f"\n{'─'*70}")
    print(f"  MUTUAL INFORMATION I(signal_dim ; context_feature)")
    print(f"{'─'*70}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    best_label = {}
    for di in range(sig_dim):
        vals = "  ".join(f"{mi_mat[di, fi]:>9.4f}" for fi in range(len(CONTEXT_KEYS)))
        best_fi = int(mi_mat[di].argmax())
        best_label[di] = CONTEXT_KEYS[best_fi] if mi_mat[di, best_fi] > 0.005 else "noise"
        print(f"  dim{di:>2}  {vals}  ← {best_label[di]}")

    # ── Vocabulary summary ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  DIMENSION ROLES (by peak MI)")
    print(f"{'─'*70}")
    for di in range(sig_dim):
        print(f"  dim{di} → {best_label[di]}")

    # ── Scout vs blind signal difference ─────────────────────────────────
    if scouts.any() and (~scouts).any():
        print(f"\n{'─'*70}")
        print(f"  SCOUT vs BLIND SIGNAL DIFFERENCE  (|mean_scout - mean_blind|)")
        print(f"{'─'*70}")
        for di in range(sig_dim):
            mu_s = float(signals[scouts,  di].mean())
            mu_b = float(signals[~scouts, di].mean())
            diff = mu_s - mu_b
            print(f"  dim{di}: scout={mu_s:+.4f}  blind={mu_b:+.4f}  Δ={diff:+.4f}")

    # ── Cluster vocabulary ────────────────────────────────────────────────
    cluster_analysis(signals, actions, scouts, ctx, k=args.k)

    # ── Lag-1 partial regression (communication test) ─────────────────────
    nb_lag1 = data.get("nb_lag1")
    nb_dist_lag1 = data.get("nb_dist_lag1")
    if nb_lag1 is not None:
        lag1_regression(signals, actions, scouts, ctx, nb_lag1, nb_dist_lag1, sig_dim)
        topographic_similarity(signals, actions, scouts, ctx, nb_lag1, sig_dim)
        categorical_vocabulary_tests(signals, scouts, ctx, sig_dim)
        lag1_direction_lrt(actions, scouts, nb_lag1, nb_dist_lag1, sig_dim)
        vq_token_direction_test(
            actions, scouts, ctx, data["vq_tokens"], data.get("nb_tok_lag1"),
            nb_dist_lag1, min_token_n=25, min_elig=50,
        )
    else:
        print(f"\n{'─'*70}")
        print(f"  LAG-1 COMMUNICATION TEST")
        print(f"{'─'*70}")
        print("  nb_scout_sig_lag1 field absent — re-run after corpus accumulates")
        print("  records with the new SignalCorpusWriter (restart required).")

    print(f"\n{'='*70}")
    print("  Done. Interpret:")
    print("  • dim with high MI(red_dist) → encodes danger proximity")
    print("  • dim with high MI(red_bear) → encodes danger direction")
    print("  • scout cluster with distinct signal → possible alarm call")
    print("  • clusters mapping to N/S/E/W → directional vocabulary")
    print("  • lag-1 β significant after controlling red_dist → genuine communication")
    print("  • VQ token direction χ²: alert codebook tokens → different flee mix")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
