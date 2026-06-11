#!/usr/bin/env python3
"""Phase 17 Verification Diagnostic — run this in a Modal Jupyter cell.

Checks whether the Gumbel-Softmax bottleneck has successfully crushed
the continuous entropy broadcasting (Protean Scattering).
"""

import json
import numpy as np
from collections import Counter

CORPUS = "/mnt/throng-runs/signal_corpus.jsonl"

print(f"Loading {CORPUS}...")
records = []
with open(CORPUS, "r") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "sig" in rec:
            records.append(rec)
        if len(records) >= 100000:
            break

if len(records) < 1000:
    print(f"Only {len(records)} records found. Let the run accumulate more data.")
else:
    sigs = np.stack([r["sig"] for r in records])
    print(f"Loaded {len(sigs)} records with {sigs.shape[1]}D signals.\n")

    # ── 1. PCA Variance Analysis ──
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5).fit(sigs)
    print("═══ PCA Variance Analysis ═══")
    print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    print(f"  PC1 = {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2 = {pca.explained_variance_ratio_[1]:.1%}")
    total_var = np.var(sigs, axis=0).sum()
    print(f"  Total signal variance: {total_var:.4f}")
    
    if pca.explained_variance_ratio_[0] > 0.40:
        print("\n⚠️  PC1 still dominates (>40%). Entropy broadcasting may NOT be fully crushed.")
        print("    If this is a fresh run, the old codebook geometry may still be lingering.")
        print("    Wait for more training steps and re-run.")
    else:
        print("\n✅  PC1 variance is low. Continuous entropy broadcasting appears crushed!")

    # ── 2. Token Distribution ──
    tokens = []
    for r in records:
        t = r.get("tok", r.get("token_id"))
        if t is not None:
            tokens.append(int(t))
    
    if tokens:
        print("\n═══ Discrete Token Analysis ═══")
        unique_tokens = len(set(tokens))
        print(f"Active tokens: {unique_tokens} / 64")
        top10 = Counter(tokens).most_common(10)
        print("Top 10 tokens:", top10)
        
        # Entropy of token distribution
        counts = np.array([c for _, c in Counter(tokens).items()], dtype=float)
        probs = counts / counts.sum()
        token_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(64)
        print(f"Token entropy: {token_entropy:.2f} bits (max = {max_entropy:.2f})")
        
        if unique_tokens <= 3:
            print("\n⚠️  Language collapse detected (≤3 active tokens).")
            print("    This is EXPECTED immediately after bottleneck hardening.")
            print("    The agents need 50-100k steps to rediscover discrete semantics.")
        elif unique_tokens >= 10:
            print("\n✅  Healthy token diversity! Agents are using 10+ discrete tokens.")
    else:
        print("\nNo token IDs found in corpus records.")

    # ── 3. Signal-Feature Correlation ──
    print("\n═══ Signal-Feature Correlations ═══")
    from scipy.stats import spearmanr
    
    for feat_name in ["norm_x", "norm_y"]:
        vals = [r.get(feat_name) for r in records if feat_name in r]
        if len(vals) >= 1000:
            vals = np.array(vals[:len(sigs)])
            if len(vals) == len(sigs):
                correlations = [abs(spearmanr(sigs[:, d], vals).correlation) for d in range(sigs.shape[1])]
                max_r = max(correlations)
                mean_r = np.mean(correlations)
                print(f"  {feat_name}: max|r| = {max_r:.4f}, mean|r| = {mean_r:.4f}")
    
    print("\nDiagnostic complete.")
