import json
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

def main():
    corpus_path = "/mnt/throng-runs/signal_corpus.jsonl"
    print(f"Loading {corpus_path}...")
    
    records_data = []
    
    with open(corpus_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except:
                continue
            if rec.get("step", 0) < 1008000:
                continue
            # Keep all agents, not just scouts
            records_data.append(rec)
            if len(records_data) >= 50000:
                break

    if not records_data:
        print("No records found.")
        return
        
    sigs = np.stack([r.get("sig", np.zeros(32)) for r in records_data], axis=0)
    toks = np.array([r.get("vq_token", -1) for r in records_data])
    
    # Extract all possible features
    features = {
        "red_dist": np.array([r.get("red_dist", 999.0) for r in records_data]),
        "red_bear": np.array([r.get("red_bear", 0.0) for r in records_data]),
        "energy": np.array([r.get("energy", 0.0) for r in records_data]),
        "neighbors": np.array([r.get("neighbors", 0.0) for r in records_data]),
        "local_resource": np.array([r.get("local_resource", 0.0) for r in records_data]),
        "adj_red": np.array([r.get("adj_red", False) for r in records_data]).astype(float),
        "adj_bg": np.array([r.get("adj_bg", False) for r in records_data]).astype(float),
        "adj_barrier": np.array([r.get("adj_barrier", False) for r in records_data]).astype(float),
        "is_scout": np.array([r.get("scout", False) for r in records_data]).astype(float)
    }
    
    print(f"Loaded {len(sigs)} records.")
    
    print("\n--- Correlation of 32D Signal with Features ---")
    for fname, fval in features.items():
        if np.var(fval) < 1e-6:
            continue
        corrs = []
        for d in range(sigs.shape[1]):
            r, p = spearmanr(sigs[:, d], fval)
            corrs.append(abs(r))
        max_r = max(corrs)
        max_d = np.argmax(corrs)
        mean_r = np.mean(corrs)
        print(f"Feature '{fname}': Max |r| = {max_r:.4f} (on dim {max_d}), Mean |r| = {mean_r:.4f}")
        
    print("\n--- Token Usage Distribution ---")
    unique, counts = np.unique(toks, return_counts=True)
    print(f"Active tokens: {len(unique)}/64. Min freq: {np.min(counts)}, Max freq: {np.max(counts)}, Mean: {np.mean(counts):.1f}")
    
    # If there's high correlation with something, print it
    print("\n--- PCA Analysis ---")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(sigs)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    for fname, fval in features.items():
        if np.var(fval) < 1e-6: continue
        r1, _ = spearmanr(X_pca[:, 0], fval)
        r2, _ = spearmanr(X_pca[:, 1], fval)
        r3, _ = spearmanr(X_pca[:, 2], fval)
        if max(abs(r1), abs(r2), abs(r3)) > 0.1:
            print(f"PC correlations with {fname}: PC1={r1:.3f}, PC2={r2:.3f}, PC3={r3:.3f}")

if __name__ == "__main__":
    main()
