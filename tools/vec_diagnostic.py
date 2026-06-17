import json
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

corpus_path = "/mnt/throng-runs/signal_corpus.jsonl"
records_data = []
with open(corpus_path, "r") as f:
    for line in f:
        if not line.strip(): continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "red_dist" not in rec or "red_bear" not in rec or "sig" not in rec:
            continue
        # Filter out invalid red_bear (-999.0)
        if rec["red_bear"] < -10:
            continue
        records_data.append(rec)
        if len(records_data) >= 50000:
            break

if not records_data:
    print("No records found.")
else:
    sigs = np.stack([r["sig"] for r in records_data], axis=0)
    
    # Relative vector to predator
    red_dist = np.array([r["red_dist"] for r in records_data])
    red_bear = np.array([r["red_bear"] for r in records_data])
    
    vec_x = red_dist * np.cos(red_bear)
    vec_y = red_dist * np.sin(red_bear)
    
    print(f"Loaded {len(sigs)} records. Testing Vector to Predator...")
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(sigs)
    
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
    r1x, _ = spearmanr(X_pca[:, 0], vec_x)
    r1y, _ = spearmanr(X_pca[:, 0], vec_y)
    r2x, _ = spearmanr(X_pca[:, 1], vec_x)
    r2y, _ = spearmanr(X_pca[:, 1], vec_y)
    
    print(f"PC1 vs vec_x: {r1x:.3f}, vec_y: {r1y:.3f}")
    print(f"PC2 vs vec_x: {r2x:.3f}, vec_y: {r2y:.3f}")

