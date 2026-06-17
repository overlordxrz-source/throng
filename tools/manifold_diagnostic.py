import os
import json
import argparse
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr

def circular_correlation(alpha, beta):
    # alpha and beta in radians
    sin_a = np.sin(alpha - np.mean(alpha))
    sin_b = np.sin(beta - np.mean(beta))
    num = np.sum(sin_a * sin_b)
    den = np.sqrt(np.sum(sin_a**2) * np.sum(sin_b**2))
    return num / den

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="/mnt/throng-runs/signal_corpus.jsonl")
    parser.add_argument("--min-step", type=int, default=1008000, help="Only process steps >= min_step")
    args = parser.parse_args()
    
    print(f"Loading {args.corpus} (min_step >= {args.min_step})...")
    
    signals = []
    bears = []
    tokens = []
    
    with open(args.corpus, "r") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            if record["step"] < args.min_step:
                continue
                
            is_scout = np.asarray(record["is_scout"])
            if not np.any(is_scout): continue
            
            # extract scout data
            sigs = np.asarray(record["signals"])
            r_bears = np.asarray(record["nearest_red_bear"])
            r_dists = np.asarray(record["nearest_red_dist"])
            toks = np.asarray(record["token_ids"])
            
            scout_mask = (r_dists <= 8.0)
            
            if np.any(scout_mask):
                signals.append(sigs[scout_mask])
                bears.append(r_bears[scout_mask])
                tokens.append(toks[scout_mask])

    if not signals:
        print("No scout signals found.")
        return
        
    X = np.concatenate(signals, axis=0)
    y_bear = np.concatenate(bears, axis=0)
    t_ids = np.concatenate(tokens, axis=0)
    
    print(f"Found {len(X)} scout emissions.")
    
    # Unique codebook embeddings
    unique_tokens, idx = np.unique(t_ids, return_index=True)
    codebook = X[idx]
    print(f"Active codebook size in corpus: {len(unique_tokens)}/64")
    
    print("\n--- PCA on all scout signals ---")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratio (PC1, PC2, PC3): {pca.explained_variance_ratio_}")
    
    # Calculate angles in the PC1-PC2 plane
    pc_angles = np.degrees(np.arctan2(X_pca[:, 1], X_pca[:, 0])) % 360.0
    
    # Convert both to radians for circular correlation
    rad_pc = np.radians(pc_angles)
    rad_bear = np.radians(y_bear)
    
    circ_corr = circular_correlation(rad_pc, rad_bear)
    print(f"Circular correlation between PC1-PC2 angle and Red Bearing: {circ_corr:.4f}")
    
    # Linear regression test: PC1 and PC2 predicting sin(bear) and cos(bear)
    sin_b = np.sin(rad_bear)
    cos_b = np.cos(rad_bear)
    
    r_sin_1, p_sin_1 = pearsonr(X_pca[:, 0], sin_b)
    r_cos_1, p_cos_1 = pearsonr(X_pca[:, 0], cos_b)
    r_sin_2, p_sin_2 = pearsonr(X_pca[:, 1], sin_b)
    r_cos_2, p_cos_2 = pearsonr(X_pca[:, 1], cos_b)
    
    print("\n--- PC vs Bearing Components ---")
    print(f"PC1 vs sin(bear): r={r_sin_1:.3f} (p={p_sin_1:.2e}) | cos(bear): r={r_cos_1:.3f} (p={p_cos_1:.2e})")
    print(f"PC2 vs sin(bear): r={r_sin_2:.3f} (p={p_sin_2:.2e}) | cos(bear): r={r_cos_2:.3f} (p={p_cos_2:.2e})")
    print(f"PC3 vs sin(bear): r={pearsonr(X_pca[:, 2], sin_b)[0]:.3f} | cos(bear): r={pearsonr(X_pca[:, 2], cos_b)[0]:.3f}")
    
    # Check if continuous signaling is stronger than discrete token signaling
    print("\n--- Variance check ---")
    # Group mean of sin(bear) by token
    token_means_sin = [np.mean(sin_b[t_ids == t]) for t in unique_tokens]
    token_means_cos = [np.mean(cos_b[t_ids == t]) for t in unique_tokens]
    print(f"Variance of sin(bear) explained by discrete token: {np.var(token_means_sin) / np.var(sin_b):.4f}")
    
    # Also print the mapping of token ID to PC1, PC2 to show the manifold
    print("\n--- Top 10 Tokens (by frequency) in PCA space ---")
    counts = np.bincount(t_ids)
    top_tokens = np.argsort(counts)[::-1][:10]
    for t in top_tokens:
        if counts[t] == 0: continue
        idx_t = np.where(t_ids == t)[0][0]
        pc_pos = X_pca[idx_t]
        print(f"Token {t:2d} (N={counts[t]:5d}): PC1={pc_pos[0]:5.2f}, PC2={pc_pos[1]:5.2f}, Angle={pc_angles[idx_t]:5.1f} deg")

if __name__ == "__main__":
    main()
