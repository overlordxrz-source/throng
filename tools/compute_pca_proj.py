import numpy as np
import json
from sklearn.decomposition import PCA

sigs = []
print("Loading corpus...", flush=True)
with open('signal_corpus.jsonl') as f:
    for line in f:
        try:
            row = json.loads(line)
            if row.get('step', 0) >= 1130000 and row.get('scout'):
                sigs.append(row['sig'][:32])
        except json.JSONDecodeError:
            continue

X = np.array(sigs)
pca = PCA()
pca.fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"Variance captured at 8D: {cumvar[7]:.4f}")

P = pca.components_[:8, :] # (8, 32)
np.save('pca_proj_8d.npy', P.T) # save (32, 8)
print("Saved PCA projection matrix to pca_proj_8d.npy")
