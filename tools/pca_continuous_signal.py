import modal
import json
import numpy as np
from sklearn.decomposition import PCA

app = modal.App("throng-pca-analysis")
vol = modal.Volume.from_name("throng-runs", create_if_missing=True)

@app.function(volumes={"/mnt/throng-runs": vol})
def compute_pca():
    sigs = []
    print("Loading corpus from /mnt/throng-runs/signal_corpus.jsonl (min-step=1000000)...")
    with open('/mnt/throng-runs/signal_corpus.jsonl') as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get('step', 0) >= 1000000 and row.get('scout'):
                    sigs.append(row['sig'][:32])
            except:
                pass
    X = np.array(sigs)
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"8D: {cumvar[7]:.3f}  16D: {cumvar[15]:.3f}  24D: {cumvar[23]:.3f}")
    
    P = pca.components_[:8, :]
    np.save('/mnt/throng-runs/pca_proj_8d.npy', P.T)
    np.save('/mnt/throng-runs/pca_proj_16d.npy', pca.components_[:16, :].T)
    print("Saved projections to /mnt/throng-runs/pca_proj_{8d,16d}.npy")

@app.local_entrypoint()
def main():
    compute_pca.remote()

