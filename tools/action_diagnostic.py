import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

corpus_path = "/mnt/throng-runs/signal_corpus.jsonl"
print(f"Loading {corpus_path}...")

records_data = []
with open(corpus_path, "r") as f:
    for line in f:
        if not line.strip(): continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "action" not in rec: continue
        records_data.append(rec)

sigs = np.stack([r.get("sig", np.zeros(32)) for r in records_data], axis=0)
actions = np.array([r["action"] for r in records_data])

print(f"Loaded {len(sigs)} records.")

# Predict action from signal
lr = LogisticRegression(max_iter=1000)
lr.fit(sigs, actions)
preds = lr.predict(sigs)
acc = accuracy_score(actions, preds)
print(f"Logistic Regression predicting Sender's Action from Signal: Accuracy = {acc:.4f} (Baseline = {np.max(np.bincount(actions))/len(actions):.4f})")

# Mean signal per action
print("\nMean PC per action:")
pca = PCA(n_components=3).fit(sigs)
X_pca = pca.transform(sigs)
for a in np.unique(actions):
    mask = actions == a
    mean_pc1 = np.mean(X_pca[mask, 0])
    mean_pc2 = np.mean(X_pca[mask, 1])
    print(f"Action {a} (N={np.sum(mask)}): PC1={mean_pc1:.2f}, PC2={mean_pc2:.2f}")

