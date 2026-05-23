"""
Extract the mean alarm signal vector from the danger cluster.
Load signal_corpus.jsonl, cluster with k=8, identify the high-scout
low-red_dist cluster, and save its centroid + survival stats.
"""
import json
import numpy as np
from sklearn.cluster import KMeans
import sys

def main(corpus_path, out_path="alarm_vector.json"):
    records = []
    with open(corpus_path) as f:
        for line in f:
            r = json.loads(line)
            if 150001 <= r["step"] <= 157201:
                records.append(r)

    signals = np.array([r["sig"] for r in records])  # (N, 32)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10).fit(signals)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print(f"Records: {len(records)}\n")
    print("Cluster analysis:")
    print(f"{'Clust':>6} {'N':>6} {'Scout%':>7} {'AvgRedDist':>10} {'DomAction':>9}")
    alarm_cid = None
    alarm_score = -1
    for cid in range(8):
        mask = labels == cid
        n = mask.sum()
        scout_pct = sum(1 for i in range(len(records)) if mask[i] and records[i].get("scout", False)) / max(n,1) * 100
        avg_red = np.mean([records[i]["red_dist"] for i in range(len(records)) if mask[i]])
        actions = [records[i]["action"] for i in range(len(records)) if mask[i]]
        dom_act = max(set(actions), key=actions.count) if actions else "?"
        print(f"{cid:>6} {n:>6} {scout_pct:>7.1f} {avg_red:>10.2f} {dom_act:>9}")
        # Score: high scout % + low red_dist = alarm cluster
        score = scout_pct / max(avg_red, 1.0)
        if score > alarm_score:
            alarm_score = score
            alarm_cid = cid

    print(f"\n🚨 ALARM CLUSTER = {alarm_cid}")
    print(f"Centroid (32-dim): {centroids[alarm_cid].round(4).tolist()}")

    # Survival rate for blind agents receiving similar signal
    blind_mask = np.array([not r.get("scout", False) for r in records])
    blind_signals = signals[blind_mask]
    # Proxy for survival: did they flee? (action != STAY=4)
    flee_actions = {"N":0, "S":1, "E":2, "W":3}
    blind_fled = np.array([r["action"] in flee_actions.values() for r in records])[blind_mask]
    dists = np.linalg.norm(blind_signals - centroids[alarm_cid], axis=1)
    close = dists < np.percentile(dists, 20)  # bottom 20% distance
    far = dists > np.percentile(dists, 80)    # top 20% distance
    print(f"\nBlind agents near alarm vector: flee rate = {blind_fled[close].mean():.3f} (n={close.sum()})")
    print(f"Blind agents far from alarm vector: flee rate = {blind_fled[far].mean():.3f} (n={far.sum()})")

    with open(out_path, "w") as f:
        json.dump({
            "alarm_cluster_id": int(alarm_cid),
            "centroid": centroids[alarm_cid].tolist(),
            "centroid_rounded": [round(x,4) for x in centroids[alarm_cid].tolist()],
            "n_cluster": int((labels==alarm_cid).sum()),
            "blind_near_flee_rate": float(blind_fled[close].mean()),
            "blind_far_flee_rate": float(blind_fled[far].mean()),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    import glob
    corpus = "/kaggle/working/throng/runs_large/run_20260522_110122/signal_corpus.jsonl"
    if not __import__("os").path.exists(corpus):
        found = glob.glob("/kaggle/working/throng/runs_large/run_*/signal_corpus.jsonl")
        if found:
            corpus = found[-1]
    main(corpus)
