"""Quick check: what steps does the corpus actually cover?"""
import json, glob, os

corpus = "/kaggle/working/throng/runs_large/run_20260522_110122/signal_corpus.jsonl"
if not os.path.exists(corpus):
    found = glob.glob("/kaggle/working/throng/runs_large/run_*/signal_corpus.jsonl")
    if found:
        corpus = found[-1]

steps = []
with open(corpus) as f:
    for line in f:
        steps.append(json.loads(line)["step"])

print(f"Corpus: {corpus}")
print(f"Records: {len(steps):,}")
print(f"Step range: {min(steps):,} – {max(steps):,}")
print(f"First 10 steps: {sorted(set(steps))[:10]}")
print(f"Unique step count: {len(set(steps))}")
