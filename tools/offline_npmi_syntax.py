import json
import argparse
from collections import defaultdict
import math

def calculate_npmi(corpus_file, condition_key, condition_lambda, min_step=0):
    # Counts
    N = 0
    token_counts = defaultdict(int)
    cond_count = 0
    joint_counts = defaultdict(int)

    with open(corpus_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except:
                continue
                
            if data.get('step', 0) < min_step:
                continue
                
            N += 1
            token = data.get('vq_token', None)
            if token is None: continue
            
            token_counts[token] += 1
            
            # Check condition
            try:
                cond_met = condition_lambda(data)
            except Exception:
                cond_met = False
                
            if cond_met:
                cond_count += 1
                joint_counts[token] += 1

    if N == 0 or cond_count == 0:
        print(f"Condition '{condition_key}' not found enough times.")
        return

    print(f"Total events: {N}")
    print(f"Condition '{condition_key}' met: {cond_count} times")
    print(f"--- NPMI Scores (Tokens > 50 occurrences) ---")
    
    results = []
    p_cond = cond_count / N
    
    for token, t_count in token_counts.items():
        if t_count < 50:
            continue
            
        p_token = t_count / N
        j_count = joint_counts.get(token, 0)
        
        if j_count == 0:
            npmi = -1.0 # Never occurs together
        else:
            p_joint = j_count / N
            pmi = math.log(p_joint / (p_token * p_cond))
            h_joint = -math.log(p_joint)
            npmi = pmi / h_joint if h_joint > 0 else 0
            
        results.append((token, npmi, t_count, j_count))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    for token, npmi, t_count, j_count in results[:10]:
        print(f"Token {token:2d} | NPMI: {npmi: 5.3f} | Occurrences: {t_count:5d} | Joint: {j_count:5d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate NPMI between VQ tokens and environmental events")
    parser.add_argument('corpus', help="Path to signal_corpus.jsonl")
    parser.add_argument('--min-step', type=int, default=0, help="Minimum step to consider")
    args = parser.parse_args()
    
    # 1. NOUN CLASSIFICATION (Static / State-Based Features)
    nouns = {
        "Predator Close (red_dist < 5.0)": lambda x: x.get('red_dist', 10.0) < 5.0,
        "Starving (energy < 0.2)": lambda x: x.get('energy', 1.0) < 0.2,
        "High Resources (resource > 0.5)": lambda x: x.get('resource', 0.0) > 0.5,
        "Social Hub (neighbors >= 2)": lambda x: x.get('neighbors', 0.0) >= 2.0,
    }
    
    # 2. VERB CLASSIFICATION (Dynamic Actions)
    verbs = {
        "Strike Action (action == 5)": lambda x: x.get('action', 0) == 5,
        "Push Action (action == 6)": lambda x: x.get('action', 0) == 6,
        "Guard Action (action == 7)": lambda x: x.get('action', 0) == 7,
        "Build Action (action == 8)": lambda x: x.get('action', 0) == 8,
    }
    
    print("\n==========================================")
    print("NOUN CLASSIFICATION (State-Based Semantics)")
    print("==========================================")
    for name, func in nouns.items():
        print(f"\nEvaluating Noun: {name}")
        calculate_npmi(args.corpus, name, func, min_step=args.min_step)

    print("\n==========================================")
    print("VERB CLASSIFICATION (Action-Based Semantics)")
    print("==========================================")
    for name, func in verbs.items():
        print(f"\nEvaluating Verb: {name}")
        calculate_npmi(args.corpus, name, func, min_step=args.min_step)
