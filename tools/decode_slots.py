"""
tools/decode_slots.py

Offline analysis for the Phase 18 multi-slot communication architecture.
Performs:
1. NPMI (Normalized Pointwise Mutual Information) analysis on 3 discrete slots.
2. Causal Token-Swap intervention scaffolding (ATE test).
"""

import argparse
import json
import numpy as np

# Mock definitions for offline testing
ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY", 5: "STRK", 6: "PUSH", 7: "GRD", 8: "BUILD", 9: "PICKUP", 10: "CRAFT", 11: "USE"}
CONTEXT_KEYS = ["red_dist", "red_bear", "resource", "energy", "wood", "stone", "tool"]

def load_corpus(path, min_step=0):
    records = []
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r["step"] >= min_step:
                    records.append(r)
            except:
                continue
    return records

def calculate_npmi(records, slot_idx, context_func, context_name):
    """
    Calculate NPMI between a specific token slot and a derived discrete context.
    slot_idx: 0, 1, or 2
    context_func: function mapping record -> discrete context bucket (e.g. str)
    """
    token_counts = {}
    context_counts = {}
    joint_counts = {}
    total = len(records)
    
    for r in records:
        toks = r.get("token_ids", [])
        if not toks or len(toks) <= slot_idx:
            continue
        tok = toks[slot_idx]
        ctx = context_func(r)
        
        token_counts[tok] = token_counts.get(tok, 0) + 1
        context_counts[ctx] = context_counts.get(ctx, 0) + 1
        joint_counts[(tok, ctx)] = joint_counts.get((tok, ctx), 0) + 1
        
    print(f"\n--- NPMI for Slot {slot_idx} vs {context_name} ---")
    results = []
    for (tok, ctx), count in joint_counts.items():
        p_xy = count / total
        p_x = token_counts[tok] / total
        p_y = context_counts[ctx] / total
        
        pmi = np.log2(p_xy / (p_x * p_y))
        npmi = pmi / -np.log2(p_xy)
        if count > 50: # Minimum support threshold
            results.append((npmi, tok, ctx, count))
            
    results.sort(reverse=True)
    for npmi, tok, ctx, count in results[:15]:
        print(f"Token {tok:2d} | Context: {ctx:10s} | NPMI: {npmi:.3f} | Support: {count}")

def test_npmi(records):
    # Slot 0: Noun -> Environmental context buckets
    def noun_context(r):
        if r.get("red_dist", 99) < 2.0: return "DANGER_NEAR"
        if r.get("resource", 99) < 2.0: return "RESOURCE_NEAR"
        return "CLEAR"
    
    # Slot 1: Verb -> Executed actions
    def verb_context(r):
        return ACTION_NAMES.get(r.get("action", -1), "UNKNOWN")
    
    # Slot 2: Modifier -> Intensity (Urgency)
    def modifier_context(r):
        energy = r.get("energy", 1.0)
        if energy < 0.2: return "CRITICAL_ENERGY"
        if energy < 0.5: return "LOW_ENERGY"
        return "HIGH_ENERGY"

    calculate_npmi(records, 0, noun_context, "Noun Context (Env)")
    calculate_npmi(records, 1, verb_context, "Verb Context (Action)")
    calculate_npmi(records, 2, modifier_context, "Modifier Context (Urgency)")

def test_ate_swap(records):
    print("\n--- Causal Intervention Scaffold (ATE Token Swap) ---")
    print("WARNING: This requires launching a frozen JAX simulation with mid-flight tensor surgery.")
    print("Steps to execute:")
    print("1. Identify highest NPMI token for a specific action (e.g., Token 42 = FLEE).")
    print("2. Initialize JAX `sim_step` with `apply_ate_swap=True`.")
    print("3. In `sim_step.py`, intercept `token_ids` matrix during communication resolve.")
    print("4. Force `token_ids[receiver_index, 1] = 42` (Verb slot).")
    print("5. Measure $\Delta P(Action=FLEE)$ over 100 rollout steps.")
    print("6. Target: ATE > 0.05. If 0.0, the channel is a cheap-talk proxy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Path to signal_corpus.jsonl")
    parser.add_argument("--min-step", type=int, default=0, help="Ignore records before this step")
    args = parser.parse_args()
    
    records = load_corpus(args.corpus, args.min_step)
    if records:
        print(f"Loaded {len(records)} records.")
        test_npmi(records)
        test_ate_swap(records)
    else:
        print("No valid records found.")
