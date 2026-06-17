"""
Standalone Training Script for Phase 17 Rosetta Stone
Unsupervised Semantic Translation (MARL VQ -> GloVe)
"""

import os
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
import json
import argparse

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jax_sim.rosetta_stone_jax import RosettaStone, rosetta_stone_loss

def load_glove_embeddings(glove_path, top_n=5000):
    """Loads a subset of GloVe embeddings, L2 normalized."""
    embeddings = []
    words = []
    print(f"Loading top {top_n} GloVe vectors from {glove_path}...")
    with open(glove_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i >= top_n:
                break
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            words.append(word)
            embeddings.append(vector)
            
    embeddings = np.array(embeddings)
    # L2 Normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return words, embeddings

def get_marl_distributions(corpus_path, vocab_size=64, min_step=992000):
    """
    Parses the signal corpus to extract co-occurrence statistics and builds 
    a distribution for each token to feed into the autoencoder.
    Calculates token frequencies for weighted Gromov-Wasserstein alignment.
    """
    print(f"Reading MARL signal corpus from {corpus_path} (min_step >= {min_step})...")
    token_counts = np.zeros(vocab_size, dtype=np.float32)
    
    with open(corpus_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if data.get('step', 0) < min_step:
                    continue
                token = data.get('vq_token', None)
                if token is not None and 0 <= token < vocab_size:
                    token_counts[token] += 1
            except Exception:
                continue
                
    # Normalize frequencies (prevent division by zero)
    total_count = np.sum(token_counts)
    if total_count > 0:
        frequencies = token_counts / total_count
    else:
        frequencies = np.ones(vocab_size, dtype=np.float32) / vocab_size
        
    print(f"Extracted frequencies for {vocab_size} tokens.")
    tokens = np.eye(vocab_size, dtype=np.float32)
    return tokens, frequencies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove-path', type=str, default='glove.6B.50d.txt', help='Path to GloVe embeddings')
    parser.add_argument('--corpus-path', type=str, default='/mnt/throng-runs/signal_corpus.jsonl')
    parser.add_argument('--llm-dim', type=int, default=50, help='Dimensionality of GloVe')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--min-step', type=int, default=992000, help='Exclude pre-burn-off vocabulary')
    args = parser.parse_args()
    
    if not os.path.exists(args.glove_path):
        print(f"WARNING: GloVe file {args.glove_path} not found. Using random continuous data for dry-run testing.")
        words = [f"word_{i}" for i in range(5000)]
        target_embs = np.random.randn(5000, args.llm_dim).astype(np.float32)
        norms = np.linalg.norm(target_embs, axis=1, keepdims=True)
        target_embs = target_embs / (norms + 1e-8)
    else:
        words, target_embs = load_glove_embeddings(args.glove_path, top_n=5000)
        
    marl_tokens, token_frequencies = get_marl_distributions(args.corpus_path, vocab_size=64, min_step=args.min_step)
    
    # Initialize Rosetta Stone
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    model = RosettaStone(vocab_size=64, llm_dim=args.llm_dim)
    dummy_marl = jnp.zeros((args.batch_size, 64))
    params = model.init(init_rng, dummy_marl, mode="fwd")
    
    tx = optax.adam(learning_rate=3e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    @jax.jit
    def train_step(state, batch_marl, batch_llm, batch_freqs, key):
        def loss_fn(p):
            # Pass batch_freqs down to loss if needed, but for now we'll just weight the GW loss matrix
            return rosetta_stone_loss(
                p, 
                model=model, 
                batch_marl=batch_marl, 
                batch_llm=batch_llm, 
                rng=key, 
                temperature=1.0, 
                gw_rank=10, 
                epsilon=1e-2,
                freq_weights=batch_freqs
            )
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    print("Beginning Gromov-Wasserstein Alignment Training...")
    
    for step in range(args.steps):
        rng, step_rng1, step_rng2 = jax.random.split(rng, 3)
        
        # Sample MARL batch
        idx_marl = jax.random.choice(step_rng1, 64, shape=(args.batch_size,), replace=True)
        batch_marl = jnp.take(marl_tokens, idx_marl, axis=0)
        batch_freqs = jnp.take(token_frequencies, idx_marl, axis=0)
        
        # Sample target batch
        idx_llm = jax.random.choice(step_rng2, len(target_embs), shape=(args.batch_size,), replace=True)
        batch_llm = jnp.take(target_embs, idx_llm, axis=0)
        
        state, metrics = train_step(state, batch_marl, batch_llm, batch_freqs, rng)
        
        if step % 200 == 0:
            print(f"Step {step:05d} | Loss: {metrics['loss_total']:.4f} | "
                  f"Cycle(Fwd): {metrics['loss_cycle_fwd']:.4f} | "
                  f"Cycle(Bwd): {metrics['loss_cycle_bwd']:.4f} | "
                  f"GW: {metrics['loss_gw']:.4f} | "
                  f"Iso: {metrics['loss_isometry']:.4f}")

    print("\n--- THE ROSETTA STONE ---")
    mapped_anchors = model.apply(params, marl_tokens, mode="fwd") # [64, 50]
    for i in range(64):
        # Find nearest 5 words
        dists = jnp.linalg.norm(target_embs - mapped_anchors[i], axis=1)
        top_indices = jnp.argsort(dists)[:5]
        top_words = [words[idx] for idx in top_indices]
        if token_frequencies[i] > 0.001:
            print(f"Token {i:02d} (freq {token_frequencies[i]:.3f}) -> {top_words}")
    
    print("Translation complete!")

if __name__ == '__main__':
    main()
