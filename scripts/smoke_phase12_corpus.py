#!/usr/bin/env python3
"""Smoke Phase 12.1 red corpus writer (no JAX sim)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from communication.analysis import SignalCorpusWriter


def main() -> None:
    sig_dim = 32
    n = 20
    alive_idx = np.arange(n)
    signals = np.random.randn(50, sig_dim).astype(np.float32)
    actions = np.random.randint(0, 5, size=50)
    token_ids = np.random.randint(0, 64, size=50)

    with tempfile.TemporaryDirectory() as td:
        blue_path = Path(td) / "signal_corpus.jsonl"
        red_path = Path(td) / "signal_corpus_red.jsonl"
        blue_w = SignalCorpusWriter(str(blue_path), sample_frac=0.5, every_n_steps=1)
        red_w = SignalCorpusWriter(str(red_path), sample_frac=0.5, every_n_steps=1)

        blue_w.maybe_record(
            step=100,
            alive_idx=alive_idx,
            signals=signals,
            actions=actions,
            is_scout=np.zeros(n, dtype=bool),
            nearest_red_dist=np.full(n, 99.0),
            nearest_red_bear=np.zeros(n),
            local_resource=np.zeros(n),
            own_energy=np.ones(n) * 0.5,
            neighbor_count=np.ones(n) * 2,
            token_ids=token_ids,
        )
        red_w.maybe_record_red(
            step=100,
            alive_idx=alive_idx,
            signals=signals,
            actions=actions,
            is_hunter=np.ones(n, dtype=bool),
            nearest_blue_dist=np.linspace(1, 7, n),
            nearest_blue_bear=np.linspace(0, 360, n),
            own_energy=np.ones(n) * 0.7,
            neighbor_count=np.ones(n) * 3,
            token_ids=token_ids,
            nb_hunter_sig_lag1=np.random.randn(n, sig_dim).astype(np.float32),
            nb_hunter_dist_lag1=np.linspace(2, 6, n),
            nb_hunter_token_lag1=np.random.randint(0, 64, size=n),
        )
        blue_w.close()
        red_w.close()

        blue_lines = blue_path.read_text().strip().splitlines()
        red_lines = red_path.read_text().strip().splitlines()
        assert len(blue_lines) >= 5, "blue corpus empty"
        assert len(red_lines) >= 5, "red corpus empty"
        rec_b = json.loads(blue_lines[0])
        rec_r = json.loads(red_lines[0])
        assert "scout" in rec_b and "red_dist" in rec_b
        assert "team" not in rec_b or rec_b.get("team") != "red"
        assert rec_r["team"] == "red"
        assert "hunter" in rec_r and "blue_dist" in rec_r
        assert "vq_token" in rec_r
        assert "nb_hunter_sig_lag1" in rec_r
        n_tok = len({json.loads(ln)["vq_token"] for ln in red_lines})
        print(f"[smoke] blue_lines={len(blue_lines)} red_lines={len(red_lines)}")
        print(f"[smoke] red unique vq_tokens in sample={n_tok}")
        print("[smoke] OK — dual corpus writers; schemas isolated")


if __name__ == "__main__":
    main()
