"""Cam's fix (2026-09-20): signal_corpus.jsonl is append-only across every
launch that ever points at it, and step numbers reset on every resume -- four
separate launches into the same 2541-lineage corpus in one evening produced
overlapping step ranges, one of them from a since-collapsed channel (measured
via a duplicate-(agent,step)-key scan; see docs/WILL_RESTART_SEP2026.md
section 5.4's 2026-09-20 entry). Without a per-launch marker, detecting that
required reconstructing launch history from train.log after the fact.

SignalCorpusWriter now tags every record with launch_id -- minted once per
process (jax_sim/main_jax.py) and shared between the blue and red writers of
the same launch, or self-minted if the caller doesn't supply one. Verified
here against the real class, writing to a real temp file.
"""

import json
import tempfile
import os

import numpy as np

from communication.analysis import SignalCorpusWriter


def _write_one_sample(writer, step=100, n=4):
    alive_idx = np.arange(n)
    signals = np.zeros((n, 4))
    actions = np.zeros(n, dtype=int)
    is_scout = np.zeros(n, dtype=bool)
    zeros = np.zeros(n)
    writer.maybe_record(
        step=step, alive_idx=alive_idx, signals=signals, actions=actions,
        is_scout=is_scout, nearest_red_dist=zeros, nearest_red_bear=zeros,
        local_resource=zeros, own_energy=zeros, neighbor_count=zeros.astype(int),
    )


def test_every_record_carries_the_supplied_launch_id():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "corpus.jsonl")
        writer = SignalCorpusWriter(path=path, sample_frac=1.0, every_n_steps=1, launch_id="launch-A")
        _write_one_sample(writer, step=100)
        writer._fh.close()

        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) > 0, "no records written -- fixture bug, not a real test"
        assert all(r["launch_id"] == "launch-A" for r in records), (
            "not every record carries the launch_id passed to the constructor"
        )


def test_two_writers_without_an_explicit_launch_id_get_different_ones():
    """Standalone use (no launch_id passed, e.g. a script or test) must still
    tag records -- self-minted, and distinct across separate writer
    instances, so records from genuinely different processes/launches are
    still distinguishable even without explicit wiring."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_a = os.path.join(tmp_dir, "a.jsonl")
        path_b = os.path.join(tmp_dir, "b.jsonl")
        writer_a = SignalCorpusWriter(path=path_a, sample_frac=1.0, every_n_steps=1)
        writer_b = SignalCorpusWriter(path=path_b, sample_frac=1.0, every_n_steps=1)
        assert writer_a.launch_id != writer_b.launch_id, (
            "two independently constructed writers minted the same launch_id -- "
            "collision risk defeats the whole point of the marker"
        )
        assert writer_a.launch_id, "launch_id must not be empty/falsy"


def test_shared_launch_id_across_blue_and_red_style_writers():
    """jax_sim/main_jax.py mints ONE launch_id and passes it to both the
    blue and red corpus writers of the same process -- verify that pattern
    actually produces matching records across both files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        shared_id = "shared-launch-xyz"
        path_blue = os.path.join(tmp_dir, "blue.jsonl")
        path_red = os.path.join(tmp_dir, "red.jsonl")
        writer_blue = SignalCorpusWriter(path=path_blue, sample_frac=1.0, every_n_steps=1, launch_id=shared_id)
        writer_red = SignalCorpusWriter(path=path_red, sample_frac=1.0, every_n_steps=1, launch_id=shared_id)
        _write_one_sample(writer_blue, step=200)
        _write_one_sample(writer_red, step=200)
        writer_blue._fh.close()
        writer_red._fh.close()

        with open(path_blue) as f:
            blue_records = [json.loads(line) for line in f if line.strip()]
        with open(path_red) as f:
            red_records = [json.loads(line) for line in f if line.strip()]
        assert all(r["launch_id"] == shared_id for r in blue_records + red_records), (
            "blue and red writers sharing one launch_id must produce records "
            "tagged with that same id in both files"
        )


if __name__ == "__main__":
    test_every_record_carries_the_supplied_launch_id()
    test_two_writers_without_an_explicit_launch_id_get_different_ones()
    test_shared_launch_id_across_blue_and_red_style_writers()
    print(
        "OK: every corpus record carries launch_id (the value passed to the "
        "constructor, or a self-minted one that's distinct across independent "
        "writer instances); the blue+red-share-one-id pattern main_jax.py uses "
        "produces matching tags across both files -- overlapping launches into "
        "the same corpus are now a groupby on launch_id, not an archaeology "
        "problem."
    )
