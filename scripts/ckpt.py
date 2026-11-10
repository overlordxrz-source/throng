"""Checkpoint + resume for long runs (Cam Nov 10: required before any run over 2 h; Modal restarts a preempted function
on the same input, so a run must pick up where it stopped).

save(path, state) writes atomically (tmp + os.replace): a preemption mid-write leaves the previous checkpoint intact.
JAX arrays are pulled to numpy; everything else is pickled as is (numpy Generators keep their bit-generator state).
The caller stores its output file's byte offset in the state and truncates the file to it on resume, so rows written
after the checkpoint are not duplicated."""
from __future__ import annotations

import os
import pickle

import jax
import numpy as np


def _host(x):
    return jax.tree_util.tree_map(lambda v: np.asarray(v) if isinstance(v, jax.Array) else v, x)


def save(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(_host(state), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def truncate(out_path: str, offset: int) -> None:
    with open(out_path, "r+b") as f:
        f.truncate(offset)
