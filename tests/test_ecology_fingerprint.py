"""Pins jax_sim.main_jax's ecology fingerprint (Cam, 2026-09-22): "a streak
counter is only meaningful within the regime it accumulated in." On resume,
if the world-defining config differs from the checkpoint's, every
comms-freeze tripwire streak and C0 capture resets.

The failure this exists to prevent actually happened: the PRESSURE counter
(11, accumulated under single-tile hearths) carried into a run with 49x the
hearth area and re-fired on its very first update, before the new geometry
could show anything. And the failure the float32 test guards against was
caught in review before landing: training_state stores the vector at
float32, so comparing the round-trip against a fresh float64 vector would
flag 0.3 -> 0.30000001 as a world change on EVERY resume.
"""

import numpy as np
import yaml

from jax_sim.main_jax import (
    DEFAULT_CONFIG, _normalize_config,
    ecology_fingerprint_vector, ecology_fingerprint_diff, ECOLOGY_FINGERPRINT_LABELS,
)
from jax_sim.ctd_ramp import HEARTH_RAMP_STAGE_N

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})


def test_same_world_survives_float32_round_trip_with_no_diff():
    fp = ecology_fingerprint_vector(_cfg, HEARTH_RAMP_STAGE_N)
    stored = np.asarray(fp, dtype=np.float32)  # what training_state actually holds
    assert ecology_fingerprint_diff(stored, fp) == [], (
        "an unchanged world must not register as changed after the float32 "
        "storage round-trip -- otherwise every resume resets every tripwire"
    )


def test_hearth_radius_change_is_detected_and_named():
    old = ecology_fingerprint_vector({**_cfg, "hearth_radius": 0}, HEARTH_RAMP_STAGE_N)
    new = ecology_fingerprint_vector({**_cfg, "hearth_radius": 3}, HEARTH_RAMP_STAGE_N)
    diff = ecology_fingerprint_diff(np.asarray(old, dtype=np.float32), new)
    assert diff == [("hearth_radius", 0.0, 3.0)]


def test_curriculum_schedule_change_is_detected():
    old = ecology_fingerprint_vector(_cfg, (1, 2, 3))
    new = ecology_fingerprint_vector(_cfg, (1, 2))
    labels = {d[0] for d in ecology_fingerprint_diff(old, new)}
    assert "hearth_ramp_n_len" in labels and "hearth_ramp_n_2" in labels


def test_catch_param_change_is_detected():
    p16 = dict(_cfg.get("phase16_combinatorial_syntax", {}))
    p16["coop_threshold_step"] = 50_000
    new = ecology_fingerprint_vector({**_cfg, "phase16_combinatorial_syntax": p16}, HEARTH_RAMP_STAGE_N)
    old = ecology_fingerprint_vector(_cfg, HEARTH_RAMP_STAGE_N)
    assert [d[0] for d in ecology_fingerprint_diff(old, new)] == ["coop_threshold_step"]


def test_non_world_keys_do_not_change_the_fingerprint():
    """Learning rate / batch size describe how fast we get there, not what
    world agents live in -- changing them must not reset tripwires."""
    old = ecology_fingerprint_vector(_cfg, HEARTH_RAMP_STAGE_N)
    new = ecology_fingerprint_vector({**_cfg, "ppo_lr": 1e-2, "ppo_minibatch_size": 7}, HEARTH_RAMP_STAGE_N)
    assert ecology_fingerprint_diff(old, new) == []


def test_vector_shape_is_fixed():
    fp = ecology_fingerprint_vector(_cfg, HEARTH_RAMP_STAGE_N)
    assert fp.shape == (len(ECOLOGY_FINGERPRINT_LABELS),)
    assert ecology_fingerprint_vector(_cfg, (1,)).shape == fp.shape


if __name__ == "__main__":
    test_same_world_survives_float32_round_trip_with_no_diff()
    test_hearth_radius_change_is_detected_and_named()
    test_curriculum_schedule_change_is_detected()
    test_catch_param_change_is_detected()
    test_non_world_keys_do_not_change_the_fingerprint()
    test_vector_shape_is_fixed()
    print("OK: ecology fingerprint detects and names world changes, ignores non-world keys, "
          "and does not cry wolf across the float32 storage round-trip.")
