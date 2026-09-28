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
from jax_sim.ctd_ramp import HEARTH_RAMP_STAGE_N, HEARTH_RAMP_ACCEPT_ANY

with open("config.yaml") as _f:
    _cfg = _normalize_config({**DEFAULT_CONFIG, **yaml.safe_load(_f)})


def _fp(cfg, n=HEARTH_RAMP_STAGE_N, accept_any=HEARTH_RAMP_ACCEPT_ANY):
    return ecology_fingerprint_vector(cfg, n, accept_any)


def test_same_world_survives_float32_round_trip_with_no_diff():
    fp = _fp(_cfg)
    stored = np.asarray(fp, dtype=np.float32)  # what training_state actually holds
    assert ecology_fingerprint_diff(stored, fp) == [], (
        "an unchanged world must not register as changed after the float32 "
        "storage round-trip -- otherwise every resume resets every tripwire"
    )


def test_hearth_radius_change_is_detected_and_named():
    old = _fp({**_cfg, "hearth_radius": 0})
    new = _fp({**_cfg, "hearth_radius": 3})
    diff = ecology_fingerprint_diff(np.asarray(old, dtype=np.float32), new)
    assert diff == [("hearth_radius", 0.0, 3.0)]


def test_curriculum_schedule_change_is_detected():
    old = _fp(_cfg, n=(1, 2, 3))
    new = _fp(_cfg, n=(1, 2))
    labels = {d[0] for d in ecology_fingerprint_diff(old, new)}
    assert "hearth_ramp_n_len" in labels and "hearth_ramp_n_2" in labels


def test_a0_a1_split_is_detected_and_named():
    """2026-09-28 (Cam): accept_any_material is exactly as world-defining as
    N -- a checkpoint trained under the old 3-stage (no accept-any) schedule
    resuming into the new A0/A1 split must reset, not silently inherit a
    stage/streak from a curriculum with a different meaning at that index."""
    old = _fp(_cfg, n=(1, 2, 3), accept_any=(False, False, False))
    new = _fp(_cfg, n=HEARTH_RAMP_STAGE_N, accept_any=HEARTH_RAMP_ACCEPT_ANY)
    labels = {d[0] for d in ecology_fingerprint_diff(old, new)}
    assert "hearth_ramp_n_len" in labels
    assert "hearth_ramp_accept_any_0" in labels  # old stage0 (False) -> new A0 (True)


def test_accept_any_change_alone_is_detected():
    old = _fp(_cfg, n=(1, 1, 2), accept_any=(False, False, False))
    new = _fp(_cfg, n=(1, 1, 2), accept_any=(True, False, False))
    diff = ecology_fingerprint_diff(old, new)
    assert diff == [("hearth_ramp_accept_any_0", 0.0, 1.0)]


def test_success_bar_change_is_detected():
    old = _fp(_cfg)
    p16 = dict(_cfg.get("ctd_competence_ramp", {}))
    p16["craft_ramp_success_bar"] = 0.10
    new = _fp({**_cfg, "ctd_competence_ramp": p16})
    assert [d[0] for d in ecology_fingerprint_diff(old, new)] == ["craft_ramp_success_bar"]


def test_catch_param_change_is_detected():
    p16 = dict(_cfg.get("phase16_combinatorial_syntax", {}))
    p16["coop_threshold_step"] = 50_000
    new = _fp({**_cfg, "phase16_combinatorial_syntax": p16})
    old = _fp(_cfg)
    assert [d[0] for d in ecology_fingerprint_diff(old, new)] == ["coop_threshold_step"]


def test_non_world_keys_do_not_change_the_fingerprint():
    """Learning rate / batch size describe how fast we get there, not what
    world agents live in -- changing them must not reset tripwires."""
    old = _fp(_cfg)
    new = _fp({**_cfg, "ppo_lr": 1e-2, "ppo_minibatch_size": 7})
    assert ecology_fingerprint_diff(old, new) == []


def test_schema_length_change_is_flagged_not_silently_truncated():
    """2026-09-28: a checkpoint saved under a shorter (pre-A0/A1-split)
    schema must not have its tail fields silently skipped by zip()'s
    truncate-to-shortest behavior -- the mismatch itself must be named."""
    old_short = _fp(_cfg)[:16]  # simulates a checkpoint saved before the 6 appended fields existed
    new = _fp(_cfg)
    diff = ecology_fingerprint_diff(old_short, new)
    assert len(diff) == 1
    assert "schema length changed" in diff[0][0]
    assert diff[0][1:] == (16.0, float(len(ECOLOGY_FINGERPRINT_LABELS)))


def test_new_fields_are_appended_not_inserted():
    """A checkpoint saved before the A0/A1 split has a fingerprint whose
    fields are a PREFIX of today's -- appended, not inserted -- so that if
    this check is ever bypassed, the pre-existing fields still line up."""
    old_len = len(ECOLOGY_FINGERPRINT_LABELS) - 6  # 5 accept_any slots + success_bar
    assert ECOLOGY_FINGERPRINT_LABELS[:old_len] == (
        "hearth_radius", "hearth_ramp_n_len",
        "hearth_ramp_n_0", "hearth_ramp_n_1", "hearth_ramp_n_2", "hearth_ramp_n_3", "hearth_ramp_n_4",
        "reward_hearth_deposit", "reward_hearth_completion", "hearth_decay_half_life_steps",
        "red_catch_radius", "coop_threshold_step",
        "reward_big_green_success", "reward_big_green_solo_penalty", "reward_small_blue", "r_coord",
    )


def test_vector_shape_is_fixed():
    fp = _fp(_cfg)
    assert fp.shape == (len(ECOLOGY_FINGERPRINT_LABELS),)
    assert _fp(_cfg, n=(1,), accept_any=(True,)).shape == fp.shape


if __name__ == "__main__":
    test_same_world_survives_float32_round_trip_with_no_diff()
    test_hearth_radius_change_is_detected_and_named()
    test_curriculum_schedule_change_is_detected()
    test_a0_a1_split_is_detected_and_named()
    test_accept_any_change_alone_is_detected()
    test_success_bar_change_is_detected()
    test_schema_length_change_is_flagged_not_silently_truncated()
    test_new_fields_are_appended_not_inserted()
    test_catch_param_change_is_detected()
    test_non_world_keys_do_not_change_the_fingerprint()
    test_vector_shape_is_fixed()
    print("OK: ecology fingerprint detects and names world changes (including the A0/A1 split "
          "and its accept_any flags), ignores non-world keys, and does not cry wolf across the "
          "float32 storage round-trip.")
