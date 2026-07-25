"""Regression tests for the Phase 18.8 receiver-necessity ecology.

Two independent things are pinned here.

1. **Wire-slot arithmetic.** `tools/causal_intervention.py` used to slice the
   40-D wire with a uniform 8-dim stride (offsets 8/16/24). The real slots are
   12/8/12 at offsets 8/20/28. A slot-0 or slot-2 intervention therefore tried
   to write a 12-vector into an 8-wide slice and raised; slot 1 had matching
   shapes but wrote to the tail of slot 0 plus the head of slot 1 and ran
   silently wrong. This is the same *class* of bug as the Phase 18.5 `loss_vq`
   index divergence, and it has now bitten twice, so it gets a test.

2. **Phase 18.8 defaults are behaviour-preserving.** Every knob in the
   `phase18_8_receiver_necessity` config block must ship in the state that
   reproduces Phase 18.7 exactly, so that pulling this branch onto the live
   cluster changes nothing until someone deliberately edits config.yaml.

Both run in milliseconds with no GPU, no checkpoint and no network.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


# ── 1. Wire-slot arithmetic ────────────────────────────────────────────────

def test_signal_slots_match_codebook_widths():
    """SIGNAL_SLOTS must describe the 12/8/12 codebooks, contiguously, after
    the 8-dim (amputated) continuous bypass."""
    from jax_sim.obs_layout import SIGNAL_SLOTS

    assert SIGNAL_SLOTS["cont"] == slice(0, 8)
    assert SIGNAL_SLOTS["slot_0"] == slice(8, 20)     # 12 wide
    assert SIGNAL_SLOTS["slot_1"] == slice(20, 28)    # 8 wide
    assert SIGNAL_SLOTS["slot_2"] == slice(28, 40)    # 12 wide

    widths = [
        SIGNAL_SLOTS[f"slot_{i}"].stop - SIGNAL_SLOTS[f"slot_{i}"].start
        for i in range(3)
    ]
    assert widths == [12, 8, 12], f"slot widths drifted: {widths}"

    # Contiguous, and the whole 40-D wire is accounted for.
    assert SIGNAL_SLOTS["cont"].stop == SIGNAL_SLOTS["slot_0"].start
    assert SIGNAL_SLOTS["slot_0"].stop == SIGNAL_SLOTS["slot_1"].start
    assert SIGNAL_SLOTS["slot_1"].stop == SIGNAL_SLOTS["slot_2"].start
    assert SIGNAL_SLOTS["slot_2"].stop == 40


def test_causal_intervention_uses_layout_not_uniform_stride():
    """The ATE instrument must derive its slices from SIGNAL_SLOTS.

    Pins the actual defect: a uniform `8 + i*8` stride silently aliases slot 1
    onto the slot-0/slot-1 boundary.
    """
    src = (REPO / "tools" / "causal_intervention.py").read_text()

    assert "SLOT_SLICES" in src, "ATE instrument no longer uses the shared layout"
    assert "8 + slot_idx*8" not in src, "uniform-stride slot arithmetic is back"

    from jax_sim.obs_layout import SIGNAL_SLOTS

    # The buggy stride and the correct layout disagree on slots 0 and 2 by
    # width, and on slot 1 by offset. If they ever agreed, this test would be
    # vacuous — assert they do not.
    for i in range(3):
        buggy = slice(8 + i * 8, 16 + i * 8)
        real = SIGNAL_SLOTS[f"slot_{i}"]
        assert buggy != real, f"slot {i}: buggy stride coincides with real layout"


def test_slot_write_shapes_are_consistent():
    """A codebook row for slot i must exactly fill slot i's wire slice."""
    from jax_sim.obs_layout import SIGNAL_SLOTS

    codebook_widths = [12, 8, 12]  # network_jax.py codebook_{0,1,2}
    for i, w in enumerate(codebook_widths):
        s = SIGNAL_SLOTS[f"slot_{i}"]
        assert s.stop - s.start == w, (
            f"slot {i}: wire slice is {s.stop - s.start} wide but codebook_{i} "
            f"emits {w} — an intervention would raise or alias"
        )


# ── 2. Phase 18.8 defaults ─────────────────────────────────────────────────

def _p188():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    assert "phase18_8_receiver_necessity" in cfg, "18.8 config block missing"
    return cfg["phase18_8_receiver_necessity"]


def test_receiver_necessity_ships_disabled():
    """Pulling this branch must not change the live ecology by itself.

    Enabling any of these is an ecology change requiring Cam sign-off (Rule 12).
    If this test fails, someone shipped an ecology change as a default — that is
    the failure mode this test exists to catch, not a reason to update the test.
    """
    p = _p188()
    for flag in (
        "strict_asymmetry",
        "contributor_only_reward",
        "consume_required_only",
        "exclude_self_from_craft_group",
    ):
        assert p[flag] is False, f"{flag} must default to Phase 18.7 behaviour"

    assert p["recipe_rotation_steps"] == 1000, "18.7 rotation period is 1000 steps"
    assert p["sender_credit_frac"] == 0.0, "sender credit is a reward change — off by default"


def test_recipe_rotation_bites_only_below_rollout_length():
    """Document the load-bearing relationship as an executable assertion.

    A recipe that outlives a PPO rollout can be memorised in the weights, which
    is what made the 18.7 `can_see_recipe` bit decorative. This test does not
    demand the fix — it pins the arithmetic so the reason is not lost.
    """
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    rollout = int(cfg.get("ppo_rollout_steps", 512))
    rotation = int(_p188()["recipe_rotation_steps"])

    memorisable = rotation >= rollout
    assert memorisable, (
        "recipe_rotation_steps dropped below ppo_rollout_steps — this is the "
        "Phase 18.8 step-3 change and it is an ecology change. Intentional? "
        "Then update this test and record the Cam sign-off in THRONG.md."
    )
