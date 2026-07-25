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


# ── 1b. Observation layout agreement ───────────────────────────────────────

def test_memory_slots_displaces_everything_after_it():
    """The network derives obs slice offsets from its own `memory_slots`.

    `causal_intervention.py` used to build the model with `memory_slots=0` while
    `init_population` created a 20-slot episodic buffer, so the environment
    emitted a 2731-D obs and the network sliced it as 1891-D. Shapes still lined
    up at every Dense layer, so nothing raised — the cultural-grid tokens were
    simply read out of the memory region and the last 840 dims were dropped.
    """
    from jax_sim.obs_layout import make_obs_layout

    kw = dict(
        signal_dim=40, symbol_dim=16, neighbor_k=6,
        local_cells=25, env_channels=15, own_state_dim=22,
    )
    without = make_obs_layout(memory_slots=0, **kw)
    with_mem = make_obs_layout(memory_slots=20, **kw)

    assert with_mem.total_dim == 2731, "config obs_dim is 2731"
    assert with_mem.total_dim - without.total_dim == 840

    # The corruption is specifically that everything *after* the memory block
    # shifts while everything before it stays put — which is why it is silent.
    assert without.loc_env_start == with_mem.loc_env_start
    assert without.loc_cult_fast_start != with_mem.loc_cult_fast_start
    assert with_mem.loc_cult_fast_start - without.loc_cult_fast_start == 840


def test_ate_tool_guards_obs_layout():
    """The ATE instrument must fail loudly on a layout mismatch, not measure
    through it."""
    src = (REPO / "tools" / "causal_intervention.py").read_text()
    assert "OBS LAYOUT MISMATCH" in src, "layout guard removed"

    # Check the code, not the prose: strip comment lines before matching, or the
    # explanatory comment above the model constructor trips this itself.
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "memory_slots=0" not in code, "hard-coded memory_slots=0 is back"
    # Both networks must take it from config — blue and red.
    assert code.count("memory_slots=_memory_slots") >= 2, (
        "blue and red must both derive memory_slots from config"
    )


# ── 1c. Craft contribution semantics ───────────────────────────────────────

def test_craft_contribution_and_free_riders():
    """Replicates the `main_jax` craft block to pin the 18.8 semantics.

    Scenario: four adjacent agents all press Craft against a 1 wood + 1 stone +
    1 flint recipe. A0/A1/A2 each hold one required material; A3 holds nothing —
    it is the free-rider whose existence made the 18.7 "receiver-necessity"
    ecology payable without listening.
    """
    import numpy as np

    gs = 128
    pos = np.array([[10, 10], [10, 11], [11, 10], [11, 11]])
    is_craft = np.array([True] * 4)
    inv = {
        "wood": np.array([1, 0, 0, 0]), "stone": np.array([0, 1, 0, 0]),
        "flint": np.array([0, 0, 1, 0]), "clay": np.zeros(4, int),
        "vine": np.array([0, 0, 0, 1]),  # A3 holds an OFF-RECIPE material
    }
    req = {"wood": 1, "stone": 1, "flint": 1, "clay": 0, "vine": 0}

    dx = np.abs(pos[:, 0:1] - pos[None, :, 0])
    dy = np.abs(pos[:, 1:2] - pos[None, :, 1])
    dx, dy = np.minimum(dx, gs - dx), np.minimum(dy, gs - dy)
    adjacent = np.maximum(dx, dy) <= 1

    craft_group = adjacent & is_craft[:, None] & is_craft[None, :]
    not_self = ~np.eye(4, dtype=bool)
    has_partner = (craft_group & not_self).any(1)

    grp = {k: (inv[k][None, :] * craft_group).sum(1) for k in inv}

    # Critical: an agent's OWN material must count toward its group total.
    # Removing the diagonal from `adjacent` to implement exclude_self would
    # break this and silently demand one extra body per craft.
    assert grp["wood"][0] == 1, "agent lost its own material from the group total"
    assert has_partner.all()

    success = is_craft & np.all([grp[k] >= req[k] for k in inv], axis=0)
    assert success.all(), "recipe is satisfied by the group"

    def paid(contrib_only, req_only):
        cons = {
            k: success & (inv[k] > 0) & ((req[k] > 0) if req_only else True)
            for k in inv
        }
        contributed = np.any([cons[k] for k in inv], axis=0)
        return (contributed if contrib_only else success), cons

    # 18.7 behaviour: the empty-handed free-rider is paid in full.
    p, _ = paid(contrib_only=False, req_only=False)
    assert p.tolist() == [True] * 4

    # 18.8 contributor_only_reward: A3 earns nothing.
    p, _ = paid(contrib_only=True, req_only=True)
    assert p.tolist() == [True, True, True, False]

    # 18.8 consume_required_only: A3's off-recipe vine is not destroyed.
    _, cons = paid(contrib_only=True, req_only=True)
    assert not cons["vine"][3], "off-recipe material was consumed"
    _, cons_legacy = paid(contrib_only=True, req_only=False)
    assert cons_legacy["vine"][3], "18.7 did consume off-recipe materials"


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
