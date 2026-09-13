"""
Pins the causal_intervention.py pass bar as two-sided (Cam's ruling,
docs/WILL_RESTART_SEP2026.md item 4). The roadmap's pre-registered bar is
`p < 0.05 AND |Delta| > 0.05`. A directional `mean_delta > 0.05` check is a
DIFFERENT hypothesis (suppression only) and structurally cannot detect an
intervention that INCREASES the target action's probability.
"""

from tools.causal_intervention import ate_passes_bar


def test_passes_on_significant_positive_delta():
    assert ate_passes_bar(p_val=0.01, mean_delta=0.10) is True


def test_passes_on_significant_negative_delta():
    """The case a directional-only check would have missed entirely."""
    assert ate_passes_bar(p_val=0.01, mean_delta=-0.10) is True


def test_fails_on_small_delta_even_if_significant():
    assert ate_passes_bar(p_val=0.001, mean_delta=0.02) is False


def test_fails_on_large_delta_if_not_significant():
    assert ate_passes_bar(p_val=0.20, mean_delta=0.10) is False


def test_fails_at_zero():
    assert ate_passes_bar(p_val=0.001, mean_delta=0.0) is False


if __name__ == "__main__":
    test_passes_on_significant_positive_delta()
    test_passes_on_significant_negative_delta()
    test_fails_on_small_delta_even_if_significant()
    test_fails_on_large_delta_if_not_significant()
    test_fails_at_zero()
    print("OK: ATE pass bar is two-sided.")
