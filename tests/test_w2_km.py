"""python -m pytest tests/test_w2_km.py --noconftest -q"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2.km import km_median  # noqa: E402


def test_no_censoring_is_lower_median():
    t = np.array([5, 1, 3, 2, 4]); assert km_median(t, np.ones(5, bool)) == (3.0, True)
    t = np.array([1, 2, 3, 4]); assert km_median(t, np.ones(4, bool)) == (2.0, True)        # S(2) = 0.5


def test_textbook_example():
    # times 3,4+,5,6+,8,9 (+ = censored): S(3)=5/6, S(5)=5/6*3/4=0.625, S(8)=0.625*1/2=0.3125 -> median 8
    t = np.array([3, 4, 5, 6, 8, 9]); e = np.array([1, 0, 1, 0, 1, 1], bool)
    assert km_median(t, e) == (8.0, True)


def test_censoring_raises_median_vs_naive():
    # censored at the very end (longer than every death): KM agrees with the ranking -> median 50 (S(50) = 0.5)
    t = np.array([10, 20, 30, 40, 50, 100, 100, 100, 100, 100]); e = t < 100
    assert km_median(t, e) == (50.0, True)
    # the retune case: late-born lives cut off SHORT (birth near the horizon). Deaths at 100..500, censored at 20..60
    t = np.array([100, 200, 300, 400, 500, 20, 30, 40, 50, 60]); e = np.array([1] * 5 + [0] * 5, bool)
    naive = np.median(t); km, reached = km_median(t, e)
    assert naive == 80.0 and km == 300.0 and reached          # the naive median is dragged down by the cut-off lives


def test_all_censored():
    assert km_median([5, 6], [False, False]) == (6.0, False)
