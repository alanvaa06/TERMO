from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from termo.features.velocity import smoothed_change
from termo.features.volatility import ewm_vol


def test_smoothed_change_of_a_straight_line() -> None:
    line = pd.Series(np.arange(100, dtype=float) * 2.0)  # +2 per day
    change = smoothed_change(line, horizon=21, delta=5)
    # mean of 2*16, 2*21 and 2*26
    assert change.iloc[-1] == pytest.approx(42.0)
    assert change.iloc[:26].isna().all()
    assert change.iloc[26:].notna().all()


def test_old_move_leaving_one_window_is_diluted() -> None:
    step = pd.Series(np.where(np.arange(80) >= 30, 30.0, 0.0))  # +30 on day 30
    change = smoothed_change(step, horizon=21, delta=5)
    assert change.iloc[30] == pytest.approx(30.0)  # today's move counts in full
    assert change.iloc[45] == pytest.approx(30.0)  # still inside the three windows
    assert change.iloc[48] == pytest.approx(20.0)  # left the 16-day window only
    assert change.iloc[53] == pytest.approx(10.0)  # left the 21-day window too
    assert change.iloc[60] == pytest.approx(0.0)  # left all of them


def test_smoothed_change_rejects_bad_windows() -> None:
    with pytest.raises(ValueError):
        smoothed_change(pd.Series([1.0, 2.0]), horizon=5, delta=5)


@given(st.floats(min_value=0.1, max_value=50.0), st.floats(min_value=1.0, max_value=200.0))
def test_vol_of_constant_absolute_moves_is_that_move(move: float, halflife: float) -> None:
    changes = pd.Series(np.tile([move, -move], 50))
    assert ewm_vol(changes, halflife).iloc[-1] == pytest.approx(move)


def test_vol_reacts_faster_with_a_short_halflife() -> None:
    changes = pd.Series([1.0] * 200 + [10.0] * 5)
    assert ewm_vol(changes, 20).iloc[-1] > ewm_vol(changes, 120).iloc[-1]


def test_vol_uses_the_past_only() -> None:
    rng = np.random.default_rng(1)
    changes = pd.Series(rng.normal(size=300))
    altered = changes.copy()
    altered.iloc[200:] = 99.0
    assert np.allclose(ewm_vol(changes, 60).iloc[:200], ewm_vol(altered, 60).iloc[:200])


def test_vol_rejects_non_positive_halflife() -> None:
    with pytest.raises(ValueError):
        ewm_vol(pd.Series([1.0, 2.0]), 0)
