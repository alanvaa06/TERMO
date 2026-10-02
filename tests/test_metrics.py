from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from termo.validation.metrics import (
    adjusted_rand,
    chance_eta_squared,
    count_jumps,
    eta_squared,
    excess_eta_squared,
    forward_change_bp,
    median_durations,
    run_lengths,
    shift_excess_eta_squared,
    weekly_last,
)


def test_adjusted_rand_ignores_the_names() -> None:
    assert adjusted_rand(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == pytest.approx(1.0)


def test_eta_squared_known_value() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])
    groups = np.array([0, 0, 1, 1])
    # group means 2 and 6, grand mean 4: between = 16, total = 20
    assert eta_squared(values, groups) == pytest.approx(0.8)


def test_eta_squared_edge_cases() -> None:
    assert eta_squared(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0])) == 0.0
    assert eta_squared(np.array([2.0, 2.0]), np.array([0, 1])) == 0.0
    assert eta_squared(np.array([]), np.array([])) == 0.0
    assert eta_squared(np.array([1.0, 1.0, 9.0, 9.0]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)


@given(
    arrays(np.float64, 30, elements=st.floats(min_value=-100.0, max_value=100.0)),
    arrays(np.int64, 30, elements=st.integers(min_value=0, max_value=3)),
)
def test_eta_squared_is_a_share(values: np.ndarray, groups: np.ndarray) -> None:
    assert -1e-9 <= eta_squared(values, groups) <= 1.0 + 1e-9


def persistent(rng: np.random.Generator, n_rows: int, n_states: int) -> np.ndarray:
    return (rng.integers(0, n_states) + np.cumsum(rng.random(n_rows) < 0.05)) % n_states


def rare_volatile_group(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """State 1 covers about 8% of the rows, in runs of about 8."""
    leave, share = 1.0 / 8.0, 0.08
    enter = leave * share / (1.0 - share)
    out = np.zeros(n_rows, dtype=int)
    draws = rng.random(n_rows)
    for t in range(1, n_rows):
        out[t] = (draws[t] > leave) if out[t - 1] == 1 else (draws[t] < enter)
    return out


def overlapping_moves(rng: np.random.Generator, vol: np.ndarray) -> np.ndarray:
    """4-row forward sums of shocks with the given volatility: like 4-week yield changes."""
    shocks = rng.normal(size=len(vol) + 4) * np.concatenate([vol, vol[-4:]])
    total = np.concatenate([[0.0], np.cumsum(shocks)])
    return total[5 : 5 + len(vol)] - total[1 : len(vol) + 1]


BLOCK, DRAWS = 26, 200


def test_chance_eta_grows_with_the_number_of_states() -> None:
    """Unrelated persistent groups explain variance by luck, and more groups explain more."""
    two, five = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        values = overlapping_moves(rng, np.ones(400))
        two.append(chance_eta_squared(values, persistent(rng, 400, 2), BLOCK, DRAWS, seed))
        five.append(chance_eta_squared(values, persistent(rng, 400, 5), BLOCK, DRAWS, seed))
    assert 0.0 < np.mean(two) < np.mean(five)


def test_excess_eta_is_zero_on_average_without_a_real_link() -> None:
    excess = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=300)
        excess.append(excess_eta_squared(values, persistent(rng, 300, 5), BLOCK, DRAWS, seed))
    assert abs(float(np.mean(excess))) < 0.01


def test_a_rare_volatile_group_earns_no_separation() -> None:
    """A group that only marks the volatile weeks knows nothing about direction.

    Sliding the groups in time (the older chance level) leaves it with credit;
    flipping signs in blocks does not.
    """
    flipped, slid = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        hot = rare_volatile_group(rng, 800)
        values = overlapping_moves(rng, np.where(hot == 1, 4.0, 1.0))
        flipped.append(excess_eta_squared(values, hot, BLOCK, DRAWS, seed))
        slid.append(shift_excess_eta_squared(values, hot))
    assert np.mean(flipped) < 0.004
    assert np.mean(slid) > 0.008


def test_excess_eta_keeps_a_real_link() -> None:
    rng = np.random.default_rng(0)
    groups = persistent(rng, 400, 2)
    values = np.where(groups == 1, 3.0, -3.0) + rng.normal(size=400)
    assert eta_squared(values, groups) > 0.8
    assert excess_eta_squared(values, groups, BLOCK, DRAWS, 0) > 0.7


def test_chance_eta_is_reproducible_and_validates_its_arguments() -> None:
    rng = np.random.default_rng(0)
    values, groups = rng.normal(size=200), persistent(rng, 200, 3)
    first = chance_eta_squared(values, groups, BLOCK, 50, seed=3)
    assert first == chance_eta_squared(values, groups, BLOCK, 50, seed=3)
    assert first != chance_eta_squared(values, groups, BLOCK, 50, seed=4)
    assert chance_eta_squared(np.array([1.0]), np.array([0]), BLOCK, 50, seed=0) == 0.0
    with pytest.raises(ValueError):
        chance_eta_squared(values, groups, 0, 50, seed=0)
    with pytest.raises(ValueError):
        chance_eta_squared(values, groups, BLOCK, 0, seed=0)


def test_shift_excess_known_value() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])
    groups = np.array([0, 0, 1, 1])
    # eta2 is 0.8; sliding the groups by 1, 2 and 3 rows gives 0, 0.8 and 0
    assert shift_excess_eta_squared(values, groups) == pytest.approx(0.8 - 0.8 / 3)


def test_forward_change_is_in_basis_points_and_blank_at_the_end() -> None:
    yields = pd.Series([4.00, 4.10, 4.05, 4.30])
    change = forward_change_bp(yields, 2)
    assert change.iloc[0] == pytest.approx(5.0)
    assert change.iloc[1] == pytest.approx(20.0)
    assert change.iloc[2:].isna().all()


def test_weekly_last_takes_the_last_available_day() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)  # Mon 1 Jan .. Tue 16 Jan
    series = pd.Series(range(12), index=index).drop(pd.Timestamp("2024-01-12"))  # no Friday
    weekly = weekly_last(series)
    assert list(weekly.index.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-11", "2024-01-16"]
    assert weekly.tolist() == [4, 8, 11]


def test_run_lengths_and_durations() -> None:
    labels = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 2])
    assert run_lengths(labels) == [(0, 3), (1, 2), (0, 1), (2, 4)]
    assert median_durations(labels, 4) == {
        0: 2.0,
        1: 2.0,
        2: 4.0,
        3: pytest.approx(math.nan, nan_ok=True),
    }
    assert count_jumps(labels) == 3
    assert run_lengths(np.array([], dtype=int)) == []


@given(st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=100))
def test_run_lengths_rebuild_the_series(raw: list[int]) -> None:
    labels = np.array(raw)
    runs = run_lengths(labels)
    rebuilt = np.concatenate([np.full(length, state) for state, length in runs])
    assert (rebuilt == labels).all()
    assert len(runs) == count_jumps(labels) + 1
