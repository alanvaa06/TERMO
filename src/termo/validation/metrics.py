"""Small, model-free measurements used by every test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

BP_PER_PERCENT = 100.0
VARIANCE_FLOOR = 1e-12  # relative; below this the values are constant up to rounding


def adjusted_rand(first: np.ndarray, second: np.ndarray) -> float:
    return float(adjusted_rand_score(first, second))


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance of `values` explained by the group means (0 to 1)."""
    if len(values) == 0:
        return 0.0
    grand_mean = values.mean()
    total = float(((values - grand_mean) ** 2).sum())
    if total <= VARIANCE_FLOOR * max(1.0, float((values**2).sum())):
        return 0.0  # constant values: nothing to explain
    between = 0.0
    for group in np.unique(groups):
        member = values[groups == group]
        between += len(member) * float((member.mean() - grand_mean) ** 2)
    return between / total


def chance_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Eta-squared that the same group series gets by luck: its mean over every time shift.

    Persistent groups explain some variance of an autocorrelated series by chance,
    and more groups explain more. Shifting the groups in time keeps their number and
    their persistence and breaks any real link with `values`.
    """
    n_rows = len(values)
    if n_rows < 2:
        return 0.0
    shifted = [eta_squared(values, np.roll(groups, shift)) for shift in range(1, n_rows)]
    return float(np.mean(shifted))


def excess_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Eta-squared above what luck alone gives. This is the separation measure."""
    return eta_squared(values, groups) - chance_eta_squared(values, groups)


def forward_change_bp(yields: pd.Series, horizon: int) -> pd.Series:
    """Change from t to t+horizon rows, in basis points; NaN where the future is not in the data."""
    return (yields.shift(-horizon) - yields) * BP_PER_PERCENT


def weekly_last(series: pd.Series) -> pd.Series:
    """The last available row of each Monday-to-Friday week."""
    return series.groupby(series.index.to_period("W-FRI")).tail(1)


def run_lengths(labels: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive runs as (state, length), in order."""
    runs: list[tuple[int, int]] = []
    if len(labels) == 0:
        return runs
    change_points = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(labels)]))
    for start, end in zip(starts, ends, strict=True):
        runs.append((int(labels[start]), int(end - start)))
    return runs


def median_durations(labels: np.ndarray, n_states: int) -> dict[int, float]:
    """Median run length per state; NaN for a state that never appears."""
    by_state: dict[int, list[int]] = {state: [] for state in range(n_states)}
    for state, length in run_lengths(labels):
        by_state[state].append(length)
    return {
        state: float(np.median(lengths)) if lengths else float("nan")
        for state, lengths in by_state.items()
    }


def count_jumps(labels: np.ndarray) -> int:
    return int((labels[1:] != labels[:-1]).sum())
