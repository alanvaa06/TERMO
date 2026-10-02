"""Small, model-free measurements used by every test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

BP_PER_PERCENT = 100.0
VARIANCE_FLOOR = 1e-12  # relative; below this the values are constant up to rounding
SIGNS = np.array([-1.0, 1.0])


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


def chance_eta_squared(
    values: np.ndarray, groups: np.ndarray, block: int, n_draws: int, seed: int
) -> float:
    """Eta-squared the same groups get by luck, once the direction of the moves is erased.

    Persistent groups explain some variance of an autocorrelated series by chance, more
    groups explain more, and a group that sits on the volatile weeks explains more still.
    Each draw gives a random sign to whole blocks of the centred values: their size stays
    where it was, next to the same groups, and any real link with direction is broken.
    """
    n_rows = len(values)
    if n_rows < 2:
        return 0.0
    if block < 1 or n_draws < 1:
        raise ValueError("block and n_draws must be positive")
    rng = np.random.default_rng(seed)
    centred = values - values.mean()
    n_blocks = -(-n_rows // block) + 1
    draws = np.empty(n_draws)
    for i in range(n_draws):
        offset = int(rng.integers(0, block))
        signs = np.repeat(rng.choice(SIGNS, size=n_blocks), block)[offset : offset + n_rows]
        draws[i] = eta_squared(centred * signs, groups)
    return float(draws.mean())


def excess_eta_squared(
    values: np.ndarray, groups: np.ndarray, block: int, n_draws: int, seed: int
) -> float:
    """Eta-squared above what luck alone gives. This is the separation measure."""
    return eta_squared(values, groups) - chance_eta_squared(values, groups, block, n_draws, seed)


def shift_excess_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Sensitivity check only: chance level from sliding the groups in time.

    Sliding the groups breaks their link with volatility as well as with direction, so a
    rare group that sits on the volatile weeks keeps some credit it did not earn. It is
    reported next to the separation measure and never decides anything.
    """
    n_rows = len(values)
    if n_rows < 2:
        return 0.0
    shifted = [eta_squared(values, np.roll(groups, shift)) for shift in range(1, n_rows)]
    return eta_squared(values, groups) - float(np.mean(shifted))


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
