"""Resampling that respects autocorrelation: blocks, never single weeks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from termo.validation.metrics import chance_eta_squared, eta_squared

CONFIDENCE_TAILS = (2.5, 97.5)
MIN_ROWS_INDEPENDENCE = 100  # so that p < 0.01 is reachable


@dataclass(frozen=True)
class EtaDifference:
    point: float
    low: float
    high: float


@dataclass(frozen=True)
class IndependenceTest:
    statistic: float
    p_value: float


def moving_block_indices(n_rows: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one moving-block bootstrap resample of length n_rows."""
    if not 1 <= block <= n_rows:
        raise ValueError("need 1 <= block <= n_rows")
    n_blocks = -(-n_rows // block)
    starts = rng.integers(0, n_rows - block + 1, size=n_blocks)
    indices: np.ndarray = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_rows]
    return indices


def paired_excess_eta_difference(
    values: np.ndarray,
    groups_a: np.ndarray,
    groups_b: np.ndarray,
    block: int,
    n_resamples: int,
    seed: int,
) -> EtaDifference:
    """Excess eta2 of A minus excess eta2 of B, with a 95% interval.

    The interval comes from resampling the same blocks of rows for both groupings.
    It is then shifted by the difference of their chance levels, so that a grouping
    with more states gets no head start.
    """
    rng = np.random.default_rng(seed)
    raw_point = eta_squared(values, groups_a) - eta_squared(values, groups_b)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = moving_block_indices(len(values), block, rng)
        draws[i] = eta_squared(values[rows], groups_a[rows]) - eta_squared(
            values[rows], groups_b[rows]
        )
    low, high = np.percentile(draws, CONFIDENCE_TAILS)
    head_start = chance_eta_squared(values, groups_a) - chance_eta_squared(values, groups_b)
    return EtaDifference(
        point=raw_point - head_start, low=float(low) - head_start, high=float(high) - head_start
    )


def chi2_statistic(groups: np.ndarray, signs: np.ndarray) -> float:
    """Pearson chi-square of the groups x signs table; 0 when the table is degenerate."""
    _, group_codes = np.unique(groups, return_inverse=True)
    _, sign_codes = np.unique(signs, return_inverse=True)
    table = np.zeros((group_codes.max() + 1, sign_codes.max() + 1))
    np.add.at(table, (group_codes, sign_codes), 1.0)
    if min(table.shape) < 2:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    return float(((table - expected) ** 2 / expected).sum())


def circular_shift_test(groups: np.ndarray, signs: np.ndarray) -> IndependenceTest:
    """Is the group/sign association larger than under every other time offset?

    Shifting the group series circularly keeps its autocorrelation and breaks its
    alignment with the signs. Every non-zero shift is used: leaving out the small
    ones makes the test reject too often, because those are the shifts most likely
    to match a large observed statistic. The smallest possible p-value is 1 / n_rows.
    """
    n_rows = len(groups)
    if n_rows < MIN_ROWS_INDEPENDENCE:
        raise ValueError(f"need at least {MIN_ROWS_INDEPENDENCE} rows")
    observed = chi2_statistic(groups, signs)
    exceed = sum(
        chi2_statistic(np.roll(groups, shift), signs) >= observed for shift in range(1, n_rows)
    )
    return IndependenceTest(statistic=observed, p_value=(1 + exceed) / n_rows)
