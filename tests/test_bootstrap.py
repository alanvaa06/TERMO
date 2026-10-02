from __future__ import annotations

import numpy as np
import pytest

from termo.validation.bootstrap import (
    chi2_statistic,
    circular_shift_test,
    moving_block_indices,
    paired_excess_eta_difference,
)
from termo.validation.metrics import eta_squared


def persistent_groups(n_rows: int, run: int) -> np.ndarray:
    return (np.arange(n_rows) // run) % 2


def test_block_indices_are_contiguous_blocks() -> None:
    indices = moving_block_indices(100, 10, np.random.default_rng(0))
    assert indices.shape == (100,)
    assert indices.min() >= 0 and indices.max() < 100
    assert (np.diff(indices.reshape(10, 10), axis=1) == 1).all()


def test_block_indices_reject_bad_block() -> None:
    with pytest.raises(ValueError):
        moving_block_indices(10, 11, np.random.default_rng(0))


def test_paired_difference_detects_the_informative_grouping() -> None:
    rng = np.random.default_rng(0)
    informative = persistent_groups(400, 20)
    noise_groups = persistent_groups(400, 7)
    values = np.where(informative == 1, 10.0, -10.0) + rng.normal(scale=5.0, size=400)
    result = paired_excess_eta_difference(values, informative, noise_groups, 26, 300, seed=0)
    assert result.point > 0.5
    assert 0.0 < result.low < result.point < result.high


def test_paired_difference_of_a_grouping_with_itself_is_zero() -> None:
    rng = np.random.default_rng(0)
    groups = persistent_groups(300, 15)
    values = rng.normal(size=300)
    result = paired_excess_eta_difference(values, groups, groups, 26, 100, seed=0)
    assert (result.point, result.low, result.high) == (0.0, 0.0, 0.0)


def test_more_states_get_no_head_start() -> None:
    """Five unrelated states against two unrelated states: no advantage after the correction."""
    raw, corrected = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=300)
        five = (rng.integers(0, 5) + np.cumsum(rng.random(300) < 0.05)) % 5
        two = (rng.integers(0, 2) + np.cumsum(rng.random(300) < 0.05)) % 2
        raw.append(eta_squared(values, five) - eta_squared(values, two))
        corrected.append(paired_excess_eta_difference(values, five, two, 26, 20, seed).point)
    assert np.mean(raw) > 0.01
    assert abs(float(np.mean(corrected))) < 0.01


def test_paired_difference_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    a, b = persistent_groups(200, 10), persistent_groups(200, 25)
    assert paired_excess_eta_difference(
        values, a, b, 20, 50, seed=7
    ) == paired_excess_eta_difference(values, a, b, 20, 50, seed=7)


def test_chi2_known_value() -> None:
    groups = np.array([0] * 20 + [1] * 20)
    signs = np.array([1] * 15 + [0] * 5 + [1] * 5 + [0] * 15)
    # expected 10 in every cell; (5^2 / 10) * 4 = 10
    assert chi2_statistic(groups, signs) == pytest.approx(10.0)
    assert chi2_statistic(groups, np.ones(40, dtype=int)) == 0.0


def test_circular_shift_finds_real_dependence() -> None:
    rng = np.random.default_rng(0)
    # Runs of random length: a strictly periodic series would match itself after a shift.
    groups = (np.cumsum(rng.random(400) < 0.05) % 2).astype(int)
    signs = np.where(rng.random(400) < 0.9, groups, 1 - groups)
    result = circular_shift_test(groups, signs)
    assert result.p_value < 0.01


def test_circular_shift_is_not_fooled_by_persistence_alone() -> None:
    """Two persistent but unrelated series: the test must keep its nominal error rate.

    A naive chi-square test rejects most of these pairs.
    """
    rejections, naive_rejections = 0, 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        # Random starting state: otherwise both series begin aligned at zero.
        groups = ((rng.integers(0, 2) + np.cumsum(rng.random(300) < 0.03)) % 2).astype(int)
        signs = ((rng.integers(0, 2) + np.cumsum(rng.random(300) < 0.03)) % 2).astype(int)
        result = circular_shift_test(groups, signs)
        rejections += result.p_value < 0.01
        naive_rejections += result.statistic > 6.63  # 1% point of chi-square with 1 d.o.f.
    # Nominal: 3 of 300. Leaving out the small shifts gives about 15.
    assert rejections <= 7
    assert naive_rejections > 100


def test_circular_shift_smallest_p_value_is_one_over_n() -> None:
    groups = np.array([0] * 60 + [1] * 60)
    result = circular_shift_test(groups, groups)
    assert result.p_value == pytest.approx(2 / 120)  # the mirror shift ties with the observed one


def test_circular_shift_needs_enough_rows() -> None:
    with pytest.raises(ValueError):
        circular_shift_test(np.zeros(50, dtype=int), np.zeros(50, dtype=int))
