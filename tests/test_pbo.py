from __future__ import annotations

import numpy as np
import pytest

from termo.validation.metrics import eta_squared
from termo.validation.pbo import block_stats, effective_n, eta_from_stats, pbo_cscv


def test_eta_from_stats_matches_the_direct_computation() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=240)
    groups = rng.integers(0, 3, size=240)
    stats = block_stats(values, groups, n_states=5, n_blocks=8)  # two states never used
    assert stats.shape == (8, 5, 3)
    assert stats[..., 0].sum() == 240
    assert eta_from_stats(stats.sum(axis=0)) == pytest.approx(eta_squared(values, groups))
    first_half = eta_from_stats(stats[:4].sum(axis=0))
    assert first_half == pytest.approx(eta_squared(values[:120], groups[:120]))


def test_pbo_is_low_when_one_configuration_is_really_better() -> None:
    rng = np.random.default_rng(0)
    truth = (np.arange(480) // 20) % 2
    values = np.where(truth == 1, 5.0, -5.0) + rng.normal(scale=3.0, size=480)
    good = block_stats(values, truth, 2, 8)
    noise = [block_stats(values, rng.integers(0, 2, size=480), 2, 8) for _ in range(9)]
    assert pbo_cscv(np.stack([good, *noise])) == 0.0


def test_pbo_is_near_half_when_every_configuration_is_noise() -> None:
    estimates = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=480)
        stats = [block_stats(values, rng.integers(0, 2, size=480), 2, 8) for _ in range(10)]
        estimates.append(pbo_cscv(np.stack(stats)))
    assert 0.3 < float(np.mean(estimates)) < 0.7


def test_pbo_needs_an_even_number_of_blocks() -> None:
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((3, 5, 2, 3)))


def test_pbo_of_identical_configurations_is_a_coin_flip() -> None:
    """Choosing among copies of one labeling is no selection at all: neither 0 nor 1."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=480)
    one = block_stats(values, rng.integers(0, 2, size=480), 2, 8)
    assert pbo_cscv(np.stack([one] * 6)) == pytest.approx(0.5)


def test_pbo_does_not_reward_a_duplicated_winner() -> None:
    rng = np.random.default_rng(1)
    truth = (np.arange(480) // 20) % 2
    values = np.where(truth == 1, 5.0, -5.0) + rng.normal(scale=3.0, size=480)
    good = block_stats(values, truth, 2, 8)
    noise = [block_stats(values, rng.integers(0, 2, size=480), 2, 8) for _ in range(6)]
    # The real winner twice: it ties with its copy, and still ranks above every noise config.
    assert pbo_cscv(np.stack([good, good, *noise])) == 0.0


def test_effective_n_counts_distinct_labelings() -> None:
    rng = np.random.default_rng(0)
    a = (np.arange(300) // 30) % 2
    b = (np.arange(300) // 50) % 3
    near_a = a.copy()
    near_a[:3] = 1 - near_a[:3]
    relabelled_a = 1 - a
    noise = rng.integers(0, 2, size=300)
    assert effective_n([a, near_a, relabelled_a], cut=0.2) == 1
    assert effective_n([a, near_a, b, noise], cut=0.2) == 3
    assert effective_n([a], cut=0.2) == 1
