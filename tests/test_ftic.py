from __future__ import annotations

import math

import numpy as np
import pytest

from termo.validation.ftic import ftic, within_cluster_ss

COMMON = {
    "wcss_saturated": 200.0,
    "saturated_states": 6,
    "n_obs": 100,
    "n_features": 4,
    "prior_states": 3,
    "prior_jumps": 2.5,
}


def test_wcss_known_value() -> None:
    features = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0], [10.0, 3.0]])
    labels = np.array([0, 0, 1, 1])
    # state 0: mean (1, 0) -> 1 + 1 ; state 1: mean (10, 2) -> 1 + 1
    assert within_cluster_ss(features, labels) == pytest.approx(4.0)


def test_ftic_hand_computed() -> None:
    # a_T = log(log 100) * log 4 = 1.527180 * 1.386294 = 2.117121
    # M   = 2 * (4 + 2.5) + 3 * (5 - 2.5) = 20.5
    # (300 - 200 + 2.117121 * 20.5) / 100 = 1.434010
    # 2 * (log 2 - log 6) = -2.197225
    assert ftic(2, 300.0, 5, **COMMON) == pytest.approx(-0.763215, abs=1e-5)


def test_ftic_penalizes_jumps_and_states() -> None:
    base = ftic(3, 250.0, 5, **COMMON)
    assert ftic(3, 250.0, 6, **COMMON) > base
    assert ftic(4, 250.0, 5, **COMMON) > base
    assert ftic(3, 240.0, 5, **COMMON) < base


def test_ftic_trades_one_jump_for_a_fixed_amount_of_fit() -> None:
    # One more jump costs a_T * K0 in WCSS units, whatever K is.
    cost = math.log(math.log(100)) * math.log(4) * 3
    for n_states in (2, 3, 4):
        assert ftic(n_states, 250.0 - cost, 6, **COMMON) == pytest.approx(
            ftic(n_states, 250.0, 5, **COMMON)
        )
