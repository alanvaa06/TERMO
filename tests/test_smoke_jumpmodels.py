"""The properties of `jumpmodels` that the rest of the project relies on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from jumpmodels.jump import JumpModel
from sklearn.metrics import adjusted_rand_score


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1, 0, 2, 1, 0], 300)
    centers = np.array([[0.0] * 6, [2.0] * 6, [-2.0] * 6])
    values = centers[truth] + rng.normal(size=(len(truth), 6))
    index = pd.bdate_range("2000-01-03", periods=len(truth))
    return pd.DataFrame(values, index=index, columns=[f"f{i}" for i in range(6)]), truth


def test_recovers_planted_regimes(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, truth = planted
    model = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    assert adjusted_rand_score(truth, np.asarray(model.labels_)) > 0.95


def test_fit_is_deterministic(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, _ = planted
    first = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    second = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    assert (np.asarray(first.labels_) == np.asarray(second.labels_)).all()


def test_online_prediction_ignores_the_future(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, _ = planted
    model = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    altered = features.copy()
    altered.iloc[1000:] = 0.0
    before = np.asarray(model.predict_online(features))[:1000]
    after = np.asarray(model.predict_online(altered))[:1000]
    assert (before == after).all()


def test_rejects_missing_values(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, _ = planted
    broken = features.copy()
    broken.iloc[5, 0] = np.nan
    with pytest.raises(AssertionError):
        JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(broken)
