from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from termo.regime.model import RegimeFitter, jump_fitter, kmeans_fitter
from termo.validation.metrics import count_jumps


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    truth = np.repeat([0, 1, 0, 1, 0], 200)
    centers = np.array([[-1.0] * 4, [1.0] * 4])
    values = centers[truth] + rng.normal(size=(len(truth), 4))
    index = pd.bdate_range("2000-01-03", periods=len(truth))
    return pd.DataFrame(values, index=index, columns=list("abcd")), truth


@pytest.mark.parametrize("fitter", [jump_fitter(2, 30.0), kmeans_fitter(2)])
def test_models_share_one_interface(
    fitter: RegimeFitter, planted: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, truth = planted
    model = fitter(features)
    assert model.n_states == 2
    for labels in (
        model.insample_labels(),
        model.online_labels(features),
        model.full_labels(features),
    ):
        assert labels.shape == (len(features),)
        assert labels.dtype.kind == "i"
        assert set(np.unique(labels)) <= {0, 1}
    assert adjusted_rand_score(truth, model.insample_labels()) > 0.5


def test_jump_penalty_makes_regimes_persistent(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, truth = planted
    jump = jump_fitter(2, 30.0)(features).insample_labels()
    kmeans = kmeans_fitter(2)(features).insample_labels()
    assert count_jumps(jump) == count_jumps(truth) == 4
    assert count_jumps(kmeans) > 10 * count_jumps(jump)


@pytest.mark.parametrize("fitter", [jump_fitter(2, 30.0), kmeans_fitter(2)])
def test_online_labels_use_the_past_only(
    fitter: RegimeFitter, planted: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, _ = planted
    model = fitter(features.iloc[:600])
    altered = features.copy()
    altered.iloc[800:] = 5.0
    assert (model.online_labels(features)[:800] == model.online_labels(altered)[:800]).all()
