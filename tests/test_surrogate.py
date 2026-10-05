"""The surrogate imitates the regime labels from the same-day features, past only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from termo.dataset import ExperimentData
from termo.regime.model import kmeans_fitter
from termo.surrogate.model import SurrogateParams, fit_surrogate, surrogate_walk
from termo.validation.walkforward import run_walkforward

PARAMS = SurrogateParams(
    n_estimators=30, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, seed=0
)


def _toy(n: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2000-01-03", periods=n)
    features = pd.DataFrame(rng.normal(size=(n, 6)), index=index, columns=list("abcdef"))
    labels = (features["a"] > 0).astype(int) + (features["b"] > 1).astype(int)
    return features, labels


def test_probabilities_have_one_column_per_phase_and_sum_to_one() -> None:
    features, labels = _toy()
    surrogate = fit_surrogate(features, labels, 3, PARAMS)
    proba = surrogate.proba(features)
    assert list(proba.columns) == [0, 1, 2] and proba.index.equals(features.index)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert balanced_accuracy_score(labels, proba.idxmax(axis=1)) > 0.9


def test_a_phase_missing_from_the_training_window_gets_probability_zero() -> None:
    features, labels = _toy()
    two = labels.clip(upper=1) * 2  # phases 0 and 2 only
    surrogate = fit_surrogate(features, two, 3, PARAMS)
    proba = surrogate.proba(features)
    assert surrogate.classes == (0, 2)
    assert (proba[1] == 0.0).all() and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_one_phase_only_cannot_be_imitated() -> None:
    features, labels = _toy()
    with pytest.raises(ValueError, match="at least two phases"):
        fit_surrogate(features, labels * 0, 3, PARAMS)


def test_the_fit_is_deterministic() -> None:
    features, labels = _toy()
    first = fit_surrogate(features, labels, 3, PARAMS).proba(features)
    second = fit_surrogate(features, labels, 3, PARAMS).proba(features)
    pd.testing.assert_frame_equal(first, second)


def test_surrogate_walk_reads_each_block_with_the_fit_of_its_cutoff(data: ExperimentData) -> None:
    walk = run_walkforward(data.refits, kmeans_fitter(2), 2, data.daily_change_10y)
    result = surrogate_walk(data.refits, walk.fits, 2, PARAMS)
    assert result.proba.index.equals(walk.oos_labels.index)
    assert len(result.surrogates) == len(data.refits)
    # a memoryless clustering is easy to imitate from the same features
    assert balanced_accuracy_score(walk.oos_labels, result.proba.idxmax(axis=1)) > 0.9
    # no look-ahead: the first block equals a surrogate fitted by hand on the first window
    refit, fit = data.refits[0], walk.fits[0]
    by_hand = fit_surrogate(refit.features.loc[: refit.cutoff], fit.insample, 2, PARAMS)
    later = refit.features.loc[refit.features.index > refit.cutoff]
    pd.testing.assert_frame_equal(result.proba.loc[later.index], by_hand.proba(later))


def test_shuffled_labels_cannot_be_imitated() -> None:
    features, labels = _toy(800)
    rng = np.random.default_rng(2)
    shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
    surrogate = fit_surrogate(features.iloc[:500], shuffled.iloc[:500], 3, PARAMS)
    guess = surrogate.proba(features.iloc[500:]).idxmax(axis=1)
    assert balanced_accuracy_score(shuffled.iloc[500:], guess) < 0.5
