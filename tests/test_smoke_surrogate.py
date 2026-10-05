"""xgboost and shap behave as spec 3 assumes, in this environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from termo.core import BOUND_PACKAGES, environment_fingerprint

PARAMS = {
    "n_estimators": 30,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 1,
}


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(600, 12)), columns=[f"f{i}" for i in range(12)])
    labels = (features["f0"] > 0).astype(int) + (features["f1"] > 1).astype(int)
    return features, labels.to_numpy()


def test_multiclass_fit_is_deterministic_and_gives_one_column_per_class() -> None:
    features, labels = _data()
    first = XGBClassifier(**PARAMS).fit(features, labels)
    second = XGBClassifier(**PARAMS).fit(features, labels)
    proba = first.predict_proba(features)
    assert proba.shape == (600, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(proba, second.predict_proba(features))


def test_shap_values_are_additive_in_log_odds() -> None:
    features, labels = _data()
    model = XGBClassifier(**PARAMS).fit(features, labels)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(features.iloc[:5]))
    assert values.shape == (5, 12, 3)  # rows, features, classes
    margin = model.predict(features.iloc[:5], output_margin=True)
    rebuilt = values.sum(axis=1) + np.asarray(explainer.expected_value)
    assert np.abs(rebuilt - margin).max() < 1e-3


def test_the_surrogate_libraries_are_part_of_the_bound_environment() -> None:
    assert {"xgboost", "shap"} <= set(BOUND_PACKAGES)
    assert {"xgboost", "shap"} <= set(environment_fingerprint())
