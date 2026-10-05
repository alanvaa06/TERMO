"""SHAP of the surrogate for the phase of each day, summed in blocks of variables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.surrogate.explain import block_map, block_sums, explain, top_variables
from termo.surrogate.model import Surrogate, SurrogateParams, fit_surrogate

PARAMS = SurrogateParams(
    n_estimators=30, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, seed=0
)
BLOCKS = (("left", ("a", "b")), ("right", ("c", "d")))
COLUMNS = ("a_1", "a_2", "b_1", "c_1", "d_1", "d_2")


def _fitted() -> tuple[pd.DataFrame, pd.Series, Surrogate]:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2000-01-03", periods=400)
    features = pd.DataFrame(rng.normal(size=(400, 6)), index=index, columns=list(COLUMNS))
    labels = (features["a_1"] > 0).astype(int) + (features["c_1"] > 1).astype(int)
    return features, labels, fit_surrogate(features, labels, 3, PARAMS)


def test_every_variable_belongs_to_exactly_one_block() -> None:
    mapping = block_map(COLUMNS, BLOCKS)
    assert mapping == {
        "a_1": "left",
        "a_2": "left",
        "b_1": "left",
        "c_1": "right",
        "d_1": "right",
        "d_2": "right",
    }
    with pytest.raises(ValueError, match="no block"):
        block_map((*COLUMNS, "z_1"), BLOCKS)
    with pytest.raises(ValueError, match="more than one block"):
        block_map(COLUMNS, (*BLOCKS, ("again", ("a",))))


def test_values_plus_base_rebuild_the_margin_of_the_phase() -> None:
    features, labels, surrogate = _fitted()
    rows = features.iloc[:20]
    explanation = explain(surrogate, rows, labels.iloc[:20])
    assert explanation.values.shape == (20, 6)
    assert list(explanation.values.columns) == list(COLUMNS)
    margin = surrogate.model.predict(rows, output_margin=True)
    expected = margin[np.arange(20), labels.iloc[:20].to_numpy()]
    rebuilt = explanation.values.sum(axis=1) + explanation.base
    assert np.abs(rebuilt.to_numpy() - expected).max() < 1e-3


def test_block_sums_keep_the_total() -> None:
    features, labels, surrogate = _fitted()
    explanation = explain(surrogate, features.iloc[:20], labels.iloc[:20])
    sums = block_sums(explanation.values, BLOCKS)
    assert list(sums.columns) == ["left", "right"]
    assert np.allclose(sums.sum(axis=1), explanation.values.sum(axis=1))
    # the label depends on a_1 and c_1: both blocks carry weight somewhere
    assert (sums.abs().mean() > 0.01).all()


def test_top_variables_are_ordered_by_absolute_contribution() -> None:
    row = pd.Series({"a_1": 0.1, "a_2": -0.9, "b_1": 0.5, "c_1": 0.0, "d_1": -0.2, "d_2": 0.3})
    assert top_variables(row, 3) == [("a_2", -0.9), ("b_1", 0.5), ("d_2", 0.3)]


def test_a_phase_the_surrogate_never_saw_has_no_explanation() -> None:
    features, labels, _ = _fitted()
    two = labels.clip(upper=1) * 2  # phases 0 and 2 only
    surrogate = fit_surrogate(features, two, 3, PARAMS)
    phases = pd.Series([0, 1, 2], index=features.index[:3])
    explanation = explain(surrogate, features.iloc[:3], phases)
    assert explanation.values.iloc[1].isna().all() and np.isnan(explanation.base.iloc[1])
    assert explanation.values.iloc[[0, 2]].notna().all().all()
    margin = surrogate.model.predict(features.iloc[:3], output_margin=True)  # margin of phase 2
    rebuilt = explanation.values.sum(axis=1) + explanation.base
    assert rebuilt.iloc[2] == pytest.approx(float(np.ravel(margin)[2]), abs=1e-3)
    assert rebuilt.iloc[0] == pytest.approx(-float(np.ravel(margin)[0]), abs=1e-3)
