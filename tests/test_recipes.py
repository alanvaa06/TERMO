"""The pipeline computes raw features through a recipe; spec 1 is the PCA recipe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from conftest import TENORS
from termo.config import CoreConfig
from termo.features.pca import CurvePCA
from termo.features.pipeline import (
    FEATURE_NAMES,
    LEVEL_CHANGE,
    PCA_RECIPE,
    fit_pipeline,
    raw_features,
)
from termo.features.velocity import smoothed_change
from termo.regime.model import kmeans_fitter
from termo.validation.stability import halves_ari
from termo.validation.walkforward import build_refits

BURN_IN = 252


@dataclass(frozen=True)
class MeanRecipe:
    """Two features from the mean yield: its level in bp and its 5-day change. Stateless."""

    @property
    def names(self) -> tuple[str, ...]:
        return ("mean", "dmean5")

    def fit(self, window: pd.DataFrame) -> MeanRecipe:
        return self

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        mean = curve[list(TENORS)].mean(axis=1) * 100.0
        return pd.DataFrame({"mean": mean, "dmean5": mean.diff(5)})

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        return smoothed_change(curve[list(TENORS)].mean(axis=1) * 100.0, 63, 10)


def test_pca_recipe_reproduces_the_spec_1_features(curve: pd.DataFrame) -> None:
    train_end = curve.index[999]
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    expected = raw_features(curve, CurvePCA.fit(curve.loc[:train_end])).iloc[BURN_IN:]
    pd.testing.assert_frame_equal(pipeline.raw(curve), expected)
    pd.testing.assert_series_equal(pipeline.level_change(curve), expected[LEVEL_CHANGE])
    assert PCA_RECIPE.names == FEATURE_NAMES


def test_the_pipeline_uses_the_recipe_it_is_given(curve: pd.DataFrame) -> None:
    train_end = curve.index[999]
    recipe = MeanRecipe()
    pipeline = fit_pipeline(curve, train_end, recipe.names, 80, recipe=recipe)
    features = pipeline.transform(curve)
    assert tuple(features.columns) == ("mean", "dmean5")
    assert features.index[0] == curve.index[80]
    assert np.isfinite(features.to_numpy()).all()
    # standardized on the training window only (tolerance covers ddof and the +/-3 sigma clip)
    train = features.loc[:train_end]
    assert np.allclose(train.mean(), 0.0, atol=1e-6)
    assert np.allclose(train.std(ddof=0), 1.0, atol=0.01)


def test_refits_and_halves_take_the_recipe(curve: pd.DataFrame, config: CoreConfig) -> None:
    recipe = MeanRecipe()
    cutoffs = [curve.index[999], curve.index[1299]]
    refits = build_refits(curve, cutoffs, recipe.names, 80, recipe=recipe)
    assert [tuple(r.features.columns) for r in refits] == [("mean", "dmean5")] * 2
    expected = recipe.level_change(curve.loc[: refits[0].block_end]).iloc[80:]
    pd.testing.assert_series_equal(refits[0].level_change, expected)
    value = halves_ari(curve, recipe.names, 80, kmeans_fitter(2), recipe=recipe)
    assert -0.5 <= value <= 1.0
