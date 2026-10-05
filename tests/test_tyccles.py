"""TYCCLES data recipe: smoothed changes over 1-9 months as causal ranks, curve, volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import TENORS
from termo.features.pipeline import fit_pipeline
from termo.features.tyccles import TycclesRecipe, causal_rank, curve_measures, smoothing_delta
from termo.features.velocity import smoothed_change

RECIPE = TycclesRecipe(
    tenors=TENORS,
    horizons=(21, 42, 63, 84, 126, 189),
    rank_windows=(126, 252),
    vol_window=21,
    vol_rank_window=252,
)


def test_causal_rank_by_hand() -> None:
    rank = causal_rank(pd.Series([3.0, 1.0, 2.0, 5.0, 4.0]), 3)
    assert rank.iloc[:2].isna().all()
    assert rank.iloc[2] == pytest.approx(2 / 3)  # 2 among (3, 1, 2)
    assert rank.iloc[3] == pytest.approx(1.0)  # 5 among (1, 2, 5)
    assert rank.iloc[4] == pytest.approx(2 / 3)  # 4 among (2, 5, 4)


def test_rank_ignores_the_future() -> None:
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=600))
    altered = series.copy()
    altered.iloc[400:] += 100.0
    pd.testing.assert_series_equal(
        causal_rank(series, 126).iloc[:400], causal_rank(altered, 126).iloc[:400]
    )


def test_smoothing_delta_follows_the_spec() -> None:
    assert [smoothing_delta(h) for h in (21, 42, 63, 84, 126, 189)] == [5, 10, 15, 21, 31, 47]


def test_one_hundred_thirty_nine_named_features(curve: pd.DataFrame) -> None:
    names = RECIPE.names
    assert len(names) == 139 and len(set(names)) == 139
    assert names[0] == "d1_21_r126" and names[-1] == "vol30_r252"
    assert "s3s10_63_r252" in names and "c5_189_r126" in names and "d30_189_r252" in names
    raw = RECIPE.raw(curve)
    assert tuple(raw.columns) == names
    assert RECIPE.burn_in_needed == 189 + 47 + 252 - 1
    assert raw.iloc[RECIPE.burn_in_needed :].notna().all().all()
    assert raw.iloc[RECIPE.burn_in_needed - 1].isna().any()
    finite = raw.dropna()
    assert ((finite > 0.0) & (finite <= 1.0)).all().all()


def test_features_are_the_ranked_smoothed_changes(curve: pd.DataFrame) -> None:
    raw = RECIPE.raw(curve)
    ten_year = curve["DGS10"] * 100.0
    expected = causal_rank(smoothed_change(ten_year, 63, 15), 252)
    pd.testing.assert_series_equal(raw["d10_63_r252"], expected, check_names=False)
    slope = (curve["DGS10"] - curve["DGS3"]) * 100.0
    expected = causal_rank(smoothed_change(slope, 21, 5), 126)
    pd.testing.assert_series_equal(raw["s3s10_21_r126"], expected, check_names=False)
    vol = (curve["DGS2"] * 100.0).diff().rolling(21).std()
    pd.testing.assert_series_equal(raw["vol2_r252"], causal_rank(vol, 252), check_names=False)


def test_curve_measures_have_the_right_sign() -> None:
    index = pd.bdate_range("2000-01-03", periods=3)
    flat = dict.fromkeys(TENORS, 5.0)
    steep = {**flat, "DGS1": 4.0, "DGS30": 6.0}
    humped = {**flat, "DGS5": 5.5}
    measures = curve_measures(pd.DataFrame([flat, steep, humped], index=index))
    assert measures.loc[index[0]].eq(0.0).all()
    assert measures.loc[index[1], "s12m5s"] == 1.0
    assert measures.loc[index[1], "s3s10"] == 0.0
    assert measures.loc[index[1], "s10s30"] == 1.0
    assert measures.loc[index[2], "c5"] == 1.0  # 2 * 5.5 - 5 - 5


def test_level_change_is_the_smoothed_change_of_the_mean_yield(curve: pd.DataFrame) -> None:
    expected = smoothed_change(curve[list(TENORS)].mean(axis=1) * 100.0, 63, 10)
    pd.testing.assert_series_equal(RECIPE.level_change(curve), expected)


def test_recipe_needs_the_tenors_of_the_curve_measures() -> None:
    with pytest.raises(ValueError, match="curve measures need"):
        TycclesRecipe(
            tenors=("DGS1", "DGS10"),
            horizons=(21,),
            rank_windows=(126,),
            vol_window=21,
            vol_rank_window=252,
        )


def test_the_pipeline_runs_the_recipe(curve: pd.DataFrame) -> None:
    pipeline = fit_pipeline(curve, curve.index[1199], RECIPE.names, 504, recipe=RECIPE)
    features = pipeline.transform(curve)
    assert features.shape[1] == 139
    assert features.index[0] == curve.index[504]
    assert np.isfinite(features.to_numpy()).all()
