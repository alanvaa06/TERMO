from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.features.pipeline import (
    FEATURE_NAMES,
    fit_pipeline,
    raw_features,
    select_columns,
)

BURN_IN = 252


@pytest.fixture(scope="module")
def train_end(curve: pd.DataFrame) -> pd.Timestamp:
    return curve.index[999]


def test_twelve_features_without_the_level(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    features = pipeline.transform(curve)
    assert tuple(features.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 12
    assert "L" not in features.columns
    assert features.index[0] == curve.index[BURN_IN]
    assert np.isfinite(features.to_numpy()).all()


def test_training_rows_are_standardized_and_clipped(
    curve: pd.DataFrame, train_end: pd.Timestamp
) -> None:
    shocked = curve.copy()
    shocked.iloc[600:] += 4.0  # a one-day jump of 400 bp: an extreme velocity and volatility
    pipeline = fit_pipeline(shocked, train_end, FEATURE_NAMES, BURN_IN)
    train = pipeline.transform(shocked).loc[:train_end]
    assert np.allclose(train.mean().to_numpy(), 0.0, atol=1e-8)
    assert np.allclose(train.std(ddof=0).to_numpy(), 1.0, atol=1e-8)
    raw = pipeline.raw(shocked).loc[:train_end]
    mean, std = raw.mean(), raw.std(ddof=0)
    assert (((raw - mean) / std).abs() > 3.0).to_numpy().any(), "the shock must create extremes"
    clipped = pipeline.clipper.transform(raw)
    assert ((clipped - mean).abs() <= 3.0 * std + 1e-9).to_numpy().all()
    assert not clipped.equals(raw)


def test_no_look_ahead_in_the_features(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    """Changing the future must not change any feature of the past."""
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    cut = 1200
    altered = curve.copy()
    altered.iloc[cut:] = altered.iloc[cut:] + 1.5
    before = pipeline.transform(curve).iloc[: cut - BURN_IN]
    after = pipeline.transform(altered).iloc[: cut - BURN_IN]
    pd.testing.assert_frame_equal(before, after)


def test_fit_uses_the_training_window_only(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    """Changing data after the training window must not change the fitted pipeline."""
    altered = curve.copy()
    altered.loc[altered.index > train_end] += 2.0
    original = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    refit = fit_pipeline(altered, train_end, FEATURE_NAMES, BURN_IN)
    assert np.array_equal(original.pca.loadings, refit.pca.loadings)
    assert np.array_equal(original.clipper.lb, refit.clipper.lb)
    assert np.array_equal(original.scaler.scaler.mean_, refit.scaler.scaler.mean_)
    pd.testing.assert_frame_equal(
        original.transform(curve.loc[:train_end]), refit.transform(altered.loc[:train_end])
    )


def test_training_window_can_start_late(curve: pd.DataFrame) -> None:
    start, end = curve.index[800], curve.index[-1]
    late = fit_pipeline(curve, end, FEATURE_NAMES, BURN_IN, train_start=start)
    window = late.transform(curve).loc[start:]
    assert np.allclose(window.mean().to_numpy(), 0.0, atol=1e-8)
    early = fit_pipeline(curve, end, FEATURE_NAMES, BURN_IN)
    assert not np.allclose(late.pca.mean_level, early.pca.mean_level)


def test_vol_features_are_logs(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    raw = raw_features(curve, pipeline.pca)
    changes = curve["DGS5"].diff() * 100.0
    vol60 = np.sqrt((changes**2).ewm(halflife=60).mean())
    vol20 = np.sqrt((changes**2).ewm(halflife=20).mean())
    assert np.allclose(raw["logvol5_60"].iloc[BURN_IN:], np.log(vol60).iloc[BURN_IN:])
    assert np.allclose(raw["logr5_20_60"].iloc[BURN_IN:], np.log(vol20 / vol60).iloc[BURN_IN:])


def test_non_finite_features_are_refused(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    flat = curve.copy()
    flat.iloc[:] = 5.0  # no changes at all: zero volatility, log of zero
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.transform(flat)


def test_select_columns_drops_the_later_of_a_correlated_pair() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    frame = pd.DataFrame({"a": a, "b": b, "a_copy": a + 0.01 * rng.normal(size=500), "c": -b})
    assert select_columns(frame, 0.8) == ("a", "b")
    assert select_columns(frame, 1.0) == ("a", "b", "a_copy", "c")
