from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.dataset import ExperimentData, first_window_columns, prepare
from termo.features.pipeline import FEATURE_NAMES


def test_first_window_columns_are_a_subset_in_order(
    pre_holdout: pd.DataFrame, config: CoreConfig
) -> None:
    columns = first_window_columns(pre_holdout, config)
    assert columns[0] == "S"
    assert [name for name in FEATURE_NAMES if name in columns] == list(columns)


def test_pre_holdout_data_never_reaches_the_holdout(data: ExperimentData) -> None:
    holdout_start = pd.Timestamp(data.config.holdout_start)
    assert data.curve.index[-1] < holdout_start
    assert all(refit.features.index[-1] < holdout_start for refit in data.refits)
    assert np.isfinite(data.full_features.to_numpy()).all()


def test_ten_year_series_are_exposed(data: ExperimentData) -> None:
    assert data.yields_10y.equals(data.curve["DGS10"])
    change = data.daily_change_10y
    assert np.isnan(change.iloc[0])
    expected = (data.curve["DGS10"].iloc[5] - data.curve["DGS10"].iloc[4]) * 100.0
    assert np.isclose(change.iloc[5], expected)


def test_feature_set_is_chosen_on_the_first_window_only(
    pre_holdout: pd.DataFrame, config: CoreConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: nothing after the first training window may influence which features are used."""
    last_row_seen: list[pd.Timestamp] = []

    def spy(frame: pd.DataFrame, limit: float) -> tuple[str, ...]:
        last_row_seen.append(frame.index[-1])
        return tuple(frame.columns)

    monkeypatch.setattr("termo.dataset.select_columns", spy)
    first_window_columns(pre_holdout, config)
    first_end = pd.Timestamp(config.first_train_end)
    assert last_row_seen == [pre_holdout.index[pre_holdout.index <= first_end][-1]]


def test_each_refit_fits_its_pipeline_on_the_past_only(
    pre_holdout: pd.DataFrame, config: CoreConfig, data: ExperimentData
) -> None:
    """Rule: PCA, clipping and scaling of a refit may not see the block it will label."""
    cutoff = data.refits[0].cutoff
    altered = pre_holdout.copy()
    altered.loc[altered.index > cutoff] += 1.0
    again = prepare(altered, config, FEATURE_NAMES)
    pd.testing.assert_frame_equal(
        again.refits[0].features.loc[:cutoff], data.refits[0].features.loc[:cutoff]
    )
    after = again.refits[0].features.loc[again.refits[0].features.index > cutoff]
    before = data.refits[0].features.loc[data.refits[0].features.index > cutoff]
    assert not after.equals(before)  # the block itself did change
