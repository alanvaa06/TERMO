from __future__ import annotations

import numpy as np
import pandas as pd

from termo.config import CoreConfig
from termo.dataset import ExperimentData, first_window_columns
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
