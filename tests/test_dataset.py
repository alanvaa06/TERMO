from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.dataset import ExperimentData, first_window_columns, prepare, recipe_for
from termo.features.pipeline import FEATURE_NAMES, PCA_RECIPE, fit_pipeline


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


def test_recipe_follows_the_feature_set(config: CoreConfig, tyccles_config: CoreConfig) -> None:
    assert recipe_for(config) is PCA_RECIPE
    recipe = recipe_for(tyccles_config)
    assert len(recipe.names) == 139
    with pytest.raises(ValueError, match="burn_in_days"):
        recipe_for(replace(tyccles_config, burn_in_days=400))


def test_collinearity_rule_can_be_switched_off(
    pre_holdout: pd.DataFrame, tyccles_config: CoreConfig
) -> None:
    columns = first_window_columns(pre_holdout, tyccles_config)
    assert columns == recipe_for(tyccles_config).names
    pruned = first_window_columns(
        pre_holdout, replace(tyccles_config, apply_collinearity_rule=True)
    )
    assert 0 < len(pruned) < 139 and set(pruned) <= set(columns)


def test_frozen_refit_is_one_fit_on_the_past_until_the_frozen_date(
    tyccles_data: ExperimentData, pre_holdout: pd.DataFrame, tyccles_config: CoreConfig
) -> None:
    assert len(tyccles_data.frozen_refits) == 1
    (frozen,) = tyccles_data.frozen_refits
    assert frozen.cutoff <= pd.Timestamp(tyccles_config.frozen_train_end)
    assert frozen.block_end == pre_holdout.index[-1]
    assert frozen.features.index[-1] == pre_holdout.index[-1]
    assert tuple(frozen.features.columns) == tyccles_data.columns
    pipeline = fit_pipeline(
        pre_holdout,
        frozen.cutoff,
        tyccles_data.columns,
        tyccles_config.burn_in_days,
        recipe=recipe_for(tyccles_config),
    )
    pd.testing.assert_frame_equal(
        frozen.features.loc[: frozen.cutoff], pipeline.transform(pre_holdout.loc[: frozen.cutoff])
    )


def test_no_frozen_refit_without_a_frozen_date(data: ExperimentData) -> None:
    assert data.frozen_refits == ()


def test_frozen_features_ignore_everything_after_the_frozen_date(
    tyccles_data: ExperimentData, pre_holdout: pd.DataFrame, tyccles_config: CoreConfig
) -> None:
    (frozen,) = tyccles_data.frozen_refits
    altered = pre_holdout.copy()
    later = altered.index > frozen.cutoff
    altered.loc[later] = altered.loc[later] * 1.5 + 2.0
    (again,) = prepare(altered, tyccles_config, tyccles_data.columns).frozen_refits
    pd.testing.assert_frame_equal(
        again.features.loc[: frozen.cutoff], frozen.features.loc[: frozen.cutoff]
    )
    after = frozen.features.index > frozen.cutoff
    assert not again.features.loc[after].equals(frozen.features.loc[after])
