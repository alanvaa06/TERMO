from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.dataset import ExperimentData
from termo.regime.model import jump_fitter, kmeans_fitter
from termo.validation.metrics import adjusted_rand
from termo.validation.walkforward import inertia_labels, refit_cutoffs, run_walkforward


class SpyModel:
    """Stands in for a regime model and tells online reading apart from full decoding."""

    n_states = 2

    def __init__(self, n_train: int) -> None:
        self._n_train = n_train

    def insample_labels(self) -> np.ndarray:
        return np.zeros(self._n_train, dtype=int)

    def online_labels(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=int)

    def full_labels(self, features: pd.DataFrame) -> np.ndarray:
        return np.ones(len(features), dtype=int)


def test_each_fit_sees_only_its_training_window(data: ExperimentData) -> None:
    """Rule: nothing is fitted on the block it is about to label."""
    last_row_seen: list[pd.Timestamp] = []
    first_row_seen: list[pd.Timestamp] = []

    def spying_fitter(features: pd.DataFrame) -> SpyModel:
        first_row_seen.append(features.index[0])
        last_row_seen.append(features.index[-1])
        return SpyModel(len(features))

    run_walkforward(data.refits, spying_fitter, 2, data.daily_change_10y)

    assert last_row_seen == [refit.cutoff for refit in data.refits]
    assert set(first_row_seen) == {data.refits[0].features.index[0]}
    for refit, last in zip(data.refits, last_row_seen, strict=True):
        assert last < refit.block_end


def test_out_of_sample_labels_come_from_the_online_reading(data: ExperimentData) -> None:
    """Rule: the label of day t may not use days after t, so full decoding is never used."""
    result = run_walkforward(
        data.refits, lambda features: SpyModel(len(features)), 2, data.daily_change_10y
    )
    assert (result.oos_labels == 0).all()  # the spy's full decoding would give 1


def test_cutoffs_step_every_refit_weeks() -> None:
    dates = pd.bdate_range("2000-01-03", "2001-12-31")
    cutoffs = refit_cutoffs(dates, pd.Timestamp("2000-12-31"), 26)
    assert [c.strftime("%Y-%m-%d") for c in cutoffs] == ["2000-12-29", "2001-06-29", "2001-12-28"]


def test_cutoffs_need_out_of_sample_dates() -> None:
    dates = pd.bdate_range("2000-01-03", "2000-06-30")
    with pytest.raises(ValueError, match="no out-of-sample"):
        refit_cutoffs(dates, pd.Timestamp("2000-12-31"), 26)
    with pytest.raises(ValueError, match="before the first"):
        refit_cutoffs(dates, pd.Timestamp("1999-12-31"), 26)


def test_blocks_tile_the_out_of_sample_period(data: ExperimentData, config: CoreConfig) -> None:
    refits = data.refits
    assert len(refits) == 8
    assert refits[0].cutoff == pd.Timestamp("1994-06-30")
    for earlier, later in zip(refits[:-1], refits[1:], strict=True):
        assert earlier.block_end == later.cutoff
    assert refits[-1].block_end == data.curve.index[-1]
    assert refits[-1].block_end < pd.Timestamp(config.holdout_start)
    for refit in refits:
        assert refit.features.index[-1] == refit.block_end
        assert refit.features.index.equals(refit.level_change.index)


def test_walkforward_reads_every_out_of_sample_day_once(data: ExperimentData) -> None:
    result = run_walkforward(data.refits, jump_fitter(2, 50.0), 2, data.daily_change_10y)
    expected = data.curve.index[data.curve.index > data.refits[0].cutoff]
    assert result.oos_labels.index.equals(expected)
    assert len(result.consecutive_ari) == len(data.refits) - 1
    assert all(-1.0 <= value <= 1.0 for value in result.consecutive_ari)


def test_walkforward_finds_the_planted_regimes_with_stable_names(
    data: ExperimentData, curve_and_regimes: tuple[pd.DataFrame, np.ndarray]
) -> None:
    curve, regimes = curve_and_regimes
    truth = pd.Series(regimes, index=curve.index)
    result = run_walkforward(data.refits, jump_fitter(2, 10.0), 2, data.daily_change_10y)
    labels = result.oos_labels
    # Online reading recognizes a new regime with a lag, so agreement is high but not perfect.
    assert adjusted_rand(truth.loc[labels.index].to_numpy(), labels.to_numpy()) > 0.4
    # State 0 is the rally (yields falling), state 1 the sell-off, in every block.
    change = data.daily_change_10y.loc[labels.index]
    assert change[labels == 0].mean() < 0 < change[labels == 1].mean()
    assert (labels == truth.loc[labels.index]).mean() > 0.8


def test_walkforward_has_no_look_ahead(data: ExperimentData) -> None:
    """Labels of a block must not change when later blocks are altered."""
    first_two = data.refits[:2]
    result = run_walkforward(data.refits, jump_fitter(2, 50.0), 2, data.daily_change_10y)
    partial = run_walkforward(first_two, jump_fitter(2, 50.0), 2, data.daily_change_10y)
    pd.testing.assert_series_equal(
        result.oos_labels.loc[partial.oos_labels.index], partial.oos_labels
    )


def test_inertia_labels_follow_the_sign_of_the_level_change(data: ExperimentData) -> None:
    labels = inertia_labels(data.refits)
    assert set(labels.unique()) <= {0, 1}
    first = data.refits[0]
    block = first.level_change.loc[first.level_change.index > first.cutoff]
    assert (labels.loc[block.index] == (block > 0).astype(int)).all()


def test_walkforward_returns_the_aligned_labels_of_every_refit(data: ExperimentData) -> None:
    walk = run_walkforward(data.refits, kmeans_fitter(2), 2, data.daily_change_10y)
    assert len(walk.fits) == len(data.refits)
    blocks = []
    for refit, fit in zip(data.refits, walk.fits, strict=True):
        assert fit.cutoff == refit.cutoff
        assert fit.insample.index.equals(refit.features.loc[: refit.cutoff].index)
        assert fit.online.index.equals(refit.features.index)
        assert set(fit.insample.unique()) <= {0, 1} and set(fit.online.unique()) <= {0, 1}
        blocks.append(fit.online.loc[fit.online.index > refit.cutoff])
    # the out-of-sample series is exactly the online labels after each cutoff, same names
    pd.testing.assert_series_equal(pd.concat(blocks), walk.oos_labels)
