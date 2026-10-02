from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.dataset import ExperimentData
from termo.regime.model import jump_fitter
from termo.validation.metrics import adjusted_rand
from termo.validation.walkforward import inertia_labels, refit_cutoffs, run_walkforward


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
