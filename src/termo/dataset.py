"""Everything a model configuration needs, prepared once and shared by all of them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from termo.config import CoreConfig
from termo.features.pipeline import FEATURE_NAMES, fit_pipeline, select_columns
from termo.validation.metrics import BP_PER_PERCENT
from termo.validation.walkforward import RefitData, build_refits, refit_cutoffs

TEN_YEAR = "DGS10"


@dataclass(frozen=True, eq=False)
class ExperimentData:
    config: CoreConfig
    curve: pd.DataFrame
    columns: tuple[str, ...]
    refits: tuple[RefitData, ...]
    full_features: pd.DataFrame  # pipeline fitted on the whole curve; used for FTIC only

    @property
    def yields_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR]

    @property
    def daily_change_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR].diff() * BP_PER_PERCENT


def first_window_columns(curve: pd.DataFrame, config: CoreConfig) -> tuple[str, ...]:
    """Apply the collinearity rule once, on the first training window."""
    first_end = pd.Timestamp(config.first_train_end)
    pipeline = fit_pipeline(curve, first_end, FEATURE_NAMES, config.burn_in_days)
    train = pipeline.transform(curve.loc[:first_end])
    return select_columns(train, config.thresholds.collinearity_max)


def prepare(curve: pd.DataFrame, config: CoreConfig, columns: Sequence[str]) -> ExperimentData:
    """Fit the feature pipeline at every refit date. `curve` must start at the sample start."""
    feature_dates = curve.index[config.burn_in_days :]
    cutoffs = refit_cutoffs(feature_dates, pd.Timestamp(config.first_train_end), config.refit_weeks)
    refits = build_refits(curve, cutoffs, columns, config.burn_in_days)
    full = fit_pipeline(curve, curve.index[-1], columns, config.burn_in_days).transform(curve)
    return ExperimentData(
        config=config,
        curve=curve,
        columns=tuple(columns),
        refits=tuple(refits),
        full_features=full,
    )
