"""Everything a model configuration needs, prepared once and shared by all of them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from termo.config import CoreConfig
from termo.features.pipeline import PCA_RECIPE, fit_pipeline, select_columns
from termo.features.recipes import Recipe
from termo.features.tyccles import TycclesRecipe
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
    frozen_refits: tuple[RefitData, ...] = ()  # one refit, never updated: the frozen K-means

    @property
    def yields_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR]

    @property
    def daily_change_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR].diff() * BP_PER_PERCENT


def recipe_for(config: CoreConfig) -> Recipe:
    """The feature recipe the configuration names, checked against its burn-in."""
    if config.feature_set != "tyccles":
        return PCA_RECIPE
    assert config.tyccles is not None  # CoreConfig validates the pair
    recipe = TycclesRecipe(
        tenors=config.series,
        horizons=config.tyccles.change_horizons_days,
        rank_windows=config.tyccles.rank_windows_days,
        vol_window=config.tyccles.vol_window_days,
        vol_rank_window=config.tyccles.vol_rank_window_days,
    )
    if config.burn_in_days < recipe.burn_in_needed:
        raise ValueError(
            f"burn_in_days must be at least {recipe.burn_in_needed} for these horizons and ranks"
        )
    return recipe


def first_window_columns(curve: pd.DataFrame, config: CoreConfig) -> tuple[str, ...]:
    """Apply the collinearity rule once, on the first training window, when the config says so."""
    recipe = recipe_for(config)
    if not config.apply_collinearity_rule:
        return recipe.names
    first_end = pd.Timestamp(config.first_train_end)
    pipeline = fit_pipeline(curve, first_end, recipe.names, config.burn_in_days, recipe=recipe)
    train = pipeline.transform(curve.loc[:first_end])
    return select_columns(train, config.thresholds.collinearity_max)


def frozen_cutoff(feature_dates: pd.DatetimeIndex, frozen_train_end: pd.Timestamp) -> pd.Timestamp:
    """Last feature date on or before the frozen date; there must be dates after it."""
    eligible = feature_dates[feature_dates <= frozen_train_end]
    if len(eligible) == 0 or eligible[-1] >= feature_dates[-1]:
        raise ValueError("frozen_train_end leaves no out-of-sample dates")
    return eligible[-1]


def prepare(curve: pd.DataFrame, config: CoreConfig, columns: Sequence[str]) -> ExperimentData:
    """Fit the feature pipeline at every refit date. `curve` must start at the sample start."""
    recipe = recipe_for(config)
    burn_in = config.burn_in_days
    feature_dates = curve.index[burn_in:]
    cutoffs = refit_cutoffs(feature_dates, pd.Timestamp(config.first_train_end), config.refit_weeks)
    refits = build_refits(curve, cutoffs, columns, burn_in, recipe=recipe)
    frozen: tuple[RefitData, ...] = ()
    if config.frozen_train_end is not None:
        cutoff = frozen_cutoff(feature_dates, pd.Timestamp(config.frozen_train_end))
        frozen = tuple(build_refits(curve, [cutoff], columns, burn_in, recipe=recipe))
    full = fit_pipeline(curve, curve.index[-1], columns, burn_in, recipe=recipe).transform(curve)
    return ExperimentData(
        config=config,
        curve=curve,
        columns=tuple(columns),
        refits=tuple(refits),
        full_features=full,
        frozen_refits=frozen,
    )
