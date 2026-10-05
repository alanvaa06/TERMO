"""Refit on the past, read the next block online, keep regime names stable."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from termo.features.pipeline import PCA_RECIPE, fit_pipeline
from termo.features.recipes import Recipe
from termo.regime.align import apply_permutation, match_to_reference, order_by_target
from termo.regime.model import RegimeFitter
from termo.validation.metrics import adjusted_rand


@dataclass(frozen=True, eq=False)
class RefitData:
    cutoff: pd.Timestamp  # last training day
    block_end: pd.Timestamp  # last day read with this fit
    features: pd.DataFrame  # standardized, every feature date up to block_end
    level_change: pd.Series  # raw smoothed 63-day change of the level, same dates


@dataclass(frozen=True, eq=False)
class WalkForwardResult:
    oos_labels: pd.Series  # daily out-of-sample labels with stable names
    consecutive_ari: tuple[float, ...]  # one per pair of consecutive refits


def refit_cutoffs(
    feature_dates: pd.DatetimeIndex, first_train_end: pd.Timestamp, refit_weeks: int
) -> list[pd.Timestamp]:
    """Last training day of each refit: first_train_end, then every refit_weeks weeks."""
    last = feature_dates[-1]
    cutoffs: list[pd.Timestamp] = []
    target = first_train_end
    while True:
        eligible = feature_dates[feature_dates <= target]
        if len(eligible) == 0:
            raise ValueError("first_train_end is before the first feature date")
        cutoff = eligible[-1]
        if cutoff >= last:
            break
        if not cutoffs or cutoff > cutoffs[-1]:
            cutoffs.append(cutoff)
        target = target + pd.Timedelta(weeks=refit_weeks)
    if not cutoffs:
        raise ValueError("no out-of-sample dates after first_train_end")
    return cutoffs


def build_refits(
    curve: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    columns: Sequence[str],
    burn_in: int,
    recipe: Recipe = PCA_RECIPE,
) -> list[RefitData]:
    """Fit the feature pipeline at every cutoff. Shared by all model configurations."""
    block_ends = [*cutoffs[1:], curve.index[-1]]
    refits: list[RefitData] = []
    for cutoff, block_end in zip(cutoffs, block_ends, strict=True):
        pipeline = fit_pipeline(curve, cutoff, columns, burn_in, recipe=recipe)
        visible = curve.loc[:block_end]
        refits.append(
            RefitData(
                cutoff=cutoff,
                block_end=block_end,
                features=pipeline.transform(visible),
                level_change=pipeline.level_change(visible),
            )
        )
    return refits


def run_walkforward(
    refits: Sequence[RefitData], fitter: RegimeFitter, n_states: int, target: pd.Series
) -> WalkForwardResult:
    """`target` names the states of the first fit (lowest in-state mean becomes state 0)."""
    previous: pd.Series | None = None
    blocks: list[pd.Series] = []
    consecutive: list[float] = []
    for refit in refits:
        train = refit.features.loc[: refit.cutoff]
        model = fitter(train)
        fitted = model.insample_labels()
        if previous is None:
            permutation = order_by_target(fitted, target.reindex(train.index).to_numpy(), n_states)
        else:
            shared = len(previous)
            if not train.index[:shared].equals(previous.index):
                raise ValueError("training windows must be nested")
            consecutive.append(adjusted_rand(previous.to_numpy(), fitted[:shared]))
            permutation = match_to_reference(previous.to_numpy(), fitted[:shared], n_states)
        previous = pd.Series(apply_permutation(fitted, permutation), index=train.index)
        online = pd.Series(
            apply_permutation(model.online_labels(refit.features), permutation),
            index=refit.features.index,
        )
        blocks.append(online.loc[online.index > refit.cutoff])
    return WalkForwardResult(oos_labels=pd.concat(blocks), consecutive_ari=tuple(consecutive))


def inertia_labels(refits: Sequence[RefitData]) -> pd.Series:
    """Baseline: 1 when the smoothed 63-day change of the level is positive, else 0."""
    blocks = [
        (refit.level_change.loc[refit.level_change.index > refit.cutoff] > 0).astype(int)
        for refit in refits
    ]
    return pd.concat(blocks)
