"""An XGBoost classifier that imitates the regime labels from the same-day features.

It gives what the jump model does not: a probability per phase. It never sees
yesterday's phase, so it disagrees with the jump model mostly around transitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from termo.validation.walkforward import RefitData, RefitLabels


@dataclass(frozen=True)
class SurrogateParams:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    seed: int


@dataclass(frozen=True, eq=False)
class Surrogate:
    model: XGBClassifier
    classes: tuple[int, ...]  # phases present in the training window, in the model's order
    n_states: int
    columns: tuple[str, ...]

    def proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """One column per phase 0..n_states-1; a phase never seen in training gets 0."""
        raw = self.model.predict_proba(features[list(self.columns)])
        out = pd.DataFrame(0.0, index=features.index, columns=list(range(self.n_states)))
        out[list(self.classes)] = raw
        return out


def fit_surrogate(
    features: pd.DataFrame, labels: pd.Series, n_states: int, params: SurrogateParams
) -> Surrogate:
    """Fit on one training window. `labels` are the regime labels of exactly those rows."""
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must cover the same rows")
    classes = tuple(sorted(int(c) for c in np.unique(labels.to_numpy())))
    if len(classes) < 2:
        raise ValueError("the surrogate needs at least two phases in the training window")
    encoded = np.searchsorted(np.asarray(classes), labels.to_numpy())
    model = XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        tree_method="hist",
        random_state=params.seed,
        n_jobs=1,
    )
    model.fit(features, encoded)
    return Surrogate(
        model=model, classes=classes, n_states=n_states, columns=tuple(features.columns)
    )


@dataclass(frozen=True, eq=False)
class SurrogateWalk:
    proba: pd.DataFrame  # out-of-sample probabilities, one row per out-of-sample day
    surrogates: tuple[Surrogate, ...]  # one per refit, same order


def surrogate_walk(
    refits: Sequence[RefitData],
    fits: Sequence[RefitLabels],
    n_states: int,
    params: SurrogateParams,
) -> SurrogateWalk:
    """Refit the surrogate at every cutoff on the labels fitted there; read the next block."""
    blocks: list[pd.DataFrame] = []
    surrogates: list[Surrogate] = []
    for refit, fit in zip(refits, fits, strict=True):
        if fit.cutoff != refit.cutoff:
            raise ValueError("refits and fitted labels are out of step")
        train = refit.features.loc[: refit.cutoff]
        surrogate = fit_surrogate(train, fit.insample, n_states, params)
        later = refit.features.loc[refit.features.index > refit.cutoff]
        blocks.append(surrogate.proba(later))
        surrogates.append(surrogate)
    return SurrogateWalk(proba=pd.concat(blocks), surrogates=tuple(surrogates))
