"""From the yield curve to the standardized daily feature vector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from termo.features.pca import BP_PER_PERCENT, CurvePCA
from termo.features.recipes import FittedRecipe, Recipe
from termo.features.velocity import smoothed_change
from termo.features.volatility import ewm_vol

FEATURE_NAMES: tuple[str, ...] = (
    "S",
    "C",
    "dL21",
    "dL63",
    "dS21",
    "dS63",
    "dC63",
    "logvol5_60",
    "logr5_20_60",
    "logr5_60_120",
    "logr2_20_60",
    "logr10_20_60",
)
LEVEL_CHANGE = "dL63"
# name, factor, horizon, delta
VELOCITIES: tuple[tuple[str, str, int, int], ...] = (
    ("dL21", "L", 21, 5),
    ("dL63", "L", 63, 10),
    ("dS21", "S", 21, 5),
    ("dS63", "S", 63, 10),
    ("dC63", "C", 63, 10),
)
CLIP_STD = 3.0


def raw_features(curve: pd.DataFrame, pca: CurvePCA) -> pd.DataFrame:
    """The 12 features before clipping and scaling. Row t uses data up to t only."""
    scores = pca.scores(curve)
    changes = curve.diff() * BP_PER_PERCENT
    out = pd.DataFrame(index=curve.index)
    out["S"] = scores["S"]
    out["C"] = scores["C"]
    for name, factor, horizon, delta in VELOCITIES:
        out[name] = smoothed_change(scores[factor], horizon, delta)
    vol5 = {halflife: ewm_vol(changes["DGS5"], halflife) for halflife in (20, 60, 120)}
    out["logvol5_60"] = np.log(vol5[60])
    out["logr5_20_60"] = np.log(vol5[20] / vol5[60])
    out["logr5_60_120"] = np.log(vol5[60] / vol5[120])
    for name, tenor in (("logr2_20_60", "DGS2"), ("logr10_20_60", "DGS10")):
        out[name] = np.log(ewm_vol(changes[tenor], 20) / ewm_vol(changes[tenor], 60))
    return out[list(FEATURE_NAMES)]


@dataclass(frozen=True, eq=False)
class FittedPcaRecipe:
    """Spec 1: PCA scores, velocities and log volatilities, loadings fixed on the window."""

    pca: CurvePCA

    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        return raw_features(curve, self.pca)

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        return raw_features(curve, self.pca)[LEVEL_CHANGE]


@dataclass(frozen=True)
class PcaRecipe:
    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def fit(self, window: pd.DataFrame) -> FittedPcaRecipe:
        return FittedPcaRecipe(pca=CurvePCA.fit(window))


PCA_RECIPE: Recipe = PcaRecipe()


@dataclass(frozen=True, eq=False)
class FittedPipeline:
    recipe: FittedRecipe
    columns: tuple[str, ...]
    burn_in: int
    clipper: DataClipperStd
    scaler: StandardScalerPD

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        """All raw features after the burn-in. `curve` must start at the sample start."""
        raw = self.recipe.raw(curve).iloc[self.burn_in :]
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError("non-finite feature values after the burn-in")
        return raw

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        """Raw smoothed 63-day change of the level after the burn-in, for the inertia baseline."""
        change = self.recipe.level_change(curve).iloc[self.burn_in :]
        if not np.isfinite(change.to_numpy()).all():
            raise ValueError("non-finite level change after the burn-in")
        return change

    def transform(self, curve: pd.DataFrame) -> pd.DataFrame:
        selected = self.raw(curve)[list(self.columns)]
        return self.scaler.transform(self.clipper.transform(selected))


def fit_pipeline(
    curve: pd.DataFrame,
    train_end: pd.Timestamp,
    columns: Sequence[str],
    burn_in: int,
    train_start: pd.Timestamp | None = None,
    recipe: Recipe = PCA_RECIPE,
) -> FittedPipeline:
    """Fit the recipe, clipping bounds and z-score on the training window only."""
    window = curve.loc[:train_end] if train_start is None else curve.loc[train_start:train_end]
    fitted = recipe.fit(window)
    raw = fitted.raw(curve.loc[:train_end]).iloc[burn_in:]
    train = raw.loc[window.index[0] :, list(columns)]
    if train.empty:
        raise ValueError("training window is empty after the burn-in")
    clipper = DataClipperStd(mul=CLIP_STD).fit(train)
    scaler = StandardScalerPD().fit(clipper.transform(train))
    return FittedPipeline(
        recipe=fitted, columns=tuple(columns), burn_in=burn_in, clipper=clipper, scaler=scaler
    )


def select_columns(standardized_train: pd.DataFrame, max_abs_corr: float) -> tuple[str, ...]:
    """Keep each column unless it correlates above the limit with one already kept."""
    corr = standardized_train.corr().abs()
    kept: list[str] = []
    for name in standardized_train.columns:
        if all(corr.loc[name, other] <= max_abs_corr for other in kept):
            kept.append(str(name))
    return tuple(kept)
