"""The TYCCLES data recipe: changes over 1-9 months as causal ranks, curve shape, volatility.

HSBC (TYCCLES, 2026) feeds K-means with yield changes over 1m-9m horizons turned into
six-month and one-year ranks, curve slopes and curvature with their changes, and realised
volatility; every change is smoothed against base effects. Spec 2 §4 fixes our adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from termo.features.pca import BP_PER_PERCENT
from termo.features.velocity import smoothed_change

LEVEL_HORIZON, LEVEL_DELTA = 63, 10  # the inertia baseline, as in spec 1
SMOOTHING_DIVISOR = 4  # delta = horizon // 4: 21 -> 5, 63 -> 15, 189 -> 47
MIN_HORIZON = 2 * SMOOTHING_DIVISOR  # so that 0 < delta < horizon

# name -> (tenors added, tenors subtracted): slope = long - short, curvature = 2 * belly - wings
CURVE_MEASURES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "s12m5s": (("DGS5",), ("DGS1",)),
    "s3s10": (("DGS10",), ("DGS3",)),
    "s10s30": (("DGS30",), ("DGS10",)),
    "c5": (("DGS5", "DGS5"), ("DGS2", "DGS10")),
}


def short_tenor(series_id: str) -> str:
    """DGS10 -> 10."""
    return series_id.removeprefix("DGS")


def smoothing_delta(horizon: int) -> int:
    return horizon // SMOOTHING_DIVISOR


def causal_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of today within the last `window` days, today included. Past only."""
    return series.rolling(window, min_periods=window).rank(pct=True)


def curve_measures(curve: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=curve.index)
    for name, (plus, minus) in CURVE_MEASURES.items():
        out[name] = sum(curve[t] for t in plus) - sum(curve[t] for t in minus)
    return out


@dataclass(frozen=True)
class TycclesRecipe:
    """Stateless: nothing is estimated, so the fitted recipe is the recipe itself."""

    tenors: tuple[str, ...]
    horizons: tuple[int, ...]
    rank_windows: tuple[int, ...]
    vol_window: int
    vol_rank_window: int

    def __post_init__(self) -> None:
        needed = {t for plus, minus in CURVE_MEASURES.values() for t in plus + minus}
        missing = sorted(needed - set(self.tenors))
        if missing:
            raise ValueError(f"curve measures need the tenors {missing}")
        if min(self.horizons) < MIN_HORIZON:
            raise ValueError(f"every horizon must be at least {MIN_HORIZON} days")
        if min(self.rank_windows) < 2 or self.vol_window < 2 or self.vol_rank_window < 2:
            raise ValueError("rank and volatility windows must be at least 2 days")

    @property
    def names(self) -> tuple[str, ...]:
        names: list[str] = []
        for tenor in self.tenors:
            names += self._change_names(f"d{short_tenor(tenor)}")
        for measure in CURVE_MEASURES:
            names += self._change_names(measure)
        names += [f"vol{short_tenor(tenor)}_r{self.vol_rank_window}" for tenor in self.tenors]
        return tuple(names)

    @property
    def burn_in_needed(self) -> int:
        """Rows before the first finite feature: longest smoothed change, then longest rank."""
        horizon = max(self.horizons)
        return horizon + smoothing_delta(horizon) + max(self.rank_windows) - 1

    def fit(self, window: pd.DataFrame) -> TycclesRecipe:
        return self

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        yields_bp = curve[list(self.tenors)] * BP_PER_PERCENT
        measures_bp = curve_measures(curve) * BP_PER_PERCENT
        columns: dict[str, pd.Series] = {}
        for tenor in self.tenors:
            columns.update(self._ranked_changes(f"d{short_tenor(tenor)}", yields_bp[tenor]))
        for measure in CURVE_MEASURES:
            columns.update(self._ranked_changes(measure, measures_bp[measure]))
        for tenor in self.tenors:
            vol = (
                yields_bp[tenor].diff().rolling(self.vol_window, min_periods=self.vol_window).std()
            )
            columns[f"vol{short_tenor(tenor)}_r{self.vol_rank_window}"] = causal_rank(
                vol, self.vol_rank_window
            )
        return pd.DataFrame(columns, index=curve.index)[list(self.names)]

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        """Level proxy: the mean of the tenors. Spec 1 used the PCA level, absent here."""
        level = curve[list(self.tenors)].mean(axis=1) * BP_PER_PERCENT
        return smoothed_change(level, LEVEL_HORIZON, LEVEL_DELTA)

    def _change_names(self, prefix: str) -> list[str]:
        return [f"{prefix}_{h}_r{w}" for h in self.horizons for w in self.rank_windows]

    def _ranked_changes(self, prefix: str, series: pd.Series) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for horizon in self.horizons:
            change = smoothed_change(series, horizon, smoothing_delta(horizon))
            for window in self.rank_windows:
                out[f"{prefix}_{horizon}_r{window}"] = causal_rank(change, window)
        return out
