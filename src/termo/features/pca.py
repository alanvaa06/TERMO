"""Level, slope and curvature of the curve from a PCA of daily yield changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BP_PER_PERCENT = 100.0
FACTORS = ("L", "S", "C")
SHORT, BELLY, TEN_YEAR, LONG = "DGS1", "DGS5", "DGS10", "DGS30"


@dataclass(frozen=True, eq=False)
class CurvePCA:
    tenors: tuple[str, ...]
    loadings: np.ndarray  # (3, n_tenors); rows are L, S, C
    mean_level: np.ndarray  # (n_tenors,), percent
    explained: np.ndarray  # (3,), share of variance of daily changes

    @classmethod
    def fit(cls, curve_train: pd.DataFrame) -> CurvePCA:
        tenors = tuple(str(name) for name in curve_train.columns)
        missing = [name for name in (SHORT, BELLY, TEN_YEAR, LONG) if name not in tenors]
        if missing:
            raise ValueError(f"sign convention needs tenors {missing}")
        changes = curve_train.diff().dropna().to_numpy() * BP_PER_PERCENT
        if len(changes) <= len(tenors):
            raise ValueError("not enough observations to fit the PCA")

        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(changes, rowvar=False))
        top = np.argsort(eigenvalues)[::-1][: len(FACTORS)]
        loadings = eigenvectors[:, top].T.copy()

        short, belly = tenors.index(SHORT), tenors.index(BELLY)
        ten, long_ = tenors.index(TEN_YEAR), tenors.index(LONG)
        # L up when the 10Y rises; S up when the curve steepens; C up when the belly cheapens.
        orientation = (
            loadings[0, ten],
            loadings[1, long_] - loadings[1, short],
            loadings[2, belly] - (loadings[2, short] + loadings[2, long_]) / 2.0,
        )
        for row, value in enumerate(orientation):
            if value < 0:
                loadings[row] *= -1.0

        return cls(
            tenors=tenors,
            loadings=loadings,
            mean_level=curve_train.mean().to_numpy(),
            explained=eigenvalues[top] / eigenvalues.sum(),
        )

    def scores(self, curve: pd.DataFrame) -> pd.DataFrame:
        """L, S, C in basis points for every date in `curve`."""
        if tuple(str(name) for name in curve.columns) != self.tenors:
            raise ValueError("curve columns do not match the fitted tenors")
        centered = (curve.to_numpy() - self.mean_level) * BP_PER_PERCENT
        return pd.DataFrame(centered @ self.loadings.T, index=curve.index, columns=list(FACTORS))
