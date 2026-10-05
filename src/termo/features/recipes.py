"""What a feature recipe must offer. The recipe computes raw features; the pipeline scales them."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class FittedRecipe(Protocol):
    """Raw features with whatever the recipe estimates fixed on a training window."""

    @property
    def names(self) -> tuple[str, ...]: ...

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        """Every raw feature for every row of `curve`. Row t uses data up to t only."""
        ...

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        """Smoothed 63-day change of the level, same rows. The inertia baseline reads its sign."""
        ...


class Recipe(Protocol):
    @property
    def names(self) -> tuple[str, ...]: ...

    def fit(self, window: pd.DataFrame) -> FittedRecipe:
        """Fix what the recipe estimates (PCA loadings, nothing for ranks) on `window` only."""
        ...
