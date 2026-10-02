"""Exponentially weighted volatility of daily yield changes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_vol(changes: pd.Series, halflife: float) -> pd.Series:
    """Root of the exponentially weighted mean of squared changes (weights sum to one)."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    return np.sqrt((changes**2).ewm(halflife=halflife, adjust=True).mean())
