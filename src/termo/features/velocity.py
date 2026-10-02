"""Change of a factor over a horizon, smoothed over neighbouring windows."""

from __future__ import annotations

import pandas as pd


def smoothed_change(series: pd.Series, horizon: int, delta: int) -> pd.Series:
    """Average of the changes over horizon-delta, horizon and horizon+delta days.

    All three windows end today, so an old move leaving one window is diluted
    while today's move counts in full.
    """
    if not 0 < delta < horizon:
        raise ValueError("need 0 < delta < horizon")
    windows = (horizon - delta, horizon, horizon + delta)
    return sum(series.diff(window) for window in windows) / float(len(windows))
