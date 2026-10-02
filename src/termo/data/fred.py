"""Download and validate one FRED series at a time."""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable

import pandas as pd

# One series per request: with several ids FRED ignores the date range for all but the first.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
MIN_YIELD = 0.0
MAX_YIELD = 25.0


class DataValidationError(ValueError):
    """A downloaded series does not look like a Treasury yield series."""


def http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "termo/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body: bytes = response.read()
    return body.decode("utf-8")


def fetch_series_csv(series_id: str, get: Callable[[str], str] = http_get) -> str:
    return get(FRED_URL.format(series_id=series_id))


def parse_series_csv(text: str, series_id: str) -> pd.Series:
    """Return the series in percent, indexed by date, with NaN where FRED has no value."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    expected = ["observation_date", series_id]
    if list(frame.columns) != expected:
        raise DataValidationError(
            f"{series_id}: expected columns {expected}, got {list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError(f"{series_id}: no rows")

    dates = pd.to_datetime(frame["observation_date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise DataValidationError(f"{series_id}: unparseable dates")
    if not (dates.is_monotonic_increasing and dates.is_unique):
        raise DataValidationError(f"{series_id}: dates must be strictly increasing")

    values = pd.to_numeric(frame[series_id], errors="coerce")
    if (frame[series_id].notna() & values.isna()).any():
        raise DataValidationError(f"{series_id}: non-numeric values")
    observed = values.dropna()
    if observed.empty:
        raise DataValidationError(f"{series_id}: no observed values")
    if ((observed < MIN_YIELD) | (observed > MAX_YIELD)).any():
        raise DataValidationError(f"{series_id}: values outside [{MIN_YIELD}, {MAX_YIELD}]")

    index = pd.DatetimeIndex(dates, name="date")
    return pd.Series(values.to_numpy(dtype=float), index=index, name=series_id)
