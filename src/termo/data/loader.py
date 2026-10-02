"""The only door to the yield curve, and the only door to the holdout."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from termo.data.fred import DataValidationError, parse_series_csv
from termo.data.snapshot import SnapshotError, read_snapshot

MAX_START_LAG_DAYS = 7
MAX_GAP_DAYS = 10  # the longest ordinary closure (holiday next to a weekend) is 4 days


class HoldoutAccessError(RuntimeError):
    """Holdout dates were requested without the final-evaluation flag."""


def load_curve(
    snapshot_dir: Path,
    series: Sequence[str],
    start: date,
    holdout_start: date,
    *,
    end: date | None = None,
    final_evaluation: bool = False,
) -> pd.DataFrame:
    """Yields in percent, one column per series, only days where every series has a value.

    Without `final_evaluation` the frame stops the day before `holdout_start`.
    """
    if not final_evaluation and end is not None and end >= holdout_start:
        raise HoldoutAccessError(
            f"end={end} reaches the holdout ({holdout_start}); pass final_evaluation=True"
        )
    frame = _snapshot_frame(snapshot_dir, series, start)

    limit: pd.Timestamp | None
    if end is not None:
        limit = pd.Timestamp(end)
    elif final_evaluation:
        limit = None
    else:
        limit = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)
    if limit is not None:
        frame = frame.loc[frame.index <= limit]
    check_coverage(frame, start)
    return frame


def snapshot_days(snapshot_dir: Path, series: Sequence[str], start: date) -> pd.DatetimeIndex:
    """Days on which every series has a value, over the whole snapshot, coverage checked.

    Dates only: a caller can size and validate the holdout without using a single yield.
    """
    frame = _snapshot_frame(snapshot_dir, series, start)
    check_coverage(frame, start)
    return frame.index


def _snapshot_frame(snapshot_dir: Path, series: Sequence[str], start: date) -> pd.DataFrame:
    texts = read_snapshot(snapshot_dir)
    missing = [name for name in series if name not in texts]
    if missing:
        raise SnapshotError(f"snapshot lacks series: {missing}")
    return complete_days([parse_series_csv(texts[name], name) for name in series], start)


def complete_days(columns: Sequence[pd.Series], start: date) -> pd.DataFrame:
    """One column per series, from `start`, keeping only days where every series has a value."""
    frame = pd.concat(list(columns), axis=1, join="outer").sort_index().dropna(how="any")
    return frame.loc[frame.index >= pd.Timestamp(start)]


def check_coverage(frame: pd.DataFrame, start: date) -> None:
    """A late start or a long hole in any series silently shortens the sample: refuse it."""
    if frame.empty:
        raise DataValidationError("no day has a value for every series")
    late = (frame.index[0] - pd.Timestamp(start)).days
    if late > MAX_START_LAG_DAYS:
        raise DataValidationError(
            f"the sample starts on {frame.index[0].date()}, {late} days after {start}"
        )
    gaps = frame.index.to_series().diff().dt.days
    if gaps.max() > MAX_GAP_DAYS:
        raise DataValidationError(
            f"gap of {int(gaps.max())} days ending on {gaps.idxmax().date()}: "
            "some series has a hole there"
        )
