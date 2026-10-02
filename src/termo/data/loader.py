"""The only door to the yield curve, and the only door to the holdout."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from termo.data.fred import parse_series_csv
from termo.data.snapshot import SnapshotError, read_snapshot


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
    texts = read_snapshot(snapshot_dir)
    missing = [name for name in series if name not in texts]
    if missing:
        raise SnapshotError(f"snapshot lacks series: {missing}")

    columns = [parse_series_csv(texts[name], name) for name in series]
    frame = pd.concat(columns, axis=1, join="outer").sort_index().dropna(how="any")
    frame = frame.loc[frame.index >= pd.Timestamp(start)]

    limit: pd.Timestamp | None
    if end is not None:
        limit = pd.Timestamp(end)
    elif final_evaluation:
        limit = None
    else:
        limit = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)
    if limit is not None:
        frame = frame.loc[frame.index <= limit]
    return frame
