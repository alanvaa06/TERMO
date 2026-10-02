from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from termo.data.fred import DataValidationError
from termo.data.loader import HoldoutAccessError, load_curve, snapshot_days
from termo.data.snapshot import SnapshotError, write_snapshot

START = date(2024, 9, 25)
HOLDOUT = date(2024, 10, 1)
SERIES = ("DGS1", "DGS2")
FILES = {
    "DGS1": (
        "observation_date,DGS1\n2024-09-24,4.00\n2024-09-25,4.01\n2024-09-26,4.02\n"
        "2024-09-27,\n2024-09-30,4.04\n2024-10-01,4.05\n2024-10-02,4.06\n"
    ),
    "DGS2": (
        "observation_date,DGS2\n2024-09-24,3.50\n2024-09-25,3.51\n2024-09-26,3.52\n"
        "2024-09-27,3.53\n2024-09-30,3.54\n2024-10-01,3.55\n2024-10-02,3.56\n"
    ),
}


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    target = tmp_path / "snap"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    return target


def test_default_load_stops_before_the_holdout(snapshot: Path) -> None:
    frame = load_curve(snapshot, SERIES, START, HOLDOUT)
    assert list(frame.columns) == list(SERIES)
    # 09-24 is before the start; 09-27 lacks DGS1; 10-01 onwards is holdout.
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2024-09-25", "2024-09-26", "2024-09-30"]


def test_requesting_holdout_dates_is_refused(snapshot: Path) -> None:
    with pytest.raises(HoldoutAccessError):
        load_curve(snapshot, SERIES, START, HOLDOUT, end=date(2024, 10, 1))


def test_final_evaluation_opens_the_holdout(snapshot: Path) -> None:
    frame = load_curve(snapshot, SERIES, START, HOLDOUT, final_evaluation=True)
    assert frame.index[-1].strftime("%Y-%m-%d") == "2024-10-02"


def test_missing_series_is_reported(snapshot: Path) -> None:
    with pytest.raises(SnapshotError, match="DGS10"):
        load_curve(snapshot, ("DGS1", "DGS10"), START, HOLDOUT)


def test_a_late_start_is_refused(snapshot: Path) -> None:
    with pytest.raises(DataValidationError, match="starts on 2024-09-24, 23 days after"):
        load_curve(snapshot, SERIES, date(2024, 9, 1), HOLDOUT)


def test_a_hole_in_one_series_is_refused(tmp_path: Path) -> None:
    days = pd.bdate_range("2024-01-01", "2024-09-30")
    full = "".join(f"{d:%Y-%m-%d},4.00\n" for d in days)
    hole_start, hole_end = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-05-31")
    holed = "".join(f"{d:%Y-%m-%d},{'' if hole_start <= d <= hole_end else '3.50'}\n" for d in days)
    target = tmp_path / "snap"
    write_snapshot(
        target,
        {"DGS1": "observation_date,DGS1\n" + full, "DGS2": "observation_date,DGS2\n" + holed},
        "2026-10-02T00:00:00+00:00",
    )
    with pytest.raises(DataValidationError, match="gap of 95 days ending on 2024-06-03"):
        load_curve(target, SERIES, date(2024, 1, 1), HOLDOUT)


def test_no_common_day_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "snap"
    write_snapshot(
        target,
        {
            "DGS1": "observation_date,DGS1\n2024-09-25,4.00\n2024-09-26,\n",
            "DGS2": "observation_date,DGS2\n2024-09-25,\n2024-09-26,3.50\n",
        },
        "2026-10-02T00:00:00+00:00",
    )
    with pytest.raises(DataValidationError, match="no day has a value"):
        load_curve(target, SERIES, START, HOLDOUT)


def test_snapshot_days_cover_the_holdout_without_returning_yields(snapshot: Path) -> None:
    days = snapshot_days(snapshot, SERIES, START)
    assert isinstance(days, pd.DatetimeIndex)
    assert list(days.strftime("%Y-%m-%d")) == [
        "2024-09-25",
        "2024-09-26",
        "2024-09-30",
        "2024-10-01",
        "2024-10-02",
    ]


def test_snapshot_days_refuse_a_hole_in_the_holdout(tmp_path: Path) -> None:
    days = pd.bdate_range("2024-09-02", "2025-03-31")
    full = "".join(f"{d:%Y-%m-%d},4.00\n" for d in days)
    hole_start, hole_end = pd.Timestamp("2024-11-01"), pd.Timestamp("2025-01-31")
    holed = "".join(f"{d:%Y-%m-%d},{'' if hole_start <= d <= hole_end else '3.50'}\n" for d in days)
    target = tmp_path / "snap"
    write_snapshot(
        target,
        {"DGS1": "observation_date,DGS1\n" + full, "DGS2": "observation_date,DGS2\n" + holed},
        "2026-10-02T00:00:00+00:00",
    )
    # The pre-holdout load is fine: the hole is entirely inside the holdout.
    assert len(load_curve(target, SERIES, date(2024, 9, 2), HOLDOUT)) == 21
    with pytest.raises(DataValidationError, match="gap of"):
        snapshot_days(target, SERIES, date(2024, 9, 2))
