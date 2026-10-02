from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from termo.data.loader import HoldoutAccessError, load_curve
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
