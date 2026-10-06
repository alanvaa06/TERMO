from __future__ import annotations

from pathlib import Path

import pytest

from termo.data.snapshot import (
    SnapshotError,
    read_snapshot,
    snapshot_downloaded_at,
    snapshot_hash,
    write_snapshot,
)

FILES = {
    "DGS1": "observation_date,DGS1\n1977-02-15,5.39\n",
    "DGS2": "observation_date,DGS2\n1977-02-15,5.95\n",
}


def test_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "2026-10-02"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    assert read_snapshot(target) == FILES


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "2026-10-02"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    with pytest.raises(FileExistsError):
        write_snapshot(target, FILES, "2026-10-03T00:00:00+00:00")


def test_detects_a_modified_file(tmp_path: Path) -> None:
    target = tmp_path / "snap"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    (target / "DGS1.csv").write_bytes(b"observation_date,DGS1\n1977-02-15,9.99\n")
    with pytest.raises(SnapshotError, match="hash mismatch for DGS1"):
        read_snapshot(target)


def test_detects_a_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "snap"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    (target / "DGS2.csv").unlink()
    with pytest.raises(SnapshotError, match="missing file for DGS2"):
        read_snapshot(target)


def test_hash_depends_on_content_not_on_download_time(tmp_path: Path) -> None:
    write_snapshot(tmp_path / "a", FILES, "2026-10-02T00:00:00+00:00")
    write_snapshot(tmp_path / "b", FILES, "2026-11-30T00:00:00+00:00")
    write_snapshot(tmp_path / "c", {**FILES, "DGS1": FILES["DGS1"] + "1977-02-16,5.40\n"}, "x")
    assert snapshot_hash(tmp_path / "a") == snapshot_hash(tmp_path / "b")
    assert snapshot_hash(tmp_path / "a") != snapshot_hash(tmp_path / "c")


def test_download_time_is_read_from_the_manifest(tmp_path: Path) -> None:
    write_snapshot(tmp_path / "a", FILES, "2026-10-02T00:00:00+00:00")
    assert snapshot_downloaded_at(tmp_path / "a") == "2026-10-02T00:00:00+00:00"
    with pytest.raises(SnapshotError, match="no manifest"):
        snapshot_downloaded_at(tmp_path / "missing")
