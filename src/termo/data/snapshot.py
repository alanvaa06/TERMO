"""Immutable, hash-checked copies of the downloaded CSV files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

MANIFEST = "manifest.json"


class SnapshotError(RuntimeError):
    """The snapshot on disk is incomplete or does not match its manifest."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(snapshot_dir: Path) -> dict[str, str]:
    path = snapshot_dir / MANIFEST
    if not path.exists():
        raise SnapshotError(f"no manifest in {snapshot_dir}")
    files: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))["files"]
    return files


def write_snapshot(
    snapshot_dir: Path, csv_by_series: Mapping[str, str], downloaded_at: str
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    files: dict[str, str] = {}
    for series_id, text in sorted(csv_by_series.items()):
        data = text.encode("utf-8")
        # Bytes, not text: Windows newline translation would change the hash.
        (snapshot_dir / f"{series_id}.csv").write_bytes(data)
        files[series_id] = _sha256(data)
    manifest = {"downloaded_at": downloaded_at, "files": files}
    (snapshot_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )


def read_snapshot(snapshot_dir: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for series_id, expected in _read_manifest(snapshot_dir).items():
        path = snapshot_dir / f"{series_id}.csv"
        if not path.exists():
            raise SnapshotError(f"missing file for {series_id}")
        data = path.read_bytes()
        if _sha256(data) != expected:
            raise SnapshotError(f"hash mismatch for {series_id}")
        texts[series_id] = data.decode("utf-8")
    return texts


def snapshot_hash(snapshot_dir: Path) -> str:
    files = _read_manifest(snapshot_dir)
    joined = "".join(f"{series_id}:{digest};" for series_id, digest in sorted(files.items()))
    return _sha256(joined.encode("utf-8"))
