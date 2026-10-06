"""One snapshot holds the curve series and the macro context series (spec 4)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from termo.config import CoreConfig
from termo.data.fred import fetch_series_csv, parse_series_csv
from termo.data.loader import check_coverage, complete_days
from termo.data.snapshot import write_snapshot
from termo.operation.config import OperationConfig


def take_operation_snapshot(
    config: CoreConfig,
    op: OperationConfig,
    snapshot_dir: Path,
    get: Callable[[str], str],
    downloaded_at: str,
) -> None:
    """Download curve and macro series; nothing is written unless the curve passes coverage.

    Macro series are parsed to prove they are valid FRED CSV, but a hole in them is not
    refused: the panel notes the last available value instead.
    """
    texts: dict[str, str] = {}
    parsed: list[pd.Series] = []
    for series_id in config.series:
        text = fetch_series_csv(series_id, get)
        parsed.append(parse_series_csv(text, series_id))
        texts[series_id] = text
    check_coverage(complete_days(parsed, config.start), config.start)
    for macro in op.macro:
        text = fetch_series_csv(macro.series_id, get)
        parse_series_csv(text, macro.series_id)
        texts[macro.series_id] = text
    write_snapshot(snapshot_dir, texts, downloaded_at)
