"""The context panel: macro series read from the snapshot, never later than the reading day."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.data.fred import parse_series_csv
from termo.data.loader import load_curve
from termo.data.snapshot import read_snapshot
from termo.operation.config import MacroSeries, OperationConfig

# A term premium or a spread may be negative (Kim-Wright THREEFYTP10 has many negative
# observations); the curve series keep the yield bounds of the parser.
MACRO_BOUNDS = (-25.0, 25.0)


def _column_name(macro: MacroSeries) -> str:
    if macro.spread_against is None:
        return macro.series_id
    return f"{macro.spread_against}_menos_{macro.series_id}"


def _label(macro: MacroSeries) -> str:
    if macro.spread_against is None:
        return macro.name
    return f"{macro.spread_against} - {macro.name}"


def macro_frame(snapshot_dir: Path, config: CoreConfig, op: OperationConfig) -> pd.DataFrame:
    """Macro columns on the curve's business days (weekend rows dropped, nothing filled).

    The whole history is read on purpose: the operation stage works on every day.
    """
    curve = load_curve(
        snapshot_dir, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    texts = read_snapshot(snapshot_dir)
    columns: dict[str, pd.Series] = {}
    for macro in op.macro:
        series = parse_series_csv(
            texts[macro.series_id], macro.series_id, bounds=MACRO_BOUNDS
        ).reindex(curve.index)
        if macro.spread_against is not None:
            series = curve[macro.spread_against] - series
        columns[_column_name(macro)] = series
    return pd.DataFrame(columns, index=curve.index)


def macro_panel(frame: pd.DataFrame, op: OperationConfig, day: date) -> list[dict[str, Any]]:
    """Value, percentile and change per macro series, using rows <= day only."""
    stamp = pd.Timestamp(day)
    rows: list[dict[str, Any]] = []
    for macro in op.macro:
        known = frame[_column_name(macro)].loc[:stamp].dropna()
        value = float(known.iloc[-1])
        value_date = known.index[-1]
        window = known.iloc[-op.percentile_window_days :]
        change = (
            value - float(known.iloc[-1 - op.change_days])
            if len(known) > op.change_days
            else float("nan")
        )
        note = (
            ""
            if value_date == stamp
            else f"último dato disponible: {value_date.date().isoformat()}"
        )
        rows.append(
            {
                "serie": _label(macro),
                "valor": value,
                "fecha_valor": value_date.date().isoformat(),
                "percentil_10a": float((window <= value).mean()),
                "cambio_21d": change,
                "dias_de_ventana": len(window),
                "nota": note,
            }
        )
    return rows
