"""The five CSV exports (spec 4, section 8): what the analysts open in Excel.

Every file is UTF-8, comma separated, and starts with two comment lines (`# snapshot_hash=...`
and `# generado=...`) before the header. Numbers are written as numbers, booleans as
True/False, dates as ISO. Nothing here looks past the data the caller passes in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from termo.config import CoreConfig
from termo.descriptive.stages import BASE, Analysis, holdout_result
from termo.operation.config import OperationConfig
from termo.operation.monthly import READING_KIND, episodes
from termo.reading import governing
from termo.validation.trials import TrialLog

PRE_HOLDOUT, HOLDOUT, SHADOW = "pre_holdout", "holdout", "sombra"
DRIVER_COLUMNS = 3
ALERT_COLUMNS = ("fecha", "de", "a", "confianza")


def _names(config: CoreConfig) -> tuple[str, ...]:
    desc = config.descriptive
    if desc is None:
        raise ValueError("the configuration has no descriptive section")
    return tuple(desc.phase_names)


def _probability_column(name: str) -> str:
    return "p_" + name.replace(" ", "_")


def _registered_holdout_end(
    model_log: TrialLog | None, analysis: Analysis, config: CoreConfig
) -> pd.Timestamp | None:
    """Last day of the registered holdout: its first `days` labelled days from holdout_start."""
    if model_log is None:
        return None
    result = holdout_result(model_log)
    if result is None:
        return None
    days = int(result["days"])
    held = analysis.labels.loc[pd.Timestamp(config.holdout_start) :].index
    if days < 1 or days > len(held):
        raise ValueError(
            f"the registered holdout has {days} days but the analysis has {len(held)} "
            "labelled days from holdout_start"
        )
    return pd.Timestamp(held[days - 1])


def _periods(
    dates: pd.DatetimeIndex, holdout_start: pd.Timestamp, holdout_end: pd.Timestamp
) -> np.ndarray:
    """pre_holdout before holdout_start, holdout through the registered end, sombra after."""
    return np.where(
        dates < holdout_start,
        PRE_HOLDOUT,
        np.where(dates <= holdout_end, HOLDOUT, SHADOW),
    )


def _write(path: Path, table: pd.DataFrame, snapshot_hash: str, generated_at: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# snapshot_hash={snapshot_hash}\n# generado={generated_at}\n")
        table.to_csv(handle, index=False, lineterminator="\n")


def _iso(dates: pd.Index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in dates]


def _readings(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> pd.DataFrame:
    columns = [
        "fecha",
        "fase",
        "nombre",
        "dias_en_fase",
        "inicio_episodio",
        *(_probability_column(n) for n in names),
        "confianza",
        "baja_confianza",
        *(
            column
            for rank in range(1, DRIVER_COLUMNS + 1)
            for column in (f"bloque_{rank}", f"aporte_{rank}")
        ),
        "alerta",
        "historia_revisada",
        "validacion",
        "snapshot_hash",
    ]
    rows: list[dict[str, Any]] = []
    for record in sorted(
        (r for r in records if r.get("kind") == READING_KIND), key=lambda r: str(r["reading_date"])
    ):
        reading = record["reading"]
        alert = record.get("alert")
        row: dict[str, Any] = {
            "fecha": str(record["reading_date"]),
            "fase": int(reading["phase"]),
            "nombre": str(reading["phase_name"]),
            "dias_en_fase": int(reading["days_in_phase"]),
            "inicio_episodio": str(reading["episode_start"]),
        }
        row.update({_probability_column(n): float(reading["probabilities"][n]) for n in names})
        row["confianza"] = float(reading["confidence"])
        row["baja_confianza"] = bool(reading["low_confidence"])
        drivers = reading["drivers"]
        for rank in range(1, DRIVER_COLUMNS + 1):
            driver = drivers[rank - 1] if rank <= len(drivers) else None
            row[f"bloque_{rank}"] = "" if driver is None else str(driver["block"])
            row[f"aporte_{rank}"] = np.nan if driver is None else float(driver["contribution"])
        row["alerta"] = "" if alert is None else f"{alert['de']} -> {alert['a']}"
        row["historia_revisada"] = not bool(record["history"]["reproduced"])
        row["validacion"] = governing(reading["validation"])[1]
        row["snapshot_hash"] = str(record["snapshot_hash"])
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _history(
    analysis: Analysis, names: Sequence[str], holdout_start: pd.Timestamp, holdout_end: pd.Timestamp
) -> pd.DataFrame:
    labels = analysis.labels
    table = pd.DataFrame(
        {
            "fecha": _iso(labels.index),
            "fase": labels.to_numpy(),
            "nombre": [names[p] for p in labels],
        }
    )
    for phase, name in enumerate(names):
        table[_probability_column(name)] = analysis.proba[phase].to_numpy(dtype=float)
    blocks = analysis.shap_blocks.reindex(labels.index)
    for block in blocks.columns.drop(BASE):
        table[str(block)] = blocks[block].to_numpy(dtype=float)
    table["base"] = blocks[BASE].to_numpy(dtype=float)
    table["periodo"] = _periods(pd.DatetimeIndex(labels.index), holdout_start, holdout_end)
    return table


def _episodes(
    analysis: Analysis,
    curve: pd.DataFrame,
    names: Sequence[str],
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> pd.DataFrame:
    table = episodes(analysis.labels, curve, names)
    starts = pd.DatetimeIndex(table["inicio"])
    table["periodo"] = _periods(starts, holdout_start, holdout_end)
    table["inicio"] = _iso(starts)
    table["fin"] = _iso(pd.DatetimeIndex(table["fin"]))
    return table


def _rolling_percentile(window: np.ndarray) -> float:
    return float((window <= window[-1]).mean())


def _macro(frame: pd.DataFrame, op: OperationConfig) -> pd.DataFrame:
    """Value, trailing percentile and change per column, every day; holes stay holes."""
    table = pd.DataFrame({"fecha": _iso(frame.index)})
    for column in frame.columns:
        known = frame[column].dropna()
        percentile = known.rolling(op.percentile_window_days, min_periods=1).apply(
            _rolling_percentile, raw=True
        )
        change = known - known.shift(op.change_days)
        table[str(column)] = frame[column].to_numpy(dtype=float)
        table[f"{column}_percentil_10a"] = percentile.reindex(frame.index).to_numpy(dtype=float)
        table[f"{column}_cambio_21d"] = change.reindex(frame.index).to_numpy(dtype=float)
    return table


def _alerts(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "fecha": str(r["alert"]["fecha"]),
            "de": str(r["alert"]["de"]),
            "a": str(r["alert"]["a"]),
            "confianza": float(r["alert"]["confianza"]),
        }
        for r in sorted(
            (r for r in records if r.get("kind") == READING_KIND and r.get("alert") is not None),
            key=lambda r: str(r["reading_date"]),
        )
    ]
    return pd.DataFrame(rows, columns=list(ALERT_COLUMNS))


def export_all(
    shadow_records: Sequence[Mapping[str, Any]],
    model_log: TrialLog | None,
    analysis: Analysis,
    curve: pd.DataFrame,
    config: CoreConfig,
    op: OperationConfig,
    macro: pd.DataFrame,
    out_dir: Path,
    snapshot_hash: str,
    generated_at: str,
    holdout_end: pd.Timestamp | None = None,
) -> dict[str, Path]:
    """Write the five CSV into `out_dir` and return `{file name: path}`.

    The periodo boundaries are `config.holdout_start` and the registered holdout end: the
    last of the first `days` labelled days from holdout_start that the model log's holdout
    result recorded. Without that result the caller must pass `holdout_end`.
    """
    names = _names(config)
    end = _registered_holdout_end(model_log, analysis, config)
    if end is None:
        end = holdout_end
    if end is None:
        raise ValueError(
            "the registered holdout end is unknown: the model log has no holdout result "
            "and no holdout_end was given"
        )
    start = pd.Timestamp(config.holdout_start)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "lecturas.csv": _readings(shadow_records, names),
        "historia_diaria.csv": _history(analysis, names, start, end),
        "episodios.csv": _episodes(analysis, curve, names, start, end),
        "macro.csv": _macro(macro, op),
        "alertas.csv": _alerts(shadow_records),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        paths[name] = out_dir / name
        _write(paths[name], table, snapshot_hash, generated_at)
    return paths
