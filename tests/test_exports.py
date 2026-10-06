"""The five CSV exports: exact columns, numbers as numbers, periods by the registered boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.criteria import APTO
from termo.descriptive.stages import BASE, Analysis, desc_trial_id
from termo.operation.config import OperationConfig, load_operation_config
from termo.operation.exports import export_all
from termo.operation.macro import _column_name, macro_panel
from termo.operation.monthly import episodes
from termo.validation.trials import TrialLog, TrialStatus

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"
NAMES = ("rally fuerte", "rally moderado", "venta")
BLOCKS = ["nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad"]
HOLDOUT_DAYS = 60  # the registered holdout covers the first 60 labelled days from holdout_start
HASH = "snap-abc"
STAMP = "2026-10-06T12:00:00Z"
READING_COLUMNS = [
    "fecha",
    "fase",
    "nombre",
    "dias_en_fase",
    "inicio_episodio",
    "p_rally_fuerte",
    "p_rally_moderado",
    "p_venta",
    "confianza",
    "baja_confianza",
    "bloque_1",
    "aporte_1",
    "bloque_2",
    "aporte_2",
    "bloque_3",
    "aporte_3",
    "alerta",
    "historia_revisada",
    "validacion",
    "snapshot_hash",
]
HISTORY_COLUMNS = [
    "fecha",
    "fase",
    "nombre",
    "p_rally_fuerte",
    "p_rally_moderado",
    "p_venta",
    *BLOCKS,
    "base",
    "periodo",
]
EPISODE_COLUMNS = [
    "inicio",
    "fin",
    "fase",
    "nombre",
    "dias",
    "cambio_10y_pb",
    "cambio_2y_pb",
    "periodo",
]
ALERT_COLUMNS = ["fecha", "de", "a", "confianza"]


@pytest.fixture(scope="module")
def op() -> OperationConfig:
    return load_operation_config(REPO_OP)


def _analysis(curve: pd.DataFrame) -> Analysis:
    """~400 business days across holdout_start (1998-04-01): phases change every ~35 days."""
    days = curve.loc["1997-10-01":"1999-03-12"].index
    labels = pd.Series((np.arange(len(days)) // 35) % 3, index=days, dtype=int)
    rng = np.random.default_rng(3)
    raw = rng.random((len(days), 3)) + 0.1
    proba = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), index=days, columns=[0, 1, 2])
    blocks = pd.DataFrame(
        rng.normal(size=(len(days), 7)) * 0.3, index=days, columns=[*BLOCKS, BASE]
    )
    top = pd.DataFrame({"top": json.dumps([["d2_63_r252", 0.9]])}, index=days)
    return Analysis(
        labels=labels, frozen_labels=labels, proba=proba, shap_blocks=blocks, shap_top=top
    )


def _model_log(path: Path, config: CoreConfig, holdout_days: int | None) -> TrialLog:
    log = TrialLog(path)
    trial = desc_trial_id(config)
    log.register(trial, "the one model", {"model": "descriptive"}, "snap", "code-1")
    log.record_result(trial, {"files": {}}, TrialStatus.KEPT, "", "labels.csv", "abc")
    log.record_report(
        {
            "stage": "diagnostic",
            "verdict": APTO,
            "final_trial_id": trial,
            "checks": [{"name": "D1_persistence", "passed": True}],
            "by_phase": None,
        }
    )
    if holdout_days is not None:
        log.open_holdout(trial)
        log.record_holdout_result(
            {"stage": "holdout", "verdict": APTO, "checks": [], "days": holdout_days, "files": {}}
        )
    return log


def _reading_record(
    analysis: Analysis, day: str, alert: dict[str, Any] | None, reproduced: bool
) -> dict[str, Any]:
    stamp = pd.Timestamp(day)
    phase = int(analysis.labels.loc[stamp])
    past = analysis.labels.loc[:stamp]
    run_start = past.index[past.ne(phase).to_numpy().nonzero()[0][-1] + 1]
    proba = analysis.proba.loc[stamp]
    blocks = analysis.shap_blocks.loc[stamp].drop(BASE)
    order = blocks.abs().sort_values(ascending=False, kind="stable").index[:3]
    return {
        "kind": "reading",
        "run_at": f"{day}T18:00:00Z",
        "snapshot_hash": f"hash-{day}",
        "reading_date": day,
        "reading": {
            "date": day,
            "phase": phase,
            "phase_name": NAMES[phase],
            "days_in_phase": int(len(past.loc[run_start:])),
            "episode_start": run_start.date().isoformat(),
            "probabilities": {NAMES[p]: float(proba[p]) for p in range(3)},
            "confidence": float(proba[phase]),
            "low_confidence": bool(proba[phase] < 0.4),
            "drivers": [{"block": str(b), "contribution": float(blocks[b])} for b in order],
            "validation": {"diagnostic": APTO, "holdout": APTO, "holdout_seen_by": "x"},
        },
        "files": {},
        "history": {"reproduced": reproduced, "differing": {}},
        "alert": alert,
        "macro": [],
        "code_commit": "code-9",
    }


def _records(analysis: Analysis) -> list[dict[str, Any]]:
    changed = analysis.labels.loc["1999-01-04":"1999-03-05"]
    day_of_change = changed.index[changed.ne(changed.shift()).to_numpy()][1].date().isoformat()
    alert = {"fecha": day_of_change, "de": "venta", "a": "rally fuerte", "confianza": 0.5}
    return [
        _reading_record(analysis, "1999-01-08", None, True),
        _reading_record(analysis, day_of_change, alert, False),
        {"kind": "evaluation", "run_at": "1999-02-10T18:00:00Z", "weeks": 26},
        _reading_record(analysis, "1999-03-05", None, True),
    ]


def _macro_frame(curve: pd.DataFrame, op: OperationConfig) -> pd.DataFrame:
    days = curve.loc["1997-10-01":"1999-03-12"].index
    rng = np.random.default_rng(11)
    columns = {
        _column_name(m): pd.Series(np.cumsum(rng.normal(size=len(days))), index=days)
        for m in op.macro
    }
    frame = pd.DataFrame(columns, index=days)
    first = frame.columns[0]
    frame.loc[days[100], first] = np.nan  # a hole: nothing is filled
    return frame


@pytest.fixture(scope="module")
def exported(
    curve: pd.DataFrame, desc2_config: CoreConfig, op: OperationConfig, tmp_path_factory: Any
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("exports")
    analysis = _analysis(curve)
    log = _model_log(root / "trials.jsonl", desc2_config, HOLDOUT_DAYS)
    records = _records(analysis)
    macro = _macro_frame(curve, op)
    out = root / "csv"
    paths = export_all(records, log, analysis, curve, desc2_config, op, macro, out, HASH, STAMP)
    return {
        "paths": paths,
        "analysis": analysis,
        "records": records,
        "macro": macro,
        "out": out,
        "log": log,
        "root": root,
    }


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def test_five_files_each_with_the_two_comment_lines_in_utf8(exported: dict[str, Any]) -> None:
    paths = exported["paths"]
    assert set(paths) == {
        "lecturas.csv",
        "historia_diaria.csv",
        "episodios.csv",
        "macro.csv",
        "alertas.csv",
    }
    for name, path in paths.items():
        assert path == exported["out"] / name
        text = path.read_bytes().decode("utf-8")
        lines = text.split("\n")
        assert lines[0] == f"# snapshot_hash={HASH}"
        assert lines[1] == f"# generado={STAMP}"
        assert not lines[2].startswith("#") and "," in lines[2]


def test_lecturas_has_one_row_per_reading_with_numbers_as_numbers(
    exported: dict[str, Any], desc2_config: CoreConfig
) -> None:
    table = _read(exported["paths"]["lecturas.csv"])
    records = [r for r in exported["records"] if r["kind"] == "reading"]
    assert list(table.columns) == READING_COLUMNS
    assert len(table) == len(records) == 3
    assert table["fecha"].tolist() == [r["reading_date"] for r in records]
    for column in ("p_rally_fuerte", "p_rally_moderado", "p_venta", "confianza", "aporte_1"):
        assert table[column].dtype == np.float64
    assert table["fase"].dtype == np.int64 and table["dias_en_fase"].dtype == np.int64
    assert table["baja_confianza"].dtype == bool and table["historia_revisada"].dtype == bool
    first = records[0]["reading"]
    row = table.iloc[0]
    assert row["nombre"] == first["phase_name"]
    assert row["inicio_episodio"] == first["episode_start"]
    assert row["dias_en_fase"] == first["days_in_phase"]
    assert row["p_venta"] == pytest.approx(first["probabilities"]["venta"])
    assert row["confianza"] == pytest.approx(first["confidence"])
    for rank, driver in enumerate(first["drivers"], start=1):
        assert row[f"bloque_{rank}"] == driver["block"]
        assert row[f"aporte_{rank}"] == pytest.approx(driver["contribution"])
    assert row["validacion"] == APTO
    assert row["snapshot_hash"] == "hash-1999-01-08"
    # the alert is "de -> a" on its own row only; the revised history flags the same row
    assert table["alerta"].isna().tolist() == [True, False, True]
    assert table.loc[1, "alerta"] == "venta -> rally fuerte"
    assert table["historia_revisada"].tolist() == [False, True, False]


def test_historia_diaria_one_row_per_labelled_day_with_periods_at_the_registered_boundaries(
    exported: dict[str, Any], desc2_config: CoreConfig
) -> None:
    analysis: Analysis = exported["analysis"]
    table = _read(exported["paths"]["historia_diaria.csv"])
    assert list(table.columns) == HISTORY_COLUMNS
    assert len(table) == len(analysis.labels)
    assert table["fecha"].tolist() == [d.date().isoformat() for d in analysis.labels.index]
    assert table["fase"].tolist() == analysis.labels.tolist()
    for column in ("p_venta", "nivel corto", "base"):
        assert table[column].dtype == np.float64
    assert table["base"].tolist() == pytest.approx(analysis.shap_blocks[BASE].tolist())
    assert table["p_rally_moderado"].tolist() == pytest.approx(analysis.proba[1].tolist())

    start = pd.Timestamp(desc2_config.holdout_start)
    registered_end = analysis.labels.loc[start:].index[HOLDOUT_DAYS - 1]
    dates = pd.to_datetime(table["fecha"])
    period = table["periodo"]
    assert (period[dates < start] == "pre_holdout").all()
    assert (period[(dates >= start) & (dates <= registered_end)] == "holdout").all()
    assert (period[dates > registered_end] == "sombra").all()
    assert set(period) == {"pre_holdout", "holdout", "sombra"}
    assert (period == "holdout").sum() == HOLDOUT_DAYS
    # the last pre-holdout day and the first day of the holdout sit on either side
    assert period[dates == dates[dates < start].max()].item() == "pre_holdout"
    assert period[dates == start].item() == "holdout"


def test_episodios_equals_monthly_episodes_plus_the_period_of_the_start(
    exported: dict[str, Any], curve: pd.DataFrame, desc2_config: CoreConfig
) -> None:
    analysis: Analysis = exported["analysis"]
    table = _read(exported["paths"]["episodios.csv"])
    assert list(table.columns) == EPISODE_COLUMNS
    expected = episodes(analysis.labels, curve, NAMES)
    assert len(table) == len(expected)
    assert table["inicio"].tolist() == [d.date().isoformat() for d in expected["inicio"]]
    assert table["fin"].tolist() == [d.date().isoformat() for d in expected["fin"]]
    assert table["fase"].tolist() == expected["fase"].tolist()
    assert table["nombre"].tolist() == expected["nombre"].tolist()
    assert table["dias"].tolist() == expected["dias"].tolist()
    assert table["cambio_10y_pb"].tolist() == pytest.approx(expected["cambio_10y_pb"].tolist())
    assert table["cambio_2y_pb"].tolist() == pytest.approx(expected["cambio_2y_pb"].tolist())
    start = pd.Timestamp(desc2_config.holdout_start)
    end = analysis.labels.loc[start:].index[HOLDOUT_DAYS - 1]
    starts = pd.to_datetime(table["inicio"])
    expected_period = np.where(
        starts < start, "pre_holdout", np.where(starts <= end, "holdout", "sombra")
    )
    assert table["periodo"].tolist() == expected_period.tolist()
    assert set(table["periodo"]) == {"pre_holdout", "holdout", "sombra"}


def test_macro_csv_value_percentile_and_change_per_column_on_every_day(
    exported: dict[str, Any], op: OperationConfig
) -> None:
    frame: pd.DataFrame = exported["macro"]
    table = _read(exported["paths"]["macro.csv"])
    expected_columns = ["fecha"]
    for column in frame.columns:
        expected_columns += [column, f"{column}_percentil_10a", f"{column}_cambio_21d"]
    assert list(table.columns) == expected_columns
    assert len(table) == len(frame)
    assert table["fecha"].tolist() == [d.date().isoformat() for d in frame.index]
    first = frame.columns[0]
    # the hole stays a hole: nothing is filled
    assert table[first].isna().sum() == 1
    assert table.loc[100, [f"{first}_percentil_10a", f"{first}_cambio_21d"]].isna().all()
    # by hand on one day: percentile within the trailing non-NaN values, change 21 rows back
    known = frame[first].dropna()
    day = frame.index[300]
    window = known.loc[:day].iloc[-op.percentile_window_days :]
    row = table.loc[300]
    assert row[first] == pytest.approx(frame.loc[day, first])
    assert row[f"{first}_percentil_10a"] == pytest.approx(float((window <= window.iloc[-1]).mean()))
    assert row[f"{first}_cambio_21d"] == pytest.approx(
        float(window.iloc[-1] - known.loc[:day].iloc[-1 - op.change_days])
    )
    assert table[f"{first}_percentil_10a"].dtype == np.float64


def test_macro_csv_agrees_with_the_panel_on_the_last_day_and_respects_the_window(
    curve: pd.DataFrame, desc2_config: CoreConfig, op: OperationConfig, tmp_path: Path
) -> None:
    short = replace(op, percentile_window_days=252, change_days=5)
    analysis = _analysis(curve)
    frame = _macro_frame(curve, short)
    log = _model_log(tmp_path / "t.jsonl", desc2_config, HOLDOUT_DAYS)
    paths = export_all(
        _records(analysis), log, analysis, curve, desc2_config, short, frame, tmp_path, HASH, STAMP
    )
    table = _read(paths["macro.csv"])
    panel = macro_panel(frame, short, frame.index[-1].date())
    for macro, row in zip(short.macro, panel, strict=True):
        column = _column_name(macro)
        assert table[column].iloc[-1] == pytest.approx(row["valor"])
        assert table[f"{column}_percentil_10a"].iloc[-1] == pytest.approx(row["percentil_10a"])
        assert table[f"{column}_cambio_21d"].iloc[-1] == pytest.approx(row["cambio_21d"])
    # a 252-day window forgets older values: day 300 is ranked against its last 252 only
    column = _column_name(short.macro[0])
    known = frame[column].dropna()
    day = frame.index[300]
    window = known.loc[:day].iloc[-252:]
    full = known.loc[:day]
    expected = float((window <= window.iloc[-1]).mean())
    assert expected != float((full <= full.iloc[-1]).mean())
    assert table.loc[300, f"{column}_percentil_10a"] == pytest.approx(expected)


def test_alertas_has_one_row_per_alert(exported: dict[str, Any]) -> None:
    table = _read(exported["paths"]["alertas.csv"])
    assert list(table.columns) == ALERT_COLUMNS
    assert len(table) == 1
    alert = next(r["alert"] for r in exported["records"] if r.get("alert"))
    assert table.iloc[0]["fecha"] == alert["fecha"]
    assert table.iloc[0]["de"] == "venta" and table.iloc[0]["a"] == "rally fuerte"
    assert table.iloc[0]["confianza"] == pytest.approx(0.5)
    assert table["confianza"].dtype == np.float64


def test_zero_alerts_gives_a_file_with_only_the_header(
    curve: pd.DataFrame, desc2_config: CoreConfig, op: OperationConfig, tmp_path: Path
) -> None:
    analysis = _analysis(curve)
    log = _model_log(tmp_path / "t.jsonl", desc2_config, HOLDOUT_DAYS)
    records = [_reading_record(analysis, "1999-01-08", None, True)]
    paths = export_all(
        records, log, analysis, curve, desc2_config, op, _macro_frame(curve, op), tmp_path / "o",
        HASH, STAMP,
    )  # fmt: skip
    text = paths["alertas.csv"].read_text(encoding="utf-8")
    assert text.splitlines()[2:] == ["fecha,de,a,confianza"]
    assert _read(paths["alertas.csv"]).empty
    assert len(_read(paths["lecturas.csv"])) == 1


def test_without_a_registered_holdout_the_caller_gives_its_end(
    curve: pd.DataFrame, desc2_config: CoreConfig, op: OperationConfig, tmp_path: Path
) -> None:
    analysis = _analysis(curve)
    frame = _macro_frame(curve, op)
    no_result = _model_log(tmp_path / "t.jsonl", desc2_config, None)
    end = analysis.labels.loc["1998-06-30":].index[0]
    for model_log in (None, no_result):
        paths = export_all(
            [], model_log, analysis, curve, desc2_config, op, frame, tmp_path / "o",
            HASH, STAMP, holdout_end=end,
        )  # fmt: skip
        table = _read(paths["historia_diaria.csv"])
        dates = pd.to_datetime(table["fecha"])
        assert (table.loc[dates > end, "periodo"] == "sombra").all()
        assert (
            table.loc[
                (dates >= pd.Timestamp(desc2_config.holdout_start)) & (dates <= end), "periodo"
            ]
            == "holdout"
        ).all()
    with pytest.raises(ValueError, match="holdout"):
        export_all([], None, analysis, curve, desc2_config, op, frame, tmp_path / "o2", HASH, STAMP)
