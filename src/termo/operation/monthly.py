"""The monthly report (spec 4, section 5): the month's readings and the past's statistics.

Everything here describes days up to the last labelled day of the month. The durations
and transitions are frequencies of the past and carry the label the operation
configuration fixes; nothing is conditioned on what happens after a reading date.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.descriptive.stages import BASE, Analysis, validation_from_log
from termo.operation.config import OperationConfig
from termo.validation.metrics import BP_PER_PERCENT, run_lengths
from termo.validation.trials import TrialLog

LEVEL_10Y, LEVEL_2Y = "DGS10", "DGS2"
READING_KIND = "reading"
EPISODE_COLUMNS = ("inicio", "fin", "fase", "nombre", "dias", "cambio_10y_pb", "cambio_2y_pb")


def episodes(labels: pd.Series, curve: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """One row per run of the same phase, with the 10Y and 2Y change inside the run (bp)."""
    rows: list[dict[str, Any]] = []
    position = 0
    for state, length in run_lengths(labels.to_numpy()):
        start, end = labels.index[position], labels.index[position + length - 1]
        rows.append(
            {
                "inicio": start,
                "fin": end,
                "fase": state,
                "nombre": names[state],
                "dias": length,
                "cambio_10y_pb": BP_PER_PERCENT
                * float(curve.loc[end, LEVEL_10Y] - curve.loc[start, LEVEL_10Y]),
                "cambio_2y_pb": BP_PER_PERCENT
                * float(curve.loc[end, LEVEL_2Y] - curve.loc[start, LEVEL_2Y]),
            }
        )
        position += length
    return pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))


def durations_by_phase(table: pd.DataFrame, names: Sequence[str]) -> list[dict[str, Any]]:
    """Episodes, median and quartiles of the run length per phase; None without episodes."""
    rows: list[dict[str, Any]] = []
    for phase, name in enumerate(names):
        lengths = table.loc[table["fase"] == phase, "dias"]
        empty = lengths.empty
        rows.append(
            {
                "fase": phase,
                "nombre": name,
                "episodios": int(len(lengths)),
                "mediana_dias": None if empty else float(lengths.median()),
                "p25_dias": None if empty else float(lengths.quantile(0.25)),
                "p75_dias": None if empty else float(lengths.quantile(0.75)),
            }
        )
    return rows


def transitions(table: pd.DataFrame, names: Sequence[str]) -> list[dict[str, Any]]:
    """Per "from" phase: episodes with a successor and the share that went to each phase.

    The last episode has no successor. A phase that never had one gives None shares.
    """
    sequence = table["fase"].tolist()
    pairs = list(zip(sequence[:-1], sequence[1:], strict=True))
    rows: list[dict[str, Any]] = []
    for phase, name in enumerate(names):
        successors = [to for origin, to in pairs if origin == phase]
        total = len(successors)
        rows.append(
            {
                "fase": phase,
                "nombre": name,
                "total": total,
                "a": {
                    target: None if total == 0 else successors.count(index) / total
                    for index, target in enumerate(names)
                },
            }
        )
    return rows


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    period = pd.Period(month, freq="M")
    return period.start_time, period.end_time


def _in_month(day: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return start <= pd.Timestamp(day) <= end


def build_monthly(
    month: str,
    shadow_records: Sequence[Mapping[str, Any]],
    model_log: TrialLog,
    analysis: Analysis,
    curve: pd.DataFrame,
    config: CoreConfig,
    op: OperationConfig,
    macro_rows: Sequence[Mapping[str, Any]],
    code_commit: str,
) -> dict[str, Any]:
    """The report's payload: the month's readings, the phase at its close, the past's
    durations and transitions up to that close, the macro panel the caller computed."""
    desc = config.descriptive
    if desc is None:
        raise ValueError("the configuration has no descriptive section")
    names = desc.phase_names
    start, end = _month_bounds(month)
    in_month = analysis.labels.loc[start:end]
    if in_month.empty:
        raise ValueError(f"no labelled days in {month}")
    last_day = in_month.index[-1]
    past = analysis.labels.loc[:last_day]
    phase, days_in_phase = run_lengths(past.to_numpy())[-1]

    readings = sorted(
        (r for r in shadow_records if r.get("kind") == READING_KIND),
        key=lambda r: str(r["reading_date"]),
    )
    month_readings = [r for r in readings if _in_month(str(r["reading_date"]), start, end)]
    lecturas = [
        {
            "fecha": str(r["reading_date"]),
            "fase": int(r["reading"]["phase"]),
            "nombre": str(r["reading"]["phase_name"]),
            "confianza": float(r["reading"]["confidence"]),
            "baja_confianza": bool(r["reading"]["low_confidence"]),
        }
        for r in month_readings
    ]
    alertas = [dict(r["alert"]) for r in month_readings if r.get("alert") is not None]
    dates = [pd.Timestamp(r["fecha"]) for r in lecturas]
    blocks = analysis.shap_blocks.drop(columns=BASE)
    bloques_promedio = (
        {} if not dates else {str(b): float(v) for b, v in blocks.loc[dates].mean().items()}
    )
    history = episodes(past, curve, names)
    return {
        "mes": month,
        "ultimo_dia": last_day.date().isoformat(),
        "fase_cierre": int(phase),
        "nombre_cierre": names[phase],
        "desde": past.index[-days_in_phase].date().isoformat(),
        "dias_en_fase": int(days_in_phase),
        "lecturas": lecturas,
        "alertas": alertas,
        "bloques_promedio": bloques_promedio,
        "duraciones": durations_by_phase(history, names),
        "transiciones": transitions(history, names),
        "rotulo": op.texts["rotulo_historia"],
        "macro": [dict(row) for row in macro_rows],
        "validacion": validation_from_log(model_log, config),
        "no_dice": op.texts["no_dice"],
        "descargo": op.texts["descargo"],
        "generado_con": {"code_commit": code_commit},
    }


def _cell(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _share(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _phase_rows_es(by_phase: Iterable[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Fase | Días | Evaluable | Duración mediana (días) | Proporción de dirección | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for row in by_phase:
        lines.append(
            f"| {row['name']} | {row['days']} | {'sí' if row['evaluable'] else 'no'} "
            f"| {_cell(row['median_duration_days'], 1)} | {_cell(row['direction_share'], 2)} "
            f"| {_cell(row['recall'], 2)} |"
        )
    return lines


def _readings_section(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## Lecturas del mes", ""]
    if not payload["lecturas"]:
        return lines + ["Sin lecturas de sombra en el mes."]
    lines += ["| Fecha | Fase | Confianza |", "|---|---|---|"]
    for reading in payload["lecturas"]:
        confidence = f"{reading['confianza']:.2f}" + (
            " (baja)" if reading["baja_confianza"] else ""
        )
        lines.append(f"| {reading['fecha']} | {reading['nombre']} | {confidence} |")
    return lines


def _alerts_section(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## Alertas del mes", ""]
    if not payload["alertas"]:
        return lines + ["Sin cambios de fase con confianza suficiente en el mes."]
    return lines + [
        f"- {a['fecha']}: {a['de']} → {a['a']} (confianza {a['confianza']:.2f})"
        for a in payload["alertas"]
    ]


def _blocks_section(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## Qué la empujó", "", "Contribución media por bloque en las lecturas del mes:", ""]
    if not payload["bloques_promedio"]:
        return lines[:2] + ["Sin lecturas en el mes: no hay contribuciones que promediar."]
    return lines + [
        f"- {block}: {value:+.2f}" for block, value in payload["bloques_promedio"].items()
    ]


def _history_sections(payload: Mapping[str, Any], label: str) -> list[str]:
    lines = [f"**{label}**", "", "## Duraciones históricas", ""]
    lines += [
        "| Fase | Episodios | Mediana (días) | P25 (días) | P75 (días) |",
        "|---|---|---|---|---|",
    ]
    for row in payload["duraciones"]:
        lines.append(
            f"| {row['nombre']} | {row['episodios']} | {_cell(row['mediana_dias'], 1)} "
            f"| {_cell(row['p25_dias'], 1)} | {_cell(row['p75_dias'], 1)} |"
        )
    names = [row["nombre"] for row in payload["transiciones"]]
    lines += ["", f"**{label}**", "", "## Transiciones", ""]
    lines += [
        "| De \\ A | " + " | ".join(names) + " | Episodios con sucesor |",
        "|---|" + "---|" * (len(names) + 1),
    ]
    for row in payload["transiciones"]:
        shares = " | ".join(_share(row["a"][name]) for name in names)
        lines.append(f"| {row['nombre']} | {shares} | {row['total']} |")
    return lines


def _macro_section(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## Contexto macro", ""]
    if not payload["macro"]:
        return lines + ["Sin series macro en el panel."]
    lines += [
        "| Serie | Valor | Percentil (10 años) | Cambio (21 días) | Nota |",
        "|---|---|---|---|---|",
    ]
    for row in payload["macro"]:
        lines.append(
            f"| {row['serie']} | {_cell(row['valor'], 2)} | {_share(row['percentil_10a'])} "
            f"| {_cell(row['cambio_21d'], 2)} | {row['nota'] or '-'} |"
        )
    return lines


def _validation_section(payload: Mapping[str, Any]) -> list[str]:
    status = payload["validacion"]
    registered = ", ".join(
        f"{v['stage']} {str(v['verdict']).upper()}" for v in status["registered_verdicts"]
    )
    lines = ["## Validación", "", f"- Veredictos registrados: {registered}"]
    if "holdout_seen_by" in status:
        lines.append(
            f"- Holdout ya visto por {status['holdout_seen_by']}: la única validación de estos "
            "nombres es el periodo de sombra."
        )
    if status["by_phase"] is None:
        lines.append("- Por fase: no registrado por este registro")
    else:
        lines += ["", "Por fase (periodo que gobierna):", "", *_phase_rows_es(status["by_phase"])]
    return lines


def render_monthly(payload: Mapping[str, Any], op: OperationConfig) -> str:
    """Spanish Markdown of the monthly report. UTF-8; nothing about days after the close."""
    label = op.texts["rotulo_historia"]
    lines = [
        f"# TERMO — ficha mensual {payload['mes']}",
        "",
        "## Fase al cierre del mes",
        "",
        f"Fase al {payload['ultimo_dia']}: **{payload['nombre_cierre']}**, "
        f"desde {payload['desde']} ({payload['dias_en_fase']} días hábiles).",
        "",
        *_readings_section(payload),
        "",
        *_alerts_section(payload),
        "",
        *_blocks_section(payload),
        "",
        *_history_sections(payload, label),
        "",
        *_macro_section(payload),
        "",
        *_validation_section(payload),
        "",
        payload["no_dice"],
        "",
        payload["descargo"],
        "",
        f"Generado con código {payload['generado_con']['code_commit']}",
        "",
    ]
    return "\n".join(lines)


def write_monthly(payload: Mapping[str, Any], op: OperationConfig, out_dir: Path) -> Path:
    """Write `<mes>.md` and `<mes>.json` (UTF-8, accents kept); return the Markdown path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / str(payload["mes"])
    markdown = stem.with_suffix(".md")
    markdown.write_text(render_monthly(payload, op), encoding="utf-8", newline="\n")
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return markdown
