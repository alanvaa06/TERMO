"""The weekly sheet in Spanish (spec 4, section 4): Markdown for the committee, plus its JSON.

The sheet describes the phase the curve is in on the reading date. It prints nothing about
days after that date and no forecast; the fixed texts come from the operation configuration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from termo.descriptive.criteria import APTO
from termo.operation.config import OperationConfig
from termo.reading import LOST_HOLDOUT, governing

FIDELITY_WARNING = "[FIDELIDAD DEL IMITADOR FALLIDA: explicaciones no validadas]"
SEEN_BY_NOTE = "Nombres de fase elegidos tras el holdout; los valida solo la sombra."
DASH = "-"


def _num(value: float | None, digits: int = 2) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return DASH if value is None else f"{value * 100:.0f}%"


def _banner(validation: Mapping[str, Any]) -> list[str]:
    period, verdict = governing(validation)
    if verdict == APTO:
        return []
    period_es = "holdout" if period == "holdout" else "diagnóstico previo al holdout"
    failed = ", ".join(validation["failed_checks"]) or "ninguno evaluable"
    return [f"**[NO VALIDADO: {period_es} {verdict.upper()}; checks fallidos: {failed}]**", ""]


def _history_banner(history: Mapping[str, Any]) -> list[str]:
    if history["reproduced"]:
        return []
    days = {d for dates in history["differing"].values() for d in dates}
    return [f"**HISTORIA REVISADA**: {len(days)} días difieren de la historia registrada", ""]


def _holdout_text(holdout: str | None) -> str:
    if holdout is None:
        return "no abierto"
    return "abierto sin resultado (perdido)" if holdout == LOST_HOLDOUT else holdout.upper()


def _phase_table(by_phase: list[Mapping[str, Any]] | None) -> list[str]:
    if by_phase is None:
        return ["- Tabla por fase: no registrada"]
    lines = [
        "| Fase | Días | Evaluable | Duración mediana (días) | Dirección | Acierto |",
        "|---|---|---|---|---|---|",
    ]
    for row in by_phase:
        lines.append(
            f"| {row['name']} | {row['days']} | {'sí' if row['evaluable'] else 'no'} "
            f"| {_num(row['median_duration_days'], 1)} | {_num(row['direction_share'])} "
            f"| {_num(row['recall'])} |"
        )
    return lines


def render_sheet(record: Mapping[str, Any], op: OperationConfig) -> str:
    reading = record["reading"]
    status = reading["validation"]
    shadow = status["sombra"]
    lines = [
        f"# TERMO — hoja semanal del {record['reading_date']}",
        "",
        f"Snapshot del {str(record['run_at'])[:10]} (hash {str(record['snapshot_hash'])[:12]})",
        "",
    ]
    alert = record["alert"]
    if alert is not None:
        change = f"{alert['de']} → {alert['a']} (confianza {alert['confianza']:.2f})"
        lines += [f"**CAMBIO DE FASE: {change}**", ""]
    lines += _banner(status)
    lines += _history_banner(record["history"])

    low = "  [CONFIANZA BAJA]" if reading["low_confidence"] else ""
    lines += [
        "## Fase actual",
        "",
        f"**{reading['phase_name']}** desde {reading['episode_start']} "
        f"({reading['days_in_phase']} días hábiles)",
        "",
        f"Confianza del imitador: {reading['confidence']:.2f}{low}",
        "",
        op.texts["nota_confianza"],
        "",
        "| Fase | Probabilidad |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.2f} |" for name, value in reading["probabilities"].items()]

    lines += ["", "## Qué la empuja", ""]
    if not reading["drivers_validated"]:
        lines += [FIDELITY_WARNING, ""]
    lines += [f"- {d['block']}: {d['contribution']:+.2f}" for d in reading["drivers"]]
    lines += ["", "Variables con mayor contribución:", ""]
    lines += [f"- {v['variable']}: {v['contribution']:+.2f}" for v in reading["top_variables"]]

    lines += [
        "",
        "## Contexto macro",
        "",
        "| Serie | Valor | Percentil (10 años) | Cambio (21 días hábiles) | Nota |",
        "|---|---|---|---|---|",
    ]
    for row in record["macro"]:
        lines.append(
            f"| {row['serie']} | {_num(row['valor'])} | {_pct(row['percentil_10a'])} "
            f"| {_num(row['cambio_21d'])} | {row['nota']} |"
        )

    registered = ", ".join(
        f"{v['stage']} {str(v['verdict']).upper()}" for v in status["registered_verdicts"]
    )
    weeks_left = shadow["proxima_evaluacion_semanas"]
    next_eval = "ya puede correrse" if weeks_left == 0 else f"en {weeks_left} semanas"
    lines += [
        "",
        "## Validación",
        "",
        f"- Veredictos registrados: {registered}",
        f"- Holdout: {_holdout_text(status['holdout'])}",
        "",
    ]
    lines += _phase_table(status["by_phase"])
    lines += [
        "",
        f"Semanas de sombra: {shadow['semanas']}",
        "",
        f"Próxima evaluación de sombra {next_eval}",
    ]
    if "holdout_seen_by" in status:
        lines += ["", SEEN_BY_NOTE]
    lines += ["", op.texts["descargo"], "", f"Generado con código {record['code_commit']}", ""]
    return "\n".join(lines)


def write_sheet(record: Mapping[str, Any], op: OperationConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / str(record["reading_date"])
    path = stem.with_suffix(".md")
    path.write_text(render_sheet(record, op), encoding="utf-8", newline="\n")
    stem.with_suffix(".json").write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return path
