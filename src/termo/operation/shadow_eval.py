"""The shadow evaluation (spec 4, section 6): the registered criteria on the shadow days.

The days after the registered holdout are the only data nobody looked at when the phase
names of this registry were chosen. The criteria, thresholds and the not-evaluable rule are
the registered ones; D5 compares the walk-forward phases with a jump model fitted once on
everything through the holdout end. The verdict is the shadow's own: no earlier verdict is
rewritten, and the model trial log is read, never written.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.core import registered_columns, verify_configuration
from termo.dataset import prepare
from termo.descriptive.criteria import calibration, evaluate, phase_table, verdict
from termo.descriptive.stages import (
    CALIBRATION_BINS,
    HOLDOUT_DIR,
    Analysis,
    disclosure,
    ensure_writable,
    frozen_map,
    holdout_result,
    limits_of,
    load_analysis,
    restrict,
)
from termo.operation.config import OperationConfig
from termo.operation.shadow import ONE_DAY, ShadowLog
from termo.validation.trials import TrialLog, TrialLogError

EVALUATION = "evaluation"
STAGE = "sombra"
REPORT_PREFIX = "sombra_eval_"
SHADOW_EVAL_LIMIT = (
    "La evaluación de sombra valida los nombres de fase elegidos tras el holdout; "
    "es la única evidencia limpia de desc2."
)
RESULT_ES = {True: "pasa", False: "FALLA", None: "no evaluable"}
VERDICT_ES = {"apto": "APTO", "no-apto": "NO APTO"}
DASH = "-"


def _holdout_end(log: TrialLog, trials_dir: Path) -> pd.Timestamp:
    """The last day of the registered holdout labels, read from the logged files only."""
    holdout = holdout_result(log)
    if holdout is None:
        raise TrialLogError("no holdout result in the log: the shadow evaluates a finished one")
    labels = load_analysis(trials_dir / HOLDOUT_DIR, holdout["files"]).labels
    return pd.Timestamp(labels.index[-1])


def shadow_config(config: CoreConfig, hold_end: pd.Timestamp) -> CoreConfig:
    """The registered configuration with the frozen cutoff moved to the holdout end.

    The shadow days are this evaluation's holdout, so its start moves to the day after
    `hold_end` (the configuration refuses a frozen cutoff inside the holdout; `prepare`
    never reads the holdout start). Nothing else changes.
    """
    end = hold_end.date()
    return replace(config, holdout_start=end + timedelta(days=1), frozen_train_end=end)


def _frozen_through(
    curve: pd.DataFrame, config: CoreConfig, columns: tuple[str, ...], hold_end: pd.Timestamp
) -> pd.Series:
    """The phases of a jump model fitted once on data through `hold_end`, the later days only."""
    frozen = frozen_map(prepare(curve, shadow_config(config, hold_end), columns))
    return frozen.loc[frozen.index > hold_end]


def _cell(value: float | None, digits: int) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _phase_rows(by_phase: Iterable[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Fase | Días | Evaluable | Duración mediana (días) | Dirección | Acierto | Episodios |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in by_phase:
        lines.append(
            f"| {row['name']} | {row['days']} | {'sí' if row['evaluable'] else 'no'} "
            f"| {_cell(row['median_duration_days'], 1)} | {_cell(row['direction_share'], 2)} "
            f"| {_cell(row['recall'], 2)} | {row['episodes']} |"
        )
    return lines


def render_evaluation(payload: Mapping[str, Any], op: OperationConfig) -> str:
    """Spanish Markdown of a shadow evaluation."""
    period = payload["period"]
    lines = [
        f"# TERMO — evaluación de sombra al {period['hasta']}",
        "",
        f"**Veredicto: {VERDICT_ES[str(payload['verdict'])]}** "
        f"(sombra, {payload['weeks']} semanas, {payload['days']} días)",
        "",
        f"Período evaluado: del {period['desde']} al {period['hasta']}, "
        "los días posteriores al holdout registrado.",
        "",
        "| Criterio | Valor | Requisito | Resultado |",
        "|---|---|---|---|",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| {check['name']} | {_cell(check['value'], 4)} | {check['requirement']} "
            f"| {RESULT_ES[check['passed']]} |"
        )
    lines += ["", "## Por fase", "", *_phase_rows(payload["by_phase"])]
    lines += ["", "## Lo que estas pruebas no demuestran", ""]
    lines += [f"- {limit}" for limit in payload["limits"]]
    found = payload["disclosure"]
    lines += ["", "## Divulgación", "", f"- Pruebas en esta bitácora: {found['n_trials_this_log']}"]
    lines += [
        f"- {prior['path']}: {prior['n_trials']} pruebas, veredicto {prior['verdict']}"
        for prior in found["prior_logs"]
    ]
    lines += [f"- Total de pruebas registradas: {found['n_trials_total']}"]
    lines += ["", op.texts["descargo"], "", f"Generado con código {payload['code_commit']}", ""]
    return "\n".join(lines)


def _write(payload: Mapping[str, Any], op: OperationConfig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(
        render_evaluation(payload, op), encoding="utf-8", newline="\n"
    )
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )


def evaluate_shadow(
    op: OperationConfig,
    config: CoreConfig,
    model_log: TrialLog,
    shadow_log: ShadowLog,
    trials_dir: Path,
    reports_dir: Path,
    curve: pd.DataFrame,
    analysis: Analysis,
    code_commit: str,
    min_weeks: int | None = None,
) -> dict[str, Any]:
    """The registered criteria on the days after the holdout end, logged and reported.

    `curve` and `analysis` are the latest shadow run's, over the whole history. The
    evaluation runs once there are `min_weeks` (default: the operation configuration's)
    distinct reading dates in the shadow log; everything is computed before anything is
    appended or written, and the record is in the shadow log before the report exists.
    """
    desc = config.descriptive
    if desc is None:
        raise TrialLogError("this configuration has no descriptive section")
    verify_configuration(model_log, config)
    minimum = op.shadow_eval_min_weeks if min_weeks is None else min_weeks
    weeks = len({str(r["reading_date"]) for r in shadow_log.readings()})
    if weeks < minimum:
        raise TrialLogError(
            f"the shadow log has {weeks} weeks of readings; at least {minimum} are needed"
        )
    hold_end = _holdout_end(model_log, trials_dir)
    shadow = restrict(analysis, hold_end + ONE_DAY)
    if len(shadow.labels) == 0:
        raise TrialLogError(f"no shadow days after the holdout end {hold_end.date().isoformat()}")
    frozen = _frozen_through(curve, config, registered_columns(model_log), hold_end)
    checks = evaluate(shadow.labels, curve, shadow.proba, frozen, config)
    first, last = shadow.labels.index[0], shadow.labels.index[-1]
    payload: dict[str, Any] = {
        "kind": EVALUATION,
        "stage": STAGE,
        "verdict": verdict(checks),
        "checks": [asdict(c) for c in checks],
        "by_phase": phase_table(shadow.labels, curve, shadow.proba, config),
        "days": int(len(shadow.labels)),
        "weeks": weeks,
        "period": {"desde": first.date().isoformat(), "hasta": last.date().isoformat()},
        "calibration": calibration(shadow.labels, shadow.proba, CALIBRATION_BINS),
        "limits": [*limits_of(desc, holdout=False), SHADOW_EVAL_LIMIT],
        "disclosure": disclosure(model_log),
        "code_commit": code_commit,
        "run_at": shadow_log.clock(),
    }
    stem = reports_dir / f"{REPORT_PREFIX}{payload['period']['hasta']}"
    ensure_writable([shadow_log.path, stem.with_suffix(".md"), stem.with_suffix(".json")])
    shadow_log.append(payload)  # the verdict is in the log before any file is written
    _write(payload, op, stem)
    return payload
