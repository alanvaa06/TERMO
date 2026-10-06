"""The shadow evaluation: the registered criteria on the days after the holdout, logged."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import SHADOW_DAYS, Desc2Registry, fake_fred_frames, make_macro
from termo.config import CoreConfig
from termo.core import registered_columns
from termo.dataset import prepare
from termo.descriptive.criteria import APTO, NO_APTO, evaluate, phase_table
from termo.descriptive.stages import (
    HOLDOUT_DIR,
    frozen_map,
    holdout_result,
    limits_of,
    load_analysis,
)
from termo.operation.config import OperationConfig, load_operation_config
from termo.operation.shadow import SHADOW_DIR, SHADOW_LOG_FILE, ShadowLog, ShadowRun, run_shadow
from termo.operation.shadow_eval import SHADOW_EVAL_LIMIT, evaluate_shadow, shadow_config
from termo.operation.snapshot import take_operation_snapshot
from termo.validation.trials import TrialLogError

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"
COMMIT = "test-commit-eval"
CHECK_NAMES = (
    "D1_persistence",
    "D2_sell_coherence",
    "D3_rally_coherence",
    "D5_map_stable",
    "D6_fidelity",
)
FORBIDDEN = ("siguiente mes", "próximo movimiento", "va a subir", "va a bajar", "pronóstico de")


@pytest.fixture(scope="module")
def op() -> OperationConfig:
    return load_operation_config(REPO_OP)


@pytest.fixture(scope="module")
def registry(
    desc2_registry: Desc2Registry, tmp_path_factory: pytest.TempPathFactory
) -> Desc2Registry:
    """A private copy of the finished registry: the shadow log written here stays here."""
    return desc2_registry.copy_to(tmp_path_factory.mktemp("termo-shadow-eval"))


@pytest.fixture(scope="module")
def op_snapshot(
    tmp_path_factory: pytest.TempPathFactory,
    curve: pd.DataFrame,
    desc2_config: CoreConfig,
    op: OperationConfig,
) -> Path:
    """The operation snapshot: the FULL curve plus the macro series, 60 days past the registry."""
    target = tmp_path_factory.mktemp("termo-shadow-eval-snapshot") / "op"
    get = fake_fred_frames([curve, make_macro(curve)])
    take_operation_snapshot(desc2_config, op, target, get, "1999-03-15T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def shadow_run(registry: Desc2Registry, op: OperationConfig, op_snapshot: Path) -> ShadowRun:
    """One completed shadow run: one reading in the shadow log, SHADOW_DAYS shadow days."""
    return run_shadow(
        op,
        registry.config,
        registry.log,
        op_snapshot,
        registry.trials_dir,
        registry.reports_dir,
        COMMIT,
        echo=lambda _: None,
    )


def _shadow_log(registry: Desc2Registry) -> ShadowLog:
    return ShadowLog(registry.trials_dir / SHADOW_DIR / SHADOW_LOG_FILE)


def _hold_end(registry: Desc2Registry) -> pd.Timestamp:
    holdout = holdout_result(registry.log)
    assert holdout is not None
    labels = load_analysis(registry.trials_dir / HOLDOUT_DIR, holdout["files"]).labels
    return pd.Timestamp(labels.index[-1])


def test_refuses_fewer_readings_than_the_minimum(
    shadow_run: ShadowRun, registry: Desc2Registry, op: OperationConfig, curve: pd.DataFrame
) -> None:
    log = _shadow_log(registry)
    before = log.path.read_bytes()
    model_before = registry.log.path.read_bytes()
    assert len(log.readings()) == 1 and op.shadow_eval_min_weeks > 1
    with pytest.raises(TrialLogError) as caught:
        evaluate_shadow(
            op,
            registry.config,
            registry.log,
            log,
            registry.trials_dir,
            registry.reports_dir,
            curve,
            shadow_run.analysis,
            COMMIT,
        )
    message = str(caught.value)
    assert "1" in message and str(op.shadow_eval_min_weeks) in message
    with pytest.raises(TrialLogError, match="2"):
        evaluate_shadow(
            op,
            registry.config,
            registry.log,
            log,
            registry.trials_dir,
            registry.reports_dir,
            curve,
            shadow_run.analysis,
            COMMIT,
            min_weeks=2,
        )
    assert log.path.read_bytes() == before
    assert registry.log.path.read_bytes() == model_before
    assert not list(registry.reports_dir.glob("sombra_eval_*"))


@pytest.fixture(scope="module")
def evaluation(
    shadow_run: ShadowRun, registry: Desc2Registry, op: OperationConfig, curve: pd.DataFrame
) -> tuple[dict[str, Any], bytes]:
    """The evaluation with the minimum lowered to one week, and the model log bytes before it."""
    model_before = registry.log.path.read_bytes()
    payload = evaluate_shadow(
        op,
        registry.config,
        registry.log,
        _shadow_log(registry),
        registry.trials_dir,
        registry.reports_dir,
        curve,
        shadow_run.analysis,
        COMMIT,
        min_weeks=1,
    )
    return payload, model_before


def test_evaluates_the_registered_criteria_on_the_shadow_days_only(
    evaluation: tuple[dict[str, Any], bytes],
    shadow_run: ShadowRun,
    registry: Desc2Registry,
    curve: pd.DataFrame,
) -> None:
    payload, _ = evaluation
    config = registry.config
    hold_end = _hold_end(registry)
    shadow = shadow_run.analysis.labels.loc[shadow_run.analysis.labels.index > hold_end]
    assert len(shadow) == SHADOW_DAYS and shadow.index[-1] == curve.index[-1]

    assert payload["kind"] == "evaluation" and payload["stage"] == "sombra"
    assert payload["verdict"] in {APTO, NO_APTO}
    assert [c["name"] for c in payload["checks"]] == list(CHECK_NAMES)
    assert payload["days"] == SHADOW_DAYS and payload["weeks"] == 1
    assert payload["period"] == {
        "desde": shadow.index[0].date().isoformat(),
        "hasta": shadow.index[-1].date().isoformat(),
    }
    assert payload["code_commit"] == COMMIT and payload["run_at"]

    # D5 against a jump model fitted ONCE through the holdout end, on the shadow days only
    frozen_through = shadow_config(config, hold_end)
    assert frozen_through.frozen_train_end == hold_end.date()
    assert frozen_through.holdout_start == hold_end.date() + timedelta(days=1)
    assert replace(frozen_through, frozen_train_end=config.frozen_train_end) == replace(
        config, holdout_start=frozen_through.holdout_start
    )
    frozen = frozen_map(prepare(curve, frozen_through, registered_columns(registry.log)))
    frozen = frozen.loc[frozen.index > hold_end]
    assert frozen.index[0] == shadow.index[0] and frozen.index.equals(shadow.index)
    proba = shadow_run.analysis.proba.loc[shadow.index]
    expected = evaluate(shadow, curve, proba, frozen, config)
    assert payload["checks"] == [asdict(c) for c in expected]
    assert payload["by_phase"] == phase_table(shadow, curve, proba, config)
    assert len(payload["by_phase"]) == 3
    assert set(payload["calibration"]) == {"brier", "reliability", "recall_by_phase"}

    desc = config.descriptive
    assert desc is not None
    assert payload["limits"] == [*limits_of(desc, holdout=False), SHADOW_EVAL_LIMIT]
    assert payload["disclosure"]["n_trials_this_log"] == 1


def test_not_evaluable_checks_agree_with_the_phase_table(
    evaluation: tuple[dict[str, Any], bytes], registry: Desc2Registry
) -> None:
    payload, _ = evaluation
    desc = registry.config.descriptive
    assert desc is not None
    checks = {c["name"]: c for c in payload["checks"]}
    evaluable = {row["phase"] for row in payload["by_phase"] if row["evaluable"]}
    for row in payload["by_phase"]:
        assert row["evaluable"] == (row["days"] >= desc.min_days_evaluable)
    assert (checks["D1_persistence"]["passed"] is None) == (not evaluable)
    assert (checks["D2_sell_coherence"]["passed"] is None) == (desc.sell_phase not in evaluable)
    rallies = {0, 1, 2} - {desc.sell_phase}
    assert (checks["D3_rally_coherence"]["passed"] is None) == (not rallies & evaluable)
    assert (checks["D6_fidelity"]["passed"] is None) == (not evaluable)
    for check in payload["checks"]:
        assert (check["value"] is None) == (check["passed"] is None)
    if not evaluable:
        assert payload["verdict"] == NO_APTO


def test_evaluation_is_logged_and_reported_without_touching_the_model_log(
    evaluation: tuple[dict[str, Any], bytes],
    shadow_run: ShadowRun,
    registry: Desc2Registry,
    op: OperationConfig,
) -> None:
    payload, model_before = evaluation
    assert registry.log.path.read_bytes() == model_before
    assert all(r["kind"] != "evaluation" for r in registry.log.records())

    log = _shadow_log(registry)
    records = log.records()
    assert [r["kind"] for r in records] == ["reading", "evaluation"]
    assert records[1] == {**payload, "at": records[1]["at"]}
    last = log.last_reading()
    assert last is not None and last == {**shadow_run.record, "at": last["at"]}

    stamp = payload["period"]["hasta"]
    markdown_path = registry.reports_dir / f"sombra_eval_{stamp}.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith(f"# TERMO — evaluación de sombra al {stamp}")
    verdict = str(payload["verdict"]).upper().replace("-", " ")
    assert f"**Veredicto: {verdict}** (sombra, 1 semanas, {SHADOW_DAYS} días)" in markdown
    assert "| Criterio | Valor | Requisito | Resultado |" in markdown
    results = {True: "pasa", False: "FALLA", None: "no evaluable"}
    for check in payload["checks"]:
        value = "-" if check["value"] is None else f"{check['value']:.4f}"
        result = results[check["passed"]]
        assert f"| {check['name']} | {value} | {check['requirement']} | {result} |" in markdown
    assert "## Por fase" in markdown
    assert (
        "| Fase | Días | Evaluable | Duración mediana (días) | Dirección | Acierto | Episodios |"
        in markdown
    )
    for row in payload["by_phase"]:
        assert f"| {row['name']} | {row['days']} |" in markdown
    assert "## Lo que estas pruebas no demuestran" in markdown
    assert SHADOW_EVAL_LIMIT in markdown
    assert "## Divulgación" in markdown
    assert op.texts["descargo"] in markdown
    assert f"Generado con código {COMMIT}" in markdown
    for phrase in FORBIDDEN:
        assert phrase not in markdown
    written = json.loads(markdown_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert written == payload
    assert "\r\n" not in markdown_path.read_bytes().decode("utf-8")
