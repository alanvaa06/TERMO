"""The monthly report: the month's readings, and the past's durations and transitions, labelled."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.criteria import APTO
from termo.descriptive.stages import BASE, Analysis, desc_trial_id
from termo.operation.config import OperationConfig, load_operation_config
from termo.operation.monthly import (
    build_monthly,
    durations_by_phase,
    episodes,
    render_monthly,
    transitions,
    write_monthly,
)
from termo.validation.trials import TrialLog, TrialStatus

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"
NAMES = ("rally fuerte", "rally moderado", "venta")
BLOCKS = ["nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad"]
BY_PHASE: list[dict[str, Any]] = [
    {
        "phase": p,
        "name": NAMES[p],
        "days": 100 + p,
        "evaluable": True,
        "median_duration_days": 30.0 + p,
        "direction_share": 0.7,
        "recall": 0.9,
        "episodes": 3,
    }
    for p in range(3)
]
MACRO_ROWS: list[dict[str, Any]] = [
    {
        "serie": "prima por plazo 10 anos (Kim-Wright)",
        "valor": 1.2345,
        "fecha_valor": "1999-02-26",
        "percentil_10a": 0.456,
        "cambio_21d": -0.1,
        "dias_de_ventana": 2520,
        "nota": "",
    },
    {
        "serie": "DGS2 - fed funds efectiva",
        "valor": 0.5,
        "fecha_valor": "1999-02-25",
        "percentil_10a": 0.8,
        "cambio_21d": 0.05,
        "dias_de_ventana": 2520,
        "nota": "último dato disponible: 1999-02-25",
    },
]


@pytest.fixture(scope="module")
def op() -> OperationConfig:
    return load_operation_config(REPO_OP)


def _hand_labels() -> tuple[pd.Series, pd.DataFrame]:
    """2 x10, 1 x5, 0 x7, 2 x3, 1 x4 on business days; a curve with known yields."""
    days = pd.bdate_range("1999-01-04", periods=29)
    labels = pd.Series([2] * 10 + [1] * 5 + [0] * 7 + [2] * 3 + [1] * 4, index=days)
    curve = pd.DataFrame(
        {
            "DGS2": [4.0 + 0.01 * i for i in range(29)],
            "DGS10": [5.0 + 0.02 * i for i in range(29)],
        },
        index=days,
    )
    return labels, curve


def test_episodes_durations_and_transitions_by_hand() -> None:
    labels, curve = _hand_labels()
    table = episodes(labels, curve, NAMES)
    assert list(table.columns) == [
        "inicio",
        "fin",
        "fase",
        "nombre",
        "dias",
        "cambio_10y_pb",
        "cambio_2y_pb",
    ]
    assert len(table) == 5
    assert table["dias"].tolist() == [10, 5, 7, 3, 4]
    assert table["fase"].tolist() == [2, 1, 0, 2, 1]
    assert table["nombre"].tolist() == [
        "venta",
        "rally moderado",
        "rally fuerte",
        "venta",
        "rally moderado",
    ]
    assert table["inicio"].tolist() == [labels.index[i] for i in (0, 10, 15, 22, 25)]
    assert table["fin"].tolist() == [labels.index[i] for i in (9, 14, 21, 24, 28)]
    # the 10Y rises 0.02 per day: an episode of n days moves 2 * (n - 1) bp; the 2Y half that
    expected_10y = [2.0 * (n - 1) for n in (10, 5, 7, 3, 4)]
    assert table["cambio_10y_pb"].round(6).tolist() == pytest.approx(expected_10y)
    assert table["cambio_2y_pb"].round(6).tolist() == pytest.approx([x / 2 for x in expected_10y])

    durations = durations_by_phase(table, NAMES)
    assert [d["fase"] for d in durations] == [0, 1, 2]
    assert durations[0] == {
        "fase": 0,
        "nombre": "rally fuerte",
        "episodios": 1,
        "mediana_dias": 7.0,
        "p25_dias": 7.0,
        "p75_dias": 7.0,
    }
    # phase 1: lengths 5 and 4 -> median 4.5, quartiles 4.25 and 4.75 (linear interpolation)
    assert durations[1]["episodios"] == 2 and durations[1]["mediana_dias"] == 4.5
    assert durations[1]["p25_dias"] == 4.25 and durations[1]["p75_dias"] == 4.75
    # phase 2: lengths 10 and 3 -> median 6.5, quartiles 4.75 and 8.25
    assert durations[2]["episodios"] == 2 and durations[2]["mediana_dias"] == 6.5
    assert durations[2]["p25_dias"] == 4.75 and durations[2]["p75_dias"] == 8.25

    # sequence 2 -> 1 -> 0 -> 2 -> 1: the last episode has no successor
    rows = {t["fase"]: t for t in transitions(table, NAMES)}
    assert rows[2]["total"] == 2 and rows[2]["nombre"] == "venta"
    assert rows[2]["a"] == {"rally fuerte": 0.0, "rally moderado": 1.0, "venta": 0.0}
    assert rows[1]["total"] == 1 and rows[1]["a"] == {
        "rally fuerte": 1.0,
        "rally moderado": 0.0,
        "venta": 0.0,
    }
    assert rows[0]["total"] == 1 and rows[0]["a"] == {
        "rally fuerte": 0.0,
        "rally moderado": 0.0,
        "venta": 1.0,
    }
    for row in rows.values():
        assert sum(row["a"].values()) == pytest.approx(1.0)


def test_a_phase_without_successor_has_no_shares_and_no_episodes_gives_none() -> None:
    days = pd.bdate_range("1999-01-04", periods=6)
    labels = pd.Series([0, 0, 0, 2, 2, 2], index=days)
    curve = pd.DataFrame({"DGS2": 4.0, "DGS10": 5.0}, index=days)
    table = episodes(labels, curve, NAMES)
    rows = {t["fase"]: t for t in transitions(table, NAMES)}
    assert rows[0]["total"] == 1 and rows[0]["a"]["venta"] == 1.0
    assert rows[2]["total"] == 0 and rows[2]["a"] == {name: None for name in NAMES}
    assert rows[1]["total"] == 0 and rows[1]["a"] == {name: None for name in NAMES}
    durations = durations_by_phase(table, NAMES)
    assert durations[1] == {
        "fase": 1,
        "nombre": "rally moderado",
        "episodios": 0,
        "mediana_dias": None,
        "p25_dias": None,
        "p75_dias": None,
    }


def _analysis(curve: pd.DataFrame, end: str) -> Analysis:
    """A fake analysis on the synthetic curve's days: venta until mid-January, then rally
    moderado through `end` (a phase change inside February would blur `desde`)."""
    days = curve.loc["1998-10-01":end].index
    change = days.get_loc(pd.Timestamp("1999-01-15"))
    labels = pd.Series([2] * change + [1] * (len(days) - change), index=days, dtype=int)
    proba = pd.DataFrame({0: 0.1, 1: 0.7, 2: 0.2}, index=days)
    proba.iloc[:change] = [0.1, 0.2, 0.7]
    blocks = pd.DataFrame(0.0, index=days, columns=[*BLOCKS, BASE])
    blocks["nivel corto"] = [0.01 * i for i in range(len(days))]
    blocks["pendientes"], blocks[BASE] = -0.5, 0.3
    top = pd.DataFrame({"top": json.dumps([["d2_63_r252", 0.9]])}, index=days)
    return Analysis(
        labels=labels, frozen_labels=labels, proba=proba, shap_blocks=blocks, shap_top=top
    )


def _model_log(path: Path, config: CoreConfig) -> TrialLog:
    """A log with the records `validation_from_log` reads: a result and a diagnostic report."""
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
            "by_phase": BY_PHASE,
        }
    )
    return log


def _reading_record(
    day: str, phase: int, confidence: float, alert: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "kind": "reading",
        "run_at": f"{day}T18:00:00Z",
        "reading_date": day,
        "reading": {
            "date": day,
            "phase": phase,
            "phase_name": NAMES[phase],
            "confidence": confidence,
            "low_confidence": confidence < 0.6,
        },
        "alert": alert,
        "code_commit": "code-9",
    }


def _records() -> list[dict[str, Any]]:
    return [
        _reading_record("1999-01-29", 1, 0.8, None),
        _reading_record("1999-02-05", 1, 0.75, None),
        {"kind": "evaluation", "run_at": "1999-02-10T18:00:00Z", "weeks": 26},
        _reading_record(
            "1999-02-19",
            1,
            0.55,
            {"fecha": "1999-02-19", "de": "venta", "a": "rally moderado", "confianza": 0.55},
        ),
        _reading_record("1999-03-05", 1, 0.9, None),
    ]


def test_monthly_payload_covers_the_month_only_from_shadow_readings_and_registered_history(
    curve: pd.DataFrame, desc2_config: CoreConfig, op: OperationConfig, tmp_path: Path
) -> None:
    log = _model_log(tmp_path / "trials.jsonl", desc2_config)
    analysis = _analysis(curve, "1999-02-26")
    payload = build_monthly(
        "1999-02", _records(), log, analysis, curve, desc2_config, op, MACRO_ROWS, "code-9"
    )
    assert payload["mes"] == "1999-02"
    assert payload["fase_cierre"] == 1 and payload["nombre_cierre"] == "rally moderado"
    assert payload["ultimo_dia"] == "1999-02-26"
    assert payload["desde"] == "1999-01-15"
    assert payload["dias_en_fase"] == len(curve.loc["1999-01-15":"1999-02-26"])
    # the month's readings only, in date order; the evaluation record is not a reading
    assert [r["fecha"] for r in payload["lecturas"]] == ["1999-02-05", "1999-02-19"]
    assert payload["lecturas"][1] == {
        "fecha": "1999-02-19",
        "fase": 1,
        "nombre": "rally moderado",
        "confianza": 0.55,
        "baja_confianza": True,
    }
    assert payload["alertas"] == [
        {"fecha": "1999-02-19", "de": "venta", "a": "rally moderado", "confianza": 0.55}
    ]
    # six blocks, averaged over the two reading days, base excluded
    blocks = analysis.shap_blocks.drop(columns=BASE)
    expected = blocks.loc[[pd.Timestamp("1999-02-05"), pd.Timestamp("1999-02-19")]].mean()
    assert set(payload["bloques_promedio"]) == set(BLOCKS)
    assert payload["bloques_promedio"]["nivel corto"] == pytest.approx(expected["nivel corto"])
    assert payload["bloques_promedio"]["pendientes"] == pytest.approx(-0.5)
    # history: venta then rally moderado, so one episode each and one transition venta -> rally
    durations = {d["fase"]: d for d in payload["duraciones"]}
    assert durations[2]["episodios"] == 1
    assert durations[2]["mediana_dias"] == len(curve.loc["1998-10-01":"1999-01-14"])
    assert durations[1]["episodios"] == 1 and durations[0]["episodios"] == 0
    shares = {t["fase"]: t["a"] for t in payload["transiciones"]}
    assert shares[2] == {"rally fuerte": 0.0, "rally moderado": 1.0, "venta": 0.0}
    assert shares[1] == {name: None for name in NAMES}  # the current phase has no successor
    assert payload["rotulo"] == op.texts["rotulo_historia"]
    assert payload["macro"] == MACRO_ROWS
    assert payload["validacion"]["diagnostic"] == APTO
    assert payload["validacion"]["holdout_seen_by"] == "desc_k3 (test)"
    assert payload["validacion"]["by_phase"] == BY_PHASE
    assert payload["no_dice"] == op.texts["no_dice"]
    assert payload["descargo"] == op.texts["descargo"]
    assert payload["generado_con"] == {"code_commit": "code-9"}

    # labels after the month (a new phase in March) change nothing in the month's report
    later = _analysis(curve, curve.index[-1].strftime("%Y-%m-%d"))
    later.labels.loc["1999-03-01":] = 0
    assert later.labels.index[-1] > pd.Timestamp("1999-02-28")
    same = build_monthly(
        "1999-02", _records(), log, later, curve, desc2_config, op, MACRO_ROWS, "code-9"
    )
    assert same == payload


def test_a_month_without_labelled_days_is_refused(
    curve: pd.DataFrame, desc2_config: CoreConfig, op: OperationConfig, tmp_path: Path
) -> None:
    log = _model_log(tmp_path / "trials.jsonl", desc2_config)
    analysis = _analysis(curve, "1999-02-26")
    with pytest.raises(ValueError, match="1999-05"):
        build_monthly("1999-05", [], log, analysis, curve, desc2_config, op, [], "code-9")


def test_monthly_render_is_spanish_and_has_no_forecast_section(
    curve: pd.DataFrame, desc2_config: CoreConfig, op: OperationConfig, tmp_path: Path
) -> None:
    log = _model_log(tmp_path / "trials.jsonl", desc2_config)
    analysis = _analysis(curve, "1999-02-26")
    payload = build_monthly(
        "1999-02", _records(), log, analysis, curve, desc2_config, op, MACRO_ROWS, "code-9"
    )
    text = render_monthly(payload, op)
    order = [
        "# TERMO — ficha mensual 1999-02",
        "## Fase al cierre del mes",
        "**rally moderado**",
        "desde 1999-01-15",
        "## Lecturas del mes",
        "| 1999-02-05 | rally moderado | 0.75 |",
        "| 1999-02-19 | rally moderado | 0.55 (baja) |",
        "## Alertas del mes",
        "1999-02-19: venta → rally moderado",
        "## Qué la empujó",
        "pendientes: -0.50",
        f"**{op.texts['rotulo_historia']}**",
        "## Duraciones históricas",
        f"**{op.texts['rotulo_historia']}**",
        "## Transiciones",
        "| venta | 0% | 100% | 0% | 1 |",
        "## Contexto macro",
        "| prima por plazo 10 anos (Kim-Wright) | 1.23 | 46% | -0.10 |",
        "último dato disponible: 1999-02-25",
        "## Validación",
        "diagnostic APTO",
        "desc_k3 (test)",
        "| rally fuerte | 100 | sí | 30.0 | 0.70 | 0.90 |",
        op.texts["no_dice"],
        op.texts["descargo"],
        "Generado con código code-9",
    ]
    position = -1
    for fragment in order:
        found = text.find(fragment, position + 1)
        assert found > position, f"missing or out of order: {fragment!r}"
        position = found
    assert "á" in text and "ó" in text
    for forbidden in (
        "Movimiento típico",
        "Qué suele venir",
        "siguiente mes",
        "próximo movimiento",
        "va a subir",
        "va a bajar",
        "después de",
    ):
        assert forbidden not in text

    written = write_monthly(payload, op, tmp_path / "fichas")
    assert written == tmp_path / "fichas" / "1999-02.md"
    assert written.read_text(encoding="utf-8") == text
    stored = json.loads((tmp_path / "fichas" / "1999-02.json").read_text(encoding="utf-8"))
    assert stored == payload
    assert "ó" in (tmp_path / "fichas" / "1999-02.json").read_text(encoding="utf-8")
