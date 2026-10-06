"""The Spanish weekly sheet: order of sections, banners, fixed texts, no forecasting language."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from termo.operation.config import MacroSeries, OperationConfig
from termo.operation.sheet_es import render_sheet, write_sheet

FORBIDDEN = ("siguiente mes", "próximo movimiento", "va a subir", "va a bajar")


@pytest.fixture
def op() -> OperationConfig:
    return OperationConfig(
        registry="desc2",
        model_config=Path("configs/desc2.yaml"),
        macro=(MacroSeries("THREEFYTP10", "prima por plazo", None),),
        alert_min_confidence=0.6,
        percentile_window_days=2520,
        change_days=21,
        shadow_eval_min_weeks=26,
        texts={
            "descargo": "TERMO describe la fase actual. No es un pronóstico de la tasa a 10 años.",
            "nota_confianza": "El imitador es sobreconfiado: 0.98 acierta cerca del 89%.",
            "rotulo_historia": "Frecuencias del pasado, no pronóstico.",
            "no_dice": "Lo que el modelo NO dice: hacia dónde irá la tasa.",
        },
    )


def _record(verdict: str = "apto") -> dict[str, Any]:
    return {
        "run_at": "2026-10-09T21:30:00+00:00",
        "snapshot_hash": "0123456789abcdef0123456789abcdef",
        "reading_date": "2026-10-09",
        "code_commit": "abc1234",
        "alert": None,
        "history": {"reproduced": True, "differing": {}},
        "macro": [
            {
                "serie": "prima por plazo 10 años (Kim-Wright)",
                "valor": 0.4567,
                "fecha_valor": "2026-10-09",
                "percentil_10a": 0.734,
                "cambio_21d": -0.1234,
                "dias_de_ventana": 2520,
                "nota": "",
            },
            {
                "serie": "DGS2 - fed funds efectiva",
                "valor": -0.5,
                "fecha_valor": "2026-10-08",
                "percentil_10a": 0.12,
                "cambio_21d": 0.05,
                "dias_de_ventana": 2520,
                "nota": "último dato disponible: 2026-10-08",
            },
        ],
        "reading": {
            "date": "2026-10-09",
            "phase": 0,
            "phase_name": "venta",
            "days_in_phase": 12,
            "episode_start": "2026-09-23",
            "probabilities": {"venta": 0.71, "rally moderado": 0.2, "rally fuerte": 0.09},
            "confidence": 0.71,
            "surrogate_agrees": True,
            "low_confidence": False,
            "drivers": [
                {"block": "nivel", "contribution": 0.812},
                {"block": "pendiente", "contribution": -0.4},
                {"block": "curvatura", "contribution": 0.05},
            ],
            "top_variables": [
                {"variable": f"var{i}", "contribution": c}
                for i, c in enumerate([0.5, -0.3, 0.2, 0.1, -0.05])
            ],
            "drivers_validated": True,
            "validation": {
                "diagnostic": "apto",
                "holdout": verdict,
                "failed_checks": [] if verdict == "apto" else ["D3_stability", "D6_fidelity"],
                "fidelity_failed": verdict != "apto",
                "by_phase": [
                    {
                        "name": "venta",
                        "days": 300,
                        "evaluable": True,
                        "median_duration_days": 21.0,
                        "direction_share": 0.8,
                        "recall": 0.6,
                        "episodes": 9,
                    },
                    {
                        "name": "rally fuerte",
                        "days": 20,
                        "evaluable": False,
                        "median_duration_days": None,
                        "direction_share": None,
                        "recall": None,
                        "episodes": 1,
                    },
                ],
                "registered_verdicts": [
                    {"stage": "diagnostic", "verdict": "apto"},
                    {"stage": "holdout", "verdict": verdict},
                ],
                "holdout_seen_by": "desc1",
                "sombra": {
                    "semanas": 5,
                    "historia_reproducida": True,
                    "proxima_evaluacion_semanas": 21,
                },
            },
            "generated_with": {"code_commit": "abc1234"},
            "disclaimer": "irrelevant english text",
        },
    }


def _order(text: str, needles: list[str]) -> None:
    last = -1
    for needle in needles:
        at = text.find(needle, last + 1)
        assert at > last, f"{needle!r} missing or out of order"
        last = at


def test_full_sheet_sections_in_order(op: OperationConfig) -> None:
    record = _record("no apto")
    record["alert"] = {
        "fecha": "2026-10-09",
        "de": "venta",
        "a": "rally moderado",
        "confianza": 0.7,
    }
    record["history"] = {
        "reproduced": False,
        "differing": {"labels.csv": ["2026-01-05", "2026-01-06"], "proba.csv": ["2026-01-06"]},
    }
    record["reading"]["validation"]["sombra"]["historia_reproducida"] = False
    record["reading"]["drivers_validated"] = False
    text = render_sheet(record, op)
    _order(
        text,
        [
            "# TERMO — hoja semanal del 2026-10-09",
            "2026-10-09",
            "0123456789ab",
            "CAMBIO DE FASE: venta → rally moderado (confianza 0.70)",
            "**[NO VALIDADO:",
            "D3_stability",
            "**HISTORIA REVISADA**: 2 días difieren de la historia registrada",
            "## Fase actual",
            "**venta** desde 2026-09-23 (12 días hábiles)",
            op.texts["nota_confianza"],
            "| rally moderado | 0.20 |",
            "## Qué la empuja",
            "[FIDELIDAD DEL IMITADOR FALLIDA: explicaciones no validadas]",
            "- nivel: +0.81",
            "- pendiente: -0.40",
            "- curvatura: +0.05",
            "- var0: +0.50",
            "- var4: -0.05",
            "## Contexto macro",
            "| prima por plazo 10 años (Kim-Wright) | 0.46 | 73% | -0.12 |",
            "último dato disponible: 2026-10-08",
            "## Validación",
            "holdout NO APTO",
            "| venta | 300 | sí | 21.0 | 0.80 | 0.60 |",
            "| rally fuerte | 20 | no | - | - | - |",
            "Semanas de sombra: 5",
            "Próxima evaluación de sombra en 21 semanas",
            "Nombres de fase elegidos tras el holdout; los valida solo la sombra.",
            op.texts["descargo"],
            "Generado con código abc1234",
        ],
    )
    assert "NO APTO" in text.split("## Fase actual")[0]
    assert "á" in text and "ó" in text


def test_apto_record_without_alert_has_no_banner_or_alert(op: OperationConfig) -> None:
    text = render_sheet(_record("apto"), op)
    assert "CAMBIO DE FASE" not in text
    assert "NO VALIDADO" not in text
    assert "HISTORIA REVISADA" not in text
    assert "FIDELIDAD" not in text
    assert "no registrada" not in text


def test_diagnostic_governs_when_no_holdout_and_by_phase_may_be_missing(
    op: OperationConfig,
) -> None:
    record = copy.deepcopy(_record("apto"))
    val = record["reading"]["validation"]
    val["holdout"] = None
    val["diagnostic"] = "no apto"
    val["failed_checks"] = []
    val["by_phase"] = None
    val["registered_verdicts"] = val["registered_verdicts"][:1]
    del val["holdout_seen_by"]
    val["sombra"]["proxima_evaluacion_semanas"] = 0
    text = render_sheet(record, op)
    assert "**[NO VALIDADO:" in text
    assert "no registrada" in text
    assert "ya puede correrse" in text
    assert "Nombres de fase" not in text


def test_no_forecasting_language_outside_fixed_texts(op: OperationConfig) -> None:
    text = render_sheet(_record("no apto"), op)
    lowered = text.lower()
    for phrase in FORBIDDEN:
        assert phrase not in lowered
    stripped = text.replace(op.texts["descargo"], "")
    assert "pronóstico" not in stripped.lower()


def test_write_sheet_writes_utf8_md_and_json(op: OperationConfig, tmp_path: Path) -> None:
    record = _record()
    path = write_sheet(record, op, tmp_path / "out")
    assert path.name == "2026-10-09.md"
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8") == render_sheet(record, op)
    loaded = json.loads((tmp_path / "out" / "2026-10-09.json").read_text(encoding="utf-8"))
    assert loaded["snapshot_hash"] == record["snapshot_hash"]
