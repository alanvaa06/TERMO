"""The operation configuration: macro series, alert rule, cadences, fixed texts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from termo.operation.config import load_operation_config

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"


def test_loads_the_repository_operation_configuration() -> None:
    op = load_operation_config(REPO_OP)
    assert [m.series_id for m in op.macro] == ["THREEFYTP10", "DFF"]
    assert op.macro[0].name == "prima por plazo 10 anos (Kim-Wright)"
    assert op.macro[1].spread_against == "DGS2"  # the proxy is DGS2 - DFF
    assert op.macro[0].spread_against is None
    assert op.alert_min_confidence == 0.6
    assert op.percentile_window_days == 2520 and op.change_days == 21
    assert op.shadow_eval_min_weeks == 26
    assert op.registry == "desc2" and op.model_config == Path("configs/desc2.yaml")
    assert "no anticipa" in op.texts["descargo"].lower()
    assert "sobreconfiado" in op.texts["nota_confianza"]
    assert "no pronóstico" in op.texts["rotulo_historia"]


def test_unknown_keys_and_bad_values_are_refused(tmp_path: Path) -> None:
    text = REPO_OP.read_text(encoding="utf-8")
    bad = tmp_path / "op.yaml"
    bad.write_text(
        text.replace("alert_min_confidence: 0.6", "alert_min_confidenc: 0.6"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_operation_config(bad)
    op = load_operation_config(REPO_OP)
    with pytest.raises(ValueError):
        replace(op, alert_min_confidence=1.5)
    with pytest.raises(ValueError):
        replace(op, shadow_eval_min_weeks=0)
    with pytest.raises(ValueError, match="distinct"):
        replace(op, macro=(op.macro[0], op.macro[0]))
