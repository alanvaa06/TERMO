"""A reading is a lookup in hashed outputs: phase, confidence, drivers, disclaimer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.stages import DISCLAIMER, Analysis
from termo.reading import build_reading, render_reading, write_reading

BLOCKS = ["nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad"]
STATUS: dict[str, str | None] = {"diagnostic": "apto", "holdout": None}


def _analysis() -> Analysis:
    days = pd.bdate_range("1997-01-01", periods=30)
    labels = pd.Series([2] * 10 + [0] * 20, index=days)
    proba = pd.DataFrame({0: 0.7, 1: 0.2, 2: 0.1}, index=days)
    proba.iloc[:10] = [0.1, 0.2, 0.7]
    proba.iloc[25] = [0.45, 0.40, 0.15]  # phase 0 still first, but below the confidence line
    proba.iloc[26] = [0.30, 0.60, 0.10]  # the surrogate prefers another phase
    blocks = pd.DataFrame(0.0, index=days, columns=[*BLOCKS, "base"])
    blocks["nivel corto"], blocks["pendientes"], blocks["volatilidad"] = 1.5, -0.8, 0.1
    blocks["curvatura"] = 0.3
    top = pd.DataFrame(
        {"top": json.dumps([["d2_63_r252", 0.9], ["s3s10_21_r126", -0.4]])}, index=days
    )
    return Analysis(
        labels=labels, frozen_labels=labels, proba=proba, shap_blocks=blocks, shap_top=top
    )


def test_reading_reports_phase_run_confidence_and_drivers(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20]
    reading = build_reading(analysis, day.date(), desc_config, STATUS)
    assert reading["date"] == day.date().isoformat()
    assert reading["phase"] == 0 and reading["phase_name"] == "rally de la parte corta"
    assert reading["days_in_phase"] == 11
    assert reading["episode_start"] == analysis.labels.index[10].date().isoformat()
    assert reading["probabilities"] == {
        "rally de la parte corta": 0.7,
        "rally de la parte larga": 0.2,
        "venta": 0.1,
    }
    assert reading["confidence"] == 0.7 and reading["low_confidence"] is False
    assert [d["block"] for d in reading["drivers"]] == ["nivel corto", "pendientes", "curvatura"]
    assert reading["drivers"][1]["contribution"] == -0.8
    assert reading["top_variables"][0] == {"variable": "d2_63_r252", "contribution": 0.9}
    assert reading["validation"] == STATUS
    assert reading["disclaimer"] == DISCLAIMER


def test_low_confidence_is_flagged_for_a_weak_or_disagreeing_surrogate(
    desc_config: CoreConfig,
) -> None:
    analysis = _analysis()
    weak = build_reading(analysis, analysis.labels.index[25].date(), desc_config, STATUS)
    assert weak["confidence"] == 0.45 and weak["low_confidence"] is True
    other = build_reading(analysis, analysis.labels.index[26].date(), desc_config, STATUS)
    assert other["low_confidence"] is True and other["surrogate_agrees"] is False


def test_a_date_without_a_reading_is_refused(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    with pytest.raises(KeyError, match="no reading"):
        build_reading(analysis, date(2030, 1, 1), desc_config, STATUS)


def test_markdown_is_ascii_and_never_speaks_about_what_comes_next(
    desc_config: CoreConfig, tmp_path: Path
) -> None:
    analysis = _analysis()
    reading = build_reading(analysis, analysis.labels.index[20].date(), desc_config, STATUS)
    text = render_reading(reading)
    assert text.isascii() and DISCLAIMER in text
    assert "rally de la parte corta" in text and "nivel corto" in text and "+1.50" in text
    for word in ("forecast", "expected move", "next month", "will "):
        assert word not in text.lower()
    write_reading(reading, tmp_path)
    stem = tmp_path / reading["date"]
    assert json.loads(stem.with_suffix(".json").read_text(encoding="utf-8")) == reading
    assert stem.with_suffix(".md").read_text(encoding="utf-8") == text
    assert not np.isnan(reading["confidence"])
