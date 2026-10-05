"""A reading is a lookup in hashed outputs: phase, confidence, drivers, disclaimer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.criteria import APTO, NO_APTO
from termo.descriptive.stages import DISCLAIMER, Analysis, restrict
from termo.reading import (
    DRIVERS_TITLE,
    FIDELITY_WARNING,
    LOST_HOLDOUT,
    build_reading,
    render_reading,
    span_periods,
    write_reading,
)
from termo.validation.trials import TrialLogError

BLOCKS = ["nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad"]
STATUS: dict[str, object] = {
    "diagnostic": APTO,
    "holdout": None,
    "failed_checks": [],
    "fidelity_failed": False,
}
GENERATED: dict[str, object] = {
    "code_commit": "code-3",
    "registered_code_commit": "code-3",
    "environment": {"python": "3.12.0"},
}


def _status(**changes: object) -> dict[str, object]:
    return {**STATUS, **changes}


def _build(analysis: Analysis, day: date, config: CoreConfig, **changes: object) -> dict[str, Any]:
    return build_reading(analysis, day, config, _status(**changes), GENERATED)


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
    reading = _build(analysis, day.date(), desc_config)
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
    assert reading["validation"] == STATUS and reading["drivers_validated"] is True
    assert reading["generated_with"] == GENERATED
    assert reading["disclaimer"] == DISCLAIMER


def test_low_confidence_is_flagged_for_a_weak_or_disagreeing_surrogate(
    desc_config: CoreConfig,
) -> None:
    analysis = _analysis()
    weak = _build(analysis, analysis.labels.index[25].date(), desc_config)
    assert weak["confidence"] == 0.45 and weak["low_confidence"] is True
    other = _build(analysis, analysis.labels.index[26].date(), desc_config)
    assert other["low_confidence"] is True and other["surrogate_agrees"] is False


def test_a_date_without_a_reading_is_refused(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    with pytest.raises(KeyError, match="no reading"):
        _build(analysis, date(2030, 1, 1), desc_config)


def test_a_reading_never_uses_data_after_its_date(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20]
    truncated = Analysis(  # nothing after the date: the day itself is the last row
        labels=analysis.labels.loc[:day],
        frozen_labels=analysis.frozen_labels.loc[:day],
        proba=analysis.proba.loc[:day],
        shap_blocks=analysis.shap_blocks.loc[:day],
        shap_top=analysis.shap_top.loc[:day],
    )
    assert truncated.labels.index[-1] == day and len(truncated.labels) < len(analysis.labels)
    whole = _build(analysis, day.date(), desc_config)
    assert _build(truncated, day.date(), desc_config) == whole


def test_the_holdout_period_keeps_the_phase_history_of_the_period_before(
    desc_config: CoreConfig,
) -> None:
    analysis = _analysis()
    split = analysis.labels.index[15]  # inside the run of phase 0 that starts on day 10
    before = Analysis(
        labels=analysis.labels.loc[: split - pd.Timedelta(days=1)],
        frozen_labels=analysis.frozen_labels.loc[: split - pd.Timedelta(days=1)],
        proba=analysis.proba.loc[: split - pd.Timedelta(days=1)],
        shap_blocks=analysis.shap_blocks.loc[: split - pd.Timedelta(days=1)],
        shap_top=analysis.shap_top.loc[: split - pd.Timedelta(days=1)],
    )
    after = restrict(analysis, split)
    joined = span_periods(before, after)
    pd.testing.assert_series_equal(joined.labels, analysis.labels)
    assert joined.proba is after.proba and joined.shap_blocks is after.shap_blocks
    assert joined.shap_top is after.shap_top and joined.frozen_labels is after.frozen_labels
    day = analysis.labels.index[20].date()
    whole = _build(analysis, day, desc_config)
    assert _build(joined, day, desc_config) == whole
    assert whole["episode_start"] == analysis.labels.index[10].date().isoformat()
    assert whole["episode_start"] < split.date().isoformat()  # the episode began before the split
    # a holdout that starts after the first missing day cannot borrow the history
    with pytest.raises(TrialLogError, match="do not join"):
        span_periods(before, restrict(analysis, analysis.labels.index[16]))
    with pytest.raises(TrialLogError, match="do not join"):
        span_periods(before, restrict(analysis, analysis.labels.index[14]))


def test_markdown_is_ascii_and_never_speaks_about_what_comes_next(
    desc_config: CoreConfig, tmp_path: Path
) -> None:
    analysis = _analysis()
    reading = _build(analysis, analysis.labels.index[20].date(), desc_config)
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


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_an_apto_reading_has_no_banner_and_upper_case_verdicts(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20].date()
    text = render_reading(_build(analysis, day, desc_config))
    assert "NOT VALIDATED" not in text and FIDELITY_WARNING not in text
    assert "- Pre-holdout diagnostic: APTO" in text and "- Holdout: not opened" in text
    assert "- Failed checks (pre-holdout diagnostic): none" in text
    assert _lines(text)[0] == f"# TERMO - reading of {day.isoformat()}"
    assert _lines(text)[2].startswith("**Phase: ")
    both = render_reading(_build(analysis, day, desc_config, holdout=APTO))
    assert "NOT VALIDATED" not in both and "- Holdout: APTO" in both
    assert "- Failed checks (holdout): none" in both


def test_a_no_apto_diagnostic_puts_a_banner_before_anything_and_keeps_the_drivers(
    desc_config: CoreConfig,
) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20].date()
    failed = ["D1_persistence", "D3_rally_coherence"]
    reading = _build(analysis, day, desc_config, diagnostic=NO_APTO, failed_checks=failed)
    assert reading["validation"]["failed_checks"] == failed
    assert reading["drivers_validated"] is True
    assert [d["block"] for d in reading["drivers"]] == ["nivel corto", "pendientes", "curvatura"]
    text = render_reading(reading)
    banner = (
        "**[NOT VALIDATED: pre-holdout diagnostic NO-APTO; "
        "failed: D1_persistence, D3_rally_coherence]**"
    )
    assert _lines(text)[:4] == [f"# TERMO - reading of {day.isoformat()}", "", banner, ""]
    assert text.count("NOT VALIDATED") == 1 and text.isascii()
    assert DRIVERS_TITLE in text and FIDELITY_WARNING not in text
    assert "- nivel corto: +1.50" in text and "- d2_63_r252: +0.90" in text
    assert "- Pre-holdout diagnostic: NO-APTO" in text
    assert "- Failed checks (pre-holdout diagnostic): D1_persistence, D3_rally_coherence" in text


def test_a_failed_fidelity_marks_the_drivers_as_not_validated(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20].date()
    failed = ["D1_persistence", "D6_fidelity"]
    reading = _build(
        analysis, day, desc_config, diagnostic=NO_APTO, failed_checks=failed, fidelity_failed=True
    )
    assert reading["drivers_validated"] is False
    assert reading["drivers"] and reading["top_variables"]  # still shown, flagged
    text = render_reading(reading)
    assert f"{DRIVERS_TITLE} {FIDELITY_WARNING}" in text
    assert "SURROGATE FIDELITY FAILED" in text and "- nivel corto: +1.50" in text
    assert "failed: D1_persistence, D6_fidelity]**" in text


def test_the_holdout_governs_once_it_has_a_result(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20].date()
    text = render_reading(
        _build(
            analysis,
            day,
            desc_config,
            diagnostic=APTO,
            holdout=NO_APTO,
            failed_checks=["D2_level_coherence"],
        )
    )
    assert "**[NOT VALIDATED: holdout NO-APTO; failed: D2_level_coherence]**" in text
    assert "- Pre-holdout diagnostic: APTO" in text and "- Holdout: NO-APTO" in text
    assert "- Failed checks (holdout): D2_level_coherence" in text
    # with no evaluable check the verdict is still not apto, and the banner says so
    none = render_reading(_build(analysis, day, desc_config, holdout=NO_APTO, failed_checks=[]))
    assert "**[NOT VALIDATED: holdout NO-APTO; failed: none evaluable]**" in none


def test_a_lost_holdout_leaves_the_diagnostic_governing_and_stays_visible(
    desc_config: CoreConfig,
) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20].date()
    lost = render_reading(
        _build(
            analysis,
            day,
            desc_config,
            diagnostic=NO_APTO,
            holdout=LOST_HOLDOUT,
            failed_checks=["D1_persistence"],
        )
    )
    assert "**[NOT VALIDATED: pre-holdout diagnostic NO-APTO; failed: D1_persistence]**" in lost
    assert f"- Holdout: {LOST_HOLDOUT}" in lost
    fine = render_reading(_build(analysis, day, desc_config, holdout=LOST_HOLDOUT))
    assert "NOT VALIDATED" not in fine and f"- Holdout: {LOST_HOLDOUT}" in fine


def test_the_reading_records_the_code_that_formatted_it(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20].date()
    same = render_reading(_build(analysis, day, desc_config))
    assert _lines(same)[-1] == "Generated with code code-3" and "(registered" not in same
    later = {**GENERATED, "code_commit": "code-9"}
    reading = build_reading(analysis, day, desc_config, STATUS, later)
    assert reading["generated_with"] == later
    text = render_reading(reading)
    assert _lines(text)[-1] == "Generated with code code-9 (registered: code-3)"
    assert _lines(text).index(DISCLAIMER) < len(_lines(text)) - 1
