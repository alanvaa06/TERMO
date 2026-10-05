"""The spec 3 command line on the synthetic curve."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.config import load_config
from termo.core import take_snapshot
from termo.desc_cli import main
from termo.validation.trials import TrialLogError
from test_cli import EXP2_SMALL_CONFIG

DESC_SMALL_CONFIG = (
    EXP2_SMALL_CONFIG.replace(
        "grid: {k: [2], jump_penalty: [0.5, 3]}", "grid: {k: [3], jump_penalty: [0.5]}"
    ).replace("frozen_train_end: 1995-12-31", "frozen_train_end: 1996-06-28")
    + """\
descriptive:
  phase_names: [rally de la parte corta, rally de la parte larga, venta]
  sell_phase: 2
  short_led_phase: 0
  long_led_phase: 1
  level_series: DGS10
  slope_long: DGS10
  slope_short: DGS2
  change_days: 63
  coherence_share_min: 0.6
  min_days_evaluable: 40
  map_ari_min: 0.6
  fidelity_min: 0.8
  low_confidence_below: 0.6
  surrogate: {n_estimators: 20, max_depth: 3, learning_rate: 0.1, subsample: 0.8,
    colsample_bytree: 0.8, seed: 0}
  blocks:
    nivel corto: [d1, d2, d3]
    nivel medio: [d5, d7]
    nivel largo: [d10, d30]
    pendientes: [s12m5s, s3s10, s10s30]
    curvatura: [c5]
    volatilidad: [vol1, vol2, vol3, vol5, vol7, vol10, vol30]
"""
)


def test_stages_and_reading_through_the_command_line(
    tmp_path: Path,
    curve: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "k: [3]" in DESC_SMALL_CONFIG and "1996-06-28" in DESC_SMALL_CONFIG
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "desc.yaml"
    config_path.write_text(DESC_SMALL_CONFIG, encoding="utf-8")
    monkeypatch.setattr("termo.desc_cli.code_identity", lambda: "code-3")
    snapshot = tmp_path / "data" / "snapshots" / "2026-10-02"
    take_snapshot(load_config(config_path), snapshot, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    stage = ["--config", str(config_path), "--snapshot", str(snapshot)]

    assert main(["register", *stage]) == 0
    assert main(["run", *stage]) == 0
    assert main(["report", *stage]) == 0
    assert (tmp_path / "trials" / "desc" / "pre_holdout" / "proba.csv").exists()
    assert (tmp_path / "reports" / "desc" / "diagnostic.md").exists()

    assert main(["read", *stage, "--date", "1997-06-02"]) == 0
    reading = tmp_path / "reports" / "desc" / "readings" / "1997-06-02.md"
    assert reading.exists() and "does not anticipate" in reading.read_text(encoding="utf-8")
    assert capsys.readouterr().out.isascii()

    # a holdout date has no reading until a holdout result exists
    with pytest.raises(TrialLogError, match="no validated reading"):
        main(["read", *stage, "--date", "1998-06-01"])
    # the holdout does not open without the explicit flag
    with pytest.raises(SystemExit):
        main(["holdout", *stage])
    log_text = (tmp_path / "trials" / "desc" / "trials.jsonl").read_text(encoding="utf-8")
    assert not any("holdout_opened" in line for line in log_text.splitlines())
